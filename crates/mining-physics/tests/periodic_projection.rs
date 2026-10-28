//! Independent physical-face oracle for the experimental periodic projection.
//! This does not qualify particle transfer, advection or a complete strip solver.
use game::gpu::pressure_3d::GpuPressure3D;
use wgpu::util::DeviceExt;

fn device() -> (wgpu::Device, wgpu::Queue) {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default())).unwrap();
    pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap()
}
fn buffer(d: &wgpu::Device, values: &[f32]) -> wgpu::Buffer {
    d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("periodic projection oracle"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}
fn read(d: &wgpu::Device, q: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<f32> {
    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut e = d.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    q.submit(Some(e.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    d.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let values = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    values
}
fn index(p: [usize; 3], n: [usize; 3]) -> usize {
    (p[2] * n[1] + p[1]) * n[0] + p[0]
}

struct Field {
    n: [usize; 3],
    cells: Vec<u32>,
    mobility: [Vec<f32>; 3],
    open: u32,
    h: f32,
}
impl Field {
    fn new(n: [usize; 3], varied: bool, open: u32) -> Self {
        let mut cells = vec![1; n.iter().product()];
        for z in 0..n[2] {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    // Include asymmetric air/fluid and solid/fluid seam pairs.
                    cells[index([x, y, z], n)] =
                        if y == n[1] - 1 || (x == 1 && y == n[1] / 2 && z == 0) {
                            0
                        } else if x == n[0] / 2 && y < n[1] / 3 && z == n[2] - 1 {
                            2
                        } else {
                            1
                        };
                }
            }
        }
        let mut mobility: [Vec<f32>; 3] = std::array::from_fn(|axis| {
            let mut shape = n;
            shape[axis] += 1;
            (0..shape.iter().product())
                .map(|i| {
                    let x = i % shape[0];
                    let y = (i / shape[0]) % shape[1];
                    let z = i / (shape[0] * shape[1]);
                    if varied {
                        0.05 + 0.95 * ((x + 2 * y + 3 * (z % n[2]) + axis * 5) % 17) as f32 / 16.
                    } else {
                        1.
                    }
                })
                .collect()
        });
        // The allocated depth plane is not a second physical seam coefficient.
        mobility[2][n.iter().product::<usize>()..].fill(7.);
        Self {
            n,
            cells,
            mobility,
            open,
            h: 0.01,
        }
    }
    fn target(&self) -> Vec<f32> {
        (0..self.cells.len())
            .map(|i| {
                let x = i % self.n[0];
                let y = (i / self.n[0]) % self.n[1];
                let z = i / (self.n[0] * self.n[1]);
                if self.cells[i] == 1 {
                    0.001 * ((x * 3 + y * 5 + z * 7) % 19) as f32 - 0.009
                } else {
                    0.
                }
            })
            .collect()
    }
    /// Visit each physical face once. Periodic W0 connects last and first
    /// cells; the allocated W-depth plane is a redundant output, not a face.
    fn faces(&self, mut visit: impl FnMut(usize, usize, Option<usize>, Option<usize>, f64)) {
        for axis in 0..3 {
            let mut shape = self.n;
            if axis != 2 {
                shape[axis] += 1;
            }
            let mut storage = self.n;
            storage[axis] += 1;
            for z in 0..shape[2] {
                for y in 0..shape[1] {
                    for x in 0..shape[0] {
                        let p = [x, y, z];
                        let face = index(p, storage);
                        let left = if axis == 2 {
                            Some(index([x, y, (z + self.n[2] - 1) % self.n[2]], self.n))
                        } else if p[axis] > 0 {
                            let mut p = p;
                            p[axis] -= 1;
                            Some(index(p, self.n))
                        } else {
                            None
                        };
                        let right = if p[axis] < self.n[axis] {
                            Some(index(p, self.n))
                        } else {
                            None
                        };
                        let solid = |i: Option<usize>, side: u32| {
                            i.map_or(self.open & (side << (axis * 2)) == 0, |i| {
                                self.cells[i] == 2
                            })
                        };
                        if solid(left, 1) || solid(right, 2) {
                            continue;
                        }
                        if left.is_none_or(|i| self.cells[i] != 1)
                            && right.is_none_or(|i| self.cells[i] != 1)
                        {
                            continue;
                        }
                        visit(
                            axis,
                            face,
                            left.filter(|&i| self.cells[i] == 1),
                            right.filter(|&i| self.cells[i] == 1),
                            self.mobility[axis][face] as f64,
                        );
                    }
                }
            }
        }
    }
    fn apply(&self, p: &[f32]) -> Vec<f64> {
        let mut out = vec![0.; self.cells.len()];
        self.faces(|_, _, left, right, a| {
            let flux = a * (left.map_or(0., |i| p[i] as f64) - right.map_or(0., |i| p[i] as f64));
            if let Some(i) = left {
                out[i] += flux;
            }
            if let Some(i) = right {
                out[i] -= flux;
            }
        });
        out
    }
    fn gradient(&self, p: &[f32]) -> [Vec<f32>; 3] {
        let mut out = std::array::from_fn(|axis| {
            let mut n = self.n;
            n[axis] += 1;
            vec![0.; n.iter().product()]
        });
        self.faces(|axis, face, left, right, a| {
            out[axis][face] = (a
                * (right.map_or(0., |i| p[i] as f64) - left.map_or(0., |i| p[i] as f64))
                / self.h as f64) as f32;
        });
        // Poison the duplicate plane: divergence must use canonical W0, then
        // the single seam owner must overwrite this stale input after projection.
        out[2][self.cells.len()..].fill(17.);
        out
    }
}

#[test]
#[ignore = "requires actual GPU; run explicitly with --ignored"]
fn periodic_faces_match_independent_flux_projection_and_all_schedules() {
    let (d, q) = device();
    let mut cases = 0;
    for n in [[9, 7, 2], [17, 13, 8], [17, 13, 12]] {
        for varied in [false, true] {
            for open in [0, 10] {
                let f = Field::new(n, varied, open);
                let target = f.target();
                let rhs = f.apply(&target);
                let velocity = f.gradient(&target);
                let mobility: Vec<_> = f
                    .mobility
                    .iter()
                    .map(|m| {
                        let mut v = vec![if varied { 1. } else { 0. }];
                        v.extend(m);
                        buffer(&d, &v)
                    })
                    .collect();
                let mut outputs = Vec::new();
                for (compact, cached) in [(false, false), (true, false), (true, true)] {
                    let v: Vec<_> = velocity.iter().map(|v| buffer(&d, v)).collect();
                    let mut solver = GpuPressure3D::new_periodic_z(
                        &d,
                        n[0] as u32,
                        n[1] as u32,
                        n[2] as u32,
                        &v[0],
                        &v[1],
                        &v[2],
                        Some([&mobility[0], &mobility[1], &mobility[2]]),
                    )
                    .unwrap();
                    solver.compact_iterations = compact;
                    solver.cached_stencil = cached;
                    solver.upload_cell_types(&q, &f.cells, f.h, f.open);
                    let mut e = d.create_command_encoder(&Default::default());
                    solver.clear_pressure(&mut e);
                    solver.encode(&mut e, 480);
                    q.submit(Some(e.finish()));
                    let p = read(&d, &q, &solver.pressure_buffer);
                    let applied = f.apply(&p);
                    let relative = (applied
                        .iter()
                        .zip(&rhs)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        / rhs.iter().map(|b| b * b).sum::<f64>())
                    .sqrt();
                    assert!(
                        relative < 1e-4,
                        "shape{n:?} varied{varied} open{open} relative{relative}"
                    );
                    let measured = solver.measure_residual(&d, &q).unwrap();
                    assert!(measured.relative_l2.unwrap() < 1e-4, "{measured:?}");
                    let got: Vec<_> = v.iter().map(|v| read(&d, &q, v)).collect();
                    let max_speed = got.iter().flatten().map(|v| v.abs()).fold(0., f32::max);
                    assert!(max_speed < 0.0002, "remaining speed{max_speed} shape{n:?}");
                    assert_eq!(
                        got[2][..n[0] * n[1]]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        got[2][f.cells.len()..]
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "seam outputs disagree"
                    );
                    outputs.push(p.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    cases += 1;
                }
                assert_eq!(outputs[0], outputs[1]);
                assert_eq!(outputs[0], outputs[2]);
            }
        }
    }
    assert_eq!(cases, 36);
    println!(
        "{cases} periodic projections: independent face flux, canonical seam, three schedules"
    );
}

#[test]
#[ignore = "requires actual GPU; run explicitly with --ignored"]
fn periodic_closed_gauge_and_incompatible_source_remain_visible() {
    let (d, q) = device();
    let n = [9, 7, 8];
    let count = n.iter().product();
    let sizes = [
        (n[0] + 1) * n[1] * n[2],
        n[0] * (n[1] + 1) * n[2],
        n[0] * n[1] * (n[2] + 1),
    ];
    let v: Vec<_> = sizes.iter().map(|&n| buffer(&d, &vec![0.; n])).collect();
    let solver = GpuPressure3D::new_periodic_z(
        &d,
        n[0] as u32,
        n[1] as u32,
        n[2] as u32,
        &v[0],
        &v[1],
        &v[2],
        None,
    )
    .unwrap();
    solver.upload_cell_types(&q, &vec![1; count], 0.01, 0);
    q.write_buffer(
        &solver.pressure_buffer,
        0,
        bytemuck::cast_slice(&vec![0.25f32; count]),
    );
    let mut e = d.create_command_encoder(&Default::default());
    solver.encode_iterations_only(&mut e, 80);
    q.submit(Some(e.finish()));
    assert!(read(&d, &q, &solver.pressure_buffer)
        .iter()
        .all(|p| p.to_bits() == 0.25f32.to_bits()));
    q.write_buffer(
        &solver.divergence_buffer,
        0,
        bytemuck::cast_slice(&vec![1f32; count]),
    );
    let mut e = d.create_command_encoder(&Default::default());
    solver.encode_iterations_only(&mut e, 80);
    q.submit(Some(e.finish()));
    assert!(
        solver
            .measure_residual(&d, &q)
            .unwrap()
            .relative_l2
            .unwrap()
            > 0.5,
        "closed incompatible RHS must not be silently accepted as converged"
    );
    for depth in [0, 1, 3, 7, 13] {
        assert!(GpuPressure3D::new_periodic_z(&d, 9, 7, depth, &v[0], &v[1], &v[2], None).is_err());
    }
}

#[test]
#[ignore = "requires actual GPU; run explicitly with --ignored"]
fn periodic_strip_matches_four_repeated_domains_bit_for_bit() {
    let (d, q) = device();
    for period in [8, 12] {
        let f = Field::new([9, 7, period], true, 10);
        let b: Vec<_> = f
            .apply(&f.target())
            .iter()
            .map(|b| (-b / (f.h * f.h) as f64) as f32)
            .collect();
        for sweeps in [1, 7, 80] {
            let solve = |repeat| {
                let n = [9, 7, period * repeat];
                let count = n.iter().product();
                let sizes = [
                    (n[0] + 1) * n[1] * n[2],
                    n[0] * (n[1] + 1) * n[2],
                    n[0] * n[1] * (n[2] + 1),
                ];
                let v: Vec<_> = sizes.iter().map(|&n| buffer(&d, &vec![0.; n])).collect();
                let mobility: Vec<_> = (0..3)
                    .map(|axis| {
                        let source = if axis == 2 {
                            &f.mobility[axis][..f.cells.len()]
                        } else {
                            &f.mobility[axis]
                        };
                        let mut m = vec![1.];
                        m.extend((0..sizes[axis]).map(|i| source[i % source.len()]));
                        buffer(&d, &m)
                    })
                    .collect();
                let solver = GpuPressure3D::new_periodic_z(
                    &d,
                    9,
                    7,
                    n[2] as u32,
                    &v[0],
                    &v[1],
                    &v[2],
                    Some([&mobility[0], &mobility[1], &mobility[2]]),
                )
                .unwrap();
                let cells: Vec<_> = (0..count).map(|i| f.cells[i % f.cells.len()]).collect();
                let rhs: Vec<_> = (0..count).map(|i| b[i % b.len()]).collect();
                solver.upload_cell_types(&q, &cells, f.h, f.open);
                q.write_buffer(&solver.divergence_buffer, 0, bytemuck::cast_slice(&rhs));
                let mut e = d.create_command_encoder(&Default::default());
                solver.clear_pressure(&mut e);
                solver.encode_iterations_only(&mut e, sweeps);
                q.submit(Some(e.finish()));
                read(&d, &q, &solver.pressure_buffer)
                    .iter()
                    .map(|p| p.to_bits())
                    .collect::<Vec<_>>()
            };
            let strip = solve(1);
            let full = solve(4);
            for tile in full.chunks_exact(strip.len()) {
                assert_eq!(tile, strip, "period {period}, sweeps {sweeps}");
            }
        }
    }
    println!("8/12-cell strip pressure equals four tiled periods at 1/7/80 sweeps");
}

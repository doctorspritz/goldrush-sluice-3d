//! Independent finite-volume face-flux oracle for the production pressure solve.
use game::gpu::pressure_3d::{GpuPressure3D, PressureResidual3D};
use serde_json::json;
use wgpu::util::DeviceExt;

fn device() -> (wgpu::Device, wgpu::Queue) {
    let a =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default())).unwrap();
    pollster::block_on(a.request_device(&Default::default(), None)).unwrap()
}
fn buffer(d: &wgpu::Device, v: &[f32]) -> wgpu::Buffer {
    d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(v),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}
fn read(d: &wgpu::Device, q: &wgpu::Queue, b: &wgpu::Buffer) -> Vec<f32> {
    let t = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: b.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = d.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(b, 0, &t, 0, b.size());
    q.submit(Some(e.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    t.slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    d.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let v = bytemuck::cast_slice(&t.slice(..).get_mapped_range()).to_vec();
    t.unmap();
    v
}
#[derive(Clone)]
struct Field {
    shape: [usize; 3],
    cells: Vec<u32>,
    mobility: [Vec<f32>; 3],
    open: u32,
    h: f32,
}
fn index(p: [usize; 3], n: [usize; 3]) -> usize {
    (p[2] * n[1] + p[1]) * n[0] + p[0]
}
impl Field {
    fn new(shape: [usize; 3], open: u32, varied: bool, obstacle: bool) -> Self {
        let count = shape.iter().product();
        let mut cells = vec![1; count];
        if obstacle {
            for z in 0..shape[2] {
                for y in 0..shape[1] {
                    for x in 0..shape[0] {
                        if y >= shape[1] * 3 / 4 {
                            cells[index([x, y, z], shape)] = 0;
                        }
                        if x >= shape[0] / 3
                            && x < shape[0] / 2
                            && y < shape[1] / 2
                            && z >= shape[2] / 3
                            && z < shape[2] * 2 / 3
                        {
                            cells[index([x, y, z], shape)] = 2;
                        }
                    }
                }
            }
        }
        let mobility = std::array::from_fn(|axis| {
            let mut n = shape;
            n[axis] += 1;
            (0..n.iter().product())
                .map(|i| {
                    if !varied {
                        return 1.;
                    }
                    // Continuous and layered phase-mobility contrasts up to 20:1.
                    let x = i % n[0];
                    let y = (i / n[0]) % n[1];
                    let z = i / (n[0] * n[1]);
                    let t = (x + 2 * y + 3 * z + axis * 5) as f32 * 0.19;
                    0.05 + 0.95 * (0.5 + 0.5 * t.sin()).powi(2)
                })
                .collect()
        });
        Self {
            shape,
            cells,
            mobility,
            open,
            h: 0.12 / shape[0] as f32,
        }
    }
    /// Accumulate each physical face once, with equal/opposite flux into its
    /// two fluid cells. Air/open faces have zero pressure; solid/closed do not
    /// carry flux. This is separate from the shader's cell-neighbour stencil.
    fn apply(&self, p: &[f64], weighted: bool) -> Vec<f64> {
        let mut result = vec![0.; self.cells.len()];
        for axis in 0..3 {
            let mut n = self.shape;
            n[axis] += 1;
            for z in 0..n[2] {
                for y in 0..n[1] {
                    for x in 0..n[0] {
                        let pos = [x, y, z];
                        let k = index(pos, n);
                        let left = if pos[axis] > 0 {
                            let mut q = pos;
                            q[axis] -= 1;
                            Some(index(q, self.shape))
                        } else {
                            None
                        };
                        let right = if pos[axis] < self.shape[axis] {
                            Some(index(pos, self.shape))
                        } else {
                            None
                        };
                        let is_fluid = |i: Option<usize>| i.is_some_and(|i| self.cells[i] == 1);
                        let is_closed = |i: Option<usize>, bit: u32| match i {
                            Some(i) => self.cells[i] == 2,
                            None => self.open & bit == 0,
                        };
                        if is_closed(left, 1 << (2 * axis)) || is_closed(right, 2 << (2 * axis)) {
                            continue;
                        }
                        let lp = left.filter(|_| is_fluid(left)).map_or(0., |i| p[i]);
                        let rp = right.filter(|_| is_fluid(right)).map_or(0., |i| p[i]);
                        let a = if weighted {
                            self.mobility[axis][k] as f64
                        } else {
                            1.
                        };
                        let flux = a * (lp - rp);
                        if let Some(i) = left.filter(|_| is_fluid(left)) {
                            result[i] += flux;
                        }
                        if let Some(i) = right.filter(|_| is_fluid(right)) {
                            result[i] -= flux;
                        }
                    }
                }
            }
        }
        result
    }
    fn solver(
        &self,
        d: &wgpu::Device,
        q: &wgpu::Queue,
        weighted: bool,
        ready: bool,
    ) -> GpuPressure3D {
        let grids = self
            .mobility
            .each_ref()
            .map(|m| buffer(d, &vec![0.; m.len()]));
        let coeff = self.mobility.each_ref().map(|m| {
            let mut c = vec![if ready { 1. } else { 0. }];
            c.extend(m);
            buffer(d, &c)
        });
        let [nx, ny, nz] = self.shape.map(|n| n as u32);
        let solver = if weighted {
            GpuPressure3D::new_with_mobility(
                d,
                nx,
                ny,
                nz,
                &grids[0],
                &grids[1],
                &grids[2],
                coeff.each_ref(),
            )
        } else {
            GpuPressure3D::new(d, nx, ny, nz, &grids[0], &grids[1], &grids[2])
        };
        solver.upload_cell_types(q, &self.cells, self.h, self.open);
        solver
    }
}
fn rhs(f: &Field, q: &wgpu::Queue, solver: &GpuPressure3D, b: &[f64]) -> Vec<f32> {
    let div: Vec<_> = b.iter().map(|b| -(*b as f32) / (f.h * f.h)).collect();
    q.write_buffer(&solver.divergence_buffer, 0, bytemuck::cast_slice(&div));
    div
}
fn check_residual(
    f: &Field,
    d: &wgpu::Device,
    q: &wgpu::Queue,
    solver: &GpuPressure3D,
    weighted: bool,
    div: &[f32],
) -> PressureResidual3D {
    let p = read(d, q, &solver.pressure_buffer);
    let ap = f.apply(&p.iter().map(|v| *v as f64).collect::<Vec<_>>(), weighted);
    let r: Vec<_> = ap
        .iter()
        .enumerate()
        .filter(|(i, _)| f.cells[*i] == 1)
        .map(|(i, a)| -(div[i] * f.h * f.h) as f64 - a)
        .collect();
    let expected_max = r.iter().map(|r| r.abs()).fold(0f64, f64::max);
    let expected_rms = (r.iter().map(|r| r * r).sum::<f64>() / r.len().max(1) as f64).sqrt();
    let report = solver.measure_residual(d, q).unwrap();
    assert_eq!(report.fluid_cells, r.len());
    assert!(
        (report.max_abs - expected_max).abs() < 2e-8,
        "GPU/CPU face-flux maximum residual: {:?}, {expected_max}",
        report
    );
    assert!(
        (report.rms - expected_rms).abs() < 2e-8,
        "GPU/CPU face-flux RMS residual: {:?}, {expected_rms}",
        report
    );
    assert_eq!(
        p,
        read(d, q, &solver.pressure_buffer),
        "diagnostic changed pressure"
    );
    assert_eq!(
        div,
        read(d, q, &solver.divergence_buffer),
        "diagnostic changed RHS"
    );
    report
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn stored_pressure_residual_matches_face_fluxes_and_rejects_nonfinite() {
    let (d, q) = device();
    for weighted in [false, true] {
        for ready in [false, true] {
            for open in [0, 1, 8, 32, 63] {
                let f = Field::new([9, 8, 7], open, true, true);
                let solver = f.solver(&d, &q, weighted, ready);
                let p: Vec<f32> = (0..f.cells.len())
                    .map(|i| 0.01 * (i as f32 * 0.73).sin())
                    .collect();
                q.write_buffer(&solver.pressure_buffer, 0, bytemuck::cast_slice(&p));
                let b: Vec<_> = (0..p.len())
                    .map(|i| 0.0003 * (i as f64 * 0.31).cos())
                    .collect();
                let div = rhs(&f, &q, &solver, &b);
                let report = check_residual(&f, &d, &q, &solver, weighted && ready, &div);
                assert!(report.relative_l2.unwrap() > 1.);
                assert!(report.max_backward_error > 0.01);
                q.write_buffer(
                    &solver.divergence_buffer,
                    0,
                    bytemuck::cast_slice(&vec![0f32; p.len()]),
                );
                assert_eq!(
                    solver.measure_residual(&d, &q).unwrap().relative_l2,
                    None,
                    "nonzero residual with zero source reported as converged"
                );
                q.write_buffer(
                    &solver.pressure_buffer,
                    0,
                    bytemuck::cast_slice(&vec![0f32; p.len()]),
                );
                assert_eq!(
                    solver.measure_residual(&d, &q).unwrap().relative_l2,
                    Some(0.)
                );
                q.write_buffer(&solver.pressure_buffer, 0, bytemuck::bytes_of(&f32::NAN));
                assert!(
                    solver.measure_residual(&d, &q).is_err(),
                    "nonfinite equation reported as measured"
                );
            }
        }
    }
    // A closed isolated cell has no pressure diagonal. A nonzero source is
    // incompatible and must remain an explicit residual, not a zero/NaN pass.
    let mut f = Field::new([3, 3, 3], 0, false, false);
    f.cells.fill(2);
    f.cells[13] = 1;
    let solver = f.solver(&d, &q, true, true);
    let mut b = vec![0.; 27];
    b[13] = 0.001;
    let div = rhs(&f, &q, &solver, &b);
    let report = check_residual(&f, &d, &q, &solver, true, &div);
    assert!((report.relative_l2.unwrap() - 1.).abs() < 1e-6);
    assert!((report.max_backward_error - 1.).abs() < 1e-6);
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn weighted_pressure_converges_across_boundaries_mobility_and_obstacles() {
    let (d, q) = device();
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for shape in [[12, 10, 8], [24, 24, 24]] {
        for open in [0, 1, 2, 4, 8, 16, 32, 63] {
            for varied in [false, true] {
                let obstacle = varied;
                let f = Field::new(shape, open, varied, obstacle);
                let solver = f.solver(&d, &q, true, true);
                let exact: Vec<_> = (0..f.cells.len())
                    .map(|i| {
                        let x = (i % shape[0]) as f64 / shape[0] as f64;
                        let y = ((i / shape[0]) % shape[1]) as f64 / shape[1] as f64;
                        let z = (i / (shape[0] * shape[1])) as f64 / shape[2] as f64;
                        0.01 * ((1.7 * x).cos() + 0.3 * y + 0.2 * (2.1 * z).sin())
                    })
                    .collect();
                let b = f.apply(&exact, true);
                let div = rhs(&f, &q, &solver, &b);
                for (total, additional) in [(120, 120), (480, 360)] {
                    let mut e = d.create_command_encoder(&Default::default());
                    solver.encode_iterations_only(&mut e, additional);
                    q.submit(Some(e.finish()));
                    let report = check_residual(&f, &d, &q, &solver, true, &div);
                    let limit = if total == 120 { 0.005 } else { 0.0001 };
                    let passed = report.relative_l2.is_some_and(|r| r < limit);
                    let row = json!({"shape":shape,"open":open,"variable_mobility":varied,"obstacle":obstacle,"sweeps":total,"relative_l2":report.relative_l2,"rms":report.rms,"max_abs":report.max_abs,"max_backward_error":report.max_backward_error,"limit":limit,"passed":passed});
                    if !passed {
                        failures.push(row.clone());
                    }
                    println!("{row}");
                    rows.push(row);
                }
            }
        }
    }
    if let Some(path) = std::env::var_os("SLUICE_PRESSURE_RECEIPT") {
        assert!(!std::path::Path::new(&path).exists(), "preserve receipt");
        std::fs::write(path, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
    }
    assert!(
        failures.is_empty(),
        "pressure convergence failures: {failures:?}"
    );
}

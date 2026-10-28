//! Complete strip-water and independent periodic stencil controls. No mineral,
//! full-width plant ownership, gameplay or calibrated-recovery claim is implied.
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;
const H: f32 = 0.0625;
type Shape = [usize; 3];
fn device() -> (wgpu::Device, wgpu::Queue) {
    let i = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });
    let a = pollster::block_on(i.request_adapter(&Default::default())).expect("real Metal adapter");
    eprintln!("adapter: {:?}", a.get_info());
    pollster::block_on(a.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: a.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap()
}
fn buffer<T: bytemuck::Pod>(d: &wgpu::Device, v: &[T]) -> Arc<wgpu::Buffer> {
    Arc::new(d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(v),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    }))
}
fn read<T: bytemuck::Pod>(d: &wgpu::Device, q: &wgpu::Queue, b: &wgpu::Buffer) -> Vec<T> {
    let s = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: b.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut e = d.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(b, 0, &s, 0, b.size());
    q.submit(Some(e.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    s.slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    d.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let v = bytemuck::cast_slice(&s.slice(..).get_mapped_range()).to_vec();
    s.unmap();
    v
}
fn run(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    source: &str,
    entry: &str,
    bindings: &[(u32, &wgpu::Buffer, bool)],
    constants: &[(&str, f64)],
    groups: [u32; 3],
) {
    let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let entries: Vec<_> = bindings
        .iter()
        .map(|(binding, _, write)| wgpu::BindGroupLayoutEntry {
            binding: *binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if *binding == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: !*write }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    });
    let pl = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let constants = constants.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        cache: None,
    });
    let entries: Vec<_> = bindings
        .iter()
        .map(|(binding, b, _)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &entries,
    });
    let mut e = d.create_command_encoder(&Default::default());
    {
        let mut p = e.begin_compute_pass(&Default::default());
        p.set_pipeline(&pipeline);
        p.set_bind_group(0, &bg, &[]);
        p.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }
    q.submit(Some(e.finish()));
}
fn index(p: Shape, n: Shape) -> usize {
    (p[2] * n[1] + p[1]) * n[0] + p[0]
}
fn nodes(n: Shape) -> impl Iterator<Item = Shape> {
    (0..n[2]).flat_map(move |z| (0..n[1]).flat_map(move |y| (0..n[0]).map(move |x| [x, y, z])))
}
fn shape(n: Shape, a: usize) -> Shape {
    let mut s = n;
    s[a] += 1;
    s
}
fn neighbour(p: Shape, axis: usize, sign: i32, n: Shape) -> Option<Shape> {
    let mut p = p.map(|v| v as i32);
    p[axis] += sign;
    p[2] = p[2].rem_euclid(n[2] as i32);
    if p[0] < 0 || p[0] >= n[0] as i32 || p[1] < 0 || p[1] >= n[1] as i32 {
        None
    } else {
        Some(p.map(|v| v as usize))
    }
}
fn groups(n: Shape) -> [u32; 3] {
    [
        (n[0] as u32).div_ceil(8),
        (n[1] as u32).div_ceil(8),
        (n[2] as u32).div_ceil(4),
    ]
}
fn hat(v: f64) -> f64 {
    (1. - v.abs()).max(0.)
}
fn sample_images(n: Shape, p: [f32; 3], values: &[f32], stagger: [f64; 3]) -> f64 {
    let mut v = 0.;
    for node in nodes(n) {
        for image in -1..=1 {
            let w = (0..3)
                .map(|a| {
                    hat(f64::from(p[a]) / f64::from(H)
                        - node[a] as f64
                        - stagger[a]
                        - if a == 2 {
                            f64::from(image) * n[2] as f64
                        } else {
                            0.
                        })
                })
                .product::<f64>();
            v += w * f64::from(values[index(node, n)]);
        }
    }
    v
}

#[test]
#[ignore = "requires real Metal GPU; run with --ignored"]
fn density_marking_gradient_and_positive_face_gather_match_physical_neighbours() {
    let (d, q) = device();
    for depth in [2, 8, 12] {
        let n = [8, 8, depth];
        let len = n.into_iter().product::<usize>();
        let g = groups(n);
        let mut cells = vec![1u32; len];
        cells[index([3, 3, depth - 1], n)] = 2;
        cells[index([2, 2, 0], n)] = 0;
        let counts: Vec<i32> = nodes(n)
            .map(|[x, y, z]| {
                if (x + 2 * y + z) % 5 == 0 {
                    0
                } else {
                    4 + ((x + y + z) % 8) as i32
                }
            })
            .collect();
        let types = buffer(&d, &cells);
        let count = buffer(&d, &counts);
        let out = buffer(&d, &vec![999f32; len]);
        let dummy = buffer(&d, &vec![0f32; len]);
        let params = buffer(
            &d,
            &[
                8u32,
                8,
                depth as u32,
                8f32.to_bits(),
                0.01f32.to_bits(),
                1,
                0,
                0,
            ],
        );
        run(
            &d,
            &q,
            include_str!("../../game/src/gpu/shaders/density_error_3d.wgsl"),
            "compute_density_error",
            &[
                (0, &params, false),
                (1, &count, false),
                (2, &types, false),
                (3, &out, true),
                (4, &dummy, false),
            ],
            &[("PERIODIC_Z", 1.)],
            g,
        );
        let got: Vec<f32> = read(&d, &q, &out);
        for p in nodes(n) {
            let i = index(p, n);
            let neighbours: Vec<u32> = (0..3)
                .flat_map(|a| [-1, 1].map(move |sign| neighbour(p, a, sign, n)))
                .map(|p| p.map_or(0, |p| cells[index(p, n)]))
                .collect();
            let mut error = (f64::from(counts[i])
                + neighbours.iter().filter(|c| **c == 2).count() as f64 * 0.5625)
                / 8.
                - 1.;
            if neighbours.contains(&0) {
                error = error.max(0.);
            }
            error = (error / 0.01).clamp(-200., 200.);
            if cells[i] != 1 {
                error = 0.;
            }
            assert!(
                (f64::from(got[i]) - error).abs() < 2e-4,
                "density {p:?}: {} vs {error}",
                got[i]
            );
        }
        // Cell expansion uses immutable home counts, including across the seam.
        let params = buffer(&d, &[8u32, 8, depth as u32, 10, 2, 0, 0, 0]);
        run(
            &d,
            &q,
            include_str!("../../game/src/gpu/shaders/fluid_cell_expand_3d.wgsl"),
            "expand_fluid_cells",
            &[(0, &params, false), (1, &count, false), (2, &types, true)],
            &[("PERIODIC_Z", 1.)],
            g,
        );
        let marked: Vec<u32> = read(&d, &q, &types);
        for p in nodes(n) {
            let i = index(p, n);
            let neighbour_count = (0..3)
                .flat_map(|a| [-1, 1].map(move |sign| neighbour(p, a, sign, n)))
                .filter(|p| p.is_some_and(|p| counts[index(p, n)] > 0))
                .count();
            let expected = if cells[i] == 2 {
                2
            } else if p[0] == 7 || p[1] == 7 {
                0
            } else {
                u32::from(counts[i] > 0 || neighbour_count >= 2)
            };
            assert_eq!(marked[i], expected, "mark {p:?}");
        }
        // Independent forward face gradient, preserving real solid/air seam pairs.
        q.write_buffer(&types, 0, bytemuck::cast_slice(&cells));
        let pressure: Vec<f32> = nodes(n)
            .map(|[x, y, z]| {
                (0.125 * x as f64 - 0.25 * y as f64
                    + 0.5 * (std::f64::consts::TAU * z as f64 / depth as f64).sin())
                    as f32
            })
            .collect();
        let pressure = buffer(&d, &pressure);
        let delta: [Arc<wgpu::Buffer>; 3] = std::array::from_fn(|_| buffer(&d, &vec![999f32; len]));
        let params = buffer(
            &d,
            &[
                8u32,
                8,
                depth as u32,
                0.001f32.to_bits(),
                0.5f32.to_bits(),
                0,
                0,
                0,
            ],
        );
        run(
            &d,
            &q,
            include_str!("../../game/src/gpu/shaders/density_position_grid_3d.wgsl"),
            "compute_position_grid",
            &[
                (0, &params, false),
                (1, &pressure, false),
                (2, &types, false),
                (3, &delta[0], true),
                (4, &delta[1], true),
                (5, &delta[2], true),
            ],
            &[("PERIODIC_Z", 1.)],
            g,
        );
        let ps: Vec<f32> = read(&d, &q, &pressure);
        let fields = delta.each_ref().map(|b| read::<f32>(&d, &q, b));
        for p in nodes(n) {
            let i = index(p, n);
            for (axis, field) in fields.iter().enumerate() {
                let adjacent = neighbour(p, axis, 1, n);
                let other = adjacent.map_or(0, |p| cells[index(p, n)]);
                let expected = if cells[i] == 2 || other == 2 {
                    0.
                } else {
                    (if other == 1 {
                        f64::from(ps[index(adjacent.unwrap(), n)])
                    } else {
                        0.
                    } - if cells[i] == 1 { f64::from(ps[i]) } else { 0. })
                        * 0.0005
                };
                assert!(
                    (f64::from(field[i]) - expected).abs() < 1e-8,
                    "gradient {p:?} axis{axis}"
                );
            }
        }
        let positions: Vec<[f32; 4]> = [0., 0.125, depth as f32 - 0.125, depth as f32 + 0.125]
            .into_iter()
            .map(|z| [3.25 * H, 4.75 * H, z * H, 0.75])
            .collect();
        let bpos = buffer(&d, &positions);
        let velocities = vec![[0.25f32, -0.125, 0.5, 0.]; 4];
        let bvel = buffer(&d, &velocities);
        let rho = buffer(&d, &[1f32, 1., 1., 2.5]);
        let status = buffer(&d, &[0u32; 4]);
        let params = buffer(
            &d,
            &[8u32, 8, depth as u32, 4, H.to_bits(), 1f32.to_bits(), 0, 0],
        );
        run(
            &d,
            &q,
            include_str!("../../game/src/gpu/shaders/density_correct_3d.wgsl"),
            "correct_positions",
            &[
                (0, &params, false),
                (1, &delta[0], false),
                (2, &delta[1], false),
                (3, &delta[2], false),
                (4, &types, false),
                (5, &bpos, true),
                (6, &bvel, true),
                (7, &rho, false),
                (8, &status, false),
            ],
            &[("PERIODIC_Z", 1.)],
            [1, 1, 1],
        );
        let got: Vec<[f32; 4]> = read(&d, &q, &bpos);
        for (i, p) in positions.iter().enumerate() {
            if i == 3 {
                assert_eq!(got[i], *p);
                continue;
            }
            for axis in 0..3 {
                let stagger = std::array::from_fn(|k| if k == axis { 1. } else { 0.5 });
                let mut expected = f64::from(p[axis])
                    + sample_images(n, [p[0], p[1], p[2]], &fields[axis], stagger) / f64::from(H);
                if axis == 2 {
                    expected = expected.rem_euclid(depth as f64 * f64::from(H));
                }
                assert!(
                    (f64::from(got[i][axis]) - expected).abs() < 1e-6,
                    "gather {i} axis{axis}"
                );
            }
            assert_eq!(got[i][3], p[3]);
        }
        assert_eq!(read::<[f32; 4]>(&d, &q, &bvel), velocities);
    }
}

#[test]
#[ignore = "requires real Metal GPU; run with --ignored"]
fn collocated_curl_and_confinement_match_analytic_periodic_flow() {
    let (d, q) = device();
    for depth in [2, 8, 12] {
        let n = [8, 8, depth];
        let len = n.into_iter().product::<usize>();
        let mut cells = vec![1u32; len];
        cells[index([3, 3, 0], n)] = 0;
        cells[index([4, 4, depth - 1], n)] = 2;
        let initial: [Vec<f32>; 3] = std::array::from_fn(|axis| {
            nodes(shape(n, axis))
                .map(|node| {
                    let p: [f64; 3] = std::array::from_fn(|a| {
                        (node[a] as f64 + if a == axis { 0. } else { 0.5 }) * f64::from(H)
                    });
                    let t = std::f64::consts::TAU * p[2] / (depth as f64 * f64::from(H));
                    match axis {
                        0 => (0.125 * p[0] + 2. * p[1] + 0.01 * t.sin()) as f32,
                        1 => (-p[0] + 0.5 * p[1] + 0.02 * t.cos()) as f32,
                        _ => ((0.5 * p[0] - p[1]) * t.cos()) as f32,
                    }
                })
                .collect()
        });
        let mut poisoned = initial.clone();
        poisoned[2][len..].fill(999.);
        let grid = poisoned.each_ref().map(|v| buffer(&d, v));
        let types = buffer(&d, &cells);
        let vort: [Arc<wgpu::Buffer>; 4] = std::array::from_fn(|_| buffer(&d, &vec![999f32; len]));
        let params = buffer(&d, &[8u32, 8, depth as u32, H.to_bits()]);
        run(
            &d,
            &q,
            include_str!("../../game/src/gpu/shaders/vorticity_3d.wgsl"),
            "compute_vorticity",
            &[
                (0, &params, false),
                (1, &grid[0], false),
                (2, &grid[1], false),
                (3, &grid[2], false),
                (4, &types, false),
                (5, &vort[0], true),
                (6, &vort[1], true),
                (7, &vort[2], true),
                (8, &vort[3], true),
            ],
            &[("PERIODIC_Z", 1.)],
            groups(n),
        );
        let actual = vort.each_ref().map(|b| read::<f32>(&d, &q, b));
        let step = std::f64::consts::TAU / depth as f64;
        let expected: Vec<[f64; 4]> = nodes(n)
            .map(|p| {
                if p[0] == 0 || p[0] == 7 || p[1] == 0 || p[1] == 7 || cells[index(p, n)] != 1 {
                    return [0.; 4];
                }
                let t = (p[2] as f64 + 0.5) * step;
                let a = -t.cos() * (step / 2.).cos() + 0.02 * t.sin() * step.sin() / f64::from(H);
                let b =
                    0.01 * t.cos() * step.sin() / f64::from(H) - 0.5 * t.cos() * (step / 2.).cos();
                [a, b, -3., (a * a + b * b + 9.).sqrt()]
            })
            .collect();
        for (i, e) in expected.iter().enumerate() {
            for axis in 0..4 {
                assert!(
                    (f64::from(actual[axis][i]) - e[axis]).abs() < 2e-5,
                    "curl depth{depth} cell{i} axis{axis}: {} vs {}",
                    actual[axis][i],
                    e[axis]
                );
            }
        }
        // Check confinement independently from each cell's normalized |curl| gradient.
        let force: Vec<[f64; 3]> = nodes(n)
            .map(|p| {
                let i = index(p, n);
                if p[0] < 2 || p[0] >= 6 || p[1] < 2 || p[1] >= 6 || cells[i] != 1 {
                    return [0.; 3];
                }
                if (0..3).any(|a| {
                    [-1, 1]
                        .into_iter()
                        .any(|s| cells[index(neighbour(p, a, s, n).unwrap(), n)] == 0)
                }) {
                    return [0.; 3];
                }
                let grad: [f64; 3] = std::array::from_fn(|a| {
                    0.5 * (f64::from(actual[3][index(neighbour(p, a, 1, n).unwrap(), n)])
                        - f64::from(actual[3][index(neighbour(p, a, -1, n).unwrap(), n)]))
                });
                let length = grad.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-5;
                let g = grad.map(|v| v / length);
                let omega = [
                    f64::from(actual[0][i]),
                    f64::from(actual[1][i]),
                    f64::from(actual[2][i]),
                ];
                [
                    (g[1] * omega[2] - g[2] * omega[1]) * 0.01,
                    (g[2] * omega[0] - g[0] * omega[2]) * 0.01,
                    (g[0] * omega[1] - g[1] * omega[0]) * 0.01,
                ]
            })
            .collect();
        let params = buffer(&d, &[8u32, 8, depth as u32, 0.01f32.to_bits()]);
        for (axis, entry) in [
            "apply_confinement_u",
            "apply_confinement_v",
            "apply_confinement_w",
        ]
        .into_iter()
        .enumerate()
        {
            run(
                &d,
                &q,
                include_str!("../../game/src/gpu/shaders/vorticity_confine_3d.wgsl"),
                entry,
                &[
                    (0, &params, false),
                    (1, &vort[0], false),
                    (2, &vort[1], false),
                    (3, &vort[2], false),
                    (4, &vort[3], false),
                    (5, &types, false),
                    (6, &grid[0], true),
                    (7, &grid[1], true),
                    (8, &grid[2], true),
                ],
                &[("PERIODIC_Z", 1.)],
                groups(shape(n, axis)),
            );
            let got: Vec<f32> = read(&d, &q, &grid[axis]);
            for node in nodes(shape(n, axis)) {
                let mut canonical = node;
                if axis == 2 {
                    canonical[2] %= depth;
                }
                let mut expected = f64::from(initial[axis][index(canonical, shape(n, axis))]);
                if node[axis] > 0 && node[axis] < n[axis] || axis == 2 {
                    let mut back = canonical;
                    if axis == 2 {
                        back[2] = (canonical[2] + depth - 1) % depth;
                    } else {
                        back[axis] -= 1;
                    }
                    expected +=
                        0.5 * (force[index(back, n)][axis] + force[index(canonical, n)][axis]);
                }
                assert!(
                    (f64::from(got[index(node, shape(n, axis))]) - expected).abs() < 1e-6,
                    "confinement axis{axis} node{node:?}"
                );
            }
            if axis == 2 {
                assert_eq!(&got[..64], &got[len..]);
            }
        }
    }
}

#[test]
#[ignore = "requires real Metal GPU; run with --ignored"]
fn signed_distance_interpolation_wraps_instead_of_creating_a_side_wall() {
    let (d, q) = device();
    for depth in [8, 12] {
        let n = [8, 8, depth];
        let values: Vec<f32> = nodes(n)
            .map(|[_, y, z]| {
                (y as f32 + 0.5) * H
                    - 0.125
                    - 0.025 * (std::f32::consts::TAU * (z as f32 + 0.5) / depth as f32).cos()
            })
            .collect();
        let input: Vec<[f32; 4]> = [
            -0.125,
            0.,
            0.125,
            depth as f32 - 0.125,
            depth as f32 + 0.125,
        ]
        .into_iter()
        .map(|z| [3.25 * H, 3.75 * H, z * H, 1.])
        .collect();
        let sdf = buffer(&d, &values);
        let p = buffer(&d, &input);
        let out = buffer(&d, &vec![0f32; input.len()]);
        let params = buffer(
            &d,
            &[
                8u32,
                8,
                depth as u32,
                input.len() as u32,
                H.to_bits(),
                0,
                0,
                0,
            ],
        );
        let source = format!(
            "{}\n{}",
            include_str!("../../game/src/gpu/shaders/sample_sdf_3d.wgsl"),
            r#"
struct Params { width:u32,height:u32,depth:u32,count:u32,cell_size:f32,pad0:u32,pad1:u32,pad2:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> sdf:array<f32>;
@group(0) @binding(2) var<storage,read> p:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> out:array<f32>;
fn cell_index(i:u32,j:u32,k:u32)->u32{return (k*params.height+j)*params.width+i;}
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>){if id.x<params.count{out[id.x]=sample_sdf(p[id.x].xyz);}}
"#
        );
        run(
            &d,
            &q,
            &source,
            "probe",
            &[
                (0, &params, false),
                (1, &sdf, false),
                (2, &p, false),
                (3, &out, true),
            ],
            &[("SDF_PERIODIC_Z", 1.)],
            [1, 1, 1],
        );
        let got: Vec<f32> = read(&d, &q, &out);
        for (i, p) in input.iter().enumerate() {
            let expected = sample_images(n, [p[0], p[1], p[2]], &values, [0.5; 3]);
            assert!((f64::from(got[i]) - expected).abs() < 1e-6);
            assert!(got[i] > 0., "periodic seam became a solid wall");
        }
    }
}

#[test]
#[ignore = "requires real Metal GPU; run with --ignored"]
fn ordinary_w_boundary_passes_cover_upper_z_and_apply_both_slip_factors() {
    let (d, q) = device();
    for depth in [12, 38] {
        let n = [12, 12, depth];
        // Reproduce the former host launch against the actual BC shader.
        // Its ceil(depth/8) launch cannot reach this upper-quarter W face.
        let old_params = buffer(&d, &[12u32, 12, depth as u32, 0, 0.5f32.to_bits(), 0, 0, 0]);
        let old_cells = buffer(&d, &vec![0u32; n.into_iter().product()]);
        let old_grid: [Arc<wgpu::Buffer>; 3] = std::array::from_fn(|axis| {
            buffer(&d, &vec![1f32; shape(n, axis).into_iter().product()])
        });
        for _ in 0..2 {
            run(
                &d,
                &q,
                include_str!("../../game/src/gpu/shaders/enforce_bc_3d.wgsl"),
                "enforce_bc_w",
                &[
                    (0, &old_params, false),
                    (1, &old_cells, false),
                    (2, &old_grid[0], true),
                    (3, &old_grid[1], true),
                    (4, &old_grid[2], true),
                ],
                &[],
                [2, 2, ((depth + 1) as u32).div_ceil(8)],
            );
        }
        let old_w: Vec<f32> = read(&d, &q, &old_grid[2]);
        assert_eq!(old_w[index([5, 2, depth - 3], shape(n, 2))], 1.);
        let mut p = vec![Vec3::new(5.25 * H, 2.5 * H, (depth as f32 - 3.25) * H)];
        let mut v = vec![Vec3::Z];
        let mut c = vec![Mat3::ZERO];
        let mut s = GpuFlip3D::new(&d, 12, 12, depth as u32, H, 1);
        s.slip_factor = 0.5;
        s.flip_ratio = 0.;
        s.density_projection_enabled = false;
        s.vorticity_epsilon = 0.;
        s.sediment_porosity_drag = 0.;
        s.sediment_rest_particles = 0.;
        s.step(
            &d,
            &q,
            &mut p,
            &mut v,
            &mut c,
            &[1.],
            &vec![0; n.into_iter().product()],
            Some(&vec![1.; n.into_iter().product()]),
            None,
            0.001,
            0.,
            0.,
            0,
        );
        assert!(
            (v[0].z - 0.25).abs() < 1e-6,
            "upper Z depth{depth}: {:?}; each of two BC passes must multiply by 0.5",
            v[0]
        );
    }
}

#[test]
#[ignore = "requires real Metal GPU; run with --ignored"]
fn periodic_water_advects_through_multiple_seams_and_retains_velocity() {
    let (d, q) = device();
    for depth in [8, 12] {
        let n = [12, 12, depth];
        let mut p = vec![Vec3::new(5.25 * H, 5.25 * H, depth as f32 * H - H / 8.)];
        let mut v = vec![Vec3::Z * 2.];
        let mut c = vec![Mat3::ZERO];
        let mut s = GpuFlip3D::new_periodic_z(&d, 12, 12, depth as u32, H, 1);
        assert!(s.is_periodic_z());
        s.flip_ratio = 1.;
        s.density_projection_enabled = false;
        s.vorticity_epsilon = 0.;
        let cells = vec![0; n.into_iter().product()];
        let sdf = vec![1.; cells.len()];
        let initial = p[0];
        for tick in 0..64 {
            s.step(
                &d,
                &q,
                &mut p,
                &mut v,
                &mut c,
                &[1.],
                &cells,
                Some(&sdf),
                None,
                0.01,
                0.,
                0.,
                0,
            );
            let expected = (initial.z + 2. * 0.01 * (tick + 1) as f32).rem_euclid(depth as f32 * H);
            let distance = (p[0].z - expected).abs();
            assert!(
                distance.min(depth as f32 * H - distance) < 2e-6,
                "tick{tick}: {:?} vs {expected}",
                p[0]
            );
            assert!(
                (v[0] - Vec3::Z * 2.).length() < 2e-5,
                "uniform velocity drift: {:?}",
                v[0]
            );
            assert!(p[0].z >= 0. && p[0].z < depth as f32 * H);
            assert!(c[0].is_finite());
        }
    }
}

struct Water {
    s: GpuFlip3D,
    p: Vec<Vec3>,
    v: Vec<Vec3>,
    c: Vec<Mat3>,
    cells: Vec<u32>,
    sdf: Vec<f32>,
}
impl Water {
    fn new(d: &wgpu::Device, depth: usize) -> Self {
        let n = [24, 14, depth];
        let cells: Vec<u32> = nodes(n)
            .map(|[x, y, _]| u32::from(y == 0 || x == 0 || ((x == 9 || x == 15) && y < 3)) * 2)
            .collect();
        let sdf: Vec<f32> = nodes(n)
            .map(|[x, y, _]| {
                let px = (x as f32 + 0.5) * H;
                let py = (y as f32 + 0.5) * H;
                let mut distance = py - H;
                for rx in [9f32, 15.] {
                    let q = Vec3::new((px - (rx + 0.5) * H).abs() - H / 2., py.abs() - 3. * H, 0.);
                    let box_sdf =
                        Vec3::new(q.x.max(0.), q.y.max(0.), 0.).length() + q.x.max(q.y).min(0.);
                    distance = distance.min(box_sdf);
                }
                distance.min(px - H)
            })
            .collect();
        let mut p = Vec::new();
        for [x, y, z] in nodes(n) {
            if !(3..21).contains(&x) || !(3..6).contains(&y) {
                continue;
            }
            for a in [0.25, 0.75] {
                for b in [0.25, 0.75] {
                    for c in [0.25, 0.75] {
                        p.push(Vec3::new(x as f32 + a, y as f32 + b, z as f32 + c) * H);
                    }
                }
            }
        }
        let v = vec![Vec3::X * 0.4; p.len()];
        let c = vec![Mat3::ZERO; p.len()];
        let mut s = GpuFlip3D::new_periodic_z(d, 24, 14, depth as u32, H, p.len());
        s.open_boundaries = 10;
        Self {
            s,
            p,
            v,
            c,
            cells,
            sdf,
        }
    }
    fn step(&mut self, d: &wgpu::Device, q: &wgpu::Queue) {
        let densities = vec![1.; self.p.len()];
        self.s.step(
            d,
            q,
            &mut self.p,
            &mut self.v,
            &mut self.c,
            &densities,
            &self.cells,
            Some(&self.sdf),
            None,
            1. / 240.,
            -9.81,
            0.25,
            40,
        );
    }
}

#[test]
#[ignore = "requires real Metal GPU; run with --ignored"]
fn evolving_riffle_water_matches_repeated_domains_and_survives_storage_growth() {
    let (d, q) = device();
    for depth in [8, 12] {
        let mut a = Water::new(&d, depth);
        let mut b = Water::new(&d, depth * 2);
        let count = a.p.len();
        let initial_y = a.p.iter().map(|p| p.y).sum::<f32>() / count as f32;
        for tick in 0..32 {
            if tick == 8 {
                let capacity = a.s.particle_capacity();
                a.s.reallocate_for_upload(&d, &q, capacity + 128).unwrap();
                assert!(a.s.is_periodic_z());
            }
            a.step(&d, &q);
            b.step(&d, &q);
            let mut max_pos = 0f32;
            let mut max_vel = 0f32;
            for i in 0..count {
                let p = a.p[i];
                let other = b.p[i];
                let dz = (p.z - other.z.rem_euclid(depth as f32 * H)).abs();
                let distance =
                    Vec3::new(p.x - other.x, p.y - other.y, dz.min(depth as f32 * H - dz));
                max_pos = max_pos.max(distance.length());
                max_vel = max_vel.max((a.v[i] - b.v[i]).length());
                assert!(p.is_finite() && a.v[i].is_finite() && a.c[i].is_finite());
                assert!(p.z >= 0. && p.z < depth as f32 * H);
                let voxel = (p / H).floor().as_uvec3();
                if voxel.x < 24 && voxel.y < 14 {
                    assert_ne!(
                        a.cells[index(
                            [voxel.x as usize, voxel.y as usize, voxel.z as usize],
                            [24, 14, depth]
                        )],
                        2,
                        "water entered a riffle/floor voxel at tick {tick}: {p:?}"
                    );
                }
            }
            assert!(
                max_pos < 2e-5 && max_vel < 0.002,
                "depth{depth} tick{tick}: pos{max_pos} vel{max_vel}"
            );
        }
        let mean_y = a.p.iter().map(|p| p.y).sum::<f32>() / count as f32;
        assert!(
            mean_y < initial_y - 0.002,
            "gravity must evolve the water, not freeze it"
        );
        assert_eq!(a.p.len() * 2, b.p.len());
    }
}

#[test]
#[ignore = "requires real Metal GPU; run with --ignored"]
fn resident_four_substep_frames_equal_complete_sync_water_states() {
    let (d, q) = device();
    let mut sync = Water::new(&d, 8);
    let mut resident = Water::new(&d, 8);
    let rho = vec![1.; resident.p.len()];
    for frame in 0..6 {
        for _ in 0..4 {
            sync.step(&d, &q);
        }
        resident.s.step_no_readback(
            &d,
            &q,
            &mut resident.p,
            &mut resident.v,
            &mut resident.c,
            &rho,
            &resident.cells,
            Some(&resident.sdf),
            None,
            1. / 240.,
            -9.81,
            0.25,
            40,
        );
        for _ in 1..4 {
            resident.s.step_in_place(
                &d,
                &q,
                resident.p.len() as u32,
                &resident.cells,
                Some(&resident.sdf),
                None,
                1. / 240.,
                -9.81,
                0.25,
                40,
            );
        }
        assert!(resident.s.request_readback(&d, &q, resident.p.len()));
        d.poll(wgpu::Maintain::Wait);
        assert_eq!(
            resident
                .s
                .try_readback(&d, &mut resident.p, &mut resident.v, &mut resident.c),
            Some(rho.len())
        );
        assert_eq!(sync.p, resident.p, "position frame{frame}");
        assert_eq!(sync.v, resident.v, "velocity frame{frame}");
        assert_eq!(sync.c, resident.c, "affine frame{frame}");
    }
}

//! Physical-face/image oracles for the experimental repeated-Z APIC transfers.
//! These controls do not qualify advection, collisions or a complete strip solver.
use game::gpu::{
    g2p_3d::{GpuG2p3D, SedimentParams3D},
    p2g_3d::GpuP2g3D,
};
use glam::{Mat3, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const H: f32 = 0.0625; // Exact dyadic coordinates isolate quantisation from indexing.
const SCALE: f64 = 100_000.;
type Shape = [usize; 3];

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });
    let a = pollster::block_on(instance.request_adapter(&Default::default()))
        .expect("real Metal adapter");
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
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::UNIFORM,
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
fn index(p: Shape, n: Shape) -> usize {
    (p[2] * n[1] + p[1]) * n[0] + p[0]
}
fn shapes(n: Shape) -> [Shape; 3] {
    std::array::from_fn(|a| {
        let mut s = n;
        s[a] += 1;
        s
    })
}
fn nodes(n: Shape) -> impl Iterator<Item = Shape> {
    (0..n[2]).flat_map(move |z| (0..n[1]).flat_map(move |y| (0..n[0]).map(move |x| [x, y, z])))
}
fn bspline(x: f64) -> f64 {
    let a = x.abs();
    if a < 0.5 {
        0.75 - a * a
    } else if a < 1.5 {
        0.5 * (1.5 - a).powi(2)
    } else {
        0.
    }
}
fn offset(p: [f32; 4], node: Shape, axis: usize, image: i32, depth: usize) -> [f64; 3] {
    std::array::from_fn(|k| {
        ((node[k] as f64 + if k == axis { 0. } else { 0.5 })
            + if k == 2 {
                f64::from(image) * depth as f64
            } else {
                0.
            })
            * f64::from(H)
            - f64::from(p[k])
    })
}
fn weight(r: [f64; 3]) -> f64 {
    r.into_iter().map(|r| bspline(r / f64::from(H))).product()
}

#[derive(Clone)]
struct Particle {
    p: [f32; 4],
    v: [f32; 4],
    rho: f32,
    c: [[f32; 4]; 3],
}
fn particles(depth: usize) -> Vec<Particle> {
    [
        0.,
        0.125,
        0.75,
        depth as f32 - 0.125,
        depth as f32 - 0.75,
        depth as f32 * 0.5,
        -0.125,
        depth as f32 + 0.125,
    ]
    .into_iter()
    .enumerate()
    .map(|(i, z)| Particle {
        p: [
            4.25 * H,
            4.75 * H,
            z * H,
            if i % 2 == 0 { 0.5 } else { 1.25 },
        ],
        v: [0.25, -0.125, 0.5, 0.],
        rho: if i % 3 == 0 { 2.5 } else { 1. },
        c: [
            [0.125, -0.25, 0.5, 0.],
            [-0.5, 0.25, -0.125, 0.],
            [0.25, 0.5, -0.25, 0.],
        ],
    })
    .collect()
}
fn planes(d: &wgpu::Device, p: &[Particle]) -> [Arc<wgpu::Buffer>; 6] {
    [
        buffer(d, &p.iter().map(|p| p.p).collect::<Vec<_>>()),
        buffer(d, &p.iter().map(|p| p.v).collect::<Vec<_>>()),
        buffer(d, &p.iter().map(|p| p.rho).collect::<Vec<_>>()),
        buffer(d, &p.iter().map(|p| p.c[0]).collect::<Vec<_>>()),
        buffer(d, &p.iter().map(|p| p.c[1]).collect::<Vec<_>>()),
        buffer(d, &p.iter().map(|p| p.c[2]).collect::<Vec<_>>()),
    ]
}

/// Compile the production scatter with readable accumulators, independently of
/// the host transfer wrapper. The oracle below enumerates physical faces and
/// periodic images, rather than reproducing the shader's particle stencil.
fn raw_scatter(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    n: Shape,
    p: &[Particle],
    include: bool,
    paired: bool,
) -> [Vec<i32>; 8] {
    let b = planes(d, p);
    let sizes = shapes(n).map(|s| s.into_iter().product::<usize>());
    let counts = n.into_iter().product::<usize>();
    let accum: [Arc<wgpu::Buffer>; 8] = std::array::from_fn(|i| {
        buffer(
            d,
            &vec![
                0i32;
                if i < 6 {
                    let m = sizes[i / 2];
                    if include {
                        m.div_ceil(64) * 64 + m
                    } else {
                        m
                    }
                } else {
                    counts
                }
            ],
        )
    });
    let params = buffer(
        d,
        &[
            H.to_bits(),
            n[0] as u32,
            n[1] as u32,
            n[2] as u32,
            p.len() as u32,
            u32::from(include),
            u32::from(paired),
            0,
        ],
    );
    let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../game/src/gpu/shaders/p2g_scatter_3d.wgsl").into(),
        ),
    });
    let constants = std::collections::HashMap::from([("PERIODIC_Z".into(), 1.)]);
    let pipe = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("scatter"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        cache: None,
    });
    let bindings = [
        &params, &b[0], &b[1], &b[3], &b[4], &b[5], &accum[0], &accum[1], &accum[2], &accum[3],
        &accum[4], &accum[5], &accum[6], &b[2], &accum[7],
    ];
    let entries: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &entries,
    });
    let mut e = d.create_command_encoder(&Default::default());
    {
        let mut pass = e.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((p.len() as u32).div_ceil(128), 1, 1);
    }
    q.submit(Some(e.finish()));
    accum.each_ref().map(|b| read(d, q, b))
}

#[test]
#[ignore = "requires real Metal GPU; run explicitly with --ignored"]
fn scatter_preserves_physical_face_mass_momentum_and_single_home_counts() {
    let (d, q) = device();
    for depth in [2, 8, 12] {
        let n = [10, 10, depth];
        let p = particles(depth);
        for (include, paired) in [(false, false), (true, false), (true, true)] {
            let got = raw_scatter(&d, &q, n, &p, include, paired);
            for (axis, shape) in shapes(n).into_iter().enumerate() {
                let len = shape.into_iter().product::<usize>();
                let mut expected = vec![[0i32; 2]; got[axis * 2].len()];
                let mut physical = shape;
                physical[2] = depth;
                for node in nodes(physical) {
                    for particle in &p {
                        if particle.rho > 1. && !include {
                            continue;
                        }
                        let target = index(node, shape)
                            + if paired && particle.rho <= 1. {
                                len.div_ceil(64) * 64
                            } else {
                                0
                            };
                        for image in -1..=1 {
                            let r = offset(particle.p, node, axis, image, depth);
                            let w = weight(r);
                            if w == 0. {
                                continue;
                            }
                            let mass = f64::from(particle.rho * particle.p[3]);
                            let v = f64::from(particle.v[axis])
                                + (0..3)
                                    .map(|k| f64::from(particle.c[axis][k]) * r[k])
                                    .sum::<f64>();
                            expected[target][0] += (mass * v * w * SCALE) as i32;
                            expected[target][1] += (mass * w * SCALE) as i32;
                        }
                    }
                }
                for (i, e) in expected.iter().enumerate() {
                    for c in 0..2 {
                        assert!((got[2*axis+c][i]-e[c]).abs()<=2,"depth{depth} include{include} paired{paired} axis{axis} node{i} component{c}: {} vs {}",got[2*axis+c][i],e[c]);
                    }
                }
                let mass: f64 = p
                    .iter()
                    .filter(|p| include || p.rho <= 1.)
                    .map(|p| f64::from(p.rho * p.p[3]))
                    .sum();
                let momentum: f64 = p
                    .iter()
                    .filter(|p| include || p.rho <= 1.)
                    .map(|p| f64::from(p.rho * p.p[3] * p.v[axis]))
                    .sum();
                let epsilon = p.len() as f64 * 27. / SCALE;
                assert!(
                    (got[2 * axis + 1]
                        .iter()
                        .map(|v| f64::from(*v) / SCALE)
                        .sum::<f64>()
                        - mass)
                        .abs()
                        < epsilon
                );
                assert!(
                    (got[2 * axis]
                        .iter()
                        .map(|v| f64::from(*v) / SCALE)
                        .sum::<f64>()
                        - momentum)
                        .abs()
                        < epsilon
                );
            }
            let mut expected_water = vec![0; n.into_iter().product()];
            let mut expected_solid = expected_water.clone();
            for p in &p {
                if !include && p.rho > 1. {
                    continue;
                }
                let k = ((p.p[2] / H).floor() as i32).rem_euclid(depth as i32) as usize;
                let i = index([(p.p[0] / H) as usize, (p.p[1] / H) as usize, k], n);
                if p.rho <= 1. {
                    expected_water[i] += 1;
                } else {
                    expected_solid[i] += 1;
                }
            }
            assert_eq!(got[6], expected_water);
            assert_eq!(got[7], expected_solid);
        }
    }
}

fn scatter(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    n: Shape,
    p: &[Particle],
    paired: bool,
) -> (GpuP2g3D, [Arc<wgpu::Buffer>; 6]) {
    let b = planes(d, p);
    let mut solver = GpuP2g3D::new_periodic_z(
        d,
        n[0] as u32,
        n[1] as u32,
        n[2] as u32,
        p.len(),
        paired,
        b[0].clone(),
        b[1].clone(),
        b[2].clone(),
        b[3].clone(),
        b[4].clone(),
        b[5].clone(),
    );
    solver.paired_water = paired;
    solver.prepare(q, p.len() as u32, H);
    let mut e = d.create_command_encoder(&Default::default());
    solver.encode(&mut e, p.len() as u32);
    q.submit(Some(e.finish()));
    (solver, b)
}

#[test]
#[ignore = "requires real Metal GPU; run explicitly with --ignored"]
fn divide_copies_canonical_seam_velocity_and_support_without_stale_data() {
    let (d, q) = device();
    for depth in [2, 8, 12] {
        for paired in [false, true] {
            let n = [10, 10, depth];
            let mut p = particles(depth);
            if !paired {
                p.iter_mut().for_each(|p| p.rho = 1.);
            }
            let (solver, b) = scatter(&d, &q, n, &p, paired);
            for round in 0..2 {
                if round == 1 {
                    p.iter_mut().for_each(|p| p.v = [0.; 4]);
                    q.write_buffer(
                        &b[1],
                        0,
                        bytemuck::cast_slice(&p.iter().map(|p| p.v).collect::<Vec<_>>()),
                    );
                    for grid in [
                        &solver.grid_u_buffer,
                        &solver.grid_v_buffer,
                        &solver.grid_w_buffer,
                    ] {
                        q.write_buffer(
                            grid,
                            0,
                            bytemuck::cast_slice(&vec![777f32; grid.size() as usize / 4]),
                        );
                    }
                    solver.prepare(&q, p.len() as u32, H);
                    let mut e = d.create_command_encoder(&Default::default());
                    solver.encode(&mut e, p.len() as u32);
                    q.submit(Some(e.finish()));
                }
                let raw = raw_scatter(&d, &q, n, &p, paired, paired);
                for (axis, grid) in [
                    &solver.grid_u_buffer,
                    &solver.grid_v_buffer,
                    &solver.grid_w_buffer,
                ]
                .into_iter()
                .enumerate()
                {
                    let got: Vec<f32> = read(&d, &q, grid);
                    let shape = shapes(n)[axis];
                    let len = shape.into_iter().product::<usize>();
                    let water_offset = if paired { len.div_ceil(64) * 64 } else { 0 };
                    for node in nodes(shape) {
                        let dst = index(node, shape);
                        let mut src = node;
                        if axis == 2 && node[2] == depth {
                            src[2] = 0;
                        }
                        let src = index(src, shape);
                        let water_sum = raw[axis * 2][water_offset + src];
                        let water_weight = raw[axis * 2 + 1][water_offset + src];
                        let sum = raw[axis * 2][src] + if paired { water_sum } else { 0 };
                        let w = raw[axis * 2 + 1][src] + if paired { water_weight } else { 0 };
                        let velocity = if w > 0 {
                            sum as f32 * 0.00001 / (w as f32 * 0.00001)
                        } else {
                            0.
                        };
                        // Metal may lower division to a reciprocal multiply;
                        // accept at most a few f32 rounding units here.
                        assert!(
                            (got[dst] - velocity).abs()
                                <= 4. * f32::EPSILON * velocity.abs().max(1.),
                            "axis{axis} node{node:?}"
                        );
                        let water = if water_weight > 0 {
                            water_sum as f32 * 0.00001 / (water_weight as f32 * 0.00001)
                        } else {
                            0.
                        };
                        if paired {
                            assert!(
                                (got[water_offset + dst] - water).abs()
                                    <= 4. * f32::EPSILON * water.abs().max(1.)
                            );
                        }
                        assert_eq!(got[water_offset + len + dst], f32::from(water_weight > 0));
                        if axis == 2 && node[2] == depth {
                            assert_eq!(got[dst].to_bits(), got[src].to_bits());
                            assert_eq!(
                                got[water_offset + len + dst].to_bits(),
                                got[water_offset + len + src].to_bits()
                            );
                            if paired {
                                assert_eq!(
                                    got[water_offset + dst].to_bits(),
                                    got[water_offset + src].to_bits()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// Enumerate each physical face and its three images; this oracle has no nearest
// node/base/stencil-index logic in common with the production gather.
fn gather_oracle(n: Shape, p: [f32; 4], field: &[Vec<f32>; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [0.; 3];
    let mut c = [[0.; 3]; 3];
    for axis in 0..3 {
        let shape = shapes(n)[axis];
        let mut physical = shape;
        physical[2] = n[2];
        let mut total = 0.;
        for node in nodes(physical) {
            for image in -1..=1 {
                let r = offset(p, node, axis, image, n[2]);
                let w = weight(r);
                let u = f64::from(field[axis][index(node, shape)]);
                v[axis] += w * u;
                total += w;
                for (k, r) in r.into_iter().enumerate() {
                    c[axis][k] += w * u * r * 4. / f64::from(H * H);
                }
            }
        }
        v[axis] /= total;
        c[axis].iter_mut().for_each(|c| *c /= total);
    }
    (v, c)
}
fn gather(
    d: &wgpu::Device,
    q: &wgpu::Queue,
    n: Shape,
    p: &[Particle],
    fields: &[Vec<f32>; 3],
    old: &[Vec<f32>; 3],
    flip: f32,
) -> (Vec<Vec3>, Vec<Mat3>) {
    let b = planes(d, p);
    let grids = fields.each_ref().map(|f| buffer(d, f));
    let olds = old.each_ref().map(|f| buffer(d, f));
    let vort = buffer(d, &vec![0f32; n.into_iter().product()]);
    let solver = GpuG2p3D::new_periodic_z(
        d,
        n[0] as u32,
        n[1] as u32,
        n[2] as u32,
        p.len(),
        b[0].clone(),
        b[1].clone(),
        b[3].clone(),
        b[4].clone(),
        b[5].clone(),
        b[2].clone(),
        &grids[0],
        &grids[1],
        &grids[2],
        &olds[0],
        &olds[1],
        &olds[2],
        &vort,
        &grids[0],
        &grids[1],
        &grids[2],
        [0; 3],
        false,
    );
    solver.upload_params(
        q,
        p.len() as u32,
        H,
        0.001,
        flip,
        SedimentParams3D::default(),
    );
    let mut e = d.create_command_encoder(&Default::default());
    solver.encode(&mut e, p.len() as u32);
    q.submit(Some(e.finish()));
    let mut v = vec![Vec3::ZERO; p.len()];
    let mut c = vec![Mat3::ZERO; p.len()];
    solver.download(d, q, p.len() as u32, &mut v, &mut c);
    (v, c)
}

#[test]
#[ignore = "requires real Metal GPU; run explicitly with --ignored"]
fn gather_matches_image_oracle_for_periodic_affine_pic_and_flip_fields() {
    let (d, q) = device();
    for depth in [2, 8, 12] {
        let n = [10, 10, depth];
        let mut p = particles(depth);
        p.iter_mut().for_each(|p| p.rho = 1.);
        for kind in 0..3 {
            let mut fields: [Vec<f32>; 3] = std::array::from_fn(|axis| {
                let shape = shapes(n)[axis];
                nodes(shape)
                    .map(|node| {
                        let r = offset([0.; 4], node, axis, 0, depth);
                        match kind {
                            0 => 0.25,
                            1 => 0.25 + (0.125 * r[0] - 0.25 * r[1]) as f32,
                            _ => {
                                0.25 + (0.125 * r[0] - 0.25 * r[1]
                                    + 0.0625
                                        * (std::f64::consts::TAU * r[2]
                                            / (depth as f64 * f64::from(H)))
                                        .sin()) as f32
                            }
                        }
                    })
                    .collect()
            });
            // Canonical readers must ignore this allocated redundant layer.
            fields[2][n.into_iter().product::<usize>()..].fill(999.);
            let old = fields
                .each_ref()
                .map(|f| f.iter().map(|v| v - 0.125).collect());
            for flip in [0., 0.97, 1.] {
                let (v, c) = gather(&d, &q, n, &p, &fields, &old, flip);
                for (i, particle) in p.iter().enumerate() {
                    let (pic, affine) = gather_oracle(n, particle.p, &fields);
                    let (before, _) = gather_oracle(n, particle.p, &old);
                    for axis in 0..3 {
                        let expected = (1. - f64::from(flip)) * pic[axis]
                            + f64::from(flip)
                                * (f64::from(particle.v[axis]) + pic[axis] - before[axis]);
                        assert!(
                            (f64::from(v[i][axis]) - expected).abs() < 2e-6,
                            "depth{depth} kind{kind} flip{flip} p{i} axis{axis}"
                        );
                        for k in 0..3 {
                            assert!(
                                (f64::from(c[i].to_cols_array()[axis * 3 + k]) - affine[axis][k])
                                    .abs()
                                    < 2e-4,
                                "affine depth{depth} p{i} axis{axis} k{k}: {:?} vs {affine:?}",
                                c[i]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires real Metal GPU; run explicitly with --ignored"]
fn tiled_domains_match_strip_scatter_gather_and_affine_state() {
    let (d, q) = device();
    for depth in [8, 12] {
        let n = [10, 10, depth];
        let mut small = particles(depth);
        small.iter_mut().for_each(|p| p.rho = 1.);
        let big: Vec<_> = (0..4)
            .flat_map(|tile| {
                small.iter().cloned().map(move |mut p| {
                    p.p[2] += tile as f32 * depth as f32 * H;
                    p
                })
            })
            .collect();
        let (a, _) = scatter(&d, &q, n, &small, false);
        let m = [10, 10, depth * 4];
        let (b, _) = scatter(&d, &q, m, &big, false);
        let fa =
            [&a.grid_u_buffer, &a.grid_v_buffer, &a.grid_w_buffer].map(|v| read::<f32>(&d, &q, v));
        let fb =
            [&b.grid_u_buffer, &b.grid_v_buffer, &b.grid_w_buffer].map(|v| read::<f32>(&d, &q, v));
        for axis in 0..3 {
            let shape = shapes(n)[axis];
            let large = shapes(m)[axis];
            let mut physical = shape;
            physical[2] = depth;
            for node in nodes(physical) {
                for tile in 0..4 {
                    let mut copy = node;
                    copy[2] += tile * depth;
                    assert_eq!(fa[axis][index(node, shape)], fb[axis][index(copy, large)]);
                }
            }
        }
        let (va, ca) = gather(&d, &q, n, &small, &fa, &fa, 0.97);
        let (vb, cb) = gather(&d, &q, m, &big, &fb, &fb, 0.97);
        for tile in 0..4 {
            for i in 0..small.len() {
                assert_eq!(va[i], vb[tile * small.len() + i]);
                assert_eq!(ca[i], cb[tile * small.len() + i]);
            }
        }
    }
}

#[test]
#[ignore = "requires real Metal GPU; run explicitly with --ignored"]
fn sediment_carrier_uses_wrapped_support_and_preserves_missing_components() {
    let (d, q) = device();
    let n = [10, 10, 8];
    let mut p = particles(8);
    p.truncate(2);
    for particle in &mut p {
        particle.rho = 2.5;
        particle.v = [0.4; 4];
    }
    p[0].p[2] = -0.125 * H;
    p[1].p[2] = 8.125 * H;
    let b = planes(&d, &p);
    let grids: [Arc<wgpu::Buffer>; 3] = std::array::from_fn(|axis| {
        let shape = shapes(n)[axis];
        let len = shape.into_iter().product::<usize>();
        let mut v = vec![0f32; len * 2];
        for node in nodes(shape) {
            let i = index(node, shape);
            if axis == 0 && node[2] == 7 {
                v[len + i] = 1.;
            }
            if axis == 2 && node[2] == 0 {
                v[i] = 1.25;
                v[len + i] = 1.;
            }
        }
        buffer(&d, &v)
    });
    let mut vorticity = vec![0f32; 800];
    vorticity[index([4, 4, 7], n)] = 1.;
    let vort = buffer(&d, &vorticity);
    let solver = GpuG2p3D::new_periodic_z(
        &d,
        10,
        10,
        8,
        2,
        b[0].clone(),
        b[1].clone(),
        b[3].clone(),
        b[4].clone(),
        b[5].clone(),
        b[2].clone(),
        &grids[0],
        &grids[1],
        &grids[2],
        &grids[0],
        &grids[1],
        &grids[2],
        &vort,
        &grids[0],
        &grids[1],
        &grids[2],
        [0; 3],
        true,
    );
    let params = SedimentParams3D {
        gravity: 0.,
        settling_velocity: 0.01,
        vorticity_lift: 1.,
        vorticity_threshold: 0.,
        ..Default::default()
    };
    solver.upload_params(&q, 2, H, 0.001, 0., params);
    let mut e = d.create_command_encoder(&Default::default());
    solver.encode(&mut e, 2);
    q.submit(Some(e.finish()));
    let mut v = [Vec3::ZERO; 2];
    let mut c = [Mat3::ZERO; 2];
    solver.download(&d, &q, 2, &mut v, &mut c);
    let a = 1. - (-0.01f32).exp();
    let expected = Vec3::new(0.4 * (1. - a), 0.4, 0.4 * (1. - a) + 1.25 * a);
    for (i, actual) in v.iter().enumerate() {
        // The particle just below zero samples the last cell's lift; the one
        // above the period samples cell zero. Carrier Y has no supported faces.
        let expected = expected
            + if i == 0 {
                Vec3::Y * 0.000009
            } else {
                Vec3::ZERO
            };
        assert!(
            (*actual - expected).length() < 1e-6,
            "carrier {i}: {:?} vs {expected:?}",
            actual
        );
    }
}

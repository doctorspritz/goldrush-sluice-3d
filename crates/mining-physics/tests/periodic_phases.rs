//! Conservative periodic water/mineral exchange. Physical marker/bed width
//! ownership, periodic DEM and calibrated sluice recovery remain separate work.
use game::gpu::{
    flip_3d::GpuFlip3D,
    g2p_3d::{GpuG2p3D, SedimentParams3D},
    phase_transfer_3d::PhaseTransfer3D,
};
use glam::{DVec3, Mat3, Vec3};
use mining_physics::phase_exchange::{FaceResponse, GrainSample};
use std::sync::Arc;
use wgpu::util::DeviceExt;
const H: f32 = 0.0625;
const DT: f32 = 1. / 120.;
type Shape = [usize; 3];
fn device() -> (wgpu::Device, wgpu::Queue) {
    let i = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..Default::default()
    });
    let a = pollster::block_on(i.request_adapter(&Default::default())).expect("real Metal GPU");
    eprintln!("adapter {:?}", a.get_info());
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
fn nodes(n: Shape) -> impl Iterator<Item = Shape> {
    (0..n[2]).flat_map(move |z| (0..n[1]).flat_map(move |y| (0..n[0]).map(move |x| [x, y, z])))
}
fn shape(n: Shape, a: usize) -> Shape {
    let mut s = n;
    s[a] += 1;
    s
}
fn weight(x: f64) -> f64 {
    let a = x.abs();
    if a < 0.5 {
        0.75 - a * a
    } else if a < 1.5 {
        0.5 * (1.5 - a).powi(2)
    } else {
        0.
    }
}
fn offset(p: [f32; 4], node: Shape, axis: usize, image: i32, depth: usize) -> DVec3 {
    DVec3::from_array(std::array::from_fn(|a| {
        ((node[a] as f64 + if a == axis { 0. } else { 0.5 })
            + if a == 2 {
                f64::from(image) * depth as f64
            } else {
                0.
            })
            * f64::from(H)
            - f64::from(p[a])
    }))
}
fn kernel(delta: DVec3) -> f64 {
    let x = delta / f64::from(H);
    weight(x.x) * weight(x.y) * weight(x.z)
}
#[derive(Clone)]
struct Particle {
    p: [f32; 4],
    v: [f32; 4],
    rho: f32,
    c: [[f32; 4]; 3],
}
fn particles(depth: usize, affine: bool) -> Vec<Particle> {
    let mut p = Vec::new();
    for z in [0.125, 0.75, depth as f32 - 0.125, depth as f32 - 0.75] {
        for (rho, vol) in [(1., 1.), (2.65, 0.25), (19.3, 0.0625)] {
            let x = 4.25 * H;
            let y = 4.75 * H;
            let c = if affine {
                [[0., -0.25, 0., 0.], [0.25, 0., 0., 0.], [0.; 4]]
            } else {
                [[0.; 4]; 3]
            };
            let v = if affine {
                [-0.25 * y, 0.25 * x, 0.03, 0.]
            } else if rho == 1. {
                [0.08, -0.01, 0.02, 0.]
            } else {
                [-0.03, 0.07, -0.01, 0.]
            };
            p.push(Particle {
                p: [x, y, z * H, vol],
                v,
                rho,
                c,
            });
        }
    }
    p
}
struct Operator {
    n: Shape,
    phase: PhaseTransfer3D,
    p: Arc<wgpu::Buffer>,
    v: Arc<wgpu::Buffer>,
    rows: [Arc<wgpu::Buffer>; 3],
    rho: Arc<wgpu::Buffer>,
    meta: Arc<wgpu::Buffer>,
    grid: [Arc<wgpu::Buffer>; 3],
    old: [Arc<wgpu::Buffer>; 3],
    vort: Arc<wgpu::Buffer>,
}
impl Operator {
    fn new(d: &wgpu::Device, n: Shape, particles: &[Particle], capacity: usize) -> Self {
        let p = buffer(
            d,
            &particles
                .iter()
                .map(|p| p.p)
                .chain(std::iter::repeat([0.; 4]))
                .take(capacity)
                .collect::<Vec<_>>(),
        );
        let v = buffer(
            d,
            &particles
                .iter()
                .map(|p| p.v)
                .chain(std::iter::repeat([0.; 4]))
                .take(capacity)
                .collect::<Vec<_>>(),
        );
        let rows = std::array::from_fn(|a| {
            buffer(
                d,
                &particles
                    .iter()
                    .map(|p| p.c[a])
                    .chain(std::iter::repeat([0.; 4]))
                    .take(capacity)
                    .collect::<Vec<_>>(),
            )
        });
        let mut metadata = vec![0f32; capacity * 4];
        for (i, p) in particles.iter().enumerate() {
            metadata[i] = p.rho;
        }
        let meta = buffer(d, &metadata);
        let rho = buffer(
            d,
            &particles
                .iter()
                .map(|p| p.rho)
                .chain(std::iter::repeat(1.))
                .take(capacity)
                .collect::<Vec<_>>(),
        );
        let grid =
            std::array::from_fn(|a| buffer(d, &vec![0f32; shape(n, a).into_iter().product()]));
        let old =
            std::array::from_fn(|a| buffer(d, &vec![0f32; shape(n, a).into_iter().product()]));
        let vort = buffer(d, &vec![0f32; n.into_iter().product()]);
        let phase = PhaseTransfer3D::new_periodic_z(
            d,
            n.map(|v| v as u32),
            capacity as u32,
            [&p, &v, &rows[0], &rows[1], &rows[2]],
            &meta,
            [&grid[0], &grid[1], &grid[2]],
        )
        .unwrap();
        Self {
            n,
            phase,
            p,
            v,
            rows,
            rho,
            meta,
            grid,
            old,
            vort,
        }
    }
    fn transfer(
        &self,
        d: &wgpu::Device,
        q: &wgpu::Queue,
        count: usize,
        gravity: f32,
        rate: f32,
        geometry: Option<&[u32]>,
    ) {
        let sed = SedimentParams3D {
            drag_coefficient: rate,
            gravity: gravity.abs(),
            ..Default::default()
        };
        self.phase.prepare(q, count as u32, H, DT, gravity, 8., sed);
        self.phase.set_volume_boundaries(q, 15);
        let mut e = d.create_command_encoder(&Default::default());
        self.phase.encode_begin_step(&mut e);
        if let Some(types) = geometry {
            let cells = buffer(d, types);
            let sdf = buffer(d, &cells_data(self.n, types));
            let bounds = self
                .phase
                .boundary_volume(d, &self.p, &self.meta, &cells, &sdf);
            self.phase
                .encode_cell_volume_with_boundaries(&mut e, count as u32, &bounds);
        } else {
            self.phase.encode_cell_volume(&mut e, count as u32);
        }
        self.phase.encode_transfer(&mut e, count as u32);
        q.submit(Some(e.finish()));
    }
    fn respond(&self, d: &wgpu::Device, q: &wgpu::Queue, gravity: f32) {
        if gravity != 0. {
            let mut v: Vec<f32> = read(d, q, &self.grid[1]);
            for x in &mut v {
                *x += gravity * DT;
            }
            q.write_buffer(&self.grid[1], 0, bytemuck::cast_slice(&v));
        }
        let mut e = d.create_command_encoder(&Default::default());
        self.phase.encode_response(&mut e);
        q.submit(Some(e.finish()));
    }
    fn status(&self, d: &wgpu::Device, q: &wgpu::Queue) -> u32 {
        read::<u32>(d, q, self.phase.status_buffer())[0]
    }
    fn gather(
        &self,
        d: &wgpu::Device,
        q: &wgpu::Queue,
        count: usize,
        capacity: usize,
        sediment: SedimentParams3D,
        flip: f32,
    ) -> (Vec<Vec3>, Vec<Mat3>) {
        let mut g = GpuG2p3D::new_periodic_with_metadata(
            d,
            self.n[0] as u32,
            self.n[1] as u32,
            self.n[2] as u32,
            capacity,
            self.p.clone(),
            self.v.clone(),
            self.rows[0].clone(),
            self.rows[1].clone(),
            self.rows[2].clone(),
            self.rho.clone(),
            &self.grid[0],
            &self.grid[1],
            &self.grid[2],
            &self.old[0],
            &self.old[1],
            &self.old[2],
            &self.vort,
            &self.phase.water[0],
            &self.phase.water[1],
            &self.phase.water[2],
            [0; 3],
            Some(self.meta.clone()),
            false,
        );
        g.set_conservative_phase_exchange(true);
        g.upload_params(q, count as u32, H, DT, flip, sediment);
        g.set_open_boundaries(q, 15);
        let mut e = d.create_command_encoder(&Default::default());
        g.encode(&mut e, count as u32);
        q.submit(Some(e.finish()));
        let mut v = vec![Vec3::ZERO; count];
        let mut c = vec![Mat3::ZERO; count];
        g.download(d, q, count as u32, &mut v, &mut c);
        (v, c)
    }
}
fn sediment(gravity: f32, rate: f32) -> SedimentParams3D {
    SedimentParams3D {
        drag_coefficient: rate,
        gravity: gravity.abs(),
        ..Default::default()
    }
}
fn cells_data(n: Shape, cells: &[u32]) -> Vec<f32> {
    assert_eq!(cells.len(), n.into_iter().product::<usize>());
    cells.iter().map(|c| if *c == 2 { -H } else { H }).collect()
}

#[test]
#[ignore = "requires actual Metal GPU; run explicitly with --ignored"]
fn periodic_implicit_response_and_adjoint_gather_match_physical_image_oracle() {
    let (d, q) = device();
    for depth in [2, 8, 12] {
        for (gravity, rate, impulse, affine, capacity) in [
            (0., 27., false, false, 12),
            (0., 27., false, false, 32),
            (-9.81, 27., true, false, 32),
            (0., 1e8, false, false, 32),
            (0., 27., false, true, 32),
        ] {
            let n = [10, 10, depth];
            let p = particles(depth, affine);
            let op = Operator::new(&d, n, &p, capacity);
            op.transfer(&d, &q, p.len(), gravity, rate, None);
            let raw = op.grid.each_ref().map(|b| read::<f32>(&d, &q, b));
            op.respond(&d, &q, gravity);
            assert_eq!(op.status(&d, &q), 0);
            let mixture = op.grid.each_ref().map(|b| read::<f32>(&d, &q, b));
            let water = op.phase.water.each_ref().map(|b| read::<f32>(&d, &q, b));
            let mobility = op.phase.mobility.each_ref().map(|b| read::<f32>(&d, &q, b));
            let mut responses: [Vec<Option<FaceResponse>>; 3] =
                std::array::from_fn(|a| vec![None; shape(n, a).into_iter().product()]);
            let mut gradients: [Vec<DVec3>; 3] =
                std::array::from_fn(|a| vec![DVec3::ZERO; shape(n, a).into_iter().product()]);
            let mut post = mixture.clone();
            for axis in 0..3 {
                let s = shape(n, axis);
                let mut physical = s;
                physical[2] = depth;
                for node in nodes(physical) {
                    let face = index(node, s);
                    let mut vw = 0.;
                    let mut pw = 0.;
                    let mut grains = Vec::new();
                    for particle in &p {
                        for image in -1..=1 {
                            let delta = offset(particle.p, node, axis, image, depth);
                            let volume = f64::from(particle.p[3]) * kernel(delta);
                            if volume == 0. {
                                continue;
                            }
                            let mut predictor = DVec3::new(
                                f64::from(particle.v[0]),
                                f64::from(particle.v[1]),
                                f64::from(particle.v[2]),
                            );
                            for a in 0..3 {
                                predictor[a] += (0..3)
                                    .map(|k| f64::from(particle.c[a][k]) * delta[k])
                                    .sum::<f64>();
                            }
                            if particle.rho == 1. {
                                vw += volume;
                                pw += volume * predictor[axis];
                            } else {
                                predictor.y += f64::from(gravity * DT);
                                grains.push(GrainSample {
                                    volume,
                                    density: f64::from(particle.rho),
                                    relaxation_rate: f64::from(rate),
                                    velocity: predictor,
                                });
                            }
                        }
                    }
                    if vw <= 1e-8 {
                        continue;
                    }
                    let mut water_predictor = DVec3::splat(pw / vw);
                    if axis == 1 {
                        water_predictor[axis] += f64::from(gravity * DT);
                    }
                    let response =
                        FaceResponse::new(vw, water_predictor, &grains, 8., f64::from(DT)).unwrap();
                    let compare = |actual: f32, expected: f64| {
                        assert!(
                            (f64::from(actual) - expected).abs() < 4e-5 * expected.abs().max(1.),
                            "depth{depth} axis{axis} face{face}: {actual} vs {expected}"
                        )
                    };
                    compare(raw[axis][face], pw / vw);
                    compare(mixture[axis][face], response.mixture_predictor[axis]);
                    compare(mobility[axis][face + 1], response.pressure_mobility);
                    let aux = PhaseTransfer3D::aux_offset(s.into_iter().product::<usize>() as u64)
                        as usize
                        + 4 * face;
                    compare(water[axis][aux], response.water_predictor[axis]);
                    compare(
                        water[axis][aux + 1],
                        response.water_pressure_mobility / response.pressure_mobility,
                    );
                    compare(
                        water[axis][aux + 3],
                        response.pressure_scale / response.pressure_mobility,
                    );
                    let volume = vw + grains.iter().map(|g| g.volume).sum::<f64>();
                    let mut gradient = DVec3::ZERO;
                    if impulse {
                        gradient[axis] = 0.005 * volume / 8.;
                    }
                    post[axis][face] = response.mixture(gradient)[axis] as f32;
                    responses[axis][face] = Some(response);
                    gradients[axis][face] = gradient;
                }
            }
            // Redundant seam response/coefficients are copies; accumulators represent
            // each physical face once. Gather must ignore a poisoned redundant velocity.
            let plane = n[0] * n[1];
            let first = depth * plane;
            let aux = PhaseTransfer3D::aux_offset(mixture[2].len() as u64) as usize;
            for i in 0..plane {
                assert_eq!(mixture[2][i].to_bits(), mixture[2][first + i].to_bits());
                assert_eq!(
                    mobility[2][i + 1].to_bits(),
                    mobility[2][first + i + 1].to_bits()
                );
                for c in 0..4 {
                    assert_eq!(
                        water[2][aux + 4 * i + c].to_bits(),
                        water[2][aux + 4 * (first + i) + c].to_bits()
                    );
                }
            }
            post[2][first..].fill(999.);
            for (b, v) in op.grid.iter().zip(&post) {
                q.write_buffer(b, 0, bytemuck::cast_slice(v));
            }
            let flip = 0.97;
            let (velocity, affine_out) =
                op.gather(&d, &q, p.len(), capacity, sediment(gravity, rate), flip);
            assert_eq!(op.status(&d, &q), 0);
            for (id, particle) in p.iter().enumerate() {
                for axis in 0..3 {
                    let s = shape(n, axis);
                    let mut physical = s;
                    physical[2] = depth;
                    let (mut pic, mut old, mut total) = (0., 0., 0.);
                    let (mut cnew, mut cold) = (DVec3::ZERO, DVec3::ZERO);
                    for node in nodes(physical) {
                        let face = index(node, s);
                        let Some(response) = responses[axis][face] else {
                            continue;
                        };
                        for image in -1..=1 {
                            let delta = offset(particle.p, node, axis, image, depth);
                            let w = kernel(delta);
                            if w == 0. {
                                continue;
                            }
                            let mut predictor = DVec3::new(
                                f64::from(particle.v[0]),
                                f64::from(particle.v[1]),
                                f64::from(particle.v[2]),
                            );
                            for a in 0..3 {
                                predictor[a] += (0..3)
                                    .map(|k| f64::from(particle.c[a][k]) * delta[k])
                                    .sum::<f64>();
                            }
                            predictor.y += f64::from(gravity * DT);
                            let value = if particle.rho == 1. {
                                response.water(gradients[axis][face])[axis]
                            } else {
                                response.grain(
                                    GrainSample {
                                        volume: 1.,
                                        density: f64::from(particle.rho),
                                        relaxation_rate: f64::from(rate),
                                        velocity: predictor,
                                    },
                                    gradients[axis][face],
                                )[axis]
                            };
                            let base = f64::from(raw[axis][face]);
                            pic += w * value;
                            old += w * base;
                            total += w;
                            let reference = if particle.rho == 1. {
                                f64::from(particle.v[axis])
                            } else {
                                0.
                            };
                            cnew += delta * (w * (value - reference) * 4. / f64::from(H * H));
                            cold += delta * (w * (base - reference) * 4. / f64::from(H * H));
                        }
                    }
                    pic /= total;
                    old /= total;
                    cnew /= total;
                    cold /= total;
                    let expected = if particle.rho == 1. {
                        f64::from(flip) * (f64::from(particle.v[axis]) + pic - old)
                            + (1. - f64::from(flip)) * pic
                    } else {
                        pic
                    };
                    assert!(
                        (f64::from(velocity[id][axis]) - expected).abs() < 5e-5,
                        "gather velocity depth{depth} particle{id} axis{axis}: {:?} vs {expected}",
                        velocity[id]
                    );
                    let expected_c = if particle.rho == 1. {
                        DVec3::new(
                            f64::from(particle.c[axis][0]),
                            f64::from(particle.c[axis][1]),
                            f64::from(particle.c[axis][2]),
                        ) * f64::from(flip)
                            + (cnew - cold) * f64::from(flip)
                            + cnew * (1. - f64::from(flip))
                    } else {
                        cnew
                    };
                    for k in 0..3 {
                        assert!(
                            (f64::from(affine_out[id].to_cols_array()[axis * 3 + k])
                                - expected_c[k])
                                .abs()
                                < 0.003,
                            "gather affine depth{depth} particle{id} axis{axis} k{k}"
                        );
                    }
                }
            }
            if gravity == 0. && !impulse {
                let before = p
                    .iter()
                    .map(|p| {
                        DVec3::new(f64::from(p.v[0]), f64::from(p.v[1]), f64::from(p.v[2]))
                            * f64::from(p.p[3] * p.rho)
                    })
                    .sum::<DVec3>();
                let after = p
                    .iter()
                    .zip(&velocity)
                    .map(|(p, v)| v.as_dvec3() * f64::from(p.p[3] * p.rho))
                    .sum::<DVec3>();
                assert!(
                    (after - before).length() < 2e-5,
                    "periodic momentum {before:?} -> {after:?}"
                );
                if !affine {
                    let e0 = p
                        .iter()
                        .map(|p| {
                            0.5 * f64::from(p.p[3] * p.rho)
                                * (f64::from(p.v[0]).powi(2)
                                    + f64::from(p.v[1]).powi(2)
                                    + f64::from(p.v[2]).powi(2))
                        })
                        .sum::<f64>();
                    let e1 = p
                        .iter()
                        .zip(&velocity)
                        .map(|(p, v)| {
                            0.5 * f64::from(p.p[3] * p.rho) * v.as_dvec3().length_squared()
                        })
                        .sum::<f64>();
                    assert!(e1 <= e0 + 1e-7, "drag created kinetic energy {e0} -> {e1}");
                }
            }
        }
    }
}

#[test]
#[ignore = "requires actual Metal GPU; run explicitly with --ignored"]
fn voxel_reflection_and_volume_are_invariant_when_a_real_block_crosses_the_seam() {
    let (d, q) = device();
    for depth in [8, 12] {
        let n = [10, 10, depth];
        let shift = 3;
        let cells: Vec<u32> = nodes(n)
            .map(|[x, y, z]| {
                if (2..7).contains(&x) && (2..7).contains(&y) && z >= depth - 2 {
                    2
                } else {
                    0
                }
            })
            .collect();
        let mut moved = vec![0; cells.len()];
        for p in nodes(n) {
            let mut copy = p;
            copy[2] = (p[2] + shift) % depth;
            moved[index(copy, n)] = cells[index(p, n)];
        }
        let mut p = particles(depth, false);
        p.retain(|p| p.p[2] < H);
        for p in &mut p {
            p.p[1] = 4.25 * H;
        }
        let extra: Vec<_> = p
            .iter()
            .cloned()
            .map(|mut p| {
                p.p[0] = 7.25 * H;
                p
            })
            .collect();
        p.extend(extra);
        let moved_p: Vec<_> = p
            .iter()
            .cloned()
            .map(|mut p| {
                p.p[2] = (p.p[2] + shift as f32 * H).rem_euclid(depth as f32 * H);
                p
            })
            .collect();
        let a = Operator::new(&d, n, &p, p.len() + 4);
        let b = Operator::new(&d, n, &moved_p, p.len() + 4);
        a.transfer(&d, &q, p.len(), 0., 27., Some(&cells));
        b.transfer(&d, &q, p.len(), 0., 27., Some(&moved));
        a.respond(&d, &q, 0.);
        b.respond(&d, &q, 0.);
        assert_eq!(a.status(&d, &q), 0);
        assert_eq!(b.status(&d, &q), 0);
        let av = read::<f32>(&d, &q, &a.phase.water[0]);
        let bv = read::<f32>(&d, &q, &b.phase.water[0]);
        let start = PhaseTransfer3D::cell_volume_offset(n.map(|v| v as u32)) as usize;
        let total = av[start..start + cells.len()]
            .iter()
            .map(|v| f64::from(*v))
            .sum::<f64>();
        let expected = p.iter().map(|p| f64::from(p.p[3])).sum::<f64>();
        assert!(
            (total - expected).abs() < 2e-5,
            "represented volume {total} vs {expected}"
        );
        for node in nodes(n) {
            let mut copy = node;
            copy[2] = (node[2] + shift) % depth;
            assert_eq!(av[start + index(node, n)], bv[start + index(copy, n)]);
            if cells[index(node, n)] == 2 {
                assert_eq!(av[start + index(node, n)], 0.);
            }
        }
        for axis in 0..3 {
            let s = shape(n, axis);
            let fa = read::<f32>(&d, &q, &a.grid[axis]);
            let fb = read::<f32>(&d, &q, &b.grid[axis]);
            let mut physical = s;
            physical[2] = depth;
            for node in nodes(physical) {
                let mut copy = node;
                copy[2] = (node[2] + shift) % depth;
                assert_eq!(
                    fa[index(node, s)],
                    fb[index(copy, s)],
                    "real wall reflection axis{axis} node{node:?}"
                );
            }
        }
        let (va, ca) = a.gather(&d, &q, p.len(), p.len() + 4, sediment(0., 27.), 0.97);
        let (vb, cb) = b.gather(&d, &q, p.len(), p.len() + 4, sediment(0., 27.), 0.97);
        assert_eq!(va, vb);
        assert_eq!(ca, cb);
    }
}

#[test]
#[ignore = "requires actual Metal GPU; run explicitly with --ignored"]
fn closed_periodic_mixture_has_constant_volume_and_complete_seam_pressure_support() {
    let (d, q) = device();
    for depth in [2, 8, 12] {
        let n = [6, 6, depth];
        let mut p = Vec::new();
        for node in nodes(n) {
            let mut sample = 0;
            for x in [0.25, 0.75] {
                for y in [0.25, 0.75] {
                    for z in [0.25, 0.75] {
                        let rho = if sample % 2 == 0 {
                            1.
                        } else if sample % 3 == 0 {
                            19.3
                        } else {
                            2.65
                        };
                        p.push(Particle {
                            p: [
                                (node[0] as f32 + x) * H,
                                (node[1] as f32 + y) * H,
                                (node[2] as f32 + z) * H,
                                1.,
                            ],
                            v: [0.; 4],
                            rho,
                            c: [[0.; 4]; 3],
                        });
                        sample += 1;
                    }
                }
            }
        }
        let op = Operator::new(&d, n, &p, p.len());
        let cells = buffer(&d, &vec![0u32; n.into_iter().product()]);
        let sdf = buffer(&d, &vec![1f32; n.into_iter().product()]);
        let boundary = op.phase.boundary_volume(&d, &op.p, &op.meta, &cells, &sdf);
        op.phase.prepare(
            &q,
            p.len() as u32,
            H,
            DT,
            0.,
            8.,
            SedimentParams3D::default(),
        );
        op.phase.set_volume_boundaries(&q, 0);
        let mut e = d.create_command_encoder(&Default::default());
        op.phase.encode_begin_step(&mut e);
        op.phase
            .encode_cell_volume_with_boundaries(&mut e, p.len() as u32, &boundary);
        op.phase.encode_transfer(&mut e, p.len() as u32);
        q.submit(Some(e.finish()));
        let v = read::<f32>(&d, &q, &op.phase.water[0]);
        let start = PhaseTransfer3D::cell_volume_offset(n.map(|v| v as u32)) as usize;
        for value in &v[start..start + n.into_iter().product::<usize>()] {
            assert!(
                (*value - 8.).abs() < 1e-5,
                "uniform represented volume {value}"
            );
        }
        let params = buffer(&d, &[6u32, 6, depth as u32, 0]);
        let (pipeline, group) = op.phase.pressure_support(&d, &params, &cells);
        let mut e = d.create_command_encoder(&Default::default());
        {
            let mut pass = e.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, (depth as u32).div_ceil(4));
        }
        q.submit(Some(e.finish()));
        assert_eq!(
            op.status(&d, &q),
            0,
            "occupied seam must see canonical carried W support"
        );
        assert!(read::<u32>(&d, &q, &cells).iter().all(|v| *v == 1));
    }
}

#[test]
#[ignore = "requires actual Metal GPU; run explicitly with --ignored"]
fn complete_periodic_conservative_runtime_retains_mixture_through_growth() {
    let (d, q) = device();
    let n = [8, 8, 8];
    let mut p = Vec::new();
    let mut rho = Vec::new();
    for node in nodes(n) {
        if !(2..6).contains(&node[0]) || !(2..6).contains(&node[1]) {
            continue;
        }
        for z in [0.25, 0.75] {
            for y in [0.25, 0.75] {
                for x in [0.25, 0.75] {
                    p.push(
                        Vec3::new(node[0] as f32 + x, node[1] as f32 + y, node[2] as f32 + z) * H,
                    );
                    rho.push(if x == 0.25 {
                        1.
                    } else if y == 0.25 {
                        2.65
                    } else {
                        19.3
                    });
                }
            }
        }
    }
    let count = p.len();
    let mut v = vec![Vec3::ZERO; count];
    let mut c = vec![Mat3::ZERO; count];
    let cells = vec![0u32; n.into_iter().product()];
    let sdf = vec![1f32; cells.len()];
    let mut s = GpuFlip3D::new_periodic_conservative(&d, 8, 8, 8, H, count).unwrap();
    assert!(s.is_periodic_z() && s.conservative_phase_exchange());
    s.open_boundaries = 0;
    s.vorticity_epsilon = 0.;
    s.set_particle_volumes(vec![1.; count]);
    for tick in 0..12 {
        if tick == 4 {
            s.reallocate_for_upload(&d, &q, count + 128).unwrap();
            assert!(s.is_periodic_z() && s.conservative_phase_exchange());
        }
        s.step(
            &d,
            &q,
            &mut p,
            &mut v,
            &mut c,
            &rho,
            &cells,
            Some(&sdf),
            None,
            1. / 480.,
            -9.81,
            0.,
            80,
        );
        assert_eq!(s.take_phase_error(), None, "phase refusal at tick{tick}");
        assert!(p.iter().all(|p| p.is_finite() && p.z >= 0. && p.z < 8. * H));
        assert!(v.iter().all(|v| v.is_finite()) && c.iter().all(|c| c.is_finite()));
    }
    assert_eq!(p.len(), count);
    assert!(
        v.iter()
            .zip(&rho)
            .any(|(v, rho)| *rho > 1. && v.y < -0.0001),
        "mineral gravity must remain active"
    );
}

#[test]
#[ignore = "requires actual Metal GPU; run explicitly with --ignored"]
fn coupled_resident_frames_and_growth_match_complete_synchronous_states() {
    let (d, q) = device();
    let mut positions = Vec::new();
    let mut rho = Vec::new();
    for node in nodes([4, 4, 8]) {
        for x in [0.25, 0.75] {
            for y in [0.25, 0.75] {
                for z in [0.25, 0.75] {
                    positions.push(
                        Vec3::new(node[0] as f32 + x, node[1] as f32 + y, node[2] as f32 + z) * H,
                    );
                    rho.push(if x == 0.25 {
                        1.
                    } else if y == 0.25 {
                        2.65
                    } else {
                        19.3
                    });
                }
            }
        }
    }
    let count = positions.len();
    let mut a = positions.clone();
    let mut b = positions;
    let mut va = vec![Vec3::ZERO; count];
    let mut vb = va.clone();
    let mut ca = vec![Mat3::ZERO; count];
    let mut cb = ca.clone();
    let mut sa = GpuFlip3D::new_periodic_conservative(&d, 4, 4, 8, H, count).unwrap();
    let mut sb = GpuFlip3D::new_periodic_conservative(&d, 4, 4, 8, H, count).unwrap();
    sa.vorticity_epsilon = 0.;
    sb.vorticity_epsilon = 0.;
    let cells = vec![0u32; 4 * 4 * 8];
    let sdf = vec![1f32; cells.len()];
    for frame in 0..3 {
        if frame == 1 {
            sa.reallocate_for_upload(&d, &q, count + 64).unwrap();
            sb.reallocate_for_upload(&d, &q, count + 64).unwrap();
        }
        for _ in 0..4 {
            sa.step(
                &d,
                &q,
                &mut a,
                &mut va,
                &mut ca,
                &rho,
                &cells,
                Some(&sdf),
                None,
                1. / 480.,
                -9.81,
                0.,
                80,
            );
        }
        sb.step_no_readback(
            &d,
            &q,
            &mut b,
            &mut vb,
            &mut cb,
            &rho,
            &cells,
            Some(&sdf),
            None,
            1. / 480.,
            -9.81,
            0.,
            80,
        );
        for _ in 1..4 {
            sb.step_in_place(
                &d,
                &q,
                count as u32,
                &cells,
                Some(&sdf),
                None,
                1. / 480.,
                -9.81,
                0.,
                80,
            );
        }
        assert!(sb.request_readback(&d, &q, count));
        d.poll(wgpu::Maintain::Wait);
        assert_eq!(sb.try_readback(&d, &mut b, &mut vb, &mut cb), Some(count));
        assert_eq!(sa.take_phase_error(), None);
        assert_eq!(sb.take_phase_error(), None);
        assert_eq!(a, b, "position frame{frame}");
        assert_eq!(va, vb, "velocity frame{frame}");
        assert_eq!(ca, cb, "affine frame{frame}");
    }
}

#[test]
#[ignore = "requires actual Metal GPU; run explicitly with --ignored"]
fn volume_feedback_sees_actual_water_gaps_across_the_periodic_seam() {
    let (d, q) = device();
    for depth in [8usize, 12] {
        for shift in [0usize, 3] {
            let n = [6, 6, depth];
            let len = n.into_iter().product::<usize>();
            let gap = (depth - 1 + shift) % depth;
            let markers: Vec<u32> = nodes(n).map(|p| u32::from(p[2] != gap)).collect();
            let markers = buffer(&d, &markers);
            let cells = buffer(&d, &vec![1u32; len]);
            let volume = buffer(&d, &vec![7f32; len]);
            let divergence = buffer(&d, &vec![0f32; len]);
            let status = buffer(&d, &[0u32; 4]);
            let params = buffer(
                &d,
                &[
                    6u32,
                    6,
                    depth as u32,
                    8f32.to_bits(),
                    DT.to_bits(),
                    1,
                    0,
                    0.25f32.to_bits(),
                ],
            );
            let source = d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../game/src/gpu/shaders/density_error_3d.wgsl").into(),
                ),
            });
            let buffers = [
                (0, &params, false),
                (2, &cells, false),
                (4, &volume, false),
                (5, &divergence, true),
                (6, &status, true),
                (7, &markers, false),
            ];
            let entries: Vec<_> = buffers
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
            let constants = std::collections::HashMap::from([
                ("PERIODIC_Z".into(), 1.),
                ("PHASE_RUNTIME_ALLOWED".into(), 1.),
            ]);
            let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pl),
                module: &source,
                entry_point: Some("add_density_rhs"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            });
            let entries: Vec<_> = buffers
                .iter()
                .map(|(binding, b, _)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: b.as_entire_binding(),
                })
                .collect();
            let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            });
            let mut encoder = d.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, (depth as u32).div_ceil(4));
            }
            q.submit(Some(encoder.finish()));
            assert_eq!(read::<u32>(&d, &q, &status)[0], 0);
            let actual = read::<f32>(&d, &q, &divergence);
            for p in nodes(n) {
                let surface =
                    p[2] == gap || (p[2] + 1) % depth == gap || (p[2] + depth - 1) % depth == gap;
                let expected = if surface { 0. } else { 0.25 / (8. * DT) };
                assert!(
                    (actual[index(p, n)] - expected).abs() < 1e-5,
                    "source depth{depth} shift{shift} cell{p:?}"
                );
            }
        }
    }
}

//! Actual-runtime gravity equilibrium diagnostic. This is independent of the
//! zero-gravity rest controls and must not be reported as passing on refusal.
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use serde_json::json;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let output = args.get(1).expect("output JSON");
    let block = args.iter().any(|a| a == "--block");
    let mask_only = args.iter().any(|a| a == "--mask-only");
    let voxel = block || mask_only || args.iter().any(|a| a == "--voxel-walls");
    let use_sdf = voxel && !mask_only;
    let geometry = if block {
        "voxel-walls-and-block"
    } else if voxel {
        "voxel-walls"
    } else {
        "exterior-walls"
    };
    let duration: f32 = args
        .iter()
        .find_map(|a| a.strip_prefix("--duration="))
        .map(|v| v.parse().unwrap())
        .unwrap_or(0.1);
    assert!(duration.is_finite() && (0.01..=2.).contains(&duration));
    assert!(!std::path::Path::new(&output).exists(), "preserve receipt");
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("real GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    let mut cases = Vec::new();
    for alpha in [0., 1.] {
        for (n, per_axis, hz, pressure_iterations) in [
            (12usize, 2usize, 120usize, 80),
            (12, 4, 120, 80),
            (24, 2, 240, 80),
            (12, 2, 240, 80),
            (12, 2, 120, 440),
            (24, 2, 240, 440),
        ] {
            let h = 0.12 / n as f32;
            let dt = 1. / hz as f32;
            let wall = if voxel { n / 12 } else { 0 };
            let mut cells = vec![0u32; n * n * n];
            let mut distances = vec![1f32; n * n * n];
            for z in 0..n {
                for y in 0..n {
                    for x in 0..n {
                        let i = (z * n + y) * n + x;
                        if voxel
                            && (x < wall || x >= n - wall || z < wall || z >= n - wall || y < wall)
                        {
                            cells[i] = 2;
                        }
                        let point = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5) * h;
                        let lower = wall as f32 * h;
                        let upper = 0.12 - lower;
                        distances[i] = (point.x - lower)
                            .min(upper - point.x)
                            .min(point.y - lower)
                            .min(point.z - lower)
                            .min(upper - point.z);
                        if block {
                            let lo = Vec3::new(0.04, 0.01, 0.04);
                            let hi = Vec3::new(0.08, 0.04, 0.08);
                            let d = (point - (lo + hi) * 0.5).abs() - (hi - lo) * 0.5;
                            let phi = d.max(Vec3::ZERO).length() + d.max_element().min(0.);
                            distances[i] = distances[i].min(phi);
                            if point.cmpge(lo).all() && point.cmplt(hi).all() {
                                cells[i] = 2;
                            }
                        }
                    }
                }
            }
            let mut p = Vec::new();
            let mut excluded_seed_points = 0;
            for z in 0..n * per_axis {
                for y in 0..(n * 2 / 3 + wall) * per_axis {
                    for x in 0..n * per_axis {
                        if cells[((z / per_axis) * n + y / per_axis) * n + x / per_axis] == 2 {
                            continue;
                        }
                        let candidate = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5)
                            * (h / per_axis as f32);
                        // Trilinear geometry rounds voxel corners. Seed the
                        // actual admissible field, not points already in it.
                        if use_sdf && sample_distance(&distances, n, h, candidate) < h * 0.001 {
                            excluded_seed_points += 1;
                            continue;
                        }
                        p.push(candidate);
                    }
                }
            }
            let initial = p.clone();
            let mut sim =
                GpuFlip3D::new_conservative(&device, n as u32, n as u32, n as u32, h, p.len())
                    .unwrap();
            sim.open_boundaries = 8;
            sim.flip_ratio = 0.97;
            sim.vorticity_epsilon = 0.;
            sim.density_projection_strength = alpha;
            sim.set_particle_volumes(vec![8. / (per_axis as f32).powi(3); p.len()]);
            let mut v = vec![Vec3::ZERO; p.len()];
            let mut c = vec![Mat3::ZERO; p.len()];
            let rho = vec![1.; p.len()];
            let sdf = use_sdf.then_some(distances.as_slice());
            let steps = (duration * hz as f32).round() as usize;
            let mut error = None;
            for step in 0..steps {
                sim.step(
                    &device,
                    &queue,
                    &mut p,
                    &mut v,
                    &mut c,
                    &rho,
                    &cells,
                    sdf,
                    None,
                    dt,
                    -9.81,
                    0.,
                    pressure_iterations,
                );
                if step == 0
                    && alpha == 0.
                    && n == 12
                    && per_axis == 2
                    && hz == 120
                    && pressure_iterations == 80
                {
                    if let Ok(path) = std::env::var("SLUICE_HYDRO_TRACE") {
                        let grids = [
                            sim.grid_u_buffer(),
                            sim.grid_v_buffer(),
                            sim.grid_w_buffer(),
                        ]
                        .map(|buffer| read_grid(&device, &queue, buffer));
                        let diagnostic = sim.read_physics_diagnostics(&device, &queue);
                        let columns: Vec<_> = [0usize, 1, 5]
                            .iter()
                            .map(|&x| {
                                let values: Vec<_> = (0..=n)
                                    .map(|y| grids[1][(5 * (n + 1) + y) * n + x])
                                    .collect();
                                json!({"x_index":x,"v_faces":values})
                            })
                            .collect();
                        let trace = json!({"geometry":geometry,"first_step":true,"fluid_cells":diagnostic.fluid_cell_count,
                            "avg_pressure":diagnostic.avg_pressure,
                            "post_projection_max_divergence":sim.compute_post_correction_divergence(&device,&queue),
                            "max_grid_components":grids.iter().map(|g|g.iter().map(|v|v.abs()).fold(0f32,f32::max)).collect::<Vec<_>>(),
                            "max_particle_components":(0..3).map(|axis|v.iter().map(|v|v[axis].abs()).fold(0f32,f32::max)).collect::<Vec<_>>(),
                            "v_columns":columns});
                        std::fs::write(path, serde_json::to_vec_pretty(&trace).unwrap()).unwrap();
                        eprintln!("hydrostatic first-step trace: {trace}");
                    }
                }
                error = sim.take_phase_error();
                if error.is_some() {
                    break;
                }
            }
            let max_speed = v.iter().map(|v| v.length()).fold(0f32, f32::max);
            let max_drift = p
                .iter()
                .zip(&initial)
                .map(|(a, b)| (*a - *b).length())
                .fold(0f32, f32::max);
            let mass = 1000. * (h as f64 / per_axis as f64).powi(3);
            let d = (h as f64).powi(2) / 4.;
            let energy = v
                .iter()
                .zip(&c)
                .map(|(v, c)| {
                    0.5 * mass
                        * (v.as_dvec3().length_squared()
                            + d * [c.x_axis, c.y_axis, c.z_axis]
                                .iter()
                                .map(|r| r.as_dvec3().length_squared())
                                .sum::<f64>())
                })
                .sum::<f64>();
            let passed = error.is_none() && max_speed < 1e-3 && max_drift < 0.02 * h;
            let row = json!({"geometry":geometry,"uses_sdf":use_sdf,"excluded_seed_points":excluded_seed_points,"grid":n,"per_axis":per_axis,"hz":hz,"alpha":alpha,
                "pressure_sweeps":pressure_iterations+sim.volume_iterations,"particles":p.len(),
                "duration_s":steps as f64/hz as f64,"max_speed_mps":max_speed,"max_drift_m":max_drift,
                "kinetic_energy_j":energy,"refusal":error,"passed":passed});
            println!("{row}");
            cases.push(row);
        }
    }
    let passed = cases.iter().all(|c| c["passed"] == true);
    std::fs::write(output, serde_json::to_vec_pretty(&json!({
        "protocol":"initially stationary 0.12m square domain, 0.08m water depth, gravity -9.81, open top, closed other faces; no flow or vorticity force",
        "geometry":geometry,"uses_sdf":use_sdf,"requested_duration_s":duration,
        "gate":"max speed < 1mm/s and max drift < 0.02 cell over the stated physical duration",
        "passed":passed,"cases":cases
    })).unwrap()).unwrap();
    if !passed {
        std::process::exit(1);
    }
}

fn read_grid(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hydrostatic diagnostic"),
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    values
}

fn sample_distance(field: &[f32], n: usize, h: f32, point: Vec3) -> f32 {
    let coordinate = point / h - Vec3::splat(0.5);
    let base = coordinate.floor();
    let fraction = coordinate - base;
    let mut result = 0.;
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let q = base + Vec3::new(x as f32, y as f32, z as f32);
                let value = field[(q.z as usize * n + q.y as usize) * n + q.x as usize];
                let weight = Vec3::new(
                    if x == 0 { 1. - fraction.x } else { fraction.x },
                    if y == 0 { 1. - fraction.y } else { fraction.y },
                    if z == 0 { 1. - fraction.z } else { fraction.z },
                );
                result += value * weight.x * weight.y * weight.z;
            }
        }
    }
    result
}

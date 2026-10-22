//! Actual-runtime gravity equilibrium diagnostic. This is independent of the
//! zero-gravity rest controls and must not be reported as passing on refusal.
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use serde_json::json;

fn main() {
    let output = std::env::args().nth(1).expect("output JSON");
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
        ] {
            let h = 0.12 / n as f32;
            let dt = 1. / hz as f32;
            let mut p = Vec::new();
            for z in 0..n * per_axis {
                for y in 0..n * per_axis * 2 / 3 {
                    for x in 0..n * per_axis {
                        p.push(
                            Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5)
                                * (h / per_axis as f32),
                        );
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
            let cells = vec![0; n * n * n];
            let mut error = None;
            for _ in 0..hz / 10 {
                sim.step(
                    &device,
                    &queue,
                    &mut p,
                    &mut v,
                    &mut c,
                    &rho,
                    &cells,
                    None,
                    None,
                    dt,
                    -9.81,
                    0.,
                    pressure_iterations,
                );
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
            let row = json!({"grid":n,"per_axis":per_axis,"hz":hz,"alpha":alpha,
                "pressure_sweeps":pressure_iterations+sim.volume_iterations,"particles":p.len(),
                "duration_s":0.1,"max_speed_mps":max_speed,"max_drift_m":max_drift,
                "kinetic_energy_j":energy,"refusal":error,"passed":passed});
            println!("{row}");
            cases.push(row);
        }
    }
    let passed = cases.iter().all(|c| c["passed"] == true);
    std::fs::write(output, serde_json::to_vec_pretty(&json!({
        "protocol":"initially stationary 0.12m square column, 0.08m water, gravity -9.81, open top, closed other faces; no SDF, flow, or vorticity force",
        "gate":"max speed < 1mm/s and max drift < 0.02 cell over 0.1 physical seconds",
        "passed":passed,"cases":cases
    })).unwrap()).unwrap();
    if !passed {
        std::process::exit(1);
    }
}

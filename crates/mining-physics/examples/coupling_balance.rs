//! Closed momentum budget for zero-gravity interphase exchange. This is a
//! diagnostic, not a recovery calibration or a prescribed target curve.
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use serde_json::json;

fn main() {
    let output = std::env::args().nth(1).expect("output JSON");
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&Default::default())).expect("real GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    let mut results = Vec::new();
    for (name, solid, dt) in [
        ("water_control", false, 1. / 120.),
        ("mixed_120hz", true, 1. / 120.),
        ("mixed_240hz", true, 1. / 240.),
    ] {
        let mut sim = GpuFlip3D::new(&device, 12, 12, 12, 0.01, 1024);
        sim.open_boundaries = 63;
        sim.density_projection_enabled = false;
        sim.vorticity_epsilon = 0.;
        sim.sediment_friction_strength = 0.;
        sim.sediment_vorticity_lift = 0.;
        sim.sediment_rest_particles = 0.;
        sim.sediment_porosity_drag = 0.;
        let mut p = Vec::new();
        for z in 4..8 {
            for y in 4..8 {
                for x in 4..8 {
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                p.push(
                                    Vec3::new(
                                        x as f32 + 0.25 + dx as f32 * 0.5,
                                        y as f32 + 0.25 + dy as f32 * 0.5,
                                        z as f32 + 0.25 + dz as f32 * 0.5,
                                    ) * 0.01,
                                );
                            }
                        }
                    }
                }
            }
        }
        let nw = p.len();
        p.push(Vec3::splat(0.06));
        let mut v = vec![Vec3::X * 0.2; p.len()];
        v[nw] = -Vec3::X * 0.2;
        let mut c = vec![Mat3::ZERO; p.len()];
        let mut densities = vec![1.; p.len()];
        if solid {
            densities[nw] = 2.65;
        }
        let mut volumes = vec![1.; p.len()];
        volumes[nw] = 16.;
        sim.set_particle_volumes(volumes.clone());
        sim.upload_grain_diameters(&queue, nw, &[if solid { 0.0003 } else { 0. }]);
        let momentum = |v: &[Vec3]| {
            v.iter()
                .zip(&densities)
                .zip(&volumes)
                .map(|((v, rho), volume)| *v * (*rho * *volume))
                .sum::<Vec3>()
        };
        let before = momentum(&v);
        let old = v.clone();
        sim.step(
            &device,
            &queue,
            &mut p,
            &mut v,
            &mut c,
            &densities,
            &vec![0; 12 * 12 * 12],
            None,
            None,
            dt,
            0.,
            0.,
            80,
        );
        let after = momentum(&v);
        let fluid_change: Vec3 = v[..nw].iter().zip(&old[..nw]).map(|(a, b)| *a - *b).sum();
        let solid_change = (v[nw] - old[nw]) * densities[nw] * volumes[nw];
        let scale = fluid_change.length() + solid_change.length();
        results.push(json!({"case":name,"dt":dt,"gravity":0,
            "momentum_before_relative_mass_mps":before.to_array(),
            "momentum_after_relative_mass_mps":after.to_array(),
            "momentum_residual":(after-before).to_array(),
            "relative_to_initial_momentum":(after-before).length()/before.length(),
            "exchange_imbalance_fraction":(fluid_change+solid_change).length()/scale.max(1e-12),
            "fluid_momentum_change":fluid_change.to_array(),"solid_momentum_change":solid_change.to_array(),
            "solid_final_velocity":v[nw].to_array()}));
    }
    std::fs::write(output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}

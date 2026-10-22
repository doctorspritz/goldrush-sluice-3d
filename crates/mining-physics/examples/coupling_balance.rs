//! Closed momentum budget for zero-gravity interphase exchange. This is a
//! diagnostic, not a recovery calibration or a prescribed target curve.
use game::gpu::flip_3d::GpuFlip3D;
use glam::{DVec3, Mat3, Vec3};
use serde_json::json;

fn main() {
    let output = std::env::args().nth(1).expect("output JSON");
    let conservative = std::env::args().any(|a| a == "--conservative");
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
    for (name, solid, dt, pressure, flip, diameter, coincident) in [
        ("water_transfer_flip", false, 1. / 120., 0, 1., 0., false),
        ("water_transfer_pic", false, 1. / 120., 0, 0., 0., false),
        ("water_transfer_blend", false, 1. / 120., 0, 0.97, 0., false),
        ("water_pressure_flip", false, 1. / 120., 80, 1., 0., false),
        (
            "water_pressure_blend",
            false,
            1. / 120.,
            80,
            0.97,
            0.,
            false,
        ),
        (
            "mixed_dry_transfer_flip",
            true,
            1. / 120.,
            0,
            1.,
            -0.0003,
            false,
        ),
        (
            "mixed_dry_transfer_blend",
            true,
            1. / 120.,
            0,
            0.97,
            -0.0003,
            false,
        ),
        ("mixed_drag_flip", true, 1. / 120., 0, 1., 0.0003, false),
        ("mixed_drag_blend", true, 1. / 120., 0, 0.97, 0.0003, false),
        ("mixed_full_120hz", true, 1. / 120., 80, 0.97, 0.0003, false),
        ("mixed_full_240hz", true, 1. / 240., 80, 0.97, 0.0003, false),
        ("coincident_drag_flip", true, 1. / 120., 0, 1., 0.0003, true),
        (
            "coincident_drag_blend",
            true,
            1. / 120.,
            0,
            0.97,
            0.0003,
            true,
        ),
    ] {
        let mut sim = if conservative {
            GpuFlip3D::new_conservative(&device, 12, 12, 12, 0.01, 1024).unwrap()
        } else {
            GpuFlip3D::new(&device, 12, 12, 12, 0.01, 1024)
        };
        sim.open_boundaries = 63;
        sim.flip_ratio = flip;
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
        if coincident {
            p.fill(Vec3::splat(0.06));
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
        sim.upload_grain_diameters(&queue, nw, &[diameter]);
        let momentum = |v: &[Vec3]| {
            v.iter()
                .zip(&densities)
                .zip(&volumes)
                .map(|((v, rho), volume)| v.as_dvec3() * (*rho as f64 * *volume as f64))
                .sum::<DVec3>()
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
            pressure,
        );
        assert!(
            sim.take_phase_error().is_none(),
            "phase response refused {name}"
        );
        let after = momentum(&v);
        let fluid_change: DVec3 = v
            .iter()
            .zip(&old)
            .zip(&densities)
            .zip(&volumes)
            .filter(|(((_, _), rho), _)| **rho <= 1.)
            .map(|(((v, old), rho), volume)| {
                (v.as_dvec3() - old.as_dvec3()) * (*rho as f64 * *volume as f64)
            })
            .sum();
        let solid_change: DVec3 = v
            .iter()
            .zip(&old)
            .zip(&densities)
            .zip(&volumes)
            .filter(|(((_, _), rho), _)| **rho > 1.)
            .map(|(((v, old), rho), volume)| {
                (v.as_dvec3() - old.as_dvec3()) * (*rho as f64 * *volume as f64)
            })
            .sum();
        if conservative {
            assert!(
                (after - before).length() / before.length() < 1e-6,
                "full runtime momentum budget failed: {name}"
            );
        }
        let scale = fluid_change.length() + solid_change.length();
        println!(
            "{name}: residual={:.9} relative={:.9} water_dP={:.9} grain_dP={:.9}",
            (after - before).x,
            (after - before).length() / before.length(),
            fluid_change.x,
            solid_change.x
        );
        results.push(json!({"case":name,"dt":dt,"gravity":0,"pressure_sweeps":pressure,"flip_ratio":flip,
            "diameter":diameter,"coincident":coincident,
            "momentum_before_relative_mass_mps":before.to_array(),
            "momentum_after_relative_mass_mps":after.to_array(),
            "momentum_residual":(after-before).to_array(),
            "relative_to_initial_momentum":(after-before).length()/before.length(),
            "combined_phase_imbalance_fraction":solid.then_some((fluid_change+solid_change).length()/scale.max(1e-12)),
            "fluid_momentum_change":fluid_change.to_array(),"solid_momentum_change":solid_change.to_array(),
            "solid_final_velocity":v[nw].to_array()}));
    }
    std::fs::write(output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

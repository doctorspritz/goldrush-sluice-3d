//! A storage change must preserve full transfers and pressure warm starts.
use game::gpu::flip_3d::{GpuFlip3D, GravelObstacle};
use glam::{Mat3, Vec3};

#[test]
#[ignore = "requires a real GPU; run explicitly with --ignored"]
fn reallocation_preserves_transfers_parameters_and_warm_pressure() {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default())).unwrap();
    // A bounded test device makes staging, rather than a vector binding, the
    // first allocation limit. All ordinary fixture buffers fit below 64 MiB.
    let mut limits = adapter.limits();
    limits.max_buffer_size = 64 * 1024 * 1024;
    limits.max_storage_buffer_binding_size = 64 * 1024 * 1024;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: limits,
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    for conservative in [false, true] {
        for projection in [false, true] {
            let run = |mode: u32| {
                let configure = |sim: &mut GpuFlip3D| {
                    sim.open_boundaries = 63;
                    sim.flip_ratio = 0.93;
                    sim.density_projection_enabled = projection;
                    sim.volume_iterations = 7;
                    sim.water_rest_particles = 6.5;
                    sim.sediment_rest_particles = 0.;
                    sim.sediment_porosity_drag = 0.;
                    sim.sediment_drag_coefficient = 3.75;
                    sim.gold_drag_multiplier = 0.37;
                    sim.vorticity_epsilon = 0.02;
                    sim.slip_factor = 0.7;
                };
                let create = |n| {
                    if conservative {
                        GpuFlip3D::new_conservative(&device, 16, 12, 8, 0.02, n).unwrap()
                    } else {
                        GpuFlip3D::new(&device, 16, 12, 8, 0.02, n)
                    }
                };
                let mut sim = create(if mode == 0 { 8192 } else { 4096 });
                configure(&mut sim);
                let too_large = (device.limits().max_buffer_size / 80 + 1) as usize;
                assert!(
                    sim.reallocate_for_upload(&device, &queue, too_large)
                        .is_err(),
                    "oversized combined readback reached allocation"
                );
                let obstacles = [GravelObstacle {
                    position: [0.15, 0.12, 0.08],
                    radius: 0.018,
                }];
                sim.upload_gravel_obstacles(&queue, &obstacles);
                let mut positions = Vec::new();
                for k in 2..6 {
                    for j in 2..10 {
                        for i in 2..14 {
                            for n in 0..8 {
                                positions.push(
                                    (Vec3::new(i as f32, j as f32, k as f32)
                                        + Vec3::new(
                                            0.25 + 0.5 * (n & 1) as f32,
                                            0.25 + 0.5 * ((n >> 1) & 1) as f32,
                                            0.25 + 0.5 * ((n >> 2) & 1) as f32,
                                        ))
                                        * 0.02,
                                );
                            }
                        }
                    }
                }
                let count = positions.len();
                let mut velocities: Vec<_> = (0..count)
                    .map(|i| Vec3::new(0.4, (i % 7) as f32 * 0.01 - 0.03, 0.02))
                    .collect();
                let mut affine = vec![Mat3::ZERO; count];
                let mut densities = vec![1.; count];
                densities[count - 2] = 2.65;
                densities[count - 1] = 19.3;
                let mut volumes = vec![1.; count];
                volumes[0] = 3.5;
                let cells = vec![0; 16 * 12 * 8];
                for step in 0..7 {
                    if step == 3 && mode == 1 {
                        sim.reallocate_for_upload(&device, &queue, 8192).unwrap();
                        assert_eq!(sim.particle_capacity(), 8192);
                        assert_eq!(sim.conservative_phase_exchange(), conservative);
                        assert_eq!(sim.flip_ratio, 0.93);
                        assert_eq!(sim.gold_drag_multiplier, 0.37);
                        assert_eq!(sim.density_projection_enabled, projection);
                        assert_eq!(sim.volume_iterations, 7);
                    } else if step == 3 && mode == 2 {
                        // A cold rebuild is an independent negative control for
                        // hidden pressure state, with identical tunable parameters.
                        sim = create(8192);
                        configure(&mut sim);
                        sim.upload_gravel_obstacles(&queue, &obstacles);
                    }
                    sim.set_particle_volumes(volumes.clone());
                    sim.upload_grain_diameters(&queue, count - 2, &[0.0003, 0.0003]);
                    assert!(sim.prepare_particle_readback(0, count));
                    sim.step_no_readback(
                        &device,
                        &queue,
                        &mut positions,
                        &mut velocities,
                        &mut affine,
                        &densities,
                        &cells,
                        None,
                        None,
                        1. / 120.,
                        -0.15,
                        0.03,
                        23,
                    );
                    if mode == 1 && step == 2 {
                        assert!(
                            sim.reallocate_for_upload(&device, &queue, 8192).is_err(),
                            "pending readback was discarded"
                        );
                    }
                    device.poll(wgpu::Maintain::Wait);
                    assert_eq!(
                        sim.try_readback(&device, &mut positions, &mut velocities, &mut affine),
                        Some(count)
                    );
                }
                assert!(positions.iter().chain(&velocities).all(|v| v.is_finite()));
                (positions, velocities, affine)
            };
            let reference = run(0);
            let grown = run(1);
            assert_eq!(
                reference, grown,
                "growth changed state; density projection={projection}"
            );
            if !projection {
                assert_ne!(
                    reference,
                    run(2),
                    "fixture failed to exercise a pressure warm start"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU; run explicitly with --ignored"]
fn a_large_live_shift_restores_and_continues_with_the_same_physical_state() {
    live_restore(false);
}
#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn conservative_large_live_shift_restores_and_continues_identically() {
    live_restore(true);
}
fn live_restore(conservative: bool) {
    use mining_sim::{session::Controls, sluice::Grain, GrainKind, Material, Placement, Session};
    let canonical = |s: &Session| {
        let mut value: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
        for name in ["spheres", "planes", "sdf"] {
            value["sluice"]["contacts"][name]
                .as_array_mut()
                .unwrap()
                .sort_by_cached_key(|row| {
                    let row = row.as_array().unwrap();
                    serde_json::to_string(&row[..row.len() - 1]).unwrap()
                });
        }
        serde_json::to_string(&value).unwrap()
    };
    let mut session = Session::new();
    session.sluice.placement = Some(Placement::suggested(&session.site));
    session.sluice.pump_on = true;
    let mass_g = std::f64::consts::PI / 6. * 0.0001_f64.powi(3) * 19_300. * 1000.;
    session.sluice.grains = (0..4103)
        .map(|i| Grain {
            volume_resolved: true,
            kind: GrainKind::Gold,
            position: Vec3::new(
                0.48 + (i % 128) as f32 * 0.0002,
                0.285,
                0.06 + (i / 128) as f32 * 0.0002,
            ),
            velocity: Vec3::ZERO,
            material: Material {
                gold_g: mass_g,
                ..Default::default()
            },
            resting: false,
            diameter_m: 0.0001,
            affine: Mat3::ZERO,
            rotation: glam::Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
        })
        .collect();
    // This is a defined initial retained population, not dynamic gold creation.
    session.initial_gold_g = session.accounted_gold();
    let mut solver = mining_physics::Hydraulics::new().unwrap();
    if conservative {
        solver = solver.with_conservative_phases();
    }
    for _ in 0..12 {
        session.step(Controls::default());
        solver.step(&mut session).unwrap();
    }
    assert!(session.sluice.grains.len() > 4096);
    let json = session.save_json().unwrap();
    let mut restored = Session::load_json(&json).unwrap();
    assert!(
        canonical(&session) == canonical(&restored),
        "large live save did not restore exactly"
    );
    for _ in 0..12 {
        session.step(Controls::default());
        solver.step(&mut session).unwrap();
    }
    let expected = canonical(&session);
    drop(solver);
    let mut solver = mining_physics::Hydraulics::new().unwrap();
    if conservative {
        solver = solver.with_conservative_phases();
    }
    for _ in 0..12 {
        restored.step(Controls::default());
        solver.step(&mut restored).unwrap();
    }
    assert!(
        expected == canonical(&restored),
        "live restoration changed the continued physical state"
    );
}

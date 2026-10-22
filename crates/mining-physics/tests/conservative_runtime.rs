//! Real FLIP runtime controls, not the stand-alone phase-response harness.
use game::gpu::flip_3d::GpuFlip3D;
use glam::{DVec3, Mat3, Vec3};
fn device() -> (wgpu::Device, wgpu::Queue) {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("real GPU");
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap()
}
fn configured(device: &wgpu::Device, capacity: usize) -> GpuFlip3D {
    configured_grid(device, capacity, 12, 0.01)
}
fn configured_grid(device: &wgpu::Device, capacity: usize, n: u32, h: f32) -> GpuFlip3D {
    let mut sim = GpuFlip3D::new_conservative(device, n, n, n, h, capacity).unwrap();
    sim.open_boundaries = 63;
    sim.flip_ratio = 0.97;
    sim.vorticity_epsilon = 0.;
    sim.sediment_friction_strength = 0.;
    sim.sediment_vorticity_lift = 0.;
    sim.sediment_porosity_drag = 0.;
    sim.sediment_rest_particles = 0.;
    sim
}
fn cloud() -> Vec<Vec3> {
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
    p
}
fn momentum(p: &[Vec3], v: &[Vec3], rho: &[f32], volume: &[f32]) -> (DVec3, DVec3) {
    p.iter().zip(v).zip(rho).zip(volume).fold(
        (DVec3::ZERO, DVec3::ZERO),
        |(linear, angular), (((p, v), rho), volume)| {
            let impulse = v.as_dvec3() * (*rho as f64 * *volume as f64);
            (linear + impulse, angular + p.as_dvec3().cross(impulse))
        },
    )
}
#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn actual_runtime_conserves_linear_momentum_with_represented_volume_repair() {
    let (device, queue) = device();
    for repair in [false, true] {
        for dt in [1. / 120., 1. / 240.] {
            let mut sim = configured(&device, 1024);
            sim.density_projection_enabled = repair;
            let mut p = cloud();
            let nw = p.len();
            for i in 0..24 {
                p.push(p[i * 17] + Vec3::new(0.0002, -0.0001, 0.0001));
            }
            let mut rho = vec![1.; p.len()];
            for (i, r) in rho[nw..].iter_mut().enumerate() {
                *r = if i % 2 == 0 { 2.65 } else { 19.3 };
            }
            let volume: Vec<f32> = (0..p.len())
                .map(|i| {
                    if i < nw {
                        0.7 + 0.3 * (i % 3) as f32
                    } else {
                        0.5 + 0.2 * (i % 4) as f32
                    }
                })
                .collect();
            sim.set_particle_volumes(volume.clone());
            sim.upload_grain_diameters(&queue, nw, &[0.0003; 24]);
            let mut v: Vec<Vec3> = (0..p.len())
                .map(|i| {
                    if i < nw {
                        Vec3::new(0.15, 0.01, -0.02)
                    } else {
                        Vec3::new(-0.06, -0.02, 0.04)
                    }
                })
                .collect();
            let before = momentum(&p, &v, &rho, &volume);
            let mut c = vec![Mat3::ZERO; p.len()];
            sim.step(
                &device,
                &queue,
                &mut p,
                &mut v,
                &mut c,
                &rho,
                &[0; 12 * 12 * 12],
                None,
                None,
                dt,
                0.,
                0.,
                80,
            );
            assert!(sim.take_phase_error().is_none());
            assert!(p.iter().chain(&v).all(|x| x.is_finite()));
            assert!(c.iter().all(|x| x.is_finite()));
            let after = momentum(&p, &v, &rho, &volume);
            let residual = (after.0 - before.0).length() / before.0.length();
            assert!(
                residual < 1e-6,
                "repair={repair},dt={dt}: linear relative={residual}"
            );
            // Generalized APIC angular momentum includes the intrinsic affine part.
            let d = 0.01f64.powi(2) / 4.;
            let intrinsic: DVec3 = c
                .iter()
                .zip(&rho)
                .zip(&volume)
                .map(|((c, r), volume)| {
                    let rows = [c.x_axis, c.y_axis, c.z_axis];
                    DVec3::new(
                        (rows[2].y - rows[1].z) as f64,
                        (rows[0].z - rows[2].x) as f64,
                        (rows[1].x - rows[0].y) as f64,
                    ) * (d * (*r as f64) * (*volume as f64))
                })
                .sum();
            let angular_error = (after.1 + intrinsic - before.1).length();
            assert!(
                angular_error < 1e-6,
                "repair={repair},dt={dt}: angular={angular_error}"
            );
            eprintln!(
                "runtime repair={repair} dt={dt}: linear={residual:.9}, angular_absolute={angular_error:.9}"
            );
        }
    }
}
#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn refusal_remains_sticky_across_substeps_and_withholds_cpu_state() {
    let (device, queue) = device();
    let mut sim = configured(&device, 32);
    sim.density_projection_enabled = false;
    let mut p = cloud()[..8].to_vec();
    let n = p.len();
    let mut v = vec![Vec3::X * 0.1; n];
    v[0] = Vec3::X * 1e12;
    let mut c = vec![Mat3::ZERO; n];
    sim.step_no_readback(
        &device,
        &queue,
        &mut p,
        &mut v,
        &mut c,
        &vec![1.; n],
        &[0; 12 * 12 * 12],
        None,
        None,
        1. / 240.,
        0.,
        0.,
        80,
    );
    // Repair the input, but the unobserved refusal must survive GPU-only steps.
    v.fill(Vec3::X * 0.1);
    sim.upload_particle_range(&queue, 0, &p, &v, &c);
    for _ in 0..2 {
        sim.step_in_place(
            &device,
            &queue,
            n as u32,
            &[0; 12 * 12 * 12],
            None,
            None,
            1. / 240.,
            0.,
            0.,
            80,
        );
    }
    assert!(sim.prepare_particle_readback(0, n));
    sim.step_in_place(
        &device,
        &queue,
        n as u32,
        &[0; 12 * 12 * 12],
        None,
        None,
        1. / 240.,
        0.,
        0.,
        80,
    );
    let mut out_p = vec![Vec3::splat(777.); n];
    let mut out_v = out_p.clone();
    let mut out_c = vec![Mat3::IDENTITY; n];
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    let status = loop {
        assert_eq!(
            sim.try_readback(&device, &mut out_p, &mut out_v, &mut out_c),
            None
        );
        if let Some(status) = sim.take_phase_error() {
            break status;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "joined refusal readback timed out"
        );
        std::thread::sleep(std::time::Duration::from_micros(100));
    };
    assert_ne!(status & 1, 0);
    assert!(out_p.iter().chain(&out_v).all(|p| *p == Vec3::splat(777.)));
    assert!(out_c.iter().all(|c| *c == Mat3::IDENTITY));
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn actual_worker_pauses_on_phase_refusal_and_keeps_the_shift_saveable() {
    use mining_physics::{
        worker::{Work, Worker},
        Hydraulics,
    };
    use mining_sim::{session::Controls, Action, Placement, Session};
    let mut session = Session::new();
    session.player.position = Vec3::new(14.6, 0.35, 15.0);
    session
        .on_action(Action::Place(Placement::suggested(&session.site)))
        .unwrap();
    session.on_action(Action::TogglePump).unwrap();
    let mut solver = Hydraulics::new().unwrap().with_conservative_phases();
    for _ in 0..4 {
        session.step(Controls::default());
        solver.step(&mut session).unwrap();
    }
    assert!(!session.sluice.water.is_empty());
    drop(solver);
    let steps = session.sluice.solver_steps;
    let water_error = session.water_balance_error_m3();
    let gold = session.accounted_gold();
    let dry = session.accounted_dry_mass_kg();
    // A finite saved affine field can exceed the transfer request bound while
    // particle velocities remain inside the existing save-admission range.
    session.sluice.water[0].affine = Mat3::IDENTITY * 1e12;
    Session::load_json(&session.save_json().unwrap()).expect("admissible fault fixture");
    let mut worker = Worker::with_session_conservative(session).unwrap();
    let wait = |worker: &mut Worker| {
        let end = std::time::Instant::now() + std::time::Duration::from_secs(15);
        loop {
            if let Some(snapshot) = worker.poll().unwrap() {
                break snapshot;
            }
            assert!(std::time::Instant::now() < end, "worker response deadline");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    };
    let _ = wait(&mut worker);
    worker
        .submit(Work {
            sequence: 1,
            advance: true,
            ticks: 2,
            ..Default::default()
        })
        .unwrap();
    let failed = wait(&mut worker);
    assert!(
        failed
            .failure
            .as_ref()
            .is_some_and(|message| message.contains("phase response refused")),
        "{:?}",
        failed.failure
    );
    assert!(failed.session.paused);
    assert_eq!(failed.session.sluice.solver_steps, steps);
    assert!(
        (failed.session.water_balance_error_m3() - water_error).abs() < 1e-8,
        "water custody changed on refusal"
    );
    assert!(
        (failed.session.sluice.water_used_m3 - failed.session.sluice.accounted_water_m3()).abs()
            < 1e-8
    );
    assert!((failed.session.accounted_gold() - gold).abs() < 1e-7);
    assert!((failed.session.accounted_dry_mass_kg() - dry).abs() < 1e-4);
    let json = failed.session.save_json().unwrap();
    Session::load_json(&json).expect("paused numerical refusal must remain reloadable");
    worker
        .submit(Work {
            sequence: 2,
            advance: true,
            ticks: 2,
            ..Default::default()
        })
        .unwrap();
    let paused = wait(&mut worker);
    assert!(paused.session.paused);
    assert_eq!(paused.session.sluice.solver_steps, steps);
    assert_eq!(
        paused.session.sim_time, failed.session.sim_time,
        "paused worker advanced after numerical refusal"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn full_runtime_freefall_keeps_all_densities_comoving() {
    let (device, queue) = device();
    for rate in [0f32, 50., 1e12] {
        let mut sim = configured(&device, 1024);
        sim.density_projection_enabled = false;
        sim.sediment_drag_coefficient = rate;
        let mut p = cloud();
        let nw = p.len();
        for i in 0..3 {
            p.push(p[120 + i]);
        }
        let mut rho = vec![1.; p.len()];
        rho[nw..].copy_from_slice(&[2.65, 7., 19.3]);
        let initial = Vec3::new(0.1, 0.03, -0.02);
        let mut v = vec![initial; p.len()];
        let mut c = vec![Mat3::ZERO; p.len()];
        let dt = 1. / 120.;
        sim.step(
            &device,
            &queue,
            &mut p,
            &mut v,
            &mut c,
            &rho,
            &[0; 12 * 12 * 12],
            None,
            None,
            dt,
            -9.81,
            0.,
            0,
        );
        assert!(sim.take_phase_error().is_none());
        let expected = initial + Vec3::Y * (-9.81 * dt);
        for (i, v) in v.iter().enumerate() {
            assert!(
                (*v - expected).length() < 2e-6,
                "rate={rate} particle={i},density={}: {v:?} vs {expected:?}",
                rho[i]
            );
        }
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn invalid_volume_constraint_refuses_before_publishing_particle_state() {
    let (device, queue) = device();
    let mut sim = configured(&device, 1024);
    sim.density_projection_strength = 2.;
    let mut p = cloud();
    let mut v = vec![Vec3::X * 0.1; p.len()];
    let mut c = vec![Mat3::ZERO; p.len()];
    let initial = (p.clone(), v.clone(), c.clone());
    let rho = vec![1.; p.len()];
    sim.step(
        &device,
        &queue,
        &mut p,
        &mut v,
        &mut c,
        &rho,
        &[0; 12 * 12 * 12],
        None,
        None,
        1. / 120.,
        0.,
        0.,
        80,
    );
    assert_eq!(sim.take_phase_error(), Some(16));
    assert_eq!(
        (p, v, c),
        initial,
        "refused volume constraint modified authoritative output"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn closed_box_at_rest_has_no_volume_constraint_motion() {
    let (device, queue) = device();
    let mut results = Vec::new();
    for geometry in 0..3 {
        for (n, per_axis) in [(12usize, 2usize), (12, 4), (24, 2)] {
            let h = 0.12 / n as f32;
            let wall = n / 12;
            let mut cells = vec![0u32; n * n * n];
            for z in 0..n {
                for y in 0..n {
                    for x in 0..n {
                        if (geometry == 1 && [x, y, z].iter().any(|&q| q < wall || q >= n - wall))
                            || (geometry == 2
                                && [x, y, z].iter().all(|&q| (n / 3..n * 2 / 3).contains(&q)))
                        {
                            cells[(z * n + y) * n + x] = 2;
                        }
                    }
                }
            }
            for (steps, dt) in [(4, 1f32 / 120.), (8, 1. / 240.)] {
                let mut p = Vec::new();
                for z in 0..n * per_axis {
                    for y in 0..n * per_axis {
                        for x in 0..n * per_axis {
                            if cells[((z / per_axis) * n + y / per_axis) * n + x / per_axis] == 2 {
                                continue;
                            }
                            p.push(
                                Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5)
                                    * (h / per_axis as f32),
                            );
                        }
                    }
                }
                let mut sim = configured_grid(&device, p.len(), n as u32, h);
                sim.open_boundaries = 0;
                let weight = 8. / per_axis.pow(3) as f32;
                sim.set_particle_volumes(vec![weight; p.len()]);
                let initial = p.clone();
                let mut v = vec![Vec3::ZERO; p.len()];
                let mut c = vec![Mat3::ZERO; p.len()];
                let rho = vec![1.; p.len()];
                for _ in 0..steps {
                    sim.step(
                        &device, &queue, &mut p, &mut v, &mut c, &rho, &cells, None, None, dt, 0.,
                        0., 80,
                    );
                    assert!(sim.take_phase_error().is_none());
                }
                let max_speed = v.iter().map(|v| v.length()).fold(0f32, f32::max);
                let max_drift = p
                    .iter()
                    .zip(&initial)
                    .map(|(p, q)| (*p - *q).length())
                    .fold(0f32, f32::max);
                let mass = 1000. * (h as f64 / per_axis as f64).powi(3);
                let d = (h as f64).powi(2) / 4.;
                let energy_j = v
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
                results.push(serde_json::json!({"geometry":geometry,"grid":n,"per_axis":per_axis,"dt":dt,"steps":steps,"particles":p.len(),"max_speed_mps":max_speed,"max_drift_m":max_drift,"energy_j":energy_j,"passed":max_speed<1e-5&&max_drift<1e-6&&energy_j<1e-10}));
            }
        }
    }
    eprintln!("resting-volume mesh/particle/time cases={}; max_speed={:.3e}; max_drift={:.3e}; max_energy_j={:.3e}",results.len(),
        results.iter().map(|r|r["max_speed_mps"].as_f64().unwrap()).fold(0f64,f64::max),results.iter().map(|r|r["max_drift_m"].as_f64().unwrap()).fold(0f64,f64::max),results.iter().map(|r|r["energy_j"].as_f64().unwrap()).fold(0f64,f64::max));
    if let Ok(path) = std::env::var("SLUICE_VOLUME_LAKE_REPORT") {
        std::fs::write(path, serde_json::to_string_pretty(&results).unwrap()).unwrap();
    }
    assert!(
        results.iter().all(|r| r["passed"] == true),
        "volume constraint moved or energized zero-gravity resting water"
    );
}

fn generalized_budget(p: &[Vec3], v: &[Vec3], c: &[Mat3], weights: &[f32]) -> (DVec3, f64) {
    let mut angular = DVec3::ZERO;
    let mut energy = 0.;
    let d = 0.01f64.powi(2) / 4.;
    for (((p, v), c), &mass) in p.iter().zip(v).zip(c).zip(weights) {
        let rows = [c.x_axis, c.y_axis, c.z_axis];
        let spin = DVec3::new(
            (rows[2].y - rows[1].z) as f64,
            (rows[0].z - rows[2].x) as f64,
            (rows[1].x - rows[0].y) as f64,
        ) * d;
        angular += (p.as_dvec3().cross(v.as_dvec3()) + spin) * mass as f64;
        energy += 0.5
            * mass as f64
            * (v.as_dvec3().length_squared()
                + d * rows
                    .iter()
                    .map(|c| c.as_dvec3().length_squared())
                    .sum::<f64>());
    }
    (angular, energy)
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn force_free_apic_flip_blends_preserve_angular_momentum_without_creating_energy() {
    let (device, queue) = device();
    let mut results = Vec::new();
    for blend in [0f32, 0.5, 0.97, 1.] {
        let mut sim = configured(&device, 1024);
        sim.flip_ratio = blend;
        sim.density_projection_enabled = false;
        let mut p = cloud();
        let weights: Vec<_> = (0..p.len()).map(|i| 0.7 + 0.3 * (i % 3) as f32).collect();
        sim.set_particle_volumes(weights.clone());
        let mut v: Vec<_> = p
            .iter()
            .map(|p| {
                Vec3::new(
                    0.1 + 0.04 * (70. * p.y).sin(),
                    0.03 * (80. * p.z).cos(),
                    0.02 * (60. * p.x).sin(),
                )
            })
            .collect();
        let mut c = vec![Mat3::ZERO; p.len()];
        let rho = vec![1.; p.len()];
        let before = generalized_budget(&p, &v, &c, &weights);
        sim.step(
            &device,
            &queue,
            &mut p,
            &mut v,
            &mut c,
            &rho,
            &[0; 12 * 12 * 12],
            None,
            None,
            1. / 120.,
            0.,
            0.,
            0,
        );
        assert!(sim.take_phase_error().is_none());
        let after = generalized_budget(&p, &v, &c, &weights);
        let angular_error = (after.0 - before.0).length();
        let energy_ratio = after.1 / before.1;
        results.push(serde_json::json!({"blend":blend,"angular_error":angular_error,"energy_ratio":energy_ratio,"passed":angular_error<1e-6&&energy_ratio<=1.000001}));
    }
    eprintln!(
        "force-free blend budgets: {}",
        serde_json::to_string(&results).unwrap()
    );
    if let Ok(path) = std::env::var("SLUICE_VOLUME_BLEND_REPORT") {
        std::fs::write(path, serde_json::to_string_pretty(&results).unwrap()).unwrap();
    }
    assert!(
        results.iter().all(|r| r["passed"] == true),
        "force-free transfer violated angular/energy budget"
    );
}

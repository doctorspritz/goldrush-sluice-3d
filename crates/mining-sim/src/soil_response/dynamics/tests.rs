use super::*;
fn patch() -> SurfacePatch {
    SurfacePatch::new(&[
        DVec3::new(-0.25, -0.1, -0.5),
        DVec3::new(-0.25, -0.1, 0.5),
        DVec3::new(0.25, -0.1, 0.5),
        DVec3::new(0.25, -0.1, -0.5),
    ])
    .unwrap()
}
fn plane() -> Plane {
    Plane::new(DVec3::ZERO, DVec3::Y).unwrap()
}
fn body(mass: f64, gap: f64) -> Body {
    Body::dynamic(
        RigidMotion {
            center_of_mass_m: DVec3::Y * (0.1 + gap),
            ..Default::default()
        },
        mass,
        DVec3::new(1.04, 1.25, 0.29) * (mass / 12.),
    )
    .unwrap()
}
fn fixed() -> Body {
    Body::fixed(RigidMotion::default()).unwrap()
}
fn audit(step: &Step) {
    let scale = (step.mechanical_before_j.abs()
        + step.mechanical_after_j.abs()
        + step.external_work_j.abs()
        + step.damping_loss_j.abs())
    .max(1.);
    assert!(
        step.energy_residual_j.abs() < 1e-8 * scale,
        "energy residual {} scale {scale}",
        step.energy_residual_j
    );
    assert!(
        step.material_work_residual_j.abs() < 1e-9 * scale,
        "material work residual {}",
        step.material_work_residual_j
    );
    assert!(
        step.contact_work_residual_j.abs() < 1e-9 * scale,
        "discrete contact work mismatch {}J",
        step.contact_work_residual_j
    );
    assert!(
        step.damping_loss_j >= -1e-10 * scale,
        "negative material dissipation"
    );
    assert!(
        step.linear_momentum_residual_ns.length() < 1e-7,
        "linear impulse residual {:?}",
        step.linear_momentum_residual_ns
    );
    assert!(
        step.angular_momentum_residual_nms.length() < 1e-7,
        "angular impulse residual {:?}",
        step.angular_momentum_residual_nms
    );
}
#[test]
fn loaded_dimensional_material_dynamics_converges_to_an_independent_oscillator() {
    let layer = NormalLayer::new(1e6, 0.2, 0.).unwrap();
    let patch = patch();
    let stiffness: f64 = 2.5e6;
    for mass in [100., 1000.] {
        let equilibrium = -mass * 9.81 / stiffness;
        let initial = 1.5 * equilibrium;
        let omega = (stiffness / mass).sqrt();
        let mut errors = Vec::new();
        for h in [1f64 / 480., 1. / 960., 1. / 1920.] {
            let mut bodies = [body(mass, initial), fixed()];
            let mut error = 0f64;
            let stride = ((1. / 480.) / h).round() as usize;
            let steps = (0.2 / h).round() as usize;
            for index in 0..steps {
                let step = step_pair(
                    layer,
                    &patch,
                    plane(),
                    bodies,
                    Loads {
                        gravity_m_s2: -DVec3::Y * 9.81,
                        ..Default::default()
                    },
                    h,
                    Settings::default(),
                )
                .unwrap();
                audit(&step);
                bodies = step.bodies;
                if (index + 1) % stride == 0 {
                    let t = (index + 1) as f64 * h;
                    let exact = equilibrium + (initial - equilibrium) * (omega * t).cos();
                    let exact_v = -(initial - equilibrium) * omega * (omega * t).sin();
                    error = error.max(
                        ((bodies[0].motion().center_of_mass_m.y - 0.1) - exact).abs()
                            / equilibrium.abs(),
                    );
                    error = error.max(
                        (bodies[0].motion().linear_velocity_m_s.y - exact_v).abs()
                            / (equilibrium.abs() * omega),
                    );
                }
            }
            errors.push(error);
        }
        assert!(
            errors[1] < 0.4 * errors[0] && errors[2] < 0.4 * errors[1],
            "oscillator failed second-order refinement: {errors:?}"
        );
        println!("DYNAMIC_OSCILLATOR mass={mass}kg normalized_errors={errors:?}");
    }
}
#[test]
fn damped_material_support_settles_to_force_over_stiffness() {
    let layer = NormalLayer::new(1e6, 0.2, 20_000.).unwrap();
    let patch = patch();
    for mass in [100., 1000.] {
        let equilibrium = -mass * 9.81 / 2.5e6;
        let mut bodies = [body(mass, 1.5 * equilibrium), fixed()];
        let mut dissipated = 0.;
        let mut peak = 0f64;
        for _ in 0..480 {
            let step = step_pair(
                layer,
                &patch,
                plane(),
                bodies,
                Loads {
                    gravity_m_s2: -DVec3::Y * 9.81,
                    ..Default::default()
                },
                1. / 960.,
                Settings::default(),
            )
            .unwrap();
            audit(&step);
            peak = peak.max(step.energy_residual_j.abs());
            dissipated += step.damping_loss_j;
            bodies = step.bodies;
        }
        let gap = bodies[0].motion().center_of_mass_m.y - 0.1;
        assert!((gap - equilibrium).abs() < 1e-6);
        assert!(bodies[0].motion().linear_velocity_m_s.length() < 1e-5);
        assert!(dissipated > 0.);
        println!("DYNAMIC_SUPPORT mass={mass}kg gap={gap}m expected={equilibrium}m dissipation={dissipated}J max_step_energy_residual={peak}J");
    }
}
#[test]
fn elastic_release_opens_without_losing_or_inventing_stored_energy() {
    let layer = NormalLayer::new(1e6, 0.2, 0.).unwrap();
    let patch = patch();
    let mut bodies = [body(100., -0.002), fixed()];
    let initial = 0.5 * 2.5e6 * 0.002f64.powi(2);
    let mut maximum = 0f64;
    for _ in 0..60 {
        let step = step_pair(
            layer,
            &patch,
            plane(),
            bodies,
            Loads::default(),
            1. / 960.,
            Settings::default(),
        )
        .unwrap();
        audit(&step);
        maximum = maximum.max(step.energy_residual_j.abs());
        assert!(step.mean_contact_forces_n[0].y >= -1e-12);
        bodies = step.bodies;
    }
    assert!(bodies[0].motion().center_of_mass_m.y > 0.1);
    assert!((bodies[0].kinetic_energy_j() - initial).abs() < 1e-7);
    assert!((bodies[0].motion().linear_velocity_m_s.y - (2. * initial / 100.).sqrt()).abs() < 1e-8);
    println!("DYNAMIC_RELEASE initial_strain={initial}J final_kinetic={}J max_step_energy_residual={maximum}J",bodies[0].kinetic_energy_j());
}
fn pair() -> [Body; 2] {
    let mut a = body(50., -0.001);
    a.motion.rotation = DQuat::from_rotation_z(0.01);
    a.motion.linear_velocity_m_s = DVec3::new(0.02, -0.1, 0.01);
    a.motion.angular_velocity_rad_s = DVec3::new(0.2, -0.1, 0.3);
    let b = Body::dynamic(
        RigidMotion {
            rotation: DQuat::from_rotation_x(-0.005),
            linear_velocity_m_s: DVec3::new(-0.01, 0.02, -0.03),
            angular_velocity_rad_s: DVec3::new(-0.1, 0.2, -0.05),
            ..Default::default()
        },
        200.,
        DVec3::new(8., 10., 12.),
    )
    .unwrap();
    [a, b]
}
#[test]
fn two_anisotropic_bodies_exchange_contact_energy_and_momentum() {
    let patch = patch();
    let layer = NormalLayer::new(100_000., 0.2, 500.).unwrap();
    let mut bodies = pair();
    let mut maximum = 0f64;
    let p0 = bodies
        .into_iter()
        .map(Body::linear_momentum_ns)
        .sum::<DVec3>();
    let l0 = bodies
        .into_iter()
        .map(Body::angular_momentum_nms)
        .sum::<DVec3>();
    for _ in 0..64 {
        let step = step_pair(
            layer,
            &patch,
            plane(),
            bodies,
            Loads::default(),
            1. / 1920.,
            Settings::default(),
        )
        .unwrap();
        audit(&step);
        maximum = maximum.max(step.energy_residual_j.abs());
        bodies = step.bodies;
    }
    assert!(
        (bodies
            .into_iter()
            .map(Body::linear_momentum_ns)
            .sum::<DVec3>()
            - p0)
            .length()
            < 1e-7
    );
    assert!(
        (bodies
            .into_iter()
            .map(Body::angular_momentum_nms)
            .sum::<DVec3>()
            - l0)
            .length()
            < 1e-7
    );
    println!(
        "DYNAMIC_ROTATING_PAIR 64 coupled partial-contact steps; max_energy_residual={maximum}J"
    );
}
#[test]
fn free_anisotropic_rotation_retains_energy_and_world_momentum_direction() {
    let patch = patch();
    let layer = NormalLayer::new(1e6, 0.2, 0.).unwrap();
    let a = Body::dynamic(
        RigidMotion {
            center_of_mass_m: DVec3::Y * 2.,
            rotation: DQuat::from_axis_angle(DVec3::new(0.2, 0.1, 0.4).normalize(), 0.5),
            linear_velocity_m_s: DVec3::new(0.1, 0.2, -0.1),
            angular_velocity_rad_s: DVec3::new(1., 1.5, 2.),
        },
        100.,
        DVec3::new(1., 2., 3.),
    )
    .unwrap();
    let mut bodies = [a, fixed()];
    let initial = a.kinetic_energy_j();
    let momentum = a.spin_momentum_nms();
    for _ in 0..120 {
        let step = step_pair(
            layer,
            &patch,
            plane(),
            bodies,
            Loads::default(),
            1. / 120.,
            Settings::default(),
        )
        .unwrap();
        audit(&step);
        bodies = step.bodies;
    }
    assert!((bodies[0].kinetic_energy_j() - initial).abs() < 1e-7);
    assert!((bodies[0].spin_momentum_nms() - momentum).length() < 1e-7);
}
#[test]
fn applied_world_loads_close_their_discrete_work_and_impulse() {
    let patch = patch();
    let layer = NormalLayer::new(1e6, 0.2, 0.).unwrap();
    let mut a = body(100., 2.);
    a.motion.angular_velocity_rad_s = DVec3::new(0.1, 0.2, 0.3);
    let mut bodies = [a, fixed()];
    let forces = DVec3::new(3., 4., -2.);
    let torque = DVec3::new(0.1, -0.2, 0.05);
    let loads = Loads {
        forces_n: [forces, DVec3::ZERO],
        torques_nm: [torque, DVec3::ZERO],
        ..Default::default()
    };
    let initial = a.spin_momentum_nms();
    for _ in 0..50 {
        let step = step_pair(
            layer,
            &patch,
            plane(),
            bodies,
            loads,
            0.001,
            Settings::default(),
        )
        .unwrap();
        audit(&step);
        bodies = step.bodies;
    }
    assert!((bodies[0].motion().linear_velocity_m_s - forces * 0.05 / 100.).length() < 1e-10);
    assert!((bodies[0].spin_momentum_nms() - initial - torque * 0.05).length() < 1e-8);
}
#[test]
fn same_material_patch_refinement_retains_coupled_body_motion() {
    let patch = patch();
    let layer = NormalLayer::new(100_000., 0.2, 500.).unwrap();
    let input = pair();
    let reference = step_pair(
        layer,
        &patch,
        plane(),
        input,
        Loads::default(),
        1. / 1920.,
        Settings::default(),
    )
    .unwrap();
    audit(&reference);
    for level in [1, 2, 3] {
        let fine = step_pair(
            layer,
            &patch.refined(level).unwrap(),
            plane(),
            input,
            Loads::default(),
            1. / 1920.,
            Settings::default(),
        )
        .unwrap();
        audit(&fine);
        for i in 0..2 {
            assert!(
                (fine.bodies[i].motion().center_of_mass_m
                    - reference.bodies[i].motion().center_of_mass_m)
                    .length()
                    < 1e-8
            );
            assert!(
                (fine.bodies[i].motion().linear_velocity_m_s
                    - reference.bodies[i].motion().linear_velocity_m_s)
                    .length()
                    < 1e-7
            );
            assert!(
                (fine.bodies[i].motion().angular_velocity_rad_s
                    - reference.bodies[i].motion().angular_velocity_rad_s)
                    .length()
                    < 1e-7
            );
        }
    }
}
#[test]
fn invalid_or_unresolved_steps_return_no_advanced_body_state() {
    let patch = patch();
    let layer = NormalLayer::new(1e6, 0.2, 500.).unwrap();
    let input = pair();
    assert!(Body::dynamic(RigidMotion::default(), -1., DVec3::ONE).is_err());
    assert!(Body::dynamic(RigidMotion::default(), 1., DVec3::new(1., 1., 4.)).is_err());
    assert!(Body::fixed(RigidMotion {
        linear_velocity_m_s: DVec3::Y,
        ..Default::default()
    })
    .is_err());
    assert!(step_pair(
        layer,
        &patch,
        plane(),
        input,
        Loads::default(),
        0.,
        Settings::default()
    )
    .is_err());
    let tiny_rotation = Settings {
        maximum_rotation_rad: 1e-12,
        ..Default::default()
    };
    assert!(step_pair(
        layer,
        &patch,
        plane(),
        input,
        Loads::default(),
        1. / 60.,
        tiny_rotation
    )
    .is_err());
    let impossible = Settings {
        maximum_iterations: 1,
        velocity_residual_tolerance: 1e-30,
        ..Default::default()
    };
    assert!(step_pair(
        layer,
        &patch,
        plane(),
        input,
        Loads::default(),
        1. / 60.,
        impossible
    )
    .is_err());
    let failed_work = step_pair(
        layer,
        &patch,
        plane(),
        input,
        Loads::default(),
        1. / 1920.,
        Settings {
            maximum_abs_work_residual_j: 1e-30,
            ..Settings::default()
        },
    );
    assert!(failed_work
        .unwrap_err()
        .contains("Patch work audit did not close"));
}

#[test]
fn coupled_motion_is_covariant_under_world_pose_and_galilean_velocity() {
    let patch = patch();
    let layer = NormalLayer::new(100_000., 0.2, 500.).unwrap();
    let original = pair();
    let h = 1. / 1920.;
    let reference = step_pair(
        layer,
        &patch,
        plane(),
        original,
        Loads::default(),
        h,
        Settings::default(),
    )
    .unwrap();
    audit(&reference);
    let rotation = DQuat::from_axis_angle(DVec3::new(0.2, -0.3, 0.7).normalize(), 0.8);
    let translation = DVec3::new(1., -2., 3.);
    let drift = DVec3::new(0.3, 0.2, -0.1);
    let mut input = original;
    for body in &mut input {
        body.motion.center_of_mass_m = rotation * body.motion.center_of_mass_m + translation;
        body.motion.rotation = rotation * body.motion.rotation;
        body.motion.linear_velocity_m_s = rotation * body.motion.linear_velocity_m_s + drift;
        body.motion.angular_velocity_rad_s = rotation * body.motion.angular_velocity_rad_s;
    }
    let transformed = step_pair(
        layer,
        &patch,
        plane(),
        input,
        Loads::default(),
        h,
        Settings::default(),
    )
    .unwrap();
    audit(&transformed);
    for i in 0..2 {
        let a = reference.bodies[i].motion();
        let b = transformed.bodies[i].motion();
        assert!(
            (rotation * a.center_of_mass_m + translation + drift * h - b.center_of_mass_m).length()
                < 1e-9
        );
        assert!((rotation * a.linear_velocity_m_s + drift - b.linear_velocity_m_s).length() < 1e-8);
        assert!((rotation * a.angular_velocity_rad_s - b.angular_velocity_rad_s).length() < 1e-8);
        assert!(
            ((rotation * a.rotation).conjugate() * b.rotation)
                .xyz()
                .length()
                < 1e-8
        );
        assert!(
            (rotation * reference.mean_contact_forces_n[i] - transformed.mean_contact_forces_n[i])
                .length()
                < 1e-6
        );
    }
}

#[test]
fn rotating_partial_contact_converges_under_physical_timestep_refinement() {
    let patch = patch();
    let layer = NormalLayer::new(100_000., 0.2, 500.).unwrap();
    let solve = |rate: usize| {
        let mut bodies = pair();
        for _ in 0..rate / 30 {
            let step = step_pair(
                layer,
                &patch,
                plane(),
                bodies,
                Loads::default(),
                1. / rate as f64,
                Settings::default(),
            )
            .unwrap();
            audit(&step);
            bodies = step.bodies;
        }
        bodies
    };
    let reference = solve(15360);
    let mut errors = Vec::new();
    for rate in [480, 960, 1920] {
        let bodies = solve(rate);
        let error = bodies
            .into_iter()
            .zip(reference)
            .map(|(a, b)| {
                let a = a.motion();
                let b = b.motion();
                (a.center_of_mass_m - b.center_of_mass_m).length()
                    + (a.linear_velocity_m_s - b.linear_velocity_m_s).length() / 30.
                    + (a.angular_velocity_rad_s - b.angular_velocity_rad_s).length() / 60.
            })
            .sum::<f64>();
        errors.push(error);
    }
    println!("ROTATING_TIMESTEP error_metric_m={errors:?}; same material/masses; reference15360Hz");
    assert!(
        errors[1] < 0.4 * errors[0] && errors[2] < 0.4 * errors[1],
        "rotating contact failed physical timestep refinement"
    );
}

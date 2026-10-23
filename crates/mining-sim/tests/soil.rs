use glam::{DVec3, Quat};
use mining_sim::{
    articulation::{self, Environment},
    site::{CELL, WIDTH},
    soil::{self, ActiveSoil, Clod},
    Excavator, MachineInput, Material, Session, Site, Vec3,
};
fn prism(width: f64, depth: f64, height: f64) -> [DVec3; 6] {
    [
        DVec3::ZERO,
        DVec3::new(width, 0., 0.),
        DVec3::new(0., 0., depth),
        DVec3::new(0., height, 0.),
        DVec3::new(width, height, 0.),
        DVec3::new(0., height, depth),
    ]
}
fn machine() -> Excavator {
    let mut m = Excavator::new(&Site::new());
    m.position = Vec3::new(16., 10., 16.);
    m.boom = 1.1;
    m.stick = -1.5;
    m.curl = -0.2;
    m.engine_on = false;
    m.enable_articulation().unwrap();
    m
}
fn totals(m: &Excavator, s: &ActiveSoil) -> (DVec3, DVec3, f64) {
    let f = &m.articulation.as_ref().unwrap().feedback;
    let t = s.totals(Vec3::ZERO);
    (
        f.linear_momentum_ns.as_dvec3() + t.momentum_ns,
        f.angular_momentum_nms.as_dvec3() + t.angular_nms,
        f.kinetic_j + f.elastic_j + t.kinetic_j,
    )
}
#[test]
fn simplex_moments_match_independent_triangular_prism_integrals() {
    let vertices = prism(0.4, 0.3, 0.2).map(|p| p + DVec3::new(20., 3., 16.));
    let (v, m) = soil::prism_moments(vertices, 12.).unwrap();
    assert!((v - 0.012).abs() < 1e-15);
    let center = DVec3::new(20. + 0.4 / 3., 3.1, 16. + 0.3 / 3.);
    assert!((m.center - center).length() < 1e-13);
    // Uniform right-triangle: Var(x)=a²/18, Var(z)=b²/18,
    // Cov(x,z)=-ab/36. Independent uniform vertical slab: Var(y)=h²/12.
    let c = m.second / 12.;
    assert!((c.x_axis.x - 0.4f64.powi(2) / 18.).abs() < 1e-14);
    assert!((c.z_axis.z - 0.3f64.powi(2) / 18.).abs() < 1e-14);
    assert!((c.y_axis.y - 0.2f64.powi(2) / 12.).abs() < 1e-14);
    assert!((c.x_axis.z + 0.4 * 0.3 / 36.).abs() < 1e-14);
    assert!(c.y_axis.x.abs() < 1e-14 && c.y_axis.z.abs() < 1e-14);
}
#[test]
fn terrain_activation_preserves_every_finite_stock_and_refines_the_same_volume() {
    let original = Site::new();
    let before = original.stored_material();
    let water = original.soil_water.stored_m3() + original.water_volume_m3();
    let node = 80 * WIDTH + 76;
    let mut measured = Vec::new();
    for n in [1, 2, 4] {
        let mut site = original.clone();
        let active = soil::activate(&mut site, &[(node, 0.12)], n).unwrap();
        assert!(active.valid());
        let mut after = site.stored_material();
        after.add(active.material());
        for [a, b] in [
            [before.soil_m3, after.soil_m3],
            [before.sand_m3, after.sand_m3],
            [before.gravel_m3, after.gravel_m3],
            [before.gold_g, after.gold_g],
            [before.pore_water_m3, after.pore_water_m3],
        ] {
            assert!((a - b).abs() < 1e-8, "stock changed {a} {b}");
        }
        assert!(
            (site.soil_water.stored_m3()
                + site.water_volume_m3()
                + active.material().pore_water_m3
                - water)
                .abs()
                < 1e-8
        );
        assert!(
            (active.material().volume() - 0.12f32 as f64 * (CELL as f64).powi(2)).abs() < 1e-10
        );
        assert!(active.geometry_volume_error_m3.abs() < 2e-7);
        let total = active.totals(Vec3::ZERO);
        let center = total.first_kg_m / total.mass_kg;
        assert!((center.x - 19.).abs() < 2e-6 && (center.z - 20.).abs() < 2e-6);
        measured.push((active.clods.len(), center, total.mass_kg));
        println!(
            "soil mesh n={n}: clods={} source={:?} geometry error={} COM={center}",
            active.clods.len(),
            active.material(),
            active.geometry_volume_error_m3
        );
    }
    for pair in measured.windows(2) {
        assert!(pair[1].0 > pair[0].0);
        assert!((pair[1].1 - pair[0].1).length() < 3e-6);
        assert!((pair[1].2 - pair[0].2).abs() < 1e-10);
    }
}
#[test]
fn activation_errors_are_transactional_and_wet_contact_is_explicitly_refused() {
    let mut site = Site::new();
    let before = serde_json::to_string(&site.snapshot()).unwrap();
    for nodes in [
        vec![],
        vec![(0, 0.1)],
        vec![(80 * WIDTH + 76, -0.1)],
        vec![(80 * WIDTH + 76, 0.1), (80 * WIDTH + 76, 0.1)],
    ] {
        assert!(soil::activate(&mut site, &nodes, 2).is_err());
        assert_eq!(before, serde_json::to_string(&site.snapshot()).unwrap());
    }
    let i = 80 * WIDTH + 76;
    site.world.water_surface[i] = site.world.ground_height(76, 80) + 0.05;
    let before = serde_json::to_string(&site.snapshot()).unwrap();
    assert!(soil::activate(&mut site, &[(i, 0.1)], 2).is_err());
    assert_eq!(before, serde_json::to_string(&site.snapshot()).unwrap());
}
#[test]
fn finite_soil_collision_reacts_on_the_floating_bucket_with_closed_momentum() {
    let mut m = machine();
    let arm = m.arm();
    let bucket_q = m.bucket_rotation();
    let q = bucket_q;
    let floor = arm.wrist + q * Vec3::new(0., -0.65, 0.4);
    let mut clod = Clod::prism(
        Material {
            sand_m3: 0.004,
            ..Default::default()
        },
        prism(0.2, 0.2, 0.2),
    )
    .unwrap();
    clod.center = floor + q * (clod.center + Vec3::new(-0.1, 0.14, -0.1));
    clod.rotation = q;
    clod.velocity = q * (-Vec3::Y);
    let mut s = ActiveSoil::from_clods(vec![clod]).unwrap();
    let before = totals(&m, &s);
    let mut contacts = 0;
    let mut maximum_residual = 0f64;
    for _ in 0..120 {
        articulation::step_soil(
            &mut m,
            MachineInput::default(),
            1. / 240.,
            &Environment::free(Vec3::ZERO),
            &mut s,
        )
        .unwrap();
        assert!(s.feedback.closed_free_system);
        contacts += s.feedback.bucket_contacts;
        maximum_residual = maximum_residual.max(s.feedback.coupled_passivity_residual_j);
    }
    let after = totals(&m, &s);
    println!("soil impact contacts={contacts}, dP={} dL={} initialK={} finalK={} max residual={maximum_residual}",(after.0-before.0).length(),(after.1-before.1).length(),before.2,after.2);
    assert!(contacts > 0);
    assert!((after.0 - before.0).length() < 0.003);
    assert!((after.1 - before.1).length() < 0.03);
    assert!(after.2 <= before.2 + 0.01);
    assert!(maximum_residual < 0.01);
    assert!(
        m.articulation
            .as_ref()
            .unwrap()
            .feedback
            .linear_momentum_ns
            .length()
            > 0.1,
        "soil did not push the actual machine"
    );
}
fn slope(degrees: f32, hz: u16) -> (f32, f32) {
    let mut m = machine();
    m.position = Vec3::new(16., 50., 16.);
    m.tracks.mechanical_hz = hz;
    m.enable_articulation().unwrap();
    let q = Quat::from_rotation_z(degrees.to_radians());
    let mut clod = Clod::prism(
        Material {
            sand_m3: 0.0003125,
            ..Default::default()
        },
        prism(0.25, 0.25, 0.01),
    )
    .unwrap();
    clod.center = q * (clod.center + Vec3::Y * 0.0001);
    clod.rotation = q;
    let initial = clod.center;
    let mut active = ActiveSoil::from_clods(vec![clod]).unwrap();
    let obstacles = [mining_sim::tracks::Obstacle {
        center: -(q * Vec3::Y) * 0.1,
        rotation: q,
        half_extents: Vec3::new(3., 0.1, 3.),
    }];
    let env = Environment {
        gravity: Vec3::new(0., -9.81, 0.),
        site: None,
        obstacles: &obstacles,
        loads: &[],
    };
    for _ in 0..(0.6 * hz as f32).round() as usize {
        articulation::step_soil(
            &mut m,
            MachineInput::default(),
            1. / hz as f32,
            &env,
            &mut active,
        )
        .unwrap();
    }
    let down = -(q * Vec3::X);
    let c = &active.clods[0];
    let result = ((c.center - initial).dot(down), c.velocity.dot(down));
    println!(
        "slope {degrees} degrees hz={hz}: displacement={} speed={} omega={} q={:?} friction={}",
        result.0,
        result.1,
        c.angular_velocity,
        c.rotation,
        c.friction()
    );
    result
}
#[test]
fn same_contact_solve_holds_below_friction_and_slides_above_it() {
    let a = slope(20., 240);
    assert!(a.0.abs() < 0.005 && a.1.abs() < 0.01);
    let b = slope(40., 240);
    let acceleration =
        9.81 * (40f32.to_radians().sin() - 30f32.to_radians().tan() * 40f32.to_radians().cos());
    assert!((b.1 - acceleration * 0.6).abs() < 0.04);
    assert!((b.0 - 0.5 * acceleration * 0.6 * 0.6).abs() < 0.02);
}
#[test]
fn physical_contact_timestep_refines_the_independent_sliding_solution() {
    let a = slope(40., 120);
    let b = slope(40., 240);
    let c = slope(40., 480);
    let acceleration =
        9.81 * (40f32.to_radians().sin() - 30f32.to_radians().tan() * 40f32.to_radians().cos());
    let expected = 0.5 * acceleration * 0.6 * 0.6;
    assert!((c.0 - expected).abs() < (a.0 - expected).abs());
    assert!((c.0 - b.0).abs() < (b.0 - a.0).abs());
}
#[test]
fn moving_soil_owns_material_and_v9_continuation_rejects_wrong_formats() {
    let mut s = Session::new();
    s.machine.enable_articulation().unwrap();
    let before = (
        s.accounted_bulk_material_m3(),
        s.accounted_dry_mass_kg(),
        s.stored_water_m3(),
        s.accounted_gold(),
    );
    s.activate_soil(&[(80 * WIDTH + 76, 0.12)], 2).unwrap();
    let after = (
        s.accounted_bulk_material_m3(),
        s.accounted_dry_mass_kg(),
        s.stored_water_m3(),
        s.accounted_gold(),
    );
    assert!(
        (before.0 - after.0).abs() < 1e-8
            && (before.1 - after.1).abs() < 1e-7
            && (before.2 - after.2).abs() < 1e-8
            && (before.3 - after.3).abs() < 1e-8
    );
    let encoded = s.save_json().unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(v["version"], 11);
    let mut restored = Session::load_json(&encoded).unwrap();
    assert_eq!(s.save_json().unwrap(), restored.save_json().unwrap());
    for _ in 0..10 {
        s.step(Default::default());
        restored.step(Default::default());
    }
    assert!(!s.paused);
    assert_eq!(s.save_json().unwrap(), restored.save_json().unwrap());
    // The live copy has derived geometry/inertia caches now; a loaded copy
    // reconstructs them. Cache lifetime must not change physical continuation.
    let mut cold = Session::load_json(&s.save_json().unwrap()).unwrap();
    for _ in 0..10 {
        s.step(Default::default());
        cold.step(Default::default());
    }
    assert!(!s.paused);
    assert_eq!(s.save_json().unwrap(), cold.save_json().unwrap());
    for version in [6, 7, 8, 9, 10] {
        v["version"] = serde_json::json!(version);
        assert!(Session::load_json(&v.to_string()).is_err());
    }
    v["version"] = serde_json::json!(11);
    v["active_soil"]["clods"][0]["vertices"][0][0] = serde_json::json!(20.);
    assert!(Session::load_json(&v.to_string()).is_err());
}
#[test]
fn prepared_bank_trial_moves_real_soil_with_actuator_commands() {
    let mut s = Session::new();
    s.prepare_soil_trial().unwrap();
    s.player.mode = mining_sim::ControlMode::Excavator;
    let mut stationary = s.clone();
    let positions = s
        .active_soil
        .as_ref()
        .unwrap()
        .clods
        .iter()
        .map(|c| c.center)
        .collect::<Vec<_>>();
    let material = s.active_soil.as_ref().unwrap().material();
    let before = (
        s.accounted_bulk_material_m3(),
        s.accounted_dry_mass_kg(),
        s.stored_water_m3(),
        s.accounted_gold(),
    );
    let mut contacts = 0;
    let start = std::time::Instant::now();
    for step in 0..300 {
        let input = match step {
            60..=119 => MachineInput {
                boom: -1.,
                ..Default::default()
            },
            120..=179 => MachineInput {
                curl: 1.,
                stick: -0.4,
                ..Default::default()
            },
            180..=239 => MachineInput {
                boom: 1.,
                ..Default::default()
            },
            _ => Default::default(),
        };
        s.step(mining_sim::session::Controls {
            machine: input,
            ..Default::default()
        });
        assert!(!s.paused, "{}", s.message);
        stationary.step(Default::default());
        assert!(!stationary.paused, "{}", stationary.message);
        contacts += s.active_soil.as_ref().unwrap().feedback.bucket_contacts;
    }
    let active = s.active_soil.as_ref().unwrap();
    let moved = active
        .clods
        .iter()
        .zip(positions)
        .filter(|(c, p)| c.center.distance(*p) > 0.02)
        .count();
    println!("bank soil clods={} material={material:?} contacts={contacts}, moved={moved}, wall={}s machine={:?} arm={:?}",active.clods.len(),start.elapsed().as_secs_f64(),s.machine.position,s.machine.arm());
    let operator_moved = active
        .clods
        .iter()
        .zip(&stationary.active_soil.as_ref().unwrap().clods)
        .filter(|(a, b)| a.center.distance(b.center) > 0.02)
        .count();
    println!("bank control: {operator_moved} clods differ by over 2 cm from the same bank without operator commands");
    assert!(contacts > 0 && moved > 0 && operator_moved > active.clods.len() / 10);
    assert!(
        (before.0 - s.accounted_bulk_material_m3()).abs() < 1e-8
            && (before.1 - s.accounted_dry_mass_kg()).abs() < 1e-7
            && (before.2 - s.stored_water_m3()).abs() < 1e-8
            && (before.3 - s.accounted_gold()).abs() < 1e-8
    );
}

#[test]
fn soil_trial_rebuilds_an_already_selected_stationary_articulation_before_time_zero() {
    let mut a = Session::new();
    let mut b = a.clone();
    b.machine.enable_articulation().unwrap();
    a.prepare_soil_trial().unwrap();
    b.prepare_soil_trial().unwrap();
    assert_eq!(a.save_json().unwrap(), b.save_json().unwrap());
    let mut moving = Session::new();
    moving.machine.enable_articulation().unwrap();
    moving.machine.articulation.as_mut().unwrap().velocity[0] = 0.1;
    assert!(moving.prepare_soil_trial().is_err());
}

#[test]
fn shell_cheek_impacts_recoil_the_machine_without_inventing_momentum_or_energy() {
    for direction in [-1., 1.] {
        let mut m = machine();
        assert_eq!(m.bucket_model(), mining_sim::bucket::Model::MountedShellV2);
        let arm = m.arm();
        let q = m.bucket_rotation();
        let mut clod = Clod::prism(
            Material {
                soil_m3: 0.08 * 0.08 * 0.06 * 0.5,
                ..Default::default()
            },
            prism(0.08, 0.08, 0.06),
        )
        .unwrap();
        clod.center = arm.wrist + q * (clod.center + Vec3::new(-0.04, -0.50, 0.4));
        clod.rotation = q;
        clod.velocity = q * Vec3::X * (direction * 2.);
        let mut soil = ActiveSoil::from_clods(vec![clod]).unwrap();
        let before = totals(&m, &soil);
        let mut contacts = 0;
        for _ in 0..120 {
            articulation::step_soil(
                &mut m,
                MachineInput::default(),
                1. / 240.,
                &Environment::free(Vec3::ZERO),
                &mut soil,
            )
            .unwrap();
            contacts += soil.feedback.bucket_contacts;
        }
        let after = totals(&m, &soil);
        println!(
            "cheek sign={direction} contacts={contacts} dP={} dL={} energy={} -> {}",
            (after.0 - before.0).length(),
            (after.1 - before.1).length(),
            before.2,
            after.2
        );
        assert!(contacts > 0);
        assert!((after.0 - before.0).length() < 0.003);
        assert!((after.1 - before.1).length() < 0.03);
        assert!(after.2 <= before.2 + 0.01);
        assert!(
            m.articulation
                .as_ref()
                .unwrap()
                .feedback
                .linear_momentum_ns
                .length()
                > 0.1
        );
    }
}

#[test]
fn actual_bank_clods_are_scooped_held_airborne_and_emptied_by_the_open_shell() {
    let mut s = Session::new();
    s.prepare_soil_trial().unwrap();
    s.player.mode = mining_sim::ControlMode::Excavator;
    let source = s.active_soil.as_ref().unwrap().material();
    let mut held = f64::INFINITY;
    for _ in 0..1080 {
        let input = mining_sim::replay::soil_trial_controls(&s, s.sim_time);
        s.step(mining_sim::session::Controls {
            machine: input,
            ..Default::default()
        });
        assert!(!s.paused, "{}", s.message);
        assert!(
            s.machine.bucket.is_empty(),
            "independent soil was captured into scalar custody"
        );
        assert_eq!(s.active_soil.as_ref().unwrap().material(), source);
        for (i, clod) in s.active_soil.as_ref().unwrap().clods.iter().enumerate() {
            let highest = clod
                .vertices()
                .iter()
                .map(|v| clod.center + clod.rotation * *v)
                .map(|p| p.y - soil::substrate_height(&s.site, p.x, p.z))
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                highest > -0.025,
                "clod {i} escaped the substrate during full cycle at {}: highest={highest}",
                s.sim_time
            );
        }
        if (9.0..10.0).contains(&s.sim_time) {
            let soil = s.active_soil.as_ref().unwrap();
            let contents = mining_sim::bucket::estimated_soil_contents(&s.machine, soil);
            held = held.min(contents.volume());
            assert!(
                contents.volume() > 0.001,
                "no meaningful load held clear of the bank"
            );
            let (w, q) = articulation::payload_frame(&s.machine);
            for c in &soil.clods {
                if mining_sim::bucket::geometry().contains(q.inverse() * (c.center.as_dvec3() - w))
                {
                    assert!(
                        c.vertices().iter().all(|v| {
                            let p = c.center + c.rotation * (*v);
                            p.y - s.site.height(p.x, p.z) > 0.25
                        }),
                        "counted clod was still supported by the bank"
                    );
                }
            }
        }
    }
    let empty =
        mining_sim::bucket::estimated_soil_contents(&s.machine, s.active_soil.as_ref().unwrap());
    println!(
        "real-bank minimum held={} L, final contents={} L",
        held * 1000.,
        empty.volume() * 1000.
    );
    assert_eq!(
        empty.volume(),
        0.,
        "opening did not release all counted clods"
    );
    assert_eq!(source, s.active_soil.as_ref().unwrap().material());
}

#[test]
fn delayed_operator_inputs_do_not_push_complete_clods_through_the_substrate() {
    for (lag, refresh) in [(2, 1), (2, 2), (4, 2), (6, 2)] {
        let mut s = Session::new();
        s.prepare_soil_trial().unwrap();
        s.player.mode = mining_sim::ControlMode::Excavator;
        let source = s.active_soil.as_ref().unwrap().material();
        let mut delayed = std::collections::VecDeque::new();
        let mut latched = MachineInput::default();
        for tick in 0..180 {
            delayed.push_back(mining_sim::replay::soil_trial_controls(&s, s.sim_time));
            let next = if delayed.len() > lag {
                delayed.pop_front().unwrap()
            } else {
                MachineInput::default()
            };
            if tick % refresh == 0 {
                latched = next;
            }
            s.step(mining_sim::session::Controls {
                machine: latched,
                ..Default::default()
            });
            assert!(!s.paused, "{}", s.message);
            let soil = s.active_soil.as_ref().unwrap();
            assert_eq!(soil.material(), source);
            for (i, clod) in soil.clods.iter().enumerate() {
                // Compare with the actual triangle surface, not the smoother
                // bilinear presentation query. A complete body below this surface
                // has escaped the substrate, even if its material still balances.
                let highest = clod
                    .vertices()
                    .iter()
                    .map(|p| clod.center + clod.rotation * *p)
                    .map(|p| p.y - soil::substrate_height(&s.site, p.x, p.z))
                    .fold(f32::NEG_INFINITY, f32::max);
                assert!(
                highest > -0.025,
                "lag={lag} refresh={refresh} clod {i} passed through terrain at t={}: best vertex clearance={highest}",
                s.sim_time
            );
            }
        }
    }
}

#[test]
fn small_fast_spin_has_finite_low_point_speed_and_closed_free_motion() {
    let mut m = machine();
    m.articulation.as_mut().unwrap().stiffness_nm_rad = [0.; 4];
    m.articulation.as_mut().unwrap().damping_nms_rad = [0.; 4];
    let mut c = Clod::prism(
        Material {
            sand_m3: 0.02 * 0.02 * 0.02 * 0.5,
            ..Default::default()
        },
        prism(0.02, 0.02, 0.02),
    )
    .unwrap();
    c.center = Vec3::new(25., 10., 25.);
    c.angular_velocity = Vec3::Y * 200.;
    assert!(c.valid(), "a small clod at low surface speed was rejected");
    let mut explosive = c.clone();
    explosive.angular_velocity = Vec3::Y * 20000.;
    assert!(!explosive.valid(), "surface-speed limit was lost");
    let mut combined = c.clone();
    combined.velocity = Vec3::X * 99.;
    combined.angular_velocity = Vec3::Y * 1000.;
    assert!(
        !combined.valid(),
        "combined translational and rotational point speed was not bounded"
    );
    let mut soil = ActiveSoil::from_clods(vec![c]).unwrap();
    let before = totals(&m, &soil);
    for _ in 0..120 {
        articulation::step_soil(
            &mut m,
            MachineInput::default(),
            1. / 240.,
            &Environment::free(Vec3::ZERO),
            &mut soil,
        )
        .unwrap();
    }
    let after = totals(&m, &soil);
    assert!((after.0 - before.0).length() < 0.003);
    assert!((after.1 - before.1).length() < 0.03);
    println!(
        "free spin total energy={} -> {}, clod energy={}",
        before.2,
        after.2,
        soil.totals(Vec3::ZERO).kinetic_j
    );
    assert!(
        (after.2 - before.2).abs() < before.2 * 0.0001 + 1e-8,
        "free spin invented or erased energy"
    );
    let expected = glam::DQuat::from_axis_angle(DVec3::Y, 100.);
    let actual = soil.clods[0].rotation.as_dquat().normalize();
    let error = expected.inverse() * actual;
    let phase_error = 2. * error.xyz().length().atan2(error.w.abs());
    println!("principal spin phase error={phase_error} rad");
    assert!(
        phase_error < 1e-4,
        "principal spin orientation lagged analytic motion"
    );
}

fn asymmetric_clod(speed: f32) -> Clod {
    let mut c = Clod::prism(
        Material {
            sand_m3: 0.09 * 0.04 * 0.03 * 0.5,
            ..Default::default()
        },
        prism(0.09, 0.04, 0.03),
    )
    .unwrap();
    c.center = Vec3::new(25., 10., 25.);
    c.rotation = Quat::from_rotation_x(0.31) * Quat::from_rotation_y(-0.4);
    c.angular_velocity = c.rotation * (Vec3::new(12., 17., 23.) * speed);
    c
}
fn torque_free_reference(initial: &Clod, steps: usize) -> (glam::DQuat, DVec3) {
    let inertia = initial.moments().inertia();
    let mut q = initial.rotation.as_dquat().normalize();
    let momentum = q * (inertia * (q.inverse() * initial.angular_velocity.as_dvec3()));
    // Independent f64 quaternion RK4 at constant WORLD angular momentum.
    // This does not call the production exponential-midpoint implementation.
    let derivative = |q: glam::DQuat| {
        let q = q.normalize();
        let omega = q * (inertia.inverse() * (q.inverse() * momentum));
        glam::DQuat::from_xyzw(omega.x, omega.y, omega.z, 0.) * q * 0.5
    };
    let h = 0.5 / steps as f64;
    for _ in 0..steps {
        let a = derivative(q);
        let b = derivative(q + a * (h * 0.5));
        let c = derivative(q + b * (h * 0.5));
        let d = derivative(q + c * h);
        q = (q + (a + b * 2. + c * 2. + d) * (h / 6.)).normalize();
    }
    (q, momentum)
}
fn rotation_distance(a: glam::DQuat, b: glam::DQuat) -> f64 {
    let d = a.inverse() * b;
    2. * d.xyz().length().atan2(d.w.abs())
}
fn measure_tumbling(initial: &Clod, reference: glam::DQuat, momentum: DVec3, hz: u16) -> [f64; 3] {
    let mut m = machine();
    m.tracks.mechanical_hz = hz;
    m.articulation.as_mut().unwrap().stiffness_nm_rad = [0.; 4];
    m.articulation.as_mut().unwrap().damping_nms_rad = [0.; 4];
    let mut soil = ActiveSoil::from_clods(vec![initial.clone()]).unwrap();
    let before = soil.totals(Vec3::ZERO);
    for _ in 0..hz / 2 {
        articulation::step_soil(
            &mut m,
            MachineInput::default(),
            1. / hz as f32,
            &Environment::free(Vec3::ZERO),
            &mut soil,
        )
        .unwrap();
    }
    let after = soil.totals(Vec3::ZERO);
    let angle = rotation_distance(reference, soil.clods[0].rotation.as_dquat().normalize());
    let de = (after.kinetic_j - before.kinetic_j).abs() / before.kinetic_j;
    let dl = (after.angular_nms - momentum).length() / momentum.length();
    assert_eq!(soil.clods[0].material(), initial.material());
    assert!(after.momentum_ns.length() < 1e-10);
    assert!(soil.clods[0].center == initial.center);
    println!("TUMBLING speed={} hz={hz} orientation_rad={angle} relative_energy={de} relative_momentum={dl}",initial.angular_velocity.length());
    [angle, de, dl]
}
#[test]
fn asymmetric_clod_tumbling_matches_independent_torque_free_reference() {
    let initial = asymmetric_clod(1.);
    let (reference, momentum) = torque_free_reference(&initial, 8192);
    assert!(rotation_distance(reference, torque_free_reference(&initial, 4096).0) < 1e-9);
    let measured: [_; 4] =
        [120, 240, 480, 960].map(|hz| measure_tumbling(&initial, reference, momentum, hz));
    for e in measured {
        assert!(
            e[0] < 5e-4 && e[1] < 5e-6 && e[2] < 5e-6,
            "torque-free error {e:?}"
        );
    }
    // Angular accuracy scales with mechanical time, so 480/960 Hz exhibit
    // second-order phase convergence. Every coarse case stays checked.
    assert!(measured[3][0] < measured[2][0] * 0.35);
}
#[test]
fn asymmetric_fast_clod_tumbling_retains_full_time_and_invariant_budgets() {
    let initial = asymmetric_clod(6.);
    assert!(initial.valid());
    let (reference, momentum) = torque_free_reference(&initial, 32768);
    assert!(rotation_distance(reference, torque_free_reference(&initial, 16384).0) < 1e-8);
    let measured: [_; 3] =
        [240, 480, 960].map(|hz| measure_tumbling(&initial, reference, momentum, hz));
    for e in measured {
        assert!(
            e[0] < 0.005 && e[1] < 1e-5 && e[2] < 5e-6,
            "fast torque-free error {e:?}"
        );
    }
    assert!(measured[2][0] < measured[0][0] * 0.8);
}

#[test]
fn unforced_zero_spin_clods_preserve_exact_pose_across_world_rebuilds() {
    for rotation in [
        Quat::IDENTITY,
        Quat::from_rotation_x(0.31) * Quat::from_rotation_y(-0.4),
    ] {
        let mut initial = asymmetric_clod(0.);
        initial.rotation = rotation;
        let mut m = machine();
        m.articulation.as_mut().unwrap().stiffness_nm_rad = [0.; 4];
        m.articulation.as_mut().unwrap().damping_nms_rad = [0.; 4];
        let mut soil = ActiveSoil::from_clods(vec![initial.clone()]).unwrap();
        for _ in 0..120 {
            articulation::step_soil(
                &mut m,
                MachineInput::default(),
                1. / 240.,
                &Environment::free(Vec3::ZERO),
                &mut soil,
            )
            .unwrap();
        }
        assert_eq!(soil.clods[0].center, initial.center);
        assert_eq!(soil.clods[0].rotation, initial.rotation);
        assert_eq!(soil.clods[0].velocity, Vec3::ZERO);
        assert_eq!(soil.clods[0].angular_velocity, Vec3::ZERO);
        assert_eq!(soil.material(), initial.material());
    }
}

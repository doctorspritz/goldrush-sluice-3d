use glam::Quat;
use mining_sim::{Excavator, MachineInput, Session, Site, Vec3};

fn plane(x_slope: f32, z_slope: f32) -> Site {
    let mut site = Site::new();
    for z in 0..128 {
        for x in 0..128 {
            let i = site.world.idx(x, z);
            site.world.bedrock_elevation[i] =
                x_slope * (x as f32 * 0.25 - 16.) + z_slope * (z as f32 * 0.25 - 16.);
            site.world.overburden_thickness[i] = 0.;
            site.world.paydirt_thickness[i] = 0.;
            site.world.gravel_thickness[i] = 0.;
            site.world.terrain_sediment[i] = 0.;
            site.world.water_surface[i] = site.world.bedrock_elevation[i];
        }
    }
    site
}
fn machine(site: &Site) -> Excavator {
    let mut m = Excavator::new(site);
    m.position = Vec3::new(16., site.height(16., 16.), 16.);
    m.boom = 1.1;
    m.stick = -1.5;
    m.curl = -0.2;
    m
}
fn run(m: &mut Excavator, site: &mut Site, input: MachineInput, dt: f32, seconds: f32) {
    for _ in 0..(seconds / dt).round() as usize {
        m.step(input, site, dt);
        assert!(
            m.valid(),
            "invalid pose/velocity: {:?} {:?}",
            m.position,
            m.tracks.velocity
        );
    }
}

#[test]
fn unsupported_engine_off_machine_falls_with_gravity_and_has_no_airborne_traction() {
    for engine in [false, true] {
        let mut site = plane(0., 0.);
        let mut m = machine(&site);
        m.position.y = 10.;
        m.engine_on = engine;
        run(
            &mut m,
            &mut site,
            MachineInput {
                drive: 1.,
                steer: 1.,
                ..Default::default()
            },
            1. / 60.,
            0.2,
        );
        assert!((m.tracks.velocity.y + 9.81 * 0.2).abs() < 0.0001, "{m:?}");
        assert!(m.tracks.velocity.x.abs() < 1e-6 && m.tracks.velocity.z.abs() < 1e-6);
        assert_eq!(m.tracks.feedback.contacts, 0);
        assert!(m.tracks.angular_velocity.length() < 1e-5);
        // Independent ballistic limit; the discrete integration error shrinks
        // with the 240 Hz contact timestep and must remain below 5 mm.
        assert!((m.position.y - (10. - 0.5 * 9.81 * 0.2f32.powi(2))).abs() < 0.005);
    }
}

#[test]
fn flat_track_reactions_support_actual_mass_and_wet_payload_without_idle_mining() {
    for load in [0., 0.20] {
        let mut site = plane(0., 0.);
        let mut m = machine(&site);
        m.bucket.gravel_m3 = load;
        m.bucket.pore_water_m3 = load * 0.2;
        let payload = m.bucket.wet_mass_kg() as f32;
        let material = m.bucket;
        m.engine_on = false;
        run(&mut m, &mut site, MachineInput::default(), 1. / 60., 2.);
        let f = m.tracks.feedback;
        eprintln!("flat load={load}: {f:?}, v={:?}", m.tracks.velocity);
        assert!((f.mass_kg - 8690. - payload).abs() < 0.01);
        assert!((f.normal_force_n / (f.mass_kg * 9.81) - 1.).abs() < 0.01);
        assert!(m.up().dot(Vec3::Y) > 0.9999);
        assert!(m.position.y.abs() < 0.002);
        assert!(m.tracks.velocity.length() < 0.005);
        assert_eq!(m.bucket, material);
        assert!(f.maximum_friction_ratio <= m.tracks.friction + 0.0001);
    }
}

#[test]
fn belt_drive_accelerates_forward_brakes_and_steers_through_ground_contact() {
    let mut site = plane(0., 0.);
    let mut m = machine(&site);
    run(
        &mut m,
        &mut site,
        MachineInput {
            drive: 1.,
            ..Default::default()
        },
        1. / 60.,
        2.,
    );
    eprintln!(
        "drive: pos={:?} v={:?} heading={}",
        m.position, m.tracks.velocity, m.heading
    );
    assert!(
        m.position.z > 17.5,
        "drive did not move forward: {:?}",
        m.position
    );
    assert!((1.2..1.6).contains(&m.tracks.velocity.z));
    assert!((m.position.x - 16.).abs() < 0.05);
    let stop = m.position;
    run(&mut m, &mut site, MachineInput::default(), 1. / 60., 1.);
    assert!(
        (m.position - stop).length() < 0.5,
        "brake teleported or slid: {:?}",
        m.position - stop
    );
    assert!(m.tracks.velocity.length() < 0.02);
    let heading = m.heading;
    run(
        &mut m,
        &mut site,
        MachineInput {
            steer: 1.,
            ..Default::default()
        },
        1. / 60.,
        1.,
    );
    let turned = (m.heading - heading)
        .sin()
        .atan2((m.heading - heading).cos());
    assert!(turned > 0.1, "steering direction/response wrong: {turned}");
}

#[test]
fn zero_friction_cannot_drive_or_steer_a_supported_machine() {
    let mut site = plane(0., 0.);
    let mut m = machine(&site);
    m.tracks.friction = 0.;
    run(
        &mut m,
        &mut site,
        MachineInput {
            drive: 1.,
            steer: 1.,
            ..Default::default()
        },
        1. / 60.,
        1.,
    );
    assert!((m.position.x - 16.).abs() < 0.001 && (m.position.z - 16.).abs() < 0.001);
    assert!(m.tracks.feedback.normal_force_n > 80_000.);
    assert_eq!(m.tracks.feedback.maximum_friction_ratio, 0.);
}

#[test]
fn tracks_follow_both_oriented_grades_and_hold_below_the_friction_angle() {
    for (a, b, heading) in [(0.12, 0., 0.), (0., 0.12, 0.), (0.08, 0.09, 0.7)] {
        let mut site = plane(a, b);
        let mut m = machine(&site);
        m.engine_on = false;
        m.heading = heading;
        let normal = Vec3::new(-a, 1., -b).normalize();
        let world = Quat::from_rotation_arc(Vec3::Y, normal);
        m.tracks.tilt = Quat::from_rotation_y(-heading) * world;
        let before = m.position;
        run(&mut m, &mut site, MachineInput::default(), 1. / 60., 2.);
        eprintln!(
            "grade ({a},{b}): tilt={} drift={:?}",
            m.tracks.feedback.tilt_deg,
            m.position - before
        );
        assert!(
            m.up().dot(normal) > 0.9999,
            "grade normal mismatch: {:?} vs {normal:?}",
            m.up()
        );
        assert!((m.position - before).length() < 0.02);
        assert!(m.tracks.velocity.length() < 0.01);
    }
}

#[test]
fn removing_track_support_tips_the_unoccupied_engine_off_machine() {
    let mut site = plane(0., 0.);
    let mut m = machine(&site);
    m.engine_on = false;
    for z in 0..128 {
        for x in 0..66 {
            let i = site.world.idx(x, z);
            site.world.bedrock_elevation[i] = -3.;
            site.world.water_surface[i] = -3.;
        }
    }
    let mut minimum_up = 1f32;
    for _ in 0..120 {
        m.step(MachineInput::default(), &mut site, 1. / 60.);
        assert!(m.valid());
        minimum_up = minimum_up.min(m.up().y);
    }
    eprintln!(
        "support loss: minimum up={minimum_up}, final={:?}",
        m.tracks.feedback
    );
    assert!(minimum_up < 0.8, "unsupported machine stayed upright");
    assert!(m.tracks.feedback.kinetic_energy_j < 1_000_000.);
}

#[test]
fn machine_contact_is_invariant_to_outer_step_batching_without_changing_terrain() {
    let mut results = Vec::new();
    for dt in [1. / 30., 1. / 60., 1. / 120.] {
        let mut site = plane(0.04, 0.02);
        let mut m = machine(&site);
        m.tracks.tilt = Quat::from_rotation_arc(Vec3::Y, Vec3::new(-0.04, 1., -0.02).normalize());
        let ground = site.world.bedrock_elevation.clone();
        run(
            &mut m,
            &mut site,
            MachineInput {
                drive: 0.5,
                steer: 0.2,
                ..Default::default()
            },
            dt,
            2.,
        );
        assert_eq!(site.world.bedrock_elevation, ground);
        results.push((m.position, m.tracks.velocity, m.heading));
    }
    eprintln!("time convergence: {results:?}");
    let reference = results[2];
    for r in &results[..2] {
        assert!(
            r.0.distance(reference.0) < 0.025,
            "position did not converge: {results:?}"
        );
        assert!(
            r.1.distance(reference.1) < 0.03,
            "velocity did not converge: {results:?}"
        );
        assert!(
            (r.2 - reference.2).abs() < 0.015,
            "yaw did not converge: {results:?}"
        );
    }
}

#[test]
fn moving_machine_save_restores_exact_dynamics_and_legacy_inventory() {
    let mut session = Session::new();
    let input = MachineInput {
        drive: 0.5,
        steer: 0.1,
        ..Default::default()
    };
    for _ in 0..10 {
        session.machine.step(input, &mut session.site, 1. / 60.);
    }
    let saved = session.save_json().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&saved).unwrap()["version"],
        6
    );
    let mut restored = Session::load_json(&saved).unwrap();
    for _ in 0..30 {
        session.machine.step(input, &mut session.site, 1. / 60.);
        restored.machine.step(input, &mut restored.site, 1. / 60.);
    }
    assert_eq!(session.machine.position, restored.machine.position);
    assert_eq!(
        session.machine.tracks.velocity,
        restored.machine.tracks.velocity
    );
    assert_eq!(session.machine.tracks.tilt, restored.machine.tracks.tilt);
    let mut invalid: serde_json::Value = serde_json::from_str(&saved).unwrap();
    invalid["version"] = 5.into();
    assert!(Session::load_json(&invalid.to_string()).is_err());
    invalid["machine"].as_object_mut().unwrap().remove("tracks");
    let legacy = Session::load_json(&invalid.to_string()).unwrap();
    assert_eq!(legacy.machine.bucket, session.machine.bucket);
    assert_eq!(legacy.machine.tracks.velocity, Vec3::ZERO);
    assert_eq!(legacy.machine.tracks.tilt, Quat::IDENTITY);
}

#[test]
fn physical_mechanical_step_refinement_reduces_ballistic_error_and_stabilizes_drive() {
    let mut fall = Vec::new();
    let mut drive = Vec::new();
    for hz in [120, 240, 480] {
        let mut site = plane(0., 0.);
        let mut m = machine(&site);
        m.tracks.mechanical_hz = hz;
        m.engine_on = false;
        m.position.y = 10.;
        run(&mut m, &mut site, MachineInput::default(), 1. / 60., 0.2);
        fall.push((m.position.y - (10. - 0.5 * 9.81 * 0.2f32.powi(2))).abs());
        let mut m = machine(&site);
        m.tracks.mechanical_hz = hz;
        run(
            &mut m,
            &mut site,
            MachineInput {
                drive: 0.5,
                steer: 0.2,
                ..Default::default()
            },
            1. / 60.,
            2.,
        );
        drive.push((m.position, m.tracks.velocity, m.heading));
    }
    eprintln!("physical cadence fall errors={fall:?}, drive={drive:?}");
    assert!(fall[1] < fall[0] * 0.6 && fall[2] < fall[1] * 0.6);
    assert!(drive[1].0.distance(drive[2].0) < 0.025);
    assert!(drive[1].1.distance(drive[2].1) < 0.03);
    assert!((drive[1].2 - drive[2].2).abs() < 0.015);
}

#[test]
fn freely_rotating_machine_keeps_its_center_of_mass_on_a_ballistic_path() {
    let mut site = plane(0., 0.);
    let mut m = machine(&site);
    m.position.y = 10.;
    m.engine_on = false;
    m.tracks.angular_velocity = Vec3::Y * 0.6;
    m.step(MachineInput::default(), &mut site, 1. / 60.);
    let start = m.tracks.feedback.center_of_mass;
    run(&mut m, &mut site, MachineInput::default(), 1. / 60., 0.2);
    let end = m.tracks.feedback.center_of_mass;
    assert!(
        (end.x - start.x).abs() < 0.0005 && (end.z - start.z).abs() < 0.0005,
        "rotation introduced COM translation: {start:?} -> {end:?}"
    );
    assert!((m.tracks.velocity.y + 9.81 * (0.2 + 1. / 60.)).abs() < 0.0001);
}

#[test]
fn fast_reversing_machine_hits_thin_equipment_without_crossing_or_mining_it() {
    let mut site = plane(0., 0.);
    let mut m = machine(&site);
    m.tracks.velocity.z = -12.;
    let wall = mining_sim::tracks::Obstacle {
        center: Vec3::new(16., 50., 12.),
        rotation: Quat::IDENTITY,
        half_extents: Vec3::new(32., 50., 0.005),
    };
    for _ in 0..60 {
        m.step_with_obstacles(MachineInput::default(), &mut site, 1. / 60., &[wall]);
        assert!(m.valid());
        // Check the actual rear track corners, not just the chassis origin:
        // a root-only check could miss over a metre of penetration.
        let rotation = m.chassis_rotation();
        for side in [-1., 1.] {
            for edge in [-1., 1.] {
                for y in [0., 0.65] {
                    let corner =
                        m.position + rotation * Vec3::new(side * 0.925 + edge * 0.225, y, -1.44);
                    assert!(
                        corner.z >= 11.995,
                        "track penetrated a thin rigid wall: {corner:?}, body={:?}",
                        m.position
                    );
                }
            }
        }
    }
    assert!(m.bucket.is_empty());
}

#[test]
fn unoccupied_machine_continues_to_fall_without_operator_input() {
    let mut s = Session::new();
    s.machine.position.y += 5.;
    s.machine.engine_on = false;
    let gold = s.accounted_gold();
    let player = s.player.position;
    for _ in 0..12 {
        s.step(Default::default());
    }
    assert!(s.machine.tracks.velocity.y < -1.9);
    assert_eq!(s.player.position, player);
    assert_eq!(s.accounted_gold(), gold);
    assert!(s.machine.bucket.is_empty());
}

#[test]
fn bucket_floor_and_linkage_rotate_rigidly_with_the_complete_chassis_frame() {
    let site = plane(0., 0.);
    let base = machine(&site);
    let flat = base.arm();
    for tilt in [
        Quat::from_rotation_x(0.3),
        Quat::from_rotation_z(-0.25),
        Quat::from_rotation_x(0.2) * Quat::from_rotation_z(0.15),
    ] {
        let mut m = base.clone();
        m.heading = 0.7;
        m.tracks.tilt = tilt;
        let q = m.chassis_rotation();
        let arm = m.arm();
        for (name, old, actual) in [
            ("shoulder", flat.shoulder, arm.shoulder),
            ("elbow", flat.elbow, arm.elbow),
            ("wrist", flat.wrist, arm.wrist),
            ("heel", flat.heel, arm.heel),
            ("tooth", flat.lip, arm.lip),
        ] {
            let expected = base.position + q * (old - base.position);
            assert!(
                actual.distance(expected) < 0.000005,
                "{name} did not rotate with chassis: {actual:?} vs {expected:?}"
            );
        }
    }
}

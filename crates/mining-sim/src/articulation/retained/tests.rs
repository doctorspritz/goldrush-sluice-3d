use super::*;
use crate::soil::{ActiveSoil, Clod};
use crate::Material;
fn bank() -> crate::Session {
    let mut s = crate::Session::new();
    s.prepare_soil_trial().unwrap();
    s
}
fn same(a: &RetainedSoil, b: &RetainedSoil, label: &str) {
    assert!(
        serde_json::to_string(a.machine()).unwrap() == serde_json::to_string(b.machine()).unwrap(),
        "{label}: machine/controller changed"
    );
    assert!(
        serde_json::to_string(a.soil()).unwrap() == serde_json::to_string(b.soil()).unwrap(),
        "{label}: canonical soil changed"
    );
    let a = a.model.as_ref().unwrap();
    let b = b.model.as_ref().unwrap();
    for (h, body) in a.world.bodies.iter() {
        let other = &b.world.bodies[h];
        for (x, y) in body
            .translation()
            .to_array()
            .into_iter()
            .chain(body.rotation().to_array())
            .chain(body.linvel().to_array())
            .chain(body.angvel().to_array())
            .zip(
                other
                    .translation()
                    .to_array()
                    .into_iter()
                    .chain(other.rotation().to_array())
                    .chain(other.linvel().to_array())
                    .chain(other.angvel().to_array()),
            )
        {
            assert_eq!(x.to_bits(), y.to_bits(), "{label}: physical body {h:?}");
        }
    }
}
#[test]
fn complete_retained_bank_owner_clones_and_restores_through_real_contact() {
    let mut s = bank();
    let obstacles = s.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&s.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let mut a = RetainedSoil::new(s.machine.clone(), s.active_soil.take().unwrap(), &env).unwrap();
    let mut b = RetainedSoil::decode(&a.encode().unwrap(), &env).unwrap();
    let mut c = a.clone();
    let initial = serde_json::to_string(&a.soil().material()).unwrap();
    for i in 0..64 {
        let input = MachineInput {
            curl: 0.1 * (i as f32 * 0.025).sin(),
            ..Default::default()
        };
        a.step(input, 1. / 240., &env).unwrap();
        b.step(input, 1. / 240., &env).unwrap();
        c.step(input, 1. / 240., &env).unwrap();
        if i % 8 == 7 {
            same(&a, &b, "restored");
            same(&a, &c, "cloned");
        }
        if i == 31 {
            let bytes = a.encode().unwrap();
            println!("retained full bank boundary bytes={}", bytes.len());
            b = RetainedSoil::decode(&bytes, &env).unwrap();
        }
    }
    assert_eq!(a.soil().clods.len(), 1040);
    assert_eq!(
        serde_json::to_string(&a.soil().material()).unwrap(),
        initial
    );
    let impulses = a
        .model
        .as_ref()
        .unwrap()
        .world
        .narrow_phase
        .contact_pairs()
        .flat_map(|p| p.solver_manifolds())
        .flat_map(|m| &m.points)
        .filter(|p| p.data.warmstart_impulse > 0. && p.data.warmstart_tangent_impulse.norm() > 0.)
        .count();
    assert!(impulses > 0);
    println!("real retained normal/friction histories={impulses}");
}
#[test]
fn refused_inputs_geometry_and_corrupt_owners_cannot_reset_contact_history() {
    let mut s = bank();
    let obstacles = s.mechanical_obstacles();
    let env = Environment {
        gravity: -Vec3::Y * 9.81,
        site: Some(&s.site),
        obstacles: &obstacles,
        loads: &[],
    };
    let mut a = RetainedSoil::new(s.machine.clone(), s.active_soil.take().unwrap(), &env).unwrap();
    a.step(MachineInput::default(), 1. / 240., &env).unwrap();
    let before = a.encode().unwrap();
    for (input, dt) in [
        (
            MachineInput {
                curl: f32::NAN,
                ..Default::default()
            },
            1. / 240.,
        ),
        (MachineInput::default(), 0.),
    ] {
        assert!(a.step(input, dt, &env).is_err());
        assert!(before == a.encode().unwrap());
        assert!(!a.failed());
    }
    let obstacle = [Obstacle {
        center: Vec3::new(20., 2., 20.),
        half_extents: Vec3::ONE,
        rotation: Quat::IDENTITY,
    }];
    let changed = Environment {
        obstacles: &obstacle,
        ..env
    };
    assert!(a
        .step(MachineInput::default(), 1. / 240., &changed)
        .is_err());
    assert!(before == a.encode().unwrap());
    assert!(RetainedSoil::decode(&before, &changed).is_err());
    for corruption in 0..3 {
        let mut saved: Saved = codec().deserialize(&before[HEADER..]).unwrap();
        match corruption {
            0 => saved.bodies.swap(0, 4),
            1 => {
                let mut machine: Excavator = serde_json::from_str(&saved.machine_json).unwrap();
                machine.position.x += 1.;
                saved.machine_json = serde_json::to_string(&machine).unwrap();
            }
            _ => {
                let mut world = ContactHistory::decode(&saved.numerical_world).unwrap();
                world
                    .world_mut()
                    .integration_parameters
                    .num_solver_iterations = 1;
                saved.numerical_world = world.encode().unwrap();
            }
        }
        let mut bytes = before[..HEADER].to_vec();
        bytes.extend(codec().serialize(&saved).unwrap());
        assert!(
            RetainedSoil::decode(&bytes, &env).is_err(),
            "corrupt owner/configuration {corruption} admitted"
        );
    }
    let mut trailing = before.clone();
    trailing.push(0);
    assert!(RetainedSoil::decode(&trailing, &env).is_err());
    assert!(RetainedSoil::decode(&before[..before.len() - 1], &env).is_err());
}
fn free_machine(swing: f32) -> Excavator {
    let mut m = Excavator::new(&Site::new());
    m.position = Vec3::new(16., 10., 16.);
    m.boom = 1.1;
    m.stick = -1.5;
    m.curl = -0.2;
    m.swing = swing;
    m.engine_on = false;
    m.enable_articulation().unwrap();
    m
}
fn prism() -> [DVec3; 6] {
    [
        DVec3::ZERO,
        DVec3::new(0.2, 0., 0.),
        DVec3::new(0., 0., 0.2),
        DVec3::new(0., 0.2, 0.),
        DVec3::new(0.2, 0.2, 0.),
        DVec3::new(0., 0.2, 0.2),
    ]
}
fn clod() -> Clod {
    Clod::prism(
        Material {
            sand_m3: 0.004,
            ..Default::default()
        },
        prism(),
    )
    .unwrap()
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
fn retained_soil_impact_closes_momentum_and_dissipates_at_two_timesteps() {
    for hz in [240, 480] {
        let mut m = free_machine(0.);
        m.tracks.mechanical_hz = hz;
        let q = m.bucket_rotation();
        let floor = m.arm().wrist + q * Vec3::new(0., -0.65, 0.4);
        let mut clod = clod();
        clod.center = floor + q * (clod.center + Vec3::new(-0.1, 0.14, -0.1));
        clod.rotation = q;
        clod.velocity = q * (-Vec3::Y);
        let soil = ActiveSoil::from_clods(vec![clod]).unwrap();
        let before = totals(&m, &soil);
        let env = Environment::free(Vec3::ZERO);
        let mut a = RetainedSoil::new(m, soil, &env).unwrap();
        let mut contacts = 0;
        let mut residual = 0f64;
        for _ in 0..hz / 2 {
            a.step(MachineInput::default(), 1. / hz as f32, &env)
                .unwrap();
            assert!(a.soil().feedback.closed_free_system);
            contacts += a.soil().feedback.bucket_contacts;
            residual = residual.max(a.soil().feedback.coupled_passivity_residual_j);
        }
        let after = totals(a.machine(), a.soil());
        println!(
            "retained impact {hz}Hz: contacts={contacts},dP={},dL={},dE={},max_residual={residual}",
            (after.0 - before.0).length(),
            (after.1 - before.1).length(),
            after.2 - before.2
        );
        assert!(contacts > 0);
        assert!((after.0 - before.0).length() < 0.003);
        assert!((after.1 - before.1).length() < 0.03);
        assert!(after.2 <= before.2 + 0.01);
        assert!(residual < 0.01);
    }
}
#[test]
fn retained_swing_chart_crosses_pi_without_losing_saved_contact_state_or_momentum() {
    let mut m = free_machine(3.13);
    m.engine_on = true;
    let mut clod = clod();
    clod.center = Vec3::new(25., 12., 25.);
    let soil = ActiveSoil::from_clods(vec![clod]).unwrap();
    let before = totals(&m, &soil);
    let initial_com = (m
        .articulation
        .as_ref()
        .unwrap()
        .feedback
        .center_of_mass
        .as_dvec3()
        * m.articulation.as_ref().unwrap().feedback.mass_kg
        + soil.totals(Vec3::ZERO).first_kg_m)
        / (m.articulation.as_ref().unwrap().feedback.mass_kg + soil.totals(Vec3::ZERO).mass_kg);
    let env = Environment::free(Vec3::ZERO);
    let mut a = RetainedSoil::new(m, soil, &env).unwrap();
    let mut previous = 3.13;
    let mut wraps = 0;
    let mut restored = None;
    for tick in 0..120 {
        let input = MachineInput {
            swing: 1.,
            ..Default::default()
        };
        a.step(input, 1. / 240., &env).unwrap();
        if let Some(b) = restored.as_mut() {
            RetainedSoil::step(b, input, 1. / 240., &env).unwrap();
            if tick % 16 == 15 {
                same(&a, b, "swing restoration");
            }
        }
        let coordinate = a.machine().articulation.as_ref().unwrap().coordinates[0];
        if previous > 3. && coordinate < -3. {
            wraps += 1;
            restored = Some(RetainedSoil::decode(&a.encode().unwrap(), &env).unwrap());
        }
        previous = coordinate;
    }
    let after = totals(a.machine(), a.soil());
    println!(
        "retained swing chart: wraps={wraps},dP={},dL={}",
        (after.0 - before.0).length(),
        (after.1 - before.1).length()
    );
    assert!(wraps > 0);
    // Match the existing half-second powered-actuator budget, distinct from
    // the stricter unpowered, small-clod impact control above. Origin angular
    // momentum includes COM cross linear drift; use the intrinsic quantity.
    let feedback = &a.machine().articulation.as_ref().unwrap().feedback;
    let st = a.soil().totals(Vec3::ZERO);
    let final_com = (feedback.center_of_mass.as_dvec3() * feedback.mass_kg + st.first_kg_m)
        / (feedback.mass_kg + st.mass_kg);
    let intrinsic = after.1 - final_com.cross(after.0);
    println!(
        "powered wrap: centroid drift={}m, intrinsic momentum={}",
        (final_com - initial_com).length(),
        intrinsic.length()
    );
    assert!((final_com - initial_com).length() < 0.002);
    assert!((after.0 - before.0).length() < 1.);
    assert!(intrinsic.length() < 5.);
}

use glam::{DVec3, Mat3, Quat};
use mining_sim::{
    loose_gold::{LooseGold, MAX_PARTICLES},
    site::{CELL, WIDTH},
    sluice::Grain,
    GrainKind, Material, Site, Vec3,
};

fn flat(depth: f32, flow: f32) -> Site {
    let mut s = Site::new();
    s.world.bedrock_elevation.fill(0.);
    s.world.paydirt_thickness.fill(0.);
    s.world.overburden_thickness.fill(0.);
    s.world.terrain_sediment.fill(0.);
    s.world.gravel_thickness.fill(0.);
    s.world.water_surface.fill(depth);
    s.hydraulic.velocity_x = vec![flow; WIDTH * WIDTH];
    s.hydraulic.velocity_z = vec![0.; WIDTH * WIDTH];
    s
}
fn grain(position: Vec3, diameter: f32) -> Grain {
    Grain {
        volume_resolved: true,
        kind: GrainKind::Gold,
        position,
        velocity: Vec3::ZERO,
        material: Material {
            gold_g: 0.01,
            ..Material::default()
        },
        resting: false,
        diameter_m: diameter,
        affine: Mat3::IDENTITY * 0.2,
        rotation: Quat::from_rotation_z(0.3),
        angular_velocity: Vec3::ZERO,
    }
}
fn run(site: &Site, grain: &Grain, dt: f32, n: usize) -> LooseGold {
    let mut owner = LooseGold::default();
    owner.admit(grain).unwrap();
    for _ in 0..n {
        owner.step(site, dt).unwrap();
    }
    owner
}

#[test]
fn dry_ballistics_and_constant_flow_match_independent_analytic_solution() {
    let mut g = grain(Vec3::new(10., 10., 10.), 0.001);
    g.velocity = Vec3::new(1.25, 0.5, -0.4);
    let dry = run(&flat(0., 0.), &g, 1. / 60., 12);
    let t = (1. / 60.) as f32 as f64 * 12.;
    let expected_v = g.velocity.as_dvec3() - DVec3::Y * (9.81 * t);
    let expected_p =
        g.position.as_dvec3() + g.velocity.as_dvec3() * t - DVec3::Y * (0.5 * 9.81 * t * t);
    assert!((dry.particles[0].velocity - expected_v).length() < 1e-12);
    assert!((dry.particles[0].position - expected_p).length() < 1e-11);
    let wet = run(&flat(20., 0.7), &g, 1. / 60., 12);
    let d = g.diameter_m as f64;
    let relative: f64 = 19.3 - 1.;
    let ws = relative * 9.81 * d * d / (18e-6 + (0.75 * relative * 9.81 * d * d * d).sqrt());
    let k = 9.81 * (1. - 1. / 19.3) / ws;
    let eq = DVec3::new(0.7_f32 as f64, -ws, 0.);
    let expected_v = eq + (g.velocity.as_dvec3() - eq) * (-k * t).exp();
    let expected_p =
        g.position.as_dvec3() + eq * t + (g.velocity.as_dvec3() - eq) * (-(-k * t).exp_m1() / k);
    assert!((wet.particles[0].velocity - expected_v).length() < 1e-11);
    assert!((wet.particles[0].position - expected_p).length() < 1e-10);
}

#[test]
fn settling_uses_physical_diameter_independent_of_parcel_weight() {
    let s = flat(20., 0.);
    for diameter in [0.00005, 0.0001, 0.0005, 0.001] {
        let g = grain(Vec3::new(10., 10., 10.), diameter);
        let a = run(&s, &g, 1. / 60., 60);
        let mut weighted = g.clone();
        weighted.material.gold_g *= 1000.;
        let b = run(&s, &weighted, 1. / 60., 60);
        assert_eq!(a.particles[0].position, b.particles[0].position);
        let d = diameter as f64;
        let ws =
            (19.3 - 1.) * 9.81 * d * d / (18e-6 + (0.75 * (19.3 - 1.) * 9.81 * d * d * d).sqrt());
        let k = 9.81 * (1. - 1. / 19.3) / ws;
        let t = (1. / 60.) as f32 as f64 * 60.;
        assert!((a.particles[0].velocity.y + ws * (1. - (-k * t).exp())).abs() < 1e-10);
    }
}

#[test]
fn flat_contact_holds_in_still_and_low_flow_but_strong_flow_moves_gold() {
    let g = grain(Vec3::new(10., 0.0005, 10.), 0.001);
    let still = run(&flat(0.2, 0.), &g, 1. / 120., 240);
    let low = run(&flat(0.2, 0.001), &g, 1. / 120., 240);
    let high = run(&flat(0.2, 2.), &g, 1. / 120., 240);
    assert!(still.particles[0].supported);
    assert!((still.particles[0].position.x - 10.).abs() < 1e-12);
    assert!((low.particles[0].position.x - 10.).abs() < 1e-10);
    assert!(high.particles[0].position.x > 10.1);
    assert!(high.particles[0].velocity.x > 0.);
    assert_eq!(high.particles[0].grain.diameter_m, g.diameter_m);
}

#[test]
fn deposition_buries_without_surface_teleport_and_erosion_uncovers() {
    let mut s = flat(0.3, 0.);
    let g = grain(Vec3::new(10., 0.0005, 10.), 0.001);
    let mut owner = run(&s, &g, 1. / 60., 1);
    let position = owner.particles[0].position;
    s.world.terrain_sediment.fill(0.05);
    owner.step(&s, 1. / 60.).unwrap();
    assert!(owner.particles[0].buried);
    assert_eq!(owner.particles[0].position, position);
    s.world.terrain_sediment.fill(0.);
    s.world.bedrock_elevation.fill(-0.02);
    owner.step(&s, 1. / 60.).unwrap();
    assert!(!owner.particles[0].buried);
    assert!(owner.particles[0].position.y < position.y);
    assert!(owner.particles[0].mobile);
}

#[test]
fn repeated_submicron_deposition_cannot_carry_gold_upward() {
    let mut s = flat(0.3, 0.);
    let g = grain(Vec3::new(10., 0.0005, 10.), 0.001);
    let mut owner = run(&s, &g, 1. / 60., 1);
    let at = owner.particles[0].position;
    for n in 1..=100 {
        s.world.terrain_sediment.fill(n as f32 * 1e-8);
        owner.step(&s, 1. / 60.).unwrap();
        assert_eq!(owner.particles[0].position, at);
    }
    assert!(owner.particles[0].buried);
}

#[test]
fn shallow_cut_excludes_deep_distant_and_suspended_gold_and_closes_exactly() {
    let mut s = flat(0., 0.);
    s.world.terrain_sediment.fill(0.2);
    let mut owner = LooseGold::default();
    for position in [
        Vec3::new(10., 0.2005, 10.),
        Vec3::new(10., 0.1, 10.),
        Vec3::new(12., 0.2005, 10.),
        Vec3::new(10., 0.4, 10.),
    ] {
        owner.admit(&grain(position, 0.001)).unwrap();
    }
    owner.step(&s, 1. / 120.).unwrap();
    let i = 40 * WIDTH + 40;
    assert!((owner.preview_cut(&s, &[(i, 0.01)]) - 0.01).abs() < 1e-12);
    assert_eq!(owner.take_cut(&s, &[(i, 0.01)]), 0.01);
    assert_eq!(owner.particles.len(), 3);
    assert!(owner.valid());
    assert!((owner.material().gold_g + owner.excavated_g - 0.04).abs() < 1e-15);
}

#[test]
fn mixed_intake_takes_only_local_mobile_mass_and_preserves_remainder_identity() {
    let s = flat(0.3, 0.);
    let mut owner = LooseGold::default();
    for p in [
        Vec3::new(10., 0.1, 10.),
        Vec3::new(10., 0.0005, 10.),
        Vec3::new(12., 0.1, 10.),
    ] {
        owner.admit(&grain(p, 0.001)).unwrap();
    }
    owner.step(&s, 1. / 120.).unwrap();
    let before = owner.particles[0].clone();
    let cell = 40 * WIDTH + 40;
    assert_eq!(owner.withdraw_mobile(cell, 0.25), 0.0025);
    assert_eq!(owner.particles[0].id, before.id);
    assert_eq!(owner.particles[0].grain.diameter_m, before.grain.diameter_m);
    assert_eq!(owner.particles[1].grain.material.gold_g, 0.01);
    assert!(owner.valid());
    assert_eq!(owner.withdraw_mobile(cell, 1.), 0.0075);
    assert_eq!(owner.particles.len(), 2);
    assert!(owner.valid());
}

#[test]
fn admission_capacity_and_numerical_refusal_are_transactional() {
    let g = grain(Vec3::new(10., 1., 10.), 0.001);
    let mut owner = LooseGold::default();
    let mut mixed = g.clone();
    mixed.material.sand_m3 = 1e-9;
    assert!(owner.admit(&mixed).is_err());
    assert!(owner.try_reserve(MAX_PARTICLES + 1).is_err());
    assert_eq!(owner.next_id, 0);
    let mut fast = g.clone();
    fast.velocity.x = 1e20;
    owner.admit(&fast).unwrap();
    let before = serde_json::to_value(&owner).unwrap();
    assert!(owner.step(&flat(0., 0.), 0.25).is_err());
    assert_eq!(serde_json::to_value(&owner).unwrap(), before);
}

#[test]
fn saved_complete_identity_and_delayed_first_impact_continue_exactly() {
    let s = flat(20., 0.7);
    let mut g = grain(Vec3::new(10., 10., 10.), 0.001);
    g.angular_velocity = Vec3::new(0.1, -0.2, 0.3);
    let mut owner = LooseGold::default();
    let dt = 1. / 60_f32;
    owner.admit_at(&g, dt as f64 * 0.75).unwrap();
    let mut restored: LooseGold =
        serde_json::from_str(&serde_json::to_string(&owner).unwrap()).unwrap();
    assert!(restored.valid());
    for _ in 0..60 {
        owner.step(&s, dt).unwrap();
        restored.step(&s, dt).unwrap();
    }
    assert_eq!(
        serde_json::to_value(&owner).unwrap(),
        serde_json::to_value(&restored).unwrap()
    );
    assert_eq!(owner.particles[0].grain.affine, g.affine);
    assert_eq!(
        owner.particles[0].grain.angular_velocity,
        g.angular_velocity
    );
    assert_eq!(owner.particles[0].delay_s, 0.);
    let mut delayed = LooseGold::default();
    delayed.admit_at(&g, dt as f64 * 0.75).unwrap();
    delayed.step(&s, dt).unwrap();
    let direct = run(&s, &g, dt * 0.25, 1);
    assert!((delayed.particles[0].position - direct.particles[0].position).length() < 1e-12);
    let mut corrupt = restored.clone();
    corrupt.particles[0].grain.position.x += 0.01;
    assert!(!corrupt.valid());
    let mut corrupt = restored.clone();
    corrupt.particles.push(corrupt.particles[0].clone());
    assert!(!corrupt.valid());
    let mut corrupt = restored.clone();
    corrupt.particles[0].grain.resting = !corrupt.particles[0].grain.resting;
    assert!(!corrupt.valid());
    let mut corrupt = restored.clone();
    corrupt.particles[0].supported = true;
    corrupt.particles[0].buried = true;
    corrupt.particles[0].mobile = false;
    corrupt.particles[0].grain.resting = true;
    assert!(!corrupt.valid());
    let mut corrupt = restored.clone();
    corrupt.particles[0].delay_s = 0.251;
    assert!(!corrupt.valid());
    assert!(owner.admit_at(&g, 0.251).is_err());
    let mut corrupt = restored;
    corrupt.model_version = 99;
    assert!(!corrupt.valid());
}

#[test]
fn timestep_refinement_preserves_free_motion_and_converges_contact() {
    let s = flat(0.4, 0.8);
    let g = grain(Vec3::new(10., 0.1, 10.), 0.001);
    let states: [LooseGold; 4] = [60, 120, 480, 960].map(|hz| run(&s, &g, 1. / hz as f32, hz));
    for state in &states {
        assert!(state.valid());
        assert!(state.particles[0].supported);
    }
    let a = states[0].particles[0].position;
    let b = states[1].particles[0].position;
    let c = states[2].particles[0].position;
    let reference = states[3].particles[0].position;
    println!(
        "GOLD_CONTACT_REFINEMENT {}",
        serde_json::json!({"hz":[60,120,480,960],"position_m":[a.to_array(),b.to_array(),c.to_array(),reference.to_array()],"error_to_960_m":[(a-reference).length(),(b-reference).length(),(c-reference).length()],"physical_diameter_m":g.diameter_m,"point_parcel_model_only":true})
    );
    assert!(
        (b - reference).length() < (a - reference).length() + 1e-8,
        "coarse={a:?}, medium={b:?}, reference={reference:?}"
    );
    assert!((b - reference).length() < 0.004);
    assert!((c - reference).length() < 0.0005);
}

#[test]
fn bare_bedrock_surface_gold_can_be_picked_up_without_inventing_paydirt() {
    let s = flat(0., 0.);
    let mut owner = LooseGold::default();
    for p in [
        Vec3::new(10., 0.0005, 10.),
        Vec3::new(10., -0.01, 10.),
        Vec3::new(12., 0.0005, 10.),
        Vec3::new(10., 0.1, 10.),
    ] {
        owner.admit(&grain(p, 0.001)).unwrap();
    }
    owner.step(&s, 1. / 120.).unwrap();
    assert_eq!(owner.take_cut(&s, &[(40 * WIDTH + 40, 0.01)]), 0.01);
    assert_eq!(owner.particles.len(), 3);
    assert!(owner.valid());
}

#[test]
fn actual_triangle_support_and_passive_slope_do_not_create_energy() {
    let mut s = flat(0., 0.);
    // Gentle plane: exact normal (-.1,1,0), below the declared static slope.
    for z in 0..WIDTH {
        for x in 0..WIDTH {
            s.world.bedrock_elevation[z * WIDTH + x] = x as f32 * CELL * 0.1;
        }
    }
    let n = DVec3::new(-0.1, 1., 0.).normalize();
    let h = mining_sim::soil::substrate_height(&s, 10., 10.) as f64;
    let g = grain(Vec3::new(10., (h + 0.0005 / n.y) as f32, 10.), 0.001);
    let mut owner = run(&s, &g, 1. / 120., 120);
    assert!((owner.particles[0].position.x - 10.).abs() < 1e-6);
    assert!(owner.particles[0].velocity.length() < 1e-8);
    // Frictionless sliding must draw its kinetic energy from gravity.
    owner.friction_mu = 0.;
    let start = owner.particles[0].position;
    let initial = 9.81 * start.y;
    for _ in 0..120 {
        owner.step(&s, 1. / 120.).unwrap();
    }
    let p = &owner.particles[0];
    assert!(p.position.x < start.x);
    let energy = 9.81 * p.position.y + 0.5 * p.velocity.length_squared();
    assert!(
        energy <= initial + 1e-5,
        "energy created: {energy} > {initial}"
    );
    assert!(
        (p.position.y
            - mining_sim::soil::substrate_height(&s, p.position.x as f32, p.position.z as f32)
                as f64
            - 0.0005 / n.y)
            .abs()
            < 1e-6
    );
}

#[test]
fn independent_boundary_export_is_a_finite_counted_sink() {
    let s = flat(0., 0.);
    let mut g = grain(Vec3::new(31.999, 10., 10.), 0.001);
    g.velocity.x = 1.;
    let mut owner = LooseGold::default();
    owner.admit(&g).unwrap();
    assert_eq!(owner.step(&s, 1. / 60.).unwrap(), 0.01);
    assert!(owner.particles.is_empty());
    assert_eq!(owner.exported_g, 0.01);
    assert!(owner.valid());
    assert_eq!(owner.step(&s, 1. / 60.).unwrap(), 0.);
}

#[test]
fn remainder_admission_advances_only_the_new_packet_and_refuses_atomically() {
    let site = flat(20., 0.7);
    let old = grain(Vec3::new(10., 10., 10.), 0.001);
    let mut owner = LooseGold::default();
    owner.admit(&old).unwrap();
    let existing = serde_json::to_value(&owner.particles[0]).unwrap();
    let mut incoming = grain(Vec3::new(12., 6., 12.), 0.0001);
    incoming.velocity = Vec3::new(0.2, -0.1, 0.3);
    let dt = 1. / 240_f32;
    owner.admit_and_advance(&incoming, &site, dt).unwrap();
    assert_eq!(serde_json::to_value(&owner.particles[0]).unwrap(), existing);
    let reference = run(&site, &incoming, dt, 1);
    assert_eq!(owner.particles[1].position, reference.particles[0].position);
    assert_eq!(owner.particles[1].velocity, reference.particles[0].velocity);
    assert_eq!(owner.particles[1].grain.diameter_m, incoming.diameter_m);
    assert_eq!(owner.particles[1].grain.material, incoming.material);
    assert!(owner.valid());
    let before = serde_json::to_value(&owner).unwrap();
    let mut fast = incoming.clone();
    fast.velocity.x = 1e20;
    assert!(owner.admit_and_advance(&fast, &flat(0., 0.), 0.25).is_err());
    assert_eq!(serde_json::to_value(&owner).unwrap(), before);
    assert!(owner.admit_and_advance(&incoming, &site, -1.).is_err());
    assert_eq!(serde_json::to_value(&owner).unwrap(), before);
    let mut exiting = grain(Vec3::new(31.999, 10., 10.), 0.001);
    exiting.velocity.x = 1.;
    owner
        .admit_and_advance(&exiting, &flat(0., 0.), 1. / 60.)
        .unwrap();
    assert_eq!(owner.particles.len(), 2);
    assert_eq!(owner.exported_g, exiting.material.gold_g);
    assert_eq!(owner.admitted_g, 0.03);
    assert_eq!(owner.next_id, 3);
    assert_eq!(serde_json::to_value(&owner.particles[0]).unwrap(), existing);
    assert!(owner.valid());
}

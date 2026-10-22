use mining_sim::{machine::MachineInput, Excavator, Material, Session, Site, Vec3};

fn flat_site() -> Site {
    let mut s = Site::new();
    for i in 0..s.gold_g.len() {
        s.world.bedrock_elevation[i] = 0.05;
        s.world.overburden_thickness[i] = 0.10;
        s.world.paydirt_thickness[i] = 0.20;
        s.world.gravel_thickness[i] = 0.;
        s.world.terrain_sediment[i] = 0.;
        s.world.water_surface[i] = 0.45;
        s.gold_g[i] = 0.20 * 0.0625 * 10.;
    }
    s
}
fn solid(s: &Site) -> f64 {
    [
        &s.world.overburden_thickness,
        &s.world.paydirt_thickness,
        &s.world.gravel_thickness,
        &s.world.terrain_sediment,
    ]
    .iter()
    .flat_map(|v| v.iter())
    .map(|v| *v as f64 * 0.0625)
    .sum()
}
fn lower(site: &mut Site, m: &mut Excavator, dt: f32, seconds: f32, cut: bool) {
    for _ in 0..(seconds / dt).round() as usize {
        m.step(
            MachineInput {
                cut,
                boom: -0.22,
                ..Default::default()
            },
            site,
            dt,
        );
    }
}

#[test]
fn holding_cut_with_a_stationary_bucket_does_not_mine() {
    let mut site = Site::new();
    let mut machine = Excavator::new(&site);
    machine.boom = 0.32;
    machine.bucket = Material::default();
    let gold = site.remaining_gold();
    for _ in 0..120 {
        machine.step(
            MachineInput {
                cut: true,
                ..Default::default()
            },
            &mut site,
            1. / 60.,
        );
    }
    assert!(
        machine.bucket.is_empty(),
        "stationary bucket mined {:?}",
        machine.bucket
    );
    assert_eq!(site.remaining_gold(), gold);
}

#[test]
fn fresh_machine_floor_and_teeth_start_clear_of_the_actual_bank() {
    let site = Site::new();
    let machine = Excavator::new(&site);
    let pose = machine.arm();
    println!("clear spawn boom: {}", machine.boom);
    let right = Vec3::new(pose.forward.z, 0., -pose.forward.x);
    for along in [0., 0.5, 1.] {
        for side in [-0.5, 0., 0.5] {
            let p = pose.heel.lerp(pose.lip, along)
                + right * (mining_sim::earthworks::BUCKET_WIDTH_M * side);
            assert!(p.y >= site.height(p.x, p.z) + 0.05);
        }
    }
}

#[test]
fn swept_cut_conserves_layers_gold_and_water_and_stops_the_tooth_at_rock() {
    let mut s = flat_site();
    let mut m = Excavator::new(&s);
    let (material, water, gold) = (solid(&s), s.water_volume_m3(), s.remaining_gold());
    lower(&mut s, &mut m, 1. / 60., 4., true);
    assert!(m.bucket.volume() > 0.02 && m.bucket.volume() <= 0.28000001);
    assert!((solid(&s) + m.bucket.volume() - material).abs() < 1e-6);
    assert!((s.remaining_gold() + m.bucket.gold_g - gold).abs() < 1e-7);
    assert!((s.water_volume_m3() - water).abs() < 1e-6);
    assert!(
        m.arm().lip.y >= 0.0498,
        "tooth passed through rock: {:?}",
        m.arm()
    );
    assert!(m.arm().heel.y >= 0.0498);
    assert!(m.digging.limited);
}

#[test]
fn passive_bucket_cannot_be_driven_through_the_ground() {
    let mut s = flat_site();
    let mut m = Excavator::new(&s);
    lower(&mut s, &mut m, 1. / 60., 5., false);
    assert!(m.bucket.is_empty());
    assert!(
        m.arm().lip.y >= 0.3498,
        "passive tool penetrated ground: {:?}",
        m.arm()
    );
    assert!(m.digging.limited);
    // It must still be possible to pull away from the contact.
    for _ in 0..30 {
        m.step(
            MachineInput {
                boom: 1.,
                ..Default::default()
            },
            &mut s,
            1. / 60.,
        );
    }
    assert!(m.arm().lip.y > 0.8);
}

#[test]
fn force_power_and_remaining_capacity_limit_the_accepted_stroke() {
    let run = |force: f32, power: f32, fill: f64| {
        let mut s = flat_site();
        let mut m = Excavator::new(&s);
        m.hydraulics.crowd_force_n = force;
        m.hydraulics.tool_power_w = power;
        m.bucket.gravel_m3 = fill;
        let mut limited = 0;
        for _ in 0..150 {
            m.step(
                MachineInput {
                    cut: true,
                    boom: -0.22,
                    ..Default::default()
                },
                &mut s,
                1. / 60.,
            );
            assert!(m.digging.soil_force_n <= force + 0.1);
            assert!(m.digging.tool_power_w <= power + 0.1);
            assert!(m.bucket.volume() <= 0.28000001);
            limited += usize::from(m.digging.limited);
        }
        (m.bucket.volume() - fill, limited)
    };
    let (normal, _) = run(41_300., 30_000., 0.);
    let (weak, stalls) = run(500., 30_000., 0.);
    let (slow, restrictions) = run(41_300., 100., 0.);
    assert!(
        normal > 0.02 && weak < normal * 0.5 && stalls > 0,
        "normal={normal}, weak={weak}"
    );
    assert!(
        slow < normal * 0.8 && restrictions > 0,
        "normal={normal}, slow={slow}"
    );
    assert!(run(41_300., 30_000., 0.27995).0 <= 0.00005001);
}

#[test]
fn repeated_stroke_converges_with_timestep_instead_of_mining_per_frame() {
    let mut volumes = Vec::new();
    for dt in [1. / 30., 1. / 60., 1. / 120.] {
        let mut s = flat_site();
        let mut m = Excavator::new(&s);
        lower(&mut s, &mut m, dt, 4., true);
        volumes.push(m.bucket.volume());
    }
    let reference = volumes[2];
    println!("swept cut volumes at 30/60/120 Hz: {volumes:?}");
    assert!(
        volumes.iter().all(|v| (v / reference - 1.).abs() < 0.001),
        "sweep convergence: {volumes:?}"
    );
}

#[test]
fn gravel_is_excavatable_in_the_same_layer_order_as_the_gpu_and_preserves_trace_gold() {
    let mut s = flat_site();
    let p = Vec3::new(19., 0.35, 22.);
    let i = s.world.idx(76, 88);
    s.world.terrain_sediment[i] = 0.06;
    s.world.gravel_thickness[i] = 0.08;
    s.world.water_surface[i] = s.world.ground_height(76, 88) + 0.1;
    let before = s.remaining_gold();
    let water = s.water_volume_m3();
    let top = s.dig(p, 0.1, 0.08, 10.);
    assert!((top.sand_m3 - 0.06 * 0.0625).abs() < 1e-9);
    assert!((top.soil_m3 - 0.02 * 0.0625).abs() < 1e-9);
    assert_eq!(top.gravel_m3, 0.);
    assert_eq!(top.gold_g, 0.);
    assert!((s.world.overburden_thickness[i] - 0.08).abs() < 1e-7);
    let rest = s.dig(p, 0.1, 1., 10.);
    assert_eq!(s.world.gravel_thickness[i], 0.);
    assert!((s.remaining_gold() + rest.gold_g - before).abs() < 1e-8);
    assert!((s.water_volume_m3() - water).abs() < 1e-7);
}

#[test]
fn saved_stroke_resumes_and_older_saves_receive_finite_machine_defaults() {
    let mut session = Session::new();
    lower(&mut session.site, &mut session.machine, 1. / 60., 1.5, true);
    let mut restored = Session::load_json(&session.save_json().unwrap()).unwrap();
    lower(&mut session.site, &mut session.machine, 1. / 60., 1., true);
    lower(
        &mut restored.site,
        &mut restored.machine,
        1. / 60.,
        1.,
        true,
    );
    assert_eq!(session.machine.bucket, restored.machine.bucket);
    assert_eq!(session.machine.arm().lip, restored.machine.arm().lip);
    let mut old: serde_json::Value = serde_json::from_str(&session.save_json().unwrap()).unwrap();
    old["machine"].as_object_mut().unwrap().remove("hydraulics");
    old["machine"].as_object_mut().unwrap().remove("digging");
    let restored = Session::load_json(&old.to_string()).unwrap();
    assert!(restored.machine.valid());
    assert_eq!(restored.machine.hydraulics.crowd_force_n, 41_300.);
}

#[test]
fn blade_pressure_has_the_independent_passive_earth_pressure_limit() {
    use mining_sim::earthworks::SoilStrength;
    // phi=0 gives Kp=1: integrate rho*g*z + 2*c over depth and width.
    let soil = SoilStrength {
        density_kg_m3: 1800.,
        cohesion_pa: 2000.,
        friction_deg: 0.,
    };
    let expected = 0.8 * (0.5 * 1800. * 9.81 * 0.3 * 0.3 + 2. * 2000. * 0.3);
    assert!((soil.resistance_n(0.3, 0.8) - expected).abs() < 0.01);
    assert_eq!(soil.resistance_n(0., 0.8), 0.);
    assert!(
        SoilStrength {
            friction_deg: 35.,
            ..soil
        }
        .resistance_n(0.3, 0.8)
            > expected
    );
}

#[test]
fn suspended_inventory_uses_water_volume_and_survives_clear_water_dilution() {
    let mut s = Session::new();
    let initial = s.accounted_bulk_material_m3();
    let point = Vec3::new(19., 0.35, 17.);
    let i = s.site.world.idx(76, 68);
    let removed = s.site.dig(point, 0.1, 0.06, 10.);
    assert_eq!(removed.gravel_m3, 0.);
    s.site.world.water_surface[i] = s.site.world.ground_height(76, 68) + 0.2;
    s.site.hydraulic.suspended_overburden = vec![0.; 128 * 128];
    s.site.hydraulic.suspended_overburden[i] =
        (removed.soil_m3 / (0.0625 * s.site.world.water_depth(76, 68) as f64)) as f32;
    assert!((s.accounted_bulk_material_m3() - initial).abs() < 1e-8);
    s.site.world.water_surface[i] += 0.2;
    s.site.hydraulic.suspended_overburden[i] *= 0.5;
    assert!((s.accounted_bulk_material_m3() - initial).abs() < 1e-8);
}

#[test]
fn rejected_chassis_travel_cannot_mine_at_the_rejected_tool_position() {
    use mining_sim::{
        session::Controls,
        sluice::{HOPPER_X, HOPPER_Z},
        ControlMode,
    };
    let mut moving = Session::new();
    moving.player.mode = ControlMode::Excavator;
    moving.machine.position = Vec3::new(
        HOPPER_X + 2.22,
        moving.site.height(HOPPER_X + 2.22, HOPPER_Z),
        HOPPER_Z,
    );
    moving.machine.heading = -std::f32::consts::FRAC_PI_2;
    moving.machine.boom = 0.32;
    let mut stationary = moving.clone();
    let mut controls = Controls {
        machine: MachineInput {
            drive: 1.,
            cut: true,
            boom: -0.2,
            ..Default::default()
        },
        ..Default::default()
    };
    moving.step(controls);
    controls.machine.drive = 0.;
    stationary.step(controls);
    assert_eq!(moving.machine.position, stationary.machine.position);
    assert_eq!(
        moving.site.world.paydirt_thickness,
        stationary.site.world.paydirt_thickness
    );
    assert_eq!(
        moving.site.world.overburden_thickness,
        stationary.site.world.overburden_thickness
    );
    assert_eq!(moving.machine.bucket, stationary.machine.bucket);
}

#[test]
fn operator_scoops_and_delivers_real_terrain_with_continuous_joint_controls() {
    use mining_sim::replay::OperatorReplay;
    let mut s = Session::new();
    let mut replay = OperatorReplay::default();
    let (initial, water) = (solid(&s.site), s.site.water_volume_m3());
    for _ in 0..15_000 {
        let controls = replay.controls(&mut s);
        s.step(controls);
        if replay.error.is_some() || replay.phase >= 15 {
            break;
        }
    }
    println!(
        "earthwork replay phase={} error={:?} mined={:.6} delivered={:.6} bucket={:.6} sim={:.3}",
        replay.phase,
        replay.error,
        s.site.excavated.volume(),
        s.delivered_m3,
        s.machine.bucket.volume(),
        s.sim_time
    );
    assert!(replay.error.is_none(), "{:?}", replay.error);
    assert_eq!(replay.phase, 15);
    assert!(
        s.delivered_m3 >= 0.4,
        "actual scoop/crowd/dump did not deliver the required finite load"
    );
    assert!(s.machine.bucket.is_empty() && s.falling.is_empty());
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-6);
    assert!((s.site.water_volume_m3() - water).abs() < 1e-6);
    assert!((solid(&s.site) + s.sluice.hopper.volume() - initial).abs() < 1e-6);
    mining_sim::replay::earthworks_receipt(&s, &replay).unwrap();
}

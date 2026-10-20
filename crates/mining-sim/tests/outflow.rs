use glam::Mat3;
use mining_sim::{
    outflow::{self, OutflowParticle},
    sluice::WaterParticle,
    Placement, Session, Vec3,
};

#[test]
fn lip_handoff_keeps_world_position_momentum_and_water_ledger() {
    let mut s = Session::new();
    let mut p = Placement::suggested(&s.site);
    p.yaw = 0.6;
    s.sluice.placement = Some(p);
    let position = Vec3::new(1.501, 0.06, 0.2);
    let velocity = Vec3::new(0.7, -0.2, 0.1);
    s.sluice.water.push(WaterParticle {
        position,
        velocity,
        affine: Mat3::ZERO,
    });
    let before = s.sluice.accounted_water_m3();
    outflow::release(&mut s.sluice);
    assert!(s.sluice.water.is_empty());
    let particle = &s.sluice.outflow[0];
    assert_eq!(
        particle.position,
        p.to_world(s.sluice.grid_to_local(position))
    );
    assert!(
        (particle.velocity - (p.forward() * 0.7 - Vec3::Y * 0.2 + p.right() * 0.1)).length() < 1e-6
    );
    assert_eq!(before, s.sluice.accounted_water_m3());
}
#[test]
fn flight_follows_exact_ballistic_arc_and_save_restore() {
    let mut s = Session::new();
    let particle = OutflowParticle {
        position: Vec3::new(14.6, 2., 19.5),
        velocity: Vec3::new(0.6, -0.1, 0.3),
    };
    s.sluice.outflow.push(particle.clone());
    let json = s.save_json().unwrap();
    let mut restored = Session::load_json(&json).unwrap();
    for _ in 0..12 {
        outflow::step(&mut s.sluice, &mut s.site, 1. / 120.);
        outflow::step(&mut restored.sluice, &mut restored.site, 1. / 120.);
    }
    let expected = particle.position + particle.velocity * 0.1 - Vec3::Y * (0.5 * 9.81 * 0.01);
    assert!((s.sluice.outflow[0].position - expected).length() < 2e-5);
    assert_eq!(
        s.sluice.outflow[0].position,
        restored.sluice.outflow[0].position
    );
    assert_eq!(s.sluice.water_returned_m3, 0.);
}
#[test]
fn impacts_deposit_volume_and_horizontal_momentum_at_landing() {
    let mut s = Session::new();
    let x = 14.625;
    let z = 20.625;
    let p = OutflowParticle {
        position: Vec3::new(x, s.site.height(x, z) + 0.002, z),
        velocity: Vec3::new(0.8, -2., -0.4),
    };
    s.sluice.outflow = vec![p; 4000];
    let before = s.sluice.accounted_water_m3();
    let water_before: f64 = (0..128 * 128)
        .map(|i| s.site.world.water_depth(i % 128, i / 128) as f64 * 0.0625)
        .sum();
    outflow::step(&mut s.sluice, &mut s.site, 1. / 60.);
    assert!(s.sluice.outflow.is_empty());
    assert!((before - s.sluice.accounted_water_m3()).abs() < 1e-14);
    let water_after: f64 = (0..128 * 128)
        .map(|i| s.site.world.water_depth(i % 128, i / 128) as f64 * 0.0625)
        .sum();
    assert!((water_after - water_before - before).abs() < 1e-8);
    let (mut mx, mut mz) = (0., 0.);
    for i in 0..128 * 128 {
        let vol = s.site.world.water_depth(i % 128, i / 128) as f64 * 0.0625;
        mx += vol * s.site.hydraulic.velocity_x[i] as f64;
        mz += vol * s.site.hydraulic.velocity_z[i] as f64;
    }
    assert!((mx - before * 0.8).abs() < 1e-8);
    assert!((mz + before * 0.4).abs() < 1e-8);
    assert!(s.site.hydraulic.impact_dirty);
}
#[test]
fn old_saves_default_to_no_airborne_water_and_pause_stops_flight() {
    let mut s = Session::new();
    let json = s.save_json().unwrap();
    let mut old: serde_json::Value = serde_json::from_str(&json).unwrap();
    old["sluice"].as_object_mut().unwrap().remove("outflow");
    assert!(Session::load_json(&old.to_string())
        .unwrap()
        .sluice
        .outflow
        .is_empty());
    s.sluice.outflow.push(OutflowParticle {
        position: Vec3::new(14., 2., 20.),
        velocity: Vec3::X,
    });
    s.paused = true;
    s.step(Default::default());
    assert_eq!(s.sluice.outflow[0].position, Vec3::new(14., 2., 20.));
}

#[test]
fn clean_water_impact_dilutes_all_suspension_without_creating_sediment() {
    let mut s = Session::new();
    let ix = 58;
    let iz = 82;
    let i = iz * 128 + ix;
    let ground = s.site.world.ground_height(ix, iz);
    s.site.world.water_surface[i] = ground + 0.02;
    s.site.hydraulic.suspended_sediment = vec![0.; 128 * 128];
    s.site.hydraulic.suspended_sediment[i] = 0.08;
    s.site.hydraulic.suspended_paydirt = vec![0.; 128 * 128];
    s.site.hydraulic.suspended_paydirt[i] = 0.03;
    let before_sand = s.site.world.water_depth(ix, iz) * s.site.hydraulic.suspended_sediment[i];
    let before_pay = s.site.world.water_depth(ix, iz) * s.site.hydraulic.suspended_paydirt[i];
    s.sluice.outflow = vec![
        OutflowParticle {
            position: Vec3::new(ix as f32 * 0.25, ground + 0.02, iz as f32 * 0.25),
            velocity: -Vec3::Y
        };
        1000
    ];
    outflow::step(&mut s.sluice, &mut s.site, 1. / 60.);
    assert!(s.sluice.outflow.is_empty());
    assert!(
        (before_sand - s.site.world.water_depth(ix, iz) * s.site.hydraulic.suspended_sediment[i])
            .abs()
            < 1e-8
    );
    assert!(
        (before_pay - s.site.world.water_depth(ix, iz) * s.site.hydraulic.suspended_paydirt[i])
            .abs()
            < 1e-8
    );
    let loaded = Session::load_json(&s.save_json().unwrap()).unwrap();
    assert!(loaded.site.hydraulic.impact_dirty);
}

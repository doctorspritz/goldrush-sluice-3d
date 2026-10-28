//! Production upload/readback and trace-gold custody, beyond isolated GPU stages.
use mining_physics::Hydraulics;
use mining_sim::{
    session::Controls,
    site::{CELL, WIDTH},
    water_balance::WaterBalance,
    Session,
};

fn sheet() -> Session {
    let mut s = Session::new();
    s.site.soil_water = Default::default();
    for i in 0..WIDTH * WIDTH {
        s.site.world.bedrock_elevation[i] = 0.;
        s.site.world.terrain_sediment[i] = 0.125;
        s.site.world.overburden_thickness[i] = 0.125;
        s.site.world.gravel_thickness[i] = 0.125;
        s.site.world.paydirt_thickness[i] = 0.125;
        s.site.world.water_surface[i] = 0.5 + 0.0008;
        s.site.gold_g[i] = 0.;
    }
    s.site.hydraulic.suspended_sediment = vec![0.; WIDTH * WIDTH];
    s.site.hydraulic.suspended_overburden = vec![0.; WIDTH * WIDTH];
    s.site.hydraulic.suspended_gravel = vec![0.; WIDTH * WIDTH];
    s.site.hydraulic.suspended_paydirt = vec![0.; WIDTH * WIDTH];
    s.site.hydraulic.suspended_gold_g = vec![0.; WIDTH * WIDTH];
    for z in 58..=86 {
        for x in 10..=38 {
            let i = z * WIDTH + x;
            s.site.hydraulic.suspended_sediment[i] = 0.15;
            s.site.hydraulic.suspended_paydirt[i] = 0.01;
            s.site.world.suspended_sediment[i] = 0.16;
        }
    }
    s.site.hydraulic.suspended_gold_g[72 * WIDTH + 24] = 0.007;
    s.sluice.placement = Some(mining_sim::Placement::suggested(&s.site));
    s.initial_gold_g = s.accounted_gold();
    s.water_balance = WaterBalance::new(s.stored_water_m3(), 0., false);
    s
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn production_film_deposition_preserves_local_gold_and_precision_stock_through_restore() {
    let mut s = sheet();
    let i = 72 * WIDTH + 24;
    let area = (CELL as f64).powi(2);
    let h = s.site.world.water_depth(i % WIDTH, i / WIDTH) as f64;
    let mobile = s.site.hydraulic.suspended_paydirt[i] as f64 * h * area;
    let initial = s.site.bed_class_volume_m3(i, 3) + mobile;
    if let Some(dir) = std::env::var_os("SLUICE_FILM_FIXTURE_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut source: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
        source["version"] = 4.into();
        source["site"]["hydraulic"]
            .as_object_mut()
            .unwrap()
            .remove("bed_remainder_m3");
        std::fs::write(dir.join("sheet-v4.json"), source.to_string()).unwrap();
    }
    let mut physics = Hydraulics::new().unwrap();
    for _ in 0..3 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    assert_eq!(s.site.hydraulic.steps, 1);
    let remaining = s.site.suspended_class_volumes_m3(i)[3];
    let residual = s.site.bed_class_volume_m3(i, 3) + remaining - initial;
    assert!(
        residual.abs() < mobile * 1e-6,
        "local production material budget: {residual}"
    );
    assert!(
        s.site.hydraulic.suspended_gold_g[i] < 1e-9,
        "stranded paydirt left its gold suspended"
    );
    assert!(
        (s.site.gold_g[i] - 0.007).abs() < 1e-9,
        "gold missed the deposited bed"
    );
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-12);
    assert!(s.site.hydraulic.bed_remainder_m3[i].iter().any(|v| *v > 0.));
    let precision = s.site.hydraulic.bed_remainder_m3.clone();
    let saved = s.save_json().unwrap();
    s = Session::load_json(&saved).unwrap();
    physics.restore();
    assert_eq!(s.site.hydraulic.bed_remainder_m3, precision);
    for _ in 0..3 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-12);
    assert!(s.site.hydraulic.valid());
    Session::load_json(&s.save_json().unwrap()).unwrap();
    println!("PRODUCTION FILM h={h} mobile={mobile} material_residual={residual} bed_gold={} suspended_gold={} precision={:?}",s.site.gold_g[i],s.site.hydraulic.suspended_gold_g[i],s.site.hydraulic.bed_remainder_m3[i]);
    if let Some(dir) = std::env::var_os("SLUICE_FILM_FIXTURE_DIR") {
        std::fs::write(
            std::path::PathBuf::from(dir).join("sheet-after-v5.json"),
            s.save_json().unwrap(),
        )
        .unwrap();
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn production_shallow_sheet_keeps_water_through_surface_projection_and_restore() {
    let mut s = sheet();
    let initial = s.stored_water_m3();
    let mut physics = Hydraulics::new().unwrap();
    for _ in 0..60 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    let first_error = s.water_balance_error_m3();
    assert!(
        first_error.abs() < initial * 1e-6,
        "thin sheet water drift: {first_error}"
    );
    let saved = s.save_json().unwrap();
    s = Session::load_json(&saved).unwrap();
    physics.restore();
    for _ in 0..60 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    let final_error = s.water_balance_error_m3();
    assert!(
        final_error.abs() < initial * 1e-6,
        "restored thin sheet water drift: {final_error}"
    );
    assert!(s.site.soil_water.surface_m3.iter().sum::<f64>() > 0.);
    println!("PROJECTION WATER initial={initial} first_error={first_error} final_error={final_error} queued={} momentum_valid={}",
        s.site.soil_water.surface_m3.iter().sum::<f64>(), s.site.soil_water.valid());
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn a_moving_film_deposits_its_gold_where_the_marked_paydirt_arrives() {
    for (vx, vz, offset) in [
        (2., 0., 1),
        (-2., 0., -1),
        (0., 2., WIDTH as isize),
        (0., -2., -(WIDTH as isize)),
    ] {
        let mut s = sheet();
        s.site.hydraulic.suspended_sediment.fill(0.);
        s.site.hydraulic.suspended_paydirt.fill(0.);
        s.site.hydraulic.suspended_gold_g.fill(0.);
        let donor = 64 * WIDTH + 56;
        let receiver = (donor as isize + offset) as usize;
        s.site.hydraulic.suspended_paydirt[donor] = 0.01;
        s.site.hydraulic.suspended_gold_g[donor] = 0.007;
        s.site.hydraulic.velocity_x = vec![vx; WIDTH * WIDTH];
        s.site.hydraulic.velocity_z = vec![vz; WIDTH * WIDTH];
        s.site.hydraulic.impact_dirty = true;
        s.initial_gold_g = s.accounted_gold();
        let mobile = s.site.suspended_class_volumes_m3(donor)[3];
        let bed_before = s.site.bed_class_volume_m3(receiver, 3);
        let mut physics = Hydraulics::new().unwrap();
        for _ in 0..3 {
            s.step(Controls::default());
            physics.step(&mut s).unwrap();
        }
        let arrived = s.site.bed_class_volume_m3(receiver, 3) - bed_before
            + s.site.suspended_class_volumes_m3(receiver)[3];
        let expected_gold = 0.007 * arrived / mobile;
        let actual_gold = s.site.gold_g[receiver] + s.site.hydraulic.suspended_gold_g[receiver];
        println!("MARKED FILM arrived={arrived} mobile={mobile} expected_gold={expected_gold} actual_gold={actual_gold}");
        assert!(
            arrived > mobile * 1e-3,
            "control did not transport a useful paydirt pulse"
        );
        assert!(
        (actual_gold - expected_gold).abs() < 0.007 * 1e-6,
        "gold did not follow local marked paydirt: expected={expected_gold} actual={actual_gold}"
    );
        assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-12);
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn local_gold_replay_continues_identically_after_restoring_a_mixed_bed_and_mobile_load() {
    let mut s = sheet();
    s.site.world.water_surface.fill(0.52);
    s.site.hydraulic.suspended_sediment.fill(0.);
    s.site.hydraulic.suspended_paydirt.fill(0.);
    s.site.hydraulic.suspended_gold_g.fill(0.);
    let i = 64 * WIDTH + 56;
    s.site.hydraulic.suspended_paydirt[i] = 0.1;
    s.site.hydraulic.suspended_gold_g[i] = 0.007;
    s.site.gold_g[i] = 1.;
    s.site.gold_g[i + 1] = 0.01;
    s.site.hydraulic.velocity_x = vec![1.; WIDTH * WIDTH];
    s.site.hydraulic.velocity_z = vec![0.; WIDTH * WIDTH];
    s.site.hydraulic.impact_dirty = true;
    s.initial_gold_g = s.accounted_gold();
    s.water_balance = WaterBalance::new(s.stored_water_m3(), 0., false);
    let mut physics = Hydraulics::new().unwrap();
    for _ in 0..6 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    let checkpoint = s.save_json().unwrap();
    for _ in 0..6 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    let expected_bed = s.site.gold_g.clone();
    let expected_mobile = s.site.hydraulic.suspended_gold_g.clone();
    let expected_export = s.site.hydraulic.exported_gold_g;
    s = Session::load_json(&checkpoint).unwrap();
    physics.restore();
    for _ in 0..6 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    assert_eq!(s.site.gold_g, expected_bed);
    assert_eq!(s.site.hydraulic.suspended_gold_g, expected_mobile);
    assert_eq!(s.site.hydraulic.exported_gold_g, expected_export);
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-12);
    println!(
        "MIXED GOLD live restore exact, bed={} mobile={} receiver_bed={} receiver_mobile={}",
        s.site.gold_g[i],
        s.site.hydraulic.suspended_gold_g[i],
        s.site.gold_g[i + 1],
        s.site.hydraulic.suspended_gold_g[i + 1]
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn collapsing_non_pay_strata_do_not_transport_a_stationary_coarse_paydirt_gold_tag() {
    let mut s = sheet();
    s.site.hydraulic.suspended_sediment.fill(0.);
    s.site.hydraulic.suspended_paydirt.fill(0.);
    s.site.hydraulic.suspended_gold_g.fill(0.);
    let i = 64 * WIDTH + 56;
    s.site.world.terrain_sediment[i] = 2.;
    for j in 0..WIDTH * WIDTH {
        s.site.world.water_surface[j] = s.site.world.ground_height(j % WIDTH, j / WIDTH);
    }
    s.site.gold_g[i] = 0.007;
    s.initial_gold_g = s.accounted_gold();
    s.water_balance = WaterBalance::new(s.stored_water_m3(), 0., false);
    let mut physics = Hydraulics::new().unwrap();
    for _ in 0..3 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    assert!(s.site.world.terrain_sediment[i] < 2.);
    assert_eq!(s.site.gold_g[i], 0.007);
    assert_eq!(s.site.hydraulic.suspended_gold_g[i], 0.);
    assert_eq!(s.site.world.paydirt_thickness[i], 0.125);
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn a_deposit_below_f64_geometric_addition_precision_still_receives_its_gold() {
    let mut s = sheet();
    s.site.hydraulic.suspended_sediment.fill(0.);
    s.site.hydraulic.suspended_paydirt.fill(0.);
    s.site.hydraulic.suspended_gold_g.fill(0.);
    let i = 64 * WIDTH + 56;
    s.site.hydraulic.suspended_paydirt[i] = 1e-15;
    s.site.hydraulic.suspended_gold_g[i] = 1e-15;
    s.initial_gold_g = s.accounted_gold();
    let mut physics = Hydraulics::new().unwrap();
    for _ in 0..3 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    assert_eq!(s.site.world.paydirt_thickness[i], 0.125);
    assert!(s.site.hydraulic.bed_remainder_m3[i][3] > 0.);
    assert!(s.site.gold_g[i] > 0.999 * s.initial_gold_g);
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-28);
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn a_finite_intake_drain_and_clean_rewet_preserve_local_deposited_and_sampled_gold() {
    let mut s = sheet();
    let i = 72 * WIDTH + 24;
    let mut physics = Hydraulics::new().unwrap();
    for _ in 0..3 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    let bed_gold = s.site.gold_g[i];
    let point = mining_sim::Vec3::new(6., 0., 18.);
    let packet = s.site.withdraw_intake(point, 1.);
    assert!(packet.water_m3 > 0.);
    assert_eq!(s.site.world.water_depth(24, 72), 0.);
    // This controlled outlet collects the finite intake sample outside the map.
    s.water_balance.boundary_outflow_m3 += packet.water_m3;
    s.site.hydraulic.exported_gold_g += packet.suspension.gold_g;
    for ch in 0..4 {
        s.site.hydraulic.exported_material_m3[ch] += packet.suspension.solid_m3[ch];
    }
    let saved = s.save_json().unwrap();
    s = Session::load_json(&saved).unwrap();
    physics.restore();
    assert_eq!(s.site.gold_g[i], bed_gold);
    let added = s.site.add_clean_inflow(point, 0.02, 0.01, 0.1);
    assert!(added > 0.);
    s.water_balance.external_inflow_m3 += added;
    for _ in 0..6 {
        s.step(Controls::default());
        physics.step(&mut s).unwrap();
    }
    assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-12);
    assert_eq!(
        s.site.gold_g[i], bed_gold,
        "clean rewet moved gold under intact covering strata"
    );
    assert!(s.water_balance_error_m3().abs() < 1e-6);
    println!(
        "DRAIN/REWET collected_water={} collected_gold={} bed_gold={} residual_gold={}",
        packet.water_m3,
        packet.suspension.gold_g,
        bed_gold,
        s.accounted_gold() - s.initial_gold_g
    );
}

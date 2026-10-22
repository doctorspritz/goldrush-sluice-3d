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

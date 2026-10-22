use mining_sim::{
    site::{CELL, WIDTH},
    water_balance::WaterBalance,
    Session, Site, Vec3,
};
const AREA: f64 = (CELL as f64) * (CELL as f64);
const I: usize = 64 * WIDTH + 56;
fn fixture(ground: f32) -> Site {
    let mut s = Site::new();
    s.soil_water = Default::default();
    s.hydraulic = Default::default();
    s.world.bedrock_elevation.fill(ground);
    s.world.terrain_sediment.fill(0.);
    s.world.overburden_thickness.fill(0.);
    s.world.gravel_thickness.fill(0.);
    s.world.paydirt_thickness.fill(0.);
    s.world.water_surface.fill(ground);
    s.gold_g.fill(0.);
    s.hydraulic.velocity_x = vec![0.; WIDTH * WIDTH];
    s.hydraulic.velocity_z = vec![0.; WIDTH * WIDTH];
    s
}
#[test]
fn gpu_projection_preserves_thin_water_momentum_and_all_four_material_classes() {
    for ground in [0.5, 32., -32.] {
        for h in [1e-7_f32, 0.00025, 0.0008, 0.1] {
            let mut s = fixture(ground);
            let classes = [0.15_f32, 0.02, 0.03, 0.01];
            for (field, c) in [
                &mut s.hydraulic.suspended_sediment,
                &mut s.hydraulic.suspended_overburden,
                &mut s.hydraulic.suspended_gravel,
                &mut s.hydraulic.suspended_paydirt,
            ]
            .into_iter()
            .zip(classes)
            {
                *field = vec![0.; WIDTH * WIDTH];
                field[I] = c;
            }
            s.hydraulic.velocity_x[I] = 2.3;
            s.hydraulic.velocity_z[I] = -0.7;
            let mut depths = vec![0.; WIDTH * WIDTH];
            depths[I] = h;
            let v = h as f64 * AREA;
            s.project_terrain_water(&depths);
            let represented = s.world.water_depth(I % WIDTH, I / WIDTH) as f64 * AREA;
            assert!((represented + s.soil_water.surface_m3[I] - v).abs() < 1e-18);
            for (axis, velocity) in [&s.hydraulic.velocity_x, &s.hydraulic.velocity_z]
                .into_iter()
                .enumerate()
            {
                let expected = v * [2.3_f32, -0.7][axis] as f64;
                assert!(
                    (represented * velocity[I] as f64
                        + s.soil_water.surface_momentum_m4_s[I][axis]
                        - expected)
                        .abs()
                        < 1e-17
                );
            }
            let mobile = s.suspended_class_volumes_m3(I);
            for (ch, c) in classes.into_iter().enumerate() {
                let total = mobile[ch] + s.bed_class_volume_m3(I, ch);
                assert!(
                    (total - c as f64 * v).abs() < 1e-18,
                    "ground={ground} h={h} class={ch}"
                );
            }
            assert!(s.hydraulic.valid());
            assert!(s.soil_water.valid());
        }
    }
}
#[test]
fn queued_flow_and_mobile_precision_survive_restore_then_intake() {
    let mut s = Session::new();
    s.site = fixture(32.);
    s.site.hydraulic.suspended_sediment = vec![0.; WIDTH * WIDTH];
    s.site.hydraulic.suspended_sediment[I] = 0.15;
    s.site.hydraulic.velocity_x[I] = 2.3;
    let mut depths = vec![0.; WIDTH * WIDTH];
    depths[I] = 0.0008;
    s.site.project_terrain_water(&depths);
    let mass = s.site.suspended_class_volumes_m3(I)[0];
    let queued = s.site.soil_water.surface_m3[I];
    assert!(queued > 0.);
    s.initial_gold_g = s.accounted_gold();
    s.water_balance = WaterBalance::new(s.stored_water_m3(), 0., false);
    let json = s.save_json().unwrap();
    let mut restored = Session::load_json(&json).unwrap();
    assert_eq!(
        restored.site.soil_water.surface_momentum_m4_s,
        s.site.soil_water.surface_momentum_m4_s
    );
    assert_eq!(
        restored.site.hydraulic.suspended_remainder_m3,
        s.site.hydraulic.suspended_remainder_m3
    );
    let packet = restored.site.withdraw_intake(Vec3::new(14., 32., 16.), 1.);
    assert!((packet.suspension.solid_m3[0] - mass).abs() < 1e-18);
    assert_eq!(restored.site.suspended_class_volumes_m3(I), [0.; 4]);
    assert_eq!(restored.site.soil_water.surface_m3[I], queued);
    let mut legacy: serde_json::Value = serde_json::from_str(&json).unwrap();
    legacy["version"] = 4.into();
    assert!(Session::load_json(&legacy.to_string()).is_err());
}
#[test]
fn repeated_subgeometry_depths_release_owned_water_with_downstream_momentum() {
    let mut s = fixture(32.);
    let h = 1e-7_f32;
    let mut depths = vec![0.; WIDTH * WIDTH];
    let mut input = 0.;
    for _ in 0..60 {
        // Emulate the next solver's original represented water plus a finite inlet.
        depths[I] = s.world.water_depth(I % WIDTH, I / WIDTH) + h;
        input += h as f64 * AREA;
        s.hydraulic.velocity_x[I] = 2.;
        s.project_terrain_water(&depths);
        s.reconcile_soil_water();
    }
    let represented = s.world.water_depth(I % WIDTH, I / WIDTH) as f64 * AREA;
    assert!(represented > 0.);
    assert!((represented + s.soil_water.surface_m3[I] - input).abs() < 1e-12);
    assert!((s.hydraulic.velocity_x[I] - 2.).abs() < 1e-6);
}

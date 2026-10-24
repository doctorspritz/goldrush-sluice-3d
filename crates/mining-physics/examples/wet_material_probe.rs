//! Conservative wet-load diagnostic states; not a human input replay.
use mining_sim::{storage, water_balance::WaterBalance, Material, Vec3};
use std::path::PathBuf;
fn main() {
    let mut args = std::env::args().skip(1);
    let directory = PathBuf::from(args.next().expect("OUTPUT DIRECTORY"));
    let source = PathBuf::from(args.next().expect("SOURCE SAVE"));
    assert!(
        !directory.join("receipt.json").exists(),
        "preserve diagnostic states"
    );
    std::fs::create_dir_all(&directory).unwrap();
    let mut s = storage::load(&source).unwrap();
    s.site.soil_water.ensure();
    for z in 80..100 {
        for x in 68..88 {
            let i = s.site.world.idx(x, z);
            s.site.soil_water.pore_m3[i] = s.site.column_pore_volume_m3(i) * 0.8;
        }
    }
    s.water_balance = WaterBalance::new(s.stored_water_m3(), s.sim_time, false);
    let initial = s.stored_water_m3();
    let initial_mass = s.accounted_dry_mass_kg();
    let mut load = Material::default();
    for z in 88..92 {
        for x in 76..80 {
            let px = x as f32 * 0.25;
            let pz = z as f32 * 0.25;
            load.add(
                s.site
                    .dig(Vec3::new(px, s.site.height(px, pz), pz), 0.05, 0.12, 1.),
            );
        }
    }
    assert!(load.volume() > 0.1 && load.pore_water_m3 > 0.02);
    s.machine.bucket.add(load);
    assert!((s.stored_water_m3() - initial).abs() < 1e-10);
    assert!((s.accounted_dry_mass_kg() - initial_mass).abs() < 1e-6);
    storage::save(&s, &directory.join("wet-bucket.save.json")).unwrap();
    let mut dry = s.clone();
    dry.machine.bucket.pore_water_m3 = 0.;
    dry.water_balance = WaterBalance::new(dry.stored_water_m3(), dry.sim_time, false);
    storage::save(&dry, &directory.join("dry-bucket.save.json")).unwrap();
    let batch = s.machine.bucket.take_fraction(1.);
    let overflow = s.sluice.add_to_hopper(batch);
    if !overflow.is_empty() {
        let p = mining_sim::Sluice::hopper_position(&s.site);
        s.site.deposit(p + Vec3::X * 1.3, overflow);
    }
    assert!((s.stored_water_m3() - initial).abs() < 1e-10);
    storage::save(&s, &directory.join("wet-flow.save.json")).unwrap();
    let receipt = serde_json::json!({"source_save":source,"initial_condition":"0.8 column saturation in specified 20x20 bank patch; water budget starts after conditioning","cut_source":"16 actual domain column cuts of 0.12 m; diagnostic transfer into bucket and hopper, not operator replay","reference_bulk_m3":load.volume(),"dry_mass_kg":load.mass_kg(),"pore_water_m3":load.pore_water_m3,"wet_mass_kg":load.wet_mass_kg(),"wet_bucket_saturation":load.saturation(),"water_error_after_transfers_m3":s.stored_water_m3()-initial,"dry_mass_error_after_transfers_kg":s.accounted_dry_mass_kg()-initial_mass,"overflow_to_site":overflow,"save_version":4,"dry_comparison":"same geometry/dry inventory, separately declared zero bucket moisture initial condition"});
    std::fs::write(
        directory.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    println!("WET MATERIAL STATES: {receipt}");
}

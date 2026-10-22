use mining_sim::{
    site::{BedAgeUnit, BED_STABILITY_MICROS, WIDTH},
    Session,
};

#[test]
fn new_clock_roundtrips_without_rescaling_physical_age() {
    let mut s = Session::new();
    s.site.hydraulic.settling_time = vec![400_000; WIDTH * WIDTH];
    s.site.hydraulic.bed_exchange_error = vec![[1e-9, -1e-9, 2e-9, -2e-9]; WIDTH * WIDTH];
    let json = s.save_json().unwrap();
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["version"], 4);
    assert_eq!(
        value["site"]["hydraulic"]["settling_time_unit"],
        "microseconds"
    );
    let loaded = Session::load_json(&json).unwrap();
    assert_eq!(
        loaded.site.hydraulic.settling_time,
        s.site.hydraulic.settling_time
    );
    let again = Session::load_json(&loaded.save_json().unwrap()).unwrap();
    assert_eq!(
        again.site.hydraulic.bed_exchange_error,
        s.site.hydraulic.bed_exchange_error
    );
    assert_eq!(
        loaded.site.hydraulic.settling_time,
        again.site.hydraulic.settling_time
    );
}

#[test]
fn version_two_tick_ages_migrate_once_with_unchanged_inventories() {
    let mut s = Session::new();
    s.site.soil_water = Default::default();
    s.water_balance.initial_stored_m3 = s.stored_water_m3();
    s.site.hydraulic.settling_time = (0..WIDTH * WIDTH)
        .map(|i| [0, 1, 10, 30, 255][i % 5])
        .collect();
    let mut old: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    old["version"] = 2.into();
    old["site"].as_object_mut().unwrap().remove("soil_water");
    old["site"]["hydraulic"]
        .as_object_mut()
        .unwrap()
        .remove("settling_time_unit");
    old["site"]["hydraulic"]
        .as_object_mut()
        .unwrap()
        .remove("bed_exchange_error");
    let loaded = Session::load_json(&old.to_string()).unwrap();
    assert_eq!(
        loaded.site.hydraulic.settling_time_unit,
        BedAgeUnit::Microseconds
    );
    for (i, age) in loaded.site.hydraulic.settling_time.iter().enumerate() {
        assert_eq!(
            *age,
            [
                0,
                50_000,
                500_000,
                BED_STABILITY_MICROS,
                BED_STABILITY_MICROS
            ][i % 5]
        );
    }
    assert_eq!(
        s.accounted_bulk_material_m3(),
        loaded.accounted_bulk_material_m3()
    );
    assert_eq!(s.stored_water_m3(), loaded.stored_water_m3());
    assert_eq!(s.accounted_gold(), loaded.accounted_gold());
    let saved = loaded.save_json().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&saved).unwrap()["version"],
        4
    );
    let resumed = Session::load_json(&saved).unwrap();
    assert_eq!(
        resumed.site.hydraulic.settling_time,
        loaded.site.hydraulic.settling_time
    );
}

#[test]
fn invalid_or_ambiguous_bed_clocks_are_rejected() {
    let mut s = Session::new();
    s.site.hydraulic.settling_time = vec![0; WIDTH * WIDTH];
    let saved: serde_json::Value = serde_json::from_str(&s.save_json().unwrap()).unwrap();
    for case in 0..7 {
        let mut value = saved.clone();
        match case {
            0 => {
                value["site"]["hydraulic"]
                    .as_object_mut()
                    .unwrap()
                    .remove("settling_time_unit");
            }
            1 => {
                value["version"] = 2.into();
            }
            2 => {
                value["site"]["hydraulic"]["settling_time"][0] = (BED_STABILITY_MICROS + 1).into();
            }
            3 => {
                value["site"]["hydraulic"]["settling_time_unit"] = "unknown".into();
            }
            4 => {
                value["version"] = 2.into();
                value["site"]["hydraulic"]
                    .as_object_mut()
                    .unwrap()
                    .remove("settling_time_unit");
                value["site"]["hydraulic"]["settling_time"][0] = 256.into();
            }
            5 => {
                value["site"]["hydraulic"]["bed_exchange_error"] =
                    serde_json::json!([[0., 0., 0., 0.]]);
            }
            _ => {
                value["site"]["hydraulic"]["bed_exchange_error"] =
                    serde_json::json!(vec![[1., 0., 0., 0.]; WIDTH * WIDTH]);
            }
        }
        assert!(
            Session::load_json(&value.to_string()).is_err(),
            "accepted invalid clock case {case}"
        );
    }
}

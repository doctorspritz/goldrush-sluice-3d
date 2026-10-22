//! Controlled river -> pump -> real FLIP/DEM -> tailings inventory experiment.
//! This is an initial-value test, not a gameplay replay or a constructed pond.
use mining_sim::{
    session::Controls,
    site::{CELL, WIDTH},
    sluice::HOPPER_Z,
    Material, Placement, Session, Site, FIXED_DT,
};

fn fixture() -> Session {
    let mut s = Session::new();
    for layer in [
        &mut s.site.world.bedrock_elevation,
        &mut s.site.world.paydirt_thickness,
        &mut s.site.world.overburden_thickness,
        &mut s.site.world.gravel_thickness,
        &mut s.site.world.terrain_sediment,
    ] {
        layer.fill(0.);
    }
    s.site.world.water_surface.fill(0.08);
    s.site.gold_g.fill(0.);
    let n = WIDTH * WIDTH;
    s.site.hydraulic.suspended_sediment = vec![0.; n];
    s.site.hydraulic.suspended_overburden = vec![0.; n];
    s.site.hydraulic.suspended_gravel = vec![0.; n];
    s.site.hydraulic.suspended_paydirt = vec![0.; n];
    s.site.hydraulic.suspended_gold_g = vec![0.; n];
    let x = (Site::river_center(HOPPER_Z) / CELL) as usize;
    let z = (HOPPER_Z / CELL) as usize;
    for cz in z - 1..=z + 1 {
        for cx in x - 1..=x + 1 {
            let i = cz * WIDTH + cx;
            s.site.hydraulic.suspended_sediment[i] = 0.001;
            s.site.hydraulic.suspended_overburden[i] = 0.0003;
            s.site.hydraulic.suspended_gravel[i] = 0.0001;
            s.site.hydraulic.suspended_paydirt[i] = 0.0003;
            s.site.world.suspended_sediment[i] = 0.0017;
            s.site.hydraulic.suspended_gold_g[i] = 0.0001;
        }
    }
    s.initial_gold_g = s.accounted_gold();
    s.water_balance.initial_stored_m3 = s.stored_water_m3();
    s.sluice.placement = Some(Placement::suggested(&s.site));
    s.sluice.pump_on = true;
    s
}

// An independent inventory in the domain's three bulk-material categories.
// Includes every stored solid and explicit boundary discharge, never cumulative
// intake/emission counters. Geological paydirt maps to gravel, as in excavation.
fn inventory(s: &Session) -> Material {
    let w = &s.site.world;
    let a = &s.site.hydraulic;
    let mut m = s.sluice.active_material();
    for other in [
        s.sluice.tailings,
        s.sluice.concentrate,
        s.machine.bucket,
        s.player.hand,
    ] {
        m.add(other);
    }
    for falling in &s.falling {
        m.add(falling.material);
    }
    for i in 0..WIDTH * WIDTH {
        let depth = w.water_depth(i % WIDTH, i / WIDTH) as f64;
        let suspended = |field: &[f32]| field.get(i).copied().unwrap_or(0.) as f64 * depth;
        let area = (CELL as f64).powi(2);
        m.sand_m3 += (w.terrain_sediment[i] as f64 + suspended(&a.suspended_sediment)) * area;
        m.soil_m3 += (w.overburden_thickness[i] as f64 + suspended(&a.suspended_overburden)) * area;
        m.gravel_m3 += (w.gravel_thickness[i] as f64
            + w.paydirt_thickness[i] as f64
            + suspended(&a.suspended_gravel)
            + suspended(&a.suspended_paydirt))
            * area;
    }
    m.sand_m3 += a.exported_material_m3[0];
    m.soil_m3 += a.exported_material_m3[1];
    m.gravel_m3 += a.exported_material_m3[2] + a.exported_material_m3[3];
    m.gold_g = s.accounted_gold();
    m
}

fn main() -> Result<(), String> {
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "intake-probe.json".into());
    if std::path::Path::new(&output).exists() {
        return Err("Preserve previous evidence: choose a new output path".into());
    }
    let mut s = fixture();
    let initial = inventory(&s);
    let mut h = mining_physics::Hydraulics::new()?;
    let started = std::time::Instant::now();
    let mut samples = Vec::new();
    let mut save_restore_exact = false;
    for tick in 0..720 {
        s.step(Controls::default());
        h.step(&mut s)?;
        if tick == 359 {
            let before = inventory(&s);
            let water = s.stored_water_m3();
            let text = s.save_json()?;
            s = Session::load_json(&text)?;
            save_restore_exact = before == inventory(&s) && water == s.stored_water_m3();
            h.restore();
        }
        if tick % 60 == 59 {
            samples.push(serde_json::json!({"sim_seconds":s.sim_time,
                "inventory":inventory(&s),"water_error_m3":s.water_balance_error_m3(),
                "pumped_water_m3":s.sluice.water_used_m3,"inlet":s.sluice.inlet,
                "grains":s.sluice.grains.len(),"tailings":s.sluice.tailings}));
        }
    }
    let final_inventory = inventory(&s);
    let errors = [
        final_inventory.sand_m3 - initial.sand_m3,
        final_inventory.soil_m3 - initial.soil_m3,
        final_inventory.gravel_m3 - initial.gravel_m3,
    ];
    let tolerance = initial.volume() * 1e-4;
    let passed = save_restore_exact
        && errors.iter().all(|e| e.abs() < tolerance)
        && (final_inventory.gold_g - initial.gold_g).abs() < 1e-9
        && s.water_balance_error_m3().abs() < 1e-4
        && s.sluice.inlet.emitted.solid_m3.iter().all(|v| *v > 0.)
        && s.sluice.inlet.emitted.gold_g > 0.
        && s.sluice.hopper == Material::default();
    let report = serde_json::json!({"passed":passed,"fixture":"flat initial-value claim; three-by-three suspended pulse; no hopper feed",
        "adapter":h.adapter_name,"initial":initial,"final":final_inventory,
        "component_errors_m3_sand_soil_gravel":errors,"component_tolerance_m3":tolerance,
        "gold_error_g":final_inventory.gold_g-initial.gold_g,"water_error_m3":s.water_balance_error_m3(),
        "save_restore_exact":save_restore_exact,"sim_seconds":s.sim_time,
        "wall_seconds":started.elapsed().as_secs_f64(),"fixed_dt":FIXED_DT,
        "inlet":s.sluice.inlet,"pumped_water_m3":s.sluice.water_used_m3,
        "boundary_diagnostics":h.boundary_diagnostics(&s),
        "tailings":s.sluice.tailings,"samples":samples,
        "limits":["No physical hose priming/pressure or cavitation model", "Terrain trace-gold transport still needs substep/local validation", "Tailings are an explicit inventory, not a constructed settling pond", "Bulk-volume convention does not establish porosity or grain-mass accuracy"]});
    std::fs::write(
        output,
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{report}");
    if passed {
        Ok(())
    } else {
        Err("Integrated intake budget gate failed; report retained".into())
    }
}

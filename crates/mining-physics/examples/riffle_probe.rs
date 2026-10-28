//! Controlled, finite tracer runs through the actual FLIP/DEM pipeline.
//! A bare channel is a geometry control, not a target recovery percentage.
use glam::{Mat3, Quat};
use mining_sim::{session::Controls, sluice::Grain, GrainKind, Material, Placement, Session, Vec3};
use std::time::Instant;
fn main() {
    let output = std::env::args().nth(1).expect("metrics JSON");
    let mut results = Vec::new();
    for (name, riffles, flow) in [
        ("riffles_nominal", true, 0.00375),
        ("bare_nominal", false, 0.00375),
        ("riffles_high", true, 0.0075),
    ] {
        let mut s = Session::new();
        s.sluice.placement = Some(Placement::suggested(&s.site));
        s.sluice.riffles_enabled = riffles;
        s.sluice.pump_on = true;
        s.sluice.flow_m3s = flow;
        let mut solver = mining_physics::Hydraulics::new().unwrap();
        let started = Instant::now();
        for _ in 0..480 {
            s.step(Controls::default());
            solver.step(&mut s).unwrap();
        }
        let mut initial = [0.0; 2];
        for i in 0..32 {
            for (kind, density, slot) in
                [(GrainKind::Gold, 19300.0, 0), (GrainKind::Sand, 2650.0, 1)]
            {
                let diameter = 0.0003_f32;
                let volume = std::f64::consts::PI * (diameter as f64).powi(3) / 6.0;
                let material = if slot == 0 {
                    Material {
                        gold_g: volume * density * 1000.0,
                        ..Default::default()
                    }
                } else {
                    Material::sand_from_solid_volume(volume)
                };
                initial[slot] += if slot == 0 {
                    material.gold_g
                } else {
                    material.volume()
                };
                s.sluice.grains.push(Grain {
                    volume_resolved: true,
                    kind,
                    position: Vec3::new(
                        0.35,
                        (s.sluice.floor_cells() + 1) as f32 * 0.01
                            - 0.35 * s.sluice.placement.unwrap().slope_deg.to_radians().tan()
                            + 0.035,
                        0.04 + i as f32 * 0.01,
                    ),
                    velocity: Vec3::new(0.35, 0.0, 0.0),
                    diameter_m: diameter,
                    material,
                    resting: false,
                    affine: Mat3::ZERO,
                    rotation: Quat::IDENTITY,
                    angular_velocity: Vec3::ZERO,
                });
            }
        }
        // Record the externally specified assay in this controlled test only.
        s.initial_gold_g += initial[0];
        let mut pocket_samples = 0usize;
        let mut reverse_samples = 0usize;
        let mut slow_samples = 0usize;
        let mut top_speed = 0.0f64;
        let mut top_samples = 0usize;
        for tick in 0..480 {
            s.step(Controls::default());
            solver.step(&mut s).unwrap();
            if tick % 30 == 29 {
                for p in &s.sluice.water {
                    let x = p.position.x;
                    let local = s.sluice.grid_to_local(p.position);
                    if x > 0.42 && x < 1.35 && ((x - 0.40) % 0.32) < 0.12 && local.y < 0.035 {
                        pocket_samples += 1;
                        reverse_samples += usize::from(p.velocity.x < -0.02);
                        slow_samples += usize::from(p.velocity.length() < 0.15);
                    }
                    if x > 0.4 && x < 1.35 && ((x - 0.4) % 0.32) < 0.03 && local.y > 0.035 {
                        top_speed += p.velocity.x as f64;
                        top_samples += 1;
                    }
                }
            }
        }
        let active_gold = s.sluice.active_material().gold_g;
        let active_sand = s.sluice.active_material().sand_m3;
        let water = s.sluice.water.len() as f64 * s.sluice.profile.particle_m3()
            + s.sluice.water_credit_m3
            + s.sluice.water_returned_m3;
        let item = serde_json::json!({"case":name,"flow_m3s":flow,"sim_seconds":s.sim_time,"wall_seconds":started.elapsed().as_secs_f64(),"mean_water_downstream_ms":s.sluice.water_speed(),"water_particles":s.sluice.water.len(),"water_depth_m":s.sluice.water_depth(),"pocket_samples":pocket_samples,"pocket_reverse_fraction":reverse_samples as f64/pocket_samples.max(1) as f64,"pocket_slow_fraction":slow_samples as f64/pocket_samples.max(1) as f64,"crest_mean_downstream_ms":top_speed/top_samples.max(1) as f64,"gold_retained_fraction":active_gold/initial[0],"same_size_quartz_retained_fraction":active_sand/initial[1],"gold_outlet_g":s.sluice.tailings.gold_g,"resting_grains":s.sluice.grains.iter().filter(|g|g.resting).count(),"water_balance_error_m3":s.sluice.water_used_m3-water,"gold_balance_error_g":s.accounted_gold()-s.initial_gold_g});
        assert!((s.sluice.water_used_m3 - water).abs() < 1e-7);
        assert!((s.accounted_gold() - s.initial_gold_g).abs() < 1e-7);
        assert!(s.sluice.valid());
        println!("{item}");
        results.push(item);
        std::fs::write(&output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
    }
}

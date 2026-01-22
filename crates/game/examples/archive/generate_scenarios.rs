use game::editor::{EditorLayout, ScenarioConfig, TestFloorPiece};
use glam::Vec3;
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scenarios_dir = Path::new("crates/game/scenarios");
    std::fs::create_dir_all(scenarios_dir)?;

    // --- TEST 1: DEM Floor Collision ---
    let mut layout = EditorLayout::new();
    layout.test_floors.push(TestFloorPiece {
        id: 1,
        y: 0.0,
    });

    let mut dem_sim = ClusterSimulation3D::new(
        Vec3::new(-10.0, -2.0, -10.0),
        Vec3::new(20.0, 10.0, 20.0),
    );
    dem_sim.normal_stiffness = 1000.0;
    dem_sim.tangential_stiffness = 500.0;
    dem_sim.restitution = 0.1;

    let sand_mass = 2650.0 * (4.0 / 3.0) * std::f32::consts::PI * 0.008f32.powi(3);
    let sand_template = ClumpTemplate3D::generate(
        ClumpShape3D::Irregular {
            count: 5,
            seed: 123,
            style: sim3d::clump::IrregularStyle3D::Round,
        },
        0.008,
        sand_mass,
    );
    let sand_idx = dem_sim.add_template(sand_template.clone());

    for i in 0..5 {
        for j in 0..5 {
            let pos = Vec3::new(
                (i as f32 - 2.0) * 0.05,
                0.5 + (j as f32) * 0.03,
                0.0,
            );
            dem_sim.spawn(sand_idx, pos, Vec3::ZERO);
        }
    }

    let mut scenario_1 = ScenarioConfig::from_layout(layout);
    scenario_1.name = Some("DEM: Floor Collision".to_string());
    scenario_1.description = Some("Particles fall from height and settle on a floor".to_string());

    scenario_1.save_json(&scenarios_dir.join("dem_floor_collision.json"))?;
    println!("Generated dem_floor_collision.json");

    // --- TEST 2: DEM Wall Collision ---
    let mut layout_2 = EditorLayout::new();
    layout_2.test_boxes.push(game::editor::TestBoxPiece {
        id: 1,
        position: Vec3::ZERO,
        width: 0.4,
        depth: 0.4,
        wall_height: 0.3,
    });

    let mut dem_sim_2 = ClusterSimulation3D::new(
        Vec3::new(-10.0, -2.0, -10.0),
        Vec3::new(20.0, 10.0, 20.0),
    );
    dem_sim_2.normal_stiffness = 1000.0;
    dem_sim_2.tangential_stiffness = 500.0;
    dem_sim_2.restitution = 0.1;
    let sand_idx_2 = dem_sim_2.add_template(sand_template.clone());

    for i in 0..10 {
        let pos = Vec3::new(
            -0.1 + (i % 2) as f32 * 0.05,
            0.15,
            -0.05 + (i / 2) as f32 * 0.03,
        );
        let vel = Vec3::new(1.0, 0.0, 0.0);
        dem_sim_2.spawn(sand_idx_2, pos, vel);
    }

    let mut scenario_2 = ScenarioConfig::from_layout(layout_2);
    scenario_2.name = Some("DEM: Wall Collision".to_string());
    scenario_2.description = Some("Particles reflect off walls in a box".to_string());
    scenario_2.save_json(&scenarios_dir.join("dem_wall_collision.json"))?;
    println!("Generated dem_wall_collision.json");

    // --- TEST 3: DEM Density Separation ---
    let mut layout_3 = EditorLayout::new();
    layout_3.test_boxes.push(game::editor::TestBoxPiece {
        id: 1,
        position: Vec3::ZERO,
        width: 0.3,
        depth: 0.3,
        wall_height: 0.4,
    });

    let mut dem_sim_3 = ClusterSimulation3D::new(
        Vec3::new(-10.0, -2.0, -10.0),
        Vec3::new(20.0, 10.0, 20.0),
    );
    dem_sim_3.normal_stiffness = 1000.0;
    dem_sim_3.restitution = 0.1;
    let sand_idx_3 = dem_sim_3.add_template(sand_template.clone());
    
    let gold_mass = 19300.0 * (4.0 / 3.0) * std::f32::consts::PI * 0.008f32.powi(3);
    let gold_template = ClumpTemplate3D::generate(
        ClumpShape3D::Irregular {
            count: 5,
            seed: 42,
            style: sim3d::clump::IrregularStyle3D::Round,
        },
        0.008,
        gold_mass,
    );
    let gold_idx_3 = dem_sim_3.add_template(gold_template);

    for i in 0..5 {
        for j in 0..5 {
            let pos = Vec3::new(
                (i as f32 - 2.0) * 0.04,
                0.25 + (j as f32) * 0.02,
                (j as f32 - 2.0) * 0.02,
            );
            if (i + j) % 2 == 0 {
                dem_sim_3.spawn(gold_idx_3, pos, Vec3::ZERO);
            } else {
                dem_sim_3.spawn(sand_idx_3, pos, Vec3::ZERO);
            }
        }
    }

    let mut scenario_3 = ScenarioConfig::from_layout(layout_3);
    scenario_3.name = Some("DEM: Density Separation".to_string());
    scenario_3.description = Some("Mixed gold and sand dropped into box".to_string());
    scenario_3.save_json(&scenarios_dir.join("dem_density_separation.json"))?;
    println!("Generated dem_density_separation.json");

    // --- TEST 4: DEM Settling Time ---
    let mut layout_4 = EditorLayout::new();
    layout_4.test_boxes.push(game::editor::TestBoxPiece {
        id: 1,
        position: Vec3::ZERO,
        width: 0.5,
        depth: 0.5,
        wall_height: 0.3,
    });

    let mut dem_sim_4 = ClusterSimulation3D::new(
        Vec3::new(-10.0, -2.0, -10.0),
        Vec3::new(20.0, 10.0, 20.0),
    );
    dem_sim_4.normal_stiffness = 1000.0;
    dem_sim_4.restitution = 0.1;
    let sand_idx_4 = dem_sim_4.add_template(sand_template.clone());

    for i in 0..50 {
        let x = ((i * 7) % 10) as f32 * 0.04 - 0.2;
        let y = 0.3 + (i as f32 * 0.005);
        let z = ((i * 13) % 10) as f32 * 0.04 - 0.2;
        let pos = Vec3::new(x, y, z);
        dem_sim_4.spawn(sand_idx_4, pos, Vec3::ZERO);
    }

    let mut scenario_4 = ScenarioConfig::from_layout(layout_4);
    scenario_4.name = Some("DEM: Settling Time".to_string());
    scenario_4.description = Some("50 particles dropped into box, should all come to rest".to_string());
    scenario_4.save_json(&scenarios_dir.join("dem_settling_time.json"))?;
    println!("Generated dem_settling_time.json");

    Ok(())
}

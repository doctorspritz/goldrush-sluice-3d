//! Water Flow Tests
//! Tests to verify water flows correctly in the heightfield simulation.

extern crate game;
extern crate pollster;
extern crate sim3d;

use sim3d::World;

/// Test that water flows from high ground to low ground in the CPU simulation.
/// This verifies the basic SWE algorithm in sim3d::World works.
#[test]
fn test_cpu_water_flows_downhill() {
    // Small world with a slope
    let mut world = World::new(32, 32, 1.0, 5.0);

    // Create a slope: higher on the left (x=0), lower on the right (x=31)
    for z in 0..32 {
        for x in 0..32 {
            let idx = world.idx(x, z);
            // Slope from 5m on left to 2m on right
            let height = 5.0 - (x as f32 / 31.0) * 3.0;
            world.bedrock_elevation[idx] = height;
            world.overburden_thickness[idx] = 0.0;
            world.paydirt_thickness[idx] = 0.0;
            world.gravel_thickness[idx] = 0.0;
            world.terrain_sediment[idx] = 0.0;
        }
    }

    // Add water at the high side (left)
    for z in 10..22 {
        for x in 5..10 {
            let idx = world.idx(x, z);
            let ground = world.ground_height(x, z);
            world.water_surface[idx] = ground + 0.5; // 0.5m of water
        }
    }

    let initial_volume = world.total_water_volume();
    println!("Initial water volume: {}", initial_volume);
    assert!(initial_volume > 0.0, "Should have initial water");

    // Step simulation many times
    for _ in 0..100 {
        world.update_water_flow(0.016);
    }

    // Check that water has moved rightward (downhill)
    // Left side should have less water, right side should have more
    let mut left_depth_sum = 0.0;
    let mut right_depth_sum = 0.0;

    for z in 10..22 {
        for x in 5..10 {
            left_depth_sum += world.water_depth(x, z);
        }
        for x in 20..25 {
            right_depth_sum += world.water_depth(x, z);
        }
    }

    println!(
        "Left depth sum: {}, Right depth sum: {}",
        left_depth_sum, right_depth_sum
    );

    // Water should have flowed downhill - right side should have more now
    // or left side should have less than initial
    let initial_left_depth = 0.5 * 5.0 * 12.0; // 5 cells wide * 12 cells deep * 0.5m depth
    assert!(
        left_depth_sum < initial_left_depth * 0.9,
        "Water should have flowed away from the high side. Initial left: {}, Current left: {}",
        initial_left_depth,
        left_depth_sum
    );
}

/// Test that water conserves mass during flow.
#[test]
fn test_cpu_water_mass_conservation() {
    let mut world = World::new(32, 32, 1.0, 5.0);

    // Flat terrain
    for z in 0..32 {
        for x in 0..32 {
            let idx = world.idx(x, z);
            world.bedrock_elevation[idx] = 3.0;
            world.overburden_thickness[idx] = 0.0;
            world.paydirt_thickness[idx] = 0.0;
            world.gravel_thickness[idx] = 0.0;
            world.terrain_sediment[idx] = 0.0;
        }
    }

    // Add water mound in center (away from boundaries which drain)
    for z in 12..20 {
        for x in 12..20 {
            let idx = world.idx(x, z);
            let ground = world.ground_height(x, z);
            world.water_surface[idx] = ground + 1.0;
        }
    }

    let initial_volume = world.total_water_volume();
    println!("Initial water volume: {}", initial_volume);

    // Step simulation - water should spread but conserve mass (mostly)
    // Note: boundaries drain water, so some loss is expected
    for i in 0..50 {
        world.update_water_flow(0.016);
        if i % 10 == 0 {
            let vol = world.total_water_volume();
            println!("Step {}: volume = {}", i, vol);
        }
    }

    let final_volume = world.total_water_volume();
    println!("Final water volume: {}", final_volume);

    // Water should mostly be conserved (some boundary drain is OK)
    // Allow up to 30% loss due to boundary draining
    let ratio = final_volume / initial_volume;
    assert!(
        ratio >= 0.7,
        "Too much water lost: {} -> {} (ratio {})",
        initial_volume,
        final_volume,
        ratio
    );
}

/// Test that water spreads out on flat terrain.
#[test]
fn test_cpu_water_spreads_on_flat_terrain() {
    let mut world = World::new(32, 32, 1.0, 3.0);

    // Flat terrain
    for z in 0..32 {
        for x in 0..32 {
            let idx = world.idx(x, z);
            world.bedrock_elevation[idx] = 3.0;
            world.overburden_thickness[idx] = 0.0;
            world.paydirt_thickness[idx] = 0.0;
            world.gravel_thickness[idx] = 0.0;
            world.terrain_sediment[idx] = 0.0;
        }
    }

    // Add water only in center cell
    let center_idx = world.idx(16, 16);
    let ground = world.ground_height(16, 16);
    world.water_surface[center_idx] = ground + 2.0; // 2m mound

    let initial_center_depth = world.water_depth(16, 16);
    println!("Initial center depth: {}", initial_center_depth);

    // Step simulation
    for _ in 0..50 {
        world.update_water_flow(0.016);
    }

    let final_center_depth = world.water_depth(16, 16);
    println!("Final center depth: {}", final_center_depth);

    // Water should have spread out - center should be lower
    assert!(
        final_center_depth < initial_center_depth * 0.5,
        "Water should have spread from center. Initial: {}, Final: {}",
        initial_center_depth,
        final_center_depth
    );

    // Neighboring cells should have some water now
    let neighbor_depth = world.water_depth(17, 16);
    println!("Neighbor depth: {}", neighbor_depth);
    assert!(
        neighbor_depth > 0.01,
        "Neighboring cell should have received water: {}",
        neighbor_depth
    );
}

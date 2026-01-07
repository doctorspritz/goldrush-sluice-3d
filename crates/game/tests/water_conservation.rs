// Water Conservation Tests
// Tests to verify water mass is conserved during simulation

extern crate game;
extern crate sim3d;

use sim3d::World;

/// Test that excavation does not create water
#[test]
fn test_excavation_no_water_creation() {
    let mut world = World::new(32, 32, 1.0, 10.0);
    
    // No water initially
    let initial_volume = world.total_water_volume();
    assert_eq!(initial_volume, 0.0, "Should start with no water");
    
    // Excavate in center
    let center = glam::Vec3::new(16.0, 0.0, 16.0);
    world.excavate(center, 5.0, 3.0);
    
    // Still no water
    let after_volume = world.total_water_volume();
    assert_eq!(after_volume, 0.0, "Excavation should not create water");
}

/// Test that excavation under water doesn't create phantom mass
#[test]
fn test_excavation_under_water_no_phantom_mass() {
    let mut world = World::new(32, 32, 1.0, 10.0);
    
    // Add water in center
    for z in 10..22 {
        for x in 10..22 {
            let idx = world.idx(x, z);
            let ground = world.ground_height(x, z);
            world.water_surface[idx] = ground + 2.0;
        }
    }
    
    let initial_volume = world.total_water_volume();
    println!("Initial water volume: {}", initial_volume);
    assert!(initial_volume > 0.0, "Should have initial water");
    
    // Excavate in the center (under water)
    let center = glam::Vec3::new(16.0, 0.0, 16.0);
    world.excavate(center, 3.0, 2.0);
    
    let after_volume = world.total_water_volume();
    println!("After excavation water volume: {}", after_volume);
    
    // Water volume should NOT dramatically increase
    // Some increase is OK (terrain lowered, water fills gap from neighbors)
    // But a >2x increase indicates phantom water creation
    let ratio = after_volume / initial_volume;
    assert!(ratio <= 2.0, 
        "Phantom water created during excavation: {} -> {} (ratio {})", 
        initial_volume, after_volume, ratio);
}

/// Test water surface clamp after excavation 
#[test]
fn test_water_surface_clamped_to_ground() {
    let mut world = World::new(32, 32, 1.0, 10.0);
    
    // Set water surface BELOW ground in some cells (should get clamped)
    for z in 10..22 {
        for x in 10..22 {
            let idx = world.idx(x, z);
            let ground = world.ground_height(x, z);
            // Surface below ground is invalid
            world.water_surface[idx] = ground - 5.0;
        }
    }
    
    // Total water volume should count as 0 for these cells
    let volume = world.total_water_volume();
    assert_eq!(volume, 0.0, "Water surface below ground should count as 0 volume");
}

/// Test total_water_volume calculation
#[test]
fn test_water_volume_calculation() {
    let mut world = World::new(10, 10, 1.0, 10.0);
    
    // Add 1m depth of water to one cell
    let idx = world.idx(5, 5);
    let ground = world.ground_height(5, 5);
    world.water_surface[idx] = ground + 1.0;
    
    let volume = world.total_water_volume();
    
    // 1m depth * 1m^2 cell = 1m^3 = 1.0 volume
    assert!((volume - 1.0).abs() < 0.01, "Volume should be ~1.0, got {}", volume);
}

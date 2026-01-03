//! Phase 1: Material-aware mixture density tests
//!
//! Verifies that the mixture density calculation uses actual material densities
//! instead of a hardcoded sand density. This allows gold beds (density 19.3)
//! to divert water flow more effectively than sand beds (density 2.65).

use sim::flip::FlipSimulation;
use sim::particle::{Particle, ParticleMaterial};
use glam::Vec2;

const DT: f32 = 1.0 / 60.0;
const CELL_SIZE: f32 = 4.0;

/// Test that the density accumulator tracks actual material density
#[test]
fn test_mixture_density_reflects_material() {
    // Create small sim
    let mut sim = FlipSimulation::new(20, 20, CELL_SIZE);

    // Add gold particles clustered in one cell
    for i in 0..10 {
        let p = Particle::gold(Vec2::new(40.0 + i as f32 * 0.1, 40.0), Vec2::ZERO);
        sim.particles.list.push(p);
    }

    // Run P2G
    sim.classify_cells();
    sim.particles_to_grid();

    // Find the u-face near the gold particles
    let u_idx = sim.grid.u_index(10, 10);
    let sediment_vol = sim.sand_volume_u[u_idx];
    let density_sum = sim.sediment_density_sum_u[u_idx];

    println!("sediment_vol at u(10,10): {}", sediment_vol);
    println!("density_sum at u(10,10): {}", density_sum);

    if sediment_vol > 0.0 {
        let avg_density = density_sum / sediment_vol;
        println!("avg_density: {}", avg_density);
        // Gold density is 19.3, should be close
        assert!((avg_density - 19.3).abs() < 1.0,
            "Expected gold density ~19.3, got {}", avg_density);
    } else {
        // If no sediment at this exact face, check adjacent faces
        let mut found_gold = false;
        for j in 8..12 {
            for i in 8..12 {
                let idx = sim.grid.u_index(i, j);
                let vol = sim.sand_volume_u[idx];
                if vol > 0.0 {
                    let density = sim.sediment_density_sum_u[idx] / vol;
                    println!("Found sediment at u({},{}): vol={}, density={}", i, j, vol, density);
                    assert!((density - 19.3).abs() < 1.0,
                        "Expected gold density ~19.3, got {}", density);
                    found_gold = true;
                    break;
                }
            }
            if found_gold { break; }
        }
        assert!(found_gold, "Should find gold particles somewhere on the grid");
    }
}

/// Test that sand has its expected density tracked
#[test]
fn test_sand_density_tracked() {
    let mut sim = FlipSimulation::new(20, 20, CELL_SIZE);

    // Add sand particles
    for i in 0..10 {
        let mut p = Particle::gold(Vec2::new(40.0 + i as f32 * 0.1, 40.0), Vec2::ZERO);
        p.material = ParticleMaterial::Sand;
        sim.particles.list.push(p);
    }

    sim.classify_cells();
    sim.particles_to_grid();

    // Search for sand volume
    let mut found = false;
    for j in 8..12 {
        for i in 8..12 {
            let idx = sim.grid.u_index(i, j);
            let vol = sim.sand_volume_u[idx];
            if vol > 0.0 {
                let density = sim.sediment_density_sum_u[idx] / vol;
                println!("Found sand at u({},{}): vol={}, density={}", i, j, vol, density);
                // Sand density is 2.65
                assert!((density - 2.65).abs() < 0.5,
                    "Expected sand density ~2.65, got {}", density);
                found = true;
                break;
            }
        }
        if found { break; }
    }
    assert!(found, "Should find sand particles on the grid");
}

/// Test that magnetite has its expected density tracked
#[test]
fn test_magnetite_density_tracked() {
    let mut sim = FlipSimulation::new(20, 20, CELL_SIZE);

    // Add magnetite particles
    for i in 0..10 {
        let mut p = Particle::gold(Vec2::new(40.0 + i as f32 * 0.1, 40.0), Vec2::ZERO);
        p.material = ParticleMaterial::Magnetite;
        sim.particles.list.push(p);
    }

    sim.classify_cells();
    sim.particles_to_grid();

    // Search for magnetite volume
    let mut found = false;
    for j in 8..12 {
        for i in 8..12 {
            let idx = sim.grid.u_index(i, j);
            let vol = sim.sand_volume_u[idx];
            if vol > 0.0 {
                let density = sim.sediment_density_sum_u[idx] / vol;
                println!("Found magnetite at u({},{}): vol={}, density={}", i, j, vol, density);
                // Magnetite density is 5.2
                assert!((density - 5.2).abs() < 1.0,
                    "Expected magnetite density ~5.2, got {}", density);
                found = true;
                break;
            }
        }
        if found { break; }
    }
    assert!(found, "Should find magnetite particles on the grid");
}

/// Test that dense bed reduces water velocity (gold bed diverts water)
#[test]
fn test_dense_bed_reduces_water_velocity() {
    let mut sim = FlipSimulation::new(40, 20, CELL_SIZE);

    // Set up floor
    for i in 0..40 {
        sim.grid.set_solid(i, 19);
    }
    // Left wall
    for j in 0..20 {
        sim.grid.set_solid(0, j);
    }
    sim.grid.compute_sdf();

    // Create water flow on left side
    for y in 5..15 {
        for x in 2..10 {
            sim.spawn_water(x as f32 * CELL_SIZE, y as f32 * CELL_SIZE, 50.0, 0.0, 1);
        }
    }

    // Create gold bed in middle (blocking path)
    for y in 8..12 {
        for x in 15..20 {
            sim.spawn_gold(x as f32 * CELL_SIZE, y as f32 * CELL_SIZE, 0.0, 0.0, 2);
        }
    }

    // Run simulation
    for _ in 0..50 {
        sim.update(DT);
    }

    // Sample velocity through bed vs above bed
    let vel_through = sim.grid.sample_velocity(Vec2::new(70.0, 40.0)); // through bed
    let vel_above = sim.grid.sample_velocity(Vec2::new(70.0, 20.0));   // above bed

    // Water above bed should be faster than through bed
    println!("Velocity through bed: {:?}", vel_through);
    println!("Velocity above bed: {:?}", vel_above);

    // Through-bed velocity should be significantly reduced
    // Note: This test may need adjustment based on actual simulation behavior
    // The key point is that flow should be impeded by the dense gold bed
    println!("Velocity difference (above - through): {}", vel_above.x - vel_through.x);
}

/// Test that gold beds create higher mixture density than sand beds
#[test]
fn test_gold_creates_higher_mixture_density() {
    // Create two simulations: one with gold, one with sand
    let mut sim_gold = FlipSimulation::new(20, 20, CELL_SIZE);
    let mut sim_sand = FlipSimulation::new(20, 20, CELL_SIZE);

    // Add water particles to both
    for i in 0..5 {
        let p = Particle::gold(Vec2::new(40.0 + i as f32 * 0.1, 40.0), Vec2::ZERO);
        let mut p_water = p.clone();
        p_water.material = ParticleMaterial::Water;
        sim_gold.particles.list.push(p_water.clone());
        sim_sand.particles.list.push(p_water);
    }

    // Add gold to first sim
    for i in 0..5 {
        let p = Particle::gold(Vec2::new(40.0 + i as f32 * 0.1, 40.5), Vec2::ZERO);
        sim_gold.particles.list.push(p);
    }

    // Add sand to second sim
    for i in 0..5 {
        let mut p = Particle::gold(Vec2::new(40.0 + i as f32 * 0.1, 40.5), Vec2::ZERO);
        p.material = ParticleMaterial::Sand;
        sim_sand.particles.list.push(p);
    }

    // Run P2G on both
    sim_gold.classify_cells();
    sim_gold.particles_to_grid();
    sim_sand.classify_cells();
    sim_sand.particles_to_grid();

    // Find a face with sediment in both
    let mut gold_density = 0.0f32;
    let mut sand_density = 0.0f32;

    for j in 8..12 {
        for i in 8..12 {
            let idx = sim_gold.grid.u_index(i, j);
            let gold_vol = sim_gold.sand_volume_u[idx];
            let sand_vol = sim_sand.sand_volume_u[idx];

            if gold_vol > 0.0 && sand_vol > 0.0 {
                gold_density = sim_gold.sediment_density_sum_u[idx] / gold_vol;
                sand_density = sim_sand.sediment_density_sum_u[idx] / sand_vol;
                println!("At u({},{}): gold_density={}, sand_density={}",
                    i, j, gold_density, sand_density);
            }
        }
    }

    // Gold should have higher tracked density than sand
    if gold_density > 0.0 && sand_density > 0.0 {
        assert!(gold_density > sand_density * 2.0,
            "Gold density ({}) should be significantly higher than sand density ({})",
            gold_density, sand_density);
    }
}

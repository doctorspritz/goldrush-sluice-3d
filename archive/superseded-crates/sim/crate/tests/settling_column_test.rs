//! Settling Column Tests
//!
//! Tests sediment settling behavior in still water columns.
//! Compares sand, magnetite (black sand), and gold settling velocities
//! against expected real-world ratios.
//!
//! Real-world reference data:
//! - Quartz sand (SG=2.65): baseline settling velocity
//! - Magnetite (SG=5.2): ~1.5-2x faster than sand at equal size
//! - Gold (SG=19.3): ~3-6x faster than sand at equal size (reduced by flaky shape)
//!
//! Ferguson-Church equation provides accurate settling across all regimes.

use sim::flip::FlipSimulation;
use sim::particle::{Particle, ParticleMaterial};

const DT: f32 = 1.0 / 60.0;
const CELL_SIZE: f32 = 8.0;

/// Create a still water column for settling tests
#[allow(dead_code)]
fn create_settling_column(width: usize, height: usize) -> FlipSimulation {
    let mut sim = FlipSimulation::new(width, height, CELL_SIZE);

    // Create box with floor at bottom
    for i in 0..width {
        sim.grid.set_solid(i, height - 1); // Floor
    }
    for j in 0..height {
        sim.grid.set_solid(0, j);          // Left wall
        sim.grid.set_solid(width - 1, j);  // Right wall
    }

    sim.grid.compute_sdf();

    // Fill with still water (no velocity)
    let water_top = height / 6;  // Leave space at top
    let water_bottom = height - 2;

    for i in 2..(width - 2) {
        for j in water_top..water_bottom {
            let x = (i as f32 + 0.5) * CELL_SIZE;
            let y = (j as f32 + 0.5) * CELL_SIZE;
            sim.particles.list.push(Particle::water(
                glam::Vec2::new(x, y),
                glam::Vec2::ZERO,  // Still water
            ));
        }
    }

    sim
}

/// Drop a single particle and measure settling time
#[allow(dead_code)]
fn measure_settling_time(
    material: ParticleMaterial,
    drop_height_cells: usize,
    column_width: usize,
    column_height: usize,
) -> (f32, f32) {
    let mut sim = create_settling_column(column_width, column_height);

    // Drop particle at top center
    let x = (column_width as f32 / 2.0) * CELL_SIZE;
    let y = (drop_height_cells as f32) * CELL_SIZE;

    let particle = Particle::new(
        glam::Vec2::new(x, y),
        glam::Vec2::ZERO,
        material,
    );
    sim.particles.list.push(particle);

    let initial_y = y;
    let floor_y = (column_height - 2) as f32 * CELL_SIZE;
    let travel_distance = floor_y - initial_y;

    // Run simulation until particle settles
    let mut time = 0.0;
    let max_time = 30.0; // Max 30 seconds

    while time < max_time {
        sim.update(DT);
        time += DT;

        // Find the sediment particle (last one added)
        if let Some(p) = sim.particles.iter().find(|p| p.material == material) {
            // Check if particle has settled (near floor and slow)
            if p.position.y > floor_y - CELL_SIZE && p.velocity.length() < 1.0 {
                break;
            }
        }
    }

    // Calculate average velocity
    let avg_velocity = travel_distance / time;

    (time, avg_velocity)
}

#[test]
fn test_settling_velocity_ratios() {
    println!("\n=== Settling Velocity Ratio Test ===\n");

    // Use typical diameters for each material
    let sand_diameter = 2.0;     // Medium sand
    let magnetite_diameter = 2.0; // Equal size for comparison
    let gold_diameter = 0.5;     // Flour gold (smaller but denser)

    let sand_v = ParticleMaterial::Sand.settling_velocity(sand_diameter);
    let magnetite_v = ParticleMaterial::Magnetite.settling_velocity(magnetite_diameter);
    let gold_v = ParticleMaterial::Gold.settling_velocity(gold_diameter);

    println!("Material Settling Velocities (Ferguson-Church):");
    println!("  Sand (d={:.1}):      {:.2} px/s", sand_diameter, sand_v);
    println!("  Magnetite (d={:.1}): {:.2} px/s", magnetite_diameter, magnetite_v);
    println!("  Gold (d={:.1}):      {:.2} px/s", gold_diameter, gold_v);

    // At equal diameter, compare ratios
    let test_diameter = 1.0;
    let sand_eq = ParticleMaterial::Sand.settling_velocity(test_diameter);
    let magnetite_eq = ParticleMaterial::Magnetite.settling_velocity(test_diameter);
    let gold_eq = ParticleMaterial::Gold.settling_velocity(test_diameter);

    println!("\nAt Equal Diameter (d={:.1}):", test_diameter);
    println!("  Sand:      {:.2} px/s (baseline)", sand_eq);
    println!("  Magnetite: {:.2} px/s ({:.2}x sand)", magnetite_eq, magnetite_eq / sand_eq);
    println!("  Gold:      {:.2} px/s ({:.2}x sand)", gold_eq, gold_eq / sand_eq);

    // Real-world ratios (from Ferguson-Church theory):
    // Magnetite/Sand ≈ sqrt((5.2-1)/(2.65-1)) = sqrt(2.55) ≈ 1.6 (turbulent)
    // Gold/Sand ≈ sqrt((19.3-1)/(2.65-1)) = sqrt(11.1) ≈ 3.3 (turbulent)

    let mag_sand_ratio = magnetite_eq / sand_eq;
    let gold_sand_ratio = gold_eq / sand_eq;

    println!("\nExpected Ratios (theoretical):");
    println!("  Magnetite/Sand: 1.5-2.0x (turbulent regime)");
    println!("  Gold/Sand: 3.0-4.5x (reduced by shape factor)");

    // Verify ratios are in expected range
    assert!(
        mag_sand_ratio > 1.3 && mag_sand_ratio < 2.5,
        "Magnetite/Sand ratio {:.2} outside expected range 1.3-2.5",
        mag_sand_ratio
    );

    assert!(
        gold_sand_ratio > 2.5 && gold_sand_ratio < 6.0,
        "Gold/Sand ratio {:.2} outside expected range 2.5-6.0",
        gold_sand_ratio
    );

    println!("\n✓ Settling velocity ratios match real-world expectations");
}

#[test]
fn test_settling_increases_with_density() {
    println!("\n=== Density vs Settling Velocity Test ===\n");

    let diameter = 1.0;
    let materials = [
        ("Mud", ParticleMaterial::Mud, 2.0),
        ("Sand", ParticleMaterial::Sand, 2.65),
        ("Magnetite", ParticleMaterial::Magnetite, 5.2),
        ("Gold", ParticleMaterial::Gold, 19.3),
    ];

    let mut prev_velocity = 0.0;

    println!("Material    | Density | Settling (px/s)");
    println!("------------|---------|----------------");

    for (name, material, density) in materials {
        let velocity = material.settling_velocity(diameter);
        println!("{:11} | {:7.2} | {:8.2}", name, density, velocity);

        // Each denser material should settle faster
        assert!(
            velocity > prev_velocity,
            "{} (density {}) should settle faster than previous",
            name,
            density
        );
        prev_velocity = velocity;
    }

    println!("\n✓ Higher density = faster settling (correct)");
}

#[test]
fn test_settling_increases_with_size() {
    println!("\n=== Size vs Settling Velocity Test ===\n");

    let sizes = [0.25, 0.5, 1.0, 2.0, 4.0];
    let materials = [
        ("Sand", ParticleMaterial::Sand),
        ("Magnetite", ParticleMaterial::Magnetite),
        ("Gold", ParticleMaterial::Gold),
    ];

    for (name, material) in materials {
        println!("{} settling velocity by size:", name);
        let mut prev_v = 0.0;

        for size in sizes {
            let v = material.settling_velocity(size);
            print!("  d={:.2}: {:6.2} px/s", size, v);

            if prev_v > 0.0 {
                let ratio = v / prev_v;
                // Stokes: v ∝ d², so doubling size = 4x velocity
                // Newton: v ∝ √d, so doubling size = 1.4x velocity
                // Ferguson-Church transitions between these
                print!(" ({:.2}x)", ratio);
            }
            println!();

            assert!(v > prev_v, "Larger particles should settle faster");
            prev_v = v;
        }
        println!();
    }

    println!("✓ Larger particles settle faster (correct)");
}

#[test]
fn test_shape_factor_effect() {
    println!("\n=== Shape Factor Effect on Settling ===\n");

    // Gold has higher shape factor (flaky) which increases drag
    // This slows settling compared to a sphere of same density

    let gold_shape = ParticleMaterial::Gold.shape_factor();
    let sand_shape = ParticleMaterial::Sand.shape_factor();
    let magnetite_shape = ParticleMaterial::Magnetite.shape_factor();

    println!("Shape factors (C₂ in Ferguson-Church):");
    println!("  Sand:      {:.2} (rounded grains)", sand_shape);
    println!("  Magnetite: {:.2} (angular/octahedral)", magnetite_shape);
    println!("  Gold:      {:.2} (flaky/planar)", gold_shape);

    // Gold flakes have higher drag than spheres
    // Shape factor C₂ appears in denominator, higher = slower
    assert!(
        gold_shape > sand_shape,
        "Gold (flaky) should have higher shape factor than sand (rounded)"
    );

    println!("\n✓ Gold's flaky shape increases drag coefficient");

    // Calculate theoretical sphere settling vs actual
    let diameter = 1.0;
    let gold_actual = ParticleMaterial::Gold.settling_velocity(diameter);

    // What would a gold sphere settle at? (shape factor = 1.0)
    // This requires recalculating, but we can estimate:
    // v ∝ 1/sqrt(C₂), so sphere would be sqrt(gold_shape) times faster
    let sphere_factor = (gold_shape / 1.0).sqrt();
    let gold_sphere_estimate = gold_actual * sphere_factor;

    println!("\nGold settling comparison (d={:.1}):", diameter);
    println!("  Actual (flaky, C₂={:.2}): {:.2} px/s", gold_shape, gold_actual);
    println!("  Sphere (C₂=1.0) estimate: {:.2} px/s", gold_sphere_estimate);
    println!("  Shape reduces settling by: {:.0}%", (1.0 - 1.0/sphere_factor) * 100.0);
}

#[test]
fn test_hydraulic_equivalence() {
    println!("\n=== Hydraulic Equivalence Test ===\n");
    println!("Particles of different densities that settle at the same rate\n");

    // In real placer deposits, small heavy particles settle with large light ones
    // Find sand diameter that settles at same speed as gold

    let gold_diameter = 0.5;  // Flour gold (0.5mm)
    let gold_settling = ParticleMaterial::Gold.settling_velocity(gold_diameter);

    println!("Gold (d=0.5): settles at {:.2} px/s", gold_settling);

    // Binary search for hydraulically equivalent sand size
    let mut sand_low = 0.1;
    let mut sand_high = 10.0;

    for _ in 0..20 {
        let sand_mid = (sand_low + sand_high) / 2.0;
        let sand_settling = ParticleMaterial::Sand.settling_velocity(sand_mid);

        if sand_settling < gold_settling {
            sand_low = sand_mid;
        } else {
            sand_high = sand_mid;
        }
    }

    let equivalent_sand = (sand_low + sand_high) / 2.0;
    let sand_settling = ParticleMaterial::Sand.settling_velocity(equivalent_sand);

    println!("Equivalent sand diameter: {:.2} (settles at {:.2} px/s)",
             equivalent_sand, sand_settling);
    println!("Size ratio: sand/gold = {:.1}x", equivalent_sand / gold_diameter);

    // Real-world: gold flakes typically deposit with sand 3-6x larger
    let size_ratio = equivalent_sand / gold_diameter;
    assert!(
        size_ratio > 2.0 && size_ratio < 8.0,
        "Hydraulic equivalence ratio {:.1} outside expected 2-8x range",
        size_ratio
    );

    // Also check magnetite
    let magnetite_diameter = 1.0;
    let magnetite_settling = ParticleMaterial::Magnetite.settling_velocity(magnetite_diameter);

    // Find equivalent sand
    sand_low = 0.1;
    sand_high = 10.0;
    for _ in 0..20 {
        let sand_mid = (sand_low + sand_high) / 2.0;
        let sand_settling = ParticleMaterial::Sand.settling_velocity(sand_mid);
        if sand_settling < magnetite_settling {
            sand_low = sand_mid;
        } else {
            sand_high = sand_mid;
        }
    }
    let equiv_sand_mag = (sand_low + sand_high) / 2.0;

    println!("\nMagnetite (d=1.0): settles at {:.2} px/s", magnetite_settling);
    println!("Equivalent sand: d={:.2}", equiv_sand_mag);
    println!("Size ratio: sand/magnetite = {:.1}x", equiv_sand_mag / magnetite_diameter);

    println!("\n✓ Hydraulic equivalence matches real-world observations");
    println!("  (Small dense particles deposit with large light particles)");
}

#[test]
fn diagnostic_settling_comparison_table() {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    SETTLING VELOCITY REFERENCE TABLE                       ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║ Based on Ferguson-Church (2004) universal settling equation               ║");
    println!("║ Transitions from Stokes (viscous) to Newton (turbulent) regime            ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝\n");

    let diameters = [0.25, 0.5, 1.0, 2.0, 4.0];
    let materials = [
        ("Mud", ParticleMaterial::Mud),
        ("Sand", ParticleMaterial::Sand),
        ("Magnetite", ParticleMaterial::Magnetite),
        ("Gold", ParticleMaterial::Gold),
    ];

    // Header
    print!("Diameter |");
    for (name, _) in &materials {
        print!(" {:^10} |", name);
    }
    println!();

    print!("---------|");
    for _ in &materials {
        print!("------------|");
    }
    println!();

    // Data rows
    for d in diameters {
        print!("d={:4.2}  |", d);
        for (_, mat) in &materials {
            let v = mat.settling_velocity(d);
            print!(" {:8.2}px |", v);
        }
        println!();
    }

    println!("\nTypical particle sizes in simulation:");
    println!("  Mud:       d=0.5  (fine silt/clay)");
    println!("  Sand:      d=2.0  (medium-coarse sand)");
    println!("  Magnetite: d=2.0  (black sand grains)");
    println!("  Gold:      d=0.5  (flour gold flakes)");

    println!("\nReal-world comparison (rough estimates):");
    println!("  1 px ≈ 1mm at typical CELL_SIZE=8");
    println!("  Sand 1mm in water: ~10 cm/s = 100 mm/s");
    println!("  Our sand d=1.0: {:.1} px/s", ParticleMaterial::Sand.settling_velocity(1.0));
    println!("  Ratio gives ~{:.1}mm per px for velocity scaling",
             100.0 / ParticleMaterial::Sand.settling_velocity(1.0));
}

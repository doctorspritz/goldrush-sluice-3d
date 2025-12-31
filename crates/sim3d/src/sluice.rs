//! 3D Sluice box geometry generation.
//!
//! Creates a sloped channel with riffles for sediment trapping.

use crate::FlipSimulation3D;

/// Configuration for sluice box geometry.
#[derive(Clone, Debug)]
pub struct SluiceConfig {
    /// Floor slope: rise per cell in X direction (0.0 = flat, 0.1 = 10% grade)
    pub slope: f32,
    /// Cells before first riffle (flat inlet section)
    pub slick_plate_len: usize,
    /// Spacing between riffles in cells
    pub riffle_spacing: usize,
    /// Height of riffles in cells
    pub riffle_height: usize,
    /// Width/thickness of riffles in cells (along X)
    pub riffle_width: usize,
}

impl Default for SluiceConfig {
    fn default() -> Self {
        Self {
            slope: 0.15,           // ~8.5 degree slope
            slick_plate_len: 8,    // Flat inlet section
            riffle_spacing: 6,     // Riffle every 6 cells
            riffle_height: 2,      // 2 cells tall
            riffle_width: 1,       // 1 cell thick
        }
    }
}

/// Create a sluice box in the simulation.
///
/// Layout (side view, X is flow direction, Y is up):
/// ```text
///                    outlet (open)
///     ___________       |
///    /   |   |   \      v
///   /====|===|====\========>
///   ^    riffles     floor slope
///   |
/// inlet
/// ```
///
/// Top view (X is flow, Z is width):
/// ```text
/// wall +-----------------+
///      |  ==  ==  ==     | <- riffles (bars across Z)
/// Z    |  ==  ==  ==     |
///      |  ==  ==  ==     |
/// wall +-----------------+
///      inlet        outlet
///            X ->
/// ```
pub fn create_sluice(sim: &mut FlipSimulation3D, config: &SluiceConfig) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    let depth = sim.grid.depth;
    let dx = sim.grid.cell_size;

    // Clear existing solids
    sim.grid.clear_solids();

    // 1. Create floor with slope
    // Floor height increases as X increases (water flows downhill from high X to low X,
    // or we tilt the other way: water enters at low X, exits at high X going downhill)
    // Let's do: floor is higher at X=0 (inlet), lower at X=width (outlet)
    // So water flows in +X direction, downhill
    for i in 0..width {
        // Floor height at this X position (in cells, from bottom)
        let floor_y = ((width - 1 - i) as f32 * config.slope) as usize;
        let floor_y = floor_y.min(height - 3); // Leave room for water

        // Fill floor cells
        for k in 0..depth {
            for j in 0..=floor_y {
                sim.grid.set_solid(i, j, k);
            }
        }
    }

    // 2. Create side walls (z=0 and z=depth-1)
    for i in 0..width {
        for j in 0..height {
            sim.grid.set_solid(i, j, 0);
            sim.grid.set_solid(i, j, depth - 1);
        }
    }

    // 3. Create left wall (inlet side) - but leave opening for water entry
    let inlet_bottom = ((width - 1) as f32 * config.slope) as usize + 1;
    let inlet_top = inlet_bottom + height / 3; // Opening is 1/3 of height
    for k in 0..depth {
        for j in 0..height {
            // Wall except for inlet opening
            if j < inlet_bottom || j > inlet_top {
                sim.grid.set_solid(0, j, k);
            }
        }
    }

    // 4. Create riffles
    // Start after slick plate, repeat every riffle_spacing cells
    let mut riffle_x = config.slick_plate_len;
    while riffle_x + config.riffle_width < width - 2 {
        // Floor height at riffle position
        let floor_y = ((width - 1 - riffle_x) as f32 * config.slope) as usize;

        // Riffle is a bar extending across Z (the channel width)
        for dx_offset in 0..config.riffle_width {
            let x = riffle_x + dx_offset;
            if x >= width {
                break;
            }

            for k in 1..depth - 1 {
                // Skip side walls
                for dy in 0..config.riffle_height {
                    let y = floor_y + 1 + dy;
                    if y < height {
                        sim.grid.set_solid(x, y, k);
                    }
                }
            }
        }

        riffle_x += config.riffle_spacing;
    }

    // 5. Compute SDF for smooth collision
    sim.grid.compute_sdf();
}

/// Spawn water at the inlet with given velocity.
pub fn spawn_inlet_water(sim: &mut FlipSimulation3D, count: usize, velocity: glam::Vec3) {
    let dx = sim.grid.cell_size;
    let width = sim.grid.width;
    let depth = sim.grid.depth;

    // Inlet position: near X=0, in the opening
    let inlet_bottom = ((width - 1) as f32 * 0.15) as usize + 1; // Match default slope
    let inlet_x = 2.0 * dx; // Just inside inlet

    // Spawn particles in a grid pattern at inlet
    let particles_per_side = (count as f32).cbrt().ceil() as usize;
    let spacing = dx * 0.4;

    let mut spawned = 0;
    for pi in 0..particles_per_side {
        for pj in 0..particles_per_side {
            for pk in 0..particles_per_side {
                if spawned >= count {
                    return;
                }

                let pos = glam::Vec3::new(
                    inlet_x + pi as f32 * spacing,
                    (inlet_bottom as f32 + 1.0 + pj as f32 * 0.5) * dx,
                    (1.0 + pk as f32 * (depth as f32 - 2.0) / particles_per_side as f32) * dx,
                );

                // Check if position is valid (not inside solid)
                if sim.grid.sample_sdf(pos) > 0.0 {
                    sim.spawn_particle_with_velocity(pos, velocity);
                    spawned += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sluice_creation() {
        let mut sim = FlipSimulation3D::new(32, 16, 8, 0.1);
        let config = SluiceConfig::default();
        create_sluice(&mut sim, &config);

        // Check that some cells are solid (floor exists)
        let mut solid_count = 0;
        for idx in 0..sim.grid.solid.len() {
            if sim.grid.solid[idx] {
                solid_count += 1;
            }
        }
        assert!(solid_count > 0, "Sluice should have solid cells");

        // Check that SDF was computed
        let center = glam::Vec3::new(1.6, 0.8, 0.4); // Should be above floor
        let sdf = sim.grid.sample_sdf(center);
        // This might be inside or outside depending on geometry
        assert!(sdf.is_finite(), "SDF should be finite");
    }

    #[test]
    fn test_inlet_spawning() {
        let mut sim = FlipSimulation3D::new(32, 16, 8, 0.1);
        let config = SluiceConfig::default();
        create_sluice(&mut sim, &config);

        let initial_count = sim.particle_count();
        spawn_inlet_water(&mut sim, 100, glam::Vec3::new(1.0, 0.0, 0.0));

        assert!(
            sim.particle_count() > initial_count,
            "Should have spawned particles"
        );
    }
}

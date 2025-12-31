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
pub fn create_sluice(sim: &mut FlipSimulation3D, _config: &SluiceConfig) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    let depth = sim.grid.depth;

    // Clear existing solids
    sim.grid.clear_solids();

    // Simple box: floor + 4 walls (closed), inlet at top of one wall
    let wall_height = height * 2 / 3; // Walls go 2/3 up

    // Floor (y=0)
    for i in 0..width {
        for k in 0..depth {
            sim.grid.set_solid(i, 0, k);
        }
    }

    // All 4 walls - completely solid (no exit)
    for j in 1..wall_height {
        // Left wall (x=0)
        for k in 0..depth {
            sim.grid.set_solid(0, j, k);
        }
        // Right wall (x=width-1) - BLOCKED
        for k in 0..depth {
            sim.grid.set_solid(width - 1, j, k);
        }
        // Front wall (z=0)
        for i in 0..width {
            sim.grid.set_solid(i, j, 0);
        }
        // Back wall (z=depth-1)
        for i in 0..width {
            sim.grid.set_solid(i, j, depth - 1);
        }
    }

    // Compute SDF
    sim.grid.compute_sdf();
}

/// Original sluice with slope - for later
pub fn create_sluice_sloped(sim: &mut FlipSimulation3D, config: &SluiceConfig) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    let depth = sim.grid.depth;

    sim.grid.clear_solids();

    // Floor with slope
    for i in 0..width {
        let floor_y = ((width - 1 - i) as f32 * config.slope) as usize;
        let floor_y = floor_y.min(height - 3);
        for k in 0..depth {
            for j in 0..=floor_y {
                sim.grid.set_solid(i, j, k);
            }
        }
    }

    // Side walls
    let wall_height = 6;
    for i in 0..width {
        let floor_y = ((width - 1 - i) as f32 * config.slope) as usize;
        for dy in 0..wall_height {
            let j = floor_y + 1 + dy;
            if j < height {
                sim.grid.set_solid(i, j, 0);
                sim.grid.set_solid(i, j, depth - 1);
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
    let height = sim.grid.height;
    let depth = sim.grid.depth;

    // Spawn above the box (walls are 2/3 height, spawn at 3/4 height)
    let spawn_y = (height as f32 * 0.75) * dx;
    let spawn_x = (width as f32 * 0.5) * dx; // Center of box
    let spawn_z = (depth as f32 * 0.5) * dx; // Center of box

    // Spawn particles in a grid pattern
    let particles_per_side = (count as f32).cbrt().ceil() as usize;
    let spacing = dx * 0.4;

    let mut spawned = 0;
    for pi in 0..particles_per_side {
        for pj in 0..particles_per_side {
            for pk in 0..particles_per_side {
                if spawned >= count {
                    return;
                }

                // Center the spawn block around spawn_x, spawn_z
                let offset_i = pi as f32 - particles_per_side as f32 / 2.0;
                let offset_k = pk as f32 - particles_per_side as f32 / 2.0;

                let pos = glam::Vec3::new(
                    spawn_x + offset_i * spacing,
                    spawn_y + pj as f32 * spacing,
                    spawn_z + offset_k * spacing,
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

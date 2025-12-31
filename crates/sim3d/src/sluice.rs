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
/// inlet (high)              outlet (low, open)
///     |                          |
///     v                          v
///   wall  ___________________
///     |  /   |   |   |       \---> water exits
///     | /====|===|===|========\
///       ^    riffles     floor slope
/// ```
///
/// Top view (X is flow, Z is width):
/// ```text
/// wall +-----------------+ open
///      |  ==  ==  ==     |---->
/// Z    |  ==  ==  ==     | exit
///      |  ==  ==  ==     |
/// wall +-----------------+
///      inlet        outlet
///            X ->
/// ```
pub fn create_sluice(sim: &mut FlipSimulation3D, config: &SluiceConfig) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    let depth = sim.grid.depth;

    sim.grid.clear_solids();

    // 1. Sloped floor: high at x=0 (inlet), low at x=width-1 (outlet)
    // floor_y = (width - 1 - i) * slope, so at i=0 it's highest, at i=width-1 it's 0
    for i in 0..width {
        let floor_y = ((width - 1 - i) as f32 * config.slope) as usize;
        let floor_y = floor_y.min(height - 4); // Leave room for walls and water
        for k in 0..depth {
            for j in 0..=floor_y {
                sim.grid.set_solid(i, j, k);
            }
        }
    }

    // 2. Side walls (z=0 and z=depth-1) - contain water laterally
    let wall_height = 5; // Cells above floor
    for i in 0..width {
        let floor_y = ((width - 1 - i) as f32 * config.slope) as usize;
        let floor_y = floor_y.min(height - 4);
        for dy in 1..=wall_height {
            let j = floor_y + dy;
            if j < height {
                sim.grid.set_solid(i, j, 0);          // Front wall
                sim.grid.set_solid(i, j, depth - 1);  // Back wall
            }
        }
    }

    // 3. Back wall at inlet (x=0) - prevent water escaping backwards
    let inlet_floor_y = ((width - 1) as f32 * config.slope) as usize;
    let inlet_floor_y = inlet_floor_y.min(height - 4);
    for dy in 1..=wall_height {
        let j = inlet_floor_y + dy;
        if j < height {
            for k in 0..depth {
                sim.grid.set_solid(0, j, k);
            }
        }
    }

    // 4. NO wall at outlet (x=width-1) - water exits freely

    // 5. Riffles: bars across Z starting after slick plate
    let mut riffle_x = config.slick_plate_len;
    while riffle_x + config.riffle_width < width - 2 {
        let floor_y = ((width - 1 - riffle_x) as f32 * config.slope) as usize;
        let floor_y = floor_y.min(height - 4);

        for dx in 0..config.riffle_width {
            let x = riffle_x + dx;
            if x >= width {
                break;
            }
            // Riffle spans the interior (not the side walls)
            for k in 1..depth - 1 {
                for dy in 1..=config.riffle_height {
                    let y = floor_y + dy;
                    if y < height {
                        sim.grid.set_solid(x, y, k);
                    }
                }
            }
        }
        riffle_x += config.riffle_spacing;
    }

    // 6. Compute SDF for smooth particle collision
    sim.grid.compute_sdf();
}

/// Spawn water at the inlet (high end of sluice) with given velocity.
///
/// Multiple horizontal emitters span the full width (Z) of the channel.
/// The inlet is at x=0 where the floor is highest. Water spawns just above
/// the floor and flows downhill toward the outlet at x=width-1.
///
/// Top view of emitters:
/// ```text
/// wall +-------------------+
///      | * * * * * * * *   |  <- emitters span Z
/// Z    |                   |
///      |                   |
/// wall +-------------------+
///      inlet          outlet
///            X ->
/// ```
pub fn spawn_inlet_water(sim: &mut FlipSimulation3D, config: &SluiceConfig, count: usize, velocity: glam::Vec3) {
    let dx = sim.grid.cell_size;
    let width = sim.grid.width;
    let depth = sim.grid.depth;

    // Inlet is at x=0, floor height is (width-1) * slope
    let inlet_floor_y = ((width - 1) as f32 * config.slope) as usize;

    // Spawn position: just past inlet wall, above floor
    let spawn_x = 2.0 * dx;
    let spawn_y_base = (inlet_floor_y as f32 + 2.0) * dx;

    // Number of emitters across the width (inside side walls)
    let num_emitters = (depth - 2).max(1); // depth-2 to stay inside walls
    let particles_per_emitter = (count / num_emitters).max(1);

    // Spacing between emitters
    let z_start = 1.5 * dx;  // Just inside front wall
    let z_end = (depth as f32 - 1.5) * dx;  // Just inside back wall
    let z_span = z_end - z_start;

    let mut spawned = 0;
    for emitter in 0..num_emitters {
        // Position this emitter along Z
        let t = if num_emitters > 1 {
            emitter as f32 / (num_emitters - 1) as f32
        } else {
            0.5
        };
        let emitter_z = z_start + t * z_span;

        // Spawn a small cluster at each emitter
        let cluster_size = (particles_per_emitter as f32).sqrt().ceil() as usize;
        let spacing = dx * 0.3;

        for pi in 0..cluster_size {
            for pj in 0..cluster_size {
                if spawned >= count {
                    return;
                }

                let pos = glam::Vec3::new(
                    spawn_x + pi as f32 * spacing,
                    spawn_y_base + pj as f32 * spacing,
                    emitter_z,
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
        spawn_inlet_water(&mut sim, &config, 100, glam::Vec3::new(1.0, 0.0, 0.0));

        assert!(
            sim.particle_count() > initial_count,
            "Should have spawned particles"
        );
    }
}

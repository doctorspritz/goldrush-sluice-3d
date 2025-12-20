//! Sluice box geometry setup
//!
//! Creates the terrain for testing vortex formation:
//! - Sloped floor
//! - Periodic riffles (vertical bars) that create vortices

use crate::flip::FlipSimulation;

/// Create a sluice box test setup
/// - slope: how much the floor rises per cell (0.3 = 17 degrees)
/// - riffle_spacing: cells between riffles
/// - riffle_height: how tall the riffles are (cells above floor)
/// - riffle_width: how wide each riffle is (cells)
pub fn create_sluice(sim: &mut FlipSimulation, slope: f32, riffle_spacing: usize, riffle_height: usize, riffle_width: usize) {
    let width = sim.grid.width;
    let height = sim.grid.height;

    // Clear all solid
    for idx in 0..sim.grid.solid.len() {
        sim.grid.solid[idx] = false;
    }

    // Create sloped floor
    // Water flows LEFT to RIGHT (downhill)
    // Left side: floor surface is HIGH (floor_y is LOW in screen coords)
    // Right side: floor surface is LOW (floor_y is HIGH in screen coords)
    let base_height = height / 4; // Start higher for more room

    for i in 0..width {
        // Floor surface goes DOWN as x increases (water flows right)
        let floor_y = base_height + (i as f32 * slope) as usize;

        // Fill everything below floor_y as solid
        for j in floor_y..height {
            sim.grid.set_solid(i, j);
        }

        // Add riffles (rectangular bars with width)
        // Check if we're within a riffle zone
        let riffle_start = riffle_spacing; // First riffle starts after spacing
        if i >= riffle_start && i < width - riffle_spacing {
            let pos_in_cycle = (i - riffle_start) % riffle_spacing;
            if pos_in_cycle < riffle_width {
                // We're within a riffle
                for dy in 0..riffle_height {
                    let riffle_y = floor_y.saturating_sub(dy + 1);
                    if riffle_y > 0 {
                        sim.grid.set_solid(i, riffle_y);
                    }
                }
            }
        }
    }

    // Add walls on left and right
    for j in 0..height {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(width - 1, j);
    }
}

/// Create a simple flat-bottom test tank with riffles
pub fn create_flat_sluice(sim: &mut FlipSimulation, riffle_spacing: usize, riffle_height: usize) {
    let width = sim.grid.width;
    let height = sim.grid.height;

    // Clear all solid
    for idx in 0..sim.grid.solid.len() {
        sim.grid.solid[idx] = false;
    }

    // Floor at bottom 20%
    let floor_y = height - height / 5;

    for i in 0..width {
        // Floor
        for j in floor_y..height {
            sim.grid.set_solid(i, j);
        }

        // Riffles
        if i > riffle_spacing && i % riffle_spacing == 0 && i < width - riffle_spacing {
            for dy in 0..riffle_height {
                let riffle_y = floor_y.saturating_sub(dy + 1);
                if riffle_y > 0 {
                    sim.grid.set_solid(i, riffle_y);
                }
            }
        }
    }

    // Walls
    for j in 0..height {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(width - 1, j);
    }
}

/// Create a simple box with no riffles for basic fluid testing
pub fn create_box(sim: &mut FlipSimulation) {
    let width = sim.grid.width;
    let height = sim.grid.height;

    // Clear all solid
    for idx in 0..sim.grid.solid.len() {
        sim.grid.solid[idx] = false;
    }

    // Floor
    for i in 0..width {
        sim.grid.set_solid(i, height - 1);
    }

    // Walls
    for j in 0..height {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(width - 1, j);
    }
}

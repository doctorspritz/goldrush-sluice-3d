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
    let slick_plate_len = 50;     // Flat inlet section for flow development

    for i in 0..width {
        // Floor surface calculation
        let floor_y = if i < slick_plate_len {
            // Slick Plate: Flat section
            base_height
        } else {
            // Sloped section: Starts dropping after slick plate
            base_height + ((i - slick_plate_len) as f32 * slope) as usize
        };

        // Fill everything below floor_y as solid
        for j in floor_y..height {
            sim.grid.set_solid(i, j);
        }

        // Add riffles (wedge shaped: ramp upstream, vertical downstream)
        // Start riffles only after the slick plate
        let riffle_start = slick_plate_len + riffle_spacing;
        
        // Check if we are in a riffle zone (periodic)
        if i >= riffle_start && i < width - riffle_spacing {
            let cycle_len = riffle_spacing;
            let rel_x = (i - riffle_start) % cycle_len;

            // Riffle geometry:
            // Gentler ramp (3:1 slope) for better flow adherance
            let ramp_slope_inv = 3; // 3 units run for 1 unit rise
            let ramp_len = riffle_height * ramp_slope_inv; 
            let total_len = ramp_len + riffle_width;

            if rel_x < total_len {
                // Calculate height at this x
                let h = if rel_x < ramp_len {
                    // Ramp section: rise 1 unit every 3 cells
                    (rel_x / ramp_slope_inv) + 1
                } else {
                    // Flat top section
                    riffle_height
                };

                // Fill solid from floor up to height h
                for dy in 0..h {
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

    // Precompute SDF with new geometry
    sim.grid.compute_sdf();
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

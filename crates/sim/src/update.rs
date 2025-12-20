//! Simulation update logic - cellular automata rules.

use crate::chunk::{Chunk, CHUNK_SIZE};
use crate::material::Material;

const GRAVITY: f32 = 0.2;
const TERMINAL_VELOCITY: f32 = 5.0;
const EROSION_THRESHOLD: f32 = 1.0;

/// Update a single chunk for one simulation frame.
///
/// Returns true if any cell moved (chunk is still active).
pub fn update_chunk(chunk: &mut Chunk, frame: u64) -> bool {
    if !chunk.is_active {
        return false;
    }

    chunk.clear_updated();
    
    // FLUID SOLVER STEP
    // Run Navier-Stokes on the grid velocities BEFORE moving particles
    crate::fluid::solve_fluid(chunk);

    let mut any_moved = false;

    // Bottom-to-top iteration (so falling particles process correctly)
    for y in (0..CHUNK_SIZE).rev() {
        // Alternate left-right direction each frame to prevent bias
        let x_range: Box<dyn Iterator<Item = usize>> = if frame & 1 == 0 {
            Box::new(0..CHUNK_SIZE)
        } else {
            Box::new((0..CHUNK_SIZE).rev())
        };

        for x in x_range {
            // Skip if already updated this frame
            if chunk.is_updated(x, y) {
                continue;
            }

            let material = chunk.get_material(x, y);

            let moved = match material {
                Material::Air | Material::Rock => false,
                _ if material.is_powder() => update_powder(chunk, x, y, frame),
                _ if material.is_liquid() => update_liquid(chunk, x, y, frame),
                _ => false,
            };

            if moved {
                any_moved = true;
            } else {
                // Mark as processed even if didn't move
                chunk.mark_updated(x, y);
            }
        }
    }

    chunk.is_active = any_moved;
    any_moved
}

/// Update powder material (Soil, Gold) - falls and slides.
fn update_powder(chunk: &mut Chunk, x: usize, y: usize, frame: u64) -> bool {
    let material = chunk.get_material(x, y);
    let density = material.density();
    let idx = Chunk::index(x, y);

    // --- SUSPENSION (Navier-Stokes Coupling) ---
    // If fluid velocity is high here, Soil lifts/moves with it.
    let vx = chunk.vel_x[idx];
    let vy = chunk.vel_y[idx];
    let mut speed = vx.hypot(vy);

    // Entrainment: Check ABOVE for fast water
    if speed < 1.0 && y > 0 {
        let idx_above = Chunk::index(x, y - 1);
        let vx_above = chunk.vel_x[idx_above];
        let vy_above = chunk.vel_y[idx_above];
        let speed_above = vx_above.hypot(vy_above);
        
        // Soil lifts easier than it moves?
        if speed_above > 1.0 {
             // Pick up velocity
             chunk.vel_x[idx] = vx_above * 0.5;
             chunk.vel_y[idx] = vy_above * 0.5 - 1.0; // Strong upward kick!
             speed = chunk.vel_x[idx].hypot(chunk.vel_y[idx]);
        }
    }

    // Threshold: Soil needs faster water to lift than Mud
    if speed > 0.8 { // Lowered threshold (1.0 -> 0.8)
        // Simple advection step
        let steps = (vx.abs().max(vy.abs()).ceil() as usize).max(1);
        let mut curr_x = x;
        let mut curr_y = y;
        
        for _ in 0..steps {
             let dx = if vx > 0.1 { 1 } else if vx < -0.1 { -1 } else { 0 };
             let dy = if vy > 0.1 { 1 } else if vy < -0.1 { -1 } else { 0 };
             if dx == 0 && dy == 0 { break; }
             
             // Check bounds
             let next_x = (curr_x as i32 + dx); 
             let next_y = (curr_y as i32 + dy);
             
             if next_x < 0 || next_x >= CHUNK_SIZE as i32 || next_y < 0 || next_y >= CHUNK_SIZE as i32 {
                 break;
             }
             let nx = next_x as usize;
             let ny = next_y as usize;

             let target_mat = chunk.get_material(nx, ny);
             if target_mat == Material::Air || target_mat.is_liquid() {
                 chunk.swap(curr_x, curr_y, nx, ny);
                 curr_x = nx;
                 curr_y = ny;
                 return true; // Moved via suspension
             }
        }
    }
    // -------------------------------------------

    // Can't fall if at bottom
    if y >= CHUNK_SIZE - 1 {
        return false;
    }

    // 1. Try to fall straight down
    let below = chunk.get_material(x, y + 1);
    if can_displace(density, below) {
        chunk.swap(x, y, x, y + 1);
        return true;
    }

    // 2. Try to slide diagonally
    // Use frame-based pseudo-random to pick direction (prevents bias)
    let try_left_first = ((x ^ y) as u64 ^ frame) & 1 == 0;

    let (dir1, dir2) = if try_left_first {
        (x.checked_sub(1), x.checked_add(1).filter(|&nx| nx < CHUNK_SIZE))
    } else {
        (x.checked_add(1).filter(|&nx| nx < CHUNK_SIZE), x.checked_sub(1))
    };

    // Try first diagonal direction
    if let Some(nx) = dir1 {
        let diag = chunk.get_material(nx, y + 1);
        if can_displace(density, diag) {
            chunk.swap(x, y, nx, y + 1);
            return true;
        }
    }

    // Try second diagonal direction
    if let Some(nx) = dir2 {
        let diag = chunk.get_material(nx, y + 1);
        if can_displace(density, diag) {
            chunk.swap(x, y, nx, y + 1);
            return true;
        }
    }

    false
}

/// Update liquid material (Water, Mud) - Moves based on Fluid Grid Velocity.
fn update_liquid(chunk: &mut Chunk, x: usize, y: usize, frame: u64) -> bool {
    let material = chunk.get_material(x, y);
    let density = material.density();
    let idx = Chunk::index(x, y);

    // Read Grid Velocity (Solved by Navier-Stokes)
    let vx = chunk.vel_x[idx];
    let vy = chunk.vel_y[idx];
    let mut speed = vx.hypot(vy);

    // Entrainment: If settled, check velocity ABOVE to be picked up
    if speed < 0.5 && y > 0 {
        let idx_above = Chunk::index(x, y - 1);
        let vx_above = chunk.vel_x[idx_above];
        let vy_above = chunk.vel_y[idx_above];
        let speed_above = vx_above.hypot(vy_above);
        
        if speed_above > 0.5 {
             // Friction/Entrainment: Pick up velocity from above
             chunk.vel_x[idx] = vx_above * 0.5;
             chunk.vel_y[idx] = vy_above * 0.5;
             
             // TURBULENT LIFT: Kick it up into the flow!
             chunk.vel_y[idx] -= 0.8; 
             
             speed = chunk.vel_x[idx].hypot(chunk.vel_y[idx]);
        }
    }

    // 1. Suspension Physics (Mud)
    if material == Material::Mud {
        if speed > 0.3 {
            // Suspended! Flows with the water.
            // No gravity, no damping.
            // Turbulence: mix vertically
            if frame % 4 == 0 {
                let jitter = if (frame + x as u64) % 2 == 0 { 0.2 } else { -0.2 };
                chunk.vel_x[idx] += jitter;
                chunk.vel_y[idx] -= 0.1; // Slight upward buoyancy bias to counteract settling
            }
        } else {
            // Settling
            chunk.vel_y[idx] += GRAVITY;
            chunk.vel_x[idx] *= 0.8; // Reduced damping (0.5 -> 0.8) to allow buildup of speed
            chunk.vel_y[idx] *= 0.8;
        }
    } else {
         // Water - always add gravity (accelerates down)
         // But careful: solver handles pressure.
         if chunk.vel_y[idx] < TERMINAL_VELOCITY {
             chunk.vel_y[idx] += GRAVITY; 
         }

         // 2. Hydrostatic Pressure (Explicit Spreading)
         // Navier-Stokes handles pressure, but this "Overdrive" ensures fast settling for gameplay.
         let pressure = if y > 0 && chunk.get_material(x, y - 1).is_liquid() { 4.0 } else { 0.2 };
         
         // Basic spreading preference
         if chunk.vel_x[idx].abs() < pressure {
             // If not already moving fast, push away from walls or random
             let mut push = 0.0;
             if x > 0 && !chunk.get_material(x-1, y).is_solid() { push -= pressure; }
             if x < CHUNK_SIZE-1 && !chunk.get_material(x+1, y).is_solid() { push += pressure; }
             
             if push != 0.0 {
                 chunk.vel_x[idx] += push * 0.5; // Apply force
             }
         }
    }

    // 2. Re-read Velocity (after gravity add)
    let vx = chunk.vel_x[idx];
    let vy = chunk.vel_y[idx];

    // 3. Move Particle (Advection)
    let steps = (vx.abs().max(vy.abs()).ceil() as usize).max(1);
    
    let mut curr_x = x;
    let mut curr_y = y;
    let mut moved = false;

    for _ in 0..steps {
         let dx = if vx > 0.1 { 1 } else if vx < -0.1 { -1 } else { 0 };
         let dy = if vy > 0.1 { 1 } else { 0 };

         if dx == 0 && dy == 0 { break; }

         let next_x = (curr_x as i32 + dx) as usize;
         let next_y = (curr_y as i32 + dy) as usize;

         if next_x >= CHUNK_SIZE || next_y >= CHUNK_SIZE {
             break; // Hit world bounds
         }

         let target = chunk.get_material(next_x, next_y);
         
         // Interaction Logic
         if target == Material::Air {
             // Move freely
             chunk.swap_material_only(curr_x, curr_y, next_x, next_y);
             curr_x = next_x;
             curr_y = next_y;
             moved = true;
         } else if target.is_liquid() {
             // Mix/Push liquids (Density check)
             if density > target.density() {
                 chunk.swap_material_only(curr_x, curr_y, next_x, next_y);
                 curr_x = next_x;
                 curr_y = next_y;
                 moved = true;
             } else {
                 // Blocked by fluid. Try side step (bounce/slide)
                let slide_dir = if dx == 0 {
                    if (frame + x as u64) % 2 == 0 { 1 } else { -1 }
                } else {
                    -dx
                };
                let slide_x = (curr_x as i32 + slide_dir) as usize;
                
                if slide_x < CHUNK_SIZE && chunk.get_material(slide_x, curr_y) == Material::Air {
                    chunk.swap_material_only(curr_x, curr_y, slide_x, curr_y);
                    curr_x = slide_x;
                    moved = true;
                }
                break;
             }
         } else if target == Material::Soil {
             // EROSION: High velocity water displaces dirt
             let speed = (vx * vx + vy * vy).sqrt();
             if speed > EROSION_THRESHOLD {
                 // Scour: Swap water and dirt
                 chunk.swap(curr_x, curr_y, next_x, next_y);
                 
                 // Transfer momentum to dirt
                 let dirt_idx = Chunk::index(curr_x, curr_y);
                 chunk.vel_x[dirt_idx] = vx * 0.8;
                 chunk.vel_y[dirt_idx] = vy * 0.5;
                 
                 // Water slows down
                 let water_idx = Chunk::index(next_x, next_y);
                 chunk.vel_x[water_idx] *= 0.5;
                 chunk.vel_y[water_idx] *= 0.5;

                 curr_x = next_x;
                 curr_y = next_y;
                 moved = true;
             } else {
                 // Hit solid ground
                 chunk.vel_x[idx] = 0.0;
                 chunk.vel_y[idx] = 0.0;
                 break;
             }
         } else {
             // Hit Rock
             if dy > 0 && dx == 0 {
                 // Hit floor, spread sideways
                 let spread_dir = if (frame + x as u64) % 2 == 0 { 1 } else { -1 };
                 chunk.vel_x[idx] = spread_dir as f32 * 2.0;
                 chunk.vel_y[idx] = 0.0;
                 // Don't zero Y yet, allows sliding?
             }
             break;
         }
    }
    
    moved
}

/// Get the height of a water column (count water cells from y downward).
fn get_column_height(chunk: &Chunk, x: usize, y: usize) -> u32 {
    let mut height = 0;
    let mut check_y = y;

    while check_y < CHUNK_SIZE {
        if chunk.get_material(x, check_y) == Material::Water {
            height += 1;
            check_y += 1;
        } else {
            break;
        }
    }

    height
}

/// Try to flow surface water from (x, y) to neighbor column nx.
/// Flows to air with support, or to shorter water columns.
fn try_surface_flow(chunk: &mut Chunk, x: usize, y: usize, nx: usize, my_height: u32) -> bool {
    // Deprecated? New velocity logic handles spreading via pressure jitter.
    // Keeping for fallback or removal.
    // Let's rely on velocity jitter for now.
    false
}

/// Check if a material with given density can displace another material.
#[inline]
fn can_displace(density: u8, target: Material) -> bool {
    match target {
        Material::Air => true,
        Material::Water => density > target.density(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soil_falls_into_air() {
        let mut chunk = Chunk::new();
        chunk.set_material(32, 32, Material::Soil);

        update_chunk(&mut chunk, 0);

        assert_eq!(chunk.get_material(32, 32), Material::Air);
        assert_eq!(chunk.get_material(32, 33), Material::Soil);
    }

    #[test]
    fn soil_slides_when_blocked() {
        let mut chunk = Chunk::new();
        chunk.set_material(32, 32, Material::Soil);
        chunk.set_material(32, 33, Material::Rock); // Block below

        update_chunk(&mut chunk, 0);

        // Should have slid to 31,33 or 33,33
        let slid_left = chunk.get_material(31, 33) == Material::Soil;
        let slid_right = chunk.get_material(33, 33) == Material::Soil;
        assert!(slid_left || slid_right, "Soil should slide diagonally");
    }

    #[test]
    fn gold_sinks_through_water() {
        let mut chunk = Chunk::new();
        chunk.set_material(32, 32, Material::Gold);
        chunk.set_material(32, 33, Material::Water);
        // Block all water escape routes so we test the density swap
        chunk.set_material(32, 34, Material::Rock); // Below
        chunk.set_material(31, 33, Material::Rock); // Left
        chunk.set_material(33, 33, Material::Rock); // Right
        chunk.set_material(31, 34, Material::Rock); // Diag left
        chunk.set_material(33, 34, Material::Rock); // Diag right

        update_chunk(&mut chunk, 0);

        assert_eq!(chunk.get_material(32, 33), Material::Gold);
        assert_eq!(chunk.get_material(32, 32), Material::Water);
    }

    #[test]
    fn water_spreads_horizontally_via_velocity() {
        let mut chunk = Chunk::new();
        chunk.set_material(32, 63, Material::Water); // At bottom
        chunk.set_material(32, 62, Material::Rock);  // Can't slide down
        
        // Run a few frames to let velocity jitter kick in
        for i in 0..10 {
            update_chunk(&mut chunk, i);
        }

        // Should spread left or right eventually
        let spread_left = chunk.get_material(31, 63) == Material::Water;
        let spread_right = chunk.get_material(33, 63) == Material::Water;
        assert!(spread_left || spread_right, "Water should spread horizontally");
    }
}

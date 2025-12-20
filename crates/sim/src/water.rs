//! Virtual Pipes Water Simulation
//!
//! Mass-based water simulation that creates continuous currents.
//! Instead of moving water particles, we track water mass per cell
//! and flow rates between cells through virtual "pipes".
//!
//! Based on: https://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html

use crate::chunk::{Chunk, CHUNK_AREA, CHUNK_SIZE};
use crate::Material;

const GRAVITY: f32 = 30.0;
const PIPE_AREA: f32 = 1.0;
const PIPE_LENGTH: f32 = 1.0;
const DAMPING: f32 = 0.995;  // Very little damping - keep momentum
const MIN_FLOW: f32 = 0.0001;
const MIN_WATER: f32 = 0.01;

/// Particles displace water - this is how much "height" a particle adds
const PARTICLE_DISPLACEMENT: f32 = 0.8;

/// Flow resistance when particles are present (0.0 = no resistance, 1.0 = full block)
const PARTICLE_RESISTANCE: f32 = 0.7;

/// Update water flows and mass for a single chunk.
/// Call this each frame for chunks with water.
pub fn update_water(chunk: &mut Chunk, dt: f32) {
    if !chunk.has_water {
        return;
    }

    // Step 1: Update flow rates based on pressure differences
    update_flows(chunk, dt);

    // Step 2: Scale outflows to prevent negative water
    scale_outflows(chunk, dt);

    // Step 3: Apply flows to update water mass
    update_water_mass(chunk, dt);

    // Check if we still have water
    chunk.has_water = chunk.water_mass.iter().any(|&w| w > MIN_WATER);
}

/// Get effective water height at a cell, accounting for particle displacement.
/// Particles take up space, pushing water upward and creating pressure.
#[inline]
fn effective_height(chunk: &Chunk, idx: usize) -> f32 {
    let water = chunk.water_mass[idx];

    // Only add displacement if there's BOTH water AND a particle
    // Dry particles don't displace anything
    if water > MIN_WATER {
        let material = chunk.materials[idx];
        if material != Material::Air && !material.is_solid() {
            return water + PARTICLE_DISPLACEMENT;
        }
    }

    water
}

/// Check if a cell has a particle (non-air, non-solid material)
#[inline]
fn has_particle(chunk: &Chunk, idx: usize) -> bool {
    let material = chunk.materials[idx];
    material != Material::Air && !material.is_solid()
}

/// Check if a particle is "packed" (touching other particles below or to sides)
/// Isolated particles don't block water flow - only packed piles do
#[inline]
fn is_particle_packed(chunk: &Chunk, x: usize, y: usize) -> bool {
    let idx = Chunk::index(x, y);
    if !has_particle(chunk, idx) {
        return false;
    }

    // Check if supported by another particle below
    if y < CHUNK_SIZE - 1 {
        let below_mat = chunk.materials[Chunk::index(x, y + 1)];
        if below_mat != Material::Air && !below_mat.is_solid() {
            return true; // Sitting on another particle
        }
        if below_mat.is_solid() {
            // On solid ground - check if neighbors also on ground
            let has_neighbor = (x > 0 && has_particle(chunk, Chunk::index(x - 1, y)))
                || (x < CHUNK_SIZE - 1 && has_particle(chunk, Chunk::index(x + 1, y)));
            if has_neighbor {
                return true; // Part of a pile on the ground
            }
        }
    }

    false // Isolated particle - doesn't block water
}

/// Get flow resistance for a cell (only PACKED particles slow down water)
#[inline]
fn flow_resistance(chunk: &Chunk, from_idx: usize, to_idx: usize, from_x: usize, from_y: usize, to_x: usize, to_y: usize) -> f32 {
    // Only apply resistance if particles are packed (in a pile)
    // Isolated particles in water don't block flow
    let from_packed = is_particle_packed(chunk, from_x, from_y);
    let to_packed = is_particle_packed(chunk, to_x, to_y);

    if from_packed || to_packed {
        1.0 - PARTICLE_RESISTANCE  // 0.3 multiplier for packed particles
    } else {
        1.0  // No resistance for isolated particles or clear water
    }
}

/// Calculate flow acceleration based on water height difference.
fn update_flows(chunk: &mut Chunk, dt: f32) {
    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);

            // Skip if solid material blocks flow
            if chunk.materials[idx].is_solid() {
                chunk.flow_right[idx] = 0.0;
                chunk.flow_down[idx] = 0.0;
                continue;
            }

            // Use effective height which accounts for particle displacement
            let height_here = effective_height(chunk, idx);

            // Flow Right
            if x < CHUNK_SIZE - 1 {
                let idx_right = Chunk::index(x + 1, y);
                if !chunk.materials[idx_right].is_solid() {
                    let height_right = effective_height(chunk, idx_right);
                    // Height difference creates pressure
                    let height_diff = height_here - height_right;

                    // Apply resistance only from PACKED particles (not isolated ones)
                    let resistance = flow_resistance(chunk, idx, idx_right, x, y, x + 1, y);

                    // Accelerate flow based on pressure
                    let acceleration = GRAVITY * height_diff * PIPE_AREA / PIPE_LENGTH * resistance;
                    chunk.flow_right[idx] += acceleration * dt;

                    // Apply damping (more damping through packed particles)
                    chunk.flow_right[idx] *= DAMPING * resistance.sqrt();

                    // Zero out tiny flows
                    if chunk.flow_right[idx].abs() < MIN_FLOW {
                        chunk.flow_right[idx] = 0.0;
                    }
                } else {
                    chunk.flow_right[idx] = 0.0;
                }
            }

            // Flow Down (gravity adds to downward pressure)
            if y < CHUNK_SIZE - 1 {
                let idx_down = Chunk::index(x, y + 1);
                if !chunk.materials[idx_down].is_solid() {
                    let height_down = effective_height(chunk, idx_down);
                    // +1 adds gravity's contribution to downward flow
                    let height_diff = height_here - height_down + 1.0;

                    // Apply resistance only from PACKED particles
                    let resistance = flow_resistance(chunk, idx, idx_down, x, y, x, y + 1);

                    let acceleration = GRAVITY * height_diff * PIPE_AREA / PIPE_LENGTH * resistance;
                    chunk.flow_down[idx] += acceleration * dt;

                    // Apply damping (more damping through packed particles)
                    chunk.flow_down[idx] *= DAMPING * resistance.sqrt();

                    // Zero out tiny flows
                    if chunk.flow_down[idx].abs() < MIN_FLOW {
                        chunk.flow_down[idx] = 0.0;
                    }
                } else {
                    chunk.flow_down[idx] = 0.0;
                }
            }
        }
    }
}

/// Scale outflows to ensure we never drain more water than available.
/// This is the key to stability - prevents negative water.
fn scale_outflows(chunk: &mut Chunk, dt: f32) {
    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);
            let water = chunk.water_mass[idx];

            if water <= MIN_WATER {
                // No water - can only receive, not send
                if chunk.flow_right[idx] > 0.0 {
                    chunk.flow_right[idx] = 0.0;
                }
                if chunk.flow_down[idx] > 0.0 {
                    chunk.flow_down[idx] = 0.0;
                }
                continue;
            }

            // Sum all outgoing flows from this cell
            let mut total_out = 0.0;

            // Right outflow (positive flow_right means going right = leaving this cell)
            if chunk.flow_right[idx] > 0.0 {
                total_out += chunk.flow_right[idx];
            }

            // Down outflow (positive flow_down means going down = leaving this cell)
            if chunk.flow_down[idx] > 0.0 {
                total_out += chunk.flow_down[idx];
            }

            // Left outflow (negative flow_right on left neighbor means coming TO us,
            // so negative value on current cell means leaving to the left)
            if x > 0 {
                let left_flow = chunk.flow_right[Chunk::index(x - 1, y)];
                // If left neighbor has negative flow_right, water flows from us to them
                if left_flow < 0.0 {
                    total_out += -left_flow;
                }
            }

            // Up outflow (negative flow_down on upper neighbor)
            if y > 0 {
                let up_flow = chunk.flow_down[Chunk::index(x, y - 1)];
                if up_flow < 0.0 {
                    total_out += -up_flow;
                }
            }

            // Scale if we would drain more than available
            let max_out = water / dt;
            if total_out > max_out && total_out > 0.0 {
                let scale = max_out / total_out;

                // Scale outgoing flows from this cell
                if chunk.flow_right[idx] > 0.0 {
                    chunk.flow_right[idx] *= scale;
                }
                if chunk.flow_down[idx] > 0.0 {
                    chunk.flow_down[idx] *= scale;
                }
            }
        }
    }
}

/// Water density for reference
const WATER_DENSITY: f32 = 10.0;

/// Flow strength needed to suspend a particle (relative to density ratio)
const SUSPENSION_FACTOR: f32 = 0.02;

/// Apply flow rates to transfer water mass AND particles between cells.
/// UNIFIED PHYSICS: Particles ARE water, just denser. They flow together.
/// Denser particles need stronger flow to move, otherwise they settle.
fn update_water_mass(chunk: &mut Chunk, dt: f32) {
    // Use scratch_a as temporary buffer to avoid order artifacts
    chunk.scratch_a.copy_from_slice(&*chunk.water_mass);

    // Use scratch_b for material moves (0 = no move, 1 = right, 2 = down, 3 = settle down)
    for i in 0..CHUNK_AREA {
        chunk.scratch_b[i] = 0.0;
    }

    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);

            // Skip solid materials
            if chunk.materials[idx].is_solid() {
                chunk.scratch_a[idx] = 0.0;
                continue;
            }

            let material = chunk.materials[idx];
            let is_particle = material != Material::Air && !material.is_solid();
            let water_here = chunk.water_mass[idx];

            // Soil becomes Mud when wet
            if material == Material::Soil && water_here > 0.3 {
                chunk.materials[idx] = Material::Mud;
            }

            // Calculate density-based flow threshold
            // Denser particles need more flow to move
            let density = material.density() as f32;
            let density_ratio = density / WATER_DENSITY;
            let flow_threshold = density_ratio * SUSPENSION_FACTOR;

            // UNIFIED: Combine Virtual Pipes flow + Navier-Stokes velocity field
            // Both contribute to particle movement - this creates turbulence effects
            let pipe_flow_h = chunk.flow_right[idx];
            let pipe_flow_v = if y < CHUNK_SIZE - 1 { chunk.flow_down[idx] } else { 0.0 };
            let vel_x = chunk.vel_x[idx];
            let vel_y = chunk.vel_y[idx];

            // Total velocity = pipe flow + NS velocity (turbulence)
            let total_vx = pipe_flow_h + vel_x * 0.5;
            let total_vy = pipe_flow_v + vel_y * 0.5;
            let total_flow = (total_vx * total_vx + total_vy * total_vy).sqrt();

            // Flow right (positive = water leaving rightward)
            if x < CHUNK_SIZE - 1 {
                let flow = chunk.flow_right[idx] * dt;
                if flow.abs() > MIN_FLOW {
                    chunk.scratch_a[idx] -= flow;
                    chunk.scratch_a[Chunk::index(x + 1, y)] += flow;
                }

                // Particle moves if combined velocity exceeds threshold
                if is_particle && total_vx > flow_threshold && chunk.scratch_b[idx] == 0.0 {
                    let target_idx = Chunk::index(x + 1, y);
                    let target_mat = chunk.materials[target_idx];
                    if !target_mat.is_solid() {
                        chunk.scratch_b[idx] = 1.0; // Mark for rightward move
                    }
                }
            }

            // Flow left (from velocity field - turbulence can push particles back)
            if x > 0 && is_particle && total_vx < -flow_threshold && chunk.scratch_b[idx] == 0.0 {
                let target_idx = Chunk::index(x - 1, y);
                let target_mat = chunk.materials[target_idx];
                if !target_mat.is_solid() {
                    chunk.scratch_b[idx] = 4.0; // Mark for leftward move
                }
            }

            // Flow down (positive = water leaving downward)
            if y < CHUNK_SIZE - 1 {
                let flow = chunk.flow_down[idx] * dt;
                if flow.abs() > MIN_FLOW {
                    chunk.scratch_a[idx] -= flow;
                    chunk.scratch_a[Chunk::index(x, y + 1)] += flow;
                }

                // Particles move down with combined flow + gravity
                if is_particle && (total_vy > 0.0 || flow > 0.0) && chunk.scratch_b[idx] == 0.0 {
                    let target_idx = Chunk::index(x, y + 1);
                    let target_mat = chunk.materials[target_idx];
                    if !target_mat.is_solid() {
                        chunk.scratch_b[idx] = 2.0; // Mark for downward move
                    }
                }
            }

            // Flow up (from strong upward turbulence - can lift light particles)
            if y > 0 && is_particle && total_vy < -flow_threshold * 2.0 && chunk.scratch_b[idx] == 0.0 {
                let target_idx = Chunk::index(x, y - 1);
                let target_mat = chunk.materials[target_idx];
                if !target_mat.is_solid() {
                    chunk.scratch_b[idx] = 5.0; // Mark for upward move
                }
            }

            // SETTLING: If particle is denser than water and not enough flow to suspend
            if is_particle && water_here > 0.1 && chunk.scratch_b[idx] == 0.0 {
                if total_flow < flow_threshold && y < CHUNK_SIZE - 1 {
                    let below_idx = Chunk::index(x, y + 1);
                    let below_mat = chunk.materials[below_idx];
                    if below_mat == Material::Air ||
                       (!below_mat.is_solid() && density > below_mat.density() as f32) {
                        chunk.scratch_b[idx] = 3.0; // Mark for settling
                    } else {
                        // Can't settle straight down - try diagonal slide (prevents walls)
                        let frame_bias = ((x + y) & 1) == 0;
                        let (dx1, dx2) = if frame_bias { (1i32, -1i32) } else { (-1i32, 1i32) };

                        for dx in [dx1, dx2] {
                            let nx = x as i32 + dx;
                            if nx >= 0 && nx < CHUNK_SIZE as i32 {
                                let diag_idx = Chunk::index(nx as usize, y + 1);
                                let diag_mat = chunk.materials[diag_idx];
                                if diag_mat == Material::Air ||
                                   (!diag_mat.is_solid() && density > diag_mat.density() as f32) {
                                    // Slide diagonally down
                                    chunk.scratch_b[idx] = if dx > 0 { 6.0 } else { 7.0 };
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // PRESSURE PUSH: Water pressure from the side pushes particles
            // Even if flow isn't carrying the particle, pressure difference pushes it
            if is_particle && chunk.scratch_b[idx] == 0.0 {
                // Check water height difference on left vs right
                let water_left = if x > 0 { chunk.water_mass[Chunk::index(x - 1, y)] } else { 0.0 };
                let water_right = if x < CHUNK_SIZE - 1 { chunk.water_mass[Chunk::index(x + 1, y)] } else { 0.0 };
                let pressure_diff = water_left - water_right;

                // Pressure threshold based on density - heavier particles need more pressure
                let pressure_threshold = density * 0.05;

                if pressure_diff > pressure_threshold && x < CHUNK_SIZE - 1 {
                    // Pressure from left pushes particle right
                    let target_idx = Chunk::index(x + 1, y);
                    let target_mat = chunk.materials[target_idx];
                    if target_mat == Material::Air ||
                       (!target_mat.is_solid() && density > target_mat.density() as f32) {
                        chunk.scratch_b[idx] = 1.0; // Push right
                    }
                } else if pressure_diff < -pressure_threshold && x > 0 {
                    // Pressure from right pushes particle left
                    let target_idx = Chunk::index(x - 1, y);
                    let target_mat = chunk.materials[target_idx];
                    if target_mat == Material::Air ||
                       (!target_mat.is_solid() && density > target_mat.density() as f32) {
                        chunk.scratch_b[idx] = 4.0; // Push left
                    }
                }
            }

            // INHERENT INSTABILITY: Particles have NO friction between each other
            // If sitting on another particle (not solid ground), try to slide sideways
            // This prevents 1-pixel towers from forming
            if is_particle && chunk.scratch_b[idx] == 0.0 && y < CHUNK_SIZE - 1 {
                let below_idx = Chunk::index(x, y + 1);
                let below_mat = chunk.materials[below_idx];

                // Only unstable if sitting on another PARTICLE (not solid ground)
                if !below_mat.is_solid() && below_mat != Material::Air {
                    // Check if we can slide diagonally down (there's space)
                    let frame_bias = ((x + y) & 1) == 0;
                    let (dx1, dx2) = if frame_bias { (1i32, -1i32) } else { (-1i32, 1i32) };

                    for dx in [dx1, dx2] {
                        let nx = x as i32 + dx;
                        if nx >= 0 && nx < CHUNK_SIZE as i32 {
                            let diag_idx = Chunk::index(nx as usize, y + 1);
                            let diag_mat = chunk.materials[diag_idx];
                            // Can slide if diagonal is empty OR less dense
                            if diag_mat == Material::Air ||
                               (!diag_mat.is_solid() && density > diag_mat.density() as f32) {
                                // Also check side is not blocked (need room to slide)
                                let side_idx = Chunk::index(nx as usize, y);
                                let side_mat = chunk.materials[side_idx];
                                if side_mat == Material::Air || !side_mat.is_solid() {
                                    chunk.scratch_b[idx] = if dx > 0 { 6.0 } else { 7.0 };
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Clamp to valid range and copy back
    for i in 0..CHUNK_AREA {
        chunk.water_mass[i] = chunk.scratch_a[i].max(0.0);
    }

    // Apply particle moves - SWAP materials and CREATE TURBULENCE
    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);
            let move_dir = chunk.scratch_b[idx];
            let from_mat = chunk.materials[idx];

            // Particles create turbulence when they move
            let momentum = from_mat.density() as f32 * 0.01;

            if move_dir == 1.0 && x < CHUNK_SIZE - 1 {
                // Swap right
                let target_idx = Chunk::index(x + 1, y);
                let to_mat = chunk.materials[target_idx];
                if !to_mat.is_solid() {
                    chunk.materials[idx] = to_mat;
                    chunk.materials[target_idx] = from_mat;
                    // Add momentum to velocity field (particle creates turbulence)
                    chunk.vel_x[idx] += momentum;
                    chunk.vel_x[target_idx] += momentum;
                }
            } else if move_dir == 4.0 && x > 0 {
                // Swap left (turbulence pushed it back)
                let target_idx = Chunk::index(x - 1, y);
                let to_mat = chunk.materials[target_idx];
                if !to_mat.is_solid() {
                    chunk.materials[idx] = to_mat;
                    chunk.materials[target_idx] = from_mat;
                    chunk.vel_x[idx] -= momentum;
                    chunk.vel_x[target_idx] -= momentum;
                }
            } else if move_dir == 2.0 && y < CHUNK_SIZE - 1 {
                // Swap down (carried by flow)
                let target_idx = Chunk::index(x, y + 1);
                let to_mat = chunk.materials[target_idx];
                if !to_mat.is_solid() {
                    chunk.materials[idx] = to_mat;
                    chunk.materials[target_idx] = from_mat;
                    chunk.vel_y[idx] += momentum;
                    chunk.vel_y[target_idx] += momentum;
                }
            } else if move_dir == 5.0 && y > 0 {
                // Swap up (turbulence lifted it)
                let target_idx = Chunk::index(x, y - 1);
                let to_mat = chunk.materials[target_idx];
                if !to_mat.is_solid() {
                    chunk.materials[idx] = to_mat;
                    chunk.materials[target_idx] = from_mat;
                    chunk.vel_y[idx] -= momentum;
                    chunk.vel_y[target_idx] -= momentum;
                }
            } else if move_dir == 3.0 && y < CHUNK_SIZE - 1 {
                // Settle down (sink through lighter material)
                let target_idx = Chunk::index(x, y + 1);
                let to_mat = chunk.materials[target_idx];
                if !to_mat.is_solid() && from_mat.density() > to_mat.density() {
                    chunk.materials[idx] = to_mat;
                    chunk.materials[target_idx] = from_mat;
                    chunk.vel_y[idx] += momentum * 0.5;
                    chunk.vel_y[target_idx] += momentum * 0.5;
                }
            } else if move_dir == 6.0 && x < CHUNK_SIZE - 1 && y < CHUNK_SIZE - 1 {
                // Diagonal slide down-right (spreading)
                let target_idx = Chunk::index(x + 1, y + 1);
                let to_mat = chunk.materials[target_idx];
                if !to_mat.is_solid() && from_mat.density() > to_mat.density() {
                    chunk.materials[idx] = to_mat;
                    chunk.materials[target_idx] = from_mat;
                    chunk.vel_x[target_idx] += momentum * 0.3;
                    chunk.vel_y[target_idx] += momentum * 0.3;
                }
            } else if move_dir == 7.0 && x > 0 && y < CHUNK_SIZE - 1 {
                // Diagonal slide down-left (spreading)
                let target_idx = Chunk::index(x - 1, y + 1);
                let to_mat = chunk.materials[target_idx];
                if !to_mat.is_solid() && from_mat.density() > to_mat.density() {
                    chunk.materials[idx] = to_mat;
                    chunk.materials[target_idx] = from_mat;
                    chunk.vel_x[target_idx] -= momentum * 0.3;
                    chunk.vel_y[target_idx] += momentum * 0.3;
                }
            }
        }
    }

    chunk.needs_render = true;
}

/// Add water at a specific position in the chunk.
/// Returns the amount actually added (may be less if cell is solid).
pub fn add_water_at(chunk: &mut Chunk, x: usize, y: usize, amount: f32) -> f32 {
    let idx = Chunk::index(x, y);

    // Can't add water to solid materials
    if chunk.materials[idx].is_solid() {
        return 0.0;
    }

    chunk.water_mass[idx] += amount;
    chunk.has_water = true;
    chunk.is_active = true;
    chunk.needs_render = true;

    amount
}

/// Remove water at a specific position.
/// Returns the amount actually removed.
pub fn remove_water_at(chunk: &mut Chunk, x: usize, y: usize, amount: f32) -> f32 {
    let idx = Chunk::index(x, y);
    let available = chunk.water_mass[idx];
    let removed = available.min(amount);

    chunk.water_mass[idx] -= removed;
    chunk.needs_render = true;

    removed
}

/// Get the effective water level at a position (for rendering/physics).
/// Returns a value from 0.0 (dry) to 1.0+ (full/overfull).
pub fn get_water_level(chunk: &Chunk, x: usize, y: usize) -> f32 {
    chunk.water_mass[Chunk::index(x, y)]
}

/// Get the flow velocity at a position (useful for particle suspension).
/// Returns (vx, vy) based on flow rates through the cell.
pub fn get_water_velocity(chunk: &Chunk, x: usize, y: usize) -> (f32, f32) {
    let idx = Chunk::index(x, y);

    // Horizontal velocity: average of flow in and out
    let mut vx = chunk.flow_right[idx]; // Flow going right
    if x > 0 {
        vx -= chunk.flow_right[Chunk::index(x - 1, y)]; // Flow coming from left (reversed)
    }
    vx *= 0.5;

    // Vertical velocity: average of flow in and out
    let mut vy = chunk.flow_down[idx]; // Flow going down
    if y > 0 {
        vy -= chunk.flow_down[Chunk::index(x, y - 1)]; // Flow coming from above (reversed)
    }
    vy *= 0.5;

    (vx, vy)
}

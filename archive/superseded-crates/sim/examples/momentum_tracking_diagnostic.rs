//! Momentum Tracking Diagnostic
//!
//! Track momentum through each step of the FLIP cycle to identify loss.
//! This isolates where the water "collapse" is happening.

use sim::flip::FlipSimulation;
use sim::grid::CellType;
use sim::particle::ParticleMaterial;
use glam::Vec2;

const WIDTH: usize = 60;
const HEIGHT: usize = 40;
const CELL_SIZE: f32 = 5.0;

fn main() {
    let mut flip = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // Simple box: floor and walls
    for i in 0..WIDTH {
        for j in 0..HEIGHT {
            // Floor
            if j >= HEIGHT - 4 {
                flip.grid.cell_type[j * WIDTH + i] = CellType::Solid;
            }
            // Walls
            if i < 2 || i >= WIDTH - 2 {
                flip.grid.cell_type[j * WIDTH + i] = CellType::Solid;
            }
        }
    }
    flip.grid.compute_sdf();

    // Create a simple water column in the middle
    let spacing = CELL_SIZE / 2.0;
    let water_left = WIDTH / 4;
    let water_right = 3 * WIDTH / 4;
    let water_top = HEIGHT / 4;
    let water_bottom = HEIGHT - 5;

    for j in water_top..water_bottom {
        for i in water_left..water_right {
            let idx = j * WIDTH + i;
            if flip.grid.cell_type[idx] != CellType::Solid {
                let base_x = i as f32 * CELL_SIZE;
                let base_y = j as f32 * CELL_SIZE;

                for pi in 0..2 {
                    for pj in 0..2 {
                        let px = base_x + (pi as f32 + 0.25) * spacing;
                        let py = base_y + (pj as f32 + 0.25) * spacing;
                        // Start with ZERO velocity (hydrostatic)
                        flip.particles.spawn_water(px, py, 0.0, 0.0);
                    }
                }
            }
        }
    }

    println!("=== MOMENTUM TRACKING DIAGNOSTIC ===");
    println!("Water particles: {}", flip.particles.len());
    println!("Starting with zero velocity (hydrostatic setup)");
    println!();

    let dt = 1.0 / 60.0;

    for frame in 0..30 {
        // Track momentum at each step
        let p_momentum_start = particle_momentum(&flip);

        // Step 1: Classify cells
        flip.classify_cells();
        flip.grid.compute_sdf();

        // Step 2: P2G
        flip.particles_to_grid();
        let grid_momentum_after_p2g = grid_momentum(&flip);

        // Step 3: Extrapolate + store old
        flip.grid.extrapolate_velocities(1);
        flip.store_old_velocities();
        let p_old_grid_vel = particle_old_grid_velocity(&flip);

        // Step 4: Apply gravity
        let grid_momentum_before_gravity = grid_momentum(&flip);
        flip.grid.apply_gravity(dt);
        let grid_momentum_after_gravity = grid_momentum(&flip);
        let gravity_delta = grid_momentum_after_gravity - grid_momentum_before_gravity;

        // Step 4b: Vorticity confinement
        {
            let grid = &mut flip.grid;
            let pile_height = &flip.pile_height;
            grid.apply_vorticity_confinement_with_piles(dt, 0.05, pile_height);
        }
        let grid_momentum_after_vort = grid_momentum(&flip);
        let vort_delta = grid_momentum_after_vort - grid_momentum_after_gravity;

        // Step 5: Pressure solve
        flip.grid.enforce_boundary_conditions();
        flip.grid.compute_divergence();
        let div_before = flip.grid.total_divergence();

        flip.grid.solve_pressure_multigrid(4);

        // Apply pressure gradient
        let grid_momentum_before_pressure = grid_momentum(&flip);
        flip.apply_pressure_gradient_two_way(dt);
        let grid_momentum_after_pressure = grid_momentum(&flip);
        let pressure_delta = grid_momentum_after_pressure - grid_momentum_before_pressure;

        // Porosity drag (should be zero with no sand)
        flip.apply_porosity_drag(dt);
        let grid_momentum_after_drag = grid_momentum(&flip);
        let drag_delta = grid_momentum_after_drag - grid_momentum_after_pressure;

        // Compute divergence after
        flip.grid.compute_divergence();
        let div_after = flip.grid.total_divergence();

        // Extrapolate for G2P
        flip.grid.extrapolate_velocities(1);

        // Step 6: G2P
        let p_momentum_before_g2p = particle_momentum(&flip);
        flip.grid_to_particles(dt);
        let p_momentum_after_g2p = particle_momentum(&flip);

        // Spatial hash and neighbors
        flip.build_spatial_hash();
        flip.compute_neighbor_counts();

        // Advection
        flip.advect_particles(dt);
        let p_momentum_after_advect = particle_momentum(&flip);

        // Final momentum
        let p_momentum_end = particle_momentum(&flip);

        // Compute net changes
        let g2p_delta = p_momentum_after_g2p - p_momentum_before_g2p;
        let advect_delta = p_momentum_after_advect - p_momentum_after_g2p;

        if frame < 10 || frame % 10 == 0 {
            println!("Frame {:2}:", frame);
            println!("  Particle momentum: start={:7.1} end={:7.1}",
                p_momentum_start.length(), p_momentum_end.length());
            println!("  Grid P2G:         ({:7.1}, {:7.1})",
                grid_momentum_after_p2g.x, grid_momentum_after_p2g.y);
            println!("  Grid after grav:  ({:7.1}, {:7.1}) delta_y={:7.1}",
                grid_momentum_after_gravity.x, grid_momentum_after_gravity.y, gravity_delta.y);
            println!("  Grid after vort:  ({:7.1}, {:7.1}) delta=({:5.1}, {:5.1})",
                grid_momentum_after_vort.x, grid_momentum_after_vort.y, vort_delta.x, vort_delta.y);
            println!("  Grid after pres:  ({:7.1}, {:7.1}) delta=({:5.1}, {:5.1})",
                grid_momentum_after_pressure.x, grid_momentum_after_pressure.y, pressure_delta.x, pressure_delta.y);
            println!("  Grid after drag:  ({:7.1}, {:7.1}) delta=({:5.1}, {:5.1})",
                grid_momentum_after_drag.x, grid_momentum_after_drag.y, drag_delta.x, drag_delta.y);
            println!("  Div: before={:7.1} after={:7.1}", div_before, div_after);
            println!("  G2P delta:        ({:7.1}, {:7.1})", g2p_delta.x, g2p_delta.y);
            println!("  Advect delta:     ({:7.1}, {:7.1})", advect_delta.x, advect_delta.y);
            println!();
        }
    }
}

/// Calculate total momentum of all water particles
fn particle_momentum(flip: &FlipSimulation) -> Vec2 {
    flip.particles.iter()
        .filter(|p| p.material == ParticleMaterial::Water)
        .map(|p| p.velocity)
        .fold(Vec2::ZERO, |acc, v| acc + v)
}

/// Calculate total "momentum" on grid (sum of velocities weighted by cell area)
fn grid_momentum(flip: &FlipSimulation) -> Vec2 {
    let mut u_sum = 0.0f32;
    let mut v_sum = 0.0f32;

    // Sum U velocities for fluid cells
    for j in 0..flip.grid.height {
        for i in 1..flip.grid.width {
            let left_idx = flip.grid.cell_index(i - 1, j);
            let right_idx = flip.grid.cell_index(i, j);
            let left_type = flip.grid.cell_type[left_idx];
            let right_type = flip.grid.cell_type[right_idx];

            if left_type == CellType::Fluid || right_type == CellType::Fluid {
                let u_idx = flip.grid.u_index(i, j);
                u_sum += flip.grid.u[u_idx];
            }
        }
    }

    // Sum V velocities for fluid cells
    for j in 1..flip.grid.height {
        for i in 0..flip.grid.width {
            let bottom_idx = flip.grid.cell_index(i, j - 1);
            let top_idx = flip.grid.cell_index(i, j);
            let bottom_type = flip.grid.cell_type[bottom_idx];
            let top_type = flip.grid.cell_type[top_idx];

            if bottom_type == CellType::Fluid || top_type == CellType::Fluid {
                let v_idx = flip.grid.v_index(i, j);
                v_sum += flip.grid.v[v_idx];
            }
        }
    }

    Vec2::new(u_sum, v_sum)
}

/// Calculate sum of old_grid_velocity stored on particles
fn particle_old_grid_velocity(flip: &FlipSimulation) -> Vec2 {
    flip.particles.iter()
        .filter(|p| p.material == ParticleMaterial::Water)
        .map(|p| p.old_grid_velocity)
        .fold(Vec2::ZERO, |acc, v| acc + v)
}

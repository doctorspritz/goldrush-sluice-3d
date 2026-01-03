//! Test pressure solver convergence.

use glam::Vec3;
use sim3d::{create_sluice, spawn_inlet_water, FlipSimulation3D, SluiceConfig, CellType};

fn compute_max_divergence(sim: &FlipSimulation3D) -> f32 {
    let grid = &sim.grid;
    let scale = 1.0 / grid.cell_size;
    let mut max_div = 0.0f32;

    for k in 0..grid.depth {
        for j in 0..grid.height {
            for i in 0..grid.width {
                let idx = grid.cell_index(i, j, k);
                if grid.cell_type[idx] != CellType::Fluid {
                    continue;
                }

                let u_right = grid.u[grid.u_index(i + 1, j, k)];
                let u_left = grid.u[grid.u_index(i, j, k)];
                let v_top = grid.v[grid.v_index(i, j + 1, k)];
                let v_bottom = grid.v[grid.v_index(i, j, k)];
                let w_front = grid.w[grid.w_index(i, j, k + 1)];
                let w_back = grid.w[grid.w_index(i, j, k)];

                let div = scale * ((u_right - u_left) + (v_top - v_bottom) + (w_front - w_back));
                max_div = max_div.max(div.abs());
            }
        }
    }
    max_div
}

fn main() {
    println!("=== PRESSURE SOLVER DIAGNOSTIC ===\n");

    // Test without sluice (open box)
    println!("=== TEST 1: Open box (no solids) ===");
    let mut sim = FlipSimulation3D::new(20, 12, 8, 0.1);
    sim.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim.flip_ratio = 0.97;
    sim.pressure_iterations = 50;

    // Spawn particles in open space
    for i in 5..15 {
        for j in 3..9 {
            for k in 2..6 {
                sim.spawn_particle_with_velocity(
                    Vec3::new(
                        (i as f32 + 0.5) * 0.1,
                        (j as f32 + 0.5) * 0.1,
                        (k as f32 + 0.5) * 0.1,
                    ),
                    Vec3::new(1.0, 0.0, 0.0),
                );
            }
        }
    }

    let dt = 1.0 / 60.0;
    for frame in 0..20 {
        let max_div = compute_max_divergence(&sim);
        let max_u = sim.grid.u.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));
        let max_v = sim.grid.v.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));
        let max_w = sim.grid.w.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));

        if frame % 5 == 0 {
            println!(
                "Frame {:2}: max_div={:8.4}, max_vel=({:8.3}, {:8.3}, {:8.3})",
                frame, max_div, max_u, max_v, max_w
            );
        }

        sim.update(dt);
    }

    // Test with sluice, tracking per-frame
    println!("\n=== TEST 2: Sluice per-frame ===");
    let mut sim2 = FlipSimulation3D::new(20, 12, 8, 0.1);
    sim2.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim2.flip_ratio = 0.97;
    sim2.pressure_iterations = 100;

    let config = SluiceConfig {
        slope: 0.08,
        slick_plate_len: 4,
        riffle_spacing: 4,
        riffle_height: 1,
        riffle_width: 1,
    };
    create_sluice(&mut sim2, &config);
    spawn_inlet_water(&mut sim2, &config, 100, Vec3::new(1.0, 0.0, 0.0));

    println!("Grid: {}x{}x{}, {} particles", sim2.grid.width, sim2.grid.height, sim2.grid.depth, sim2.particle_count());

    for frame in 0..30 {
        // Measure BEFORE update
        let max_div = compute_max_divergence(&sim2);
        let max_u = sim2.grid.u.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));
        let max_v = sim2.grid.v.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));

        if frame % 3 == 0 || max_u > 50.0 {
            println!(
                "Frame {:2}: max_div={:10.4}, max_u={:10.3}, max_v={:10.3}, particles={}",
                frame, max_div, max_u, max_v, sim2.particle_count()
            );
        }

        // If exploding, stop
        if max_u > 1000.0 {
            println!("!!! EXPLOSION at frame {} !!!", frame);
            break;
        }

        sim2.update(dt);
    }

    // Test with sluice but NO pressure solve
    println!("\n=== TEST 3: Sluice NO PRESSURE ===");
    let mut sim3 = FlipSimulation3D::new(20, 12, 8, 0.1);
    sim3.gravity = Vec3::new(0.0, -9.8, 0.0);
    sim3.flip_ratio = 0.97;
    sim3.pressure_iterations = 0; // Skip pressure!

    create_sluice(&mut sim3, &config);
    spawn_inlet_water(&mut sim3, &config, 100, Vec3::new(1.0, 0.0, 0.0));

    for frame in 0..30 {
        let max_div = compute_max_divergence(&sim3);
        let max_u = sim3.grid.u.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));
        let max_v = sim3.grid.v.iter().fold(0.0f32, |a, &b| a.abs().max(b.abs()));

        if frame % 3 == 0 || max_u > 50.0 {
            println!(
                "Frame {:2}: max_div={:10.4}, max_u={:10.3}, max_v={:10.3}",
                frame, max_div, max_u, max_v
            );
        }

        if max_u > 1000.0 {
            println!("!!! EXPLOSION at frame {} !!!", frame);
            break;
        }

        sim3.update(dt);
    }
}

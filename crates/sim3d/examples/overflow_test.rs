//! Headless test to investigate why water doesn't overflow tall riffles
//!
//! This test creates a sluice with 3-cell tall riffles and continuously
//! emits water to see if it accumulates and overflows.

use sim3d::*;

const GRID_WIDTH: usize = 48;  // Shorter for faster test
const GRID_HEIGHT: usize = 16;
const GRID_DEPTH: usize = 8;
const CELL_SIZE: f32 = 0.05;

fn create_sluice_with_tall_riffles(sim: &mut FlipSimulation3D) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    let depth = sim.grid.depth;

    // Floor slope: 4 cells left, 1 cell right
    let floor_height_left = 4;
    let floor_height_right = 1;

    // TALL riffles: 3 cells high (the issue being investigated)
    let riffle_spacing = 6;
    let riffle_height = 3;
    let riffle_start_x = 8;
    let riffle_end_x = width - 4;

    for k in 0..depth {
        for j in 0..height {
            for i in 0..width {
                let t = i as f32 / (width - 1) as f32;
                let floor_height = floor_height_left as f32 * (1.0 - t) + floor_height_right as f32 * t;
                let floor_j = floor_height as usize;

                let is_riffle = i >= riffle_start_x && i < riffle_end_x &&
                    (i - riffle_start_x) % riffle_spacing < 2 &&
                    j <= floor_j + riffle_height &&
                    j > floor_j;

                let is_boundary =
                    i == 0 ||                       // Left wall
                    i == width - 1 ||               // Right wall
                    j <= floor_j ||                 // Sloped floor
                    j == height - 1 ||              // Ceiling
                    k == 0 || k == depth - 1 ||     // Z walls
                    is_riffle;                      // Riffles

                if is_boundary {
                    sim.grid.set_solid(i, j, k);
                }
            }
        }
    }

    // CRITICAL: Compute SDF after setting solid cells - required for particle collision!
    sim.grid.compute_sdf();

    println!("Created sluice: {}x{}x{}, riffle_height={}", width, height, depth, riffle_height);
}

fn emit_particles(sim: &mut FlipSimulation3D, count: usize) {
    let cell_size = CELL_SIZE;

    // Emit at x=2, above the floor (which is 4 cells high on left)
    let emit_x = 2.0 * cell_size;
    let center_z = GRID_DEPTH as f32 * cell_size * 0.5;
    // Emit at height 8 (above floor + 3-cell riffle = 4+3=7)
    let emit_y = 8.0 * cell_size;

    // Inlet velocity: water enters with horizontal flow (simulates upstream flow)
    let inlet_velocity = Vec3::new(0.5, 0.0, 0.0);  // 0.5 m/s downstream

    for _ in 0..count {
        let x = emit_x + rand::random::<f32>() * cell_size;
        let z = center_z + (rand::random::<f32>() - 0.5) * 2.0 * cell_size;
        let y = emit_y + rand::random::<f32>() * 0.5 * cell_size;

        sim.spawn_particle_with_velocity(Vec3::new(x, y, z), inlet_velocity);
    }
}

fn analyze_water_height(sim: &FlipSimulation3D) -> (f32, f32, Vec<(usize, f32, usize)>) {
    let cell_size = CELL_SIZE;
    let mut max_y = 0.0f32;
    let mut max_x = 0.0f32;

    // Track max height per 8-cell X region
    let mut region_data: Vec<(f32, usize)> = vec![(0.0, 0); 6];

    for p in &sim.particles.list {
        max_y = max_y.max(p.position.y);
        max_x = max_x.max(p.position.x);

        let region = (p.position.x / cell_size / 8.0) as usize;
        if region < 6 {
            region_data[region].0 = region_data[region].0.max(p.position.y);
            region_data[region].1 += 1;
        }
    }

    let regions: Vec<(usize, f32, usize)> = region_data
        .iter()
        .enumerate()
        .map(|(i, (h, c))| (i, h / cell_size, *c))
        .collect();

    (max_y / cell_size, max_x / cell_size, regions)
}

fn debug_velocities_at_riffle(sim: &FlipSimulation3D) {
    let cell_size = CELL_SIZE;
    let k = GRID_DEPTH / 2; // Middle Z

    // First, find where particles ACTUALLY are
    println!("\n=== DEBUG: Where are the particles? ===");
    let mut x_dist = [0u32; 12];
    let mut y_dist = [0u32; 12];
    for p in &sim.particles.list {
        let xi = (p.position.x / cell_size).min(11.0) as usize;
        let yi = (p.position.y / cell_size).min(11.0) as usize;
        x_dist[xi] += 1;
        y_dist[yi] += 1;
    }
    print!("X distribution: ");
    for (i, c) in x_dist.iter().enumerate() {
        if *c > 0 { print!("X{}:{} ", i, c); }
    }
    println!();
    print!("Y distribution: ");
    for (i, c) in y_dist.iter().enumerate() {
        if *c > 0 { print!("Y{}:{} ", i, c); }
    }
    println!();

    // Sample a few actual particle positions
    println!("\nSample particles:");
    for (i, p) in sim.particles.list.iter().take(10).enumerate() {
        let xi = (p.position.x / cell_size) as usize;
        let yi = (p.position.y / cell_size) as usize;
        let zi = (p.position.z / cell_size) as usize;
        println!("  {}: pos=({:.2},{:.2},{:.2}) cell=({},{},{})",
            i, p.position.x, p.position.y, p.position.z, xi, yi, zi);
    }

    // Now check cell types near water level
    println!("\n=== Cell types at water height (j=4-9, X=1-8, k={}) ===", k);
    for j in (4..=9).rev() {
        print!("j={:2}: ", j);
        for i in 1..=8 {
            let idx = sim.grid.cell_index(i, j, k);
            let ct = sim.grid.cell_type[idx];
            let ct_char = match ct {
                CellType::Solid => 'S',
                CellType::Fluid => 'F',
                CellType::Air => '.',
            };
            print!("{}", ct_char);
        }
        println!();
    }

    // Check cell types at other Z slices
    println!("\n=== Cell types at different Z slices (j=6, X=1-8) ===");
    for kk in 1..GRID_DEPTH-1 {
        print!("k={}: ", kk);
        for i in 1..=8 {
            let idx = sim.grid.cell_index(i, 6, kk);
            let ct = sim.grid.cell_type[idx];
            let ct_char = match ct {
                CellType::Solid => 'S',
                CellType::Fluid => 'F',
                CellType::Air => '.',
            };
            print!("{}", ct_char);
        }
        println!();
    }

    // Velocities near the first riffle
    println!("\n=== Velocities near first riffle (X=6-10, j=5-8) ===");
    for j in (5..=8).rev() {
        print!("j={}: ", j);
        for i in 6..=10 {
            let idx = sim.grid.cell_index(i, j, k);
            let ct = sim.grid.cell_type[idx];
            let ct_char = match ct {
                CellType::Solid => 'S',
                CellType::Fluid => 'F',
                CellType::Air => '.',
            };

            let u_idx = sim.grid.u_index(i + 1, j, k);
            let u_val = sim.grid.u[u_idx];
            let v_idx = sim.grid.v_index(i, j + 1, k);
            let v_val = sim.grid.v[v_idx];

            print!("[{} u={:+.2} v={:+.2}] ", ct_char, u_val, v_val);
        }
        println!();
    }

    // Debug divergence and pressure in fluid region
    println!("\n=== Divergence and Pressure (X=3-7, j=4-6) ===");
    for j in (4..=6).rev() {
        print!("j={}: ", j);
        for i in 3..=7 {
            let idx = sim.grid.cell_index(i, j, k);
            let ct = sim.grid.cell_type[idx];
            let ct_char = match ct {
                CellType::Solid => 'S',
                CellType::Fluid => 'F',
                CellType::Air => '.',
            };
            let div = sim.grid.divergence[idx];
            let p = sim.grid.pressure[idx];
            print!("[{} d={:+.3} p={:+.1}] ", ct_char, div, p);
        }
        println!();
    }

    // Count particles per Y-layer more precisely
    println!("\n=== Particle count per Y-cell ===");
    let mut y_counts = vec![0u32; GRID_HEIGHT];
    for p in &sim.particles.list {
        let j = ((p.position.y / cell_size) as usize).min(GRID_HEIGHT - 1);
        y_counts[j] += 1;
    }
    for (j, count) in y_counts.iter().enumerate() {
        if *count > 0 {
            println!("  Y{}: {} particles", j, count);
        }
    }
}

fn main() {
    println!("=== Overflow Test: 3-cell tall riffles ===\n");

    let mut sim = FlipSimulation3D::new(GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE);
    // Tilted gravity: sluice slopes from j=4 (left) to j=1 (right) over 48 cells
    // Angle ≈ atan(3/48) ≈ 3.6 degrees, sin(angle) ≈ 0.0625
    // gravity_x = 9.8 * sin(angle) ≈ 0.6 m/s² (pushes water downstream)
    sim.gravity = Vec3::new(0.6, -9.8, 0.0);
    sim.flip_ratio = 0.97;
    sim.pressure_iterations = 100;  // Reduced - issue was flow setup, not convergence

    create_sluice_with_tall_riffles(&mut sim);

    let dt = 1.0 / 60.0;
    let mut max_particles_seen = 0usize;

    // Run for 600 frames (10 seconds) with slow particle emission
    for frame in 0..600 {
        // Emit only 20 particles per frame - slow fill to let pressure equilibrate
        if sim.particles.len() < 20000 {
            emit_particles(&mut sim, 20);
        }

        sim.update(dt);
        max_particles_seen = max_particles_seen.max(sim.particles.len());

        // Debug: check cells where particles are (j=4-8)
        if frame == 10 || frame == 50 || frame == 100 {
            let k = GRID_DEPTH / 2;
            println!("Frame {} debug:", frame);
            for j in (4..=8).rev() {
                print!("  j={}: ", j);
                for i in 1..=7 {
                    let idx = sim.grid.cell_index(i, j, k);
                    let ct = match sim.grid.cell_type[idx] {
                        CellType::Solid => 'S',
                        CellType::Fluid => 'F',
                        CellType::Air => '.',
                    };
                    let p = sim.grid.pressure[idx];
                    let d = sim.grid.divergence[idx];
                    print!("[{} d={:+.2} p={:.2}] ", ct, d, p);
                }
                println!();
            }
        }

        // Print diagnostics every 60 frames (1 second)
        if frame % 60 == 0 {
            let (max_y_cell, max_x_cell, regions) = analyze_water_height(&sim);

            println!("Frame {:4} | Particles: {:5} | MaxY: {:5.1} cells | MaxX: {:5.1} cells",
                     frame, sim.particles.len(), max_y_cell, max_x_cell);

            // Print region breakdown
            print!("  Regions: ");
            for (i, max_h, count) in &regions {
                let overflow = if *max_h > 7.0 { "↑" } else { " " };
                print!("[X{}-{}: h={:.1}{} n={}] ", i*8, (i+1)*8, max_h, overflow, count);
            }
            println!();

            // Check if water is getting past the first riffle (x=8-9)
            let first_riffle_region = 1; // X=8-15
            if regions.len() > first_riffle_region {
                let (_, h, _) = regions[first_riffle_region];
                if h > 7.0 {
                    println!("  ✓ Water ABOVE first riffle (height {} > 7)", h);
                } else if h > 0.0 {
                    println!("  ✗ Water at first riffle but not overflowing (height {} <= 7)", h);
                }
            }
        }
    }

    // Debug velocities at riffle BEFORE final summary
    debug_velocities_at_riffle(&sim);

    println!("\n=== Final Summary ===");
    println!("Max particles seen: {}", max_particles_seen);
    println!("Final particle count: {}", sim.particles.len());

    let (max_y_cell, max_x_cell, regions) = analyze_water_height(&sim);
    println!("Final max Y: {:.1} cells (riffle top at 7)", max_y_cell);
    println!("Final max X: {:.1} cells", max_x_cell);

    // Check if overflow happened
    if max_x_cell > 16.0 {
        println!("\n✓ SUCCESS: Water flowed past first few riffles");
    } else if max_y_cell > 7.0 {
        println!("\n⚠ PARTIAL: Water rose above riffle height but didn't flow far");
    } else {
        println!("\n✗ FAILURE: Water never exceeded riffle height");
    }
}

//! GPU DEM Settling Regression Tests
//!
//! These tests verify the core DEM physics behavior to prevent regression:
//! 1. Angle of repose - particles spread laterally, not stack vertically
//! 2. Settling velocity - particles come to rest, no perpetual vibration
//!
//! Run with: cargo test -p game --test gpu_dem_settling -- --nocapture

use game::gpu::dem::GpuDemSolver;
use glam::Vec2;
use sim::flip::FlipSimulation;
use sim::particle::{Particle, ParticleMaterial, Particles};
use std::sync::Arc;

const CELL_SIZE: f32 = 2.0;
const WIDTH: usize = 160;
const HEIGHT: usize = 120;
const DT: f32 = 1.0 / 60.0;
const GRAVITY: f32 = 200.0;

// Test thresholds - these define "working" behavior
const MIN_HORIZONTAL_SPREAD: f32 = 30.0; // Particles must spread at least this much
const MIN_ASPECT_RATIO: f32 = 0.5; // spread/height must be > this (pile, not tower)
const MAX_SETTLED_SPEED: f32 = 1.0; // Average speed must be below this when settled
const MAX_SETTLING_FRAMES: usize = 300; // Must settle within this many frames

struct TestGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

async fn create_test_gpu() -> Option<TestGpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;

    // Request higher limits for DEM solver (needs 10+ storage buffers)
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffers_per_shader_stage = 16;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("DEM Test Device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .ok()?;

    Some(TestGpu {
        device: Arc::new(device),
        queue: Arc::new(queue),
    })
}

fn build_box(sim: &mut FlipSimulation) {
    let w = sim.grid.width;
    let h = sim.grid.height;

    // Floor (2 cells thick)
    for i in 0..w {
        sim.grid.set_solid(i, h - 1);
        sim.grid.set_solid(i, h - 2);
    }

    // Left wall
    for j in 0..h {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(1, j);
    }

    // Right wall
    for j in 0..h {
        sim.grid.set_solid(w - 1, j);
        sim.grid.set_solid(w - 2, j);
    }

    sim.grid.compute_sdf();
}

struct PileStats {
    x_spread: f32,
    y_range: f32,
    avg_speed: f32,
    max_speed: f32,
}

fn compute_pile_stats(particles: &Particles) -> PileStats {
    let mut x_min = f32::MAX;
    let mut x_max = f32::MIN;
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;
    let mut total_speed = 0.0;
    let mut max_speed = 0.0f32;
    let mut count = 0;

    for p in particles.iter() {
        if !p.is_sediment() {
            continue;
        }
        x_min = x_min.min(p.position.x);
        x_max = x_max.max(p.position.x);
        y_min = y_min.min(p.position.y);
        y_max = y_max.max(p.position.y);

        let speed = p.velocity.length();
        total_speed += speed;
        max_speed = max_speed.max(speed);
        count += 1;
    }

    if count == 0 {
        return PileStats {
            x_spread: 0.0,
            y_range: 0.0,
            avg_speed: 0.0,
            max_speed: 0.0,
        };
    }

    PileStats {
        x_spread: x_max - x_min,
        y_range: y_max - y_min,
        avg_speed: total_speed / count as f32,
        max_speed,
    }
}

/// Test 1: Angle of Repose
///
/// When particles are poured in a narrow column, they should spread laterally
/// to form a pile with a natural angle of repose, NOT stack in a vertical tower.
///
/// This tests:
/// - Surface roughness (collision normal perturbation)
/// - Tangential friction
/// - Proper collision resolution
#[test]
fn test_angle_of_repose() {
    let result = pollster::block_on(async {
        let Some(gpu) = create_test_gpu().await else {
            println!("SKIP: No GPU adapter available");
            return None;
        };

        let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
        build_box(&mut sim);

        let mut dem =
            GpuDemSolver::new_headless(&gpu.device, &gpu.queue, WIDTH as u32, HEIGHT as u32, 10_000);

        // Pour particles in a narrow column (3 wide)
        let center_x = (WIDTH as f32 / 2.0) * CELL_SIZE;
        let drop_y = 20.0 * CELL_SIZE;
        let spacing = CELL_SIZE * 0.7;
        let num_particles = 200;
        let particles_per_row = 3;
        let rows = num_particles / particles_per_row;

        for row in 0..rows {
            for col in 0..particles_per_row {
                let x = center_x + (col as f32 - 1.0) * spacing;
                let y = drop_y + row as f32 * spacing;
                sim.particles.list.push(Particle::new(
                    Vec2::new(x, y),
                    Vec2::ZERO,
                    ParticleMaterial::Sand,
                ));
            }
        }

        // Run simulation for 300 frames
        for _ in 0..300 {
            sim.grid.compute_sdf();
            dem.upload_sdf_headless(&gpu.device, &gpu.queue, &sim.grid.sdf);
            dem.execute_headless(
                &gpu.device,
                &gpu.queue,
                &mut sim.particles,
                CELL_SIZE,
                DT,
                GRAVITY,
                -1.0,
            );
        }

        Some(compute_pile_stats(&sim.particles))
    });

    let Some(stats) = result else {
        return; // Skip if no GPU
    };

    let aspect_ratio = stats.x_spread / stats.y_range.max(1.0);

    println!("=== Angle of Repose Test ===");
    println!("  Horizontal spread: {:.1} (min: {})", stats.x_spread, MIN_HORIZONTAL_SPREAD);
    println!("  Vertical range: {:.1}", stats.y_range);
    println!("  Aspect ratio: {:.2} (min: {})", aspect_ratio, MIN_ASPECT_RATIO);
    println!("  Final avg speed: {:.3}", stats.avg_speed);

    assert!(
        stats.x_spread >= MIN_HORIZONTAL_SPREAD,
        "Particles stacking vertically! spread={:.1} < {} - angle of repose broken",
        stats.x_spread,
        MIN_HORIZONTAL_SPREAD
    );

    assert!(
        aspect_ratio >= MIN_ASPECT_RATIO,
        "Pile too vertical! aspect_ratio={:.2} < {} - particles not spreading",
        aspect_ratio,
        MIN_ASPECT_RATIO
    );
}

/// Test 2: Settling Velocity
///
/// Particles dropped onto a floor should come to rest within a reasonable
/// number of frames, with zero (or near-zero) velocity.
///
/// This tests:
/// - Floor collision with slop (penetration tolerance)
/// - Sleep system
/// - No perpetual vibration
#[test]
fn test_settling_velocity() {
    let result = pollster::block_on(async {
        let Some(gpu) = create_test_gpu().await else {
            println!("SKIP: No GPU adapter available");
            return None;
        };

        let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
        build_box(&mut sim);

        let mut dem =
            GpuDemSolver::new_headless(&gpu.device, &gpu.queue, WIDTH as u32, HEIGHT as u32, 10_000);

        // Drop a row of particles
        let y = 50.0 * CELL_SIZE;
        for i in 0..20 {
            let x = (40.0 + i as f32 * 2.0) * CELL_SIZE;
            sim.particles.list.push(Particle::new(
                Vec2::new(x, y),
                Vec2::ZERO,
                ParticleMaterial::Sand,
            ));
        }

        // Track when we settle
        let mut settled_frame = None;
        let mut settled_count = 0;

        for frame in 0..MAX_SETTLING_FRAMES {
            sim.grid.compute_sdf();
            dem.upload_sdf_headless(&gpu.device, &gpu.queue, &sim.grid.sdf);
            dem.execute_headless(
                &gpu.device,
                &gpu.queue,
                &mut sim.particles,
                CELL_SIZE,
                DT,
                GRAVITY,
                -1.0,
            );

            let stats = compute_pile_stats(&sim.particles);

            // Check if settled (30 consecutive frames of low velocity)
            if stats.avg_speed < MAX_SETTLED_SPEED && stats.max_speed < MAX_SETTLED_SPEED * 2.0 {
                settled_count += 1;
                if settled_count >= 30 && settled_frame.is_none() {
                    settled_frame = Some(frame);
                }
            } else {
                settled_count = 0;
            }
        }

        let final_stats = compute_pile_stats(&sim.particles);
        Some((settled_frame, final_stats))
    });

    let Some((settled_frame, stats)) = result else {
        return; // Skip if no GPU
    };

    println!("=== Settling Velocity Test ===");
    println!("  Settled at frame: {:?}", settled_frame);
    println!("  Final avg speed: {:.4} (max: {})", stats.avg_speed, MAX_SETTLED_SPEED);
    println!("  Final max speed: {:.4}", stats.max_speed);

    assert!(
        settled_frame.is_some(),
        "Particles never settled! Still moving after {} frames (avg_speed={:.3})",
        MAX_SETTLING_FRAMES,
        stats.avg_speed
    );

    assert!(
        stats.avg_speed < MAX_SETTLED_SPEED,
        "Particles still vibrating! avg_speed={:.3} > {}",
        stats.avg_speed,
        MAX_SETTLED_SPEED
    );
}

/// Test 3: No Mid-Air Pause
///
/// Particles falling through air should accelerate continuously under gravity.
/// They should NOT pause or slow down before hitting something.
///
/// This tests:
/// - Sleep system doesn't affect falling particles
/// - Support propagation only from floor
#[test]
fn test_no_midair_pause() {
    let result = pollster::block_on(async {
        let Some(gpu) = create_test_gpu().await else {
            println!("SKIP: No GPU adapter available");
            return None;
        };

        let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
        build_box(&mut sim);

        let mut dem =
            GpuDemSolver::new_headless(&gpu.device, &gpu.queue, WIDTH as u32, HEIGHT as u32, 10_000);

        // Single particle high up
        let start_y = 20.0 * CELL_SIZE;
        sim.particles.list.push(Particle::new(
            Vec2::new((WIDTH as f32 / 2.0) * CELL_SIZE, start_y),
            Vec2::ZERO,
            ParticleMaterial::Sand,
        ));

        // Track velocity over time - should increase monotonically until floor
        let mut velocities = Vec::new();
        let mut hit_floor = false;

        for frame in 0..120 {
            sim.grid.compute_sdf();
            dem.upload_sdf_headless(&gpu.device, &gpu.queue, &sim.grid.sdf);
            dem.execute_headless(
                &gpu.device,
                &gpu.queue,
                &mut sim.particles,
                CELL_SIZE,
                DT,
                GRAVITY,
                -1.0,
            );

            let p = &sim.particles.list[0];
            let vy = p.velocity.y;
            velocities.push(vy);

            // Check if we hit floor (velocity suddenly drops or reverses)
            if vy < velocities.get(velocities.len().saturating_sub(2)).copied().unwrap_or(0.0) * 0.5 {
                hit_floor = true;
                break;
            }
        }

        Some((velocities, hit_floor))
    });

    let Some((velocities, hit_floor)) = result else {
        return; // Skip if no GPU
    };

    println!("=== No Mid-Air Pause Test ===");
    println!("  Hit floor: {}", hit_floor);
    println!("  Velocity samples: {:?}", &velocities[..velocities.len().min(10)]);

    // Check that velocity increased monotonically until floor
    let mut had_pause = false;
    for i in 1..velocities.len() {
        if velocities[i] < velocities[i - 1] * 0.9 && !hit_floor {
            // Velocity decreased significantly before hitting floor = mid-air pause
            had_pause = true;
            println!("  PAUSE at frame {}: vy dropped from {:.2} to {:.2}",
                     i, velocities[i - 1], velocities[i]);
        }
    }

    assert!(
        !had_pause || hit_floor,
        "Particle paused mid-air! This indicates broken support propagation."
    );
}

/// Test 4: Particles Near Vertical Walls Don't Get Stuck
///
/// Particles falling next to a vertical wall should fall normally and reach
/// the floor, not pause mid-air or get stuck to the wall.
///
/// This tests:
/// - Gradient-based floor detection correctly identifies walls as NOT floors
/// - Sleep system doesn't trigger near walls (grad.y < 0.5 for vertical surfaces)
#[test]
fn test_vertical_wall_no_stuck() {
    let result = pollster::block_on(async {
        let Some(gpu) = create_test_gpu().await else {
            println!("SKIP: No GPU adapter available");
            return None;
        };

        let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
        build_box(&mut sim);
        let mut dem =
            GpuDemSolver::new_headless(&gpu.device, &gpu.queue, WIDTH as u32, HEIGHT as u32, 1000);

        // Drop particles RIGHT NEXT to the left wall (within 1 cell)
        let wall_x = 3.0 * CELL_SIZE; // Just past the 2-cell thick wall
        let drop_y = 20.0 * CELL_SIZE;
        let floor_y = (HEIGHT as f32 - 3.0) * CELL_SIZE;

        for i in 0..5 {
            sim.particles.list.push(Particle::new(
                Vec2::new(wall_x + i as f32 * CELL_SIZE * 0.5, drop_y + i as f32 * CELL_SIZE),
                Vec2::ZERO,
                ParticleMaterial::Sand,
            ));
        }

        // Run simulation for 200 frames - enough time to fall
        for _ in 0..200 {
            sim.grid.compute_sdf();
            dem.upload_sdf_headless(&gpu.device, &gpu.queue, &sim.grid.sdf);
            dem.execute_headless(
                &gpu.device,
                &gpu.queue,
                &mut sim.particles,
                CELL_SIZE,
                DT,
                GRAVITY,
                -1.0,
            );
        }

        // Check all particles reached floor (not stuck mid-air)
        let all_reached_floor = sim.particles.iter().all(|p| {
            p.position.y > floor_y - CELL_SIZE * 2.0 // Within 2 cells of floor
        });

        let min_y = sim.particles.iter().map(|p| p.position.y).fold(f32::MAX, f32::min);
        let max_y = sim.particles.iter().map(|p| p.position.y).fold(f32::MIN, f32::max);

        Some((all_reached_floor, min_y, max_y, floor_y))
    });

    let Some((all_reached, min_y, max_y, floor_y)) = result else {
        return;
    };

    println!("=== Vertical Wall No Stuck Test ===");
    println!("  Floor Y: {:.1}", floor_y);
    println!("  Particle Y range: {:.1} - {:.1}", min_y, max_y);
    println!("  All reached floor: {}", all_reached);

    assert!(
        all_reached,
        "Particles stuck mid-air near wall! Y range: {:.1}-{:.1}, floor at {:.1}",
        min_y, max_y, floor_y
    );
}

/// Test 5: Static Particles Are Frozen
///
/// Particles marked as STATIC should not move regardless of physics.
/// This is the foundation of the two-state model where settled piles act as solids.
///
/// This tests:
/// - Static particles skip all physics integration
/// - Dynamic particles still fall normally
/// - Position remains unchanged for static particles over 100 frames
#[test]
fn test_static_particles_frozen() {
    let result = pollster::block_on(async {
        let Some(gpu) = create_test_gpu().await else {
            println!("SKIP: No GPU adapter available");
            return None;
        };

        let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
        build_box(&mut sim);
        let mut dem =
            GpuDemSolver::new_headless(&gpu.device, &gpu.queue, WIDTH as u32, HEIGHT as u32, 1000);

        // Create 10 particles: 5 static (even indices), 5 dynamic (odd indices)
        // Place them mid-air so dynamic ones would fall
        let start_y = 20.0 * CELL_SIZE;
        for i in 0..10 {
            sim.particles.list.push(Particle::new(
                Vec2::new(80.0 * CELL_SIZE + i as f32 * CELL_SIZE * 2.0, start_y),
                Vec2::ZERO,
                ParticleMaterial::Sand,
            ));
        }

        // Record initial positions
        let initial_positions: Vec<Vec2> = sim.particles.iter().map(|p| p.position).collect();

        // Upload static states: even indices = static (1), odd = dynamic (0)
        let mut static_states = vec![0u32; 10];
        for i in (0..10).step_by(2) {
            static_states[i] = 1; // Mark even indices as static
        }
        dem.upload_static_states_headless(&gpu.queue, &static_states);

        // Run 100 frames
        for _ in 0..100 {
            sim.grid.compute_sdf();
            dem.upload_sdf_headless(&gpu.device, &gpu.queue, &sim.grid.sdf);
            dem.execute_headless(
                &gpu.device,
                &gpu.queue,
                &mut sim.particles,
                CELL_SIZE,
                DT,
                GRAVITY,
                -1.0,
            );
        }

        // Check positions
        let final_positions: Vec<Vec2> = sim.particles.iter().map(|p| p.position).collect();
        Some((initial_positions, final_positions))
    });

    let Some((initial, final_pos)) = result else {
        return;
    };

    println!("=== Static Particles Frozen Test ===");

    // Static particles (even indices) should have ZERO position change
    let mut static_moved = false;
    for i in (0..10).step_by(2) {
        let delta = (final_pos[i] - initial[i]).length();
        println!("  Static particle {}: delta = {:.4}", i, delta);
        if delta > 0.001 {
            static_moved = true;
            println!("    ERROR: Static particle moved!");
        }
    }

    // Dynamic particles (odd indices) should have fallen
    let mut dynamic_fell = true;
    for i in (1..10).step_by(2) {
        let delta_y = final_pos[i].y - initial[i].y;
        println!("  Dynamic particle {}: delta_y = {:.2}", i, delta_y);
        if delta_y < 10.0 {
            dynamic_fell = false;
            println!("    ERROR: Dynamic particle didn't fall enough!");
        }
    }

    assert!(
        !static_moved,
        "Static particles moved! They should be completely frozen."
    );
    assert!(
        dynamic_fell,
        "Dynamic particles didn't fall! Gravity should still work for dynamic particles."
    );
}

/// Test 6: Settling Particles Become Static
///
/// Particles that settle on the floor should transition from DYNAMIC to STATIC.
/// This is Level 3 of the two-state model implementation.
///
/// This tests:
/// - Dynamic particles transition to STATIC when slow + supported
/// - Static particles have zero velocity
/// - Transition happens within reasonable time
#[test]
fn test_settling_becomes_static() {
    let result = pollster::block_on(async {
        let Some(gpu) = create_test_gpu().await else {
            println!("SKIP: No GPU adapter available");
            return None;
        };

        let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
        build_box(&mut sim);
        let mut dem =
            GpuDemSolver::new_headless(&gpu.device, &gpu.queue, WIDTH as u32, HEIGHT as u32, 1000);

        // Drop 50 particles onto floor
        let drop_y = 30.0 * CELL_SIZE;
        for i in 0..50 {
            let x = 60.0 * CELL_SIZE + (i % 10) as f32 * CELL_SIZE * 2.0;
            let y = drop_y + (i / 10) as f32 * CELL_SIZE * 2.0;
            sim.particles.list.push(Particle::new(
                Vec2::new(x, y),
                Vec2::ZERO,
                ParticleMaterial::Sand,
            ));
        }

        // Run 300 frames (5 seconds at 60fps)
        for frame in 0..300 {
            sim.grid.compute_sdf();
            dem.upload_sdf_headless(&gpu.device, &gpu.queue, &sim.grid.sdf);
            dem.execute_headless(
                &gpu.device,
                &gpu.queue,
                &mut sim.particles,
                CELL_SIZE,
                DT,
                GRAVITY,
                -1.0,
            );

            // Debug: print state at key frames
            if frame == 100 || frame == 200 || frame == 299 {
                let states = dem.download_static_states_headless(&gpu.device, &gpu.queue, 50);
                let static_count = states.iter().filter(|&&s| s == 1).count();
                let avg_vel: f32 = sim.particles.iter().take(50).map(|p| p.velocity.length()).sum::<f32>() / 50.0;
                let avg_y: f32 = sim.particles.iter().take(50).map(|p| p.position.y).sum::<f32>() / 50.0;
                println!("  Frame {}: static={}/50, avg_vel={:.2}, avg_y={:.1}", frame, static_count, avg_vel, avg_y);
            }
        }

        // Download static states
        let static_states = dem.download_static_states_headless(&gpu.device, &gpu.queue, 50);

        // Count how many are static
        let static_count = static_states.iter().filter(|&&s| s == 1).count();

        // Get velocities of static particles
        let velocities: Vec<f32> = sim.particles.iter()
            .take(50)
            .enumerate()
            .filter(|(i, _)| static_states.get(*i).copied() == Some(1))
            .map(|(_, p)| p.velocity.length())
            .collect();

        let max_static_velocity = velocities.iter().copied().fold(0.0f32, f32::max);

        Some((static_count, max_static_velocity))
    });

    let Some((static_count, max_vel)) = result else {
        return;
    };

    println!("=== Settling Becomes Static Test ===");
    println!("  Static particles: {}/50 ({:.0}%)", static_count, static_count as f32 / 50.0 * 100.0);
    println!("  Max velocity of static particles: {:.4}", max_vel);

    // At least 80% should be static
    assert!(
        static_count >= 40,
        "Only {}/50 particles became static! Expected at least 40 (80%)",
        static_count
    );

    // Static particles should have near-zero velocity
    assert!(
        max_vel < 0.1,
        "Static particles have velocity {}! Expected < 0.1",
        max_vel
    );
}

// NOTE: Tests for "gold falls faster" and "gold sinks through sand" were REMOVED
// because they tested for physically incorrect behavior:
//
// 1. In vacuum/no-drag, ALL objects fall at the same acceleration (g=9.8m/s²)
//    Terminal velocity differences only exist with air resistance.
//
// 2. Settled granular material acts as a SOLID. Gold sitting on a sand pile
//    stays on top. Density stratification only occurs with AGITATION
//    (water flow, vibration, shaking table).

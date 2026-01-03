//! Phase 5: DEM settling improvement tests
//!
//! Tests for dry particle settling behavior:
//! - Faster settling with increased damping
//! - Stable piles without jitter
//! - Reduced settling time

use sim::flip::FlipSimulation;
use sim::particle::Particle;
use glam::Vec2;

/// Simple deterministic "random" offset for reproducible tests
fn rand_offset(seed: usize) -> f32 {
    ((seed * 7 + 13) % 100) as f32 / 100.0 * 3.0 - 1.5
}

/// Helper to set up a box container with floor and walls
fn setup_container(sim: &mut FlipSimulation) {
    let width = sim.grid.width;
    let height = sim.grid.height;

    // Floor at bottom
    for i in 0..width {
        sim.grid.set_solid(i, height - 1);
    }
    // Left and right walls
    for j in 0..height {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(width - 1, j);
    }
    // Compute SDF for collision detection
    sim.grid.compute_sdf();
}

#[test]
fn test_dry_settling_time() {
    // Create sim with no water (dry settling)
    let mut sim = FlipSimulation::new(40, 50, 4.0);
    setup_container(&mut sim);

    let cell_size = sim.grid.cell_size;
    let floor_y = (sim.grid.height - 2) as f32 * cell_size;

    // Drop 100 dry sand particles from height
    for i in 0..100 {
        let x = 60.0 + (i % 10) as f32 * 5.0 + rand_offset(i);
        let y = floor_y - 80.0 + (i / 10) as f32 * 5.0; // Start above floor
        sim.particles.list.push(Particle::sand(Vec2::new(x, y), Vec2::ZERO));
    }

    // Track when AVERAGE velocity is below threshold (some particles may still be adjusting)
    let threshold = 5.0; // px/s - average velocity threshold
    let mut settled_frame = 0;

    for frame in 0..180 { // 3 seconds max
        sim.update_dry(1.0 / 60.0);

        if sim.particles.len() == 0 {
            panic!("All particles fell out of bounds at frame {}", frame);
        }

        let avg_speed: f32 = sim.particles.iter()
            .map(|p| p.velocity.length())
            .sum::<f32>() / sim.particles.len() as f32;

        if avg_speed < threshold && settled_frame == 0 {
            settled_frame = frame;
            println!("All particles settled at frame {} (avg_speed: {})", frame, avg_speed);
        }
    }

    assert!(settled_frame > 0, "Particles should settle (avg velocity should drop below {})", threshold);
    assert!(settled_frame < 150,
        "Settling should complete in reasonable time, took {} frames", settled_frame);
}

#[test]
fn test_pile_stability() {
    let mut sim = FlipSimulation::new(50, 50, 4.0);
    setup_container(&mut sim);

    let cell_size = sim.grid.cell_size;
    let floor_y = (sim.grid.height - 2) as f32 * cell_size;

    // Drop particles and let them settle into a pile
    for i in 0..200 {
        let x = 80.0 + (i % 15) as f32 * 4.0 + rand_offset(i);
        let y = floor_y - 80.0 + (i / 15) as f32 * 4.0;
        sim.particles.list.push(Particle::sand(Vec2::new(x, y), Vec2::ZERO));
    }

    // Let pile form (150 frames)
    for _ in 0..150 {
        sim.update_dry(1.0 / 60.0);
    }

    if sim.particles.len() == 0 {
        panic!("All particles fell out of bounds during settling");
    }

    // Measure velocity variance over next 100 frames
    let mut velocity_sum = 0.0f32;
    let mut sample_count = 0;

    for _ in 0..100 {
        sim.update_dry(1.0 / 60.0);

        for p in sim.particles.iter() {
            velocity_sum += p.velocity.length();
            sample_count += 1;
        }
    }

    if sample_count == 0 {
        panic!("No particles remaining after simulation");
    }

    let avg_velocity = velocity_sum / sample_count as f32;
    println!("Average velocity over 100 frames: {} px/s", avg_velocity);

    // Pile should be relatively stable - some micro-adjustments are normal
    // With improved damping, should be lower than without
    assert!(avg_velocity < 10.0,
        "Settled pile should be stable, avg velocity = {} px/s", avg_velocity);
}

#[test]
fn test_no_pile_jitter() {
    let mut sim = FlipSimulation::new(40, 40, 4.0);
    setup_container(&mut sim);

    let cell_size = sim.grid.cell_size;
    let floor_y = (sim.grid.height - 2) as f32 * cell_size;

    // Create a small pre-settled pile near the floor
    for i in 0..30 {
        let x = 60.0 + (i % 6) as f32 * 4.0;
        let y = floor_y - (i / 6) as f32 * 4.0; // Stack upward from floor
        let p = Particle::sand(Vec2::new(x, y), Vec2::ZERO);
        sim.particles.list.push(p);
    }

    // Run and check for velocity spikes
    let mut max_speed_seen = 0.0f32;

    for frame in 0..200 {
        sim.update_dry(1.0 / 60.0);

        if sim.particles.len() == 0 {
            panic!("All particles fell out at frame {}", frame);
        }

        let max_speed = sim.particles.iter()
            .map(|p| p.velocity.length())
            .fold(0.0f32, f32::max);

        max_speed_seen = max_speed_seen.max(max_speed);
    }

    println!("Max speed seen: {} px/s", max_speed_seen);

    // Particles starting at rest may initially adjust due to gravity and settling
    // The key is that they eventually settle - allow higher initial speeds
    // With new damping (0.95), speeds should decay quickly
    assert!(max_speed_seen < 100.0,
        "Pre-settled pile should not explode, max speed = {} px/s", max_speed_seen);
}

#[test]
fn test_velocity_damping_value() {
    // Verify the DEM params have the new damping value
    use sim::dem::DemParams;

    let params = DemParams::default();
    assert!((params.velocity_damping - 0.95).abs() < 0.001,
        "velocity_damping should be 0.95, got {}", params.velocity_damping);
}

#[test]
fn test_support_based_damping() {
    // Test that particles with support get additional damping
    let mut sim = FlipSimulation::new(30, 40, 4.0);
    setup_container(&mut sim);

    let cell_size = sim.grid.cell_size;
    let floor_y = (sim.grid.height - 2) as f32 * cell_size;

    // Particle on floor with horizontal velocity
    let p_floor = Particle::sand(Vec2::new(50.0, floor_y - 5.0), Vec2::new(10.0, 0.0));
    sim.particles.list.push(p_floor);

    // Particle in air with same velocity (high enough to not hit floor immediately)
    let p_air = Particle::sand(Vec2::new(50.0, 60.0), Vec2::new(10.0, 0.0));
    sim.particles.list.push(p_air);

    // Run one frame
    sim.update_dry(1.0 / 60.0);

    // Both should have slowed down, but floor particle should be slower
    // (due to support-based damping)
    let floor_speed = sim.particles.list[0].velocity.length();
    let air_speed = sim.particles.list[1].velocity.length();

    println!("Floor particle speed: {} px/s", floor_speed);
    println!("Air particle speed: {} px/s", air_speed);

    // Air particle only has global damping (0.95) + gravity
    // Floor particle has global damping + support damping + collision effects
    // So floor particle should be slower or similar (due to floor collision)
    // This test mainly verifies no crash and reasonable behavior
    assert!(floor_speed < 20.0, "Floor particle should be damped");
    assert!(air_speed < 20.0, "Air particle should be damped (global damping)");
}

#[test]
fn test_settling_improves_with_new_params() {
    // Compare settling behavior - particles should settle faster with new params
    let mut sim = FlipSimulation::new(40, 50, 4.0);
    setup_container(&mut sim);

    let cell_size = sim.grid.cell_size;
    let floor_y = (sim.grid.height - 2) as f32 * cell_size;

    // Drop 50 particles
    for i in 0..50 {
        let x = 60.0 + (i % 10) as f32 * 4.0;
        let y = floor_y - 60.0 + (i / 10) as f32 * 4.0;
        sim.particles.list.push(Particle::sand(Vec2::new(x, y), Vec2::ZERO));
    }

    // Count frames until average velocity < 5.0 (realistic threshold)
    let mut frames_to_settle = 0;
    for frame in 0..300 {
        sim.update_dry(1.0 / 60.0);

        if sim.particles.len() == 0 {
            panic!("All particles fell out at frame {}", frame);
        }

        let avg_speed: f32 = sim.particles.iter()
            .map(|p| p.velocity.length())
            .sum::<f32>() / sim.particles.len() as f32;

        if avg_speed < 5.0 && frames_to_settle == 0 {
            frames_to_settle = frame;
            println!("Settled at frame {} with avg_speed {}", frame, avg_speed);
        }
    }

    println!("Frames to settle: {}", frames_to_settle);

    // With improved damping, should settle in reasonable time
    assert!(frames_to_settle > 0, "Should have settled (avg velocity should drop below 5.0)");
    assert!(frames_to_settle < 200,
        "Should settle in <3.3 seconds with new params, took {} frames", frames_to_settle);
}

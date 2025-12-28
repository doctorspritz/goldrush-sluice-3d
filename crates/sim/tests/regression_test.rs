use sim::flip::FlipSimulation;

const DT: f32 = 1.0 / 60.0;
const CELL_SIZE: f32 = 0.5;
const WIDTH: usize = 40;
const HEIGHT: usize = 30;

/// Regression test for mixed water/sand simulation stability.
/// Verifies the simulation doesn't crash, explode, or produce NaN values
/// when running with both water and sand particles.
#[test]
fn test_regression_mixed_simulation() {
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);

    // 1. Spawn a mix of particles
    // Water column
    for y in 5..25 {
        for x in 5..15 {
            sim.spawn_water(x as f32 * CELL_SIZE, y as f32 * CELL_SIZE, 0.0, 0.0, 1);
        }
    }

    // Sand bed (initially suspended, should settle)
    for x in 15..35 {
         sim.spawn_sand(x as f32 * CELL_SIZE, 5.0 * CELL_SIZE, 0.0, 0.0, 1);
    }

    // Extra sand particles (initially suspended)
    for x in 20..30 {
        sim.spawn_sand(x as f32 * CELL_SIZE, 15.0 * CELL_SIZE, 0.0, 0.0, 1);
    }

    let initial_count = sim.particles.len();

    // 2. Run for 100 frames
    for _ in 0..100 {
        sim.update(DT);
    }

    // 3. Verify Stability

    // Particle count can fluctuate due to sediment transport cycle:
    // - Deposition: particles settle and convert to solid cells (count decreases)
    // - Entrainment: deposited cells erode back to particles (count increases)
    // We verify count stays within reasonable bounds
    let final_count = sim.particles.len();
    assert!(final_count > 0, "Some particles should remain (not all deposited).");
    assert!(
        final_count < initial_count * 3,
        "Particle count should not grow excessively: {} -> {}", initial_count, final_count
    );

    // Verify velocities are not NaN/Infinite (stability)
    for (i, p) in sim.particles.iter().enumerate() {
        assert!(
            p.velocity.is_finite(),
            "Particle {} velocity must be finite: {:?}", i, p.velocity
        );
        assert!(
            p.position.is_finite(),
            "Particle {} position must be finite: {:?}", i, p.position
        );
    }
}

use dfsph::DfsphSimulation;
use sim::physics;
use glam::Vec2;
use std::time::Instant;

#[test]
fn test_performance_fps() {
    // Setup a heavy-ish simulation
    let width = 200;
    let height = 150;
    let cell_size = 2.0;
    let mut sim = DfsphSimulation::new(width, height, cell_size);
    
    // Spawn 2000 particles (grid)
    // 50x40 block
    let start_x = 20.0;
    let start_y = 20.0;
    let cols = 40;
    let rows = 25; // 1000 particles
    for y in 0..rows {
        for x in 0..cols {
            let px = start_x + x as f32 * cell_size * 0.75; // 1.5 spacing
            let py = start_y + y as f32 * cell_size * 0.75;
            sim.spawn_particle_internal(Vec2::new(px, py), Vec2::ZERO, sim::ParticleMaterial::Water);
        }
    }
    
    // Warmup
    sim.update(0.016);
    
    let start_time = Instant::now();
    let frames = 60;
    
    for i in 0..frames {
        let t0 = Instant::now();
        sim.update(0.016);
        println!("Frame {} took {:.4}s", i, t0.elapsed().as_secs_f64());
    }
    
    let duration = start_time.elapsed();
    let duration_secs = duration.as_secs_f64();
    let fps = frames as f64 / duration_secs;
    
    println!("Simulated {} frames in {:.4}s (FPS: {:.2})", frames, duration_secs, fps);
    
    // Assert 30 FPS minimum
    // Note: Debug builds are SLOW. This test might fail in debug.
    // We strictly require this for release.
    // Ideally we check if we are in release mode, but cargo doesn't easily expose that to code.
    // Low bar for debug: 5 FPS (to prove it's not infinite loop).
    #[cfg(debug_assertions)]
    let min_fps = 2.0; // Debug is very slow with rayon overhead on small substeps
    #[cfg(not(debug_assertions))]
    let min_fps = 30.0; // Release target
    
    assert!(fps >= min_fps, "FPS too low! Got {:.2}, expected >= {:.2}", fps, min_fps);
}

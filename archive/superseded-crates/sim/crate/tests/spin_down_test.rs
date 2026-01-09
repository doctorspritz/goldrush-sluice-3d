//! Regression test for fluid "sluggishness" (effective viscosity)
//! 
//! Verified behavior:
//! A swirling pool of water should maintain its angular momentum for a reasonable time.
//! Rapid spin-down indicates numerical damping (effective viscosity) is too high.

use sim::FlipSimulation;
use glam::Vec2;

#[test]
fn test_spin_down() {
    const WIDTH: usize = 32;
    const HEIGHT: usize = 32;
    const CELL_SIZE: f32 = 4.0;
    const DT: f32 = 1.0 / 60.0;
    
    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    
    // 1. Create a circular container (approx) or just a box
    // Let's use a box for simplicity, but only fill the center
    for i in 0..WIDTH {
        sim.grid.set_solid(i, HEIGHT - 1);
        sim.grid.set_solid(i, 0);
    }
    for j in 0..HEIGHT {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(WIDTH - 1, j);
    }
    sim.grid.compute_sdf();
    
    // 2. Spawn a blob of water in the center
    let center = Vec2::new(WIDTH as f32 * CELL_SIZE * 0.5, HEIGHT as f32 * CELL_SIZE * 0.5);
    let radius = 10.0 * CELL_SIZE;
    
    for i in 5..25 {
        for j in 5..25 {
            let x = i as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            let y = j as f32 * CELL_SIZE + CELL_SIZE * 0.5;
            let pos = Vec2::new(x, y);
            if pos.distance(center) < radius {
                sim.spawn_water(x, y, 0.0, 0.0, 1);
            }
        }
    }
    
    // 3. Apply an initial rotational velocity vortex
    for p in sim.particles.list.iter_mut() {
        let r = p.position - center;
        // Tangent vector (-y, x)
        let tangent = Vec2::new(-r.y, r.x).normalize_or_zero();
        let dist = r.length();
        // Velocity increases with distance (rigid body like) or vortex? 
        // Let's do rigid body rotation 
        let speed = dist * 2.0; // arbitrary angular velocity
        p.velocity = tangent * speed;
    }
    
    // Measure initial angular momentum (approx)
    let initial_am = calculate_angular_momentum(&sim, center);
    println!("Initial AM: {}", initial_am);
    
    // 4. Run for 200 frames
    for i in 0..200 {
        sim.update(DT);
        if i % 50 == 0 {
            let am = calculate_angular_momentum(&sim, center);
            println!("Frame {}: AM = {:.1} ({:.1}% of initial)", 
                i, am, am / initial_am * 100.0);
        }
    }
    
    let final_am = calculate_angular_momentum(&sim, center);
    let retention = final_am / initial_am;
    println!("Final AM retention: {:.1}%", retention * 100.0);
    
    // Pass condition: Should retain significant momentum
    // In a very damped PIC simulation, this might drop to <10%
    // In a lively FLIP/APIC, it should stay >50% (minus wall friction)
    // We'll set a threshold that requires the "fix" to pass, or at least benchmarks it.
    // For now, let's assert > 30% to start.
    assert!(retention > 0.30, 
        "Water spun down too fast! Retention: {:.1}%", retention * 100.0);
}

fn calculate_angular_momentum(sim: &FlipSimulation, center: Vec2) -> f32 {
    let mut am = 0.0;
    for p in sim.particles.iter() {
        let r_vec = p.position - center;
        let v_vec = p.velocity;
        // 2D cross product scalar (r_x * v_y - r_y * v_x)
        let cross = r_vec.x * v_vec.y - r_vec.y * v_vec.x;
        am += cross.abs(); // Use abs sum for magnitude of swirl
    }
    am
}

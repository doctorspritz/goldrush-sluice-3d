// Diagnostic: Single particle momentum through FLIP cycle
use sim::FlipSimulation;
use sim::particle::{Particle, ParticleMaterial};
use glam::Vec2;

fn main() {
    let width = 32;
    let height = 32;
    let cell_size = 1.0;
    let dt = 1.0 / 60.0;
    
    let mut sim = FlipSimulation::new(width, height, cell_size);
    
    // Single particle in center with known velocity
    let pos = Vec2::new(16.5, 16.5);  // Center of cell
    let vel = Vec2::new(100.0, 0.0);
    sim.particles.list.push(Particle::new(pos, vel, ParticleMaterial::Water));
    
    println!("BEFORE: velocity = {:?}, |v| = {:.4}", vel, vel.length());
    
    // Run isolated FLIP cycle (P2G → extrapolate → store_old → G2P)
    sim.run_isolated_flip_cycle_with_extrapolation(dt);
    
    let new_vel = sim.particles.list[0].velocity;
    let old_grid = sim.particles.list[0].old_grid_velocity;
    let ratio = new_vel.length() / vel.length();
    
    println!("\nAFTER:");
    println!("  velocity = {:?}, |v| = {:.4}", new_vel, new_vel.length());
    println!("  old_grid_velocity = {:?}", old_grid);
    println!("  ratio = {:.4}, loss = {:.2}%", ratio, (1.0 - ratio) * 100.0);
    
    // Calculate what we expect from FLIP/PIC blend
    // flip_velocity = particle_velocity + (new_grid - old_grid)
    // If old_grid = new_grid (no forces), flip = particle_velocity
    // final = 0.97 * flip + 0.03 * pic
    // final = 0.97 * particle_velocity + 0.03 * grid_velocity
    let expected_pic_influence = (old_grid - vel) * 0.03;
    println!("\nEXPECTED PIC influence: {:?}", expected_pic_influence);
    println!("If old_grid = vel, no PIC loss should occur");
}

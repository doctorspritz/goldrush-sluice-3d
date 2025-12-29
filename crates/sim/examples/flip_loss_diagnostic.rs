// Diagnostic: Investigate seed 1 momentum gain
use sim::FlipSimulation;
use sim::particle::{Particle, ParticleMaterial};
use glam::Vec2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn total_water_momentum_vector(sim: &FlipSimulation) -> Vec2 {
    sim.particles.iter()
        .filter(|p| p.material == ParticleMaterial::Water)
        .map(|p| p.velocity)
        .fold(Vec2::ZERO, |acc, v| acc + v)
}

fn create_sim_with_random_velocities(seed: u64, particle_count: usize) -> FlipSimulation {
    let width = 32;
    let height = 32;
    let cell_size = 1.0;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut sim = FlipSimulation::new(width, height, cell_size);

    for _ in 0..particle_count {
        let x = rng.gen_range(5.0..27.0);
        let y = rng.gen_range(5.0..27.0);
        let speed = rng.gen_range(1.0..100.0);
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let vel = Vec2::new(speed * angle.cos(), speed * angle.sin());
        let particle = Particle::new(Vec2::new(x, y), vel, ParticleMaterial::Water);
        sim.particles.list.push(particle);
    }
    sim
}

fn main() {
    // Test both seed 0 (passed) and seed 1 (failed)
    for seed in 0..5u64 {
        let mut sim = create_sim_with_random_velocities(seed, 500);
        let before = total_water_momentum_vector(&sim);
        
        sim.run_isolated_flip_cycle_with_extrapolation(1.0/60.0);
        
        let after = total_water_momentum_vector(&sim);
        let delta = after - before;
        let ratio = after.length() / before.length();
        
        println!("Seed {}: before={:.2}, after={:.2}, delta={:?}, ratio={:.4}", 
            seed, before.length(), after.length(), delta, ratio);
    }
    
    println!("\n--- Checking affine velocity contribution ---");
    let mut sim = create_sim_with_random_velocities(1, 500);
    
    // Calculate total affine contribution to P2G
    let mut total_affine = Vec2::ZERO;
    for p in sim.particles.iter() {
        // The affine term C*offset averages out spatially, but might not sum to zero
        // C is typically small for non-rotating flow
        let c_norm = p.affine_velocity.x_axis.length() + p.affine_velocity.y_axis.length();
        total_affine += Vec2::new(c_norm, 0.0);
    }
    println!("Total affine velocity norm: {:?}", total_affine.x);
}

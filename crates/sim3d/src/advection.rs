//! Particle advection and boundary handling for 3D FLIP.

use glam::Vec3;

use crate::grid::Grid3D;
use crate::particle::Particles3D;

/// Advect particles using simple Euler integration.
pub fn advect_particles(particles: &mut Particles3D, dt: f32) {
    for particle in &mut particles.list {
        particle.position += particle.velocity * dt;
    }
}

/// Clamp particles to domain boundaries and reflect velocities.
pub fn enforce_particle_boundaries(particles: &mut Particles3D, grid: &Grid3D) {
    let min = Vec3::splat(grid.cell_size * 0.5);
    let max = Vec3::new(
        grid.world_width() - grid.cell_size * 0.5,
        grid.world_height() - grid.cell_size * 0.5,
        grid.world_depth() - grid.cell_size * 0.5,
    );

    for particle in &mut particles.list {
        // X bounds
        if particle.position.x < min.x {
            particle.position.x = min.x;
            particle.velocity.x = particle.velocity.x.abs() * 0.1; // Damped reflection
        }
        if particle.position.x > max.x {
            particle.position.x = max.x;
            particle.velocity.x = -particle.velocity.x.abs() * 0.1;
        }

        // Y bounds (floor and ceiling)
        if particle.position.y < min.y {
            particle.position.y = min.y;
            particle.velocity.y = particle.velocity.y.abs() * 0.1;
        }
        if particle.position.y > max.y {
            particle.position.y = max.y;
            particle.velocity.y = -particle.velocity.y.abs() * 0.1;
        }

        // Z bounds
        if particle.position.z < min.z {
            particle.position.z = min.z;
            particle.velocity.z = particle.velocity.z.abs() * 0.1;
        }
        if particle.position.z > max.z {
            particle.position.z = max.z;
            particle.velocity.z = -particle.velocity.z.abs() * 0.1;
        }
    }
}

/// Apply gravity to all particles.
pub fn apply_gravity_to_particles(particles: &mut Particles3D, gravity: Vec3, dt: f32) {
    for particle in &mut particles.list {
        particle.velocity += gravity * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle3D;

    #[test]
    fn test_advection() {
        let mut particles = Particles3D::new();
        particles.list.push(Particle3D::new(
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 2.0, 3.0),
        ));

        advect_particles(&mut particles, 0.5);

        let p = &particles.list[0];
        assert!((p.position.x - 1.5).abs() < 1e-6);
        assert!((p.position.y - 2.0).abs() < 1e-6);
        assert!((p.position.z - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_boundary_clamping() {
        let grid = Grid3D::new(4, 4, 4, 1.0);
        let mut particles = Particles3D::new();

        // Particle outside bounds
        particles.list.push(Particle3D::new(
            Vec3::new(-1.0, 5.0, 2.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ));

        enforce_particle_boundaries(&mut particles, &grid);

        let p = &particles.list[0];
        // Should be clamped to [0.5, 3.5] in each dimension
        assert!(p.position.x >= 0.5);
        assert!(p.position.y <= 3.5);
    }
}

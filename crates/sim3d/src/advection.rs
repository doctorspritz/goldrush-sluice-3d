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

/// Clamp particles to domain boundaries and handle solid collisions via SDF.
///
/// IMPORTANT: Outlet (x=max) and top (y=max) are OPEN boundaries - particles
/// can exit there and should be removed by the caller.
pub fn enforce_particle_boundaries(particles: &mut Particles3D, grid: &Grid3D) {
    let min = Vec3::splat(grid.cell_size * 0.5);
    let max = Vec3::new(
        grid.world_width() - grid.cell_size * 0.5,
        grid.world_height() - grid.cell_size * 0.5,
        grid.world_depth() - grid.cell_size * 0.5,
    );

    for particle in &mut particles.list {
        // Inlet (x=0): CLOSED - bounce back
        if particle.position.x < min.x {
            particle.position.x = min.x;
            particle.velocity.x = particle.velocity.x.abs() * 0.1;
        }
        // Outlet (x=max): OPEN - let particles exit (will be removed later)
        // Don't clamp or bounce - just let them go

        // Floor (y=0): CLOSED - bounce
        if particle.position.y < min.y {
            particle.position.y = min.y;
            particle.velocity.y = particle.velocity.y.abs() * 0.1;
        }
        // Top (y=max): OPEN - let particles exit

        // Side walls (z=0, z=max): CLOSED - bounce
        if particle.position.z < min.z {
            particle.position.z = min.z;
            particle.velocity.z = particle.velocity.z.abs() * 0.1;
        }
        if particle.position.z > max.z {
            particle.position.z = max.z;
            particle.velocity.z = -particle.velocity.z.abs() * 0.1;
        }

        // SDF collision with solid geometry (riffles, floor, etc.)
        let sdf = grid.sample_sdf(particle.position);
        if sdf < 0.0 {
            // Particle is inside solid - push out along gradient
            let normal = grid.sdf_gradient(particle.position);
            let penetration = -sdf + grid.cell_size * 0.1; // Small margin
            particle.position += normal * penetration;

            // Reflect velocity component into solid
            let vel_into_solid = particle.velocity.dot(normal);
            if vel_into_solid < 0.0 {
                // Remove velocity into solid, total inelastic collision
                particle.velocity -= normal * vel_into_solid * 1.0;
            }
        }
    }
}

/// Remove particles that have exited through open boundaries.
/// Call this after advection to clean up exited particles.
pub fn remove_exited_particles(particles: &mut Particles3D, grid: &Grid3D) -> usize {
    let max_x = grid.world_width();
    let max_y = grid.world_height();

    let before = particles.list.len();
    particles.list.retain(|p| {
        // Keep if inside domain and valid
        p.position.x < max_x
            && p.position.y < max_y
            && p.position.x >= 0.0
            && p.position.y >= 0.0
            && p.velocity.is_finite()
            && p.position.is_finite()
    });
    before - particles.list.len()
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

        // Particle outside bounds on inlet side (x<0) - should be clamped
        particles.list.push(Particle3D::new(
            Vec3::new(-1.0, 2.0, 2.0),
            Vec3::new(-1.0, 0.0, 0.0),
        ));

        enforce_particle_boundaries(&mut particles, &grid);

        let p = &particles.list[0];
        // X should be clamped to min (0.5)
        assert!(p.position.x >= 0.5, "X should be clamped at inlet");
        // Y is within bounds, shouldn't change
        assert!((p.position.y - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_outlet_open() {
        let grid = Grid3D::new(4, 4, 4, 1.0);
        let mut particles = Particles3D::new();

        // Particle at outlet (x > max) - should NOT be clamped (open boundary)
        particles.list.push(Particle3D::new(
            Vec3::new(5.0, 2.0, 2.0),
            Vec3::new(1.0, 0.0, 0.0),
        ));

        enforce_particle_boundaries(&mut particles, &grid);

        let p = &particles.list[0];
        // X should NOT be clamped - outlet is open
        assert!(
            p.position.x > 4.0,
            "Outlet should be open, particle should pass through"
        );
        // Velocity should be preserved
        assert!(
            p.velocity.x > 0.0,
            "Velocity should be preserved at open outlet"
        );
    }

    #[test]
    fn test_remove_exited_particles() {
        let grid = Grid3D::new(4, 4, 4, 1.0);
        let mut particles = Particles3D::new();

        // Particle inside domain
        particles
            .list
            .push(Particle3D::new(Vec3::new(2.0, 2.0, 2.0), Vec3::ZERO));
        // Particle exited through outlet
        particles
            .list
            .push(Particle3D::new(Vec3::new(5.0, 2.0, 2.0), Vec3::ZERO));
        // Particle exited through top
        particles
            .list
            .push(Particle3D::new(Vec3::new(2.0, 5.0, 2.0), Vec3::ZERO));

        let removed = remove_exited_particles(&mut particles, &grid);

        assert_eq!(removed, 2, "Should remove 2 exited particles");
        assert_eq!(particles.list.len(), 1, "Should have 1 particle remaining");
        assert!(
            (particles.list[0].position.x - 2.0).abs() < 0.01,
            "Remaining particle should be the inside one"
        );
    }
}

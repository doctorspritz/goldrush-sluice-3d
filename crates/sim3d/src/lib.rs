//! 3D FLIP/APIC Fluid Simulation
//!
//! A minimal 3D implementation of the FLIP (Fluid Implicit Particle) method
//! with APIC (Affine Particle-In-Cell) transfers for angular momentum conservation.
//!
//! # Example
//!
//! ```
//! use sim3d::FlipSimulation3D;
//! use glam::Vec3;
//!
//! let mut sim = FlipSimulation3D::new(16, 16, 16, 0.1);
//!
//! // Spawn some particles
//! for i in 0..4 {
//!     for j in 0..8 {
//!         for k in 0..4 {
//!             sim.spawn_particle(Vec3::new(
//!                 (i as f32 + 0.5) * 0.1,
//!                 (j as f32 + 0.5) * 0.1,
//!                 (k as f32 + 0.5) * 0.1,
//!             ));
//!         }
//!     }
//! }
//!
//! // Run simulation step
//! sim.update(1.0 / 60.0);
//! ```

pub mod advection;
pub mod clump;
pub mod constants;
pub mod grid;
pub mod heightfield;
pub mod kernels;
pub mod particle;
pub mod pressure;
pub mod terrain_generator;
pub mod test_geometry;
pub mod transfer;
pub mod world;

pub use clump::{
    Clump3D, ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, FluidParams, FluidVelocityField,
    IrregularStyle3D, SdfParams,
};
pub use glam::{Mat3, Vec3};
pub use grid::{CellType, Grid3D};
pub use heightfield::Heightfield;
pub use particle::{Particle3D, Particles3D};
pub use terrain_generator::{generate_klondike_terrain, get_creek_source, TerrainConfig};
pub use test_geometry::{TestBox, TestChute, TestFloor, TestRamp, TestSdfGenerator};
pub use world::{ExcavationResult, TerrainMaterial, World, WorldParams};

use transfer::TransferBuffers;

/// 3D FLIP fluid simulation.
pub struct FlipSimulation3D {
    /// The MAC grid for pressure and velocity
    pub grid: Grid3D,
    /// All particles in the simulation
    pub particles: Particles3D,

    /// Transfer buffers (pre-allocated to avoid per-frame allocation)
    transfer_buffers: TransferBuffers,

    /// Gravity vector (default: -Y)
    pub gravity: Vec3,
    /// FLIP/PIC blend ratio (0.97 = 97% FLIP, 3% PIC)
    pub flip_ratio: f32,
    /// Number of pressure solver iterations
    pub pressure_iterations: usize,

    /// Current simulation frame
    pub frame: u32,
}

impl FlipSimulation3D {
    /// Create a new simulation with the given grid dimensions.
    pub fn new(width: usize, height: usize, depth: usize, cell_size: f32) -> Self {
        let grid = Grid3D::new(width, height, depth, cell_size);
        let transfer_buffers = TransferBuffers::new(&grid);

        Self {
            grid,
            particles: Particles3D::new(),
            transfer_buffers,
            gravity: Vec3::new(0.0, -9.8, 0.0),
            flip_ratio: 0.97,
            pressure_iterations: 50,
            frame: 0,
        }
    }

    /// Spawn a particle at the given position with zero velocity.
    pub fn spawn_particle(&mut self, position: Vec3) {
        self.particles.spawn_at(position);
    }

    /// Spawn a particle with the given position and velocity.
    pub fn spawn_particle_with_velocity(&mut self, position: Vec3, velocity: Vec3) {
        self.particles.spawn(position, velocity);
    }

    /// Spawn a sediment particle with given density.
    pub fn spawn_sediment(&mut self, position: Vec3, velocity: Vec3, density: f32) {
        self.particles
            .spawn_with_density(position, velocity, density);
    }

    /// Run one simulation step.
    pub fn update(&mut self, dt: f32) {
        if self.particles.is_empty() {
            return;
        }

        // 1. Classify cells based on particle positions
        self.classify_cells();

        // 2. P2G: Transfer particle velocities to grid
        transfer::particles_to_grid(&mut self.grid, &self.particles, &mut self.transfer_buffers);

        // 3. Apply boundary conditions BEFORE storing old velocities
        // This ensures particles near walls don't lose velocity due to BC enforcement
        pressure::enforce_boundary_conditions(&mut self.grid);

        // 4. Store old velocities for FLIP delta (after BC so delta is correct)
        self.grid.store_old_velocities();

        // 5. Apply gravity to grid velocities
        self.apply_gravity(dt);

        // 5.5. Clamp grid velocities BEFORE pressure solve for CFL stability
        // This ensures the pressure solver operates on stable input and its
        // divergence-free output is not invalidated by post-solve clamping.
        self.clamp_grid_velocities();

        // 6. Pressure projection (make velocity divergence-free)
        pressure::compute_divergence(&mut self.grid);
        pressure::solve_pressure_jacobi(&mut self.grid, self.pressure_iterations);
        pressure::apply_pressure_gradient(&mut self.grid);
        pressure::enforce_boundary_conditions(&mut self.grid);

        // 7. G2P: Transfer grid velocities back to particles
        transfer::grid_to_particles(&self.grid, &mut self.particles, self.flip_ratio);

        // 8. Advect particles
        advection::advect_particles(&mut self.particles, dt);
        advection::enforce_particle_boundaries(&mut self.particles, &self.grid);

        // 9. Remove particles that exited through open boundaries
        advection::remove_exited_particles(&mut self.particles, &self.grid);

        self.frame += 1;
    }

    /// Classify cells as Fluid, Air, or Solid based on particle positions.
    fn classify_cells(&mut self) {
        // Reset all non-solid cells to Air
        self.grid.reset_cell_types();

        // Mark cells containing particles as Fluid
        for particle in &self.particles.list {
            let (i, j, k) = self.grid.world_to_cell(particle.position);

            if self.grid.cell_in_bounds(i, j, k) {
                let idx = self.grid.cell_index(i as usize, j as usize, k as usize);
                if self.grid.cell_type[idx] != CellType::Solid {
                    self.grid.cell_type[idx] = CellType::Fluid;
                }
            }
        }
    }

    /// Apply gravity to V velocities on the grid.
    fn apply_gravity(&mut self, dt: f32) {
        let gravity_y = self.gravity.y * dt;

        // Apply to all V (vertical) velocities
        for v in &mut self.grid.v {
            *v += gravity_y;
        }

        // Apply to U (horizontal) if gravity has X component
        if self.gravity.x.abs() > 1e-6 {
            let gravity_x = self.gravity.x * dt;
            for u in &mut self.grid.u {
                *u += gravity_x;
            }
        }

        // Apply to W (depth) if gravity has Z component
        if self.gravity.z.abs() > 1e-6 {
            let gravity_z = self.gravity.z * dt;
            for w in &mut self.grid.w {
                *w += gravity_z;
            }
        }
    }

    /// Clamp grid velocities to maintain CFL stability.
    ///
    /// Without clamping, grid velocities can explode to astronomical values
    /// (10^26 m/s observed) when particles interact with complex solid geometry.
    /// This causes the simulation to become numerically unstable.
    ///
    /// CFL condition: max_vel < dx/dt
    /// For typical parameters (dx=0.025m, dt=1/120s): max_vel < 3 m/s
    /// We clamp to 10 m/s to allow some headroom while staying stable.
    fn clamp_grid_velocities(&mut self) {
        const MAX_GRID_VEL: f32 = 10.0;

        for u in &mut self.grid.u {
            *u = u.clamp(-MAX_GRID_VEL, MAX_GRID_VEL);
        }
        for v in &mut self.grid.v {
            *v = v.clamp(-MAX_GRID_VEL, MAX_GRID_VEL);
        }
        for w in &mut self.grid.w {
            *w = w.clamp(-MAX_GRID_VEL, MAX_GRID_VEL);
        }
    }

    /// Get total particle count.
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    /// Get world bounds as (min, max) vectors.
    pub fn world_bounds(&self) -> (Vec3, Vec3) {
        (
            Vec3::ZERO,
            Vec3::new(
                self.grid.world_width(),
                self.grid.world_height(),
                self.grid.world_depth(),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_creation() {
        let sim = FlipSimulation3D::new(16, 32, 8, 0.1);
        assert_eq!(sim.grid.width, 16);
        assert_eq!(sim.grid.height, 32);
        assert_eq!(sim.grid.depth, 8);
        assert_eq!(sim.particle_count(), 0);
    }

    #[test]
    fn test_spawn_particles() {
        let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);
        sim.spawn_particle(Vec3::new(1.0, 1.0, 1.0));
        sim.spawn_particle_with_velocity(Vec3::new(2.0, 2.0, 2.0), Vec3::new(0.1, 0.0, 0.0));
        assert_eq!(sim.particle_count(), 2);
    }

    #[test]
    fn test_update_with_particles() {
        let mut sim = FlipSimulation3D::new(8, 8, 8, 0.5);

        // Spawn a block of particles
        for i in 1..3 {
            for j in 4..6 {
                for k in 1..3 {
                    sim.spawn_particle(Vec3::new(
                        (i as f32 + 0.5) * 0.5,
                        (j as f32 + 0.5) * 0.5,
                        (k as f32 + 0.5) * 0.5,
                    ));
                }
            }
        }

        let initial_count = sim.particle_count();

        // Run a few steps
        for _ in 0..10 {
            sim.update(1.0 / 60.0);
        }

        // Particles should still exist (not escaped)
        assert_eq!(sim.particle_count(), initial_count);

        // Particles should have fallen due to gravity
        let avg_y: f32 = sim.particles.list.iter().map(|p| p.position.y).sum::<f32>()
            / sim.particle_count() as f32;

        // Initial Y was around 2.25-2.75, should have fallen
        assert!(
            avg_y < 2.5,
            "Particles should have fallen, avg_y = {}",
            avg_y
        );
    }

    #[test]
    fn test_cell_classification() {
        let mut sim = FlipSimulation3D::new(4, 4, 4, 1.0);

        // Spawn particle in cell (1,1,1)
        sim.spawn_particle(Vec3::new(1.5, 1.5, 1.5));

        sim.classify_cells();

        let idx = sim.grid.cell_index(1, 1, 1);
        assert_eq!(sim.grid.cell_type[idx], CellType::Fluid);

        // Empty cell should be Air
        let idx_empty = sim.grid.cell_index(3, 3, 3);
        assert_eq!(sim.grid.cell_type[idx_empty], CellType::Air);
    }

    #[test]
    fn test_hydrostatic_equilibrium() {
        // A column of still water should remain approximately still
        let mut sim = FlipSimulation3D::new(4, 8, 4, 0.5);
        sim.pressure_iterations = 100;

        // Fill a column with particles
        for i in 1..3 {
            for j in 1..5 {
                for k in 1..3 {
                    sim.spawn_particle(Vec3::new(
                        (i as f32 + 0.5) * 0.5,
                        (j as f32 + 0.5) * 0.5,
                        (k as f32 + 0.5) * 0.5,
                    ));
                }
            }
        }

        // Run for many frames
        for _ in 0..200 {
            sim.update(1.0 / 120.0);
        }

        // Check that particles haven't exploded
        let max_vel = sim
            .particles
            .list
            .iter()
            .map(|p| p.velocity.length())
            .fold(0.0f32, f32::max);

        assert!(max_vel < 10.0, "Velocities exploded: max_vel = {}", max_vel);
    }
}

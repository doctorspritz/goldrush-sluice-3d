//! Particle-grid transfer (P2G, G2P).
//!
//! Core APIC transfer operations using quadratic B-spline kernels.

use super::FlipSimulation;

// TODO: Move from mod.rs:
// - particles_to_grid
// - store_old_velocities
// - grid_to_particles

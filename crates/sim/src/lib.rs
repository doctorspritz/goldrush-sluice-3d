//! Goldrush Fluid Miner - Simulation Library
//!
//! PIC/FLIP fluid simulation with:
//! - Particle-based fluid (water, mud)
//! - MAC grid for pressure solving
//! - Vortex formation behind riffles
//!
//! This crate is framework-agnostic - it handles simulation only.
//! Use the `game` crate for rendering with Macroquad.

pub mod physics;
pub mod flip;
pub mod grid;
pub mod particle;
pub mod sluice;
pub mod pbf;

pub use flip::FlipSimulation;
pub use grid::{CellType, Grid};
pub use particle::{Particle, ParticleMaterial, Particles};
pub use sluice::{create_box, create_flat_sluice, create_sluice};
pub use pbf::{PbfSimulation, PbfParticle};

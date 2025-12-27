//! Goldrush Fluid Miner - Simulation Library
//!
//! WATER-ONLY PIC/FLIP fluid simulation with:
//! - Particle-based water fluid
//! - MAC grid for pressure solving
//! - Vortex formation behind riffles
//!
//! Sediment code archived in: archive/sediment_archive.rs
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
pub use particle::{Particle, Particles};
pub use sluice::{
    create_box, create_flat_sluice, create_sluice, create_sluice_with_mode,
    compute_surface_heightfield, get_riffle_cells,
    RiffleMode, SluiceConfig,
};
pub use pbf::{PbfSimulation, PbfParticle};

//! Goldrush Fluid Miner - Simulation Library
//!
//! PIC/FLIP fluid simulation with:
//! - Particle-based fluid (water, mud)
//! - MAC grid for pressure solving
//! - Vortex formation behind riffles
//!
//! This crate is framework-agnostic - it handles simulation only.
//! Use the `game` crate for rendering with Macroquad.

// Old CA-based modules (kept for reference, will be removed)
pub mod cell;
pub mod chunk;
pub mod fluid;
pub mod material;
pub mod update;
pub mod water;
pub mod world;

// New PIC/FLIP modules
pub mod flip;
pub mod grid;
pub mod particle;
pub mod sediment;
pub mod sluice;
pub mod pbf; // PBF for granular phase

// Re-export old types (for backwards compatibility during transition)
pub use cell::Cell;
pub use chunk::{Chunk, CHUNK_AREA, CHUNK_SIZE};
pub use material::Material;
pub use world::World;

// Re-export new PIC/FLIP types
pub use flip::FlipSimulation;
pub use grid::{CellType, Grid};
pub use particle::{Particle, Particles};
pub use sediment::{Sediment, SedimentParticle, SedimentState, SedimentType};
pub use sluice::{create_box, create_flat_sluice, create_sluice};
pub use pbf::{PbfSimulation, PbfParticle};

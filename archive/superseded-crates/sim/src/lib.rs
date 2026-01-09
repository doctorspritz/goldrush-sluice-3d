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
pub mod terrain;
pub mod excavation;
pub mod collapse;
pub mod sluice;
pub mod pbf;
pub mod dem;
pub mod clump;

pub use flip::FlipSimulation;
pub use grid::{CellType, Grid};
pub use particle::{Particle, ParticleMaterial, Particles};
pub use terrain::{HeightfieldTerrain, TerrainLayer, TerrainMaterial};
pub use excavation::{ExcavatedParticle, Tool, ToolKind};
pub use collapse::CollapseEvent;
pub use sluice::{
    create_box, create_flat_sluice, create_sluice, create_sluice_with_mode,
    compute_surface_heightfield, get_riffle_cells,
    RiffleMode, SluiceConfig,
};
pub use pbf::{PbfSimulation, PbfParticle};
pub use dem::{DemSimulation, DemParams};
pub use clump::{Clump, ClumpShape, ClumpTemplate};

//! Game library crate - exposes GPU infrastructure for examples and tests

pub mod app;
pub mod editor;
pub mod equipment_geometry;
pub mod gpu;
pub mod multigrid;
pub mod sluice_geometry;
pub mod scenario;
pub mod test_harness;
pub mod tools;
pub mod washplant;
pub mod water_heightfield;
pub mod water_emitter;
pub mod example_utils;

#[cfg(feature = "panning")]
pub mod panning;

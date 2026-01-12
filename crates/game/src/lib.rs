//! Game library crate - exposes GPU infrastructure for examples and tests

pub mod gpu;
pub mod sluice_geometry;
pub mod equipment_geometry;
pub mod test_harness;
pub mod tools;
pub mod water_heightfield;

#[cfg(feature = "panning")]
pub mod panning;

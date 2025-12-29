//! Signed Distance Field computation and sampling.
//!
//! SDF represents distance to solid boundaries (negative inside solid).

use super::Grid;
use glam::Vec2;

// TODO: Move from mod.rs:
// - compute_sdf
// - sample_sdf
// - sdf_gradient
// - bed height methods

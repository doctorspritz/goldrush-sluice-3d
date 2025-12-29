//! Pressure projection and multigrid solver.
//!
//! Enforces incompressibility via pressure solve.

use super::Grid;

// TODO: Move from mod.rs:
// - compute_divergence
// - solve_pressure
// - solve_pressure_multigrid
// - apply_pressure_gradient
// - All mg_* methods

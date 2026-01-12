//! Unified app framework for GPU-accelerated examples.
//!
//! This module provides shared infrastructure to reduce boilerplate across examples.

pub mod context;
pub mod uniforms;

pub use context::GpuContext;
pub use uniforms::ViewUniforms;

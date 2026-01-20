//! Reusable GPU compute pipeline builder for FLIP simulation.
//!
//! This module provides a builder pattern to reduce boilerplate when creating
//! compute pipelines. Instead of manually handling shader modules, bind group layouts,
//! and pipeline layouts, you can use the fluent API:
//!
//! ```ignore
//! let (pipeline, layout) = PipelineBuilder::new(device)
//!     .shader_source(include_str!("shaders/gravity_3d.wgsl"))
//!     .label("gravity_3d")
//!     .entry_point("apply_gravity")
//!     .uniform_buffer(std::mem::size_of::<GravityParams3D>())
//!     .storage_buffer(true)   // read-only
//!     .storage_buffer(false)  // read-write
//!     .storage_buffer(true)   // read-only
//!     .build();
//! ```

use std::num::NonZeroU64;

/// Builder for GPU compute pipelines.
///
/// Handles the boilerplate of creating shader modules, bind group layouts, and pipeline layouts.
/// Collects buffer binding specifications and generates the necessary wgpu structures.
pub struct PipelineBuilder<'a> {
    device: &'a wgpu::Device,
    shader_source: Option<&'a str>,
    label: Option<&'a str>,
    entry_point: &'a str,
    bindings: Vec<BufferBinding>,
}

/// Specification for a single buffer binding.
#[derive(Clone, Debug)]
enum BufferBinding {
    Uniform(Option<NonZeroU64>),
    Storage { read_only: bool },
}

impl<'a> PipelineBuilder<'a> {
    /// Create a new pipeline builder.
    pub fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            shader_source: None,
            label: None,
            entry_point: "main",
            bindings: Vec::new(),
        }
    }

    /// Set the WGSL shader source code.
    pub fn shader_source(mut self, source: &'a str) -> Self {
        self.shader_source = Some(source);
        self
    }

    /// Set the label for the pipeline (used in debugging and profiling).
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// Set the entry point function name (default: "main").
    pub fn entry_point(mut self, entry_point: &'a str) -> Self {
        self.entry_point = entry_point;
        self
    }

    /// Add a uniform buffer binding.
    ///
    /// Uniform buffers are typically used for small parameter structs.
    /// If `min_size` is provided, it's used as the minimum binding size hint.
    pub fn uniform_buffer(mut self, min_size: Option<NonZeroU64>) -> Self {
        self.bindings.push(BufferBinding::Uniform(min_size));
        self
    }

    /// Add a uniform buffer binding with explicit size.
    ///
    /// Convenience method that wraps the size in `NonZeroU64`.
    pub fn uniform_buffer_size(self, size: u64) -> Self {
        self.uniform_buffer(NonZeroU64::new(size))
    }

    /// Add a storage buffer binding.
    ///
    /// - `read_only = true`: Buffer is read-only in the shader
    /// - `read_only = false`: Buffer is read-write in the shader
    pub fn storage_buffer(mut self, read_only: bool) -> Self {
        self.bindings.push(BufferBinding::Storage { read_only });
        self
    }

    /// Add multiple storage buffers at once.
    ///
    /// Each boolean in the slice indicates whether that buffer is read-only.
    pub fn storage_buffers(mut self, read_only_flags: &[bool]) -> Self {
        for &read_only in read_only_flags {
            self.bindings.push(BufferBinding::Storage { read_only });
        }
        self
    }

    /// Build the compute pipeline and return both the pipeline and its bind group layout.
    ///
    /// Returns `(ComputePipeline, BindGroupLayout)`.
    ///
    /// # Panics
    /// Panics if shader_source was not set.
    pub fn build(self) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader_source = self
            .shader_source
            .expect("shader_source must be set before building");

        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: self.label.or(Some("Pipeline Shader")),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Build bind group layout entries
        let entries: Vec<wgpu::BindGroupLayoutEntry> = self
            .bindings
            .iter()
            .enumerate()
            .map(|(binding, spec)| match spec {
                BufferBinding::Uniform(min_size) => wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: *min_size,
                    },
                    count: None,
                },
                BufferBinding::Storage { read_only } => wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: *read_only,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            })
            .collect();

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: self
                        .label
                        .map(|l| Box::leak(format!("{} Bind Group Layout", l).into_boxed_str())
                            as &str),
                    entries: &entries,
                });

        // Create pipeline layout
        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: self
                        .label
                        .map(|l| Box::leak(format!("{} Pipeline Layout", l).into_boxed_str())
                            as &str),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: self
                .label
                .map(|l| Box::leak(format!("{} Pipeline", l).into_boxed_str()) as &str),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(self.entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        (pipeline, bind_group_layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // API shape is validated by compilation - no runtime test needed without GPU
}

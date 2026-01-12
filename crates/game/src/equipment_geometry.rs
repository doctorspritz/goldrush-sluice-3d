//! Equipment geometry generation for wash plant components.
//!
//! This module provides parametric shape generators for wash plant equipment:
//! - Grate: Parallel bars with gaps (screening/filtering)
//! - Box: Hollow rectangular container with open top
//! - Gutter: U-shaped channel (half-pipe)
//! - Hopper: Inverted pyramid for material collection
//! - Chute: Angled slide/ramp
//! - Frame: Rectangular outline (edges only, no faces)
//! - Baffle: Vertical plate/wall (flow director)
//!
//! Each shape follows the same pattern:
//! - Config struct for parameters
//! - Builder struct for mesh generation
//! - is_solid(i,j,k) for solid cell detection
//! - solid_cells() iterator
//! - build_mesh() for rendering
//! - GPU upload/update methods

use wgpu::util::DeviceExt;

// Re-export SluiceVertex for all equipment shapes
pub use crate::sluice_geometry::SluiceVertex;

// ============================================================================
// GRATE - Parallel bars with gaps
// ============================================================================

#[derive(Clone, Debug)]
pub struct GrateConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,

    /// Number of cells between bars
    pub bar_spacing: usize,
    /// Thickness of each bar in cells
    pub bar_thickness: usize,
    /// Orientation: 0 = bars parallel to X axis, 1 = bars parallel to Z axis
    pub orientation: usize,

    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}

impl Default for GrateConfig {
    fn default() -> Self {
        Self {
            grid_width: 20,
            grid_height: 4,
            grid_depth: 20,
            cell_size: 0.25,
            bar_spacing: 3,
            bar_thickness: 1,
            orientation: 0,
            color_top: [0.55, 0.50, 0.45, 1.0],
            color_side: [0.45, 0.40, 0.35, 1.0],
            color_bottom: [0.35, 0.30, 0.25, 1.0],
        }
    }
}

impl GrateConfig {
    pub fn is_solid(&self, i: usize, _j: usize, k: usize) -> bool {
        if self.orientation == 0 {
            // Bars parallel to X axis (vary along Z)
            k % (self.bar_spacing + self.bar_thickness) < self.bar_thickness
        } else {
            // Bars parallel to Z axis (vary along X)
            i % (self.bar_spacing + self.bar_thickness) < self.bar_thickness
        }
    }
}

pub struct GrateGeometryBuilder {
    config: GrateConfig,
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl GrateGeometryBuilder {
    pub fn new(config: GrateConfig) -> Self {
        Self {
            config,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    pub fn config(&self) -> &GrateConfig {
        &self.config
    }

    pub fn solid_cells(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;

        (0..depth).flat_map(move |k| {
            (0..height).flat_map(move |j| {
                (0..width).filter_map(move |i| {
                    if self.config.is_solid(i, j, k) {
                        Some((i, j, k))
                    } else {
                        None
                    }
                })
            })
        })
    }

    pub fn build_mesh<F>(&mut self, is_solid: F)
    where
        F: Fn(usize, usize, usize) -> bool,
    {
        self.vertices.clear();
        self.indices.clear();

        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;
        let cs = self.config.cell_size;

        let color_top = self.config.color_top;
        let color_side = self.config.color_side;
        let color_bottom = self.config.color_bottom;

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if !is_solid(i, j, k) {
                        continue;
                    }

                    let x0 = i as f32 * cs;
                    let x1 = (i + 1) as f32 * cs;
                    let y0 = j as f32 * cs;
                    let y1 = (j + 1) as f32 * cs;
                    let z0 = k as f32 * cs;
                    let z1 = (k + 1) as f32 * cs;

                    // -X face
                    if i == 0 || !is_solid(i - 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                            SluiceVertex::new([x0, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    // +X face
                    if i == width - 1 || !is_solid(i + 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    // -Y face
                    if j == 0 || !is_solid(i, j - 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z1], color_bottom),
                            SluiceVertex::new([x0, y0, z1], color_bottom),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    // +Y face
                    if j == height - 1 || !is_solid(i, j + 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z1], color_top),
                            SluiceVertex::new([x0, y1, z1], color_top),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    // -Z face
                    if k == 0 || !is_solid(i, j, k - 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    // +Z face
                    if k == depth - 1 || !is_solid(i, j, k + 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }
                }
            }
        }
    }

    pub fn vertices(&self) -> &[SluiceVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Grate Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );

        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Grate Index Buffer"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
    }

    pub fn update(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        if self.vertex_buffer.is_none() {
            self.upload(device);
            return;
        }

        let vb = self.vertex_buffer.as_ref().unwrap();
        let ib = self.index_buffer.as_ref().unwrap();

        let vertex_bytes = bytemuck::cast_slice(&self.vertices);
        let index_bytes = bytemuck::cast_slice(&self.indices);

        if vb.size() >= vertex_bytes.len() as u64 && ib.size() >= index_bytes.len() as u64 {
            queue.write_buffer(vb, 0, vertex_bytes);
            queue.write_buffer(ib, 0, index_bytes);
        } else {
            self.upload(device);
        }
    }

    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

// ============================================================================
// BOX - Hollow rectangular container with open top
// ============================================================================

#[derive(Clone, Debug)]
pub struct BoxConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,

    /// Thickness of walls in cells
    pub wall_thickness: usize,

    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}

impl Default for BoxConfig {
    fn default() -> Self {
        Self {
            grid_width: 20,
            grid_height: 10,
            grid_depth: 20,
            cell_size: 0.25,
            wall_thickness: 1,
            color_top: [0.55, 0.50, 0.45, 1.0],
            color_side: [0.45, 0.40, 0.35, 1.0],
            color_bottom: [0.35, 0.30, 0.25, 1.0],
        }
    }
}

impl BoxConfig {
    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        let wt = self.wall_thickness;

        // Floor
        if j < wt {
            return true;
        }

        // Walls (not top)
        if i < wt || i >= self.grid_width - wt || k < wt || k >= self.grid_depth - wt {
            return true;
        }

        false
    }
}

pub struct BoxGeometryBuilder {
    config: BoxConfig,
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl BoxGeometryBuilder {
    pub fn new(config: BoxConfig) -> Self {
        Self {
            config,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    pub fn config(&self) -> &BoxConfig {
        &self.config
    }

    pub fn solid_cells(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;

        (0..depth).flat_map(move |k| {
            (0..height).flat_map(move |j| {
                (0..width).filter_map(move |i| {
                    if self.config.is_solid(i, j, k) {
                        Some((i, j, k))
                    } else {
                        None
                    }
                })
            })
        })
    }

    pub fn build_mesh<F>(&mut self, is_solid: F)
    where
        F: Fn(usize, usize, usize) -> bool,
    {
        self.vertices.clear();
        self.indices.clear();

        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;
        let cs = self.config.cell_size;

        let color_top = self.config.color_top;
        let color_side = self.config.color_side;
        let color_bottom = self.config.color_bottom;

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if !is_solid(i, j, k) {
                        continue;
                    }

                    let x0 = i as f32 * cs;
                    let x1 = (i + 1) as f32 * cs;
                    let y0 = j as f32 * cs;
                    let y1 = (j + 1) as f32 * cs;
                    let z0 = k as f32 * cs;
                    let z1 = (k + 1) as f32 * cs;

                    if i == 0 || !is_solid(i - 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                            SluiceVertex::new([x0, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if i == width - 1 || !is_solid(i + 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == 0 || !is_solid(i, j - 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z1], color_bottom),
                            SluiceVertex::new([x0, y0, z1], color_bottom),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == height - 1 || !is_solid(i, j + 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z1], color_top),
                            SluiceVertex::new([x0, y1, z1], color_top),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if k == 0 || !is_solid(i, j, k - 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if k == depth - 1 || !is_solid(i, j, k + 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }
                }
            }
        }
    }

    pub fn vertices(&self) -> &[SluiceVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Box Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );

        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Box Index Buffer"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
    }

    pub fn update(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        if self.vertex_buffer.is_none() {
            self.upload(device);
            return;
        }

        let vb = self.vertex_buffer.as_ref().unwrap();
        let ib = self.index_buffer.as_ref().unwrap();

        let vertex_bytes = bytemuck::cast_slice(&self.vertices);
        let index_bytes = bytemuck::cast_slice(&self.indices);

        if vb.size() >= vertex_bytes.len() as u64 && ib.size() >= index_bytes.len() as u64 {
            queue.write_buffer(vb, 0, vertex_bytes);
            queue.write_buffer(ib, 0, index_bytes);
        } else {
            self.upload(device);
        }
    }

    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

// ============================================================================
// GUTTER - U-shaped channel
// ============================================================================

#[derive(Clone, Debug)]
pub struct GutterConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,

    /// Width of gutter channel in cells
    pub channel_width: usize,
    /// Depth of gutter channel in cells
    pub channel_depth: usize,
    /// Floor thickness in cells
    pub floor_thickness: usize,

    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}

impl Default for GutterConfig {
    fn default() -> Self {
        Self {
            grid_width: 20,
            grid_height: 8,
            grid_depth: 20,
            cell_size: 0.25,
            channel_width: 10,
            channel_depth: 4,
            floor_thickness: 1,
            color_top: [0.55, 0.50, 0.45, 1.0],
            color_side: [0.45, 0.40, 0.35, 1.0],
            color_bottom: [0.35, 0.30, 0.25, 1.0],
        }
    }
}

impl GutterConfig {
    pub fn is_solid(&self, _i: usize, _j: usize, k: usize) -> bool {
        let z_center = self.grid_depth / 2;
        let half_width = self.channel_width / 2;

        // Floor
        if _j < self.floor_thickness {
            return true;
        }

        // Left and right walls
        if k < z_center.saturating_sub(half_width) || k >= z_center + half_width {
            if _j < self.channel_depth + self.floor_thickness {
                return true;
            }
        }

        false
    }
}

pub struct GutterGeometryBuilder {
    config: GutterConfig,
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl GutterGeometryBuilder {
    pub fn new(config: GutterConfig) -> Self {
        Self {
            config,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    pub fn config(&self) -> &GutterConfig {
        &self.config
    }

    pub fn solid_cells(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;

        (0..depth).flat_map(move |k| {
            (0..height).flat_map(move |j| {
                (0..width).filter_map(move |i| {
                    if self.config.is_solid(i, j, k) {
                        Some((i, j, k))
                    } else {
                        None
                    }
                })
            })
        })
    }

    pub fn build_mesh<F>(&mut self, is_solid: F)
    where
        F: Fn(usize, usize, usize) -> bool,
    {
        self.vertices.clear();
        self.indices.clear();

        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;
        let cs = self.config.cell_size;

        let color_top = self.config.color_top;
        let color_side = self.config.color_side;
        let color_bottom = self.config.color_bottom;

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if !is_solid(i, j, k) {
                        continue;
                    }

                    let x0 = i as f32 * cs;
                    let x1 = (i + 1) as f32 * cs;
                    let y0 = j as f32 * cs;
                    let y1 = (j + 1) as f32 * cs;
                    let z0 = k as f32 * cs;
                    let z1 = (k + 1) as f32 * cs;

                    if i == 0 || !is_solid(i - 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                            SluiceVertex::new([x0, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if i == width - 1 || !is_solid(i + 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == 0 || !is_solid(i, j - 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z1], color_bottom),
                            SluiceVertex::new([x0, y0, z1], color_bottom),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == height - 1 || !is_solid(i, j + 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z1], color_top),
                            SluiceVertex::new([x0, y1, z1], color_top),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if k == 0 || !is_solid(i, j, k - 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if k == depth - 1 || !is_solid(i, j, k + 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }
                }
            }
        }
    }

    pub fn vertices(&self) -> &[SluiceVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gutter Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );

        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gutter Index Buffer"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
    }

    pub fn update(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        if self.vertex_buffer.is_none() {
            self.upload(device);
            return;
        }

        let vb = self.vertex_buffer.as_ref().unwrap();
        let ib = self.index_buffer.as_ref().unwrap();

        let vertex_bytes = bytemuck::cast_slice(&self.vertices);
        let index_bytes = bytemuck::cast_slice(&self.indices);

        if vb.size() >= vertex_bytes.len() as u64 && ib.size() >= index_bytes.len() as u64 {
            queue.write_buffer(vb, 0, vertex_bytes);
            queue.write_buffer(ib, 0, index_bytes);
        } else {
            self.upload(device);
        }
    }

    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

// ============================================================================
// HOPPER - Inverted pyramid for material collection
// ============================================================================

#[derive(Clone, Debug)]
pub struct HopperConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,

    /// Top opening width
    pub top_width: usize,
    /// Top opening depth
    pub top_depth: usize,
    /// Bottom opening width
    pub bottom_width: usize,
    /// Bottom opening depth
    pub bottom_depth: usize,
    /// Wall thickness in cells
    pub wall_thickness: usize,

    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}

impl Default for HopperConfig {
    fn default() -> Self {
        Self {
            grid_width: 20,
            grid_height: 12,
            grid_depth: 20,
            cell_size: 0.25,
            top_width: 16,
            top_depth: 16,
            bottom_width: 4,
            bottom_depth: 4,
            wall_thickness: 1,
            color_top: [0.55, 0.50, 0.45, 1.0],
            color_side: [0.45, 0.40, 0.35, 1.0],
            color_bottom: [0.35, 0.30, 0.25, 1.0],
        }
    }
}

impl HopperConfig {
    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        if j >= self.grid_height {
            return false;
        }

        // Linear interpolation for tapered walls
        let t = j as f32 / (self.grid_height - 1).max(1) as f32;

        let width_at_j =
            (self.bottom_width as f32 * (1.0 - t) + self.top_width as f32 * t) as usize;
        let depth_at_j =
            (self.bottom_depth as f32 * (1.0 - t) + self.top_depth as f32 * t) as usize;

        let x_min = (self.grid_width - width_at_j) / 2;
        let x_max = x_min + width_at_j;
        let z_min = (self.grid_depth - depth_at_j) / 2;
        let z_max = z_min + depth_at_j;

        // Wall check
        if i < x_min || i >= x_max || k < z_min || k >= z_max {
            return false;
        }

        // Check if in wall thickness region (outer shell)
        let is_outer_x = i < x_min + self.wall_thickness || i >= x_max - self.wall_thickness;
        let is_outer_z = k < z_min + self.wall_thickness || k >= z_max - self.wall_thickness;

        is_outer_x || is_outer_z
    }
}

pub struct HopperGeometryBuilder {
    config: HopperConfig,
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl HopperGeometryBuilder {
    pub fn new(config: HopperConfig) -> Self {
        Self {
            config,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    pub fn config(&self) -> &HopperConfig {
        &self.config
    }

    pub fn solid_cells(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;

        (0..depth).flat_map(move |k| {
            (0..height).flat_map(move |j| {
                (0..width).filter_map(move |i| {
                    if self.config.is_solid(i, j, k) {
                        Some((i, j, k))
                    } else {
                        None
                    }
                })
            })
        })
    }

    pub fn build_mesh<F>(&mut self, is_solid: F)
    where
        F: Fn(usize, usize, usize) -> bool,
    {
        self.vertices.clear();
        self.indices.clear();

        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;
        let cs = self.config.cell_size;

        let color_top = self.config.color_top;
        let color_side = self.config.color_side;
        let color_bottom = self.config.color_bottom;

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if !is_solid(i, j, k) {
                        continue;
                    }

                    let x0 = i as f32 * cs;
                    let x1 = (i + 1) as f32 * cs;
                    let y0 = j as f32 * cs;
                    let y1 = (j + 1) as f32 * cs;
                    let z0 = k as f32 * cs;
                    let z1 = (k + 1) as f32 * cs;

                    if i == 0 || !is_solid(i - 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                            SluiceVertex::new([x0, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if i == width - 1 || !is_solid(i + 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == 0 || !is_solid(i, j - 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z1], color_bottom),
                            SluiceVertex::new([x0, y0, z1], color_bottom),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == height - 1 || !is_solid(i, j + 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z1], color_top),
                            SluiceVertex::new([x0, y1, z1], color_top),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if k == 0 || !is_solid(i, j, k - 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if k == depth - 1 || !is_solid(i, j, k + 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }
                }
            }
        }
    }

    pub fn vertices(&self) -> &[SluiceVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Hopper Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );

        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Hopper Index Buffer"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
    }

    pub fn update(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        if self.vertex_buffer.is_none() {
            self.upload(device);
            return;
        }

        let vb = self.vertex_buffer.as_ref().unwrap();
        let ib = self.index_buffer.as_ref().unwrap();

        let vertex_bytes = bytemuck::cast_slice(&self.vertices);
        let index_bytes = bytemuck::cast_slice(&self.indices);

        if vb.size() >= vertex_bytes.len() as u64 && ib.size() >= index_bytes.len() as u64 {
            queue.write_buffer(vb, 0, vertex_bytes);
            queue.write_buffer(ib, 0, index_bytes);
        } else {
            self.upload(device);
        }
    }

    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

// ============================================================================
// CHUTE - Angled slide/ramp
// ============================================================================

#[derive(Clone, Debug)]
pub struct ChuteConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,

    /// Floor height at start (upstream)
    pub floor_height_start: usize,
    /// Floor height at end (downstream)
    pub floor_height_end: usize,
    /// Height of side walls above floor
    pub side_wall_height: usize,
    /// Wall thickness in cells
    pub wall_thickness: usize,

    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}

impl Default for ChuteConfig {
    fn default() -> Self {
        Self {
            grid_width: 30,
            grid_height: 12,
            grid_depth: 12,
            cell_size: 0.25,
            floor_height_start: 8,
            floor_height_end: 2,
            side_wall_height: 3,
            wall_thickness: 1,
            color_top: [0.55, 0.50, 0.45, 1.0],
            color_side: [0.45, 0.40, 0.35, 1.0],
            color_bottom: [0.35, 0.30, 0.25, 1.0],
        }
    }
}

impl ChuteConfig {
    pub fn floor_height_at(&self, x: usize) -> usize {
        let t = x as f32 / (self.grid_width - 1).max(1) as f32;
        let height = self.floor_height_start as f32 * (1.0 - t) + self.floor_height_end as f32 * t;
        height as usize
    }

    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        let floor_j = self.floor_height_at(i);
        let wall_top = floor_j + self.side_wall_height;

        // Floor
        if j <= floor_j {
            return true;
        }

        // Side walls
        let wt = self.wall_thickness;
        if (k < wt || k >= self.grid_depth - wt) && j <= wall_top {
            return true;
        }

        false
    }
}

pub struct ChuteGeometryBuilder {
    config: ChuteConfig,
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl ChuteGeometryBuilder {
    pub fn new(config: ChuteConfig) -> Self {
        Self {
            config,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    pub fn config(&self) -> &ChuteConfig {
        &self.config
    }

    pub fn solid_cells(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;

        (0..depth).flat_map(move |k| {
            (0..height).flat_map(move |j| {
                (0..width).filter_map(move |i| {
                    if self.config.is_solid(i, j, k) {
                        Some((i, j, k))
                    } else {
                        None
                    }
                })
            })
        })
    }

    pub fn build_mesh<F>(&mut self, is_solid: F)
    where
        F: Fn(usize, usize, usize) -> bool,
    {
        self.vertices.clear();
        self.indices.clear();

        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;
        let cs = self.config.cell_size;

        let color_top = self.config.color_top;
        let color_side = self.config.color_side;
        let color_bottom = self.config.color_bottom;

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if !is_solid(i, j, k) {
                        continue;
                    }

                    let x0 = i as f32 * cs;
                    let x1 = (i + 1) as f32 * cs;
                    let y0 = j as f32 * cs;
                    let y1 = (j + 1) as f32 * cs;
                    let z0 = k as f32 * cs;
                    let z1 = (k + 1) as f32 * cs;

                    if i == 0 || !is_solid(i - 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                            SluiceVertex::new([x0, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if i == width - 1 || !is_solid(i + 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == 0 || !is_solid(i, j - 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z1], color_bottom),
                            SluiceVertex::new([x0, y0, z1], color_bottom),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == height - 1 || !is_solid(i, j + 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z1], color_top),
                            SluiceVertex::new([x0, y1, z1], color_top),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if k == 0 || !is_solid(i, j, k - 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if k == depth - 1 || !is_solid(i, j, k + 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }
                }
            }
        }
    }

    pub fn vertices(&self) -> &[SluiceVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Chute Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );

        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Chute Index Buffer"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
    }

    pub fn update(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        if self.vertex_buffer.is_none() {
            self.upload(device);
            return;
        }

        let vb = self.vertex_buffer.as_ref().unwrap();
        let ib = self.index_buffer.as_ref().unwrap();

        let vertex_bytes = bytemuck::cast_slice(&self.vertices);
        let index_bytes = bytemuck::cast_slice(&self.indices);

        if vb.size() >= vertex_bytes.len() as u64 && ib.size() >= index_bytes.len() as u64 {
            queue.write_buffer(vb, 0, vertex_bytes);
            queue.write_buffer(ib, 0, index_bytes);
        } else {
            self.upload(device);
        }
    }

    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

// ============================================================================
// FRAME - Rectangular outline (edges only)
// ============================================================================

#[derive(Clone, Debug)]
pub struct FrameConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,

    /// Thickness of beams in cells
    pub beam_thickness: usize,

    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}

impl Default for FrameConfig {
    fn default() -> Self {
        Self {
            grid_width: 20,
            grid_height: 15,
            grid_depth: 20,
            cell_size: 0.25,
            beam_thickness: 1,
            color_top: [0.55, 0.50, 0.45, 1.0],
            color_side: [0.45, 0.40, 0.35, 1.0],
            color_bottom: [0.35, 0.30, 0.25, 1.0],
        }
    }
}

impl FrameConfig {
    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        let bt = self.beam_thickness;

        // Corner check (for vertical beams)
        let is_corner_x = i < bt || i >= self.grid_width - bt;
        let is_corner_z = k < bt || k >= self.grid_depth - bt;

        // Bottom and top edge check (for horizontal beams)
        let is_bottom = j < bt;
        let is_top = j >= self.grid_height - bt;

        // Edge beam check (for horizontal beams along X or Z axis)
        let is_edge_x = i < bt || i >= self.grid_width - bt;
        let is_edge_z = k < bt || k >= self.grid_depth - bt;

        // 4 vertical corner beams (full height at corners)
        if is_corner_x && is_corner_z {
            return true;
        }

        // 8 horizontal edge beams (4 bottom + 4 top)
        // Bottom: 4 beams connecting bottom corners (along edges only)
        // Top: 4 beams connecting top corners (along edges only)
        if (is_bottom || is_top) && (is_edge_x != is_edge_z) {
            return true;
        }

        false
    }
}

pub struct FrameGeometryBuilder {
    config: FrameConfig,
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl FrameGeometryBuilder {
    pub fn new(config: FrameConfig) -> Self {
        Self {
            config,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    pub fn config(&self) -> &FrameConfig {
        &self.config
    }

    pub fn solid_cells(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;

        (0..depth).flat_map(move |k| {
            (0..height).flat_map(move |j| {
                (0..width).filter_map(move |i| {
                    if self.config.is_solid(i, j, k) {
                        Some((i, j, k))
                    } else {
                        None
                    }
                })
            })
        })
    }

    pub fn build_mesh<F>(&mut self, is_solid: F)
    where
        F: Fn(usize, usize, usize) -> bool,
    {
        self.vertices.clear();
        self.indices.clear();

        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;
        let cs = self.config.cell_size;

        let color_top = self.config.color_top;
        let color_side = self.config.color_side;
        let color_bottom = self.config.color_bottom;

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if !is_solid(i, j, k) {
                        continue;
                    }

                    let x0 = i as f32 * cs;
                    let x1 = (i + 1) as f32 * cs;
                    let y0 = j as f32 * cs;
                    let y1 = (j + 1) as f32 * cs;
                    let z0 = k as f32 * cs;
                    let z1 = (k + 1) as f32 * cs;

                    if i == 0 || !is_solid(i - 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                            SluiceVertex::new([x0, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if i == width - 1 || !is_solid(i + 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == 0 || !is_solid(i, j - 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z1], color_bottom),
                            SluiceVertex::new([x0, y0, z1], color_bottom),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == height - 1 || !is_solid(i, j + 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z1], color_top),
                            SluiceVertex::new([x0, y1, z1], color_top),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if k == 0 || !is_solid(i, j, k - 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if k == depth - 1 || !is_solid(i, j, k + 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }
                }
            }
        }
    }

    pub fn vertices(&self) -> &[SluiceVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Frame Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );

        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Frame Index Buffer"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
    }

    pub fn update(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        if self.vertex_buffer.is_none() {
            self.upload(device);
            return;
        }

        let vb = self.vertex_buffer.as_ref().unwrap();
        let ib = self.index_buffer.as_ref().unwrap();

        let vertex_bytes = bytemuck::cast_slice(&self.vertices);
        let index_bytes = bytemuck::cast_slice(&self.indices);

        if vb.size() >= vertex_bytes.len() as u64 && ib.size() >= index_bytes.len() as u64 {
            queue.write_buffer(vb, 0, vertex_bytes);
            queue.write_buffer(ib, 0, index_bytes);
        } else {
            self.upload(device);
        }
    }

    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

// ============================================================================
// BAFFLE - Vertical plate/wall (flow director)
// ============================================================================

#[derive(Clone, Debug)]
pub struct BaffleConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,

    /// Thickness of baffle in cells
    pub thickness: usize,
    /// Orientation: 0 = XY plane (perpendicular to Z), 1 = ZY plane (perpendicular to X)
    pub orientation: usize,

    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}

impl Default for BaffleConfig {
    fn default() -> Self {
        Self {
            grid_width: 20,
            grid_height: 10,
            grid_depth: 20,
            cell_size: 0.25,
            thickness: 1,
            orientation: 0,
            color_top: [0.55, 0.50, 0.45, 1.0],
            color_side: [0.45, 0.40, 0.35, 1.0],
            color_bottom: [0.35, 0.30, 0.25, 1.0],
        }
    }
}

impl BaffleConfig {
    pub fn is_solid(&self, i: usize, _j: usize, k: usize) -> bool {
        if self.orientation == 0 {
            // XY plane (perpendicular to Z) - baffle in middle of Z dimension
            let z_center = self.grid_depth / 2;
            k >= z_center.saturating_sub(self.thickness / 2)
                && k < z_center + (self.thickness + 1) / 2
        } else {
            // ZY plane (perpendicular to X) - baffle in middle of X dimension
            let x_center = self.grid_width / 2;
            i >= x_center.saturating_sub(self.thickness / 2)
                && i < x_center + (self.thickness + 1) / 2
        }
    }
}

pub struct BaffleGeometryBuilder {
    config: BaffleConfig,
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl BaffleGeometryBuilder {
    pub fn new(config: BaffleConfig) -> Self {
        Self {
            config,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    pub fn config(&self) -> &BaffleConfig {
        &self.config
    }

    pub fn solid_cells(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;

        (0..depth).flat_map(move |k| {
            (0..height).flat_map(move |j| {
                (0..width).filter_map(move |i| {
                    if self.config.is_solid(i, j, k) {
                        Some((i, j, k))
                    } else {
                        None
                    }
                })
            })
        })
    }

    pub fn build_mesh<F>(&mut self, is_solid: F)
    where
        F: Fn(usize, usize, usize) -> bool,
    {
        self.vertices.clear();
        self.indices.clear();

        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;
        let cs = self.config.cell_size;

        let color_top = self.config.color_top;
        let color_side = self.config.color_side;
        let color_bottom = self.config.color_bottom;

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if !is_solid(i, j, k) {
                        continue;
                    }

                    let x0 = i as f32 * cs;
                    let x1 = (i + 1) as f32 * cs;
                    let y0 = j as f32 * cs;
                    let y1 = (j + 1) as f32 * cs;
                    let z0 = k as f32 * cs;
                    let z1 = (k + 1) as f32 * cs;

                    if i == 0 || !is_solid(i - 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                            SluiceVertex::new([x0, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if i == width - 1 || !is_solid(i + 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == 0 || !is_solid(i, j - 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z1], color_bottom),
                            SluiceVertex::new([x0, y0, z1], color_bottom),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if j == height - 1 || !is_solid(i, j + 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z1], color_top),
                            SluiceVertex::new([x0, y1, z1], color_top),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    if k == 0 || !is_solid(i, j, k - 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    if k == depth - 1 || !is_solid(i, j, k + 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }
                }
            }
        }
    }

    pub fn vertices(&self) -> &[SluiceVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Baffle Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );

        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Baffle Index Buffer"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
    }

    pub fn update(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        if self.vertex_buffer.is_none() {
            self.upload(device);
            return;
        }

        let vb = self.vertex_buffer.as_ref().unwrap();
        let ib = self.index_buffer.as_ref().unwrap();

        let vertex_bytes = bytemuck::cast_slice(&self.vertices);
        let index_bytes = bytemuck::cast_slice(&self.indices);

        if vb.size() >= vertex_bytes.len() as u64 && ib.size() >= index_bytes.len() as u64 {
            queue.write_buffer(vb, 0, vertex_bytes);
            queue.write_buffer(ib, 0, index_bytes);
        } else {
            self.upload(device);
        }
    }

    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

// ============================================================================
// SHAKER DECK - Angled deck with grid of holes and side walls
// ============================================================================

#[derive(Clone, Debug)]
pub struct ShakerConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,

    /// Hole spacing in cells (center to center)
    pub hole_spacing: usize,
    /// Hole radius in cells
    pub hole_radius: usize,
    /// Deck thickness in cells
    pub deck_thickness: usize,
    /// Floor height at start (upstream, x=0)
    pub floor_height_start: usize,
    /// Floor height at end (downstream, x=max)
    pub floor_height_end: usize,
    /// Side wall height above deck in cells
    pub wall_height: usize,
    /// Side wall thickness in cells
    pub wall_thickness: usize,
    /// Gutter floor height at upstream end (lower, where it drains)
    pub gutter_floor_start: usize,
    /// Gutter floor height at downstream end (higher)
    pub gutter_floor_end: usize,
    /// Chute length at downstream end in cells (for deck overs)
    pub chute_length: usize,
    /// Gutter exit chute length at upstream end
    pub gutter_chute_length: usize,

    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
    pub color_gutter: [f32; 4],
}

impl Default for ShakerConfig {
    fn default() -> Self {
        Self {
            grid_width: 60,
            grid_height: 50,
            grid_depth: 30,
            cell_size: 0.02,
            hole_spacing: 4,
            hole_radius: 1,
            deck_thickness: 2,
            floor_height_start: 35, // Deck in upper portion
            floor_height_end: 25,
            wall_height: 10,
            wall_thickness: 2,
            gutter_floor_start: 3,            // Low at upstream (drains here)
            gutter_floor_end: 10,             // Higher at downstream (slopes back)
            chute_length: 8,                  // Chute at downstream end for deck overs
            gutter_chute_length: 6,           // Chute at upstream for gutter water
            color_top: [0.5, 0.5, 0.55, 1.0], // Metallic grey-blue
            color_side: [0.4, 0.4, 0.45, 1.0],
            color_bottom: [0.3, 0.3, 0.35, 1.0],
            color_gutter: [0.35, 0.35, 0.4, 1.0], // Darker for gutter
        }
    }
}

impl ShakerConfig {
    /// Get the deck floor height at a given x position (angled deck - high at start, low at end)
    pub fn floor_height_at(&self, x: usize) -> usize {
        let t = x as f32 / (self.grid_width - 1).max(1) as f32;
        let height = self.floor_height_start as f32 * (1.0 - t) + self.floor_height_end as f32 * t;
        height as usize
    }

    /// Get the gutter floor height at a given x position (angled - low at start, high at end)
    /// This creates a slope that drains water back toward the upstream end (x=0)
    pub fn gutter_floor_at(&self, x: usize) -> usize {
        let t = x as f32 / (self.grid_width - 1).max(1) as f32;
        let height = self.gutter_floor_start as f32 * (1.0 - t) + self.gutter_floor_end as f32 * t;
        height as usize
    }

    /// Check if a cell is a hole in the deck
    pub fn is_hole(&self, i: usize, k: usize) -> bool {
        if self.hole_spacing == 0 || self.hole_radius == 0 {
            return false;
        }

        // Find the center of the nearest hole in a grid pattern
        let spacing = self.hole_spacing;
        let offset_z = self.wall_thickness + self.hole_spacing / 2;

        // Grid of holes with spacing
        let hole_i = ((i + spacing / 2) / spacing) * spacing;
        let hole_k = ((k.saturating_sub(offset_z) + spacing / 2) / spacing) * spacing + offset_z;

        // Check if within radius (squared distance)
        let di = (i as i32 - hole_i as i32).abs() as usize;
        let dk = (k as i32 - hole_k as i32).abs() as usize;
        let dist_sq = di * di + dk * dk;
        let radius_sq = self.hole_radius * self.hole_radius;

        dist_sq <= radius_sq
    }

    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        let deck_floor_j = self.floor_height_at(i);
        let deck_top = deck_floor_j + self.deck_thickness;
        let wall_top = deck_top + self.wall_height;
        let gutter_floor_j = self.gutter_floor_at(i);

        // Side walls (full height from gutter to above deck)
        let in_left_wall = k < self.wall_thickness;
        let in_right_wall = k >= self.grid_depth - self.wall_thickness;

        // Side walls extend from gutter floor to wall top
        if (in_left_wall || in_right_wall) && j >= gutter_floor_j && j < wall_top {
            return true;
        }

        // Angled gutter floor (solid floor below deck to catch water, slopes toward x=0)
        if j == gutter_floor_j {
            return true;
        }

        // Deck floor (with holes)
        if j >= deck_floor_j && j < deck_top {
            // Check if in interior deck area (not walls)
            if k >= self.wall_thickness && k < self.grid_depth - self.wall_thickness {
                // Check if NOT a hole
                return !self.is_hole(i, k);
            }
            return true; // Wall regions are solid
        }

        // Downstream chute - back wall that directs deck overs down
        if i >= self.grid_width - self.chute_length {
            // Back wall of chute (closes off the downstream end)
            if i == self.grid_width - 1 && j >= gutter_floor_j && j < wall_top {
                return true;
            }
        }

        // Upstream gutter exit chute - directs gutter water down to sluice
        if i < self.gutter_chute_length {
            // Front wall above gutter (blocks deck entry except at top)
            if i == 0 {
                // Solid wall from gutter floor up to deck floor (gutter exit below)
                // Opening at bottom for gutter water to exit
                if j >= gutter_floor_j + 2 && j < deck_floor_j {
                    return true;
                }
                // Wall above deck for material containment
                if j >= deck_top && j < wall_top {
                    return true;
                }
            }
        }

        false
    }
}

pub struct ShakerGeometryBuilder {
    config: ShakerConfig,
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl ShakerGeometryBuilder {
    pub fn new(config: ShakerConfig) -> Self {
        Self {
            config,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    pub fn config(&self) -> &ShakerConfig {
        &self.config
    }

    pub fn solid_cells(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;

        (0..depth).flat_map(move |k| {
            (0..height).flat_map(move |j| {
                (0..width).filter_map(move |i| {
                    if self.config.is_solid(i, j, k) {
                        Some((i, j, k))
                    } else {
                        None
                    }
                })
            })
        })
    }

    pub fn build_mesh<F>(&mut self, is_solid: F)
    where
        F: Fn(usize, usize, usize) -> bool,
    {
        self.vertices.clear();
        self.indices.clear();

        let width = self.config.grid_width;
        let height = self.config.grid_height;
        let depth = self.config.grid_depth;
        let cs = self.config.cell_size;

        let color_top = self.config.color_top;
        let color_side = self.config.color_side;
        let color_bottom = self.config.color_bottom;

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    if !is_solid(i, j, k) {
                        continue;
                    }

                    let x0 = i as f32 * cs;
                    let x1 = (i + 1) as f32 * cs;
                    let y0 = j as f32 * cs;
                    let y1 = (j + 1) as f32 * cs;
                    let z0 = k as f32 * cs;
                    let z1 = (k + 1) as f32 * cs;

                    // -X face
                    if i == 0 || !is_solid(i - 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                            SluiceVertex::new([x0, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    // +X face
                    if i == width - 1 || !is_solid(i + 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    // -Y face
                    if j == 0 || !is_solid(i, j - 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z1], color_bottom),
                            SluiceVertex::new([x0, y0, z1], color_bottom),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    // +Y face
                    if j == height - 1 || !is_solid(i, j + 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z1], color_top),
                            SluiceVertex::new([x0, y1, z1], color_top),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }

                    // -Z face
                    if k == 0 || !is_solid(i, j, k - 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    }

                    // +Z face
                    if k == depth - 1 || !is_solid(i, j, k + 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[
                            base,
                            base + 2,
                            base + 1,
                            base,
                            base + 3,
                            base + 2,
                        ]);
                    }
                }
            }
        }
    }

    pub fn vertices(&self) -> &[SluiceVertex] {
        &self.vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shaker Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            }),
        );

        self.index_buffer = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shaker Index Buffer"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }),
        );
    }

    pub fn update(&mut self, queue: &wgpu::Queue, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        if self.vertex_buffer.is_none() {
            self.upload(device);
            return;
        }

        let vb = self.vertex_buffer.as_ref().unwrap();
        let ib = self.index_buffer.as_ref().unwrap();

        let vertex_bytes = bytemuck::cast_slice(&self.vertices);
        let index_bytes = bytemuck::cast_slice(&self.indices);

        if vb.size() >= vertex_bytes.len() as u64 && ib.size() >= index_bytes.len() as u64 {
            queue.write_buffer(vb, 0, vertex_bytes);
            queue.write_buffer(ib, 0, index_bytes);
        } else {
            self.upload(device);
        }
    }

    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Grate tests
    #[test]
    fn test_grate_default_config() {
        let config = GrateConfig::default();
        assert_eq!(config.grid_width, 20);
        assert_eq!(config.bar_spacing, 3);
    }

    #[test]
    fn test_grate_solid_cells() {
        let config = GrateConfig::default();
        let builder = GrateGeometryBuilder::new(config);
        let count = builder.solid_cells().count();
        assert!(count > 0);
    }

    #[test]
    fn test_grate_build_mesh() {
        let config = GrateConfig::default();
        let mut builder = GrateGeometryBuilder::new(config.clone());
        builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

        assert!(!builder.vertices().is_empty());
        assert!(!builder.indices().is_empty());
        assert_eq!(builder.indices().len() % 3, 0);
    }

    // Box tests
    #[test]
    fn test_box_default_config() {
        let config = BoxConfig::default();
        assert_eq!(config.grid_width, 20);
        assert_eq!(config.wall_thickness, 1);
    }

    #[test]
    fn test_box_hollow_interior() {
        let config = BoxConfig::default();
        let center_i = config.grid_width / 2;
        let center_j = config.grid_height / 2;
        let center_k = config.grid_depth / 2;
        assert!(!config.is_solid(center_i, center_j, center_k));
    }

    #[test]
    fn test_box_build_mesh() {
        let config = BoxConfig::default();
        let mut builder = BoxGeometryBuilder::new(config.clone());
        builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

        assert!(!builder.vertices().is_empty());
        assert!(!builder.indices().is_empty());
        assert_eq!(builder.indices().len() % 3, 0);
    }

    // Gutter tests
    #[test]
    fn test_gutter_default_config() {
        let config = GutterConfig::default();
        assert_eq!(config.grid_width, 20);
        assert_eq!(config.channel_width, 10);
    }

    #[test]
    fn test_gutter_build_mesh() {
        let config = GutterConfig::default();
        let mut builder = GutterGeometryBuilder::new(config.clone());
        builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

        assert!(!builder.vertices().is_empty());
        assert!(!builder.indices().is_empty());
        assert_eq!(builder.indices().len() % 3, 0);
    }

    // Hopper tests
    #[test]
    fn test_hopper_default_config() {
        let config = HopperConfig::default();
        assert_eq!(config.top_width, 16);
        assert_eq!(config.bottom_width, 4);
        assert!(config.top_width > config.bottom_width);
    }

    #[test]
    fn test_hopper_build_mesh() {
        let config = HopperConfig::default();
        let mut builder = HopperGeometryBuilder::new(config.clone());
        builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

        assert!(!builder.vertices().is_empty());
        assert!(!builder.indices().is_empty());
        assert_eq!(builder.indices().len() % 3, 0);
    }

    // Chute tests
    #[test]
    fn test_chute_default_config() {
        let config = ChuteConfig::default();
        assert_eq!(config.grid_width, 30);
        assert!(config.floor_height_start > config.floor_height_end);
    }

    #[test]
    fn test_chute_floor_slope() {
        let config = ChuteConfig::default();
        let h0 = config.floor_height_at(0);
        let h_end = config.floor_height_at(config.grid_width - 1);
        assert_eq!(h0, config.floor_height_start);
        assert_eq!(h_end, config.floor_height_end);
    }

    #[test]
    fn test_chute_build_mesh() {
        let config = ChuteConfig::default();
        let mut builder = ChuteGeometryBuilder::new(config.clone());
        builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

        assert!(!builder.vertices().is_empty());
        assert!(!builder.indices().is_empty());
        assert_eq!(builder.indices().len() % 3, 0);
    }

    // Frame tests
    #[test]
    fn test_frame_default_config() {
        let config = FrameConfig::default();
        assert_eq!(config.beam_thickness, 1);
    }

    #[test]
    fn test_frame_build_mesh() {
        let config = FrameConfig::default();
        let mut builder = FrameGeometryBuilder::new(config.clone());
        builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

        assert!(!builder.vertices().is_empty());
        assert!(!builder.indices().is_empty());
        assert_eq!(builder.indices().len() % 3, 0);
    }

    // Baffle tests
    #[test]
    fn test_baffle_default_config() {
        let config = BaffleConfig::default();
        assert_eq!(config.thickness, 1);
        assert_eq!(config.orientation, 0);
    }

    #[test]
    fn test_baffle_orientation() {
        let mut config = BaffleConfig::default();
        config.orientation = 0;
        let count_z = BaffleGeometryBuilder::new(config.clone())
            .solid_cells()
            .count();

        config.orientation = 1;
        let count_x = BaffleGeometryBuilder::new(config).solid_cells().count();

        assert_eq!(count_z, count_x);
    }

    #[test]
    fn test_baffle_build_mesh() {
        let config = BaffleConfig::default();
        let mut builder = BaffleGeometryBuilder::new(config.clone());
        builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

        assert!(!builder.vertices().is_empty());
        assert!(!builder.indices().is_empty());
        assert_eq!(builder.indices().len() % 3, 0);
    }

    // Edge case tests
    #[test]
    fn test_zero_dimensions_safe() {
        let config = GrateConfig {
            grid_width: 0,
            grid_height: 0,
            grid_depth: 0,
            ..Default::default()
        };
        let builder = GrateGeometryBuilder::new(config);
        assert_eq!(builder.solid_cells().count(), 0);
    }

    #[test]
    fn test_clear() {
        let config = GrateConfig::default();
        let mut builder = GrateGeometryBuilder::new(config.clone());
        builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

        assert!(!builder.vertices().is_empty());
        builder.clear();
        assert!(builder.vertices().is_empty());
        assert!(builder.indices().is_empty());
    }
}

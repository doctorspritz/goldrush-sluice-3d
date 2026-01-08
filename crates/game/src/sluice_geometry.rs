//! Reusable sluice geometry generation and rendering.
//!
//! This module provides a templatable sluice geometry builder that can:
//! - Generate solid cell positions for a sloped sluice with riffles
//! - Build an indexed mesh of exposed faces for efficient rendering
//! - Upload/update GPU buffers for wgpu rendering
//!
//! # Example
//!
//! ```rust,ignore
//! use game::sluice_geometry::{SluiceConfig, SluiceGeometryBuilder, SluiceVertex};
//!
//! // Configure the sluice
//! let config = SluiceConfig {
//!     grid_width: 160,
//!     grid_height: 24,
//!     grid_depth: 24,
//!     cell_size: 0.25,
//!     floor_height_left: 10,
//!     floor_height_right: 3,
//!     riffle_spacing: 12,
//!     riffle_height: 2,
//!     riffle_thickness: 2,
//!     riffle_start_x: 12,
//!     riffle_end_pad: 8,
//!     wall_margin: 4,
//!     exit_width_fraction: 0.67,  // 2/3 of depth
//!     exit_height: 8,
//!     ..Default::default()
//! };
//!
//! // Build geometry
//! let mut builder = SluiceGeometryBuilder::new(config);
//!
//! // Generate solid cells (call your grid's set_solid for each)
//! for (i, j, k) in builder.solid_cells() {
//!     sim.grid.set_solid(i, j, k);
//! }
//!
//! // Build the mesh from a solidity check function
//! builder.build_mesh(|i, j, k| sim.grid.is_solid(i, j, k));
//!
//! // Upload to GPU
//! builder.upload(&device);
//!
//! // Use for rendering
//! let vertices = builder.vertices();
//! let indices = builder.indices();
//! ```

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Vertex type for sluice mesh rendering.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct SluiceVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl SluiceVertex {
    pub fn new(position: [f32; 3], color: [f32; 4]) -> Self {
        Self { position, color }
    }

    /// Returns the vertex buffer layout for wgpu pipelines.
    pub fn buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SluiceVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// Configuration for sluice geometry generation.
#[derive(Clone, Debug)]
pub struct SluiceConfig {
    /// Grid dimensions (in cells)
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,

    /// Size of each cell in world units
    pub cell_size: f32,

    /// Floor height at left (upstream) end, in cells
    pub floor_height_left: usize,
    /// Floor height at right (downstream) end, in cells
    pub floor_height_right: usize,

    /// Spacing between riffles, in cells
    pub riffle_spacing: usize,
    /// Height of riffles above floor, in cells
    pub riffle_height: usize,
    /// Thickness of each riffle, in cells
    pub riffle_thickness: usize,
    /// X position where riffles start, in cells
    pub riffle_start_x: usize,
    /// Padding from right edge where riffles end, in cells
    pub riffle_end_pad: usize,

    /// Wall height above floor + riffle, in cells
    pub wall_margin: usize,

    /// Exit opening width as fraction of depth (0.0 to 1.0)
    pub exit_width_fraction: f32,
    /// Exit opening height above floor, in cells
    pub exit_height: usize,

    /// Color for top faces (most visible)
    pub color_top: [f32; 4],
    /// Color for side faces
    pub color_side: [f32; 4],
    /// Color for bottom faces (least visible)
    pub color_bottom: [f32; 4],
}

impl Default for SluiceConfig {
    fn default() -> Self {
        Self {
            grid_width: 160,
            grid_height: 24,
            grid_depth: 24,
            cell_size: 0.25,
            floor_height_left: 10,
            floor_height_right: 3,
            riffle_spacing: 12,
            riffle_height: 2,
            riffle_thickness: 2,
            riffle_start_x: 12,
            riffle_end_pad: 8,
            wall_margin: 4,
            exit_width_fraction: 0.67,
            exit_height: 8,
            color_top: [0.55, 0.50, 0.45, 1.0],
            color_side: [0.45, 0.40, 0.35, 1.0],
            color_bottom: [0.35, 0.30, 0.25, 1.0],
        }
    }
}

impl SluiceConfig {
    /// Calculate the floor height at a given X position (in cells).
    pub fn floor_height_at(&self, x: usize) -> usize {
        let t = x as f32 / (self.grid_width - 1).max(1) as f32;
        let height = self.floor_height_left as f32 * (1.0 - t)
            + self.floor_height_right as f32 * t;
        height as usize
    }

    /// Check if a cell position is a riffle.
    pub fn is_riffle(&self, i: usize, j: usize) -> bool {
        let riffle_end_x = self.grid_width.saturating_sub(self.riffle_end_pad);
        let floor_j = self.floor_height_at(i);

        i >= self.riffle_start_x
            && i < riffle_end_x
            && (i - self.riffle_start_x) % self.riffle_spacing < self.riffle_thickness
            && j <= floor_j + self.riffle_height
            && j > floor_j
    }

    /// Check if a cell position is in the exit opening.
    pub fn is_exit(&self, i: usize, j: usize, k: usize) -> bool {
        let exit_start_z = ((1.0 - self.exit_width_fraction) / 2.0 * self.grid_depth as f32) as usize;
        let exit_end_z = self.grid_depth - exit_start_z;
        let floor_j = self.floor_height_at(i);

        i == self.grid_width - 1
            && k >= exit_start_z
            && k < exit_end_z
            && j > floor_j
            && j <= floor_j + self.exit_height
    }

    /// Check if a cell should be solid based on sluice geometry.
    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        let floor_j = self.floor_height_at(i);
        let wall_top = floor_j + self.riffle_height + self.wall_margin;

        // Floor
        if j <= floor_j {
            return true;
        }

        // Riffles
        if self.is_riffle(i, j) {
            return true;
        }

        // Left wall
        if i == 0 && j <= wall_top {
            return true;
        }

        // Right wall (except exit)
        if i == self.grid_width - 1 && !self.is_exit(i, j, k) && j <= wall_top {
            return true;
        }

        // Front and back walls
        if (k == 0 || k == self.grid_depth - 1) && j <= wall_top {
            return true;
        }

        false
    }

    /// Get the slope angle in degrees.
    pub fn slope_degrees(&self) -> f32 {
        let drop = (self.floor_height_left as i32 - self.floor_height_right as i32) as f32;
        let run = self.grid_width as f32;
        (drop / run).atan().to_degrees()
    }

    /// Get the number of riffles.
    pub fn num_riffles(&self) -> usize {
        let riffle_end_x = self.grid_width.saturating_sub(self.riffle_end_pad);
        if riffle_end_x > self.riffle_start_x {
            (riffle_end_x - self.riffle_start_x) / self.riffle_spacing
        } else {
            0
        }
    }
}

/// Builder for sluice geometry and mesh.
pub struct SluiceGeometryBuilder {
    config: SluiceConfig,
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl SluiceGeometryBuilder {
    /// Create a new geometry builder with the given configuration.
    pub fn new(config: SluiceConfig) -> Self {
        Self {
            config,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &SluiceConfig {
        &self.config
    }

    /// Iterator over all solid cell positions (i, j, k).
    ///
    /// Use this to populate your simulation grid's solid cells.
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

    /// Build the mesh from a solidity check function.
    ///
    /// The `is_solid` function should return true if the cell at (i, j, k) is solid.
    /// Only exposed faces (adjacent to non-solid cells) are included in the mesh.
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

                    // -X face (left)
                    if i == 0 || !is_solid(i - 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                            SluiceVertex::new([x0, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
                    }

                    // +X face (right)
                    if i == width - 1 || !is_solid(i + 1, j, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                    }

                    // -Y face (bottom)
                    if j == 0 || !is_solid(i, j - 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z0], color_bottom),
                            SluiceVertex::new([x1, y0, z1], color_bottom),
                            SluiceVertex::new([x0, y0, z1], color_bottom),
                        ]);
                        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                    }

                    // +Y face (top) - most visible
                    if j == height - 1 || !is_solid(i, j + 1, k) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z0], color_top),
                            SluiceVertex::new([x1, y1, z1], color_top),
                            SluiceVertex::new([x0, y1, z1], color_top),
                        ]);
                        self.indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
                    }

                    // -Z face (front)
                    if k == 0 || !is_solid(i, j, k - 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z0], color_side),
                            SluiceVertex::new([x1, y0, z0], color_side),
                            SluiceVertex::new([x1, y1, z0], color_side),
                            SluiceVertex::new([x0, y1, z0], color_side),
                        ]);
                        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                    }

                    // +Z face (back)
                    if k == depth - 1 || !is_solid(i, j, k + 1) {
                        let base = self.vertices.len() as u32;
                        self.vertices.extend_from_slice(&[
                            SluiceVertex::new([x0, y0, z1], color_side),
                            SluiceVertex::new([x1, y0, z1], color_side),
                            SluiceVertex::new([x1, y1, z1], color_side),
                            SluiceVertex::new([x0, y1, z1], color_side),
                        ]);
                        self.indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
                    }
                }
            }
        }
    }

    /// Build mesh using the config's own is_solid check.
    ///
    /// This is a convenience method when you don't need a custom solidity function.
    pub fn build_mesh_from_config(&mut self) {
        // We need to clone the config to avoid borrow issues
        let config = self.config.clone();
        self.build_mesh(|i, j, k| config.is_solid(i, j, k));
    }

    /// Get the vertices.
    pub fn vertices(&self) -> &[SluiceVertex] {
        &self.vertices
    }

    /// Get the indices.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Get the number of indices (for draw calls).
    pub fn num_indices(&self) -> u32 {
        self.indices.len() as u32
    }

    /// Get the number of triangles.
    pub fn num_triangles(&self) -> usize {
        self.indices.len() / 3
    }

    /// Upload the mesh to GPU buffers.
    pub fn upload(&mut self, device: &wgpu::Device) {
        if self.vertices.is_empty() {
            return;
        }

        self.vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sluice Vertex Buffer"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        }));

        self.index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sluice Index Buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        }));
    }

    /// Update existing GPU buffers, or create new ones if needed.
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

        // Check if buffers are large enough
        if vb.size() >= vertex_bytes.len() as u64 && ib.size() >= index_bytes.len() as u64 {
            queue.write_buffer(vb, 0, vertex_bytes);
            queue.write_buffer(ib, 0, index_bytes);
        } else {
            // Recreate larger buffers
            self.upload(device);
        }
    }

    /// Get the vertex buffer (if uploaded).
    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref()
    }

    /// Get the index buffer (if uploaded).
    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref()
    }

    /// Clear all mesh data.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Print summary statistics about the sluice.
    pub fn print_summary(&self) {
        let config = &self.config;
        println!(
            "Sluice: {}x{}x{} grid, cell_size={:.3}",
            config.grid_width, config.grid_height, config.grid_depth, config.cell_size
        );
        println!(
            "  Slope: {:.1}° ({} → {} cells)",
            config.slope_degrees(),
            config.floor_height_left,
            config.floor_height_right
        );
        println!(
            "  {} riffles (spacing={}, height={}, thickness={})",
            config.num_riffles(),
            config.riffle_spacing,
            config.riffle_height,
            config.riffle_thickness
        );
        if !self.vertices.is_empty() {
            println!(
                "  Mesh: {} vertices, {} indices ({} triangles)",
                self.vertices.len(),
                self.indices.len(),
                self.num_triangles()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SluiceConfig::default();
        assert_eq!(config.grid_width, 160);
        assert_eq!(config.grid_height, 24);
        assert_eq!(config.grid_depth, 24);
    }

    #[test]
    fn test_floor_height_interpolation() {
        let config = SluiceConfig {
            grid_width: 100,
            floor_height_left: 10,
            floor_height_right: 0,
            ..Default::default()
        };

        assert_eq!(config.floor_height_at(0), 10);
        // At x=50: t = 50/99 ≈ 0.505, height = 10 * (1 - 0.505) ≈ 4.95 → 4
        assert_eq!(config.floor_height_at(50), 4);
        assert_eq!(config.floor_height_at(99), 0);
    }

    #[test]
    fn test_solid_cells_iterator() {
        let config = SluiceConfig {
            grid_width: 10,
            grid_height: 5,
            grid_depth: 5,
            floor_height_left: 2,
            floor_height_right: 1,
            riffle_spacing: 100, // No riffles in this small test
            wall_margin: 1,
            ..Default::default()
        };

        let builder = SluiceGeometryBuilder::new(config);
        let solid_count = builder.solid_cells().count();
        assert!(solid_count > 0, "Should have some solid cells");
    }

    #[test]
    fn test_build_mesh() {
        let config = SluiceConfig {
            grid_width: 10,
            grid_height: 5,
            grid_depth: 5,
            floor_height_left: 2,
            floor_height_right: 1,
            riffle_spacing: 100,
            wall_margin: 1,
            ..Default::default()
        };

        let mut builder = SluiceGeometryBuilder::new(config);
        builder.build_mesh_from_config();

        assert!(!builder.vertices().is_empty(), "Should have vertices");
        assert!(!builder.indices().is_empty(), "Should have indices");
        assert_eq!(builder.indices().len() % 3, 0, "Indices should be triangles");
    }

    #[test]
    fn test_slope_degrees() {
        let config = SluiceConfig {
            grid_width: 160,
            floor_height_left: 10,
            floor_height_right: 4,
            ..Default::default()
        };

        let slope = config.slope_degrees();
        // 6 cell drop over 160 cells ≈ 2.1°
        assert!(slope > 2.0 && slope < 2.5, "Slope should be around 2°, got {}", slope);
    }

    #[test]
    fn test_num_riffles() {
        let config = SluiceConfig {
            grid_width: 160,
            riffle_start_x: 12,
            riffle_end_pad: 8,
            riffle_spacing: 12,
            ..Default::default()
        };

        let num = config.num_riffles();
        // (160 - 8 - 12) / 12 = 140 / 12 = 11
        assert_eq!(num, 11);
    }
}

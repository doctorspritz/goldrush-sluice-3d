//! Test SDF geometry for isolated physics tests.
//!
//! This module provides methods to set up test signed distance fields:
//! - Test floor for simple collision tests
//! - Test box for enclosed container tests

use crate::sluice_geometry::SluiceVertex;
use glam::Vec3;
use sim3d::test_geometry::{TestBox, TestFloor, TestSdfGenerator};

use super::constants::SIM_CELL_SIZE;
use super::MultiGridSim;

impl MultiGridSim {
    /// Set up test SDF using TestFloor geometry for isolated floor collision tests.
    pub fn setup_test_floor(&mut self, floor_y: f32) {
        let cell_size = SIM_CELL_SIZE;
        let width = 40usize; // 1m
        let height = 60usize; // 1.5m
        let depth = 40usize; // 1m
        let offset = Vec3::new(-0.5, -0.5, -0.5); // Grid origin in world space

        // TestFloor works in WORLD coordinates (cell_center returns world pos)
        let floor = TestFloor::with_thickness(floor_y, cell_size * 4.0);
        let mut gen = TestSdfGenerator::new(width, height, depth, cell_size, offset);
        gen.add_floor(&floor);

        self.test_sdf = Some(gen.sdf);
        self.test_sdf_dims = (width, height, depth);
        self.test_sdf_cell_size = cell_size;
        self.test_sdf_offset = offset;

        println!(
            "Test SDF: floor at world y={} with {}x{}x{} grid, offset {:?}",
            floor_y, width, height, depth, offset
        );

        // Generate debug mesh for floor
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let _base_idx = 0;
        let color = [0.5, 0.5, 0.5, 1.0]; // Grey floor

        let x_min = offset.x;
        let x_max = offset.x + width as f32 * cell_size;
        let z_min = offset.z;
        let z_max = offset.z + depth as f32 * cell_size;
        let y_top = floor_y;
        let y_bottom = floor_y - cell_size * 4.0;

        // Top face
        vertices.push(SluiceVertex::new([x_min, y_top, z_min], color)); // 0
        vertices.push(SluiceVertex::new([x_max, y_top, z_min], color)); // 1
        vertices.push(SluiceVertex::new([x_min, y_top, z_max], color)); // 2
        vertices.push(SluiceVertex::new([x_max, y_top, z_max], color)); // 3

        // Bottom face
        vertices.push(SluiceVertex::new([x_min, y_bottom, z_min], color)); // 4
        vertices.push(SluiceVertex::new([x_max, y_bottom, z_min], color)); // 5
        vertices.push(SluiceVertex::new([x_min, y_bottom, z_max], color)); // 6
        vertices.push(SluiceVertex::new([x_max, y_bottom, z_max], color)); // 7

        // Indices
        // Top
        indices.extend_from_slice(&[0, 2, 1, 1, 2, 3]);
        // Bottom
        indices.extend_from_slice(&[4, 5, 6, 6, 5, 7]);
        // Sides
        indices.extend_from_slice(&[0, 1, 4, 4, 1, 5]); // Front
        indices.extend_from_slice(&[2, 6, 3, 3, 6, 7]); // Back
        indices.extend_from_slice(&[0, 4, 2, 2, 4, 6]); // Left
        indices.extend_from_slice(&[1, 3, 5, 5, 3, 7]); // Right

        self.test_mesh = Some((vertices, indices));
    }

    /// Set up test SDF using TestBox geometry for isolated box collision tests.
    pub fn setup_test_box(&mut self, center: Vec3, width: f32, depth: f32, wall_height: f32) {
        let cell_size = SIM_CELL_SIZE;
        let grid_width = 60usize; // 1.5m
        let grid_height = 60usize; // 1.5m
        let grid_depth = 60usize; // 1.5m
        let offset = center - Vec3::splat(cell_size * grid_width as f32 / 2.0);

        // TestBox works in WORLD coordinates (cell_center returns world pos)
        let wall_t = cell_size * 4.0;
        let floor_t = cell_size * 4.0;

        let test_box = TestBox::with_thickness(center, width, depth, wall_height, wall_t, floor_t);
        let mut gen = TestSdfGenerator::new(grid_width, grid_height, grid_depth, cell_size, offset);
        gen.add_box(&test_box);

        self.test_sdf = Some(gen.sdf);
        self.test_sdf_dims = (grid_width, grid_height, grid_depth);
        self.test_sdf_cell_size = cell_size;
        self.test_sdf_offset = offset;

        println!(
            "Test SDF: box at world {:?}, size {}x{}x{}, offset {:?}",
            center, width, depth, wall_height, offset
        );

        // Generate debug mesh for box
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let color = [0.5, 0.5, 0.5, 1.0];

        // Helper to add a box
        let mut add_box = |min: Vec3, max: Vec3| {
            let base = vertices.len() as u32;
            // Top (y=max.y)
            vertices.push(SluiceVertex::new([min.x, max.y, min.z], color)); // 0
            vertices.push(SluiceVertex::new([max.x, max.y, min.z], color)); // 1
            vertices.push(SluiceVertex::new([min.x, max.y, max.z], color)); // 2
            vertices.push(SluiceVertex::new([max.x, max.y, max.z], color)); // 3
                                                                            // Bottom (y=min.y)
            vertices.push(SluiceVertex::new([min.x, min.y, min.z], color)); // 4
            vertices.push(SluiceVertex::new([max.x, min.y, min.z], color)); // 5
            vertices.push(SluiceVertex::new([min.x, min.y, max.z], color)); // 6
            vertices.push(SluiceVertex::new([max.x, min.y, max.z], color)); // 7

            // Indices (same as floor)
            let i = [
                0, 2, 1, 1, 2, 3, 4, 5, 6, 6, 5, 7, 0, 1, 4, 4, 1, 5, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2,
                4, 6, 1, 3, 5, 5, 3, 7,
            ];
            indices.extend(i.iter().map(|idx| idx + base));
        };

        // Floor
        let _floor_top = center.y - wall_height * 0.5; // Assuming box center is center of void? No, test_box logic varies.
                                                       // Let's assume passed center is center of the VOID region.
                                                       // Box extends from -width/2 to +width/2 relative to center.
        let min_x = center.x - width * 0.5;
        let max_x = center.x + width * 0.5;
        let min_z = center.z - depth * 0.5;
        let max_z = center.z + depth * 0.5;

        let min_y = center.y; // Floor surface
                              // Add Floor plate
        add_box(
            Vec3::new(min_x - wall_t, min_y - floor_t, min_z - wall_t),
            Vec3::new(max_x + wall_t, min_y, max_z + wall_t),
        );

        // Walls (up to min_y + wall_height)
        let max_y = min_y + wall_height;

        // -X Wall
        add_box(
            Vec3::new(min_x - wall_t, min_y, min_z),
            Vec3::new(min_x, max_y, max_z),
        );
        // +X Wall
        add_box(
            Vec3::new(max_x, min_y, min_z),
            Vec3::new(max_x + wall_t, max_y, max_z),
        );
        // -Z Wall
        add_box(
            Vec3::new(min_x - wall_t, min_y, min_z - wall_t),
            Vec3::new(max_x + wall_t, max_y, min_z),
        );
        // +Z Wall
        add_box(
            Vec3::new(min_x - wall_t, min_y, max_z),
            Vec3::new(max_x + wall_t, max_y, max_z + wall_t),
        );

        self.test_mesh = Some((vertices, indices));
    }

    /// Clear the test SDF (return to using piece SDFs).
    pub fn clear_test_sdf(&mut self) {
        self.test_sdf = None;
        self.test_mesh = None;
    }
}

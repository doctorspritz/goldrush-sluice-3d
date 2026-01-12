//! Washplant Editor - Layout Editor for Gutters and Sluices
//!
//! A simple CAD-style editor for placing and configuring washplant pieces.
//!
//! Run with: cargo run --example washplant_editor --release
//!
//! Controls:
//!   WASD / QE      - Move camera
//!   Mouse drag     - Rotate camera
//!   Scroll         - Zoom in/out
//!   Arrow keys     - Move selected piece (XZ plane)
//!   Shift+Up/Down  - Move selected piece (Y axis)
//!   R / Shift+R    - Rotate 90 degrees CW/CCW
//!   [ / ]          - Decrease/increase angle
//!   - / =          - Decrease/increase length
//!   , / .          - Decrease/increase width
//!   1              - Place gutter mode
//!   2              - Place sluice mode
//!   3              - Place emitter mode
//!   Enter          - Confirm placement
//!   Escape         - Cancel / Deselect
//!   Delete         - Remove selected piece
//!   Shift+S        - Save layout to JSON
//!   L              - Load layout from JSON
//!   P              - Toggle Play mode (run simulation)
//!   N              - Toggle snap mode (outlet→inlet alignment)

use bytemuck::{Pod, Zeroable};
use game::editor::{
    EditorLayout, EditorMode, EmitterPiece, GutterPiece, Rotation, Selection, SluicePiece,
};
use game::gpu::flip_3d::GpuFlip3D;
use game::gpu::fluid_renderer::ScreenSpaceFluidRenderer;
use game::sluice_geometry::SluiceVertex;
use glam::{Mat3, Mat4, Vec3};
use sim3d::FlipSimulation3D;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const MOVE_STEP: f32 = 0.1; // Movement step in meters
const ANGLE_STEP: f32 = 2.0; // Angle adjustment in degrees
const LENGTH_STEP: f32 = 0.1; // Length adjustment in meters
const WIDTH_STEP: f32 = 0.05; // Width adjustment in meters
const SNAP_DISTANCE: f32 = 0.15; // Snap threshold in meters

// Simulation constants
const SIM_CELL_SIZE: f32 = 0.025; // 2.5cm cells (larger for speed)
const SIM_MAX_PARTICLES: usize = 50_000;
const SIM_PRESSURE_ITERS: usize = 30; // Reduced for speed
const SIM_SUBSTEPS: u32 = 2;
const SIM_GRAVITY: f32 = -9.8;

// Grid size limits for reasonable FPS
const MAX_GRID_WIDTH: usize = 120;
const MAX_GRID_HEIGHT: usize = 80;
const MAX_GRID_DEPTH: usize = 60;

/// Simple random float [0, 1) using atomic counter
fn rand_float() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let seed = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Simple xorshift-style hash
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x = x ^ (x >> 31);
    (x as f32) / (u64::MAX as f32)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

// Colors for pieces
const GUTTER_COLOR: [f32; 4] = [0.55, 0.50, 0.45, 1.0];
const GUTTER_SELECTED: [f32; 4] = [0.8, 0.7, 0.3, 1.0];
const SLUICE_COLOR: [f32; 4] = [0.3, 0.5, 0.7, 1.0];
const SLUICE_SELECTED: [f32; 4] = [0.4, 0.7, 0.9, 1.0];
const EMITTER_COLOR: [f32; 4] = [0.2, 0.6, 0.9, 1.0];
const EMITTER_SELECTED: [f32; 4] = [0.4, 0.8, 1.0, 1.0];
const PREVIEW_COLOR: [f32; 4] = [0.5, 0.8, 0.5, 0.7];
const GRID_COLOR: [f32; 4] = [0.3, 0.3, 0.3, 0.5];

// ============================================================================
// Multi-Grid Simulation Types
// ============================================================================

/// Which type of piece this simulation belongs to
#[derive(Clone, Copy, Debug)]
enum PieceKind {
    Gutter(usize), // index into layout.gutters
    Sluice(usize), // index into layout.sluices
}

/// Per-piece simulation grid
struct PieceSimulation {
    kind: PieceKind,

    // Grid configuration
    grid_offset: Vec3, // World position of grid origin
    grid_dims: (usize, usize, usize),
    cell_size: f32,

    // Simulation state
    sim: FlipSimulation3D,
    gpu_flip: Option<GpuFlip3D>,

    // Particle data buffers
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine_vels: Vec<Mat3>,
    densities: Vec<f32>,
}

/// Defines particle transfer between two pieces
struct PieceTransfer {
    from_piece: usize, // Index into MultiGridSim::pieces
    to_piece: usize,

    // Capture region (in from_piece's sim-space)
    capture_min: Vec3,
    capture_max: Vec3,

    // Injection position (in to_piece's sim-space)
    inject_pos: Vec3,
    inject_vel: Vec3,

    // Particles in transit
    transit_queue: Vec<(Vec3, Vec3, f32)>, // (position, velocity, density)
    transit_time: f32,
}

/// Multi-grid simulation manager
struct MultiGridSim {
    pieces: Vec<PieceSimulation>,
    transfers: Vec<PieceTransfer>,
    frame: u32,
}

impl MultiGridSim {
    fn new() -> Self {
        Self {
            pieces: Vec::new(),
            transfers: Vec::new(),
            frame: 0,
        }
    }

    /// Add a gutter piece simulation
    fn add_gutter(
        &mut self,
        device: &wgpu::Device,
        gutter: &GutterPiece,
        gutter_idx: usize,
    ) -> usize {
        // Calculate grid dimensions based on gutter size (use max_width for variable-width gutters)
        let cell_size = SIM_CELL_SIZE;
        let margin = cell_size * 4.0;
        let max_width = gutter.max_width();

        let width = ((gutter.length + margin * 2.0) / cell_size).ceil() as usize;
        let height = ((gutter.wall_height + margin + 0.5) / cell_size).ceil() as usize;
        let depth = ((max_width + margin * 2.0) / cell_size).ceil() as usize;

        // Clamp to reasonable sizes
        let width = width.clamp(10, 60);
        let height = height.clamp(10, 40);
        let depth = depth.clamp(10, 40);

        // Grid origin is at gutter center minus margin
        let (dir_x, dir_z) = match gutter.rotation {
            Rotation::R0 => (1.0f32, 0.0f32),
            Rotation::R90 => (0.0, 1.0),
            Rotation::R180 => (-1.0, 0.0),
            Rotation::R270 => (0.0, -1.0),
        };

        // Grid offset must account for both length and width based on rotation
        // For R0 (dir_x=1, dir_z=0): length along X, width along Z
        // For R90 (dir_x=0, dir_z=1): length along Z, width along X
        // Use max_width for grid sizing to accommodate variable-width gutters
        let grid_offset = Vec3::new(
            gutter.position.x
                - gutter.length / 2.0 * dir_x.abs()
                - max_width / 2.0 * dir_z.abs()
                - margin,
            gutter.position.y - margin,
            gutter.position.z
                - gutter.length / 2.0 * dir_z.abs()
                - max_width / 2.0 * dir_x.abs()
                - margin,
        );

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);
        sim.pressure_iterations = SIM_PRESSURE_ITERS;

        // Mark gutter solid cells (in local grid space)
        // Use max_width for centering so both inlet and outlet widths fit
        let gutter_local = GutterPiece {
            id: 0, // Temporary local piece
            position: Vec3::new(
                margin + gutter.length / 2.0,
                margin,
                margin + gutter.max_width() / 2.0,
            ),
            rotation: Rotation::R0, // Local space, no rotation needed
            angle_deg: gutter.angle_deg,
            length: gutter.length,
            width: gutter.width,
            end_width: gutter.end_width,
            wall_height: gutter.wall_height,
        };
        Self::mark_gutter_solid_cells(&mut sim, &gutter_local, cell_size);

        sim.grid.compute_sdf();

        let mut gpu_flip = GpuFlip3D::new(
            device,
            width as u32,
            height as u32,
            depth as u32,
            cell_size,
            20000, // Max particles per piece
        );
        // Gutter: +X boundary is open (outlet) - particles exit here for transfer
        // Bit 1 (value 2) = +X open
        gpu_flip.open_boundaries = 2;

        let idx = self.pieces.len();
        self.pieces.push(PieceSimulation {
            kind: PieceKind::Gutter(gutter_idx),
            grid_offset,
            grid_dims: (width, height, depth),
            cell_size,
            sim,
            gpu_flip: Some(gpu_flip),
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
        });

        idx
    }

    /// Add a sluice piece simulation
    fn add_sluice(
        &mut self,
        device: &wgpu::Device,
        sluice: &SluicePiece,
        sluice_idx: usize,
    ) -> usize {
        let cell_size = SIM_CELL_SIZE;
        let margin = cell_size * 4.0;

        let width = ((sluice.length + margin * 2.0) / cell_size).ceil() as usize;
        // Height needs to accommodate water coming from above (gutters can be higher)
        let height = ((0.8 + margin) / cell_size).ceil() as usize; // Taller to catch incoming water
        let depth = ((sluice.width + margin * 2.0) / cell_size).ceil() as usize;

        let width = width.clamp(10, 80);
        let height = height.clamp(10, 30);
        let depth = depth.clamp(10, 40);

        // Grid origin based on rotation
        let (dir_x, dir_z) = match sluice.rotation {
            Rotation::R0 => (1.0f32, 0.0f32),
            Rotation::R90 => (0.0, 1.0),
            Rotation::R180 => (-1.0, 0.0),
            Rotation::R270 => (0.0, -1.0),
        };

        // Grid offset must account for both length and width based on rotation
        let grid_offset = Vec3::new(
            sluice.position.x
                - sluice.length / 2.0 * dir_x.abs()
                - sluice.width / 2.0 * dir_z.abs()
                - margin,
            sluice.position.y - margin,
            sluice.position.z
                - sluice.length / 2.0 * dir_z.abs()
                - sluice.width / 2.0 * dir_x.abs()
                - margin,
        );

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);
        sim.pressure_iterations = SIM_PRESSURE_ITERS;

        // Mark sluice solid cells
        let sluice_local = SluicePiece {
            id: 0, // Temporary local piece
            position: Vec3::new(
                margin + sluice.length / 2.0,
                margin,
                margin + sluice.width / 2.0,
            ),
            rotation: Rotation::R0,
            length: sluice.length,
            width: sluice.width,
            slope_deg: sluice.slope_deg,
            riffle_spacing: sluice.riffle_spacing,
            riffle_height: sluice.riffle_height,
        };
        Self::mark_sluice_solid_cells(&mut sim, &sluice_local, cell_size);

        sim.grid.compute_sdf();

        let mut gpu_flip = GpuFlip3D::new(
            device,
            width as u32,
            height as u32,
            depth as u32,
            cell_size,
            30000,
        );
        // Sluice: -X (inlet) and +X (outlet) boundaries are open
        // Bit 0 (1) = -X open, Bit 1 (2) = +X open -> combined = 3
        gpu_flip.open_boundaries = 3;

        let idx = self.pieces.len();
        self.pieces.push(PieceSimulation {
            kind: PieceKind::Sluice(sluice_idx),
            grid_offset,
            grid_dims: (width, height, depth),
            cell_size,
            sim,
            gpu_flip: Some(gpu_flip),
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
        });

        idx
    }

    /// Mark gutter solid cells using cell-index approach (like friction_sluice)
    /// Supports variable width gutters (funnel effect) via width_at()
    fn mark_gutter_solid_cells(sim: &mut FlipSimulation3D, gutter: &GutterPiece, cell_size: f32) {
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;

        // Gutter position gives center in X/Z and base floor height in Y
        let center_i = (gutter.position.x / cell_size).round() as i32;
        let base_j = (gutter.position.y / cell_size).round() as i32;
        let center_k = (gutter.position.z / cell_size).round() as i32;

        // Channel dimensions in cells
        let half_len_cells = ((gutter.length / 2.0) / cell_size).ceil() as i32;
        // Inlet (start) half width for back wall
        let inlet_half_wid_cells = ((gutter.width / 2.0) / cell_size).ceil() as i32;

        // Floor height drop due to angle (in cells)
        let angle_rad = gutter.angle_deg.to_radians();
        let total_drop = gutter.length * angle_rad.tan();
        let half_drop_cells = ((total_drop / 2.0) / cell_size).round() as i32;

        // Floor heights at inlet (left) and outlet (right)
        let floor_j_left = base_j + half_drop_cells; // Inlet is higher
        let floor_j_right = base_j - half_drop_cells; // Outlet is lower

        // Wall parameters
        let wall_height_cells = ((gutter.wall_height / cell_size).ceil() as i32).max(8);
        let wall_thick_cells = 2_i32;

        // Channel bounds in i (length direction)
        let i_start = (center_i - half_len_cells).max(0) as usize;
        let i_end = ((center_i + half_len_cells) as usize).min(width);

        // Outlet width (at end of channel)
        let outlet_half_wid_cells = ((gutter.width_at(1.0) / 2.0) / cell_size).ceil() as i32;

        for i in 0..width {
            // Calculate position along gutter (0.0 = inlet, 1.0 = outlet)
            let i_i = i as i32;
            let t = if i_i <= center_i - half_len_cells {
                0.0
            } else if i_i >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };

            // Variable width at this position (funnel effect)
            let local_width = gutter.width_at(t);
            let half_wid_cells = ((local_width / 2.0) / cell_size).ceil() as i32;

            // Channel bounds in k (width direction) - variable!
            let k_start = (center_k - half_wid_cells).max(0) as usize;
            let k_end = ((center_k + half_wid_cells) as usize).min(depth);

            // Compute floor_j directly from mesh floor position to align with visual
            // Mesh floor Y at this position = gutter.position.y + half_drop - t * total_drop
            // Solid top should be at mesh floor, so floor_j = floor(mesh_floor_Y / cell_size)
            let mesh_floor_y = gutter.position.y + (total_drop / 2.0) - t * total_drop;
            let floor_j = (mesh_floor_y / cell_size).floor() as i32;

            // Look BOTH forward AND backward to cover staircase transitions
            // At floor drops, both adjacent cells need the higher floor_j to prevent diagonal gaps
            let t_next = if i_i + 1 >= center_i + half_len_cells {
                1.0
            } else if i_i + 1 <= center_i - half_len_cells {
                0.0
            } else {
                ((i_i + 1 - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };
            let mesh_floor_y_next = gutter.position.y + (total_drop / 2.0) - t_next * total_drop;
            let floor_j_next = (mesh_floor_y_next / cell_size).floor() as i32;

            let t_prev = if i_i - 1 <= center_i - half_len_cells {
                0.0
            } else if i_i - 1 >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - 1 - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };
            let mesh_floor_y_prev = gutter.position.y + (total_drop / 2.0) - t_prev * total_drop;
            let floor_j_prev = (mesh_floor_y_prev / cell_size).floor() as i32;

            // Use the HIGHEST of prev, current, and next floor_j to prevent staircase gaps
            let effective_floor_j = floor_j.max(floor_j_next).max(floor_j_prev);
            let wall_top_j = effective_floor_j + wall_height_cells;

            // Is this position past the channel outlet? (in the margin zone leading to grid edge)
            let past_outlet = i >= i_end;

            for k in 0..depth {
                let k_i = k as i32;
                let in_channel_width = k >= k_start && k < k_end;
                let in_channel_length = i >= i_start && i < i_end;

                // For the margin zone past the outlet, use outlet dimensions
                let in_outlet_chute = past_outlet
                    && k_i >= (center_k - outlet_half_wid_cells)
                    && k_i < (center_k + outlet_half_wid_cells);

                // Outlet floor_j computed from mesh position at t=1.0
                let outlet_floor_j = ((gutter.position.y - total_drop / 2.0) / cell_size).floor() as i32;

                for j in 0..height {
                    let j_i = j as i32;

                    // Floor - fill ALL cells at and below effective_floor_j within channel
                    // Use effective_floor_j for channel, outlet_floor_j for outlet chute
                    let is_channel_floor =
                        in_channel_length && in_channel_width && j_i <= effective_floor_j;
                    let is_outlet_chute_floor = in_outlet_chute && j_i <= outlet_floor_j;
                    let is_floor = is_channel_floor || is_outlet_chute_floor;

                    // Side walls - mark ALL cells outside channel as solid (not just thin strip)
                    // This prevents particles from escaping through gaps at grid edges
                    // Also extend walls into margin zone past outlet
                    let at_left_wall = k_i < (center_k - half_wid_cells);
                    let at_right_wall = k_i >= (center_k + half_wid_cells);
                    let is_side_wall_channel = (at_left_wall || at_right_wall)
                        && in_channel_length
                        && j_i <= wall_top_j
                        && j_i >= 0;

                    // Walls in the margin zone past outlet (using outlet width)
                    let at_left_wall_outlet = k_i < (center_k - outlet_half_wid_cells);
                    let at_right_wall_outlet = k_i >= (center_k + outlet_half_wid_cells);
                    let is_side_wall_outlet = (at_left_wall_outlet || at_right_wall_outlet)
                        && past_outlet
                        && j_i <= outlet_floor_j + wall_height_cells
                        && j_i >= 0;

                    let is_side_wall = is_side_wall_channel || is_side_wall_outlet;

                    // Back wall at inlet (left end, outside channel)
                    // Use inlet width for back wall
                    let at_back = i_i >= (center_i - half_len_cells - wall_thick_cells)
                        && i_i < (center_i - half_len_cells);
                    let is_back_wall = at_back
                        && j_i <= wall_top_j
                        && j_i >= 0
                        && k_i >= (center_k - inlet_half_wid_cells - wall_thick_cells)
                        && k_i < (center_k + inlet_half_wid_cells + wall_thick_cells);

                    if is_floor || is_side_wall || is_back_wall {
                        sim.grid.set_solid(i, j, k);
                    }
                }
            }
        }
    }

    /// Mark sluice solid cells using cell-index approach (like friction_sluice)
    fn mark_sluice_solid_cells(sim: &mut FlipSimulation3D, sluice: &SluicePiece, cell_size: f32) {
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;

        // Sluice position gives center in X/Z and base floor height in Y
        let center_i = (sluice.position.x / cell_size).round() as i32;
        let center_k = (sluice.position.z / cell_size).round() as i32;

        // Channel dimensions in cells
        let half_len_cells = ((sluice.length / 2.0) / cell_size).ceil() as i32;
        let half_wid_cells = ((sluice.width / 2.0) / cell_size).ceil() as i32;

        // Floor height drop due to slope
        let slope_rad = sluice.slope_deg.to_radians();
        let total_drop = sluice.length * slope_rad.tan();

        // Riffle parameters in cells
        let riffle_spacing_cells = (sluice.riffle_spacing / cell_size).round() as i32;
        let riffle_height_cells = (sluice.riffle_height / cell_size).ceil() as i32;
        let riffle_thick_cells = 2_i32;

        // Wall parameters
        let wall_height_cells = 12_i32; // Enough wall height

        // Channel bounds
        let i_start = (center_i - half_len_cells).max(0) as usize;
        let i_end = ((center_i + half_len_cells) as usize).min(width);
        let k_start = (center_k - half_wid_cells).max(0) as usize;
        let k_end = ((center_k + half_wid_cells) as usize).min(depth);

        // Inlet floor_j computed from mesh position at t=0
        let inlet_floor_j = ((sluice.position.y + total_drop / 2.0) / cell_size).floor() as i32;

        for i in 0..width {
            let i_i = i as i32;

            // Calculate t along sluice (0 = inlet, 1 = outlet)
            let t = if i_i <= center_i - half_len_cells {
                0.0
            } else if i_i >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };

            // Compute floor_j from mesh floor position to align with visual
            let mesh_floor_y = sluice.position.y + (total_drop / 2.0) - t * total_drop;
            let floor_j = (mesh_floor_y / cell_size).floor() as i32;

            // Look BOTH forward AND backward to cover staircase transitions
            let t_next = if i_i + 1 >= center_i + half_len_cells {
                1.0
            } else if i_i + 1 <= center_i - half_len_cells {
                0.0
            } else {
                ((i_i + 1 - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };
            let mesh_floor_y_next = sluice.position.y + (total_drop / 2.0) - t_next * total_drop;
            let floor_j_next = (mesh_floor_y_next / cell_size).floor() as i32;

            let t_prev = if i_i - 1 <= center_i - half_len_cells {
                0.0
            } else if i_i - 1 >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - 1 - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };
            let mesh_floor_y_prev = sluice.position.y + (total_drop / 2.0) - t_prev * total_drop;
            let floor_j_prev = (mesh_floor_y_prev / cell_size).floor() as i32;

            // Use HIGHEST of prev, current, and next to prevent staircase gaps
            let effective_floor_j = floor_j.max(floor_j_next).max(floor_j_prev);
            let wall_top_j = effective_floor_j + riffle_height_cells + wall_height_cells;

            // Check if this i position is on a riffle
            let dist_from_start = i_i - (center_i - half_len_cells);
            let is_riffle_x = if riffle_spacing_cells > 0 && dist_from_start > 4 {
                (dist_from_start % riffle_spacing_cells) < riffle_thick_cells
            } else {
                false
            };

            // Is this position before the channel inlet? (inlet chute zone)
            let before_inlet = (i as i32) < (center_i - half_len_cells);

            for k in 0..depth {
                let k_i = k as i32;
                let in_channel_width = k >= k_start && k < k_end;
                let in_channel_length = i >= i_start && i < i_end;

                // Inlet chute: extend floor from grid edge to channel inlet
                let in_inlet_chute = before_inlet
                    && k_i >= (center_k - half_wid_cells)
                    && k_i < (center_k + half_wid_cells);

                for j in 0..height {
                    let j_i = j as i32;

                    // Floor - fill ALL cells at and below effective_floor_j within channel
                    let is_channel_floor = j_i <= effective_floor_j && in_channel_length && in_channel_width;

                    // Inlet chute floor - use inlet floor height
                    let is_inlet_chute_floor = in_inlet_chute && j_i <= inlet_floor_j;

                    let is_floor = is_channel_floor || is_inlet_chute_floor;

                    // Riffles - extend above floor at regular intervals
                    let is_riffle = is_riffle_x
                        && in_channel_width
                        && in_channel_length
                        && j_i > effective_floor_j
                        && j_i <= effective_floor_j + riffle_height_cells;

                    // Side walls - mark ALL cells outside channel as solid (not just thin strip)
                    // This prevents particles from escaping through gaps at grid edges
                    // Also extend walls into inlet chute zone
                    let at_left_wall = k_i < (center_k - half_wid_cells);
                    let at_right_wall = k_i >= (center_k + half_wid_cells);
                    let is_side_wall_channel = (at_left_wall || at_right_wall)
                        && in_channel_length
                        && j_i <= wall_top_j
                        && j_i >= 0;
                    let is_side_wall_inlet = (at_left_wall || at_right_wall)
                        && before_inlet
                        && j_i <= inlet_floor_j + wall_height_cells
                        && j_i >= 0;
                    let is_side_wall = is_side_wall_channel || is_side_wall_inlet;

                    // Back wall at inlet - NO back wall since we have inlet chute now
                    // Particles enter from the inlet side

                    if is_floor || is_riffle || is_side_wall {
                        sim.grid.set_solid(i, j, k);
                    }
                }
            }
        }
    }

    /// Add a transfer between pieces
    fn add_transfer(
        &mut self,
        from_piece: usize,
        to_piece: usize,
        capture_min: Vec3,
        capture_max: Vec3,
        inject_pos: Vec3,
        inject_vel: Vec3,
    ) {
        self.transfers.push(PieceTransfer {
            from_piece,
            to_piece,
            capture_min,
            capture_max,
            inject_pos,
            inject_vel,
            transit_queue: Vec::new(),
            transit_time: 0.05,
        });
    }

    /// Emit particles into a specific piece
    fn emit_into_piece(&mut self, piece_idx: usize, world_pos: Vec3, velocity: Vec3, count: usize) {
        if piece_idx >= self.pieces.len() {
            return;
        }

        let piece = &mut self.pieces[piece_idx];
        let sim_pos = world_pos - piece.grid_offset;

        for _ in 0..count {
            // Small random spread
            let spread = 0.01;
            let offset = Vec3::new(
                (rand_float() - 0.5) * spread,
                (rand_float() - 0.5) * spread,
                (rand_float() - 0.5) * spread,
            );

            piece.positions.push(sim_pos + offset);
            piece.velocities.push(velocity);
            piece.affine_vels.push(Mat3::ZERO);
            piece.densities.push(1.0);

            // Also add to CPU sim
            piece.sim.particles.spawn(sim_pos + offset, velocity);
        }
    }

    /// Step all piece simulations
    fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        // Step each piece
        for piece in &mut self.pieces {
            if piece.positions.is_empty() {
                continue;
            }

            // Create cell types
            let (gw, gh, gd) = piece.grid_dims;
            let cell_count = gw * gh * gd;
            let mut cell_types = vec![0u32; cell_count];

            // Mark particle cells as fluid
            for pos in &piece.positions {
                let i = (pos.x / piece.cell_size).floor() as isize;
                let j = (pos.y / piece.cell_size).floor() as isize;
                let k = (pos.z / piece.cell_size).floor() as isize;
                if i >= 0
                    && (i as usize) < gw
                    && j >= 0
                    && (j as usize) < gh
                    && k >= 0
                    && (k as usize) < gd
                {
                    let idx = (k as usize) * gw * gh + (j as usize) * gw + i as usize;
                    cell_types[idx] = 1; // FLUID
                }
            }

            let sdf = Some(piece.sim.grid.sdf.clone());

            if let Some(gpu_flip) = &mut piece.gpu_flip {
                gpu_flip.step(
                    device,
                    queue,
                    &mut piece.positions,
                    &mut piece.velocities,
                    &mut piece.affine_vels,
                    &piece.densities,
                    &cell_types,
                    sdf.as_deref(),
                    None,
                    dt,
                    SIM_GRAVITY,
                    0.0,
                    SIM_PRESSURE_ITERS as u32,
                );
            }
        }

        // Process transfers - physically accurate world-space transformation
        // Pass 1: collect particles to transfer with full state
        #[derive(Clone)]
        struct TransferParticle {
            local_pos: Vec3,
            vel: Vec3,
            affine: Mat3,
            density: f32,
        }
        let mut transfer_data: Vec<Vec<TransferParticle>> =
            vec![Vec::new(); self.transfers.len()];

        // Also collect grid offsets for coordinate transformation
        let mut transfer_offsets: Vec<(Vec3, Vec3)> = Vec::new();
        for transfer in self.transfers.iter() {
            let from_offset = self.pieces[transfer.from_piece].grid_offset;
            let to_offset = self.pieces[transfer.to_piece].grid_offset;
            transfer_offsets.push((from_offset, to_offset));
        }

        for (tidx, transfer) in self.transfers.iter().enumerate() {
            let from_piece = &self.pieces[transfer.from_piece];

            // Debug: track max X position of particles in this piece
            let mut max_x = f32::NEG_INFINITY;
            let mut count_near_outlet = 0;

            for i in 0..from_piece.positions.len() {
                let pos = from_piece.positions[i];
                if pos.x > max_x {
                    max_x = pos.x;
                }
                // Count particles near the capture X range
                if pos.x >= transfer.capture_min.x - 0.1 {
                    count_near_outlet += 1;
                }

                if pos.x >= transfer.capture_min.x
                    && pos.x <= transfer.capture_max.x
                    && pos.y >= transfer.capture_min.y
                    && pos.y <= transfer.capture_max.y
                    && pos.z >= transfer.capture_min.z
                    && pos.z <= transfer.capture_max.z
                {
                    transfer_data[tidx].push(TransferParticle {
                        local_pos: pos,
                        vel: from_piece.velocities[i],
                        affine: from_piece.affine_vels[i],
                        density: from_piece.densities[i],
                    });
                }
            }

            // Print debug info every 60 frames (more frequent)
            if self.frame % 60 == 0 && !from_piece.positions.is_empty() {
                // Also track Y range and particle distribution
                let mut min_y = f32::INFINITY;
                let mut avg_x = 0.0;
                let mut min_x = f32::INFINITY;
                for pos in &from_piece.positions {
                    if pos.y < min_y {
                        min_y = pos.y;
                    }
                    if pos.x < min_x {
                        min_x = pos.x;
                    }
                    avg_x += pos.x;
                }
                avg_x /= from_piece.positions.len() as f32;

                println!(
                    "  Gutter: x=[{:.3},{:.3}], avg={:.3}, min_y={:.3}, cap_x=[{:.3},{:.3}], near={}, cap={}",
                    min_x,
                    max_x,
                    avg_x,
                    min_y,
                    transfer.capture_min.x,
                    transfer.capture_max.x,
                    count_near_outlet,
                    transfer_data[tidx].len()
                );
            }
        }

        // Pass 2: remove captured particles from source
        for (tidx, transfer) in self.transfers.iter().enumerate() {
            if transfer_data[tidx].is_empty() {
                continue;
            }

            let from_piece = &mut self.pieces[transfer.from_piece];
            let mut i = 0;
            while i < from_piece.positions.len() {
                let pos = from_piece.positions[i];
                if pos.x >= transfer.capture_min.x
                    && pos.x <= transfer.capture_max.x
                    && pos.y >= transfer.capture_min.y
                    && pos.y <= transfer.capture_max.y
                    && pos.z >= transfer.capture_min.z
                    && pos.z <= transfer.capture_max.z
                {
                    from_piece.positions.swap_remove(i);
                    from_piece.velocities.swap_remove(i);
                    from_piece.affine_vels.swap_remove(i);
                    from_piece.densities.swap_remove(i);
                } else {
                    i += 1;
                }
            }
        }

        // Pass 3: inject into target with world-space coordinate transformation
        const GRAVITY: f32 = 9.81;

        for (tidx, transfer) in self.transfers.iter().enumerate() {
            if transfer_data[tidx].is_empty() {
                continue;
            }

            let (from_offset, to_offset) = transfer_offsets[tidx];
            let to_piece = &mut self.pieces[transfer.to_piece];

            for particle in &transfer_data[tidx] {
                // Convert local position to world space
                let world_pos = particle.local_pos + from_offset;

                // Convert world position to target's local space
                let target_local_pos = world_pos - to_offset;

                // Calculate height drop for gravity acceleration
                let height_drop = from_offset.y - to_offset.y;
                let mut new_vel = particle.vel;

                // Add gravity acceleration for height drop (v^2 = v0^2 + 2*g*h)
                if height_drop > 0.0 {
                    // Falling: add downward velocity from gravity
                    let gravity_vel = (2.0 * GRAVITY * height_drop).sqrt();
                    new_vel.y -= gravity_vel;
                }

                // Only clamp if significantly outside grid - allow particles at edges
                let grid_dims = to_piece.grid_dims;
                let cell_size = to_piece.cell_size;
                let min_bound = cell_size; // Allow near edge
                let max_x = (grid_dims.0 as f32) * cell_size - cell_size;
                let max_y = (grid_dims.1 as f32) * cell_size - cell_size;
                let max_z = (grid_dims.2 as f32) * cell_size - cell_size;
                let clamped_pos = Vec3::new(
                    target_local_pos.x.clamp(min_bound, max_x),
                    target_local_pos.y.clamp(min_bound, max_y),
                    target_local_pos.z.clamp(min_bound, max_z),
                );

                to_piece.positions.push(clamped_pos);
                to_piece.velocities.push(new_vel);
                to_piece.affine_vels.push(particle.affine); // Preserve affine velocity!
                to_piece.densities.push(particle.density);
                to_piece.sim.particles.spawn(clamped_pos, new_vel);
            }

            if !transfer_data[tidx].is_empty() {
                static TRANSFER_PRINTED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !TRANSFER_PRINTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    println!(
                        "Transfer: {} particles from piece {} to piece {} (world-space)",
                        transfer_data[tidx].len(),
                        transfer.from_piece,
                        transfer.to_piece
                    );
                }
            }
        }

        // Pass 4: Remove particles that have exited far past grid boundaries
        // These are particles that flowed out of open boundaries and weren't captured by any transfer
        let exit_margin = SIM_CELL_SIZE * 20.0; // Allow some distance before removal
        for piece in &mut self.pieces {
            let (gw, gh, gd) = piece.grid_dims;
            let max_x = (gw as f32) * piece.cell_size + exit_margin;
            let max_y = (gh as f32) * piece.cell_size + exit_margin;
            let max_z = (gd as f32) * piece.cell_size + exit_margin;
            let min_bound = -exit_margin;

            let mut i = 0;
            while i < piece.positions.len() {
                let pos = piece.positions[i];
                let out_of_bounds = pos.x < min_bound
                    || pos.x > max_x
                    || pos.y < min_bound
                    || pos.y > max_y
                    || pos.z < min_bound
                    || pos.z > max_z;

                if out_of_bounds {
                    piece.positions.swap_remove(i);
                    piece.velocities.swap_remove(i);
                    piece.affine_vels.swap_remove(i);
                    piece.densities.swap_remove(i);
                } else {
                    i += 1;
                }
            }
        }

        self.frame += 1;
    }

    /// Get total particle count across all pieces
    fn total_particles(&self) -> usize {
        self.pieces.iter().map(|p| p.positions.len()).sum()
    }
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    layout: EditorLayout,

    // Editor state
    mode: EditorMode,
    selection: Selection,
    preview_gutter: GutterPiece,
    preview_sluice: SluicePiece,
    preview_emitter: EmitterPiece,

    // Camera state
    camera_yaw: f32,
    camera_pitch: f32,
    camera_distance: f32,
    camera_target: Vec3,

    // Input state
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    shift_pressed: bool,
    snap_enabled: bool, // Enable snapping to other pieces

    // Simulation state (Play mode)
    is_simulating: bool,
    multi_sim: Option<MultiGridSim>,
    fluid_renderer: Option<ScreenSpaceFluidRenderer>,

    // Legacy single-grid fields (kept for compatibility during transition)
    sim: Option<FlipSimulation3D>,
    gpu_flip: Option<GpuFlip3D>,
    sim_grid_offset: Vec3,
    sim_grid_dims: (usize, usize, usize),
    sim_frame: u32,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    affine_vels: Vec<Mat3>,
    densities: Vec<f32>,
}

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    // Mesh rendering
    mesh_pipeline: wgpu::RenderPipeline,

    // Shared
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    // Depth buffer
    depth_texture: wgpu::TextureView,
}

impl App {
    fn new() -> Self {
        // Try to auto-load layout from file, fall back to pre-connected layout
        let layout = match EditorLayout::load_json(Path::new("editor_layout.json")) {
            Ok(loaded) => {
                println!("Loaded from editor_layout.json");
                loaded
            }
            Err(_) => {
                println!("No saved layout - using pre-connected gutter+sluice");
                EditorLayout::new_connected()
            }
        };

        println!("=== Washplant Editor ===");
        println!();
        println!("Controls:");
        println!("  WASD / QE      - Move camera");
        println!("  Mouse drag     - Rotate camera");
        println!("  Scroll         - Zoom in/out");
        println!("  Arrow keys     - Move selected (XZ plane)");
        println!("  Shift+Up/Down  - Move Y axis");
        println!("  R / Shift+R    - Rotate CW/CCW");
        println!("  [ / ]          - Adjust angle");
        println!("  - / =          - Adjust length/velocity");
        println!("  , / .          - Adjust start width");
        println!("  Shift+, / .    - Adjust end width (gutter funnel)");
        println!("  1              - Place gutter");
        println!("  2              - Place sluice");
        println!("  3              - Place emitter");
        println!("  Enter          - Confirm placement");
        println!("  Escape         - Cancel / Deselect");
        println!("  Delete         - Remove selected");
        println!("  Shift+S        - Save layout");
        println!("  L              - Load layout");
        println!("  P              - Play/Stop simulation");
        println!("  N              - Toggle snap (outlet→inlet)");

        // Print initial status
        println!(
            "[SELECT] Selection: none | Gutters: {} | Sluices: {} | Emitters: {}",
            layout.gutters.len(),
            layout.sluices.len(),
            layout.emitters.len()
        );

        Self {
            window: None,
            gpu: None,
            layout,
            mode: EditorMode::Select,
            selection: Selection::None,
            preview_gutter: GutterPiece::default(),
            preview_sluice: SluicePiece::default(),
            preview_emitter: EmitterPiece::default(),
            camera_yaw: 0.5,
            camera_pitch: 0.6,
            camera_distance: 5.0,
            camera_target: Vec3::new(0.0, 0.5, 0.0),
            mouse_pressed: false,
            last_mouse_pos: None,
            shift_pressed: false,
            snap_enabled: true,
            // Simulation state
            is_simulating: false,
            multi_sim: None,
            fluid_renderer: None,
            // Legacy single-grid fields
            sim: None,
            gpu_flip: None,
            sim_grid_offset: Vec3::ZERO,
            sim_grid_dims: (0, 0, 0),
            sim_frame: 0,
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
        }
    }

    fn camera_position(&self) -> Vec3 {
        self.camera_target
            + Vec3::new(
                self.camera_distance * self.camera_yaw.cos() * self.camera_pitch.cos(),
                self.camera_distance * self.camera_pitch.sin(),
                self.camera_distance * self.camera_yaw.sin() * self.camera_pitch.cos(),
            )
    }

    fn print_status(&self) {
        let mode_str = match self.mode {
            EditorMode::Select => "SELECT",
            EditorMode::PlaceGutter => "PLACE GUTTER",
            EditorMode::PlaceSluice => "PLACE SLUICE",
            EditorMode::PlaceEmitter => "PLACE EMITTER",
        };
        let sel_str = match self.selection {
            Selection::None => "none".to_string(),
            Selection::Gutter(i) => format!("Gutter #{}", i),
            Selection::Sluice(i) => format!("Sluice #{}", i),
            Selection::Emitter(i) => format!("Emitter #{}", i),
        };
        println!(
            "[{}] Selection: {} | Gutters: {} | Sluices: {} | Emitters: {}",
            mode_str,
            sel_str,
            self.layout.gutters.len(),
            self.layout.sluices.len(),
            self.layout.emitters.len()
        );
    }

    /// Start the fluid simulation (multi-grid approach)
    fn start_simulation(&mut self) {
        if self.layout.emitters.is_empty() {
            println!("Cannot start simulation: no emitters placed!");
            return;
        }

        let Some(gpu) = &self.gpu else {
            println!("Cannot start simulation: no GPU state!");
            return;
        };

        // Create multi-grid simulation
        let mut multi_sim = MultiGridSim::new();

        println!("=== Starting Multi-Grid Simulation ===");

        // Add a piece simulation for each gutter
        for (idx, gutter) in self.layout.gutters.iter().enumerate() {
            let piece_idx = multi_sim.add_gutter(&gpu.device, gutter, idx);
            let piece = &multi_sim.pieces[piece_idx];
            println!(
                "  Gutter #{}: grid {}x{}x{} at {:?}",
                idx, piece.grid_dims.0, piece.grid_dims.1, piece.grid_dims.2, piece.grid_offset
            );
        }

        // Add a piece simulation for each sluice
        for (idx, sluice) in self.layout.sluices.iter().enumerate() {
            let piece_idx = multi_sim.add_sluice(&gpu.device, sluice, idx);
            let piece = &multi_sim.pieces[piece_idx];
            println!(
                "  Sluice #{}: grid {}x{}x{} at {:?}",
                idx, piece.grid_dims.0, piece.grid_dims.1, piece.grid_dims.2, piece.grid_offset
            );
        }

        println!("  Total pieces: {}", multi_sim.pieces.len());

        // Create transfers from gutters to sluices
        // For now, connect first gutter to first sluice if both exist
        let num_gutters = self.layout.gutters.len();
        let num_sluices = self.layout.sluices.len();

        if num_gutters > 0 && num_sluices > 0 {
            let gutter_idx = 0;
            let sluice_idx = num_gutters; // Sluices start after gutters in pieces array

            let gutter = &self.layout.gutters[0];
            let sluice = &self.layout.sluices[0];

            // Capture region: generous area at outlet end of gutter
            // Captures any particle that reaches the outlet region - prevents particles from falling through gaps
            let margin = SIM_CELL_SIZE * 4.0;
            let gutter_local_center_x = margin + gutter.length / 2.0;
            let outlet_x = gutter_local_center_x + gutter.length / 2.0;
            // Capture is a generous region: last 20% of gutter length + some overshoot
            let capture_depth = gutter.length * 0.2 + SIM_CELL_SIZE * 4.0;
            let capture_x_min = outlet_x - capture_depth;
            let capture_x_max = outlet_x + SIM_CELL_SIZE * 10.0; // Extend past outlet to catch fast particles
            // Y: full range - capture at any height (particles may have fallen)
            let capture_y_min = -1.0; // Allow negative Y for fallen particles
            let capture_y_max = 10.0; // Generous - capture everything
            // Z: full grid width to catch any lateral drift
            let center_z = margin + gutter.max_width() / 2.0;
            let half_width = gutter.max_width() / 2.0 + SIM_CELL_SIZE * 4.0;
            let capture_z_min = center_z - half_width;
            let capture_z_max = center_z + half_width;

            println!("    Capture region (gutter local): X=[{:.3}, {:.3}], Y=[{:.3}, {:.3}], Z=[{:.3}, {:.3}]",
                capture_x_min, capture_x_max, capture_y_min, capture_y_max, capture_z_min, capture_z_max);

            // Injection position: at inlet of sluice (in sluice's local grid space)
            let _sluice_piece = &multi_sim.pieces[sluice_idx];
            let sluice_margin = SIM_CELL_SIZE * 4.0;

            // Sluice inlet is at the high end (start) of the sluice
            // In local space: margin + length/2 - length/2 = margin (start of sluice)
            // Plus account for the slope - inlet is higher
            let sluice_half_drop = sluice.height_drop() / 2.0;
            let inject_x = sluice_margin; // At the inlet (start) of sluice
            let inject_y = sluice_margin + sluice_half_drop + 0.05; // Just above floor at inlet
            let inject_z = sluice_margin + sluice.width / 2.0; // Centered
            let inject_pos = Vec3::new(inject_x, inject_y, inject_z);

            // Injection velocity: along sluice slope direction
            let inject_vel = Vec3::new(0.5, -0.2, 0.0); // Forward along sluice

            multi_sim.add_transfer(
                gutter_idx,
                sluice_idx,
                Vec3::new(capture_x_min, capture_y_min, capture_z_min),
                Vec3::new(capture_x_max, capture_y_max, capture_z_max),
                inject_pos,
                inject_vel,
            );

            println!("  Created transfer: Gutter #{} -> Sluice #{}", 0, 0);

            // Debug: show grid positions for checking overlap
            let gutter_piece = &multi_sim.pieces[gutter_idx];
            let sluice_piece = &multi_sim.pieces[sluice_idx];
            let gutter_outlet_world = gutter.outlet_position();
            let sluice_inlet_world = sluice.inlet_position();
            println!("    Gutter outlet (world): {:?}", gutter_outlet_world);
            println!("    Sluice inlet (world): {:?}", sluice_inlet_world);
            println!("    Gutter grid offset: {:?}", gutter_piece.grid_offset);
            println!("    Sluice grid offset: {:?}", sluice_piece.grid_offset);

            // Check where gutter outlet maps to in sluice local space
            let outlet_in_sluice_local = gutter_outlet_world - sluice_piece.grid_offset;
            println!("    Gutter outlet in sluice local: {:?}", outlet_in_sluice_local);
            println!("    Sluice grid size: {:?}", sluice_piece.grid_dims);
        }

        // Create fluid renderer
        let mut fluid_renderer = ScreenSpaceFluidRenderer::new(&gpu.device, gpu.config.format);
        fluid_renderer.particle_radius = SIM_CELL_SIZE * 0.5;
        fluid_renderer.resize(&gpu.device, gpu.config.width, gpu.config.height);

        self.multi_sim = Some(multi_sim);
        self.fluid_renderer = Some(fluid_renderer);
        self.sim_frame = 0;
        self.is_simulating = true;

        println!("Multi-grid simulation started! Press P to stop.");
    }

    /// Stop the simulation
    fn stop_simulation(&mut self) {
        self.is_simulating = false;
        self.sim = None;
        self.gpu_flip = None;
        self.fluid_renderer = None;
        self.positions.clear();
        self.velocities.clear();
        self.affine_vels.clear();
        self.densities.clear();
        println!("Simulation stopped.");
    }

    /// Mark solid cells in the simulation grid from editor geometry
    fn mark_geometry_solids(&self, sim: &mut FlipSimulation3D) -> usize {
        let mut count = 0;
        let cell_size = SIM_CELL_SIZE;
        let offset = self.sim_grid_offset;

        // Mark gutter solids
        for gutter in &self.layout.gutters {
            count += self.mark_gutter_cells(sim, gutter, cell_size, offset);
        }

        // Mark sluice solids
        for sluice in &self.layout.sluices {
            count += self.mark_sluice_cells(sim, sluice, cell_size, offset);
        }

        // Mark floor (y=0 plane as solid)
        let grid_width = sim.grid.width;
        let grid_depth = sim.grid.depth;
        for i in 0..grid_width {
            for k in 0..grid_depth {
                let world_y = offset.y + 0.5 * cell_size;
                if world_y < 0.0 {
                    sim.grid.set_solid(i, 0, k);
                    count += 1;
                }
            }
        }

        count
    }

    /// Mark cells for a gutter piece
    fn mark_gutter_cells(
        &self,
        sim: &mut FlipSimulation3D,
        gutter: &GutterPiece,
        cell_size: f32,
        offset: Vec3,
    ) -> usize {
        let mut count = 0;
        let pos = gutter.position;

        // Gutter is a U-channel: floor + two side walls
        // Apply rotation to the gutter orientation
        let (dir_x, dir_z) = match gutter.rotation {
            Rotation::R0 => (1.0f32, 0.0f32),
            Rotation::R90 => (0.0, 1.0),
            Rotation::R180 => (-1.0, 0.0),
            Rotation::R270 => (0.0, -1.0),
        };

        let half_len = gutter.length / 2.0;
        let half_wid = gutter.width / 2.0;
        let wall_thick = 0.02; // 2cm wall/floor thickness
        let angle_rad = gutter.angle_deg.to_radians();

        // Iterate over cells in the gutter region
        let steps_len = (gutter.length / cell_size).ceil() as i32;
        let steps_wid = (gutter.width / cell_size).ceil() as i32 + 4;
        let steps_h = (gutter.wall_height / cell_size).ceil() as i32 + 2;

        for li in -2..=steps_len + 2 {
            for wi in -2..=steps_wid {
                for hi in 0..=steps_h {
                    // Local coordinates
                    let local_l = (li as f32 - steps_len as f32 / 2.0) * cell_size;
                    let local_w = (wi as f32 - steps_wid as f32 / 2.0) * cell_size;
                    let local_h = hi as f32 * cell_size;

                    // Check if this is floor or wall
                    let is_floor = local_h < wall_thick
                        && local_l.abs() <= half_len
                        && local_w.abs() <= half_wid;
                    let is_left_wall = local_w < -half_wid + wall_thick
                        && local_w > -half_wid - wall_thick
                        && local_h < gutter.wall_height;
                    let is_right_wall = local_w > half_wid - wall_thick
                        && local_w < half_wid + wall_thick
                        && local_h < gutter.wall_height;

                    if is_floor || is_left_wall || is_right_wall {
                        // Apply angle to height (tilted floor)
                        let height_offset = local_l * angle_rad.tan();

                        // Transform to world coordinates
                        let world_x = pos.x + local_l * dir_x - local_w * dir_z;
                        let world_y = pos.y + local_h + height_offset;
                        let world_z = pos.z + local_l * dir_z + local_w * dir_x;

                        // Convert to grid indices
                        let gi = ((world_x - offset.x) / cell_size).floor() as i32;
                        let gj = ((world_y - offset.y) / cell_size).floor() as i32;
                        let gk = ((world_z - offset.z) / cell_size).floor() as i32;

                        if gi >= 0
                            && gi < sim.grid.width as i32
                            && gj >= 0
                            && gj < sim.grid.height as i32
                            && gk >= 0
                            && gk < sim.grid.depth as i32
                        {
                            sim.grid.set_solid(gi as usize, gj as usize, gk as usize);
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// Mark cells for a sluice piece
    fn mark_sluice_cells(
        &self,
        sim: &mut FlipSimulation3D,
        sluice: &SluicePiece,
        cell_size: f32,
        offset: Vec3,
    ) -> usize {
        let mut count = 0;
        let pos = sluice.position;

        // Sluice is a sloped channel with riffles
        let (dir_x, dir_z) = match sluice.rotation {
            Rotation::R0 => (1.0f32, 0.0f32),
            Rotation::R90 => (0.0, 1.0),
            Rotation::R180 => (-1.0, 0.0),
            Rotation::R270 => (0.0, -1.0),
        };

        let half_len = sluice.length / 2.0;
        let half_wid = sluice.width / 2.0;
        let slope_rad = sluice.slope_deg.to_radians();

        let steps_len = (sluice.length / cell_size).ceil() as i32;
        let steps_wid = (sluice.width / cell_size).ceil() as i32 + 4;

        for li in -2..=steps_len + 2 {
            for wi in -2..=steps_wid {
                let local_l = (li as f32 - steps_len as f32 / 2.0) * cell_size;
                let local_w = (wi as f32 - steps_wid as f32 / 2.0) * cell_size;

                // Floor height varies along length (slope)
                let floor_h = (-local_l + half_len) * slope_rad.tan();

                // Check for floor
                let is_floor = local_l.abs() <= half_len && local_w.abs() <= half_wid;

                // Check for riffles (perpendicular bars)
                let riffle_pos = (local_l + half_len) % sluice.riffle_spacing;
                let is_riffle = is_floor && riffle_pos < sluice.riffle_spacing * 0.15;

                // Check for side walls
                let is_wall = local_w.abs() > half_wid - 0.02 && local_w.abs() < half_wid + 0.02;

                if is_floor || is_wall {
                    let wall_h = if is_wall { 0.15 } else { 0.0 }; // Wall height
                    let riffle_h = if is_riffle {
                        sluice.riffle_spacing * 0.3
                    } else {
                        0.0
                    };

                    for hi in 0..=((riffle_h + wall_h) / cell_size).ceil() as i32 + 1 {
                        let local_h = hi as f32 * cell_size;

                        let world_x = pos.x + local_l * dir_x - local_w * dir_z;
                        let world_y = pos.y + floor_h + local_h;
                        let world_z = pos.z + local_l * dir_z + local_w * dir_x;

                        let gi = ((world_x - offset.x) / cell_size).floor() as i32;
                        let gj = ((world_y - offset.y) / cell_size).floor() as i32;
                        let gk = ((world_z - offset.z) / cell_size).floor() as i32;

                        if gi >= 0
                            && gi < sim.grid.width as i32
                            && gj >= 0
                            && gj < sim.grid.height as i32
                            && gk >= 0
                            && gk < sim.grid.depth as i32
                        {
                            sim.grid.set_solid(gi as usize, gj as usize, gk as usize);
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }

    /// Emit particles from all emitters
    fn emit_from_emitters(&mut self) {
        let Some(sim) = &mut self.sim else { return };
        let grid_offset = self.sim_grid_offset;

        for emitter in &self.layout.emitters {
            // Calculate emit count based on rate (particles per second at 60 FPS)
            let emit_count = (emitter.rate / 60.0).ceil() as usize;

            // Get emitter direction based on rotation (points in +X by default)
            let base_dir = match emitter.rotation {
                Rotation::R0 => Vec3::new(1.0, 0.0, 0.0),
                Rotation::R90 => Vec3::new(0.0, 0.0, 1.0),
                Rotation::R180 => Vec3::new(-1.0, 0.0, 0.0),
                Rotation::R270 => Vec3::new(0.0, 0.0, -1.0),
            };

            let spread_rad = emitter.spread_deg.to_radians();

            for _ in 0..emit_count {
                // Spray bar: spread along perpendicular axis based on width
                let perp = Vec3::new(-base_dir.z, 0.0, base_dir.x);
                let width_offset = (rand_float() - 0.5) * emitter.width;
                // Small random offset in emit direction and Y for natural spray
                let depth_offset = rand_float() * emitter.radius;
                let y_offset = (rand_float() - 0.5) * emitter.radius * 0.5;
                let pos_offset = perp * width_offset + base_dir * depth_offset + Vec3::Y * y_offset;

                // Random spread angles (yaw and pitch)
                let spread_yaw = (rand_float() - 0.5) * spread_rad;
                let spread_pitch = (rand_float() - 0.5) * spread_rad * 0.5;

                // Apply spread to base direction
                let cy = spread_yaw.cos();
                let sy = spread_yaw.sin();
                let cp = spread_pitch.cos();
                let sp = spread_pitch.sin();

                // Rotate base_dir by yaw (around Y) then add pitch
                let dir_after_yaw = Vec3::new(
                    base_dir.x * cy - base_dir.z * sy,
                    base_dir.y,
                    base_dir.x * sy + base_dir.z * cy,
                );
                let velocity = Vec3::new(
                    dir_after_yaw.x * cp,
                    dir_after_yaw.y + sp,
                    dir_after_yaw.z * cp,
                )
                .normalize()
                    * emitter.velocity;

                // Convert world position to simulation grid position
                let world_pos = emitter.position + pos_offset;
                let sim_pos = world_pos - grid_offset;

                // Check grid bounds
                let grid_max = Vec3::new(
                    sim.grid.width as f32 * SIM_CELL_SIZE,
                    sim.grid.height as f32 * SIM_CELL_SIZE,
                    sim.grid.depth as f32 * SIM_CELL_SIZE,
                );

                if sim_pos.x > 0.0
                    && sim_pos.y > 0.0
                    && sim_pos.z > 0.0
                    && sim_pos.x < grid_max.x
                    && sim_pos.y < grid_max.y
                    && sim_pos.z < grid_max.z
                {
                    sim.spawn_particle_with_velocity(sim_pos, velocity);
                }
            }
        }

        // Debug: print spawn info once
        static PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !PRINTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            println!("Emitter debug:");
            println!("  Grid offset: {:?}", grid_offset);
            for (i, emitter) in self.layout.emitters.iter().enumerate() {
                let sim_pos = emitter.position - grid_offset;
                println!(
                    "  Emitter {}: world={:?} sim={:?}",
                    i, emitter.position, sim_pos
                );
            }
        }
    }

    /// Emit particles from emitters into multi-grid simulation
    fn emit_from_emitters_multi(&mut self) {
        let Some(multi_sim) = &mut self.multi_sim else {
            return;
        };

        // For now, emit into the first gutter (piece index 0 if it exists)
        // Later: find which piece the emitter is above/near
        let target_piece = if multi_sim.pieces.is_empty() {
            return;
        } else {
            0
        };

        for emitter in &self.layout.emitters {
            // Calculate emit count based on rate (particles per second at 60 FPS)
            let emit_count = (emitter.rate / 60.0).ceil() as usize;

            // Get emitter direction based on rotation
            let base_dir = match emitter.rotation {
                Rotation::R0 => Vec3::new(1.0, 0.0, 0.0),
                Rotation::R90 => Vec3::new(0.0, 0.0, 1.0),
                Rotation::R180 => Vec3::new(-1.0, 0.0, 0.0),
                Rotation::R270 => Vec3::new(0.0, 0.0, -1.0),
            };

            let spread_rad = emitter.spread_deg.to_radians();

            for _ in 0..emit_count {
                // Spray bar: spread along perpendicular axis based on width
                let perp = Vec3::new(-base_dir.z, 0.0, base_dir.x);
                let width_offset = (rand_float() - 0.5) * emitter.width;
                // Small random offset in emit direction and Y for natural spray
                let depth_offset = rand_float() * emitter.radius;
                let y_offset = (rand_float() - 0.5) * emitter.radius * 0.5;
                let pos_offset = perp * width_offset + base_dir * depth_offset + Vec3::Y * y_offset;

                // Random spread
                let spread_yaw = (rand_float() - 0.5) * spread_rad;
                let spread_pitch = (rand_float() - 0.5) * spread_rad * 0.5;
                let cy = spread_yaw.cos();
                let sy = spread_yaw.sin();
                let cp = spread_pitch.cos();
                let sp = spread_pitch.sin();
                let dir_after_yaw = Vec3::new(
                    base_dir.x * cy - base_dir.z * sy,
                    base_dir.y,
                    base_dir.x * sy + base_dir.z * cy,
                );
                let velocity = Vec3::new(
                    dir_after_yaw.x * cp,
                    dir_after_yaw.y + sp,
                    dir_after_yaw.z * cp,
                )
                .normalize()
                    * emitter.velocity;

                let world_pos = emitter.position + pos_offset;

                // Emit into target piece
                multi_sim.emit_into_piece(target_piece, world_pos, velocity, 1);
            }
        }

        // Debug once
        static PRINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !PRINTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            println!("Multi-grid emitter debug:");
            if let Some(piece) = multi_sim.pieces.first() {
                println!("  Target piece grid offset: {:?}", piece.grid_offset);
            }
            for (i, emitter) in self.layout.emitters.iter().enumerate() {
                println!("  Emitter {}: world={:?}", i, emitter.position);
            }
        }
    }

    /// Prepare particle data for GPU transfer
    fn prepare_gpu_inputs(&mut self) {
        let Some(sim) = &self.sim else { return };

        let particle_count = sim.particles.list.len();
        self.positions.clear();
        self.positions.reserve(particle_count);
        self.velocities.clear();
        self.velocities.reserve(particle_count);
        self.affine_vels.clear();
        self.affine_vels.reserve(particle_count);
        self.densities.clear();
        self.densities.reserve(particle_count);

        for p in &sim.particles.list {
            self.positions.push(p.position);
            self.velocities.push(p.velocity);
            self.affine_vels.push(p.affine_velocity);
            self.densities.push(p.density);
        }
    }

    fn handle_key(&mut self, key: KeyCode) {
        match key {
            // Mode keys
            KeyCode::Digit1 => {
                self.mode = EditorMode::PlaceGutter;
                self.selection = Selection::None;
                println!("Mode: PLACE GUTTER (arrows to position, Enter to place)");
            }
            KeyCode::Digit2 => {
                self.mode = EditorMode::PlaceSluice;
                self.selection = Selection::None;
                println!("Mode: PLACE SLUICE (arrows to position, Enter to place)");
            }
            KeyCode::Digit3 => {
                self.mode = EditorMode::PlaceEmitter;
                self.selection = Selection::None;
                println!("Mode: PLACE EMITTER (arrows to position, R to rotate, Enter to place)");
            }
            KeyCode::Escape => {
                if self.mode != EditorMode::Select {
                    self.mode = EditorMode::Select;
                    println!("Mode: SELECT");
                } else {
                    self.selection = Selection::None;
                    println!("Deselected");
                }
            }
            KeyCode::Enter => {
                match self.mode {
                    EditorMode::PlaceGutter => {
                        let mut gutter = self.preview_gutter.clone();
                        gutter.id = self.layout.next_id();
                        let pos = gutter.position;
                        self.layout.gutters.push(gutter);
                        let idx = self.layout.gutters.len() - 1;
                        self.selection = Selection::Gutter(idx);
                        self.mode = EditorMode::Select;
                        println!("Placed Gutter #{} at {:?}", idx, pos);
                        // Reset preview for next placement
                        self.preview_gutter = GutterPiece::default();
                    }
                    EditorMode::PlaceSluice => {
                        let mut sluice = self.preview_sluice.clone();
                        sluice.id = self.layout.next_id();
                        let pos = sluice.position;
                        self.layout.sluices.push(sluice);
                        let idx = self.layout.sluices.len() - 1;
                        self.selection = Selection::Sluice(idx);
                        self.mode = EditorMode::Select;
                        println!("Placed Sluice #{} at {:?}", idx, pos);
                        // Reset preview for next placement
                        self.preview_sluice = SluicePiece::default();
                    }
                    EditorMode::PlaceEmitter => {
                        let mut emitter = self.preview_emitter.clone();
                        emitter.id = self.layout.next_id();
                        let pos = emitter.position;
                        self.layout.emitters.push(emitter);
                        let idx = self.layout.emitters.len() - 1;
                        self.selection = Selection::Emitter(idx);
                        self.mode = EditorMode::Select;
                        println!("Placed Emitter #{} at {:?}", idx, pos);
                        // Reset preview for next placement
                        self.preview_emitter = EmitterPiece::default();
                    }
                    EditorMode::Select => {}
                }
                self.print_status();
            }
            KeyCode::Delete | KeyCode::Backspace => {
                match self.selection {
                    Selection::Gutter(idx) => {
                        self.layout.remove_gutter(idx);
                        self.selection = Selection::None;
                        println!("Removed gutter");
                    }
                    Selection::Sluice(idx) => {
                        self.layout.remove_sluice(idx);
                        self.selection = Selection::None;
                        println!("Removed sluice");
                    }
                    Selection::Emitter(idx) => {
                        self.layout.remove_emitter(idx);
                        self.selection = Selection::None;
                        println!("Removed emitter");
                    }
                    Selection::None => {}
                }
                self.print_status();
            }

            // Movement keys
            KeyCode::ArrowUp => {
                let delta = if self.shift_pressed {
                    Vec3::new(0.0, MOVE_STEP, 0.0)
                } else {
                    Vec3::new(0.0, 0.0, -MOVE_STEP)
                };
                self.move_selected_or_preview(delta);
            }
            KeyCode::ArrowDown => {
                let delta = if self.shift_pressed {
                    Vec3::new(0.0, -MOVE_STEP, 0.0)
                } else {
                    Vec3::new(0.0, 0.0, MOVE_STEP)
                };
                self.move_selected_or_preview(delta);
            }
            KeyCode::ArrowLeft => {
                self.move_selected_or_preview(Vec3::new(-MOVE_STEP, 0.0, 0.0));
            }
            KeyCode::ArrowRight => {
                self.move_selected_or_preview(Vec3::new(MOVE_STEP, 0.0, 0.0));
            }

            // Rotation keys
            KeyCode::KeyR => {
                if self.shift_pressed {
                    self.rotate_selected_ccw();
                } else {
                    self.rotate_selected_cw();
                }
            }

            // Adjustment keys
            KeyCode::BracketLeft => {
                self.adjust_selected_angle(-ANGLE_STEP);
            }
            KeyCode::BracketRight => {
                self.adjust_selected_angle(ANGLE_STEP);
            }
            KeyCode::Minus => {
                self.adjust_selected_length(-LENGTH_STEP);
            }
            KeyCode::Equal => {
                self.adjust_selected_length(LENGTH_STEP);
            }
            KeyCode::Comma if self.shift_pressed => {
                self.adjust_selected_end_width(-WIDTH_STEP);
            }
            KeyCode::Comma => {
                self.adjust_selected_width(-WIDTH_STEP);
            }
            KeyCode::Period if self.shift_pressed => {
                self.adjust_selected_end_width(WIDTH_STEP);
            }
            KeyCode::Period => {
                self.adjust_selected_width(WIDTH_STEP);
            }

            // Save/Load (Ctrl+S or just hold shift and press S)
            KeyCode::KeyS if self.shift_pressed => {
                let path = Path::new("editor_layout.json");
                match self.layout.save_json(path) {
                    Ok(_) => println!("Saved to {}", path.display()),
                    Err(e) => eprintln!("Save failed: {}", e),
                }
            }
            KeyCode::KeyL => {
                let path = Path::new("editor_layout.json");
                match EditorLayout::load_json(path) {
                    Ok(layout) => {
                        self.layout = layout;
                        self.selection = Selection::None;
                        println!("Loaded from {}", path.display());
                        self.print_status();
                    }
                    Err(e) => eprintln!("Load failed: {}", e),
                }
            }

            // Play/Stop simulation
            KeyCode::KeyP => {
                if self.is_simulating {
                    self.stop_simulation();
                } else {
                    self.start_simulation();
                }
            }

            // Click simulation for selection (Tab cycles through pieces)
            KeyCode::Tab => {
                self.cycle_selection();
            }

            // WASD camera movement
            KeyCode::KeyW => {
                // Move camera target forward (in camera's look direction on XZ plane)
                let forward = Vec3::new(-self.camera_yaw.sin(), 0.0, -self.camera_yaw.cos());
                self.camera_target += forward * MOVE_STEP * 2.0;
            }
            KeyCode::KeyS if !self.shift_pressed => {
                // Move camera target backward (opposite of forward) - but not when shift is pressed (that's save)
                let forward = Vec3::new(-self.camera_yaw.sin(), 0.0, -self.camera_yaw.cos());
                self.camera_target -= forward * MOVE_STEP * 2.0;
            }
            KeyCode::KeyA => {
                // Move camera target left
                let right = Vec3::new(self.camera_yaw.cos(), 0.0, -self.camera_yaw.sin());
                self.camera_target -= right * MOVE_STEP * 2.0;
            }
            KeyCode::KeyD => {
                // Move camera target right
                let right = Vec3::new(self.camera_yaw.cos(), 0.0, -self.camera_yaw.sin());
                self.camera_target += right * MOVE_STEP * 2.0;
            }
            KeyCode::KeyQ => {
                // Move camera target down
                self.camera_target.y -= MOVE_STEP * 2.0;
            }
            KeyCode::KeyE => {
                // Move camera target up
                self.camera_target.y += MOVE_STEP * 2.0;
            }

            KeyCode::KeyN => {
                // Toggle snap mode
                self.snap_enabled = !self.snap_enabled;
                println!("Snap: {}", if self.snap_enabled { "ON" } else { "OFF" });
            }

            _ => {}
        }
    }

    fn move_selected_or_preview(&mut self, delta: Vec3) {
        match self.mode {
            EditorMode::PlaceGutter => {
                self.preview_gutter.translate(delta);
                if self.snap_enabled {
                    self.snap_gutter_preview();
                }
            }
            EditorMode::PlaceSluice => {
                self.preview_sluice.translate(delta);
                if self.snap_enabled {
                    self.snap_sluice_preview();
                }
            }
            EditorMode::PlaceEmitter => {
                self.preview_emitter.translate(delta);
                // Emitters don't snap (they emit into gutters from above)
            }
            EditorMode::Select => match self.selection {
                Selection::Gutter(idx) => {
                    if let Some(g) = self.layout.gutters.get_mut(idx) {
                        g.translate(delta);
                    }
                    if self.snap_enabled {
                        self.snap_selected_gutter(idx);
                    }
                }
                Selection::Sluice(idx) => {
                    if let Some(s) = self.layout.sluices.get_mut(idx) {
                        s.translate(delta);
                    }
                    if self.snap_enabled {
                        self.snap_selected_sluice(idx);
                    }
                }
                Selection::Emitter(idx) => {
                    if let Some(e) = self.layout.emitters.get_mut(idx) {
                        e.translate(delta);
                    }
                    // Emitters don't snap
                }
                Selection::None => {}
            },
        }
    }

    /// Snap preview gutter's inlet to nearest outlet
    fn snap_gutter_preview(&mut self) {
        let inlet = self.preview_gutter.inlet_position();

        // Check all gutter outlets
        for g in &self.layout.gutters {
            let outlet = g.outlet_position();
            let dist = (inlet - outlet).length();
            if dist < SNAP_DISTANCE {
                // Snap: move gutter so inlet aligns with outlet
                let adjustment = outlet - inlet;
                self.preview_gutter.position += adjustment;
                println!("Snapped to gutter outlet (dist={:.3})", dist);
                return;
            }
        }

        // Check all sluice outlets (less common but possible)
        for s in &self.layout.sluices {
            let outlet = s.outlet_position();
            let dist = (inlet - outlet).length();
            if dist < SNAP_DISTANCE {
                let adjustment = outlet - inlet;
                self.preview_gutter.position += adjustment;
                println!("Snapped to sluice outlet (dist={:.3})", dist);
                return;
            }
        }
    }

    /// Snap preview sluice's inlet to nearest outlet
    fn snap_sluice_preview(&mut self) {
        let inlet = self.preview_sluice.inlet_position();

        // Check all gutter outlets (most common connection)
        for g in &self.layout.gutters {
            let outlet = g.outlet_position();
            let dist = (inlet - outlet).length();
            if dist < SNAP_DISTANCE {
                let adjustment = outlet - inlet;
                self.preview_sluice.position += adjustment;
                println!("Snapped to gutter outlet (dist={:.3})", dist);
                return;
            }
        }

        // Check all sluice outlets
        for s in &self.layout.sluices {
            let outlet = s.outlet_position();
            let dist = (inlet - outlet).length();
            if dist < SNAP_DISTANCE {
                let adjustment = outlet - inlet;
                self.preview_sluice.position += adjustment;
                println!("Snapped to sluice outlet (dist={:.3})", dist);
                return;
            }
        }
    }

    /// Snap selected gutter's inlet to nearest outlet
    fn snap_selected_gutter(&mut self, idx: usize) {
        let inlet = match self.layout.gutters.get(idx) {
            Some(g) => g.inlet_position(),
            None => return,
        };

        // Check all other gutter outlets
        for (i, g) in self.layout.gutters.iter().enumerate() {
            if i == idx { continue; }
            let outlet = g.outlet_position();
            let dist = (inlet - outlet).length();
            if dist < SNAP_DISTANCE {
                let adjustment = outlet - inlet;
                if let Some(gutter) = self.layout.gutters.get_mut(idx) {
                    gutter.position += adjustment;
                    println!("Snapped to gutter #{} outlet (dist={:.3})", i, dist);
                }
                return;
            }
        }

        // Check all sluice outlets
        for (i, s) in self.layout.sluices.iter().enumerate() {
            let outlet = s.outlet_position();
            let dist = (inlet - outlet).length();
            if dist < SNAP_DISTANCE {
                let adjustment = outlet - inlet;
                if let Some(gutter) = self.layout.gutters.get_mut(idx) {
                    gutter.position += adjustment;
                    println!("Snapped to sluice #{} outlet (dist={:.3})", i, dist);
                }
                return;
            }
        }
    }

    /// Snap selected sluice's inlet to nearest outlet
    fn snap_selected_sluice(&mut self, idx: usize) {
        let inlet = match self.layout.sluices.get(idx) {
            Some(s) => s.inlet_position(),
            None => return,
        };

        // Check all gutter outlets
        for (i, g) in self.layout.gutters.iter().enumerate() {
            let outlet = g.outlet_position();
            let dist = (inlet - outlet).length();
            if dist < SNAP_DISTANCE {
                let adjustment = outlet - inlet;
                if let Some(sluice) = self.layout.sluices.get_mut(idx) {
                    sluice.position += adjustment;
                    println!("Snapped to gutter #{} outlet (dist={:.3})", i, dist);
                }
                return;
            }
        }

        // Check all other sluice outlets
        for (i, s) in self.layout.sluices.iter().enumerate() {
            if i == idx { continue; }
            let outlet = s.outlet_position();
            let dist = (inlet - outlet).length();
            if dist < SNAP_DISTANCE {
                let adjustment = outlet - inlet;
                if let Some(sluice) = self.layout.sluices.get_mut(idx) {
                    sluice.position += adjustment;
                    println!("Snapped to sluice #{} outlet (dist={:.3})", i, dist);
                }
                return;
            }
        }
    }

    fn rotate_selected_cw(&mut self) {
        match self.mode {
            EditorMode::PlaceGutter => {
                self.preview_gutter.rotate_cw();
                println!(
                    "Preview rotation: {}°",
                    self.preview_gutter.rotation.degrees()
                );
            }
            EditorMode::PlaceSluice => {
                self.preview_sluice.rotate_cw();
                println!(
                    "Preview rotation: {}°",
                    self.preview_sluice.rotation.degrees()
                );
            }
            EditorMode::PlaceEmitter => {
                self.preview_emitter.rotate_cw();
                println!(
                    "Preview rotation: {}°",
                    self.preview_emitter.rotation.degrees()
                );
            }
            EditorMode::Select => match self.selection {
                Selection::Gutter(idx) => {
                    if let Some(g) = self.layout.gutters.get_mut(idx) {
                        g.rotate_cw();
                        println!("Gutter rotation: {}°", g.rotation.degrees());
                    }
                }
                Selection::Sluice(idx) => {
                    if let Some(s) = self.layout.sluices.get_mut(idx) {
                        s.rotate_cw();
                        println!("Sluice rotation: {}°", s.rotation.degrees());
                    }
                }
                Selection::Emitter(idx) => {
                    if let Some(e) = self.layout.emitters.get_mut(idx) {
                        e.rotate_cw();
                        println!("Emitter rotation: {}°", e.rotation.degrees());
                    }
                }
                Selection::None => {}
            },
        }
    }

    fn rotate_selected_ccw(&mut self) {
        match self.mode {
            EditorMode::PlaceGutter => {
                self.preview_gutter.rotate_ccw();
                println!(
                    "Preview rotation: {}°",
                    self.preview_gutter.rotation.degrees()
                );
            }
            EditorMode::PlaceSluice => {
                self.preview_sluice.rotate_ccw();
                println!(
                    "Preview rotation: {}°",
                    self.preview_sluice.rotation.degrees()
                );
            }
            EditorMode::PlaceEmitter => {
                self.preview_emitter.rotate_ccw();
                println!(
                    "Preview rotation: {}°",
                    self.preview_emitter.rotation.degrees()
                );
            }
            EditorMode::Select => match self.selection {
                Selection::Gutter(idx) => {
                    if let Some(g) = self.layout.gutters.get_mut(idx) {
                        g.rotate_ccw();
                        println!("Gutter rotation: {}°", g.rotation.degrees());
                    }
                }
                Selection::Sluice(idx) => {
                    if let Some(s) = self.layout.sluices.get_mut(idx) {
                        s.rotate_ccw();
                        println!("Sluice rotation: {}°", s.rotation.degrees());
                    }
                }
                Selection::Emitter(idx) => {
                    if let Some(e) = self.layout.emitters.get_mut(idx) {
                        e.rotate_ccw();
                        println!("Emitter rotation: {}°", e.rotation.degrees());
                    }
                }
                Selection::None => {}
            },
        }
    }

    fn adjust_selected_angle(&mut self, delta: f32) {
        match self.mode {
            EditorMode::PlaceGutter => {
                self.preview_gutter.adjust_angle(delta);
                println!("Preview angle: {:.1}°", self.preview_gutter.angle_deg);
            }
            EditorMode::PlaceSluice => {
                self.preview_sluice.adjust_slope(delta);
                println!("Preview slope: {:.1}°", self.preview_sluice.slope_deg);
            }
            EditorMode::PlaceEmitter => {
                self.preview_emitter.adjust_spread(delta);
                println!("Preview spread: {:.1}°", self.preview_emitter.spread_deg);
            }
            EditorMode::Select => match self.selection {
                Selection::Gutter(idx) => {
                    if let Some(g) = self.layout.gutters.get_mut(idx) {
                        g.adjust_angle(delta);
                        println!("Gutter angle: {:.1}°", g.angle_deg);
                    }
                }
                Selection::Sluice(idx) => {
                    if let Some(s) = self.layout.sluices.get_mut(idx) {
                        s.adjust_slope(delta);
                        println!("Sluice slope: {:.1}°", s.slope_deg);
                    }
                }
                Selection::Emitter(idx) => {
                    if let Some(e) = self.layout.emitters.get_mut(idx) {
                        e.adjust_spread(delta);
                        println!("Emitter spread: {:.1}°", e.spread_deg);
                    }
                }
                Selection::None => {}
            },
        }
    }

    fn adjust_selected_length(&mut self, delta: f32) {
        match self.mode {
            EditorMode::PlaceGutter => {
                self.preview_gutter.adjust_length(delta);
                println!("Preview length: {:.2}m", self.preview_gutter.length);
            }
            EditorMode::PlaceSluice => {
                self.preview_sluice.adjust_length(delta);
                println!("Preview length: {:.2}m", self.preview_sluice.length);
            }
            EditorMode::PlaceEmitter => {
                self.preview_emitter.adjust_velocity(delta);
                println!("Preview velocity: {:.2}m/s", self.preview_emitter.velocity);
            }
            EditorMode::Select => match self.selection {
                Selection::Gutter(idx) => {
                    if let Some(g) = self.layout.gutters.get_mut(idx) {
                        g.adjust_length(delta);
                        println!("Gutter length: {:.2}m", g.length);
                    }
                }
                Selection::Sluice(idx) => {
                    if let Some(s) = self.layout.sluices.get_mut(idx) {
                        s.adjust_length(delta);
                        println!("Sluice length: {:.2}m", s.length);
                    }
                }
                Selection::Emitter(idx) => {
                    if let Some(e) = self.layout.emitters.get_mut(idx) {
                        e.adjust_velocity(delta);
                        println!("Emitter velocity: {:.2}m/s", e.velocity);
                    }
                }
                Selection::None => {}
            },
        }
    }

    fn adjust_selected_width(&mut self, delta: f32) {
        match self.mode {
            EditorMode::PlaceGutter => {
                self.preview_gutter.adjust_width(delta);
                println!("Preview width: {:.2}m", self.preview_gutter.width);
            }
            EditorMode::PlaceSluice => {
                self.preview_sluice.adjust_width(delta);
                println!("Preview width: {:.2}m", self.preview_sluice.width);
            }
            EditorMode::PlaceEmitter => {
                self.preview_emitter.adjust_width(delta);
                println!("Preview width: {:.2}m", self.preview_emitter.width);
            }
            EditorMode::Select => match self.selection {
                Selection::Gutter(idx) => {
                    if let Some(g) = self.layout.gutters.get_mut(idx) {
                        g.adjust_width(delta);
                        println!("Gutter width: {:.2}m", g.width);
                    }
                }
                Selection::Sluice(idx) => {
                    if let Some(s) = self.layout.sluices.get_mut(idx) {
                        s.adjust_width(delta);
                        println!("Sluice width: {:.2}m", s.width);
                    }
                }
                Selection::Emitter(idx) => {
                    if let Some(e) = self.layout.emitters.get_mut(idx) {
                        e.adjust_width(delta);
                        println!("Emitter width: {:.2}m", e.width);
                    }
                }
                Selection::None => {}
            },
        }
    }

    fn adjust_selected_end_width(&mut self, delta: f32) {
        // End width only applies to gutters (for funnel effect)
        match self.mode {
            EditorMode::PlaceGutter => {
                self.preview_gutter.adjust_end_width(delta);
                println!(
                    "Preview gutter: width {:.2}m -> {:.2}m",
                    self.preview_gutter.width, self.preview_gutter.end_width
                );
            }
            EditorMode::PlaceSluice | EditorMode::PlaceEmitter => {
                println!("End width only applies to gutters");
            }
            EditorMode::Select => match self.selection {
                Selection::Gutter(idx) => {
                    if let Some(g) = self.layout.gutters.get_mut(idx) {
                        g.adjust_end_width(delta);
                        println!("Gutter: width {:.2}m -> {:.2}m", g.width, g.end_width);
                    }
                }
                Selection::Sluice(_) | Selection::Emitter(_) => {
                    println!("End width only applies to gutters");
                }
                Selection::None => {}
            },
        }
    }

    fn cycle_selection(&mut self) {
        let total = self.layout.piece_count();
        if total == 0 {
            self.selection = Selection::None;
            return;
        }

        let num_gutters = self.layout.gutters.len();
        let num_sluices = self.layout.sluices.len();

        let current_idx = match self.selection {
            Selection::None => 0,
            Selection::Gutter(i) => i + 1,
            Selection::Sluice(i) => num_gutters + i + 1,
            Selection::Emitter(i) => num_gutters + num_sluices + i + 1,
        };

        let next_idx = current_idx % total;

        if next_idx < num_gutters {
            self.selection = Selection::Gutter(next_idx);
            println!("Selected Gutter #{}", next_idx);
        } else if next_idx < num_gutters + num_sluices {
            let sluice_idx = next_idx - num_gutters;
            self.selection = Selection::Sluice(sluice_idx);
            println!("Selected Sluice #{}", sluice_idx);
        } else {
            let emitter_idx = next_idx - num_gutters - num_sluices;
            self.selection = Selection::Emitter(emitter_idx);
            println!("Selected Emitter #{}", emitter_idx);
        }
    }

    async fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 16,
                        ..wgpu::Limits::default()
                    }
                    .using_resolution(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_texture = Self::create_depth_texture(&device, size.width, size.height);

        let mesh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SluiceVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.gpu = Some(GpuState {
            surface,
            device,
            queue,
            config,
            mesh_pipeline,
            uniform_buffer,
            bind_group,
            depth_texture,
        });
    }

    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&Default::default())
    }

    fn build_gutter_mesh(
        &self,
        gutter: &GutterPiece,
        color: [f32; 4],
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let pos = gutter.position;
        let rot = gutter.rotation.radians();
        let half_len = gutter.length / 2.0;
        // Variable width: inlet (start) vs outlet (end)
        let half_wid_inlet = gutter.width / 2.0;
        let half_wid_outlet = gutter.end_width / 2.0;
        let wall_h = gutter.wall_height;
        let half_drop = gutter.height_drop() / 2.0;

        // Transform function for rotation
        let transform = |x: f32, y: f32, z: f32| -> [f32; 3] {
            let rx = x * rot.cos() - z * rot.sin();
            let rz = x * rot.sin() + z * rot.cos();
            [pos.x + rx, pos.y + y, pos.z + rz]
        };

        // Floor (angled, tapered width)
        // Inlet at -half_len (high), outlet at +half_len (low)
        let floor_verts = [
            transform(-half_len, half_drop, -half_wid_inlet),   // inlet left
            transform(-half_len, half_drop, half_wid_inlet),    // inlet right
            transform(half_len, -half_drop, half_wid_outlet),   // outlet right
            transform(half_len, -half_drop, -half_wid_outlet),  // outlet left
        ];
        let base = vertices.len() as u32;
        for v in &floor_verts {
            vertices.push(SluiceVertex::new(*v, color));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

        // Left wall (tapered)
        let wall_color = [color[0] * 0.8, color[1] * 0.8, color[2] * 0.8, color[3]];
        let left_verts = [
            transform(-half_len, half_drop, -half_wid_inlet),
            transform(-half_len, half_drop + wall_h, -half_wid_inlet),
            transform(half_len, -half_drop + wall_h, -half_wid_outlet),
            transform(half_len, -half_drop, -half_wid_outlet),
        ];
        let base = vertices.len() as u32;
        for v in &left_verts {
            vertices.push(SluiceVertex::new(*v, wall_color));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

        // Right wall (tapered)
        let right_verts = [
            transform(-half_len, half_drop, half_wid_inlet),
            transform(-half_len, half_drop + wall_h, half_wid_inlet),
            transform(half_len, -half_drop + wall_h, half_wid_outlet),
            transform(half_len, -half_drop, half_wid_outlet),
        ];
        let base = vertices.len() as u32;
        for v in &right_verts {
            vertices.push(SluiceVertex::new(*v, wall_color));
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);

        (vertices, indices)
    }

    fn build_sluice_mesh(
        &self,
        sluice: &SluicePiece,
        color: [f32; 4],
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let pos = sluice.position;
        let rot = sluice.rotation.radians();
        let half_len = sluice.length / 2.0;
        let half_wid = sluice.width / 2.0;
        let half_drop = sluice.height_drop() / 2.0;
        let wall_h = 0.1_f32;

        let transform = |x: f32, y: f32, z: f32| -> [f32; 3] {
            let rx = x * rot.cos() - z * rot.sin();
            let rz = x * rot.sin() + z * rot.cos();
            [pos.x + rx, pos.y + y, pos.z + rz]
        };

        // Floor
        let floor_verts = [
            transform(-half_len, half_drop, -half_wid),
            transform(-half_len, half_drop, half_wid),
            transform(half_len, -half_drop, half_wid),
            transform(half_len, -half_drop, -half_wid),
        ];
        let base = vertices.len() as u32;
        for v in &floor_verts {
            vertices.push(SluiceVertex::new(*v, color));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

        // Walls
        let wall_color = [color[0] * 0.8, color[1] * 0.8, color[2] * 0.8, color[3]];

        // Left wall
        let left_verts = [
            transform(-half_len, half_drop, -half_wid),
            transform(-half_len, half_drop + wall_h, -half_wid),
            transform(half_len, -half_drop + wall_h, -half_wid),
            transform(half_len, -half_drop, -half_wid),
        ];
        let base = vertices.len() as u32;
        for v in &left_verts {
            vertices.push(SluiceVertex::new(*v, wall_color));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);

        // Right wall
        let right_verts = [
            transform(-half_len, half_drop, half_wid),
            transform(-half_len, half_drop + wall_h, half_wid),
            transform(half_len, -half_drop + wall_h, half_wid),
            transform(half_len, -half_drop, half_wid),
        ];
        let base = vertices.len() as u32;
        for v in &right_verts {
            vertices.push(SluiceVertex::new(*v, wall_color));
        }
        indices.extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);

        // Riffles
        let riffle_color = [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6, color[3]];
        let riffle_h = sluice.riffle_height;
        let num_riffles = (sluice.length / sluice.riffle_spacing) as i32;

        for i in 1..num_riffles {
            let t = i as f32 / num_riffles as f32;
            let x = -half_len + t * sluice.length;
            let y = half_drop * (1.0 - 2.0 * t);

            let riffle_verts = [
                transform(x - 0.01, y, -half_wid + 0.02),
                transform(x - 0.01, y + riffle_h, -half_wid + 0.02),
                transform(x - 0.01, y + riffle_h, half_wid - 0.02),
                transform(x - 0.01, y, half_wid - 0.02),
            ];
            let base = vertices.len() as u32;
            for v in &riffle_verts {
                vertices.push(SluiceVertex::new(*v, riffle_color));
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        (vertices, indices)
    }

    fn build_emitter_mesh(
        &self,
        emitter: &EmitterPiece,
        color: [f32; 4],
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let pos = emitter.position;
        let rot = emitter.rotation.radians();
        let r = emitter.radius;

        let transform = |x: f32, y: f32, z: f32| -> [f32; 3] {
            let rx = x * rot.cos() - z * rot.sin();
            let rz = x * rot.sin() + z * rot.cos();
            [pos.x + rx, pos.y + y, pos.z + rz]
        };

        // Draw as a cone/nozzle shape pointing in emission direction
        let nozzle_len = 0.15_f32;

        // Back circle (8 segments)
        let segments = 8;
        let back_color = [color[0] * 0.7, color[1] * 0.7, color[2] * 0.7, color[3]];

        // Create triangular faces for cone
        for i in 0..segments {
            let a1 = (i as f32 / segments as f32) * std::f32::consts::TAU;
            let a2 = ((i + 1) as f32 / segments as f32) * std::f32::consts::TAU;

            let y1 = a1.cos() * r;
            let z1 = a1.sin() * r;
            let y2 = a2.cos() * r;
            let z2 = a2.sin() * r;

            // Back face triangle
            let base = vertices.len() as u32;
            vertices.push(SluiceVertex::new(transform(0.0, 0.0, 0.0), back_color));
            vertices.push(SluiceVertex::new(transform(0.0, y1, z1), back_color));
            vertices.push(SluiceVertex::new(transform(0.0, y2, z2), back_color));
            indices.extend_from_slice(&[base, base + 2, base + 1]);

            // Side face (cone to tip)
            let base = vertices.len() as u32;
            vertices.push(SluiceVertex::new(transform(0.0, y1, z1), color));
            vertices.push(SluiceVertex::new(transform(0.0, y2, z2), color));
            vertices.push(SluiceVertex::new(transform(nozzle_len, 0.0, 0.0), color));
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }

        // Direction indicator line (extend further to show direction)
        let tip_color = [0.9, 0.9, 1.0, 1.0];
        let arrow_len = nozzle_len + emitter.velocity * 0.1;

        // Small arrow lines at the tip
        let base = vertices.len() as u32;
        let arrow_size = 0.03_f32;
        vertices.push(SluiceVertex::new(
            transform(nozzle_len, 0.0, 0.0),
            tip_color,
        ));
        vertices.push(SluiceVertex::new(transform(arrow_len, 0.0, 0.0), tip_color));
        vertices.push(SluiceVertex::new(
            transform(arrow_len - arrow_size, arrow_size, 0.0),
            tip_color,
        ));
        vertices.push(SluiceVertex::new(
            transform(arrow_len - arrow_size, -arrow_size, 0.0),
            tip_color,
        ));
        indices.extend_from_slice(&[base, base + 1]); // Main line
        indices.extend_from_slice(&[base + 1, base + 2]); // Arrow head
        indices.extend_from_slice(&[base + 1, base + 3]); // Arrow head

        (vertices, indices)
    }

    fn build_grid_mesh(&self) -> (Vec<SluiceVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        // Ground plane grid
        let size = 5.0_f32;
        let step = 0.5_f32;

        for i in 0..=((size * 2.0 / step) as i32) {
            let x = -size + i as f32 * step;
            let base = vertices.len() as u32;
            vertices.push(SluiceVertex::new([x, 0.0, -size], GRID_COLOR));
            vertices.push(SluiceVertex::new([x, 0.0, size], GRID_COLOR));
            indices.extend_from_slice(&[base, base + 1]);
        }

        for i in 0..=((size * 2.0 / step) as i32) {
            let z = -size + i as f32 * step;
            let base = vertices.len() as u32;
            vertices.push(SluiceVertex::new([-size, 0.0, z], GRID_COLOR));
            vertices.push(SluiceVertex::new([size, 0.0, z], GRID_COLOR));
            indices.extend_from_slice(&[base, base + 1]);
        }

        (vertices, indices)
    }

    fn render(&mut self) {
        if self.gpu.is_none() || self.window.is_none() {
            return;
        }

        let camera_pos = self.camera_position();
        let view_matrix = Mat4::look_at_rh(camera_pos, self.camera_target, Vec3::Y);

        let size = self.window.as_ref().unwrap().inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let proj_matrix = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.01, 100.0);
        let view_proj = proj_matrix * view_matrix;

        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        };

        // Step simulation if running
        // Collect all world-space particle positions for rendering
        let mut world_positions: Vec<Vec3> = Vec::new();

        if self.is_simulating {
            // Check if using multi-grid or legacy
            let using_multi_grid = self.multi_sim.is_some();

            if using_multi_grid {
                // Multi-grid simulation path
                // Emit first (takes &mut self)
                self.emit_from_emitters_multi();

                // Now step and collect (reborrow multi_sim)
                if let Some(multi_sim) = &mut self.multi_sim {
                    // Step all piece simulations
                    if let Some(gpu) = &self.gpu {
                        let dt = 1.0 / 60.0;
                        multi_sim.step(&gpu.device, &gpu.queue, dt);
                    }

                    // Collect all particles from all pieces in world space
                    for piece in &multi_sim.pieces {
                        for sim_pos in &piece.positions {
                            let world_pos = *sim_pos + piece.grid_offset;
                            world_positions.push(world_pos);
                        }
                    }

                    self.sim_frame += 1;

                    // Debug output periodically
                    if self.sim_frame % 60 == 0 {
                        let total = multi_sim.total_particles();
                        println!(
                            "Frame {}: {} particles ({} pieces)",
                            self.sim_frame,
                            total,
                            multi_sim.pieces.len()
                        );
                        if !world_positions.is_empty() {
                            println!("  First particle world pos: {:?}", world_positions[0]);
                        }
                    }
                }
            } else {
                // Legacy single-grid simulation path (fallback)
                self.emit_from_emitters();
                self.prepare_gpu_inputs();

                let (gw, gh, gd) = self.sim_grid_dims;
                let cell_count = gw * gh * gd;
                let mut cell_types: Vec<u32> = vec![0; cell_count];

                for pos in &self.positions {
                    let i = (pos.x / SIM_CELL_SIZE).floor() as isize;
                    let j = (pos.y / SIM_CELL_SIZE).floor() as isize;
                    let k = (pos.z / SIM_CELL_SIZE).floor() as isize;
                    if i >= 0
                        && (i as usize) < gw
                        && j >= 0
                        && (j as usize) < gh
                        && k >= 0
                        && (k as usize) < gd
                    {
                        let idx = (k as usize) * gw * gh + (j as usize) * gw + (i as usize);
                        cell_types[idx] = 1;
                    }
                }

                let sdf = self.sim.as_ref().map(|s| s.grid.sdf.clone());

                if let (Some(gpu), Some(gpu_flip)) = (&self.gpu, &mut self.gpu_flip) {
                    let dt = 1.0 / 60.0;
                    gpu_flip.step(
                        &gpu.device,
                        &gpu.queue,
                        &mut self.positions,
                        &mut self.velocities,
                        &mut self.affine_vels,
                        &self.densities,
                        &cell_types,
                        sdf.as_deref(),
                        None,
                        dt,
                        SIM_GRAVITY,
                        0.0,
                        SIM_PRESSURE_ITERS as u32,
                    );
                }

                // Convert to world space for rendering
                for sim_pos in &self.positions {
                    world_positions.push(*sim_pos + self.sim_grid_offset);
                }

                self.sim_frame += 1;

                if self.sim_frame % 60 == 0 {
                    println!(
                        "Frame {}: {} particles (legacy)",
                        self.sim_frame,
                        self.positions.len()
                    );
                }
            }
        }

        // Write uniforms
        let gpu = self.gpu.as_ref().unwrap();
        gpu.queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Collect all meshes
        let mut all_vertices: Vec<SluiceVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        // Grid
        let (grid_v, grid_i) = self.build_grid_mesh();
        let base = all_vertices.len() as u32;
        all_vertices.extend(grid_v);
        all_indices.extend(grid_i.iter().map(|i| i + base));

        // Gutters
        for (idx, gutter) in self.layout.gutters.iter().enumerate() {
            let color = if matches!(self.selection, Selection::Gutter(i) if i == idx) {
                GUTTER_SELECTED
            } else {
                GUTTER_COLOR
            };
            let (verts, inds) = self.build_gutter_mesh(gutter, color);
            let base = all_vertices.len() as u32;
            all_vertices.extend(verts);
            all_indices.extend(inds.iter().map(|i| i + base));
        }

        // Sluices
        for (idx, sluice) in self.layout.sluices.iter().enumerate() {
            let color = if matches!(self.selection, Selection::Sluice(i) if i == idx) {
                SLUICE_SELECTED
            } else {
                SLUICE_COLOR
            };
            let (verts, inds) = self.build_sluice_mesh(sluice, color);
            let base = all_vertices.len() as u32;
            all_vertices.extend(verts);
            all_indices.extend(inds.iter().map(|i| i + base));
        }

        // Emitters
        for (idx, emitter) in self.layout.emitters.iter().enumerate() {
            let color = if matches!(self.selection, Selection::Emitter(i) if i == idx) {
                EMITTER_SELECTED
            } else {
                EMITTER_COLOR
            };
            let (verts, inds) = self.build_emitter_mesh(emitter, color);
            let base = all_vertices.len() as u32;
            all_vertices.extend(verts);
            all_indices.extend(inds.iter().map(|i| i + base));
        }

        // Preview piece
        match self.mode {
            EditorMode::PlaceGutter => {
                let (verts, inds) = self.build_gutter_mesh(&self.preview_gutter, PREVIEW_COLOR);
                let base = all_vertices.len() as u32;
                all_vertices.extend(verts);
                all_indices.extend(inds.iter().map(|i| i + base));
            }
            EditorMode::PlaceSluice => {
                let (verts, inds) = self.build_sluice_mesh(&self.preview_sluice, PREVIEW_COLOR);
                let base = all_vertices.len() as u32;
                all_vertices.extend(verts);
                all_indices.extend(inds.iter().map(|i| i + base));
            }
            EditorMode::PlaceEmitter => {
                let (verts, inds) = self.build_emitter_mesh(&self.preview_emitter, PREVIEW_COLOR);
                let base = all_vertices.len() as u32;
                all_vertices.extend(verts);
                all_indices.extend(inds.iter().map(|i| i + base));
            }
            EditorMode::Select => {}
        }

        // Render particles as small quads (already in world space)
        if self.is_simulating && !world_positions.is_empty() {
            let particle_size = SIM_CELL_SIZE * 0.4;
            let particle_color: [f32; 4] = [0.2, 0.5, 0.9, 1.0]; // Blue water

            // Limit particles rendered for performance (4 verts + 6 indices per particle)
            let max_render = 5000.min(world_positions.len());

            for i in 0..max_render {
                let world_pos = world_positions[i];

                // Create a small billboard quad facing the camera
                let to_cam = (camera_pos - world_pos).normalize();
                let right = to_cam.cross(Vec3::Y).normalize() * particle_size;
                let up = Vec3::Y * particle_size;

                let p0 = world_pos - right - up;
                let p1 = world_pos + right - up;
                let p2 = world_pos + right + up;
                let p3 = world_pos - right + up;

                let base = all_vertices.len() as u32;
                all_vertices.push(SluiceVertex {
                    position: p0.to_array(),
                    color: particle_color,
                });
                all_vertices.push(SluiceVertex {
                    position: p1.to_array(),
                    color: particle_color,
                });
                all_vertices.push(SluiceVertex {
                    position: p2.to_array(),
                    color: particle_color,
                });
                all_vertices.push(SluiceVertex {
                    position: p3.to_array(),
                    color: particle_color,
                });
                all_indices.extend_from_slice(&[
                    base,
                    base + 1,
                    base + 2,
                    base,
                    base + 2,
                    base + 3,
                ]);
            }
        }

        // Create buffers
        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&all_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&all_indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        // Render
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&Default::default());

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&gpu.mesh_pipeline);
            pass.set_bind_group(0, &gpu.bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..all_indices.len() as u32, 0, 0..1);
        }

        // NOTE: fluid_renderer uses GPU buffer positions which are in sim-space.
        // For now, we render particles as simple quads in the mesh pass above.
        // The particle geometry is added before this point.

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.window.as_ref().unwrap().request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes()
            .with_title("Washplant Editor")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 800));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window.clone());

        pollster::block_on(self.init_gpu(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    self.mouse_pressed = state == ElementState::Pressed;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_pressed {
                    if let Some((lx, ly)) = self.last_mouse_pos {
                        let dx = position.x - lx;
                        let dy = position.y - ly;
                        self.camera_yaw += dx as f32 * 0.005;
                        self.camera_pitch =
                            (self.camera_pitch + dy as f32 * 0.005).clamp(-1.4, 1.4);
                    }
                }
                self.last_mouse_pos = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.5).clamp(1.0, 20.0);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // Track shift state
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) => {
                        self.shift_pressed = event.state == ElementState::Pressed;
                    }
                    _ => {}
                }

                if event.state.is_pressed() {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        self.handle_key(key);
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = size.width.max(1);
                    gpu.config.height = size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                    gpu.depth_texture =
                        Self::create_depth_texture(&gpu.device, size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }
}

const MESH_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: count solid cells in a simulation grid
    fn count_solids(sim: &FlipSimulation3D) -> usize {
        let mut count = 0;
        for i in 0..sim.grid.width {
            for j in 0..sim.grid.height {
                for k in 0..sim.grid.depth {
                    if sim.grid.is_solid(i, j, k) {
                        count += 1;
                    }
                }
            }
        }
        count
    }

    /// Test helper: check if a specific cell is solid
    fn is_solid_at(sim: &FlipSimulation3D, i: usize, j: usize, k: usize) -> bool {
        if i < sim.grid.width && j < sim.grid.height && k < sim.grid.depth {
            sim.grid.is_solid(i, j, k)
        } else {
            false
        }
    }

    /// Test helper: print a cross-section of the grid (j slice)
    fn print_j_slice(sim: &FlipSimulation3D, j: usize) {
        println!("=== J={} slice (X horizontal, Z vertical) ===", j);
        for k in (0..sim.grid.depth).rev() {
            let mut row = String::new();
            for i in 0..sim.grid.width {
                if sim.grid.is_solid(i, j, k) {
                    row.push('#');
                } else {
                    row.push('.');
                }
            }
            println!("k={:2}: {}", k, row);
        }
    }

    /// Test helper: print a cross-section of the grid (k slice)
    fn print_k_slice(sim: &FlipSimulation3D, k: usize) {
        println!("=== K={} slice (X horizontal, Y vertical) ===", k);
        for j in (0..sim.grid.height).rev() {
            let mut row = String::new();
            for i in 0..sim.grid.width {
                if sim.grid.is_solid(i, j, k) {
                    row.push('#');
                } else {
                    row.push('.');
                }
            }
            println!("j={:2}: {}", j, row);
        }
    }

    #[test]
    fn test_gutter_floor_is_solid_below() {
        // Create a simple gutter in a small grid
        let cell_size = 0.025;
        let width = 40;
        let height = 20;
        let depth = 20;

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);

        // Gutter positioned at center of grid
        let gutter = GutterPiece {
            id: 0,
            position: Vec3::new(0.5, 0.25, 0.25), // Center at (20, 10, 10) in cells
            rotation: Rotation::R0,
            angle_deg: 0.0, // Flat floor for simplicity
            length: 0.5,    // 20 cells
            width: 0.25,    // 10 cells
            end_width: 0.25,
            wall_height: 0.1,
        };

        MultiGridSim::mark_gutter_solid_cells(&mut sim, &gutter, cell_size);

        // Print debug visualization
        println!("\nGutter test - flat floor at j=10");
        print_k_slice(&sim, 10); // Center depth slice

        // The gutter should have floor at approximately j=10 (0.25m / 0.025m)
        // All cells at j <= floor_j should be solid within the channel bounds
        let center_i = (gutter.position.x / cell_size).round() as usize; // 20
        let center_k = (gutter.position.z / cell_size).round() as usize; // 10
        let floor_j = (gutter.position.y / cell_size).round() as usize; // 10

        println!(
            "Expected center: i={}, k={}, floor_j={}",
            center_i, center_k, floor_j
        );

        // Check that cells at and below floor are solid
        for j in 0..=floor_j {
            let solid = is_solid_at(&sim, center_i, j, center_k);
            assert!(
                solid,
                "Cell at center ({},{},{}) should be solid (floor), but is not",
                center_i, j, center_k
            );
        }

        // Check that cells above floor (in channel) are NOT solid
        let above_floor = is_solid_at(&sim, center_i, floor_j + 2, center_k);
        assert!(
            !above_floor,
            "Cell at ({},{},{}) should NOT be solid (above floor)",
            center_i,
            floor_j + 2,
            center_k
        );

        let solid_count = count_solids(&sim);
        println!("Total solid cells: {}", solid_count);
        assert!(solid_count > 0, "Should have some solid cells");
    }

    #[test]
    fn test_gutter_angled_floor() {
        // Test that angled floor has different heights at inlet vs outlet
        let cell_size = 0.025;
        let width = 60;
        let height = 30;
        let depth = 20;

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);

        // Gutter with 15 degree angle
        let gutter = GutterPiece {
            id: 0,
            position: Vec3::new(0.75, 0.375, 0.25), // Center at (30, 15, 10) in cells
            rotation: Rotation::R0,
            angle_deg: 15.0, // Angled floor
            length: 0.75,    // 30 cells
            width: 0.25,     // 10 cells
            end_width: 0.25,
            wall_height: 0.15,
        };

        MultiGridSim::mark_gutter_solid_cells(&mut sim, &gutter, cell_size);

        // Print debug visualization
        println!("\nGutter test - angled floor (15 deg)");
        print_k_slice(&sim, 10); // Center depth slice

        let center_i = (gutter.position.x / cell_size).round() as i32;
        let center_k = (gutter.position.z / cell_size).round() as usize;
        let half_len_cells = ((gutter.length / 2.0) / cell_size).ceil() as i32;

        let inlet_i = (center_i - half_len_cells + 2) as usize; // Near inlet (left)
        let outlet_i = (center_i + half_len_cells - 2) as usize; // Near outlet (right)

        // Find floor height at inlet and outlet by scanning
        let mut inlet_floor_j = 0;
        let mut outlet_floor_j = 0;

        for j in (0..height).rev() {
            if is_solid_at(&sim, inlet_i, j, center_k) {
                inlet_floor_j = j;
                break;
            }
        }
        for j in (0..height).rev() {
            if is_solid_at(&sim, outlet_i, j, center_k) {
                outlet_floor_j = j;
                break;
            }
        }

        println!(
            "Inlet (i={}) floor_j = {}, Outlet (i={}) floor_j = {}",
            inlet_i, inlet_floor_j, outlet_i, outlet_floor_j
        );

        // Inlet should be HIGHER than outlet (water flows down)
        assert!(
            inlet_floor_j > outlet_floor_j,
            "Inlet floor ({}) should be higher than outlet floor ({})",
            inlet_floor_j,
            outlet_floor_j
        );
    }

    #[test]
    fn test_gutter_walls_exist() {
        // Test that side walls exist outside the channel
        let cell_size = 0.025;
        let width = 40;
        let height = 20;
        let depth = 24;

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);

        // Gutter positioned at center of grid
        let gutter = GutterPiece {
            id: 0,
            position: Vec3::new(0.5, 0.25, 0.3), // Center at (20, 10, 12) in cells
            rotation: Rotation::R0,
            angle_deg: 0.0,
            length: 0.5,  // 20 cells
            width: 0.25,  // 10 cells (half_wid = 5 cells)
            end_width: 0.25,
            wall_height: 0.15,
        };

        MultiGridSim::mark_gutter_solid_cells(&mut sim, &gutter, cell_size);

        let center_i = (gutter.position.x / cell_size).round() as usize;
        let center_k = (gutter.position.z / cell_size).round() as usize;
        let half_wid_cells = ((gutter.width / 2.0) / cell_size).ceil() as usize;
        let floor_j = (gutter.position.y / cell_size).round() as usize;

        println!("\nGutter walls test");
        println!(
            "center_i={}, center_k={}, half_wid_cells={}, floor_j={}",
            center_i, center_k, half_wid_cells, floor_j
        );

        // Print cross-section at floor level + 2 to see walls
        print_j_slice(&sim, floor_j + 2);

        // Check left wall (k < center_k - half_wid_cells)
        let left_wall_k = center_k - half_wid_cells - 1;
        let wall_j = floor_j + 3; // Above floor
        let left_wall_solid = is_solid_at(&sim, center_i, wall_j, left_wall_k);
        println!(
            "Left wall at ({},{},{}) solid: {}",
            center_i, wall_j, left_wall_k, left_wall_solid
        );

        // Check right wall (k > center_k + half_wid_cells)
        let right_wall_k = center_k + half_wid_cells;
        let right_wall_solid = is_solid_at(&sim, center_i, wall_j, right_wall_k);
        println!(
            "Right wall at ({},{},{}) solid: {}",
            center_i, wall_j, right_wall_k, right_wall_solid
        );

        // Check that channel center is NOT solid (above floor)
        let channel_solid = is_solid_at(&sim, center_i, wall_j, center_k);
        println!(
            "Channel center at ({},{},{}) solid: {}",
            center_i, wall_j, center_k, channel_solid
        );

        assert!(
            left_wall_solid || right_wall_solid,
            "At least one wall should be solid"
        );
        assert!(
            !channel_solid,
            "Channel center above floor should NOT be solid"
        );
    }

    #[test]
    fn test_sluice_floor_is_solid_below() {
        // Create a simple sluice in a small grid
        let cell_size = 0.025;
        let width = 60;
        let height = 20;
        let depth = 24;

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);

        // Sluice positioned at center of grid
        let sluice = SluicePiece {
            id: 0,
            position: Vec3::new(0.75, 0.25, 0.3), // Center at (30, 10, 12) in cells
            rotation: Rotation::R0,
            length: 1.0,  // 40 cells
            width: 0.3,   // 12 cells
            slope_deg: 0.0, // Flat for simplicity
            riffle_spacing: 0.2,
            riffle_height: 0.02,
        };

        MultiGridSim::mark_sluice_solid_cells(&mut sim, &sluice, cell_size);

        // Print debug visualization
        println!("\nSluice test - flat floor");
        print_k_slice(&sim, 12); // Center depth slice

        let center_i = (sluice.position.x / cell_size).round() as usize;
        let center_k = (sluice.position.z / cell_size).round() as usize;
        let floor_j = (sluice.position.y / cell_size).round() as usize;

        println!(
            "Expected center: i={}, k={}, floor_j={}",
            center_i, center_k, floor_j
        );

        // Check that cells at and below floor are solid
        for j in 0..=floor_j {
            let solid = is_solid_at(&sim, center_i, j, center_k);
            assert!(
                solid,
                "Cell at center ({},{},{}) should be solid (floor), but is not",
                center_i, j, center_k
            );
        }

        let solid_count = count_solids(&sim);
        println!("Total solid cells: {}", solid_count);
        assert!(solid_count > 0, "Should have some solid cells");
    }

    #[test]
    fn test_sdf_negative_inside_solid() {
        // Test that SDF is negative inside solid cells
        let cell_size = 0.025;
        let width = 40;
        let height = 20;
        let depth = 20;

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);

        // Simple gutter
        let gutter = GutterPiece {
            id: 0,
            position: Vec3::new(0.5, 0.25, 0.25),
            rotation: Rotation::R0,
            angle_deg: 0.0,
            length: 0.5,
            width: 0.25,
            end_width: 0.25,
            wall_height: 0.1,
        };

        MultiGridSim::mark_gutter_solid_cells(&mut sim, &gutter, cell_size);
        sim.grid.compute_sdf();

        let center_i = (gutter.position.x / cell_size).round() as usize;
        let center_k = (gutter.position.z / cell_size).round() as usize;
        let floor_j = (gutter.position.y / cell_size).round() as usize;

        // Get SDF value at floor (should be negative - inside solid)
        let idx_floor = center_k * width * height + (floor_j / 2) * width + center_i;
        let sdf_inside = sim.grid.sdf[idx_floor];

        // Get SDF value above floor (should be positive - outside solid)
        let idx_above = center_k * width * height + (floor_j + 3) * width + center_i;
        let sdf_outside = sim.grid.sdf[idx_above];

        println!("SDF inside floor (j={}): {}", floor_j / 2, sdf_inside);
        println!("SDF above floor (j={}): {}", floor_j + 3, sdf_outside);

        assert!(
            sdf_inside < 0.0,
            "SDF inside solid should be negative, got {}",
            sdf_inside
        );
        assert!(
            sdf_outside > 0.0,
            "SDF outside solid should be positive, got {}",
            sdf_outside
        );
    }

    #[test]
    fn test_gutter_grid_offset_calculation() {
        // This test verifies that the grid_offset calculation correctly positions
        // the gutter in world space. This was a bug where Z offset didn't account
        // for gutter width.
        let cell_size = SIM_CELL_SIZE;
        let margin = cell_size * 4.0;

        // Simulate the grid offset calculation from add_gutter()
        let gutter = GutterPiece {
            id: 1,
            position: Vec3::new(-1.4, 1.0, -0.4),
            rotation: Rotation::R0,
            angle_deg: 15.0,
            length: 1.2,
            width: 0.3,
            end_width: 0.3,
            wall_height: 0.1,
        };

        let (dir_x, dir_z) = match gutter.rotation {
            Rotation::R0 => (1.0f32, 0.0f32),
            Rotation::R90 => (0.0, 1.0),
            Rotation::R180 => (-1.0, 0.0),
            Rotation::R270 => (0.0, -1.0),
        };

        // Current (buggy) grid offset calculation
        let buggy_grid_offset = Vec3::new(
            gutter.position.x - gutter.length / 2.0 * dir_x.abs() - margin,
            gutter.position.y - margin,
            gutter.position.z - gutter.length / 2.0 * dir_z.abs() - margin,
        );

        // Correct grid offset calculation (accounts for width on perpendicular axis)
        let correct_grid_offset = Vec3::new(
            gutter.position.x
                - gutter.length / 2.0 * dir_x.abs()
                - gutter.width / 2.0 * dir_z.abs()
                - margin,
            gutter.position.y - margin,
            gutter.position.z
                - gutter.length / 2.0 * dir_z.abs()
                - gutter.width / 2.0 * dir_x.abs()
                - margin,
        );

        // Local gutter position (from add_gutter)
        let gutter_local_pos = Vec3::new(
            margin + gutter.length / 2.0,
            margin,
            margin + gutter.width / 2.0,
        );

        // World position = grid_offset + local_pos
        let buggy_world_pos = buggy_grid_offset + gutter_local_pos;
        let correct_world_pos = correct_grid_offset + gutter_local_pos;

        println!("\nGrid offset test:");
        println!("Original gutter position: {:?}", gutter.position);
        println!("Buggy grid_offset: {:?}", buggy_grid_offset);
        println!("Correct grid_offset: {:?}", correct_grid_offset);
        println!("Local gutter position: {:?}", gutter_local_pos);
        println!("Buggy world position: {:?}", buggy_world_pos);
        println!("Correct world position: {:?}", correct_world_pos);

        // The buggy version has wrong Z
        println!(
            "Buggy Z error: {} (should be 0)",
            (buggy_world_pos.z - gutter.position.z).abs()
        );

        // The correct version should match exactly
        let x_err = (correct_world_pos.x - gutter.position.x).abs();
        let y_err = (correct_world_pos.y - gutter.position.y).abs();
        let z_err = (correct_world_pos.z - gutter.position.z).abs();

        println!("Correct position errors: x={}, y={}, z={}", x_err, y_err, z_err);

        // Assert correct version matches original position
        assert!(
            x_err < 0.001,
            "X position error {} > 0.001",
            x_err
        );
        assert!(
            y_err < 0.001,
            "Y position error {} > 0.001",
            y_err
        );
        assert!(
            z_err < 0.001,
            "Z position error {} > 0.001",
            z_err
        );

        // This demonstrates the bug - the buggy version has wrong Z
        let buggy_z_err = (buggy_world_pos.z - gutter.position.z).abs();
        println!("Buggy Z error = {} (expected ~{})", buggy_z_err, gutter.width / 2.0);

        // The bug should cause approximately width/2 error
        assert!(
            buggy_z_err > 0.1,
            "Buggy calculation should have significant Z error"
        );
    }

    /// Helper: simulate transfer coordinate transformation
    fn simulate_transfer(
        particle_local_pos: Vec3,
        from_grid_offset: Vec3,
        to_grid_offset: Vec3,
        to_grid_dims: (usize, usize, usize),
        cell_size: f32,
    ) -> (Vec3, bool) {
        // Convert to world space
        let world_pos = particle_local_pos + from_grid_offset;

        // Convert to target local space
        let target_local_pos = world_pos - to_grid_offset;

        // Check if within bounds (with 1 cell margin)
        let min_bound = cell_size;
        let max_x = (to_grid_dims.0 as f32) * cell_size - cell_size;
        let max_y = (to_grid_dims.1 as f32) * cell_size - cell_size;
        let max_z = (to_grid_dims.2 as f32) * cell_size - cell_size;

        let in_bounds = target_local_pos.x >= min_bound
            && target_local_pos.x <= max_x
            && target_local_pos.y >= min_bound
            && target_local_pos.y <= max_y
            && target_local_pos.z >= min_bound
            && target_local_pos.z <= max_z;

        // Clamp (this is what causes teleportation if out of bounds)
        let clamped_pos = Vec3::new(
            target_local_pos.x.clamp(min_bound, max_x),
            target_local_pos.y.clamp(min_bound, max_y),
            target_local_pos.z.clamp(min_bound, max_z),
        );

        (clamped_pos, in_bounds)
    }

    #[test]
    fn test_transfer_perfectly_aligned() {
        // Gutter and sluice at same Y level, sluice inlet at gutter outlet X
        let cell_size = 0.025;

        // Gutter: outlet at world (1.0, 0.5, 0.0)
        let gutter_grid_offset = Vec3::new(0.0, 0.4, -0.15);
        let gutter_outlet_local = Vec3::new(1.0, 0.1, 0.15); // outlet in gutter local space

        // Sluice: inlet at world (1.0, 0.5, 0.0) - perfectly aligned
        // Grid starts before inlet to contain it
        let sluice_grid_offset = Vec3::new(0.9, 0.4, -0.15);
        let sluice_grid_dims = (80, 36, 20); // 2m x 0.9m x 0.5m

        let (result_pos, in_bounds) = simulate_transfer(
            gutter_outlet_local,
            gutter_grid_offset,
            sluice_grid_offset,
            sluice_grid_dims,
            cell_size,
        );

        println!("Perfectly aligned transfer:");
        println!("  Particle local pos: {:?}", gutter_outlet_local);
        println!("  World pos: {:?}", gutter_outlet_local + gutter_grid_offset);
        println!("  Result in sluice local: {:?}", result_pos);
        println!("  In bounds: {}", in_bounds);

        assert!(in_bounds, "Perfectly aligned transfer should be in bounds");

        // Position should be preserved (within cell tolerance)
        let world_before = gutter_outlet_local + gutter_grid_offset;
        let world_after = result_pos + sluice_grid_offset;
        let error = (world_before - world_after).length();
        assert!(
            error < cell_size,
            "Position should be preserved, error {} >= {}",
            error,
            cell_size
        );
    }

    #[test]
    fn test_transfer_gutter_above_sluice() {
        // Gutter outlet is ABOVE sluice inlet (common case - water falls down)
        let cell_size = 0.025;

        // Gutter: outlet at world (1.0, 0.8, 0.0) - higher up
        let gutter_grid_offset = Vec3::new(0.0, 0.7, -0.15);
        let gutter_outlet_local = Vec3::new(1.0, 0.1, 0.15);

        // Sluice: inlet at world (1.0, 0.5, 0.0) - lower
        // Grid must extend UP to catch incoming water
        let sluice_grid_offset = Vec3::new(0.9, 0.4, -0.15);
        let sluice_grid_dims = (80, 36, 20); // 36 cells = 0.9m height

        let (result_pos, in_bounds) = simulate_transfer(
            gutter_outlet_local,
            gutter_grid_offset,
            sluice_grid_offset,
            sluice_grid_dims,
            cell_size,
        );

        println!("Gutter above sluice transfer:");
        println!("  Gutter outlet world: {:?}", gutter_outlet_local + gutter_grid_offset);
        println!("  Result in sluice local: {:?}", result_pos);
        println!("  Sluice grid Y range: {} to {}", cell_size, sluice_grid_dims.1 as f32 * cell_size - cell_size);
        println!("  In bounds: {}", in_bounds);

        assert!(
            in_bounds,
            "Transfer from above should be in bounds (sluice grid tall enough)"
        );

        // Position should be preserved
        let world_before = gutter_outlet_local + gutter_grid_offset;
        let world_after = result_pos + sluice_grid_offset;
        let error = (world_before - world_after).length();
        assert!(
            error < cell_size,
            "Position should be preserved, error {} >= {}",
            error,
            cell_size
        );
    }

    #[test]
    fn test_transfer_gutter_below_sluice_fails() {
        // Gutter outlet is BELOW sluice inlet (physically impossible - water can't flow up)
        let cell_size = 0.025;

        // Gutter: outlet at world (1.0, 0.3, 0.0) - BELOW sluice
        let gutter_grid_offset = Vec3::new(0.0, 0.2, -0.15);
        let gutter_outlet_local = Vec3::new(1.0, 0.1, 0.15);

        // Sluice: inlet at world (1.0, 0.5, 0.0) - ABOVE gutter outlet
        let sluice_grid_offset = Vec3::new(0.9, 0.4, -0.15);
        let sluice_grid_dims = (80, 36, 20);

        let (result_pos, in_bounds) = simulate_transfer(
            gutter_outlet_local,
            gutter_grid_offset,
            sluice_grid_offset,
            sluice_grid_dims,
            cell_size,
        );

        println!("Gutter below sluice transfer (should fail):");
        println!("  Gutter outlet world Y: {}", (gutter_outlet_local + gutter_grid_offset).y);
        println!("  Sluice grid Y min: {}", sluice_grid_offset.y + cell_size);
        println!("  Result in sluice local: {:?}", result_pos);
        println!("  In bounds: {}", in_bounds);

        // This SHOULD be out of bounds (below sluice grid)
        assert!(
            !in_bounds,
            "Transfer from below should be OUT of bounds (water can't flow up)"
        );

        // The clamped position should be different from the original (teleportation)
        let world_before = gutter_outlet_local + gutter_grid_offset;
        let world_after = result_pos + sluice_grid_offset;
        let error = (world_before - world_after).length();
        println!("  Position error (teleportation): {}", error);
        assert!(
            error > cell_size,
            "Below transfer should cause teleportation, error {} <= {}",
            error,
            cell_size
        );
    }

    #[test]
    fn test_particle_does_not_fall_through_gutter_floor() {
        // This test verifies that particles don't fall through the gutter floor
        // by running the simulation and checking Y positions stay above floor level.
        let cell_size = 0.025;
        let width = 48;
        let height = 28;
        let depth = 20;

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);
        sim.pressure_iterations = 40;

        // Set up a gutter with moderate angle
        let margin = cell_size * 4.0;
        let gutter = GutterPiece {
            id: 1,
            position: Vec3::new(
                margin + 0.5, // 1m gutter centered
                margin,
                margin + 0.15, // 0.3m width centered
            ),
            rotation: Rotation::R0,
            angle_deg: 10.0,
            length: 1.0,
            width: 0.3,
            end_width: 0.3,
            wall_height: 0.1,
        };

        MultiGridSim::mark_gutter_solid_cells(&mut sim, &gutter, cell_size);
        sim.grid.compute_sdf();

        // Count solid cells to verify floor exists
        let mut solid_count = 0;
        for j in 0..height {
            for k in 0..depth {
                for i in 0..width {
                    if sim.grid.is_solid(i, j, k) {
                        solid_count += 1;
                    }
                }
            }
        }
        println!("Solid cells: {}", solid_count);
        assert!(solid_count > 100, "Floor should have many solid cells");

        // Calculate the floor position at the center of the gutter
        let center_i = (gutter.position.x / cell_size).round() as usize;
        let center_k = (gutter.position.z / cell_size).round() as usize;

        // Print solid cells at center column
        println!("Solid cells at center_i={}, center_k={}:", center_i, center_k);
        for j in 0..height.min(10) {
            let solid = sim.grid.is_solid(center_i, j, center_k);
            let sdf = sim.grid.sdf[sim.grid.cell_index(center_i, j, center_k)];
            println!("  j={:2}: solid={}, sdf={:.3}", j, solid, sdf);
        }

        // Print Z cross-section to verify side walls
        println!("Z cross-section at center_i={}, j=6:", center_i);
        for k in 0..depth {
            let solid = sim.grid.is_solid(center_i, 6, k);
            let s = if solid { "#" } else { "." };
            print!("{}", s);
        }
        println!(" (k=0..{})", depth);

        // Find the actual floor height at center (highest j that's solid in the channel)
        let mut floor_j = 0;
        for j in 0..height {
            if sim.grid.is_solid(center_i, j, center_k) {
                floor_j = j;
            }
        }
        println!("Floor at center: j={} (y={:.3}m)", floor_j, floor_j as f32 * cell_size);

        // Spawn multiple particles above the floor (single particle causes pressure instability)
        // Spawn only in the center of the gutter to avoid side wall issues
        let particle_start_y = (floor_j as f32 + 4.0) * cell_size; // 4 cells above floor
        let num_particles = 30;
        println!("Spawning {} particles at Y={:.3}m, center_k={}", num_particles, particle_start_y, center_k);

        for i in 0..num_particles {
            // Spread particles along the center line of the gutter
            let t = i as f32 / num_particles as f32;
            // Only use the middle 60% of gutter length to avoid outlet issues
            let x = (center_i as f32 - 8.0 + t * 16.0) * cell_size;
            let z = center_k as f32 * cell_size; // Center of gutter width
            let y = particle_start_y + (i % 3) as f32 * cell_size * 0.5; // Slight Y variation

            let pos = Vec3::new(x, y, z);
            sim.spawn_particle_with_velocity(pos, Vec3::ZERO);
        }
        assert_eq!(sim.particles.list.len(), num_particles, "Should have {} particles", num_particles);

        // Run simulation for many steps
        let dt = 0.008;
        let steps = 200;
        let floor_y = floor_j as f32 * cell_size;
        let mut min_y_seen = f32::MAX;
        let mut min_y_in_channel = f32::MAX; // Only track particles inside channel bounds

        for step in 0..steps {
            let count_before = sim.particles.list.len();

            sim.update(dt);

            let count_after = sim.particles.list.len();

            if sim.particles.list.is_empty() {
                println!("All particles removed at step {}!", step);
                println!("  Floor Y: {:.4}", floor_y);
                panic!("All particles fell through floor and were removed!");
            }

            // Track minimum Y across all particles
            // Account for angled floor - floor height varies with X position
            let angle_rad = gutter.angle_deg.to_radians();
            let center_x = gutter.position.x;
            let half_len = gutter.length / 2.0;
            let half_wid = gutter.width / 2.0;
            let center_z = gutter.position.z;

            for p in &sim.particles.list {
                min_y_seen = min_y_seen.min(p.position.y);

                // Only check floor penetration for particles INSIDE the channel
                // Particles that exit through the outlet are expected to fall freely
                let dx = p.position.x - center_x;
                let dz = (p.position.z - center_z).abs();
                let in_channel_x = dx >= -half_len && dx <= half_len;
                let in_channel_z = dz <= half_wid;

                if !in_channel_x || !in_channel_z {
                    continue; // Skip particles outside channel (expected behavior)
                }

                // Track min Y for in-channel particles
                min_y_in_channel = min_y_in_channel.min(p.position.y);

                // Calculate floor Y at this particle's X position
                let local_floor_y = gutter.position.y - dx * angle_rad.tan();

                // Check each particle hasn't fallen through its local floor
                if p.position.y < local_floor_y - cell_size * 2.0 {
                    println!(
                        "Step {}: particle at ({:.3},{:.3},{:.3}) fell below floor at y={:.4}",
                        step, p.position.x, p.position.y, p.position.z, local_floor_y
                    );
                    panic!(
                        "Particle fell through floor! y={:.4} < local_floor={:.4} - 2*cell (step {})",
                        p.position.y, local_floor_y, step
                    );
                }
            }

            // Print progress every 20 steps
            if step % 20 == 0 {
                println!(
                    "Step {:3}: {} particles, min_y={:.3}, floor_y={:.3}",
                    step, count_after, min_y_seen, floor_y
                );
            }
        }

        let final_count = sim.particles.list.len();
        let final_y = sim.particles.list.iter().map(|p| p.position.y).fold(f32::MAX, f32::min);
        println!("After {} steps:", steps);
        println!("  Final particle count: {}", final_count);
        println!("  Final Y (any): {:.4}", final_y);
        println!("  Min Y seen (any): {:.4}", min_y_seen);
        println!("  Min Y in channel: {:.4}", min_y_in_channel);
        println!("  Floor Y at center: {:.4}", floor_y);

        // The per-step check above validates particles stay above their LOCAL floor.
        // The gutter has an angled floor so the outlet is lower than the center.
        // For a 10° angle and 1m gutter, outlet floor is ~0.088m lower than center.
        // Just verify the simulation ran without the per-step panic triggering.
        println!("Test passed: {} particles remain, none fell through their local floor", final_count);
    }

    #[test]
    fn test_particle_does_not_fall_through_sluice_floor() {
        // This test verifies that particles don't fall through the sluice floor
        // by running the simulation and checking Y positions stay above floor level.
        let cell_size = 0.025;
        let width = 80;
        let height = 20;
        let depth = 28;

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);
        sim.pressure_iterations = 40;

        // Set up a sluice with moderate slope
        let margin = cell_size * 4.0;
        let sluice = SluicePiece {
            id: 1,
            position: Vec3::new(
                margin + 1.0, // 2m sluice centered
                margin + 0.1,
                margin + 0.2, // 0.4m width centered
            ),
            rotation: Rotation::R0,
            slope_deg: 8.0,
            length: 2.0,
            width: 0.4,
            riffle_spacing: 0.15,
            riffle_height: 0.02,
        };

        MultiGridSim::mark_sluice_solid_cells(&mut sim, &sluice, cell_size);
        sim.grid.compute_sdf();

        // Count solid cells to verify floor exists
        let mut solid_count = 0;
        for j in 0..height {
            for k in 0..depth {
                for i in 0..width {
                    if sim.grid.is_solid(i, j, k) {
                        solid_count += 1;
                    }
                }
            }
        }
        println!("Solid cells: {}", solid_count);
        assert!(solid_count > 100, "Floor should have many solid cells");

        // Calculate the floor position at the center of the sluice
        let center_i = (sluice.position.x / cell_size).round() as usize;
        let center_k = (sluice.position.z / cell_size).round() as usize;

        // Print solid cells at center column
        println!("Solid cells at center_i={}, center_k={}:", center_i, center_k);
        for j in 0..height.min(10) {
            let solid = sim.grid.is_solid(center_i, j, center_k);
            let sdf = sim.grid.sdf[sim.grid.cell_index(center_i, j, center_k)];
            println!("  j={:2}: solid={}, sdf={:.3}", j, solid, sdf);
        }

        // Print Z cross-section to verify side walls
        println!("Z cross-section at center_i={}, j=8:", center_i);
        for k in 0..depth {
            let solid = sim.grid.is_solid(center_i, 8, k);
            let s = if solid { "#" } else { "." };
            print!("{}", s);
        }
        println!(" (k=0..{})", depth);

        // Find the actual floor height at center (highest j that's solid in the channel)
        let mut floor_j = 0;
        for j in 0..height {
            if sim.grid.is_solid(center_i, j, center_k) {
                floor_j = j;
            }
        }
        println!("Floor at center: j={} (y={:.3}m)", floor_j, floor_j as f32 * cell_size);

        // Spawn multiple particles above the floor
        let particle_start_y = (floor_j as f32 + 4.0) * cell_size; // 4 cells above floor
        let num_particles = 40;
        println!("Spawning {} particles at Y={:.3}m, center_k={}", num_particles, particle_start_y, center_k);

        for i in 0..num_particles {
            // Spread particles along the center line of the sluice
            let t = i as f32 / num_particles as f32;
            // Only use the middle 60% of sluice length to avoid outlet issues
            let x = (center_i as f32 - 16.0 + t * 32.0) * cell_size;
            let z = center_k as f32 * cell_size; // Center of sluice width
            let y = particle_start_y + (i % 3) as f32 * cell_size * 0.5; // Slight Y variation

            let pos = Vec3::new(x, y, z);
            sim.spawn_particle_with_velocity(pos, Vec3::ZERO);
        }
        assert_eq!(sim.particles.list.len(), num_particles, "Should have {} particles", num_particles);

        // Run simulation for many steps
        let dt = 0.008;
        let steps = 200;
        let floor_y = floor_j as f32 * cell_size;
        let mut min_y_seen = f32::MAX;
        let mut min_y_in_channel = f32::MAX;

        for step in 0..steps {
            sim.update(dt);

            if sim.particles.list.is_empty() {
                println!("All particles removed at step {}!", step);
                println!("  Floor Y: {:.4}", floor_y);
                panic!("All particles fell through floor and were removed!");
            }

            // Track minimum Y across all particles
            // Account for angled floor - floor height varies with X position
            let slope_rad = sluice.slope_deg.to_radians();
            let center_x = sluice.position.x;
            let half_len = sluice.length / 2.0;
            let half_wid = sluice.width / 2.0;
            let center_z = sluice.position.z;

            for p in &sim.particles.list {
                min_y_seen = min_y_seen.min(p.position.y);

                // Only check floor penetration for particles INSIDE the channel
                let dx = p.position.x - center_x;
                let dz = (p.position.z - center_z).abs();
                let in_channel_x = dx >= -half_len && dx <= half_len;
                let in_channel_z = dz <= half_wid;

                if !in_channel_x || !in_channel_z {
                    continue; // Skip particles outside channel
                }

                min_y_in_channel = min_y_in_channel.min(p.position.y);

                // Calculate floor Y at this particle's X position
                let local_floor_y = sluice.position.y - dx * slope_rad.tan();

                // Check each particle hasn't fallen through its local floor
                if p.position.y < local_floor_y - cell_size * 2.0 {
                    println!(
                        "Step {}: particle at ({:.3},{:.3},{:.3}) fell below floor at y={:.4}",
                        step, p.position.x, p.position.y, p.position.z, local_floor_y
                    );
                    panic!(
                        "Particle fell through sluice floor! y={:.4} < local_floor={:.4} - 2*cell (step {})",
                        p.position.y, local_floor_y, step
                    );
                }
            }

            // Print progress every 20 steps
            if step % 20 == 0 {
                println!(
                    "Step {:3}: {} particles, min_y={:.3}, floor_y={:.3}",
                    step, sim.particles.list.len(), min_y_seen, floor_y
                );
            }
        }

        let final_count = sim.particles.list.len();
        println!("After {} steps:", steps);
        println!("  Final particle count: {}", final_count);
        println!("  Min Y seen (any): {:.4}", min_y_seen);
        println!("  Min Y in channel: {:.4}", min_y_in_channel);
        println!("  Floor Y at center: {:.4}", floor_y);
        println!("Test passed: {} particles remain, none fell through their local floor", final_count);
    }

    #[test]
    fn test_gutter_outlet_chute_visualization() {
        // Create a gutter matching the real simulation setup
        let cell_size = 0.025;
        let margin_cells = 4;
        let margin = margin_cells as f32 * cell_size;

        let gutter = GutterPiece {
            id: 0,
            position: Vec3::new(margin + 0.5, margin, margin + 0.15), // Similar to real setup
            rotation: Rotation::R0,
            angle_deg: 10.0,
            length: 1.0,
            width: 0.3,
            end_width: 0.3,
            wall_height: 0.1,
        };

        let width = ((gutter.length + margin * 2.0) / cell_size).ceil() as usize;
        let height = ((0.4 + margin * 2.0) / cell_size).ceil() as usize;
        let depth = ((gutter.max_width() + margin * 2.0) / cell_size).ceil() as usize;

        println!("Grid dimensions: {}x{}x{}", width, height, depth);
        println!("Gutter length: {}m = {} cells", gutter.length, gutter.length / cell_size);
        println!("Margin: {}m = {} cells", margin, margin_cells);

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);
        MultiGridSim::mark_gutter_solid_cells(&mut sim, &gutter, cell_size);

        let center_i = (gutter.position.x / cell_size).round() as i32;
        let half_len_cells = ((gutter.length / 2.0) / cell_size).ceil() as i32;
        let i_end = (center_i + half_len_cells) as usize;

        println!("\nGutter center_i: {}, i_end (channel end): {}", center_i, i_end);
        println!("Grid width: {} cells", width);
        println!("Outlet chute should extend from i={} to i={}", i_end, width);

        // Print the center k slice to see floor profile
        let center_k = (gutter.position.z / cell_size).round() as usize;
        println!("\nCenter K slice (k={}):", center_k);
        print_k_slice(&sim, center_k);

        // Also print a J slice at floor level to see side wall extent
        let floor_j = 2; // Near outlet floor level
        println!("\nFloor-level J slice (j={}):", floor_j);
        print_j_slice(&sim, floor_j);

        // Specifically check the outlet region (last 10 columns)
        println!("\n=== Outlet region check ===");
        let outlet_check_start = (width - 10).max(0);
        for i in outlet_check_start..width {
            let mut floor_j = None;
            for j in (0..height).rev() {
                if sim.grid.is_solid(i, j, center_k) {
                    floor_j = Some(j);
                    break;
                }
            }
            let marker = if i >= i_end { "*CHUTE*" } else { "channel" };
            println!("  i={}: floor_j={:?} {}", i, floor_j, marker);
        }

        // Verify that floor extends to the end of the grid
        let last_i = width - 1;
        let mut has_floor_at_end = false;
        for j in 0..height {
            if sim.grid.is_solid(last_i, j, center_k) {
                has_floor_at_end = true;
                println!("\nFloor found at grid edge (i={}, j={}, k={})", last_i, j, center_k);
                break;
            }
        }

        assert!(has_floor_at_end, "Outlet chute floor should extend to grid edge (i={})", last_i);
    }

    #[test]
    fn test_angled_gutter_sdf_profile() {
        // Test SDF values along the length of an angled gutter
        // to verify the floor collision will work correctly
        let cell_size = 0.025;
        let width = 60;
        let height = 30;
        let depth = 20;

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);

        // Angled gutter (10 degrees, like the real simulation)
        let margin = cell_size * 4.0;
        let gutter = GutterPiece {
            id: 0,
            position: Vec3::new(
                margin + 0.5,  // 1m gutter centered
                margin + 0.15, // Base height with margin
                margin + 0.15, // Center in Z
            ),
            rotation: Rotation::R0,
            angle_deg: 10.0,
            length: 1.0,
            width: 0.3,
            end_width: 0.3,
            wall_height: 0.15,
        };

        MultiGridSim::mark_gutter_solid_cells(&mut sim, &gutter, cell_size);
        sim.grid.compute_sdf();

        let center_k = (gutter.position.z / cell_size).round() as usize;
        let center_i = (gutter.position.x / cell_size).round() as usize;
        let half_len_cells = ((gutter.length / 2.0) / cell_size).ceil() as usize;

        // Calculate expected floor heights
        let angle_rad = gutter.angle_deg.to_radians();
        let total_drop = gutter.length * angle_rad.tan();
        let half_drop = total_drop / 2.0;

        println!("=== Angled Gutter SDF Profile ===");
        println!("Gutter: angle={}°, length={}m, total_drop={:.3}m", gutter.angle_deg, gutter.length, total_drop);
        println!("Center: i={}, k={}", center_i, center_k);
        println!("Half length cells: {}", half_len_cells);
        println!("");

        // Sample at inlet, 1/4, 1/2, 3/4, outlet positions
        let sample_positions = [
            ("inlet", center_i as i32 - half_len_cells as i32),
            ("1/4", center_i as i32 - half_len_cells as i32 / 2),
            ("center", center_i as i32),
            ("3/4", center_i as i32 + half_len_cells as i32 / 2),
            ("outlet", center_i as i32 + half_len_cells as i32),
        ];

        for (name, i) in sample_positions.iter() {
            if *i < 0 || *i >= width as i32 {
                println!("{}: out of bounds (i={})", name, i);
                continue;
            }

            let i_u = *i as usize;

            // Calculate expected mesh floor height at this position
            let dx = (*i as f32 - center_i as f32) * cell_size;
            let t = (dx / gutter.length + 0.5).clamp(0.0, 1.0); // 0 at inlet, 1 at outlet
            let mesh_floor_y = gutter.position.y + half_drop - t * total_drop;

            // Find actual solid floor
            let mut solid_floor_j = None;
            for j in (0..height).rev() {
                if sim.grid.is_solid(i_u, j, center_k) {
                    solid_floor_j = Some(j);
                    break;
                }
            }

            println!("{} (i={}, t={:.2}):", name, i, t);
            println!("  Mesh floor Y: {:.4}m", mesh_floor_y);
            if let Some(floor_j) = solid_floor_j {
                let solid_top_y = (floor_j + 1) as f32 * cell_size;
                println!("  Solid top (j={}) Y: {:.4}m", floor_j, solid_top_y);
                println!("  Gap (solid_top - mesh): {:.4}m ({:.1} cells)", solid_top_y - mesh_floor_y, (solid_top_y - mesh_floor_y) / cell_size);
            } else {
                println!("  NO SOLID FLOOR FOUND!");
            }

            // Sample SDF at mesh floor level
            let test_y = mesh_floor_y + cell_size * 0.5; // Just above mesh surface
            let test_pos = Vec3::new(*i as f32 * cell_size + cell_size * 0.5, test_y, center_k as f32 * cell_size + cell_size * 0.5);
            let sdf = sim.grid.sample_sdf(test_pos);
            println!("  SDF at Y={:.4}m: {:.4}", test_y, sdf);

            if sdf >= 0.0 {
                println!("  ⚠️  WARNING: Positive SDF above mesh floor - particles will fall through!");
            }
            println!("");
        }

        // Now verify particles don't fall through by checking SDF just above mesh floor
        println!("\n=== Particle Fall-Through Test ===");
        let mut any_fallthrough = false;
        for i in (center_i - half_len_cells)..=(center_i + half_len_cells) {
            let dx = (i as f32 - center_i as f32) * cell_size;
            let t = (dx / gutter.length + 0.5).clamp(0.0, 1.0);
            let mesh_floor_y = gutter.position.y + half_drop - t * total_drop;

            // Test at 0.5 cells above mesh floor
            let test_y = mesh_floor_y + cell_size * 0.5;
            let test_pos = Vec3::new(i as f32 * cell_size + cell_size * 0.5, test_y, center_k as f32 * cell_size + cell_size * 0.5);
            let sdf = sim.grid.sample_sdf(test_pos);

            if sdf > cell_size * 0.1 { // More than 0.1 cells positive = in air
                println!("FALLTHROUGH at i={}: SDF={:.4} at y={:.4} (mesh floor={:.4})", i, sdf, test_y, mesh_floor_y);
                any_fallthrough = true;
            }
        }

        if !any_fallthrough {
            println!("✓ No fall-through detected - all positions above mesh have negative or near-zero SDF");
        }

        assert!(!any_fallthrough, "Found fall-through positions!");
    }
}

//! Multi-grid simulation for washplant pieces.

use crate::editor::{GutterPiece, Rotation, ShakerDeckPiece, SluicePiece};
use crate::equipment_geometry::{BoxConfig, BoxGeometryBuilder};
use crate::gpu::flip_3d::GpuFlip3D;
use crate::sluice_geometry::SluiceVertex;
use glam::{Mat3, Vec3};
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, SdfParams};
use sim3d::test_geometry::{TestBox, TestFloor, TestSdfGenerator};
use sim3d::FlipSimulation3D;
use wgpu::util::DeviceExt;

const SIM_CELL_SIZE: f32 = 0.025; // 2.5cm cells
const SIM_PRESSURE_ITERS: usize = 120; // Increased for better density projection convergence
const SIM_GRAVITY: f32 = -9.8;

const DEM_CLUMP_RADIUS: f32 = 0.008; // 8mm clumps
const DEM_GOLD_DENSITY: f32 = 19300.0; // kg/m^3
const DEM_SAND_DENSITY: f32 = 2650.0; // kg/m^3
const DEM_WATER_DENSITY: f32 = 1000.0; // kg/m^3
const DEM_DRAG_COEFF: f32 = 5.0; // Water drag coefficient

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

// ============================================================================
// Multi-Grid Simulation Types
// ============================================================================

/// Which type of piece this simulation belongs to
#[derive(Clone, Copy, Debug)]
pub enum PieceKind {
    Gutter(usize),     // index into layout.gutters
    Sluice(usize),     // index into layout.sluices
    ShakerDeck(usize), // index into layout.shaker_decks
    TestBox,           // equipment_geometry test box
}

/// Per-piece simulation grid
pub struct PieceSimulation {
    pub kind: PieceKind,

    // Grid configuration
    pub grid_offset: Vec3, // World position of grid origin
    pub grid_dims: (usize, usize, usize),
    pub cell_size: f32,

    // Simulation state
    pub sim: FlipSimulation3D,
    pub gpu_flip: Option<GpuFlip3D>,
    pub sdf_buffer: Option<wgpu::Buffer>,

    // Particle data buffers
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub affine_vels: Vec<Mat3>,
    pub densities: Vec<f32>,
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
pub struct MultiGridSim {
    pub pieces: Vec<PieceSimulation>,
    transfers: Vec<PieceTransfer>,
    frame: u32,

    // DEM simulation (global, not per-piece)
    pub dem_sim: ClusterSimulation3D,
    pub gpu_dem: Option<crate::gpu::dem_3d::GpuDem3D>,
    pub gold_template_idx: usize,
    pub sand_template_idx: usize,
    pub gpu_test_sdf_buffer: Option<wgpu::Buffer>,

    // Test SDF for isolated physics tests (used instead of piece SDFs when set)
    test_sdf: Option<Vec<f32>>,
    test_sdf_dims: (usize, usize, usize),
    test_sdf_cell_size: f32,
    test_sdf_offset: Vec3,
    // Renderable mesh for test geometry
    pub test_mesh: Option<(Vec<SluiceVertex>, Vec<u32>)>,
}

impl MultiGridSim {
    pub fn new() -> Self {
        // Create DEM simulation with large bounds (covers all pieces)
        let mut dem_sim =
            ClusterSimulation3D::new(Vec3::new(-10.0, -2.0, -10.0), Vec3::new(20.0, 10.0, 20.0));

        // Reduce stiffness for stability with small particles
        // Default 6000 N/m causes particles to explode on collision
        dem_sim.normal_stiffness = 100.0;
        dem_sim.tangential_stiffness = 50.0;
        dem_sim.restitution = 0.1; // Lower bounce

        // Create gold template (heavy, ~8mm clumps)
        // Gold: 19300 kg/m^3, water: 1000 kg/m^3
        // Volume of 8mm sphere ~ 2.68e-7 m^3, mass ~ 5.17g for gold
        let gold_particle_mass =
            DEM_GOLD_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * DEM_CLUMP_RADIUS.powi(3);
        let gold_template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 5,
                seed: 42,
                style: sim3d::clump::IrregularStyle3D::Round,
            },
            DEM_CLUMP_RADIUS,
            gold_particle_mass,
        );
        let gold_template_idx = dem_sim.add_template(gold_template);

        // Create sand/gangue template (lighter)
        let sand_particle_mass =
            DEM_SAND_DENSITY * (4.0 / 3.0) * std::f32::consts::PI * DEM_CLUMP_RADIUS.powi(3);
        let sand_template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 4,
                seed: 123,
                style: sim3d::clump::IrregularStyle3D::Sharp,
            },
            DEM_CLUMP_RADIUS,
            sand_particle_mass,
        );
        let sand_template_idx = dem_sim.add_template(sand_template);

        Self {
            pieces: Vec::new(),
            transfers: Vec::new(),
            frame: 0,
            dem_sim,
            gpu_dem: None,
            gold_template_idx,
            sand_template_idx,
            test_sdf: None,
            test_sdf_dims: (0, 0, 0),
            test_sdf_cell_size: SIM_CELL_SIZE,
            test_sdf_offset: Vec3::ZERO,
            gpu_test_sdf_buffer: None,
            test_mesh: None,
        }
    }

    /// Initialize GPU DEM simulation
    pub fn init_gpu_dem(
        &mut self,
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
    ) {
        let mut gpu_dem =
            crate::gpu::dem_3d::GpuDem3D::new(device.clone(), queue.clone(), 50000, 10, 100000);

        // Sync templates
        for template in &self.dem_sim.templates {
            gpu_dem.add_template(template.clone());
        }

        // Sync existing clumps
        for clump in &self.dem_sim.clumps {
            gpu_dem.spawn_clump(clump.template_idx as u32, clump.position, clump.velocity);
        }

        // Create SDF buffers for existing pieces
        for piece in &mut self.pieces {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Piece SDF Buffer"),
                contents: bytemuck::cast_slice(&piece.sim.grid.sdf),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            piece.sdf_buffer = Some(buffer);
        }

        if let Some(test_sdf) = &self.test_sdf {
            println!(
                "Init GPU DEM: Test SDF len={}, sample[0]={}",
                test_sdf.len(),
                test_sdf[0]
            );
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test SDF Buffer"),
                contents: bytemuck::cast_slice(test_sdf),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            self.gpu_test_sdf_buffer = Some(buffer);
        }

        // Initialize params
        gpu_dem.stiffness = self.dem_sim.normal_stiffness;
        gpu_dem.damping = 4.0; // Critical damping ~4.5 for sand mass 0.005kg

        self.gpu_dem = Some(gpu_dem);
    }

    /// Set up test SDF using TestFloor geometry for isolated floor collision tests
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

    /// Set up test SDF using TestBox geometry for isolated box collision tests
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

    /// Clear the test SDF (return to using piece SDFs)
    pub fn clear_test_sdf(&mut self) {
        self.test_sdf = None;
        self.test_mesh = None;
    }

    /// Step the DEM simulation with the appropriate SDF.
    /// Uses test_sdf if set (for isolated tests), otherwise uses first piece's SDF.
    pub fn step_dem(&mut self, dt: f32) {
        // Prefer test SDF if set (isolated DEM tests)
        if let Some(ref sdf) = self.test_sdf {
            let (gw, gh, gd) = self.test_sdf_dims;
            let sdf_params = sim3d::clump::SdfParams {
                sdf,
                grid_width: gw,
                grid_height: gh,
                grid_depth: gd,
                cell_size: self.test_sdf_cell_size,
                grid_offset: self.test_sdf_offset,
            };
            self.dem_sim.step_with_sdf(dt, &sdf_params);
        } else if !self.pieces.is_empty() {
            // Fall back to first piece's SDF
            let piece = &self.pieces[0];
            let (gw, gh, gd) = piece.grid_dims;
            let sdf_params = sim3d::clump::SdfParams {
                sdf: &piece.sim.grid.sdf,
                grid_width: gw,
                grid_height: gh,
                grid_depth: gd,
                cell_size: piece.cell_size,
                grid_offset: piece.grid_offset,
            };
            self.dem_sim.step_with_sdf(dt, &sdf_params);
        } else {
            // No SDF available - step without collision
            self.dem_sim.step(dt);
        }
    }

    /// Add a gutter piece simulation
    pub fn add_gutter(
        &mut self,
        device: &wgpu::Device,
        gutter: &GutterPiece,
        gutter_idx: usize,
    ) -> usize {
        self.add_gutter_internal(Some(device), gutter, gutter_idx)
    }

    /// Add a gutter piece simulation without GPU (headless tests).
    pub fn add_gutter_headless(&mut self, gutter: &GutterPiece, gutter_idx: usize) -> usize {
        self.add_gutter_internal(None, gutter, gutter_idx)
    }

    fn add_gutter_internal(
        &mut self,
        device: Option<&wgpu::Device>,
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

        let gpu_flip = device.map(|device| {
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
            gpu_flip
        });

        let sdf_buffer = device.map(|device| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gutter SDF Buffer"),
                contents: bytemuck::cast_slice(&sim.grid.sdf),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        });

        let idx = self.pieces.len();
        self.pieces.push(PieceSimulation {
            kind: PieceKind::Gutter(gutter_idx),
            grid_offset,
            grid_dims: (width, height, depth),
            cell_size,
            sim,
            gpu_flip,
            sdf_buffer,
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
        });

        idx
    }

    /// Add a sluice piece simulation
    pub fn add_sluice(
        &mut self,
        device: &wgpu::Device,
        sluice: &SluicePiece,
        sluice_idx: usize,
    ) -> usize {
        self.add_sluice_internal(Some(device), sluice, sluice_idx)
    }

    /// Add a sluice piece simulation without GPU (headless tests).
    pub fn add_sluice_headless(&mut self, sluice: &SluicePiece, sluice_idx: usize) -> usize {
        self.add_sluice_internal(None, sluice, sluice_idx)
    }

    fn add_sluice_internal(
        &mut self,
        device: Option<&wgpu::Device>,
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

        let gpu_flip = device.map(|device| {
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
            gpu_flip
        });

        let sdf_buffer = device.map(|device| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sluice SDF Buffer"),
                contents: bytemuck::cast_slice(&sim.grid.sdf),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        });

        let idx = self.pieces.len();
        self.pieces.push(PieceSimulation {
            kind: PieceKind::Sluice(sluice_idx),
            grid_offset,
            grid_dims: (width, height, depth),
            cell_size,
            sim,
            gpu_flip,
            sdf_buffer,
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
        });

        idx
    }

    /// Add a shaker deck piece simulation
    pub fn add_shaker_deck(
        &mut self,
        device: &wgpu::Device,
        deck: &ShakerDeckPiece,
        deck_idx: usize,
    ) -> usize {
        self.add_shaker_deck_internal(Some(device), deck, deck_idx)
    }

    /// Add a shaker deck piece simulation without GPU (headless tests).
    pub fn add_shaker_deck_headless(&mut self, deck: &ShakerDeckPiece, deck_idx: usize) -> usize {
        self.add_shaker_deck_internal(None, deck, deck_idx)
    }

    fn add_shaker_deck_internal(
        &mut self,
        device: Option<&wgpu::Device>,
        deck: &ShakerDeckPiece,
        deck_idx: usize,
    ) -> usize {
        let cell_size = SIM_CELL_SIZE;
        let margin = cell_size * 4.0;
        let max_width = deck.max_width();

        let width = ((deck.length + margin * 2.0) / cell_size).ceil() as usize;
        // Height includes wall height + headroom for particles
        let height = ((deck.wall_height + margin + 0.3) / cell_size).ceil() as usize;
        let depth = ((max_width + margin * 2.0) / cell_size).ceil() as usize;

        let width = width.clamp(10, 60);
        let height = height.clamp(10, 40);
        let depth = depth.clamp(10, 40);

        // Grid origin based on rotation
        let (dir_x, dir_z) = match deck.rotation {
            Rotation::R0 => (1.0f32, 0.0f32),
            Rotation::R90 => (0.0, 1.0),
            Rotation::R180 => (-1.0, 0.0),
            Rotation::R270 => (0.0, -1.0),
        };

        // Grid offset accounting for rotation
        let grid_offset = Vec3::new(
            deck.position.x
                - deck.length / 2.0 * dir_x.abs()
                - max_width / 2.0 * dir_z.abs()
                - margin,
            deck.position.y - margin,
            deck.position.z
                - deck.length / 2.0 * dir_z.abs()
                - max_width / 2.0 * dir_x.abs()
                - margin,
        );

        let mut sim = FlipSimulation3D::new(width, height, depth, cell_size);
        sim.pressure_iterations = SIM_PRESSURE_ITERS;

        // Mark shaker deck solid cells (walls only - grid is porous)
        let deck_local = ShakerDeckPiece {
            id: 0,
            position: Vec3::new(
                margin + deck.length / 2.0,
                margin,
                margin + deck.max_width() / 2.0,
            ),
            rotation: Rotation::R0,
            length: deck.length,
            width: deck.width,
            end_width: deck.end_width,
            tilt_deg: deck.tilt_deg,
            hole_size: deck.hole_size,
            wall_height: deck.wall_height,
            bar_thickness: deck.bar_thickness,
        };
        Self::mark_shaker_deck_solid_cells(&mut sim, &deck_local, cell_size);

        sim.grid.compute_sdf();

        let gpu_flip = device.map(|device| {
            let mut gpu_flip = GpuFlip3D::new(
                device,
                width as u32,
                height as u32,
                depth as u32,
                cell_size,
                20000,
            );
            // Shaker deck: +X (outlet) open for oversize material
            // -Y (bottom) open for fines falling through the perforated deck
            // Bit 1 (2) = +X, Bit 2 (4) = -Y -> combined = 6
            gpu_flip.open_boundaries = 6; // +X and -Y open
            gpu_flip
        });

        let sdf_buffer = device.map(|device| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shaker Deck SDF Buffer"),
                contents: bytemuck::cast_slice(&sim.grid.sdf),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        });

        let idx = self.pieces.len();
        self.pieces.push(PieceSimulation {
            kind: PieceKind::ShakerDeck(deck_idx),
            grid_offset,
            grid_dims: (width, height, depth),
            cell_size,
            sim,
            gpu_flip,
            sdf_buffer,
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
        });

        idx
    }

    /// Add a closed box simulation using equipment_geometry (GPU).
    pub fn add_equipment_box(
        &mut self,
        device: &wgpu::Device,
        grid_offset: Vec3,
        grid_width: usize,
        grid_height: usize,
        grid_depth: usize,
        wall_thickness: usize,
    ) -> usize {
        self.add_equipment_box_internal(
            Some(device),
            grid_offset,
            grid_width,
            grid_height,
            grid_depth,
            wall_thickness,
        )
    }

    /// Add a closed box simulation using equipment_geometry (headless).
    pub fn add_equipment_box_headless(
        &mut self,
        grid_offset: Vec3,
        grid_width: usize,
        grid_height: usize,
        grid_depth: usize,
        wall_thickness: usize,
    ) -> usize {
        self.add_equipment_box_internal(
            None,
            grid_offset,
            grid_width,
            grid_height,
            grid_depth,
            wall_thickness,
        )
    }

    fn add_equipment_box_internal(
        &mut self,
        device: Option<&wgpu::Device>,
        grid_offset: Vec3,
        grid_width: usize,
        grid_height: usize,
        grid_depth: usize,
        wall_thickness: usize,
    ) -> usize {
        let cell_size = SIM_CELL_SIZE;
        let mut sim = FlipSimulation3D::new(grid_width, grid_height, grid_depth, cell_size);
        sim.pressure_iterations = SIM_PRESSURE_ITERS;

        let mut box_builder = BoxGeometryBuilder::new(BoxConfig {
            grid_width,
            grid_height,
            grid_depth,
            cell_size,
            wall_thickness,
            ..BoxConfig::default()
        });

        for (i, j, k) in box_builder.solid_cells() {
            sim.grid.set_solid(i, j, k);
        }
        sim.grid.compute_sdf();

        let config = box_builder.config().clone();
        box_builder.build_mesh(|i, j, k| config.is_solid(i, j, k));

        let mut vertices = Vec::with_capacity(box_builder.vertices().len());
        for v in box_builder.vertices() {
            let mut v = *v;
            v.position[0] += grid_offset.x;
            v.position[1] += grid_offset.y;
            v.position[2] += grid_offset.z;
            vertices.push(v);
        }

        self.test_mesh = Some((vertices, box_builder.indices().to_vec()));

        let gpu_flip = device.map(|device| {
            let mut gpu_flip = GpuFlip3D::new(
                device,
                grid_width as u32,
                grid_height as u32,
                grid_depth as u32,
                cell_size,
                200_000,
            );
            gpu_flip.open_boundaries = 0; // closed box
            gpu_flip
        });

        let sdf_buffer = device.map(|device| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Equipment Box SDF Buffer"),
                contents: bytemuck::cast_slice(&sim.grid.sdf),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        });

        let idx = self.pieces.len();
        self.pieces.push(PieceSimulation {
            kind: PieceKind::TestBox,
            grid_offset,
            grid_dims: (grid_width, grid_height, grid_depth),
            cell_size,
            sim,
            gpu_flip,
            sdf_buffer,
            positions: Vec::new(),
            velocities: Vec::new(),
            affine_vels: Vec::new(),
            densities: Vec::new(),
        });

        idx
    }

    /// Mark gutter solid cells using cell-index approach (like friction_sluice)
    /// Supports variable width gutters (funnel effect) via width_at()
    pub fn mark_gutter_solid_cells(
        sim: &mut FlipSimulation3D,
        gutter: &GutterPiece,
        cell_size: f32,
    ) {
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
        let _floor_j_left = base_j + half_drop_cells; // Inlet is higher
        let _floor_j_right = base_j - half_drop_cells; // Outlet is lower

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
                let outlet_floor_j =
                    ((gutter.position.y - total_drop / 2.0) / cell_size).floor() as i32;

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
    pub fn mark_sluice_solid_cells(
        sim: &mut FlipSimulation3D,
        sluice: &SluicePiece,
        cell_size: f32,
    ) {
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
                    let is_channel_floor =
                        j_i <= effective_floor_j && in_channel_length && in_channel_width;

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

    /// Mark shaker deck solid cells - walls only (grid is porous)
    /// Supports variable width (funnel effect)
    /// Marks side walls, back wall, and grate bars as solid
    pub fn mark_shaker_deck_solid_cells(
        sim: &mut FlipSimulation3D,
        deck: &ShakerDeckPiece,
        cell_size: f32,
    ) {
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;

        // Deck position gives center in X/Z and base height in Y
        let center_i = (deck.position.x / cell_size).round() as i32;
        let center_k = (deck.position.z / cell_size).round() as i32;

        // Channel dimensions in cells
        let half_len_cells = ((deck.length / 2.0) / cell_size).ceil() as i32;
        // Variable width: inlet vs outlet
        let inlet_half_wid_cells = ((deck.width / 2.0) / cell_size).ceil() as i32;
        let outlet_half_wid_cells = ((deck.end_width / 2.0) / cell_size).ceil() as i32;

        // Floor height drop due to tilt
        let tilt_rad = deck.tilt_deg.to_radians();
        let total_drop = deck.length * tilt_rad.tan();

        // Wall parameters - deck walls are solid
        let wall_height_cells = ((deck.wall_height / cell_size).ceil() as i32).max(4);
        let wall_thick_cells = 2_i32;

        // Grate parameters - bars and holes
        let bar_spacing = deck.hole_size + deck.bar_thickness;
        let bar_thick_cells = (deck.bar_thickness / cell_size).ceil() as i32;
        let hole_cells = (deck.hole_size / cell_size).floor() as i32;
        let pattern_cells = bar_thick_cells + hole_cells;

        // Cross bar spacing (every 3rd bar_spacing)
        let cross_spacing = bar_spacing * 3.0;
        let cross_pattern_cells = (cross_spacing / cell_size).round() as i32;

        // Channel bounds in i (length direction)
        let i_start = (center_i - half_len_cells).max(0) as usize;
        let i_end = ((center_i + half_len_cells) as usize).min(width);

        for i in 0..width {
            let i_i = i as i32;

            // Calculate position along deck (0.0 = inlet, 1.0 = outlet)
            let t = if i_i <= center_i - half_len_cells {
                0.0
            } else if i_i >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };

            // Variable width at this position (funnel effect)
            let local_half_wid_cells = (inlet_half_wid_cells as f32
                + (outlet_half_wid_cells - inlet_half_wid_cells) as f32 * t)
                as i32;

            // Compute floor_j from mesh floor position
            let mesh_floor_y = deck.position.y + (total_drop / 2.0) - t * total_drop;
            let floor_j = (mesh_floor_y / cell_size).floor() as i32;
            let wall_top_j = floor_j + wall_height_cells;

            let in_channel_length = i >= i_start && i < i_end;

            // Check if this X position is on a cross bar
            let dist_from_start = i_i - (center_i - half_len_cells);
            let on_cross_bar = if cross_pattern_cells > 0 {
                (dist_from_start % cross_pattern_cells) < bar_thick_cells
            } else {
                false
            };

            for k in 0..depth {
                let k_i = k as i32;

                // Check if this Z position is on a longitudinal bar
                let dist_from_center_z = (k_i - center_k).abs();
                let on_long_bar = if pattern_cells > 0 {
                    (dist_from_center_z % pattern_cells) < bar_thick_cells
                } else {
                    true // If no pattern, treat as solid
                };

                // Within channel width?
                let in_channel_width = k_i >= (center_k - local_half_wid_cells)
                    && k_i < (center_k + local_half_wid_cells);

                for j in 0..height {
                    let j_i = j as i32;

                    // Side walls - mark cells outside channel width as solid
                    let at_left_wall = k_i < (center_k - local_half_wid_cells);
                    let at_right_wall = k_i >= (center_k + local_half_wid_cells);
                    let is_side_wall = (at_left_wall || at_right_wall)
                        && in_channel_length
                        && j_i <= wall_top_j
                        && j_i >= floor_j;

                    // Back wall at inlet
                    let at_back = i_i >= (center_i - half_len_cells - wall_thick_cells)
                        && i_i < (center_i - half_len_cells);
                    let is_back_wall = at_back
                        && j_i <= wall_top_j
                        && j_i >= floor_j
                        && k_i >= (center_k - inlet_half_wid_cells - wall_thick_cells)
                        && k_i < (center_k + inlet_half_wid_cells + wall_thick_cells);

                    // Grate bars - at floor level, where bars exist (not holes)
                    // Bars are solid, holes let particles through
                    // Make bars 4 cells thick to prevent tunneling
                    let at_floor_level = j_i >= floor_j && j_i <= floor_j + 3;
                    let is_grate_bar = in_channel_length
                        && in_channel_width
                        && at_floor_level
                        && (on_long_bar || on_cross_bar);

                    if is_side_wall || is_back_wall || is_grate_bar {
                        sim.grid.set_solid(i, j, k);
                    }
                }
            }
        }
    }

    /// Add a transfer between pieces
    pub fn add_transfer(
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

    /// Emit particles with specific density into a piece
    pub fn emit_into_piece_with_density(
        &mut self,
        piece_idx: usize,
        world_pos: Vec3,
        velocity: Vec3,
        density: f32,
        count: usize,
    ) {
        if piece_idx >= self.pieces.len() {
            return;
        }

        let piece = &mut self.pieces[piece_idx];
        let sim_pos = world_pos - piece.grid_offset;

        for _ in 0..count {
            let spread = 0.01;
            let offset = Vec3::new(
                (rand_float() - 0.5) * spread,
                (rand_float() - 0.5) * spread,
                (rand_float() - 0.5) * spread,
            );

            piece.positions.push(sim_pos + offset);
            piece.velocities.push(velocity);
            piece.affine_vels.push(Mat3::ZERO);
            piece.densities.push(density);

            piece.sim.particles.spawn(sim_pos + offset, velocity);
        }
    }

    /// Emit water particles (density = 1.0)
    pub fn emit_into_piece(
        &mut self,
        piece_idx: usize,
        world_pos: Vec3,
        velocity: Vec3,
        count: usize,
    ) {
        self.emit_into_piece_with_density(piece_idx, world_pos, velocity, 1.0, count);
    }

    /// Step all piece simulations using GPU FLIP.
    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        self.step_internal(Some((device, queue)), dt);
    }

    /// Step all piece simulations on CPU (headless tests).
    pub fn step_headless(&mut self, dt: f32) {
        self.step_internal(None, dt);
    }

    fn step_internal(&mut self, gpu: Option<(&wgpu::Device, &wgpu::Queue)>, dt: f32) {
        // Step each piece
        for piece in &mut self.pieces {
            if piece.positions.is_empty() && piece.sim.particles.is_empty() {
                continue;
            }

            match (gpu, piece.gpu_flip.as_mut()) {
                (Some((device, queue)), Some(gpu_flip)) => {
                    // Create cell types (solid + fluid markers for BC and pressure)
                    let (gw, gh, gd) = piece.grid_dims;
                    let cell_count = gw * gh * gd;
                    let mut cell_types = vec![0u32; cell_count];

                    for (idx, is_solid) in piece.sim.grid.solid.iter().enumerate() {
                        if *is_solid {
                            cell_types[idx] = 2; // SOLID
                        }
                    }

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
                            if cell_types[idx] != 2 {
                                cell_types[idx] = 1; // FLUID
                            }
                        }
                    }
                    let sdf = Some(piece.sim.grid.sdf.clone());

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
                _ => {
                    // CPU FLIP update (headless)
                    piece.sim.update(dt);

                    piece.positions.clear();
                    piece.velocities.clear();
                    piece.affine_vels.clear();
                    piece.densities.clear();

                    for particle in &piece.sim.particles.list {
                        piece.positions.push(particle.position);
                        piece.velocities.push(particle.velocity);
                        piece.affine_vels.push(particle.affine_velocity);
                        piece.densities.push(particle.density);
                    }
                }
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
        let mut transfer_data: Vec<Vec<TransferParticle>> = vec![Vec::new(); self.transfers.len()];

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

        // Step DEM simulation
        if let Some(gpu_dem) = &mut self.gpu_dem {
            let (device, queue) = gpu.expect("GPU DEM requires device/queue");
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("DEM Step"),
            });

            // Sub-stepping for stability (e.g., 10 sub-steps per frame)
            let sub_steps = 10;
            let sub_dt = dt / sub_steps as f32;

            for _ in 0..sub_steps {
                // 1. Particle-Particle collision and hashing
                gpu_dem.prepare_step(&mut encoder, sub_dt);

                // 2. SDF collision for each piece
                for piece in &self.pieces {
                    if let Some(sdf_buffer) = &piece.sdf_buffer {
                        let (gw, gh, gd) = piece.grid_dims;
                        let sdf_params = crate::gpu::dem_3d::GpuSdfParams {
                            grid_offset: [
                                piece.grid_offset.x,
                                piece.grid_offset.y,
                                piece.grid_offset.z,
                                0.0,
                            ],
                            grid_dims: [gw as u32, gh as u32, gd as u32, 0],
                            cell_size: piece.cell_size,
                            pad0: 0.0,
                            pad1: 0.0,
                            pad2: 0.0,
                        };
                        gpu_dem.apply_sdf_collision_pass(&mut encoder, sdf_buffer, &sdf_params);
                    }
                }

                // 2b. SDF collision for test SDF if set
                if let Some(sdf_buffer) = &self.gpu_test_sdf_buffer {
                    let (gw, gh, gd) = self.test_sdf_dims;
                    let sdf_params = crate::gpu::dem_3d::GpuSdfParams {
                        grid_offset: [
                            self.test_sdf_offset.x,
                            self.test_sdf_offset.y,
                            self.test_sdf_offset.z,
                            0.0,
                        ],
                        grid_dims: [gw as u32, gh as u32, gd as u32, 0],
                        cell_size: self.test_sdf_cell_size,
                        pad0: 0.0,
                        pad1: 0.0,
                        pad2: 0.0,
                    };
                    gpu_dem.apply_sdf_collision_pass(&mut encoder, sdf_buffer, &sdf_params);
                }

                // 3. Integration
                gpu_dem.finish_step(&mut encoder);
            }

            queue.submit(Some(encoder.finish()));

            // Perform readback so CPU side (rendering) stays in sync
            let particles = pollster::block_on(gpu_dem.readback(device));
            for (i, p) in particles.iter().enumerate() {
                if i < self.dem_sim.clumps.len() {
                    self.dem_sim.clumps[i].position =
                        Vec3::new(p.position[0], p.position[1], p.position[2]);
                    self.dem_sim.clumps[i].velocity =
                        Vec3::new(p.velocity[0], p.velocity[1], p.velocity[2]);
                    self.dem_sim.clumps[i].angular_velocity = Vec3::new(
                        p.angular_velocity[0],
                        p.angular_velocity[1],
                        p.angular_velocity[2],
                    );
                    self.dem_sim.clumps[i].rotation = glam::Quat::from_xyzw(
                        p.orientation[0],
                        p.orientation[1],
                        p.orientation[2],
                        p.orientation[3],
                    );
                }
            }
        } else if !self.dem_sim.clumps.is_empty() {
            // Apply water-DEM coupling forces (only if fluid pieces exist)
            let has_fluid = self.pieces.iter().any(|piece| !piece.positions.is_empty());
            if has_fluid {
                // Apply water-DEM coupling forces to each clump
                for clump in &mut self.dem_sim.clumps {
                    let template = &self.dem_sim.templates[clump.template_idx];

                    // Buoyancy force: F_b = rho_water * V * g (upward)
                    let particle_volume =
                        (4.0 / 3.0) * std::f32::consts::PI * template.particle_radius.powi(3);
                    let total_volume = particle_volume * template.local_offsets.len() as f32;
                    let buoyancy_force = DEM_WATER_DENSITY * total_volume * 9.81;

                    // Drag force: F_d = 0.5 * C_d * rho_water * A * v^2
                    let area = std::f32::consts::PI * template.bounding_radius.powi(2);
                    let speed = clump.velocity.length();
                    let drag_force = if speed > 0.001 {
                        0.5 * DEM_DRAG_COEFF * DEM_WATER_DENSITY * area * speed * speed
                    } else {
                        0.0
                    };

                    clump.velocity.y += buoyancy_force * dt / template.mass;

                    if speed > 0.001 {
                        let drag_dir = -clump.velocity.normalize();
                        let drag_dv = drag_force * dt / template.mass;
                        clump.velocity += drag_dir * drag_dv.min(speed);
                    }
                }
            }

            // Normal operation: use piece SDFs
            // First, do DEM integration step (forces, velocity, position)
            self.dem_sim.step(dt);

            // Then apply collision response against EACH piece's SDF
            for piece in &self.pieces {
                let (gw, gh, gd) = piece.grid_dims;
                let sdf_params = SdfParams {
                    sdf: &piece.sim.grid.sdf,
                    grid_width: gw,
                    grid_height: gh,
                    grid_depth: gd,
                    cell_size: piece.cell_size,
                    grid_offset: piece.grid_offset,
                };
                self.dem_sim.collision_response_only(dt, &sdf_params, true);
            }

            // Remove clumps that fall too far below
            self.dem_sim.clumps.retain(|c| c.position.y > -2.0);
        }

        self.frame += 1;
    }

    /// Get total particle count across all pieces
    pub fn total_particles(&self) -> usize {
        let piece_count: usize = self.pieces.iter().map(|p| p.positions.len()).sum();
        piece_count + self.dem_sim.clumps.len()
    }

    /// Count occupied grid cells for a piece based on particle positions.
    pub fn occupied_cell_count(&self, piece_idx: usize) -> usize {
        let Some(piece) = self.pieces.get(piece_idx) else {
            return 0;
        };

        let (gw, gh, gd) = piece.grid_dims;
        let cell_count = gw * gh * gd;
        if cell_count == 0 {
            return 0;
        }

        let mut occupied = vec![false; cell_count];
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
                occupied[idx] = true;
            }
        }

        occupied.iter().filter(|&&v| v).count()
    }

    /// Approximate occupied volume (in m^3) from occupied grid cells.
    pub fn occupied_volume(&self, piece_idx: usize) -> f32 {
        let Some(piece) = self.pieces.get(piece_idx) else {
            return 0.0;
        };
        let count = self.occupied_cell_count(piece_idx);
        count as f32 * piece.cell_size.powi(3)
    }

    /// Approximate total occupied volume (in m^3) across all pieces.
    pub fn total_occupied_volume(&self) -> f32 {
        (0..self.pieces.len())
            .map(|idx| self.occupied_volume(idx))
            .sum()
    }
}

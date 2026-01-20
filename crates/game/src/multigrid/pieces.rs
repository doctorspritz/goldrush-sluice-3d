//! Piece management for the multigrid simulation.
//!
//! This module handles adding different equipment pieces to the simulation:
//! - Gutters (with GPU and headless variants)
//! - Sluices (with GPU and headless variants)
//! - Shaker decks (with GPU and headless variants)
//! - Equipment boxes (test geometry)

use crate::editor::{GutterPiece, Rotation, ShakerDeckPiece, SluicePiece};
use crate::equipment_geometry::{BoxConfig, BoxGeometryBuilder, GeometryConfig};
use crate::gpu::flip_3d::GpuFlip3D;
use glam::Vec3;
use sim3d::FlipSimulation3D;
use wgpu::util::DeviceExt;

use super::constants::{SIM_CELL_SIZE, SIM_PRESSURE_ITERS};
use super::types::{MultiGridSim, PieceKind, PieceSimulation};

impl MultiGridSim {
    /// Add a gutter piece simulation.
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

    /// Add a sluice piece simulation.
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

    /// Add a shaker deck piece simulation.
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

    /// Add a transfer between pieces.
    pub fn add_transfer(
        &mut self,
        from_piece: usize,
        to_piece: usize,
        capture_min: Vec3,
        capture_max: Vec3,
        inject_pos: Vec3,
        inject_vel: Vec3,
    ) {
        self.transfers.push(super::types::PieceTransfer {
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
}

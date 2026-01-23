//! Simulation stepping for the multigrid solver.
//!
//! This module contains the main simulation loop:
//! - FLIP fluid simulation stepping (GPU and CPU)
//! - Particle transfer between pieces
//! - DEM simulation stepping with fluid coupling

use glam::{Mat3, Vec3};
use sim3d::clump::SdfParams;

use super::constants::{
    rand_float, DEM_DRAG_COEFF, DEM_WATER_DENSITY, SIM_CELL_SIZE, SIM_GRAVITY,
};
use super::MultiGridSim;

impl MultiGridSim {
    /// Step the DEM simulation with the appropriate SDF.
    /// Uses test_sdf if set (for isolated tests), otherwise uses first piece's SDF.
    pub fn step_dem(&mut self, dt: f32) {
        // Prefer test SDF if set (isolated DEM tests)
        if let Some(ref sdf) = self.test_sdf {
            let (gw, gh, gd) = self.test_sdf_dims;
            let sdf_params = SdfParams {
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
            let sdf_params = SdfParams {
                sdf: &piece.sim.grid.sdf(),
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

    /// Emit particles with specific density into a piece.
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

    /// Emit water particles (density = 1.0).
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

                    for (idx, is_solid) in piece.sim.grid.solid().iter().enumerate() {
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
                    let sdf = Some(piece.sim.grid.sdf());

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
                        self.pressure_iters as u32,
                    );
                }
                _ => {
                    // CPU FLIP update (headless)
                    piece.sim.update(dt);

                    piece.positions.clear();
                    piece.velocities.clear();
                    piece.affine_vels.clear();
                    piece.densities.clear();

                    for particle in piece.sim.particles.list() {
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
                    sdf: &piece.sim.grid.sdf(),
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

    /// Get total particle count across all pieces.
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

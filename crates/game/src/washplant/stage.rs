use crate::equipment_geometry::{
    GrateConfig, GrateGeometryBuilder, HopperConfig, HopperGeometryBuilder, SluiceVertex,
};
use crate::gpu::flip_3d::GpuFlip3D;
use crate::sluice_geometry::{SluiceConfig, SluiceGeometryBuilder};
use crate::washplant::config::{
    EquipmentType, GrizzlyStageConfig, HopperStageConfig, ShakerStageConfig, SluiceStageConfig,
    StageConfig,
};
use crate::washplant::metrics::StageMetrics;
use glam::{Mat3, Vec3};
use sim3d::{FlipSimulation3D, Grid3D};

/// A single processing stage in the washplant.
pub struct WashplantStage {
    /// Configuration.
    pub config: StageConfig,

    /// CPU simulation state.
    pub sim: FlipSimulation3D,

    /// GPU acceleration (optional - None if headless).
    pub gpu_flip: Option<GpuFlip3D>,

    /// World-space offset for rendering.
    pub world_offset: Vec3,

    /// Mesh data for rendering equipment.
    pub vertices: Vec<SluiceVertex>,
    pub indices: Vec<u32>,

    /// GPU buffers (created on first render).
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,

    /// Metrics.
    pub metrics: StageMetrics,

    /// Particle data cache for GPU sync.
    positions_cache: Vec<Vec3>,
    velocities_cache: Vec<Vec3>,
    affine_cache: Vec<Mat3>,
    densities_cache: Vec<f32>,
    cell_types_cache: Vec<u32>,
}

impl WashplantStage {
    /// Create a new stage from configuration.
    pub fn new(config: StageConfig) -> Self {
        // 1. Create FLIP simulation with grid dimensions from config.
        let mut sim = FlipSimulation3D::new(
            config.grid_width,
            config.grid_height,
            config.grid_depth,
            config.cell_size,
        );

        // 2. Build equipment geometry and mark solid cells.
        let (vertices, indices) = Self::build_equipment_geometry(&config, &mut sim.grid);

        // 3. Compute SDF for collision.
        sim.grid.compute_sdf();

        Self {
            world_offset: config.world_offset,
            metrics: StageMetrics::new(config.name),
            config,
            sim,
            gpu_flip: None,
            vertices,
            indices,
            vertex_buffer: None,
            index_buffer: None,
            positions_cache: Vec::new(),
            velocities_cache: Vec::new(),
            affine_cache: Vec::new(),
            densities_cache: Vec::new(),
            cell_types_cache: Vec::new(),
        }
    }

    /// Build equipment geometry based on type, returns (vertices, indices).
    fn build_equipment_geometry(
        config: &StageConfig,
        grid: &mut Grid3D,
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        match &config.equipment {
            EquipmentType::Hopper(hopper_cfg) => Self::build_hopper(config, hopper_cfg, grid),
            EquipmentType::Grizzly(grizzly_cfg) => Self::build_grizzly(config, grizzly_cfg, grid),
            EquipmentType::Shaker(shaker_cfg) => Self::build_shaker(config, shaker_cfg, grid),
            EquipmentType::Sluice(sluice_cfg) => Self::build_sluice(config, sluice_cfg, grid),
        }
    }

    fn build_hopper(
        config: &StageConfig,
        hopper_cfg: &HopperStageConfig,
        grid: &mut Grid3D,
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        let mut hopper_config = HopperConfig::default();
        hopper_config.grid_width = config.grid_width;
        hopper_config.grid_height = config.grid_height;
        hopper_config.grid_depth = config.grid_depth;
        hopper_config.cell_size = config.cell_size;
        hopper_config.top_width =
            Self::cells_from_meters(hopper_cfg.top_width, config.cell_size, config.grid_width);
        hopper_config.top_depth =
            Self::cells_from_meters(hopper_cfg.top_depth, config.cell_size, config.grid_depth);
        hopper_config.bottom_width = Self::cells_from_meters(
            hopper_cfg.bottom_width,
            config.cell_size,
            config.grid_width,
        );
        hopper_config.bottom_depth = Self::cells_from_meters(
            hopper_cfg.bottom_depth,
            config.cell_size,
            config.grid_depth,
        );
        hopper_config.wall_thickness = hopper_cfg.wall_thickness.max(1);

        let mut builder = HopperGeometryBuilder::new(hopper_config.clone());

        // Mark solid cells in grid.
        for (i, j, k) in builder.solid_cells() {
            grid.set_solid(i, j, k);
        }

        // Build mesh.
        builder.build_mesh(|i, j, k| hopper_config.is_solid(i, j, k));

        (builder.vertices().to_vec(), builder.indices().to_vec())
    }

    fn build_grizzly(
        config: &StageConfig,
        grizzly_cfg: &GrizzlyStageConfig,
        grid: &mut Grid3D,
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        let mut grate_config = GrateConfig::default();
        grate_config.grid_width = config.grid_width;
        grate_config.grid_height = config.grid_height;
        grate_config.grid_depth = config.grid_depth;
        grate_config.cell_size = config.cell_size;
        grate_config.bar_spacing = grizzly_cfg.bar_spacing.max(1);
        grate_config.bar_thickness = grizzly_cfg.bar_thickness.max(1);
        grate_config.orientation = 0; // bars parallel to X (flow direction)

        let mut builder = GrateGeometryBuilder::new(grate_config.clone());

        for (i, j, k) in builder.solid_cells() {
            grid.set_solid(i, j, k);
        }

        builder.build_mesh(|i, j, k| grate_config.is_solid(i, j, k));

        (builder.vertices().to_vec(), builder.indices().to_vec())
    }

    fn build_shaker(
        config: &StageConfig,
        shaker_cfg: &ShakerStageConfig,
        grid: &mut Grid3D,
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        // Shaker uses GrateConfig with holes.
        let mut grate_config = GrateConfig::default();
        grate_config.grid_width = config.grid_width;
        grate_config.grid_height = config.grid_height;
        grate_config.grid_depth = config.grid_depth;
        grate_config.cell_size = config.cell_size;
        grate_config.bar_spacing = Self::cells_from_meters(
            shaker_cfg.hole_spacing,
            config.cell_size,
            config.grid_depth,
        );
        grate_config.bar_thickness = Self::cells_from_meters(
            shaker_cfg.deck_thickness,
            config.cell_size,
            config.grid_depth,
        );
        grate_config.orientation = 0;

        let mut builder = GrateGeometryBuilder::new(grate_config.clone());

        for (i, j, k) in builder.solid_cells() {
            grid.set_solid(i, j, k);
        }

        builder.build_mesh(|i, j, k| grate_config.is_solid(i, j, k));

        (builder.vertices().to_vec(), builder.indices().to_vec())
    }

    fn build_sluice(
        config: &StageConfig,
        sluice_cfg: &SluiceStageConfig,
        grid: &mut Grid3D,
    ) -> (Vec<SluiceVertex>, Vec<u32>) {
        let mut sluice_config = SluiceConfig::default();
        sluice_config.grid_width = config.grid_width;
        sluice_config.grid_height = config.grid_height;
        sluice_config.grid_depth = config.grid_depth;
        sluice_config.cell_size = config.cell_size;
        sluice_config.floor_height_left = sluice_cfg.floor_height_left;
        sluice_config.floor_height_right = sluice_cfg.floor_height_right;
        sluice_config.riffle_spacing = sluice_cfg.riffle_spacing;
        sluice_config.riffle_height = sluice_cfg.riffle_height;
        sluice_config.riffle_thickness = sluice_cfg.riffle_thickness;
        sluice_config.riffle_start_x = 10; // slick plate at start
        sluice_config.riffle_end_pad = 5;
        sluice_config.exit_width_fraction = 0.8;
        sluice_config.exit_height = 8;
        sluice_config.wall_margin = sluice_cfg.wall_margin;

        let mut builder = SluiceGeometryBuilder::new(sluice_config);

        for (i, j, k) in builder.solid_cells() {
            grid.set_solid(i, j, k);
        }

        builder.build_mesh(|i, j, k| grid.is_solid(i, j, k));

        (builder.vertices().to_vec(), builder.indices().to_vec())
    }

    /// Initialize GPU backend.
    pub fn init_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        use wgpu::util::DeviceExt;

        // Create GpuFlip3D.
        let mut gpu_flip = GpuFlip3D::new(
            device,
            self.config.grid_width as u32,
            self.config.grid_height as u32,
            self.config.grid_depth as u32,
            self.config.cell_size,
            self.config.max_particles,
        );
        gpu_flip.upload_sdf(queue, &self.sim.grid.sdf);
        self.gpu_flip = Some(gpu_flip);

        // Create vertex buffer.
        if !self.vertices.is_empty() {
            self.vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} vertex buffer", self.config.name)),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }));
        }

        // Create index buffer.
        if !self.indices.is_empty() {
            self.index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{} index buffer", self.config.name)),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX,
            }));
        }
    }

    /// Simulation tick (CPU or GPU).
    pub fn tick(
        &mut self,
        dt: f32,
        device: Option<&wgpu::Device>,
        queue: Option<&wgpu::Queue>,
    ) {
        let start = std::time::Instant::now();

        if self.gpu_flip.is_some() && device.is_some() && queue.is_some() {
            // Prepare inputs BEFORE borrowing gpu_flip to satisfy borrow checker
            self.prepare_gpu_inputs();

            let gpu_flip = self.gpu_flip.as_mut().unwrap();
            let device = device.unwrap();
            let queue = queue.unwrap();
            let sdf = self.sim.grid.sdf.as_slice();
            gpu_flip.step(
                device,
                queue,
                &mut self.positions_cache,
                &mut self.velocities_cache,
                &mut self.affine_cache,
                &self.densities_cache,
                &self.cell_types_cache,
                Some(sdf),
                None,
                dt,
                self.sim.gravity.y,
                0.0,
                self.sim.pressure_iterations as u32,
            );
            self.apply_gpu_results(self.positions_cache.len());
        } else {
            // CPU fallback.
            self.sim.update(dt);
        }

        self.update_metrics(start.elapsed().as_secs_f32() * 1000.0);
    }

    /// Spawn a water particle.
    pub fn spawn_water(&mut self, position: Vec3, velocity: Vec3) {
        if self.sim.particles.list.len() < self.config.max_particles {
            self.sim.spawn_particle_with_velocity(position, velocity);
            self.metrics.particles_entered += 1;
        }
    }

    /// Spawn a sediment particle.
    pub fn spawn_sediment(&mut self, position: Vec3, velocity: Vec3, density: f32) {
        if self.sim.particles.list.len() < self.config.max_particles {
            self.sim.spawn_sediment(position, velocity, density);
            self.metrics.particles_entered += 1;
        }
    }

    /// Remove particle at index, returns its data.
    pub fn remove_particle(&mut self, idx: usize) -> Option<(Vec3, Vec3, f32)> {
        if idx < self.sim.particles.list.len() {
            let p = self.sim.particles.list.swap_remove(idx);
            self.metrics.particles_exited += 1;
            Some((p.position, p.velocity, p.density))
        } else {
            None
        }
    }

    /// Get particle count.
    pub fn particle_count(&self) -> usize {
        self.sim.particles.list.len()
    }

    /// Get vertex/index buffers for rendering.
    pub fn render_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer, u32)> {
        match (&self.vertex_buffer, &self.index_buffer) {
            (Some(vb), Some(ib)) => Some((vb, ib, self.indices.len() as u32)),
            _ => None,
        }
    }

    /// Get grid dimensions.
    pub fn grid_size(&self) -> (usize, usize, usize) {
        (
            self.config.grid_width,
            self.config.grid_height,
            self.config.grid_depth,
        )
    }

    /// Get cell size.
    pub fn cell_size(&self) -> f32 {
        self.config.cell_size
    }

    fn cells_from_meters(value: f32, cell_size: f32, max_cells: usize) -> usize {
        let max_cells = max_cells.max(1);
        if cell_size <= 0.0 || !cell_size.is_finite() {
            return max_cells;
        }
        let raw = (value / cell_size).round();
        let clamped = raw.clamp(1.0, max_cells as f32);
        clamped as usize
    }

    fn prepare_gpu_inputs(&mut self) {
        let particle_count = self.sim.particles.list.len();
        self.positions_cache.clear();
        self.velocities_cache.clear();
        self.affine_cache.clear();
        self.densities_cache.clear();
        self.positions_cache.reserve(particle_count);
        self.velocities_cache.reserve(particle_count);
        self.affine_cache.reserve(particle_count);
        self.densities_cache.reserve(particle_count);

        for p in &self.sim.particles.list {
            self.positions_cache.push(p.position);
            self.velocities_cache.push(p.velocity);
            self.affine_cache.push(p.affine_velocity);
            self.densities_cache.push(p.density);
        }

        self.rebuild_cell_types();
    }

    fn rebuild_cell_types(&mut self) {
        let grid = &self.sim.grid;
        let cell_count = grid.width * grid.height * grid.depth;
        self.cell_types_cache.clear();
        self.cell_types_cache.resize(cell_count, 0);

        for (idx, &sdf_val) in grid.sdf.iter().enumerate() {
            if sdf_val < 0.0 {
                self.cell_types_cache[idx] = 2; // Solid
            }
        }

        let cell_size = grid.cell_size;
        for pos in &self.positions_cache {
            let i = (pos.x / cell_size) as i32;
            let j = (pos.y / cell_size) as i32;
            let k = (pos.z / cell_size) as i32;
            if i >= 0
                && i < grid.width as i32
                && j >= 0
                && j < grid.height as i32
                && k >= 0
                && k < grid.depth as i32
            {
                let idx = k as usize * grid.width * grid.height + j as usize * grid.width + i as usize;
                if self.cell_types_cache[idx] != 2 {
                    self.cell_types_cache[idx] = 1; // Fluid
                }
            }
        }
    }

    fn apply_gpu_results(&mut self, count: usize) {
        let limit = count.min(self.sim.particles.list.len());
        for (i, p) in self.sim.particles.list.iter_mut().enumerate().take(limit) {
            if i < self.positions_cache.len() {
                p.position = self.positions_cache[i];
            }
            if i < self.velocities_cache.len() {
                p.velocity = self.velocities_cache[i];
            }
            if i < self.affine_cache.len() {
                p.affine_velocity = self.affine_cache[i];
            }
        }
    }

    fn update_metrics(&mut self, last_tick_ms: f32) {
        let mut sediment_particles = 0;
        for p in &self.sim.particles.list {
            if p.is_sediment() {
                sediment_particles += 1;
            }
        }
        let total = self.sim.particles.list.len();
        self.metrics.total_particles = total;
        self.metrics.sediment_particles = sediment_particles;
        self.metrics.water_particles = total.saturating_sub(sediment_particles);
        self.metrics.last_tick_ms = last_tick_ms;
        if self.metrics.avg_tick_ms == 0.0 {
            self.metrics.avg_tick_ms = last_tick_ms;
        } else {
            self.metrics.avg_tick_ms = self.metrics.avg_tick_ms * 0.9 + last_tick_ms * 0.1;
        }
    }
}

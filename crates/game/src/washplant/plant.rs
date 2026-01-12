//! Washplant orchestrator - manages multiple stages, transfers, and camera control

use super::{PlantConfig, PlantMetrics, TransferZone, WashplantStage, AABB};
use glam::Vec3;

/// Camera targeting mode
#[derive(Clone, Debug)]
pub enum CameraTarget {
    /// Overview of entire plant
    Overview,
    /// Focused on specific stage
    Stage(usize),
}

/// Multi-stage washplant orchestrator
pub struct Washplant {
    stages: Vec<WashplantStage>,
    transfers: Vec<TransferZone>,
    metrics: PlantMetrics,
    camera_target: CameraTarget,
    focused_stage: Option<usize>,
    paused: bool,
    frame: u64,
}

impl Washplant {
    /// Create new washplant from configuration
    pub fn new(config: PlantConfig) -> Self {
        // Create stages with grid dimensions based on equipment type
        let mut stages = Vec::new();
        for stage_config in &config.stages {
            // Grid dimensions: 80x60x40, cell_size 0.01m
            let mut stage = WashplantStage::new(80, 60, 40, 0.01);
            stage.build_equipment_geometry(stage_config);
            stages.push(stage);
        }

        let mut plant = Self {
            stages,
            transfers: Vec::new(),
            metrics: PlantMetrics::new(),
            camera_target: CameraTarget::Overview,
            focused_stage: None,
            paused: false,
            frame: 0,
        };

        plant.create_transfer_zones();
        plant.metrics.start();
        plant
    }

    /// Create default 4-stage plant
    pub fn default_plant() -> Self {
        Self::new(PlantConfig::standard_4_stage())
    }

    /// Create transfer zones between adjacent stages
    fn create_transfer_zones(&mut self) {
        self.transfers.clear();

        for i in 0..(self.stages.len() - 1) {
            let source = &self.stages[i];
            let _dest = &self.stages[i + 1];

            let (sw, sh, sd) = source.grid_size();
            let s_cell = source.cell_size();

            // Exit region: bottom-right corner of source stage (10% of grid)
            let exit_min = [
                (sw as f32 * 0.9 * s_cell),
                0.0,
                (sd as f32 * 0.9 * s_cell),
            ];
            let exit_max = [
                (sw as f32 * s_cell),
                (sh as f32 * 0.2 * s_cell),
                (sd as f32 * s_cell),
            ];
            let capture_region = AABB::new(exit_min, exit_max);

            // Transit duration: 0.5 seconds
            let transfer = TransferZone::new(capture_region, 0.5);
            self.transfers.push(transfer);
        }
    }

    /// Initialize GPU acceleration for all stages
    pub fn init_gpu(&mut self, device: &wgpu::Device, max_particles: usize) {
        let per_stage = max_particles / self.stages.len().max(1);
        for stage in &mut self.stages {
            stage.init_gpu(device, per_stage);
        }
    }

    /// Main simulation tick
    pub fn tick(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        self.frame += 1;
        self.process_transfers();

        // Tick each stage (CPU-only for now, GPU sync would be separate)
        for stage in &mut self.stages {
            stage.tick(dt);
        }

        self.update_metrics();
        self.metrics.tick();
    }

    /// Sync stages to GPU (if GPU enabled)
    pub fn sync_to_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        if self.paused {
            return;
        }

        self.frame += 1;
        self.process_transfers();

        for stage in &mut self.stages {
            stage.sync_to_gpu(device, queue, dt);
        }

        self.update_metrics();
        self.metrics.tick();
    }

    /// Process particle transfers between stages
    fn process_transfers(&mut self) {
        let current_time = self.frame as f32 * 0.016; // Approximate time in seconds

        for transfer_idx in 0..self.transfers.len() {
            // Process arrivals first
            let arrived = self.transfers[transfer_idx].tick(current_time);

            // Spawn arrived particles in destination stage
            if !arrived.is_empty() && transfer_idx + 1 < self.stages.len() {
                let dest_stage = &mut self.stages[transfer_idx + 1];
                let (dw, dh, dd) = dest_stage.grid_size();
                let d_cell = dest_stage.cell_size();

                // Inlet position: top-left corner
                for _ in arrived {
                    let inlet_pos = Vec3::new(
                        dw as f32 * 0.1 * d_cell,
                        dh as f32 * 0.8 * d_cell,
                        dd as f32 * 0.1 * d_cell,
                    );
                    dest_stage.spawn_water(inlet_pos);
                }
            }

            // Capture particles from source stage
            if transfer_idx < self.stages.len() {
                let source_stage = &mut self.stages[transfer_idx];
                let particle_count = source_stage.particle_count();
                let mut to_remove = Vec::new();

                for idx in 0..particle_count {
                    if let Some(pos) = source_stage.particle_position(idx) {
                        let pos_arr = [pos.x, pos.y, pos.z];

                        if self.transfers[transfer_idx].should_capture(pos_arr) {
                            self.transfers[transfer_idx].enqueue(idx, current_time);
                            to_remove.push(idx);
                        }
                    }
                }

                // Remove in reverse order to maintain indices
                for idx in to_remove.into_iter().rev() {
                    source_stage.remove_particle(idx);
                }
            }
        }
    }

    /// Update plant-wide metrics
    fn update_metrics(&mut self) {
        // Note: PlantMetrics fields are private, so we don't update them directly.
        // Metrics are tracked internally by PlantMetrics::tick()
    }

    /// Get total particle count across all stages
    pub fn total_particles(&self) -> usize {
        self.stages.iter().map(|s| s.particle_count()).sum()
    }

    /// Focus camera on specific stage or overview
    pub fn focus_stage(&mut self, index: Option<usize>) {
        self.focused_stage = index;
        self.camera_target = match index {
            Some(idx) => CameraTarget::Stage(idx),
            None => CameraTarget::Overview,
        };
    }

    /// Get camera parameters for current target
    pub fn camera_params(&self) -> (f32, f32, f32, Vec3) {
        match &self.camera_target {
            CameraTarget::Overview => {
                // Overview: show entire plant
                let plant_center = self.plant_center();
                (0.5, 0.4, 10.0, plant_center)
            }
            CameraTarget::Stage(idx) => {
                // Focused view of specific stage
                if let Some(stage) = self.stages.get(*idx) {
                    let stage_center = self.stage_center(stage);
                    (0.0, 0.3, 5.0, stage_center)
                } else {
                    // Fallback to overview
                    let plant_center = self.plant_center();
                    (0.5, 0.4, 10.0, plant_center)
                }
            }
        }
    }

    /// Calculate plant center from all stage bounds
    fn plant_center(&self) -> Vec3 {
        if self.stages.is_empty() {
            return Vec3::ZERO;
        }

        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for stage in &self.stages {
            let (w, h, d) = stage.grid_size();
            let cell_size = stage.cell_size();
            let stage_max = Vec3::new(
                w as f32 * cell_size,
                h as f32 * cell_size,
                d as f32 * cell_size,
            );

            min = min.min(Vec3::ZERO);
            max = max.max(stage_max);
        }

        (min + max) / 2.0
    }

    /// Calculate stage center
    fn stage_center(&self, stage: &WashplantStage) -> Vec3 {
        let (w, h, d) = stage.grid_size();
        let cell_size = stage.cell_size();
        Vec3::new(
            w as f32 * cell_size / 2.0,
            h as f32 * cell_size / 2.0,
            d as f32 * cell_size / 2.0,
        )
    }

    /// Toggle pause state
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Get immutable reference to stage
    pub fn stage(&self, idx: usize) -> Option<&WashplantStage> {
        self.stages.get(idx)
    }

    /// Get mutable reference to stage
    pub fn stage_mut(&mut self, idx: usize) -> Option<&mut WashplantStage> {
        self.stages.get_mut(idx)
    }

    /// Spawn water particles at inlet of first stage
    pub fn spawn_inlet_water(&mut self, count: usize) {
        if let Some(hopper) = self.stages.first_mut() {
            let (w, h, d) = hopper.grid_size();
            let cell_size = hopper.cell_size();

            // Inlet region at top of hopper (top 20% height, center 50% width/depth)
            let inlet_min = Vec3::new(
                w as f32 * 0.25 * cell_size,
                h as f32 * 0.8 * cell_size,
                d as f32 * 0.25 * cell_size,
            );
            let inlet_max = Vec3::new(
                w as f32 * 0.75 * cell_size,
                h as f32 * cell_size,
                d as f32 * 0.75 * cell_size,
            );

            let mut rng = rand::XorShift64::new(self.frame);

            for _ in 0..count {
                let pos = Vec3::new(
                    inlet_min.x + rng.next_f32() * (inlet_max.x - inlet_min.x),
                    inlet_min.y + rng.next_f32() * (inlet_max.y - inlet_min.y),
                    inlet_min.z + rng.next_f32() * (inlet_max.z - inlet_min.z),
                );
                hopper.spawn_water(pos);
            }
        }
    }

    /// Get status string for display
    pub fn status_string(&self) -> String {
        let focused = self
            .focused_stage
            .map(|i| format!("Stage {}", i))
            .unwrap_or_else(|| "None".to_string());

        format!(
            "Frame: {} | Paused: {} | Focused: {} | {}",
            self.frame,
            self.paused,
            focused,
            self.metrics.format_summary()
        )
    }

    /// Get number of stages
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get frame counter
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Check if paused
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Get metrics reference
    pub fn metrics(&self) -> &PlantMetrics {
        &self.metrics
    }
}

/// Simple XorShift PRNG for particle spawning
mod rand {
    pub struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        pub fn new(seed: u64) -> Self {
            Self {
                state: if seed == 0 { 1 } else { seed },
            }
        }

        pub fn next(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        pub fn next_f32(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_plant_creation() {
        let plant = Washplant::default_plant();
        assert_eq!(plant.stage_count(), 4);
        assert_eq!(plant.transfers.len(), 3); // 4 stages = 3 transfers
        assert_eq!(plant.frame(), 0);
        assert!(!plant.is_paused());
    }

    #[test]
    fn test_stage_access() {
        let plant = Washplant::default_plant();
        assert!(plant.stage(0).is_some());
        assert!(plant.stage(3).is_some());
        assert!(plant.stage(4).is_none());
    }

    #[test]
    fn test_focus_stage() {
        let mut plant = Washplant::default_plant();
        plant.focus_stage(Some(2));
        assert_eq!(plant.focused_stage, Some(2));

        plant.focus_stage(None);
        assert_eq!(plant.focused_stage, None);
    }

    #[test]
    fn test_toggle_pause() {
        let mut plant = Washplant::default_plant();
        assert!(!plant.is_paused());

        plant.toggle_pause();
        assert!(plant.is_paused());

        plant.toggle_pause();
        assert!(!plant.is_paused());
    }

    #[test]
    fn test_xorshift_prng() {
        let mut rng = rand::XorShift64::new(12345);
        let val1 = rng.next();
        let val2 = rng.next();
        assert_ne!(val1, val2);

        let f32_val = rng.next_f32();
        assert!(f32_val >= 0.0 && f32_val <= 1.0);
    }
}

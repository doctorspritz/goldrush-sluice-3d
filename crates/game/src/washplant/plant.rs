use crate::washplant::config::{PlantConfig, TransferConfig};
use crate::washplant::metrics::PlantMetrics;
use crate::washplant::stage::WashplantStage;
use crate::washplant::transfer::{TransferZone, AABB};
use glam::Vec3;

/// Camera target for viewing the plant.
#[derive(Clone, Copy, Debug)]
pub enum CameraTarget {
    /// View entire plant.
    Overview,
    /// Focus on specific stage (index).
    Stage(usize),
}

/// Main washplant orchestrator.
pub struct Washplant {
    /// All processing stages.
    pub stages: Vec<WashplantStage>,

    /// Transfer zones between stages.
    pub transfers: Vec<TransferZone>,

    /// Plant-wide metrics.
    pub metrics: PlantMetrics,

    /// Current camera target.
    pub camera_target: CameraTarget,

    /// Focused stage index (for keyboard navigation).
    focused_stage: Option<usize>,

    /// Paused state.
    pub paused: bool,

    /// Frame counter.
    pub frame: u64,
}

impl Washplant {
    /// Create washplant from configuration.
    pub fn new(config: PlantConfig) -> Self {
        let PlantConfig {
            stages: stage_configs,
            transfers: transfer_configs,
        } = config;

        let stages: Vec<WashplantStage> =
            stage_configs.into_iter().map(WashplantStage::new).collect();

        let transfers = Self::create_transfer_zones(&stages, &transfer_configs);

        let stage_names: Vec<String> = stages.iter().map(|s| s.config.name.clone()).collect();
        let metrics = PlantMetrics::new(&stage_names);

        Self {
            stages,
            transfers,
            metrics,
            camera_target: CameraTarget::Overview,
            focused_stage: None,
            paused: false,
            frame: 0,
        }
    }

    /// Create default 4-stage washplant.
    pub fn default_plant() -> Self {
        Self::new(PlantConfig::default())
    }

    /// Create transfer zones from transfer configs.
    fn create_transfer_zones(
        stages: &[WashplantStage],
        transfer_configs: &[TransferConfig],
    ) -> Vec<TransferZone> {
        transfer_configs
            .iter()
            .filter_map(|tc| {
                let from_stage = stages.get(tc.from_stage)?;
                let to_stage = stages.get(tc.to_stage)?;

                let (gw, gh, gd) = from_stage.grid_size();
                let cs = from_stage.cell_size();

                // Use capture_depth_cells from config
                let capture_start_x = gw.saturating_sub(tc.capture_depth_cells) as f32 * cs;
                let capture_aabb = AABB::new(
                    Vec3::new(capture_start_x, 0.0, 0.0),
                    Vec3::new(gw as f32 * cs, gh as f32 * cs, gd as f32 * cs),
                );

                // Use inject_offset from config (normalized 0-1 coordinates)
                let (tw, th, td) = to_stage.grid_size();
                let tcs = to_stage.cell_size();
                let inject_pos = Vec3::new(
                    tc.inject_offset[0] * tw as f32 * tcs,
                    tc.inject_offset[1] * th as f32 * tcs,
                    tc.inject_offset[2] * td as f32 * tcs,
                );

                // Use exit_direction from config
                let exit_dir = Vec3::from_array(tc.exit_direction).normalize_or_zero();

                // Use inject_velocity from config
                let inject_vel = Vec3::from_array(tc.inject_velocity);

                Some(
                    TransferZone::new(
                        tc.from_stage,
                        tc.to_stage,
                        capture_aabb,
                        exit_dir,
                        inject_pos,
                    )
                    .with_inject_velocity(inject_vel)
                    .with_transit_time(tc.transit_time),
                )
            })
            .collect()
    }

    /// Initialize GPU backends for all stages.
    pub fn init_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for stage in &mut self.stages {
            stage.init_gpu(device, queue);
        }
    }

    /// Main simulation tick.
    pub fn tick(&mut self, dt: f32, device: Option<&wgpu::Device>, queue: Option<&wgpu::Queue>) {
        if self.paused {
            return;
        }

        self.frame += 1;

        for stage in &mut self.stages {
            stage.tick(dt, device, queue);
        }

        self.process_transfers(dt);
        self.update_metrics();
    }

    /// Process particle transfers between stages.
    fn process_transfers(&mut self, dt: f32) {
        if self.transfers.is_empty() || self.stages.is_empty() {
            return;
        }

        let mut removals: Vec<Vec<usize>> = vec![Vec::new(); self.stages.len()];

        for transfer in &mut self.transfers {
            let captured = {
                let from_stage = match self.stages.get(transfer.from_stage) {
                    Some(stage) => stage,
                    None => continue,
                };

                let mut captured = Vec::new();
                for (idx, particle) in from_stage.sim.particles.list().iter().enumerate() {
                    if transfer.should_capture(particle.position, particle.velocity) {
                        captured.push((
                            idx,
                            particle.position,
                            particle.velocity,
                            particle.density,
                            particle.is_sediment(),
                        ));
                    }
                }
                captured
            };

            if !captured.is_empty() {
                for (_, pos, vel, density, is_sediment) in &captured {
                    transfer.enqueue(*pos, *vel, *density, *is_sediment);
                }
                removals[transfer.from_stage].extend(captured.iter().map(|(idx, _, _, _, _)| *idx));
            }

            let ready = transfer.tick(dt);
            if let Some(to_stage) = self.stages.get_mut(transfer.to_stage) {
                for tp in ready {
                    if tp.is_sediment {
                        to_stage.spawn_sediment(tp.position, tp.velocity, tp.density);
                    } else {
                        to_stage.spawn_water(tp.position, tp.velocity);
                    }
                }
            }
        }

        for (stage_idx, indices) in removals.iter_mut().enumerate() {
            if indices.is_empty() {
                continue;
            }
            indices.sort_unstable();
            indices.dedup();
            indices.sort_unstable_by(|a, b| b.cmp(a));
            if let Some(stage) = self.stages.get_mut(stage_idx) {
                for idx in indices.drain(..) {
                    stage.remove_particle(idx);
                }
            }
        }
    }

    /// Update plant-wide metrics.
    fn update_metrics(&mut self) {
        self.metrics.tick();
        for (i, stage) in self.stages.iter().enumerate() {
            if let Some(metrics) = self.metrics.stages.get_mut(i) {
                *metrics = stage.metrics.clone();
            }
        }
    }

    /// Get total particle count across all stages.
    pub fn total_particles(&self) -> usize {
        self.stages.iter().map(|s| s.particle_count()).sum()
    }

    /// Focus on a specific stage (1-indexed for keyboard).
    pub fn focus_stage(&mut self, stage_num: usize) {
        if stage_num == 0 {
            self.camera_target = CameraTarget::Overview;
            self.focused_stage = None;
        } else if stage_num <= self.stages.len() {
            self.camera_target = CameraTarget::Stage(stage_num - 1);
            self.focused_stage = Some(stage_num - 1);
        }
    }

    /// Get camera position and target for current view.
    pub fn camera_params(&self) -> (Vec3, Vec3) {
        if self.stages.is_empty() {
            return (Vec3::new(5.0, 5.0, 10.0), Vec3::ZERO);
        }

        match self.camera_target {
            CameraTarget::Overview => {
                let mut min = Vec3::splat(f32::MAX);
                let mut max = Vec3::splat(f32::MIN);

                for stage in &self.stages {
                    let offset = stage.world_offset;
                    let (gw, gh, gd) = stage.grid_size();
                    let cs = stage.cell_size();

                    min = min.min(offset);
                    max =
                        max.max(offset + Vec3::new(gw as f32 * cs, gh as f32 * cs, gd as f32 * cs));
                }

                let center = (min + max) * 0.5;
                let size = (max - min).length();
                let eye = center + Vec3::new(size * 0.5, size * 0.8, size * 1.2);

                (eye, center)
            }
            CameraTarget::Stage(idx) => {
                if let Some(stage) = self.stages.get(idx) {
                    let (gw, gh, gd) = stage.grid_size();
                    let cs = stage.cell_size();

                    let center = stage.world_offset
                        + Vec3::new(
                            gw as f32 * cs * 0.5,
                            gh as f32 * cs * 0.5,
                            gd as f32 * cs * 0.5,
                        );

                    let size = (gw.max(gd) as f32 * cs).max(gh as f32 * cs);
                    let eye = center + Vec3::new(size * 0.3, size * 0.6, size * 1.0);

                    (eye, center)
                } else {
                    (Vec3::new(5.0, 5.0, 10.0), Vec3::ZERO)
                }
            }
        }
    }

    /// Toggle pause state.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Get stage by index.
    pub fn stage(&self, idx: usize) -> Option<&WashplantStage> {
        self.stages.get(idx)
    }

    /// Get mutable stage by index.
    pub fn stage_mut(&mut self, idx: usize) -> Option<&mut WashplantStage> {
        self.stages.get_mut(idx)
    }

    /// Spawn water at the first stage (hopper inlet).
    pub fn spawn_inlet_water(&mut self, count: usize) {
        if let Some(stage) = self.stages.first_mut() {
            let (gw, gh, gd) = stage.grid_size();
            let cs = stage.cell_size();

            for _ in 0..count {
                let x = gw as f32 * cs * 0.5 + (rand::random::<f32>() - 0.5) * cs * 4.0;
                let y = gh as f32 * cs * 0.9;
                let z = gd as f32 * cs * 0.5 + (rand::random::<f32>() - 0.5) * cs * 4.0;
                let vel = Vec3::new(0.0, -0.5, 0.0);

                stage.spawn_water(Vec3::new(x, y, z), vel);
            }
        }
    }

    /// Get status string for HUD.
    pub fn status_string(&self) -> String {
        format!(
            "{} | Stages: {} | Frame: {}",
            self.metrics.format_summary(),
            self.stages.len(),
            self.frame
        )
    }
}

/// Simple random for particle spawn jitter.
mod rand {
    use std::cell::Cell;

    thread_local! {
        static SEED: Cell<u32> = Cell::new(12345);
    }

    pub fn random<T: From<f32>>() -> T {
        SEED.with(|s| {
            let mut x = s.get();
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            s.set(x);
            T::from(x as f32 / u32::MAX as f32)
        })
    }
}

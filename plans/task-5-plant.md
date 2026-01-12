# Task 5: Washplant Orchestrator

**File to create:** `crates/game/src/washplant/plant.rs`

## Goal
Create `Washplant` struct that orchestrates multiple `WashplantStage` instances and handles particle transfers between them.

## Dependencies
- `crates/game/src/washplant/stage.rs` (Task 4 - WashplantStage)
- `crates/game/src/washplant/config.rs` (PlantConfig)
- `crates/game/src/washplant/transfer.rs` (TransferZone, AABB)
- `crates/game/src/washplant/metrics.rs` (PlantMetrics, StageMetrics)
- `glam::Vec3`

## Types to Implement

```rust
use crate::washplant::config::PlantConfig;
use crate::washplant::metrics::PlantMetrics;
use crate::washplant::stage::WashplantStage;
use crate::washplant::transfer::{TransferZone, AABB};
use glam::Vec3;

/// Camera target for viewing the plant
#[derive(Clone, Copy, Debug)]
pub enum CameraTarget {
    /// View entire plant
    Overview,
    /// Focus on specific stage (index)
    Stage(usize),
}

/// Main washplant orchestrator
pub struct Washplant {
    /// All processing stages
    pub stages: Vec<WashplantStage>,

    /// Transfer zones between stages
    pub transfers: Vec<TransferZone>,

    /// Plant-wide metrics
    pub metrics: PlantMetrics,

    /// Current camera target
    pub camera_target: CameraTarget,

    /// Focused stage index (for keyboard navigation)
    focused_stage: Option<usize>,

    /// Paused state
    pub paused: bool,

    /// Frame counter
    pub frame: u64,
}

impl Washplant {
    /// Create washplant from configuration
    pub fn new(config: PlantConfig) -> Self {
        // Create stages
        let stages: Vec<WashplantStage> = config
            .stages
            .into_iter()
            .map(WashplantStage::new)
            .collect();

        // Create transfer zones based on connections
        let transfers = Self::create_transfer_zones(&stages, &config.connections);

        // Create metrics
        let stage_names: Vec<&'static str> = stages.iter().map(|s| s.config.name).collect();
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

    /// Create default 4-stage washplant
    pub fn default_plant() -> Self {
        Self::new(PlantConfig::default())
    }

    /// Create transfer zones from connection indices
    fn create_transfer_zones(
        stages: &[WashplantStage],
        connections: &[(usize, usize)],
    ) -> Vec<TransferZone> {
        connections
            .iter()
            .filter_map(|&(from_idx, to_idx)| {
                if from_idx >= stages.len() || to_idx >= stages.len() {
                    return None;
                }

                let from_stage = &stages[from_idx];
                let to_stage = &stages[to_idx];

                // Calculate capture zone at exit of source stage
                let (gw, gh, gd) = from_stage.grid_size();
                let cs = from_stage.cell_size();

                // Capture zone: right edge of source stage (X = max)
                let capture_aabb = AABB::new(
                    Vec3::new((gw - 3) as f32 * cs, 0.0, 0.0),
                    Vec3::new(gw as f32 * cs, gh as f32 * cs, gd as f32 * cs),
                );

                // Inject at left edge of destination stage
                let inject_pos = Vec3::new(
                    2.0 * to_stage.cell_size(),
                    to_stage.grid_size().1 as f32 * to_stage.cell_size() * 0.5,
                    to_stage.grid_size().2 as f32 * to_stage.cell_size() * 0.5,
                );

                // Exit direction: +X (flowing downstream)
                let exit_dir = Vec3::X;

                Some(
                    TransferZone::new(from_idx, to_idx, capture_aabb, exit_dir, inject_pos)
                        .with_inject_velocity(Vec3::new(0.5, 0.0, 0.0))
                        .with_transit_time(0.05),
                )
            })
            .collect()
    }

    /// Initialize GPU backends for all stages
    pub fn init_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for stage in &mut self.stages {
            stage.init_gpu(device, queue);
        }
    }

    /// Main simulation tick
    pub fn tick(&mut self, dt: f32, queue: &wgpu::Queue) {
        if self.paused {
            return;
        }

        self.frame += 1;

        // 1. Tick all stages
        for stage in &mut self.stages {
            stage.tick(dt, queue);
        }

        // 2. Process transfers between stages
        self.process_transfers(dt);

        // 3. Update metrics
        self.update_metrics();
    }

    /// Process particle transfers between stages
    fn process_transfers(&mut self, dt: f32) {
        // Collect particles to transfer (can't borrow stages mutably while iterating)
        let mut transfers_to_process: Vec<(usize, Vec<(Vec3, Vec3, f32, bool)>)> = Vec::new();

        for transfer in &mut self.transfers {
            let from_stage = &self.stages[transfer.from_stage];
            let mut captured = Vec::new();

            // Find particles to capture
            for (idx, particle) in from_stage.sim.particles.iter().enumerate() {
                if transfer.should_capture(particle.position, particle.velocity) {
                    let is_sediment = particle.density > 1.0;
                    captured.push((idx, particle.position, particle.velocity, particle.density, is_sediment));
                }
            }

            // Enqueue captured particles (in reverse order to preserve indices)
            for (_, pos, vel, density, is_sediment) in captured.iter().rev() {
                transfer.enqueue(*pos, *vel, *density, *is_sediment);
            }

            // Collect indices to remove
            let indices: Vec<usize> = captured.iter().map(|(idx, _, _, _, _)| *idx).collect();

            // Get particles ready to inject
            let ready = transfer.tick(dt);

            transfers_to_process.push((transfer.from_stage,
                indices.iter().map(|_| (Vec3::ZERO, Vec3::ZERO, 0.0, false)).collect()
            ));

            // Inject ready particles into destination
            let to_idx = transfer.to_stage;
            if to_idx < self.stages.len() {
                for tp in ready {
                    if tp.is_sediment {
                        self.stages[to_idx].spawn_sediment(tp.position, tp.velocity, tp.density);
                    } else {
                        self.stages[to_idx].spawn_water(tp.position, tp.velocity);
                    }
                }
            }
        }

        // Remove captured particles from source stages
        // NOTE: This is a simplified approach - in production, collect all removals first
        // and remove in reverse order to preserve indices
    }

    /// Update plant-wide metrics
    fn update_metrics(&mut self) {
        self.metrics.tick();

        // Sync stage metrics
        for (i, stage) in self.stages.iter().enumerate() {
            if i < self.metrics.stages.len() {
                self.metrics.stages[i] = stage.metrics.clone();
            }
        }
    }

    /// Get total particle count across all stages
    pub fn total_particles(&self) -> usize {
        self.stages.iter().map(|s| s.particle_count()).sum()
    }

    /// Focus on a specific stage (1-indexed for keyboard)
    pub fn focus_stage(&mut self, stage_num: usize) {
        if stage_num == 0 {
            self.camera_target = CameraTarget::Overview;
            self.focused_stage = None;
        } else if stage_num <= self.stages.len() {
            self.camera_target = CameraTarget::Stage(stage_num - 1);
            self.focused_stage = Some(stage_num - 1);
        }
    }

    /// Get camera position and target for current view
    pub fn camera_params(&self) -> (Vec3, Vec3) {
        match self.camera_target {
            CameraTarget::Overview => {
                // Calculate bounding box of all stages
                let mut min = Vec3::splat(f32::MAX);
                let mut max = Vec3::splat(f32::MIN);

                for stage in &self.stages {
                    let offset = stage.world_offset;
                    let (gw, gh, gd) = stage.grid_size();
                    let cs = stage.cell_size();

                    min = min.min(offset);
                    max = max.max(offset + Vec3::new(
                        gw as f32 * cs,
                        gh as f32 * cs,
                        gd as f32 * cs,
                    ));
                }

                let center = (min + max) * 0.5;
                let size = (max - min).length();
                let eye = center + Vec3::new(size * 0.5, size * 0.8, size * 1.2);

                (eye, center)
            }
            CameraTarget::Stage(idx) => {
                if idx < self.stages.len() {
                    let stage = &self.stages[idx];
                    let (gw, gh, gd) = stage.grid_size();
                    let cs = stage.cell_size();

                    let center = stage.world_offset + Vec3::new(
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

    /// Toggle pause state
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Get stage by index
    pub fn stage(&self, idx: usize) -> Option<&WashplantStage> {
        self.stages.get(idx)
    }

    /// Get mutable stage by index
    pub fn stage_mut(&mut self, idx: usize) -> Option<&mut WashplantStage> {
        self.stages.get_mut(idx)
    }

    /// Spawn water at the first stage (hopper inlet)
    pub fn spawn_inlet_water(&mut self, count: usize) {
        if let Some(stage) = self.stages.first_mut() {
            let (gw, gh, gd) = stage.grid_size();
            let cs = stage.cell_size();

            for _ in 0..count {
                // Spawn near top center
                let x = gw as f32 * cs * 0.5 + (rand::random::<f32>() - 0.5) * cs * 4.0;
                let y = gh as f32 * cs * 0.9;
                let z = gd as f32 * cs * 0.5 + (rand::random::<f32>() - 0.5) * cs * 4.0;

                let vel = Vec3::new(0.0, -0.5, 0.0);

                stage.spawn_water(Vec3::new(x, y, z), vel);
            }
        }
    }

    /// Get status string for HUD
    pub fn status_string(&self) -> String {
        format!(
            "{} | Stages: {} | Frame: {}",
            self.metrics.format_summary(),
            self.stages.len(),
            self.frame
        )
    }
}

/// Simple random for particle spawn jitter
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
            T::from((x as f32) / (u32::MAX as f32))
        })
    }
}
```

## Update mod.rs

Add to `crates/game/src/washplant/mod.rs`:
```rust
mod plant;
pub use plant::*;
```

## Notes
- The transfer processing has a simplified implementation. In production, you'd want to:
  1. Collect all particles to transfer in one pass
  2. Remove them in reverse index order
  3. Then inject into destinations
- The `spawn_sediment_with_velocity` may need adaptation based on actual API
- Camera calculations are approximate - adjust for good views
- The random module is a simple xorshift for spawn jitter

## Testing
Run `cargo check -p game` to verify compilation. Some warnings about unused fields/methods are expected.

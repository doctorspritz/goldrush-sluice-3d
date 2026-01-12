# Task 3: Metrics Types

**File to create:** `crates/game/src/washplant/metrics.rs`

## Goal
Define metrics tracking for washplant performance monitoring.

## Types to Implement

```rust
use std::collections::VecDeque;
use std::time::Instant;

/// Metrics for a single stage
#[derive(Clone, Debug, Default)]
pub struct StageMetrics {
    /// Stage name for display
    pub name: &'static str,

    /// Current particle counts
    pub total_particles: usize,
    pub water_particles: usize,
    pub sediment_particles: usize,

    /// Throughput tracking
    pub particles_entered: u64,
    pub particles_exited: u64,

    /// Mass tracking (kg)
    pub water_mass_kg: f32,
    pub sediment_mass_kg: f32,

    /// Performance
    pub last_tick_ms: f32,
    pub avg_tick_ms: f32,
}

impl StageMetrics {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            ..Default::default()
        }
    }

    pub fn throughput_per_second(&self, elapsed_seconds: f32) -> f32 {
        if elapsed_seconds > 0.0 {
            self.particles_exited as f32 / elapsed_seconds
        } else {
            0.0
        }
    }
}

/// Rolling average calculator
#[derive(Clone, Debug)]
pub struct RollingAverage {
    samples: VecDeque<f32>,
    capacity: usize,
    sum: f32,
}

impl RollingAverage {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }

    pub fn push(&mut self, value: f32) {
        if self.samples.len() >= self.capacity {
            if let Some(old) = self.samples.pop_front() {
                self.sum -= old;
            }
        }
        self.samples.push_back(value);
        self.sum += value;
    }

    pub fn average(&self) -> f32 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.sum / self.samples.len() as f32
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }
}

/// Plant-wide metrics
#[derive(Clone, Debug)]
pub struct PlantMetrics {
    /// Per-stage metrics
    pub stages: Vec<StageMetrics>,

    /// Overall throughput (tons per hour)
    pub throughput_tph: f32,

    /// Recovery tracking
    pub gold_input_kg: f32,
    pub gold_recovered_kg: f32,
    pub gangue_input_kg: f32,
    pub gangue_rejected_kg: f32,

    /// Water usage
    pub water_input_m3: f32,
    pub water_recycled_m3: f32,

    /// Timing
    pub start_time: Option<Instant>,
    pub elapsed_seconds: f32,
    pub frame_count: u64,

    /// FPS tracking
    fps_samples: RollingAverage,
    last_fps_update: Option<Instant>,
    frames_since_fps_update: u32,
}

impl Default for PlantMetrics {
    fn default() -> Self {
        Self {
            stages: Vec::new(),
            throughput_tph: 0.0,
            gold_input_kg: 0.0,
            gold_recovered_kg: 0.0,
            gangue_input_kg: 0.0,
            gangue_rejected_kg: 0.0,
            water_input_m3: 0.0,
            water_recycled_m3: 0.0,
            start_time: None,
            elapsed_seconds: 0.0,
            frame_count: 0,
            fps_samples: RollingAverage::new(60),
            last_fps_update: None,
            frames_since_fps_update: 0,
        }
    }
}

impl PlantMetrics {
    pub fn new(stage_names: &[&'static str]) -> Self {
        Self {
            stages: stage_names.iter().map(|&name| StageMetrics::new(name)).collect(),
            start_time: Some(Instant::now()),
            last_fps_update: Some(Instant::now()),
            ..Default::default()
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.last_fps_update = Some(Instant::now());
    }

    pub fn tick(&mut self) {
        self.frame_count += 1;
        self.frames_since_fps_update += 1;

        if let Some(start) = self.start_time {
            self.elapsed_seconds = start.elapsed().as_secs_f32();
        }

        // Update FPS every 500ms
        if let Some(last) = self.last_fps_update {
            let elapsed = last.elapsed().as_secs_f32();
            if elapsed >= 0.5 {
                let fps = self.frames_since_fps_update as f32 / elapsed;
                self.fps_samples.push(fps);
                self.frames_since_fps_update = 0;
                self.last_fps_update = Some(Instant::now());
            }
        }
    }

    pub fn fps(&self) -> f32 {
        self.fps_samples.average()
    }

    pub fn gold_recovery_percent(&self) -> f32 {
        if self.gold_input_kg > 0.0 {
            (self.gold_recovered_kg / self.gold_input_kg) * 100.0
        } else {
            0.0
        }
    }

    pub fn total_particles(&self) -> usize {
        self.stages.iter().map(|s| s.total_particles).sum()
    }

    /// Format metrics for HUD display
    pub fn format_summary(&self) -> String {
        format!(
            "FPS: {:.1} | Particles: {} | Elapsed: {:.1}s | TPH: {:.2}",
            self.fps(),
            self.total_particles(),
            self.elapsed_seconds,
            self.throughput_tph
        )
    }

    /// Format per-stage breakdown
    pub fn format_stages(&self) -> String {
        self.stages
            .iter()
            .map(|s| format!("{}: {} (W:{} S:{})",
                s.name, s.total_particles, s.water_particles, s.sediment_particles))
            .collect::<Vec<_>>()
            .join(" | ")
    }
}
```

## Update mod.rs

Add to `crates/game/src/washplant/mod.rs`:
```rust
mod metrics;
pub use metrics::*;
```

## Testing
Run `cargo check -p game` to verify compilation.

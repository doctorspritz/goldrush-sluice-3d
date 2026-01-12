use std::time::Instant;

/// Per-stage performance and particle metrics
#[derive(Clone, Debug)]
pub struct StageMetrics {
    pub particles_in: usize,
    pub particles_out: usize,
    pub gold_in: f32,
    pub gold_out: f32,
}

/// Rolling average calculator for smoothing metrics
#[derive(Clone, Debug)]
pub struct RollingAverage {
    values: Vec<f32>,
    capacity: usize,
}

impl RollingAverage {
    pub fn new(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, value: f32) {
        if self.values.len() >= self.capacity {
            self.values.remove(0);
        }
        self.values.push(value);
    }

    pub fn average(&self) -> f32 {
        if self.values.is_empty() {
            0.0
        } else {
            self.values.iter().sum::<f32>() / self.values.len() as f32
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// Top-level metrics for the entire wash plant
pub struct PlantMetrics {
    start_time: Option<Instant>,
    last_tick: Option<Instant>,
    fps_avg: RollingAverage,
    total_gold_in: f32,
    total_gold_recovered: f32,
    total_particles: usize,
}

impl PlantMetrics {
    pub fn new() -> Self {
        Self {
            start_time: None,
            last_tick: None,
            fps_avg: RollingAverage::new(60),
            total_gold_in: 0.0,
            total_gold_recovered: 0.0,
            total_particles: 0,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.last_tick = Some(Instant::now());
    }

    pub fn tick(&mut self) {
        if let Some(last) = self.last_tick {
            let now = Instant::now();
            let dt = now.duration_since(last).as_secs_f32();
            if dt > 0.0 {
                self.fps_avg.push(1.0 / dt);
            }
            self.last_tick = Some(now);
        }
    }

    pub fn fps(&self) -> f32 {
        self.fps_avg.average()
    }

    pub fn gold_recovery_percent(&self) -> f32 {
        if self.total_gold_in > 0.0 {
            (self.total_gold_recovered / self.total_gold_in) * 100.0
        } else {
            0.0
        }
    }

    pub fn total_particles(&self) -> usize {
        self.total_particles
    }

    pub fn format_summary(&self) -> String {
        format!(
            "FPS: {:.1} | Particles: {} | Gold Recovery: {:.1}%",
            self.fps(),
            self.total_particles,
            self.gold_recovery_percent()
        )
    }

    pub fn format_stages(&self, stages: &[StageMetrics]) -> String {
        let mut output = String::from("Stage Metrics:\n");
        for (i, stage) in stages.iter().enumerate() {
            let recovery = if stage.gold_in > 0.0 {
                (stage.gold_out / stage.gold_in) * 100.0
            } else {
                0.0
            };
            output.push_str(&format!(
                "  Stage {}: {} in / {} out | Gold: {:.2}g in / {:.2}g out ({:.1}%)\n",
                i + 1,
                stage.particles_in,
                stage.particles_out,
                stage.gold_in,
                stage.gold_out,
                recovery
            ));
        }
        output
    }
}

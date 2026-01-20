use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulationMetrics {
    pub frame: u32,
    pub particle_count: usize,
    pub mean_velocity: [f32; 3],
    pub mean_density: f32,
    pub settled_count: usize,
    pub additional_metrics: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenValue {
    pub scenario_name: String,
    pub target_frame: u32,
    pub metrics: SimulationMetrics,
    pub tolerances: HashMap<String, f32>,
}

impl GoldenValue {
    pub fn compare(&self, actual: &SimulationMetrics) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Standard metrics comparison
        if actual.particle_count != self.metrics.particle_count {
            errors.push(format!(
                "Particle count mismatch: expected {}, got {}",
                self.metrics.particle_count, actual.particle_count
            ));
        }

        // Add more comparisons with tolerances...
        for (name, &tolerance) in &self.tolerances {
            let expected = self.metrics.additional_metrics.get(name).cloned().unwrap_or(0.0);
            let got = actual.additional_metrics.get(name).cloned().unwrap_or(0.0);
            if (expected - got).abs() > tolerance {
                errors.push(format!(
                    "Metric '{}' out of tolerance: expected {}, got {}, tol {}",
                    name, expected, got, tolerance
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn save_json(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let value = serde_json::from_str(&content)?;
        Ok(value)
    }
}

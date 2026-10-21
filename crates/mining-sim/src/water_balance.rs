//! Complete stored/source/discharge accounting. Pumping and landing are internal
//! transfers; cumulative landed water must never be counted as current storage.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterBalance {
    pub initial_stored_m3: f64,
    pub external_inflow_m3: f64,
    pub boundary_outflow_m3: f64,
    pub baseline_sim_time: f64,
    pub legacy_baseline: bool,
}
impl WaterBalance {
    pub fn new(stored: f64, time: f64, legacy: bool) -> Self {
        Self {
            initial_stored_m3: stored,
            external_inflow_m3: 0.,
            boundary_outflow_m3: 0.,
            baseline_sim_time: time,
            legacy_baseline: legacy,
        }
    }
    pub fn error_m3(&self, stored: f64) -> f64 {
        stored + self.boundary_outflow_m3 - self.initial_stored_m3 - self.external_inflow_m3
    }
    pub fn valid(&self) -> bool {
        [
            self.initial_stored_m3,
            self.external_inflow_m3,
            self.boundary_outflow_m3,
            self.baseline_sim_time,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.)
    }
}

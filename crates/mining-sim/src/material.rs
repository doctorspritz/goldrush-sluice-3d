use serde::{Deserialize, Serialize};

/// Bulk volume plus trace gold. Parcels carry statistical weights; visual grain size
/// does not create or destroy mass. Gold is measured in grams throughout the domain.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Material {
    pub soil_m3: f64,
    pub sand_m3: f64,
    pub gravel_m3: f64,
    pub gold_g: f64,
}

impl Material {
    pub fn volume(self) -> f64 {
        self.soil_m3 + self.sand_m3 + self.gravel_m3
    }
    pub fn is_empty(self) -> bool {
        self.volume() < 1e-10 && self.gold_g < 1e-10
    }
    pub fn mass_kg(self) -> f64 {
        self.soil_m3 * 1800.0
            + self.sand_m3 * 1600.0
            + self.gravel_m3 * 2000.0
            + self.gold_g / 1000.0
    }
    pub fn add(&mut self, other: Self) {
        self.soil_m3 += other.soil_m3;
        self.sand_m3 += other.sand_m3;
        self.gravel_m3 += other.gravel_m3;
        self.gold_g += other.gold_g;
    }
    pub fn scaled(self, f: f64) -> Self {
        Self {
            soil_m3: self.soil_m3 * f,
            sand_m3: self.sand_m3 * f,
            gravel_m3: self.gravel_m3 * f,
            gold_g: self.gold_g * f,
        }
    }
    pub fn take_fraction(&mut self, fraction: f64) -> Self {
        let f = fraction.clamp(0.0, 1.0);
        let taken = self.scaled(f);
        *self = self.scaled(1.0 - f);
        taken
    }
    pub fn take_volume(&mut self, volume: f64) -> Self {
        if self.volume() < 1e-12 {
            return Self::default();
        }
        self.take_fraction(volume.max(0.0) / self.volume())
    }
    pub fn valid(self) -> bool {
        [self.soil_m3, self.sand_m3, self.gravel_m3, self.gold_g]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    }
}

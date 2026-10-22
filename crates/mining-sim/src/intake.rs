//! Custody of suspension between the terrain intake and the existing FLIP/DEM
//! inlet. This is a finite transfer buffer, not a hopper or a new fluid solver.
use crate::Material;
use serde::{Deserialize, Serialize};

pub const ADMISSION_BATCH: usize = 32;

/// Historical `solid_m3` wire field: reference bulk volumes in terrain order
/// (fines, soil, gravel, paydirt), with the same fixed density basis as Material.
/// Convert to physical grain solids before computing displacement or optics.
/// Gold is a separate mass tag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Suspension {
    pub solid_m3: [f64; 4],
    pub gold_g: f64,
}
impl Suspension {
    pub fn add(&mut self, other: Self) {
        for (a, b) in self.solid_m3.iter_mut().zip(other.solid_m3) {
            *a += b;
        }
        self.gold_g += other.gold_g;
    }
    pub fn take_fraction(&mut self, fraction: f64) -> Self {
        let f = fraction.clamp(0., 1.);
        let mut out = Self::default();
        for (a, b) in self.solid_m3.iter_mut().zip(&mut out.solid_m3) {
            *b = *a * f;
            *a -= *b;
        }
        out.gold_g = self.gold_g * f;
        self.gold_g -= out.gold_g;
        out
    }
    pub fn material(self) -> Material {
        Material {
            sand_m3: self.solid_m3[0],
            soil_m3: self.solid_m3[1],
            gravel_m3: self.solid_m3[2] + self.solid_m3[3],
            gold_g: self.gold_g,
        }
    }
    pub fn valid(self) -> bool {
        self.solid_m3
            .iter()
            .chain(std::iter::once(&self.gold_g))
            .all(|v| v.is_finite() && *v >= 0.)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct IntakePacket {
    pub water_m3: f64,
    pub suspension: Suspension,
}

/// `ready` shares the existing water_credit_m3 carrier. `pending` owns material
/// already admitted with water but awaiting a bounded grain emission batch.
/// The cumulative fields diagnose transfers and are never counted as storage.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Inlet {
    pub ready: Suspension,
    pub pending: Suspension,
    pub withdrawn: Suspension,
    pub emitted: Suspension,
    pub emission_clock: f32,
    pub sequence: u64,
    pub last_water_m3: f64,
}
impl Inlet {
    pub fn material(&self) -> Material {
        let mut m = self.ready.material();
        m.add(self.pending.material());
        m
    }
    pub fn valid(&self) -> bool {
        [self.ready, self.pending, self.withdrawn, self.emitted]
            .iter()
            .all(|s| s.valid())
            && self.emission_clock.is_finite()
            && (0.0..=0.251).contains(&self.emission_clock)
            && self.last_water_m3.is_finite()
            && self.last_water_m3 >= 0.
            && self
                .withdrawn
                .solid_m3
                .iter()
                .enumerate()
                .all(|(i, total)| {
                    let stored = self.ready.solid_m3[i] + self.pending.solid_m3[i];
                    (stored + self.emitted.solid_m3[i] - total).abs() <= (total * 1e-9).max(1e-15)
                })
            && (self.ready.gold_g + self.pending.gold_g + self.emitted.gold_g
                - self.withdrawn.gold_g)
                .abs()
                <= (self.withdrawn.gold_g * 1e-9).max(1e-15)
    }
}

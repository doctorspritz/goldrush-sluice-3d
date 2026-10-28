use serde::{Deserialize, Serialize};

pub const MINERAL_DENSITY_KG_M3: f64 = 2650.;
pub const GOLD_DENSITY_KG_M3: f64 = 19300.;
/// Reference dry bulk densities, in terrain channel order: fines, soil, gravel,
/// paydirt. These inherited prototype assumptions are not a site calibration.
pub const REFERENCE_DRY_DENSITIES_KG_M3: [f64; 4] = [1600., 1800., 2000., 2000.];

/// Reference bulk inventory plus trace gold, never literal grain-solid volume.
/// Keep these conserved quantities when actual void space/compaction is added.
/// Gold is measured in grams throughout the domain.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Material {
    pub soil_m3: f64,
    pub sand_m3: f64,
    pub gravel_m3: f64,
    pub gold_g: f64,
    /// Water carried in the material's reference void space; never grain solids.
    #[serde(default)]
    pub pore_water_m3: f64,
}

impl Material {
    pub fn volume(self) -> f64 {
        self.soil_m3 + self.sand_m3 + self.gravel_m3
    }
    pub fn is_empty(self) -> bool {
        self.volume() < 1e-10 && self.gold_g < 1e-10 && self.pore_water_m3 < 1e-12
    }
    pub fn mass_kg(self) -> f64 {
        self.mineral_mass_kg() + self.gold_g / 1000.0
    }
    pub fn wet_mass_kg(self) -> f64 {
        self.mass_kg() + self.pore_water_m3 * 1000.
    }
    /// Reference geometry remains fixed until physical compaction is modeled.
    pub fn pore_volume_m3(self) -> f64 {
        (self.volume() - self.solid_volume_m3()).max(0.)
    }
    pub fn saturation(self) -> f64 {
        (self.pore_water_m3 / self.pore_volume_m3().max(1e-30)).clamp(0., 1.)
    }
    pub fn mineral_mass_kg(self) -> f64 {
        self.soil_m3 * REFERENCE_DRY_DENSITIES_KG_M3[1]
            + self.sand_m3 * REFERENCE_DRY_DENSITIES_KG_M3[0]
            + self.gravel_m3 * REFERENCE_DRY_DENSITIES_KG_M3[2]
    }
    pub fn mineral_solid_volume_m3(self) -> f64 {
        self.mineral_mass_kg() / MINERAL_DENSITY_KG_M3
    }
    pub fn solid_volume_m3(self) -> f64 {
        self.mineral_solid_volume_m3() + self.gold_g * 0.001 / GOLD_DENSITY_KG_M3
    }
    /// Construct a specified quartz tracer without counting its reference pore
    /// space as another fraction of a physical grain.
    pub fn sand_from_solid_volume(solid_m3: f64) -> Self {
        Self {
            sand_m3: solid_m3 * MINERAL_DENSITY_KG_M3 / REFERENCE_DRY_DENSITIES_KG_M3[0],
            ..Self::default()
        }
    }
    pub fn add(&mut self, other: Self) {
        self.soil_m3 += other.soil_m3;
        self.sand_m3 += other.sand_m3;
        self.gravel_m3 += other.gravel_m3;
        self.gold_g += other.gold_g;
        self.pore_water_m3 += other.pore_water_m3;
    }
    pub fn scaled(self, f: f64) -> Self {
        Self {
            soil_m3: self.soil_m3 * f,
            sand_m3: self.sand_m3 * f,
            gravel_m3: self.gravel_m3 * f,
            gold_g: self.gold_g * f,
            pore_water_m3: self.pore_water_m3 * f,
        }
    }
    pub fn take_fraction(&mut self, fraction: f64) -> Self {
        let f = fraction.clamp(0.0, 1.0);
        let taken = self.scaled(f);
        *self = self.scaled(1.0 - f);
        taken
    }
    /// Handling occupancy includes independently added gold solids. Reference
    /// mineral packing remains the inherited prototype, not a compaction law.
    pub fn handling_volume_m3(self) -> f64 {
        self.volume() + self.gold_g * 0.001 / GOLD_DENSITY_KG_M3
    }
    pub fn take_handling_volume(&mut self, volume: f64) -> Self {
        let available = self.handling_volume_m3();
        if available <= 0. {
            return Self::default();
        }
        self.take_fraction(volume.max(0.) / available)
    }
    pub fn take_volume(&mut self, volume: f64) -> Self {
        if self.volume() == 0. && self.gold_g > 0. {
            // Pure gold still occupies finite physical volume. Keep mineral
            // reference-volume behavior unchanged for ordinary bulk feed.
            return self.take_fraction(volume.max(0.) / self.solid_volume_m3());
        }
        if self.volume() < 1e-12 {
            return Self::default();
        }
        self.take_fraction(volume.max(0.0) / self.volume())
    }
    pub fn valid(self) -> bool {
        [
            self.soil_m3,
            self.sand_m3,
            self.gravel_m3,
            self.gold_g,
            self.pore_water_m3,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0)
            && self.pore_water_m3 <= self.pore_volume_m3() + self.volume() * 1e-6 + 1e-12
    }
}

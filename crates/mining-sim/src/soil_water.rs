//! Finite column moisture and water carried with excavated material. This is
//! inventory custody on the current reference geometry, not a Richards solver,
//! compaction model or pore-pressure constitutive law.
use crate::{
    site::{CELL, WIDTH},
    Material, Site,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct SoilWater {
    /// Physical water inside reference void space, in cubic metres per column.
    pub pore_m3: Vec<f64>,
    /// Expelled water awaiting a representable f32 surface increment. This is
    /// real water custody (counted in the ledger), not an error correction.
    pub surface_m3: Vec<f64>,
}
impl SoilWater {
    pub fn stored_m3(&self) -> f64 {
        self.pore_m3.iter().chain(&self.surface_m3).sum()
    }
    pub fn valid(&self) -> bool {
        [&self.pore_m3, &self.surface_m3].iter().all(|v| {
            (v.is_empty() || v.len() == WIDTH * WIDTH)
                && v.iter().all(|x| x.is_finite() && *x >= 0. && *x < 100.)
        })
    }
    pub fn ensure(&mut self) {
        if self.pore_m3.is_empty() {
            self.pore_m3 = vec![0.; WIDTH * WIDTH];
        }
        if self.surface_m3.is_empty() {
            self.surface_m3 = vec![0.; WIDTH * WIDTH];
        }
    }
}
impl Site {
    pub fn column_material_inventory(&self, i: usize) -> Material {
        let area = (CELL as f64).powi(2);
        Material {
            soil_m3: self.world.overburden_thickness[i] as f64 * area,
            sand_m3: self.world.terrain_sediment[i] as f64 * area,
            gravel_m3: (self.world.gravel_thickness[i] as f64
                + self.world.paydirt_thickness[i] as f64)
                * area,
            gold_g: self.gold_g[i],
            ..Default::default()
        }
    }
    pub fn column_pore_volume_m3(&self, i: usize) -> f64 {
        self.column_material_inventory(i).pore_volume_m3()
    }
    pub fn column_saturation(&self, i: usize) -> f64 {
        (self.soil_water.pore_m3.get(i).copied().unwrap_or(0.)
            / self.column_pore_volume_m3(i).max(1e-30))
        .clamp(0., 1.)
    }
    /// Initialize a new claim with saturated soil only in already ponded
    /// columns. Legacy saves use zero explicit moisture; never invent water
    /// while restoring their recorded balance.
    pub(crate) fn initialize_ponded_soil(&mut self) {
        self.soil_water.ensure();
        for i in 0..WIDTH * WIDTH {
            if self.world.water_depth(i % WIDTH, i / WIDTH) > 0. {
                self.soil_water.pore_m3[i] = self.column_pore_volume_m3(i);
            }
        }
    }
    pub(crate) fn take_pore_water(&mut self, i: usize, volume: f64) -> f64 {
        let Some(water) = self.soil_water.pore_m3.get_mut(i) else {
            return 0.;
        };
        let take = volume.max(0.).min(*water);
        *water -= take;
        take
    }
    /// Release water when erosion removes its available void space. On the
    /// current column model this is a drained response. Transport with a moving
    /// wet failure block and dynamic infiltration remain separate requirements.
    pub fn reconcile_soil_water(&mut self) {
        if self.soil_water.pore_m3.is_empty() && self.soil_water.surface_m3.is_empty() {
            return;
        }
        self.soil_water.ensure();
        for i in 0..WIDTH * WIDTH {
            let capacity = self.column_pore_volume_m3(i);
            let excess = (self.soil_water.pore_m3[i] - capacity).max(0.);
            self.soil_water.pore_m3[i] -= excess;
            self.soil_water.surface_m3[i] += excess;
            self.flush_soil_water(i);
        }
    }
    pub(crate) fn add_pore_water(&mut self, i: usize, volume: f64) {
        if volume <= 0. {
            return;
        }
        self.soil_water.ensure();
        let room = (self.column_pore_volume_m3(i) - self.soil_water.pore_m3[i]).max(0.);
        let retained = volume.min(room);
        self.soil_water.pore_m3[i] += retained;
        self.soil_water.surface_m3[i] += volume - retained;
        self.flush_soil_water(i);
    }
    fn flush_soil_water(&mut self, i: usize) {
        let pending = self.soil_water.surface_m3[i];
        if pending <= 0. {
            return;
        }
        let area = (CELL as f64).powi(2);
        let before_surface = self.world.water_surface[i];
        let before_depth = self.world.water_depth(i % WIDTH, i / WIDTH);
        let requested = before_surface as f64 + pending / area;
        let mut surface = requested as f32;
        if surface as f64 > requested {
            surface = surface.next_down();
        }
        if surface <= before_surface {
            return;
        }
        let ground = self.world.ground_height(i % WIDTH, i / WIDTH);
        let mut after_depth = (surface - ground).max(0.);
        let mut added = (after_depth as f64 - before_depth as f64) * area;
        // Match the actual represented depth used by all water ledgers.
        for _ in 0..4 {
            if added <= pending {
                break;
            }
            surface = surface.next_down();
            after_depth = (surface - ground).max(0.);
            added = (after_depth as f64 - before_depth as f64) * area;
        }
        if surface <= before_surface || added <= 0. || added > pending {
            return;
        }
        let fraction = before_depth as f64 / after_depth as f64;
        let a = &mut self.hydraulic;
        for channel in [
            &mut a.suspended_sediment,
            &mut a.suspended_overburden,
            &mut a.suspended_gravel,
            &mut a.suspended_paydirt,
            &mut a.velocity_x,
            &mut a.velocity_z,
        ] {
            if let Some(value) = channel.get_mut(i) {
                *value = (*value as f64 * fraction) as f32;
            }
        }
        self.world.water_surface[i] = surface;
        self.world.suspended_sediment[i] = [
            &a.suspended_sediment,
            &a.suspended_overburden,
            &a.suspended_gravel,
            &a.suspended_paydirt,
        ]
        .iter()
        .map(|v| v.get(i).copied().unwrap_or(0.))
        .sum();
        self.soil_water.surface_m3[i] -= added;
        a.impact_dirty = true;
    }
}

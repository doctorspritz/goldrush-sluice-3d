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
    /// Horizontal momentum of queued free water (m³ × m/s), never pore water.
    #[serde(default)]
    pub surface_momentum_m4_s: Vec<[f64; 2]>,
}
impl SoilWater {
    pub fn stored_m3(&self) -> f64 {
        self.pore_m3.iter().chain(&self.surface_m3).sum()
    }
    pub fn valid(&self) -> bool {
        [&self.pore_m3, &self.surface_m3].iter().all(|v| {
            (v.is_empty() || v.len() == WIDTH * WIDTH)
                && v.iter().all(|x| x.is_finite() && *x >= 0. && *x < 100.)
        }) && (self.surface_momentum_m4_s.is_empty()
            || self.surface_momentum_m4_s.len() == WIDTH * WIDTH)
            && self.surface_momentum_m4_s.iter().enumerate().all(|(i, m)| {
                m.iter().all(|x| x.is_finite() && x.abs() < 1e6)
                    && (self.surface_m3.get(i).copied().unwrap_or(0.) > 0. || *m == [0.; 2])
            })
    }
    pub fn ensure(&mut self) {
        if self.pore_m3.is_empty() {
            self.pore_m3 = vec![0.; WIDTH * WIDTH];
        }
        if self.surface_momentum_m4_s.is_empty() {
            self.surface_momentum_m4_s = vec![[0.; 2]; WIDTH * WIDTH];
        }
        if self.surface_m3.is_empty() {
            self.surface_m3 = vec![0.; WIDTH * WIDTH];
        }
    }
}
impl Site {
    pub fn column_material_inventory(&self, i: usize) -> Material {
        Material {
            soil_m3: self.bed_class_volume_m3(i, 1),
            sand_m3: self.bed_class_volume_m3(i, 0),
            gravel_m3: self.bed_class_volume_m3(i, 2) + self.bed_class_volume_m3(i, 3),
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
    pub(crate) fn receive_surface_water(&mut self, i: usize, volume: f64, momentum: [f64; 2]) {
        self.soil_water.ensure();
        self.soil_water.surface_m3[i] += volume;
        for (axis, value) in momentum.into_iter().enumerate() {
            self.soil_water.surface_momentum_m4_s[i][axis] += value;
        }
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
        let inventory = self.suspended_class_volumes_m3(i);
        let fraction = added / pending;
        let a = &mut self.hydraulic;
        for (axis, velocity) in [&mut a.velocity_x, &mut a.velocity_z]
            .into_iter()
            .enumerate()
        {
            let moved = self.soil_water.surface_momentum_m4_s[i][axis] * fraction;
            if moved != 0. || !velocity.is_empty() {
                if velocity.is_empty() {
                    *velocity = vec![0.; WIDTH * WIDTH];
                }
                velocity[i] = ((before_depth as f64 * area * velocity[i] as f64 + moved)
                    / (after_depth as f64 * area)) as f32;
            }
            self.soil_water.surface_momentum_m4_s[i][axis] -= moved;
        }
        self.soil_water.surface_m3[i] -= added;
        if self.soil_water.surface_m3[i] == 0. {
            self.soil_water.surface_momentum_m4_s[i] = [0.; 2];
        }
        self.world.water_surface[i] = surface;
        self.set_suspension_inventory_m3(i, inventory);
    }
}

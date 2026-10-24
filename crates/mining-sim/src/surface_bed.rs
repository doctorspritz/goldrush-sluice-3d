//! Counted material below the f32 geometric layer increment. The positive
//! remainder is real reference-bulk bed inventory, separate from numerical
//! exchange feedback. Physics exchange runs in the existing GPU terrain bed stage.
use crate::{
    site::{CELL, WIDTH},
    Site,
};
const AREA: f64 = (CELL as f64) * (CELL as f64);
impl Site {
    pub fn bed_class_volume_m3(&self, i: usize, class: usize) -> f64 {
        self.bed_layer(i, class) as f64 * AREA
            + self
                .hydraulic
                .bed_remainder_m3
                .get(i)
                .map(|v| v[class])
                .unwrap_or(0.)
    }
    pub(crate) fn bed_layer(&self, i: usize, class: usize) -> f32 {
        match class {
            0 => self.world.terrain_sediment[i],
            1 => self.world.overburden_thickness[i],
            2 => self.world.gravel_thickness[i],
            3 => self.world.paydirt_thickness[i],
            _ => unreachable!(),
        }
    }
    pub(crate) fn add_bed_volume_m3(&mut self, i: usize, class: usize, volume: f64) {
        if volume <= 0. {
            return;
        }
        if self.hydraulic.bed_remainder_m3.is_empty() {
            self.hydraulic.bed_remainder_m3 = vec![[0.; 4]; WIDTH * WIDTH];
        }
        // Accumulate tiny positive transfers before adding them to a large layer.
        let low = self.hydraulic.bed_remainder_m3[i][class] + volume;
        let high = self.bed_layer(i, class);
        let quantum = (high.next_up() as f64 - high as f64) * AREA;
        if low < quantum {
            self.hydraulic.bed_remainder_m3[i][class] = low;
        } else {
            self.set_bed_volume_m3(i, class, high as f64 * AREA + low);
        }
    }
    pub(crate) fn set_bed_volume_m3(&mut self, i: usize, class: usize, volume: f64) {
        let volume = volume.max(0.);
        let mut height = (volume / AREA) as f32;
        if height as f64 * AREA > volume {
            height = height.next_down();
        }
        let remainder = (volume - height as f64 * AREA).max(0.);
        match class {
            0 => self.world.terrain_sediment[i] = height,
            1 => self.world.overburden_thickness[i] = height,
            2 => self.world.gravel_thickness[i] = height,
            3 => self.world.paydirt_thickness[i] = height,
            _ => unreachable!(),
        }
        if remainder > 0. || !self.hydraulic.bed_remainder_m3.is_empty() {
            if self.hydraulic.bed_remainder_m3.is_empty() {
                self.hydraulic.bed_remainder_m3 = vec![[0.; 4]; WIDTH * WIDTH];
            }
            self.hydraulic.bed_remainder_m3[i][class] = remainder;
        }
    }
}

impl Site {
    pub fn suspended_class_volumes_m3(&self, i: usize) -> [f64; 4] {
        let volume = self.world.water_depth(i % WIDTH, i / WIDTH) as f64 * AREA;
        let a = &self.hydraulic;
        let fields = [
            &a.suspended_sediment,
            &a.suspended_overburden,
            &a.suspended_gravel,
            &a.suspended_paydirt,
        ];
        std::array::from_fn(|ch| {
            fields[ch].get(i).copied().unwrap_or(0.) as f64 * volume
                + a.suspended_remainder_m3.get(i).map(|v| v[ch]).unwrap_or(0.)
        })
    }
    pub(crate) fn set_suspension_inventory_m3(&mut self, i: usize, inventory: [f64; 4]) {
        let volume = self.world.water_depth(i % WIDTH, i / WIDTH) as f64 * AREA;
        if volume == 0. {
            for (class, amount) in inventory.iter().enumerate() {
                self.add_bed_volume_m3(i, class, *amount);
            }
            self.world.water_surface[i] = self.world.ground_height(i % WIDTH, i / WIDTH);
        }
        let a = &mut self.hydraulic;
        let mut residual = [0.; 4];
        let mut aggregate = 0.;
        for (ch, field) in [
            &mut a.suspended_sediment,
            &mut a.suspended_overburden,
            &mut a.suspended_gravel,
            &mut a.suspended_paydirt,
        ]
        .into_iter()
        .enumerate()
        {
            let old = field.get(i).copied().unwrap_or(0.);
            let mut c = if volume == 0. {
                0.
            } else if old as f64 * volume == inventory[ch] {
                old
            } else {
                (inventory[ch] / volume) as f32
            };
            if c as f64 * volume > inventory[ch] {
                c = c.next_down();
            }
            if volume > 0. {
                residual[ch] = (inventory[ch] - c as f64 * volume).max(0.);
            }
            if c != 0. || !field.is_empty() {
                if field.is_empty() {
                    *field = vec![0.; WIDTH * WIDTH];
                }
                field[i] = c;
            }
            aggregate += c;
        }
        if residual != [0.; 4] || !a.suspended_remainder_m3.is_empty() {
            if a.suspended_remainder_m3.is_empty() {
                a.suspended_remainder_m3 = vec![[0.; 4]; WIDTH * WIDTH];
            }
            a.suspended_remainder_m3[i] = residual;
        }
        self.world.suspended_sediment[i] = aggregate;
        a.impact_dirty = true;
    }
    /// Project GPU depths into World surface heights without creating water.
    /// Unrepresentable free water retains its momentum in the positive queue;
    /// concentration rounding retains actual mobile stock separately.
    pub fn project_terrain_water(&mut self, depths: &[f32]) {
        assert_eq!(depths.len(), WIDTH * WIDTH);
        assert!(depths.iter().all(|h| h.is_finite() && *h >= 0.));
        self.soil_water.ensure();
        for (i, &depth) in depths.iter().enumerate() {
            let a = &self.hydraulic;
            let fields = [
                &a.suspended_sediment,
                &a.suspended_overburden,
                &a.suspended_gravel,
                &a.suspended_paydirt,
            ];
            let inventory = std::array::from_fn(|ch| {
                fields[ch].get(i).copied().unwrap_or(0.) as f64 * depth as f64 * AREA
                    + a.suspended_remainder_m3.get(i).map(|v| v[ch]).unwrap_or(0.)
            });
            let ground = self.world.ground_height(i % WIDTH, i / WIDTH);
            let mut surface = ground + depth;
            while (surface - ground).max(0.) > depth {
                surface = surface.next_down();
            }
            let represented = (surface - ground).max(0.);
            let pending = (depth as f64 - represented as f64).max(0.) * AREA;
            self.soil_water.surface_m3[i] += pending;
            for (axis, velocity) in [&a.velocity_x, &a.velocity_z].into_iter().enumerate() {
                self.soil_water.surface_momentum_m4_s[i][axis] +=
                    pending * velocity.get(i).copied().unwrap_or(0.) as f64;
            }
            self.world.water_surface[i] = surface;
            self.set_suspension_inventory_m3(i, inventory);
            if represented == 0. {
                for velocity in [
                    &mut self.hydraulic.velocity_x,
                    &mut self.hydraulic.velocity_z,
                ] {
                    if let Some(v) = velocity.get_mut(i) {
                        *v = 0.;
                    }
                }
            }
        }
    }
}

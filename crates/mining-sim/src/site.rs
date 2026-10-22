use crate::{Material, Vec3};
use serde::{Deserialize, Serialize};
use sim3d::World;

pub const WIDTH: usize = 128;
pub const CELL: f32 = 0.25;
pub const EXTENT: f32 = WIDTH as f32 * CELL;
pub const BED_STABILITY_MICROS: u32 = 1_500_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BedAgeUnit {
    #[serde(rename = "legacy_20hz_steps")]
    Legacy20HzSteps,
    #[default]
    #[serde(rename = "microseconds")]
    Microseconds,
}
fn legacy_bed_age_unit() -> BedAgeUnit {
    BedAgeUnit::Legacy20HzSteps
}

#[derive(Clone)]
pub struct Site {
    pub world: World,
    pub gold_g: Vec<f64>,
    pub revision: u64,
    pub excavated: Material,
    pub hydraulic: HeightfieldState,
}

/// Authoritative extra channels from the original GPU heightfield. Geology,
/// surface and trace-gold arrays remain in Site/World.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct HeightfieldState {
    pub velocity_x: Vec<f32>,
    pub velocity_z: Vec<f32>,
    /// CPU water impacts changed momentum and suspension since the latest GPU step.
    #[serde(default)]
    pub impact_dirty: bool,
    pub flux_x: Vec<f32>,
    pub flux_z: Vec<f32>,
    /// Separate solid/water volume concentrations, not layer depths or kg/m².
    pub suspended_sediment: Vec<f32>,
    pub suspended_overburden: Vec<f32>,
    pub suspended_gravel: Vec<f32>,
    pub suspended_paydirt: Vec<f32>,
    /// Wet-bed age since disturbance, with an explicit clock for old saves.
    pub settling_time: Vec<u32>,
    #[serde(default = "legacy_bed_age_unit")]
    pub settling_time_unit: BedAgeUnit,
    /// Signed numerical exchange remainder (metres of class volume per area).
    /// Feedback prevents accumulated bias; this is not another material store.
    #[serde(default)]
    pub bed_exchange_error: Vec<[f32; 4]>,
    pub suspended_gold_g: Vec<f64>,
    pub clock: f32,
    pub steps: u64,
    /// Material actually discharged through map boundaries, retained in ledgers.
    #[serde(default)]
    pub exported_material_m3: [f64; 4],
    #[serde(default)]
    pub exported_gold_g: f64,
}
impl HeightfieldState {
    pub fn normalize_bed_age(&mut self) {
        if self.settling_time_unit == BedAgeUnit::Legacy20HzSteps {
            for age in &mut self.settling_time {
                *age = age.saturating_mul(50_000).min(BED_STABILITY_MICROS);
            }
            self.settling_time_unit = BedAgeUnit::Microseconds;
        }
    }
    pub fn valid(&self) -> bool {
        let n = WIDTH * WIDTH;
        [
            &self.velocity_x,
            &self.velocity_z,
            &self.flux_x,
            &self.flux_z,
            &self.suspended_sediment,
            &self.suspended_overburden,
            &self.suspended_gravel,
            &self.suspended_paydirt,
        ]
        .iter()
        .all(|v| {
            (v.is_empty() || v.len() == n) && v.iter().all(|x| x.is_finite() && x.abs() < 10000.0)
        }) && [
            &self.suspended_sediment,
            &self.suspended_overburden,
            &self.suspended_gravel,
            &self.suspended_paydirt,
        ]
        .iter()
        .all(|v| v.iter().all(|c| *c >= 0.))
            && (self.settling_time.is_empty() || self.settling_time.len() == n)
            && (self.bed_exchange_error.is_empty() || self.bed_exchange_error.len() == n)
            && self
                .bed_exchange_error
                .iter()
                .flatten()
                .all(|v| v.is_finite() && v.abs() < 1.)
            && self.settling_time.iter().all(|age| {
                *age <= match self.settling_time_unit {
                    BedAgeUnit::Legacy20HzSteps => 255,
                    BedAgeUnit::Microseconds => BED_STABILITY_MICROS,
                }
            })
            && (self.suspended_gold_g.is_empty() || self.suspended_gold_g.len() == n)
            && self
                .suspended_gold_g
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0)
            && self.clock.is_finite()
            && (0.0..=0.051).contains(&self.clock)
            && self
                .exported_material_m3
                .iter()
                .all(|v| v.is_finite() && *v >= 0.)
            && self.exported_gold_g.is_finite()
            && self.exported_gold_g >= 0.
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SiteSnapshot {
    bedrock: Vec<f32>,
    paydirt: Vec<f32>,
    overburden: Vec<f32>,
    sediment: Vec<f32>,
    water: Vec<f32>,
    flow_x: Vec<f32>,
    flow_z: Vec<f32>,
    gold_g: Vec<f64>,
    excavated: Material,
    gravel: Vec<f32>,
    hydraulic: HeightfieldState,
}
impl SiteSnapshot {
    pub(crate) fn bed_age_unit(&self) -> BedAgeUnit {
        self.hydraulic.settling_time_unit
    }
}

impl Default for Site {
    fn default() -> Self {
        Self::new()
    }
}

impl Site {
    pub fn new() -> Self {
        let mut world = World::new(WIDTH, WIDTH, CELL, 0.0);
        world.params.open_boundaries = true;
        world.params.manning_n = 0.036;
        let mut gold_g = vec![0.0; WIDTH * WIDTH];
        for z in 0..WIDTH {
            for x in 0..WIDTH {
                let i = world.idx(x, z);
                let px = x as f32 * CELL;
                let pz = z as f32 * CELL;
                let river_x = Self::river_center(pz);
                let bank = ((px - river_x).abs() - 1.65).max(0.0);
                let river_cut = (bank / 1.1).clamp(0.0, 1.0);
                let ridge =
                    (-(px - 19.5).powi(2) / 40.0).exp() * ((pz - 19.8) / 2.6).clamp(0.0, 1.0);
                let noise = (px * 0.51).sin() * (pz * 0.37).cos() * 0.09;
                let remote_hill = ((px - 25.0) / 7.0).max(0.0).powi(2) * 1.4;
                world.bedrock_elevation[i] = 0.02 + pz * 0.004 + noise * river_cut;
                world.paydirt_thickness[i] = (0.16 + 0.67 * ridge + remote_hill) * river_cut;
                world.overburden_thickness[i] = 0.10 * river_cut;
                world.gravel_thickness[i] = 0.0;
                // Flat service pad keeps the excavator and hopper accessible.
                if (11.0..25.0).contains(&px) && (12.0..20.0).contains(&pz) {
                    world.bedrock_elevation[i] = 0.05;
                    world.paydirt_thickness[i] = 0.20;
                    world.overburden_thickness[i] = 0.10;
                }
                // A pregraded, excavatable tailrace provides an actual gravity
                // route from the installed plant to the river. No water source
                // or prescribed runoff velocity is added here.
                let end_x = Self::river_center(18.5);
                let t = (15.25 - px) / (15.25 - end_x);
                if (0.0..=1.0).contains(&t) {
                    let center_z = 19.7 + (18.5 - 19.7) * t;
                    let distance = (pz - center_z).abs();
                    let bank = ((0.70 - distance) / 0.40).clamp(0., 1.);
                    let weight = bank * bank * (3. - 2. * bank);
                    let bed = 0.27 + (0.02 + 18.5 * 0.004 - 0.27) * t;
                    let old = world.ground_height(x, z);
                    let mut cut = ((old - bed).max(0.) * weight)
                        .min(world.overburden_thickness[i] + world.paydirt_thickness[i]);
                    let soil = cut.min(world.overburden_thickness[i]);
                    world.overburden_thickness[i] -= soil;
                    cut -= soil;
                    world.paydirt_thickness[i] -= cut;
                }
                let h = world.ground_height(x, z);
                world.water_surface[i] = if (px - river_x).abs() < 1.7 {
                    h + 0.13
                } else {
                    h
                };
                let grade =
                    9.0 + 13.0 * (-(px - 19.5).powi(2) / 28.0 - (pz - 23.0).powi(2) / 36.0).exp();
                gold_g[i] = world.paydirt_thickness[i] as f64 * (CELL * CELL) as f64 * grade as f64;
            }
        }
        Self {
            world,
            gold_g,
            revision: 0,
            excavated: Material::default(),
            hydraulic: HeightfieldState::default(),
        }
    }
    pub fn river_center(z: f32) -> f32 {
        6.4 + (z * 0.15).sin() * 0.7
    }
    pub fn height(&self, x: f32, z: f32) -> f32 {
        let fx = (x / CELL).clamp(0.0, (WIDTH - 1) as f32);
        let fz = (z / CELL).clamp(0.0, (WIDTH - 1) as f32);
        let ix = fx.floor() as usize;
        let iz = fz.floor() as usize;
        let nx = (ix + 1).min(WIDTH - 1);
        let nz = (iz + 1).min(WIDTH - 1);
        let a = self.world.ground_height(ix, iz) * (1.0 - fx.fract())
            + self.world.ground_height(nx, iz) * fx.fract();
        let b = self.world.ground_height(ix, nz) * (1.0 - fx.fract())
            + self.world.ground_height(nx, nz) * fx.fract();
        a * (1.0 - fz.fract()) + b * fz.fract()
    }
    pub fn water_depth(&self, x: f32, z: f32) -> f32 {
        let ix = (x / CELL).clamp(0.0, (WIDTH - 1) as f32) as usize;
        let iz = (z / CELL).clamp(0.0, (WIDTH - 1) as f32) as usize;
        self.world.water_depth(ix, iz)
    }
    pub fn bounds(p: Vec3, margin: f32) -> bool {
        p.is_finite()
            && p.x > margin
            && p.z > margin
            && p.x < EXTENT - margin
            && p.z < EXTENT - margin
    }
    pub fn remaining_gold(&self) -> f64 {
        self.gold_g.iter().sum::<f64>() + self.hydraulic.suspended_gold_g.iter().sum::<f64>()
    }
    pub fn water_volume_m3(&self) -> f64 {
        (0..WIDTH * WIDTH)
            .map(|i| self.world.water_depth(i % WIDTH, i / WIDTH) as f64)
            .sum::<f64>()
            * (CELL * CELL) as f64
    }
    pub fn bulk_solid_volume_m3(&self) -> f64 {
        let ground = [
            &self.world.terrain_sediment,
            &self.world.gravel_thickness,
            &self.world.overburden_thickness,
            &self.world.paydirt_thickness,
        ]
        .iter()
        .flat_map(|v| v.iter())
        .map(|v| *v as f64)
        .sum::<f64>();
        let fields = [
            &self.hydraulic.suspended_sediment,
            &self.hydraulic.suspended_overburden,
            &self.hydraulic.suspended_gravel,
            &self.hydraulic.suspended_paydirt,
        ];
        let suspended = if fields.iter().all(|v| v.is_empty()) {
            0.
        } else {
            (0..WIDTH * WIDTH)
                .map(|i| {
                    let concentration = fields
                        .iter()
                        .map(|v| v.get(i).copied().unwrap_or(0.) as f64)
                        .sum::<f64>();
                    concentration * self.world.water_depth(i % WIDTH, i / WIDTH) as f64
                })
                .sum::<f64>()
        };
        (ground + suspended) * (CELL as f64).powi(2)
    }
    /// Add the existing circular river source as clean water. Suspension is
    /// stored as concentration, so keeping C fixed would create dirt whenever
    /// source water is added to an eroding cell. Return the represented volume
    /// actually added after f32 surface/depth rounding for the water ledger.
    pub fn add_clean_inflow(&mut self, position: Vec3, rate: f32, radius: f32, dt: f32) -> f64 {
        if !position.is_finite()
            || !Self::bounds(position, 0.)
            || !rate.is_finite()
            || !radius.is_finite()
            || !dt.is_finite()
            || rate <= 0.
            || radius <= 0.
            || dt <= 0.
            || !(rate * dt).is_finite()
        {
            return 0.;
        }
        let radius = radius.min(WIDTH as f32 * CELL);
        let r = (radius / CELL).ceil() as i32;
        let (cx, cz) = ((position.x / CELL) as i32, (position.z / CELL) as i32);
        let mut old = Vec::new();
        for dz in -r..=r {
            for dx in -r..=r {
                let (x, z) = (cx + dx, cz + dz);
                if (dx * dx + dz * dz) as f32 > (radius / CELL).powi(2)
                    || !(0..WIDTH as i32).contains(&x)
                    || !(0..WIDTH as i32).contains(&z)
                {
                    continue;
                }
                let (x, z) = (x as usize, z as usize);
                old.push((z * WIDTH + x, self.world.water_depth(x, z)));
            }
        }
        self.world.add_inflow(position, rate, radius, dt);
        let mut added = 0.;
        for (i, before) in old {
            let after = self.world.water_depth(i % WIDTH, i / WIDTH);
            if after <= before {
                continue;
            }
            added += (after as f64 - before as f64) * (CELL as f64).powi(2);
            let dilution = before as f64 / after as f64;
            let mut aggregate = 0.;
            for field in [
                &mut self.hydraulic.suspended_sediment,
                &mut self.hydraulic.suspended_overburden,
                &mut self.hydraulic.suspended_gravel,
                &mut self.hydraulic.suspended_paydirt,
            ] {
                if let Some(c) = field.get_mut(i) {
                    *c = (*c as f64 * dilution) as f32;
                    aggregate += *c;
                }
            }
            self.world.suspended_sediment[i] = aggregate;
        }
        if added > 0. {
            self.hydraulic.impact_dirty = true;
            self.revision += 1;
        }
        added
    }
    pub fn dig(&mut self, p: Vec3, radius: f32, depth: f32, capacity: f64) -> Material {
        if !p.is_finite()
            || !radius.is_finite()
            || !depth.is_finite()
            || !capacity.is_finite()
            || radius <= 0.0
            || depth <= 0.0
            || capacity <= 1e-8
            || !Self::bounds(p, 0.5)
        {
            return Material::default();
        }
        let r = ((radius / CELL).ceil() as i32).min(WIDTH as i32);
        let (cx, cz) = ((p.x / CELL) as i32, (p.z / CELL) as i32);
        let mut cells = Vec::new();
        for z in (cz - r).max(0)..=(cz + r).min(WIDTH as i32 - 1) {
            for x in (cx - r).max(0)..=(cx + r).min(WIDTH as i32 - 1) {
                if ((x - cx).pow(2) + (z - cz).pow(2)) as f32 <= (radius / CELL).powi(2) {
                    cells.push(self.world.idx(x as usize, z as usize));
                }
            }
        }
        let depth = depth
            .min((capacity / (cells.len().max(1) as f64 * (CELL * CELL) as f64)) as f32 * 0.999);
        self.excavate_columns(cells.into_iter().map(|i| (i, depth)).collect())
    }
    pub fn deposit(&mut self, p: Vec3, m: Material) {
        let cx = (p.x / CELL).clamp(2.0, (WIDTH - 3) as f32) as usize;
        let cz = (p.z / CELL).clamp(2.0, (WIDTH - 3) as f32) as usize;
        for dz in -2i32..=2 {
            for dx in -2i32..=2 {
                let i = self
                    .world
                    .idx((cx as i32 + dx) as usize, (cz as i32 + dz) as usize);
                let scale = 1.0 / (25.0 * (CELL * CELL) as f64);
                let depth = self.world.water_depth(i % WIDTH, i / WIDTH);
                self.world.paydirt_thickness[i] += (m.gravel_m3 * scale) as f32;
                self.world.overburden_thickness[i] += (m.soil_m3 * scale) as f32;
                self.world.terrain_sediment[i] += (m.sand_m3 * scale) as f32;
                self.gold_g[i] += m.gold_g / 25.0;
                self.world.water_surface[i] =
                    self.world.ground_height(i % WIDTH, i / WIDTH) + depth;
            }
        }
        self.revision += 1;
    }
    /// Withdraw a well-mixed local sample, including all four suspension
    /// channels and their gold tag. Use the represented depth change, not the
    /// requested f32 decrement, so the receiving ledger matches the donor.
    pub fn withdraw_intake(&mut self, p: Vec3, volume: f32) -> crate::intake::IntakePacket {
        let mut packet = crate::intake::IntakePacket::default();
        if !p.is_finite() || !Self::bounds(p, 0.) || !volume.is_finite() || volume <= 0. {
            return packet;
        }
        let x = (p.x / CELL) as i32;
        let z = (p.z / CELL) as i32;
        // A compact 3x3 finite-volume intake. Sample all wet donor cells
        // proportionally; draining the first corner of a 7x7 square biased
        // both the source location and the concentration presented to the pump.
        let mut cells = Vec::with_capacity(9);
        for dz in -1..=1 {
            for dx in -1..=1 {
                let ix = x + dx;
                let iz = z + dz;
                if ix < 0 || iz < 0 || ix >= WIDTH as i32 || iz >= WIDTH as i32 {
                    continue;
                }
                let i = self.world.idx(ix as usize, iz as usize);
                let before = self.world.water_depth(ix as usize, iz as usize) as f64;
                cells.push((i, before));
            }
        }
        let area = (CELL as f64).powi(2);
        let available = cells.iter().map(|(_, depth)| depth * area).sum::<f64>();
        let fraction = (volume as f64 / available.max(1e-30)).min(1.);
        for (i, before) in cells {
            let requested = before * area * fraction;
            if requested <= 0. {
                continue;
            }
            let target = (self.world.water_surface[i] as f64 - requested / area)
                .max(self.world.ground_height(i % WIDTH, i / WIDTH) as f64);
            let mut surface = target as f32;
            // Round toward retained water: a sub-ULP request cannot draw
            // extra water merely to satisfy a nominal pump setting.
            if (surface as f64) < target {
                surface = surface.next_up();
            }
            self.world.water_surface[i] = surface;
            let after = self.world.water_depth(i % WIDTH, i / WIDTH) as f64;
            let taken = (before - after).max(0.) * area;
            for (channel, field) in [
                &self.hydraulic.suspended_sediment,
                &self.hydraulic.suspended_overburden,
                &self.hydraulic.suspended_gravel,
                &self.hydraulic.suspended_paydirt,
            ]
            .iter()
            .enumerate()
            {
                packet.suspension.solid_m3[channel] +=
                    field.get(i).copied().unwrap_or(0.) as f64 * taken;
            }
            if let Some(gold) = self.hydraulic.suspended_gold_g.get_mut(i) {
                let removed = *gold * ((before - after) / before.max(1e-30)).clamp(0., 1.);
                *gold -= removed;
                packet.suspension.gold_g += removed;
            }
            packet.water_m3 += taken;
        }
        if packet.water_m3 > 0. {
            self.hydraulic.impact_dirty = true;
            self.revision += 1;
        }
        packet
    }
    pub fn raycast(&self, origin: Vec3, direction: Vec3, reach: f32) -> Option<Vec3> {
        if !origin.is_finite() || !direction.is_finite() {
            return None;
        }
        let direction = direction.normalize_or_zero();
        let mut t = 0.0;
        while t < reach {
            let p = origin + direction * t;
            if Self::bounds(p, 0.1) && p.y <= self.height(p.x, p.z) {
                return Some(Vec3::new(p.x, self.height(p.x, p.z), p.z));
            }
            t += 0.06;
        }
        None
    }
    pub fn snapshot(&self) -> SiteSnapshot {
        SiteSnapshot {
            bedrock: self.world.bedrock_elevation.clone(),
            paydirt: self.world.paydirt_thickness.clone(),
            overburden: self.world.overburden_thickness.clone(),
            sediment: self.world.terrain_sediment.clone(),
            water: self.world.water_surface.clone(),
            flow_x: self.world.water_flow_x.clone(),
            flow_z: self.world.water_flow_z.clone(),
            gold_g: self.gold_g.clone(),
            excavated: self.excavated,
            gravel: self.world.gravel_thickness.clone(),
            hydraulic: self.hydraulic.clone(),
        }
    }
    pub fn restore(s: SiteSnapshot) -> Result<Self, String> {
        let n = WIDTH * WIDTH;
        for (name, v) in [
            ("bedrock", &s.bedrock),
            ("paydirt", &s.paydirt),
            ("gravel", &s.gravel),
            ("overburden", &s.overburden),
            ("sediment", &s.sediment),
            ("water", &s.water),
        ] {
            let minimum = if name == "bedrock" || name == "water" {
                -100.0
            } else {
                0.0
            };
            if v.len() != n
                || v.iter()
                    .any(|v| !v.is_finite() || *v < minimum || *v > 100.0)
            {
                return Err(format!("Invalid {name} terrain data"));
            }
        }
        if s.gold_g.len() != n
            || s.gold_g.iter().any(|v| !v.is_finite() || *v < 0.0)
            || s.flow_x.len() != (WIDTH + 1) * WIDTH
            || s.flow_z.len() != WIDTH * (WIDTH + 1)
            || s.flow_x
                .iter()
                .chain(&s.flow_z)
                .any(|v| !v.is_finite() || v.abs() > 100.0)
            || !s.excavated.valid()
            || !s.hydraulic.valid()
        {
            return Err("Invalid site quantities".into());
        }
        let mut site = Self::new();
        site.world.bedrock_elevation = s.bedrock;
        site.world.paydirt_thickness = s.paydirt;
        site.world.overburden_thickness = s.overburden;
        site.world.terrain_sediment = s.sediment;
        site.world.water_surface = s.water;
        site.world.water_flow_x = s.flow_x;
        site.world.water_flow_z = s.flow_z;
        site.gold_g = s.gold_g;
        site.excavated = s.excavated;
        site.world.gravel_thickness = s.gravel;
        site.hydraulic = s.hydraulic;
        site.hydraulic.normalize_bed_age();
        site.revision = 1;
        Ok(site)
    }
}

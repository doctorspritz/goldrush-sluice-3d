use crate::{Material, Vec3};
use serde::{Deserialize, Serialize};
use sim3d::World;

pub const WIDTH: usize = 128;
pub const CELL: f32 = 0.25;
pub const EXTENT: f32 = WIDTH as f32 * CELL;

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
    pub flux_x: Vec<f32>,
    pub flux_z: Vec<f32>,
    pub suspended_sediment: Vec<f32>,
    pub suspended_overburden: Vec<f32>,
    pub suspended_gravel: Vec<f32>,
    pub suspended_paydirt: Vec<f32>,
    pub settling_time: Vec<u32>,
    pub suspended_gold_g: Vec<f64>,
    pub clock: f32,
    pub steps: u64,
}
impl HeightfieldState {
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
        }) && (self.settling_time.is_empty() || self.settling_time.len() == n)
            && (self.suspended_gold_g.is_empty() || self.suspended_gold_g.len() == n)
            && self
                .suspended_gold_g
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0)
            && self.clock.is_finite()
            && (0.0..=0.051).contains(&self.clock)
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
        // Uniform depth cap bounds the actual set of cells touched by World::excavate,
        // including its integer-radius footprint (which is larger than pi*r*r).
        let r = (radius / CELL).ceil() as i32;
        let cell_count = (-r..=r)
            .flat_map(|z| (-r..=r).map(move |x| (x, z)))
            .filter(|(x, z)| (*x as f32).powi(2) + (*z as f32).powi(2) <= (radius / CELL).powi(2))
            .count();
        let depth =
            depth.min((capacity / (cell_count as f64 * (CELL * CELL) as f64)) as f32 * 0.999);
        let before_pd = self.world.paydirt_thickness.clone();
        let before_gr = self.world.gravel_thickness.clone();
        let before_ob = self.world.overburden_thickness.clone();
        let before_sd = self.world.terrain_sediment.clone();
        self.world.excavate(p, radius, depth);
        let mut batch = Material::default();
        for i in 0..WIDTH * WIDTH {
            let pd = (before_pd[i] - self.world.paydirt_thickness[i]).max(0.0) as f64;
            let gold = if before_pd[i] > 0.0 {
                self.gold_g[i] * pd / before_pd[i] as f64
            } else {
                0.0
            };
            self.gold_g[i] = (self.gold_g[i] - gold).max(0.0);
            batch.gold_g += gold;
            batch.gravel_m3 += (pd
                + (before_gr[i] - self.world.gravel_thickness[i]).max(0.0) as f64)
                * (CELL * CELL) as f64;
            batch.soil_m3 += (before_ob[i] - self.world.overburden_thickness[i]).max(0.0) as f64
                * (CELL * CELL) as f64;
            batch.sand_m3 += (before_sd[i] - self.world.terrain_sediment[i]).max(0.0) as f64
                * (CELL * CELL) as f64;
        }
        if !batch.is_empty() {
            self.revision += 1;
            self.excavated.add(batch);
        }
        batch
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
                self.world.paydirt_thickness[i] += (m.gravel_m3 * scale) as f32;
                self.world.overburden_thickness[i] += (m.soil_m3 * scale) as f32;
                self.world.terrain_sediment[i] += (m.sand_m3 * scale) as f32;
                self.gold_g[i] += m.gold_g / 25.0;
            }
        }
        self.revision += 1;
    }
    pub fn withdraw_water(&mut self, p: Vec3, volume: f32) -> f32 {
        let x = (p.x / CELL) as i32;
        let z = (p.z / CELL) as i32;
        let mut remaining = volume.max(0.0);
        for dz in -3..=3 {
            for dx in -3..=3 {
                let ix = x + dx;
                let iz = z + dz;
                if ix < 0 || iz < 0 || ix >= WIDTH as i32 || iz >= WIDTH as i32 {
                    continue;
                }
                let i = self.world.idx(ix as usize, iz as usize);
                let available = self.world.water_depth(ix as usize, iz as usize) * CELL * CELL;
                let taken = available.min(remaining);
                self.world.water_surface[i] -= taken / (CELL * CELL);
                remaining -= taken;
            }
        }
        volume.max(0.0) - remaining
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
        site.revision = 1;
        Ok(site)
    }
}

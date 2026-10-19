use crate::{Material, Site, Vec3};
use glam::{Mat3, Quat};
use serde::{Deserialize, Serialize};

pub const LENGTH: f32 = 3.4;
pub const CHANNEL_WIDTH: f32 = 0.72;
pub const HOPPER_X: f32 = 14.6;
pub const HOPPER_Z: f32 = 17.0;
pub const PARCEL_VOLUME: f64 = 0.002;
pub const MAX_GRAINS: usize = 4096;
pub const FLUID_CELL: f32 = 0.02;
pub const WATER_PARTICLE_M3: f64 = (FLUID_CELL * FLUID_CELL * FLUID_CELL / 8.0) as f64;
pub const MAX_WATER_PARTICLES: usize = 160_000;

/// Original FLIP/APIC state, in the local simulation grid. Kept in the save,
/// rather than recreated when a camera changes or a shift is restored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub affine: Mat3,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Placement {
    pub inlet: Vec3,
    pub yaw: f32,
    pub slope_deg: f32,
}
impl Placement {
    pub fn suggested(site: &Site) -> Self {
        Self {
            inlet: Vec3::new(
                HOPPER_X,
                site.height(HOPPER_X, HOPPER_Z) + 0.85,
                HOPPER_Z + 0.95,
            ),
            yaw: 0.0,
            slope_deg: 6.0,
        }
    }
    pub fn forward(self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos())
    }
    pub fn right(self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin())
    }
    pub fn to_world(self, p: Vec3) -> Vec3 {
        self.inlet
            + self.forward() * p.x
            + Vec3::Y * (p.y - p.x * self.slope_deg.to_radians().tan())
            + self.right() * p.z
    }
    pub fn floor_y(self, x: f32) -> f32 {
        self.inlet.y - x * self.slope_deg.to_radians().tan()
    }
    pub fn outlet(self) -> Vec3 {
        self.to_world(Vec3::new(LENGTH, 0.0, 0.0))
    }
    pub fn validate(self, site: &Site, machine: Vec3) -> Result<(), String> {
        if !self.inlet.is_finite()
            || !self.yaw.is_finite()
            || !(3.0..=12.0).contains(&self.slope_deg)
        {
            return Err("Slope must be 3–12 degrees".into());
        }
        let port = Self::suggested(site).inlet;
        if Vec3::new(self.inlet.x - port.x, 0.0, self.inlet.z - port.z).length() > 1.05 {
            return Err("Move the inlet to the hopper outlet".into());
        }
        if self.forward().dot(Vec3::Z) < 0.6 {
            return Err("Face downstream from the hopper".into());
        }
        let pump = Vec3::new(Site::river_center(HOPPER_Z), 0.0, HOPPER_Z);
        if Vec3::new(self.inlet.x - pump.x, 0.0, self.inlet.z - pump.z).length() > 12.0 {
            return Err("Water hose cannot reach".into());
        }
        for i in 0..=12 {
            let x = LENGTH * i as f32 / 12.0;
            for z in [-CHANNEL_WIDTH * 0.5, CHANNEL_WIDTH * 0.5] {
                let p = self.to_world(Vec3::new(x, 0.0, z));
                if !Site::bounds(p, 0.7) {
                    return Err("Outside the claim".into());
                }
                let clearance = p.y - site.height(p.x, p.z);
                if !(0.10..=1.7).contains(&clearance) {
                    return Err("Adjust position or slope: supports must reach firm ground".into());
                }
                if Vec3::new(p.x - machine.x, 0.0, p.z - machine.z).length() < 1.9 {
                    return Err("The excavator blocks this placement".into());
                }
                if site.water_depth(p.x, p.z) > 0.25 {
                    return Err("Place supports on dry ground".into());
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GrainKind {
    Sand,
    Gravel,
    Gold,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Grain {
    pub kind: GrainKind,
    pub position: Vec3,
    pub velocity: Vec3,
    pub material: Material,
    pub resting: bool,
    pub diameter_m: f32,
    pub affine: Mat3,
    pub rotation: Quat,
    pub angular_velocity: Vec3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sluice {
    pub placement: Option<Placement>,
    pub hopper: Material,
    pub grains: Vec<Grain>,
    pub tailings: Material,
    pub concentrate: Material,
    pub pump_on: bool,
    pub feed_on: bool,
    pub flow_m3s: f32,
    pub feed_m3s: f32,
    pub processed_m3: f64,
    pub water_used_m3: f64,
    pub cleanups: u32,
    pub time: f32,
    pub water: Vec<WaterParticle>,
    pub water_credit_m3: f64,
    pub water_returned_m3: f64,
    pub solver_steps: u64,
    pub contacts: sim3d::clump::ContactHistory3D,
    feed_credit: f64,
    sequence: u64,
}
impl Default for Sluice {
    fn default() -> Self {
        Self {
            placement: None,
            hopper: Material::default(),
            grains: Vec::new(),
            tailings: Material::default(),
            concentrate: Material::default(),
            pump_on: false,
            feed_on: false,
            flow_m3s: 0.006,
            feed_m3s: 0.006,
            processed_m3: 0.0,
            water_used_m3: 0.0,
            cleanups: 0,
            time: 0.0,
            water: Vec::new(),
            water_credit_m3: 0.0,
            water_returned_m3: 0.0,
            solver_steps: 0,
            contacts: Default::default(),
            feed_credit: 0.0,
            sequence: 0,
        }
    }
}
impl Sluice {
    pub fn hopper_position(site: &Site) -> Vec3 {
        Vec3::new(HOPPER_X, site.height(HOPPER_X, HOPPER_Z), HOPPER_Z)
    }
    pub fn hopper_top(site: &Site) -> f32 {
        site.height(HOPPER_X, HOPPER_Z) + 1.55
    }
    pub fn add_to_hopper(&mut self, material: Material) -> Material {
        let mut input = material;
        let room = (0.85 - self.hopper.volume()).max(0.0);
        self.hopper.add(input.take_volume(room));
        input
    }
    pub fn active_material(&self) -> Material {
        let mut m = self.hopper;
        for g in &self.grains {
            m.add(g.material);
        }
        m
    }
    /// Volume and positions come from FLIP; there is no analytic water-depth
    /// or sediment-capture model in the game layer.
    pub fn water_depth(&self) -> f32 {
        self.water.len() as f32 * WATER_PARTICLE_M3 as f32 / (LENGTH * CHANNEL_WIDTH)
    }
    pub fn water_speed(&self) -> f32 {
        if self.water.is_empty() {
            return 0.0;
        }
        self.water.iter().map(|p| p.velocity.x).sum::<f32>() / self.water.len() as f32
    }
    pub fn floor_cells(&self) -> usize {
        let slope = self
            .placement
            .map(|p| p.slope_deg)
            .unwrap_or(6.0)
            .to_radians()
            .tan();
        (LENGTH * slope / FLUID_CELL).ceil() as usize + 4
    }
    /// FLIP uses X downstream, Y vertical, Z across the channel. Placement uses
    /// a sloping reference plane, so convert once at the presentation boundary.
    pub fn grid_to_local(&self, p: Vec3) -> Vec3 {
        let slope = self
            .placement
            .map(|p| p.slope_deg)
            .unwrap_or(6.0)
            .to_radians()
            .tan();
        Vec3::new(
            p.x,
            p.y - (self.floor_cells() + 1) as f32 * FLUID_CELL + p.x * slope,
            p.z - CHANNEL_WIDTH * 0.5 - FLUID_CELL,
        )
    }
    pub fn feed(&mut self, dt: f32, water_available: bool) {
        if self.placement.is_none() {
            return;
        }
        self.time += dt;
        if self.feed_on && self.pump_on && water_available && self.grains.len() + 3 < MAX_GRAINS {
            self.feed_credit =
                (self.feed_credit + self.feed_m3s as f64 * dt as f64).min(PARCEL_VOLUME * 2.0);
            while self.feed_credit >= PARCEL_VOLUME
                && !self.hopper.is_empty()
                && self.grains.len() + 3 < MAX_GRAINS
            {
                let m = self.hopper.take_volume(PARCEL_VOLUME);
                self.processed_m3 += m.volume();
                self.feed_credit -= PARCEL_VOLUME;
                self.emit(m);
            }
        }
    }
    fn emit(&mut self, m: Material) {
        self.sequence += 1;
        let z = ((self.sequence.wrapping_mul(1664525) % 1000) as f32 / 1000.0 - 0.5)
            * CHANNEL_WIDTH
            * 0.70;
        let sand = Material {
            soil_m3: m.soil_m3,
            sand_m3: m.sand_m3,
            ..Material::default()
        };
        let gravel = Material {
            gravel_m3: m.gravel_m3,
            ..Material::default()
        };
        let gold = Material {
            gold_g: m.gold_g,
            ..Material::default()
        };
        for (kind, material) in [
            (GrainKind::Sand, sand),
            (GrainKind::Gravel, gravel),
            (GrainKind::Gold, gold),
        ] {
            let size_sample = ((self.sequence.wrapping_mul(1103515245).wrapping_add(12345) >> 8)
                % 1000) as f32
                / 999.0;
            let diameter_m = match kind {
                GrainKind::Gold => 0.00015 + size_sample * 0.00105,
                GrainKind::Sand => 0.0002 + size_sample * 0.0010,
                GrainKind::Gravel => 0.001 + size_sample * 0.003,
            };
            if !material.is_empty() {
                self.grains.push(Grain {
                    kind,
                    position: Vec3::new(
                        0.12,
                        (self.floor_cells() + 1) as f32 * FLUID_CELL + 0.06,
                        z + CHANNEL_WIDTH * 0.5 + FLUID_CELL,
                    ),
                    velocity: Vec3::new(0.4, 0.0, 0.0),
                    material,
                    resting: false,
                    diameter_m,
                    affine: Mat3::ZERO,
                    rotation: Quat::IDENTITY,
                    angular_velocity: Vec3::ZERO,
                });
            }
        }
    }
    pub fn cleanup(&mut self) -> Result<Material, String> {
        if self.pump_on || self.feed_on {
            return Err("Stop the feed and pump before lifting the mats".into());
        }
        if self.grains.is_empty() {
            return Err("The mats are empty".into());
        }
        let mut m = Material::default();
        for g in self.grains.drain(..) {
            m.add(g.material);
        }
        self.contacts = Default::default();
        self.concentrate.add(m);
        self.cleanups += 1;
        Ok(m)
    }
    pub fn relocate(&mut self, p: Placement, site: &Site, machine: Vec3) -> Result<(), String> {
        if self.pump_on || !self.grains.is_empty() || !self.water.is_empty() {
            return Err("Stop the pump and clean the mats before moving the sluice".into());
        }
        p.validate(site, machine)?;
        self.placement = Some(p);
        Ok(())
    }
    pub fn valid(&self) -> bool {
        self.hopper.valid()
            && self.tailings.valid()
            && self.concentrate.valid()
            && self.hopper.volume() <= 0.85001
            && self.grains.len() <= MAX_GRAINS
            && self.water.len() <= MAX_WATER_PARTICLES
            && self.contacts.spheres.len() < MAX_GRAINS * 20
            && self.contacts.planes.len() < MAX_GRAINS * 20
            && self.contacts.sdf.len() < MAX_GRAINS * 20
            && self.contacts.spheres.iter().all(|(a, b, _, _, v)| {
                *a < self.grains.len() && *b < self.grains.len() && v.is_finite()
            })
            && self
                .contacts
                .planes
                .iter()
                .all(|(a, _, _, _, v)| *a < self.grains.len() && v.is_finite())
            && self
                .contacts
                .sdf
                .iter()
                .all(|(a, _, v)| *a < self.grains.len() && v.is_finite())
            && self.water.iter().all(|p| {
                p.position.is_finite()
                    && p.position.length() < 20.0
                    && p.velocity.is_finite()
                    && p.velocity.length() < 100.0
                    && p.affine.is_finite()
            })
            && self.water_credit_m3.is_finite()
            && self.water_credit_m3 >= 0.0
            && self.water_credit_m3 < 1.0
            && self.water_returned_m3.is_finite()
            && self.water_returned_m3 >= 0.0
            && self.grains.iter().all(|g| {
                g.affine.is_finite()
                    && g.rotation.is_finite()
                    && g.angular_velocity.is_finite()
                    && g.position.is_finite()
                    && g.velocity.is_finite()
                    && g.material.valid()
                    && g.position.length() < 100.0
                    && g.velocity.length() < 100.0
                    && (0.00005..=0.02).contains(&g.diameter_m)
            })
            && [self.flow_m3s, self.feed_m3s, self.time]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
            && (0.001..=0.120).contains(&self.flow_m3s)
            && (0.001..=0.02).contains(&self.feed_m3s)
            && self.processed_m3.is_finite()
            && self.processed_m3 >= 0.0
            && self.water_used_m3.is_finite()
            && self.water_used_m3 >= 0.0
            && self.feed_credit.is_finite()
            && (0.0..=PARCEL_VOLUME * 2.01).contains(&self.feed_credit)
    }
}

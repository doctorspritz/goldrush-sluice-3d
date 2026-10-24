use crate::{Material, Site, Vec3};
use glam::{Mat3, Quat};
use serde::{Deserialize, Serialize};

pub const LENGTH: f32 = 1.5;
pub const CHANNEL_WIDTH: f32 = 0.38;
pub const HOPPER_X: f32 = 14.6;
pub const HOPPER_Z: f32 = 17.0;
pub const PARCEL_VOLUME: f64 = 0.000002; // 2 mL weighted parcels, not 2-litre lumps
/// Initial derived solver allocation, not a physical admission limit.
pub const INITIAL_GRAIN_STORAGE: usize = 4096;
/// Historical API alias; model admission no longer uses this value.
pub const MAX_GRAINS: usize = INITIAL_GRAIN_STORAGE;
pub const FLUID_CELL: f32 = 0.01;
pub const WATER_PARTICLE_M3: f64 = (FLUID_CELL * FLUID_CELL * FLUID_CELL / 8.0) as f64;
pub const MAX_WATER_PARTICLES: usize = 300_000;

/// Saves without a profile are the first playable build: retain their grid,
/// particle volumes and geometry instead of reinterpreting live water at 1 cm.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum HydraulicProfile {
    #[default]
    Legacy,
    OriginalDemo,
}
impl HydraulicProfile {
    pub fn length(self) -> f32 {
        match self {
            Self::Legacy => 3.4,
            Self::OriginalDemo => LENGTH,
        }
    }
    pub fn width(self) -> f32 {
        match self {
            Self::Legacy => 0.72,
            Self::OriginalDemo => CHANNEL_WIDTH,
        }
    }
    pub fn cell(self) -> f32 {
        match self {
            Self::Legacy => 0.02,
            Self::OriginalDemo => FLUID_CELL,
        }
    }
    pub fn capacity(self) -> usize {
        match self {
            Self::Legacy => 400_000,
            Self::OriginalDemo => MAX_WATER_PARTICLES,
        }
    }
    pub fn particle_m3(self) -> f64 {
        let dx = self.cell();
        (dx * dx * dx / 8.0) as f64
    }
    pub fn outlet_buffer(self) -> usize {
        match self {
            Self::Legacy => 0,
            Self::OriginalDemo => 12,
        }
    }
    pub fn substeps(self) -> u32 {
        match self {
            Self::Legacy => 2,
            Self::OriginalDemo => 4,
        }
    }
}

/// Original FLIP/APIC state, in the local simulation grid. Kept in the save,
/// rather than recreated when a camera changes or a shift is restored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaterParticle {
    /// Presentation identity only; never used by the physical solver.
    #[serde(default)]
    pub id: u64,
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
            slope_deg: (26.0_f32 / 150.0).atan().to_degrees(),
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
        self.validate_profile(site, machine, HydraulicProfile::OriginalDemo)
    }
    fn validate_profile(
        self,
        site: &Site,
        machine: Vec3,
        profile: HydraulicProfile,
    ) -> Result<(), String> {
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
            let x = profile.length() * i as f32 / 12.0;
            for z in [-profile.width() * 0.5, profile.width() * 0.5] {
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
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContactModel {
    #[default]
    LegacyDamping,
    Coulomb,
}
/// Explicit prototype grading, not a measured claim PSD. Old shifts keep
/// their original soil/sand grouping and diameters when more feed is emitted.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum FeedGrading {
    #[default]
    Legacy,
    Alluvial,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Grain {
    #[serde(default)]
    pub volume_resolved: bool,
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

impl Grain {
    pub fn represented_grains(&self) -> f64 {
        if self.volume_resolved {
            self.solid_volume() / (std::f64::consts::PI / 6. * (self.diameter_m as f64).powi(3))
        } else {
            1.
        }
    }
    pub fn solid_volume(&self) -> f64 {
        self.material.solid_volume_m3()
    }
    pub fn collision_radius(&self) -> f32 {
        let physical = self.diameter_m * 0.5;
        if !self.volume_resolved {
            return physical;
        }
        let one_grain = std::f64::consts::PI / 6.0 * (self.diameter_m as f64).powi(3);
        if self.solid_volume() <= one_grain * (1.0 + 1e-5) {
            return physical;
        }
        // Coarse DEM parcel: packed volume controls excluded space; the
        // physical grain diameter separately controls settling and rendering.
        physical
            .max((self.solid_volume() * 3.0 / (4.0 * std::f64::consts::PI * 0.60)).cbrt() as f32)
    }
    pub fn mass_kg(&self) -> f32 {
        self.material.mass_kg() as f32
    }
    pub fn density_kg_m3(&self) -> f32 {
        (self.material.mass_kg() / self.solid_volume().max(1e-30)) as f32
    }
}
fn riffles_present() -> bool {
    true
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sluice {
    #[serde(default)]
    pub profile: HydraulicProfile,
    #[serde(default)]
    pub granular_packets: bool,
    #[serde(default)]
    pub contact_model: ContactModel,
    #[serde(default)]
    pub feed_grading: FeedGrading,
    /// Research resolution control. Physical diameters remain independent.
    #[serde(default)]
    pub parcel_volume_m3: Option<f64>,
    #[serde(default = "riffles_present")]
    pub riffles_enabled: bool,
    #[serde(default)]
    pub fluid_clock: f32,
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
    #[serde(default)]
    next_water_id: u64,
    /// Water outside the FLIP channel, in world coordinates; owns real volume.
    #[serde(default)]
    pub outflow: Vec<crate::outflow::OutflowParticle>,
    pub water_credit_m3: f64,
    /// Moisture released at the paydirt chute, distinct from the pump carrier.
    #[serde(default)]
    pub feed_water_credit_m3: f64,
    #[serde(default)]
    pub inlet: crate::intake::Inlet,
    pub water_returned_m3: f64,
    pub solver_steps: u64,
    pub contacts: sim3d::clump::ContactHistory3D,
    feed_credit: f64,
    #[serde(default)]
    pending_gold_g: f64,
    sequence: u64,
}
impl Default for Sluice {
    fn default() -> Self {
        Self {
            profile: HydraulicProfile::OriginalDemo,
            granular_packets: true,
            contact_model: ContactModel::Coulomb,
            feed_grading: FeedGrading::Alluvial,
            parcel_volume_m3: None,
            riffles_enabled: true,
            fluid_clock: 0.0,
            placement: None,
            hopper: Material::default(),
            grains: Vec::new(),
            tailings: Material::default(),
            concentrate: Material::default(),
            pump_on: false,
            feed_on: false,
            flow_m3s: 0.00375,
            feed_m3s: 0.0001,
            processed_m3: 0.0,
            water_used_m3: 0.0,
            cleanups: 0,
            time: 0.0,
            water: Vec::new(),
            next_water_id: 0,
            outflow: Vec::new(),
            water_credit_m3: 0.0,
            feed_water_credit_m3: 0.0,
            inlet: Default::default(),
            water_returned_m3: 0.0,
            solver_steps: 0,
            contacts: Default::default(),
            feed_credit: 0.0,
            pending_gold_g: 0.0,
            sequence: 0,
        }
    }
}
impl Sluice {
    /// One-time migration for saves predating persistent water identities.
    pub fn ensure_water_ids(&mut self) {
        if self.next_water_id != 0 {
            return;
        }
        let mut next = self
            .water
            .iter()
            .map(|p| p.id)
            .chain(self.outflow.iter().map(|p| p.id))
            .max()
            .unwrap_or(0)
            + 1;
        for id in self
            .water
            .iter_mut()
            .map(|p| &mut p.id)
            .chain(self.outflow.iter_mut().map(|p| &mut p.id))
        {
            if *id == 0 {
                *id = next;
                next += 1;
            }
        }
        self.next_water_id = next;
    }
    pub fn allocate_water_id(&mut self) -> u64 {
        self.ensure_water_ids();
        let id = self.next_water_id;
        self.next_water_id += 1;
        id
    }
    pub fn parcel_volume(&self) -> f64 {
        if self.granular_packets {
            self.parcel_volume_m3.unwrap_or(PARCEL_VOLUME)
        } else {
            0.002
        }
    }
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
        m.add(self.inlet.material());
        m.gold_g += self.pending_gold_g;
        for g in &self.grains {
            m.add(g.material);
        }
        m
    }
    pub fn accept_intake(&mut self, packet: crate::intake::IntakePacket) {
        self.water_credit_m3 += packet.water_m3;
        self.inlet.ready.add(packet.suspension);
        self.inlet.withdrawn.add(packet.suspension);
    }
    pub fn intake_has_room(&self) -> bool {
        self.inlet.material().volume()
            < self
                .parcel_volume()
                .min(PARCEL_VOLUME)
                .max(std::f64::consts::PI / 6. * 0.01_f64.powi(3))
                * crate::intake::ADMISSION_BATCH as f64
    }
    /// Called with precisely the water volume admitted to FLIP. Suspended
    /// material leaves the same carrier fraction and remains in inlet custody
    /// until actual DEM parcels can be created; no hopper/feed gate is involved.
    pub fn admit_intake_water(&mut self, water_m3: f64) {
        let fraction = water_m3 / self.water_credit_m3.max(1e-30);
        self.inlet
            .pending
            .add(self.inlet.ready.take_fraction(fraction));
        self.water_credit_m3 = (self.water_credit_m3 - water_m3).max(0.);
    }
    pub fn emit_intake_grains(&mut self, dt: f32) {
        if self.placement.is_none() {
            return;
        }
        self.inlet.emission_clock = (self.inlet.emission_clock + dt).min(0.25);
        let parcel_volume = self.parcel_volume().min(PARCEL_VOLUME);
        if self.inlet.pending.material().volume() < parcel_volume
            && self.inlet.emission_clock < 0.2499
            && self.pump_on
        {
            return;
        }
        // Match the terrain solver's four D50 values, rather than resampling
        // pumped fines as the dry hopper's much coarser alluvial grading.
        let mut budget = crate::intake::ADMISSION_BATCH;
        while budget > 0 {
            let mut progressed = false;
            for (channel, diameter, kind) in [
                (0, 0.0001, GrainKind::Sand),
                (1, 0.001, GrainKind::Sand),
                (2, 0.01, GrainKind::Gravel),
                (3, 0.002, GrainKind::Gravel),
                (4, 0.0001, GrainKind::Gold),
            ] {
                if budget == 0 {
                    break;
                }
                let one_grain = std::f64::consts::PI / 6. * (diameter as f64).powi(3);
                let one_reference = if channel == 4 {
                    one_grain
                } else {
                    one_grain * crate::material::MINERAL_DENSITY_KG_M3
                        / crate::material::REFERENCE_DRY_DENSITIES_KG_M3[channel]
                };
                let available = if channel == 4 {
                    self.inlet.pending.gold_g / (19_300. * 1000.)
                } else {
                    self.inlet.pending.solid_m3[channel]
                };
                if available < one_reference {
                    continue;
                }
                let mut load = crate::intake::Suspension::default();
                if channel == 4 {
                    load.gold_g = self
                        .inlet
                        .pending
                        .gold_g
                        .min(parcel_volume.max(one_grain) * 19_300. * 1000.);
                    self.inlet.pending.gold_g -= load.gold_g;
                } else {
                    load.solid_m3[channel] =
                        self.inlet.pending.solid_m3[channel].min(parcel_volume.max(one_reference));
                    self.inlet.pending.solid_m3[channel] -= load.solid_m3[channel];
                }
                self.inlet.sequence = self.inlet.sequence.wrapping_add(1);
                let sample =
                    |factor: f64| ((self.inlet.sequence as f64 + 0.5) * factor).fract() as f32;
                let mut grain = Grain {
                    volume_resolved: true,
                    kind,
                    position: Vec3::new(
                        self.profile.cell() * (2. + sample(0.754877666)),
                        (self.floor_cells() + 1) as f32 * self.profile.cell() + 0.04,
                        self.profile.width() * 0.5
                            + self.profile.cell()
                            + (sample(0.438579021) - 0.5) * self.profile.width() * 0.25,
                    ),
                    velocity: self.inlet_velocity(),
                    material: load.material(),
                    resting: false,
                    diameter_m: diameter,
                    affine: Mat3::ZERO,
                    rotation: Quat::IDENTITY,
                    angular_velocity: Vec3::ZERO,
                };
                // A weighted parcel has a larger contact envelope than a water
                // sample. Release its complete envelope clear of the inlet wall.
                grain.position.x += grain.collision_radius();
                self.grains.push(grain);
                self.inlet.emitted.add(load);
                budget -= 1;
                progressed = true;
            }
            if !progressed || budget == 0 {
                break;
            }
        }
        self.inlet.emission_clock = 0.;
    }
    /// Retain the existing specified inlet velocity. Pump pressure, nozzle
    /// geometry and the external reaction force are separate calibration work.
    pub fn inlet_velocity(&self) -> Vec3 {
        if self.profile == HydraulicProfile::OriginalDemo {
            Vec3::new(0.5, -0.05, 0.)
        } else {
            Vec3::new(0.4, 0., 0.)
        }
    }
    /// The same finite chute footprint used by granular paydirt, in grid
    /// coordinates. Released moisture starts here, not at the pump nozzle.
    pub fn feed_water_position(&self, sequence: u64) -> Vec3 {
        let sample = |factor: f64| ((sequence as f64 + 0.5) * factor).fract() as f32;
        Vec3::new(
            0.10 + 0.04 * sample(0.754877666),
            (self.floor_cells() + 1) as f32 * self.profile.cell()
                + 0.05
                + 0.02 * sample(0.569840291),
            (sample(0.618033989) - 0.5) * self.profile.width() * 0.70
                + self.profile.width() * 0.5
                + self.profile.cell(),
        )
    }
    /// Volume and positions come from FLIP; there is no analytic water-depth
    /// or sediment-capture model in the game layer.
    pub fn water_depth(&self) -> f32 {
        self.water.len() as f32 * self.profile.particle_m3() as f32
            / (self.profile.length() * self.profile.width())
    }
    pub fn outlet(&self) -> Option<Vec3> {
        self.placement
            .map(|p| p.to_world(Vec3::new(self.profile.length(), 0.0, 0.0)))
    }
    pub fn accounted_water_m3(&self) -> f64 {
        self.stored_water_m3() + self.water_returned_m3
    }
    pub fn stored_water_m3(&self) -> f64 {
        (self.water.len() + self.outflow.len()) as f64 * self.profile.particle_m3()
            + self.water_credit_m3
            + self.feed_water_credit_m3
    }
    pub fn grid_velocity_to_world(&self, v: Vec3) -> Vec3 {
        let p = self.placement.expect("placed water domain");
        p.forward() * v.x + Vec3::Y * v.y + p.right() * v.z
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
            .unwrap_or((26.0_f32 / 150.0).atan().to_degrees())
            .to_radians()
            .tan();
        let drop = self.profile.length() * slope / self.profile.cell();
        let cells = match self.profile {
            HydraulicProfile::Legacy => drop.ceil(),
            HydraulicProfile::OriginalDemo => drop.round(),
        };
        cells as usize + 4
    }
    /// FLIP uses X downstream, Y vertical, Z across the channel. Placement uses
    /// a sloping reference plane, so convert once at the presentation boundary.
    pub fn grid_to_local(&self, p: Vec3) -> Vec3 {
        let slope = self
            .placement
            .map(|p| p.slope_deg)
            .unwrap_or((26.0_f32 / 150.0).atan().to_degrees())
            .to_radians()
            .tan();
        Vec3::new(
            p.x,
            p.y - (self.floor_cells() + 1) as f32 * self.profile.cell() + p.x * slope,
            p.z - self.profile.width() * 0.5 - self.profile.cell(),
        )
    }
    pub fn feed(&mut self, dt: f32, water_available: bool) {
        if self.placement.is_none() {
            return;
        }
        self.time += dt;
        if !self.feed_on && self.pending_gold_g > 0.0 {
            self.hopper.gold_g += self.pending_gold_g;
            self.pending_gold_g = 0.0;
        }
        if self.feed_on && self.pump_on && water_available {
            self.feed_credit = (self.feed_credit + self.feed_m3s as f64 * dt as f64)
                .min(self.parcel_volume() * if self.granular_packets { 32.0 } else { 2.0 });
            while self.feed_credit >= self.parcel_volume() && !self.hopper.is_empty() {
                let moisture = self.hopper.pore_water_m3
                    * (self.parcel_volume() / self.hopper.volume().max(1e-30)).min(1.);
                let room = self
                    .profile
                    .capacity()
                    .saturating_sub(self.water.len() + self.outflow.len())
                    as f64
                    * self.profile.particle_m3()
                    - self.water_credit_m3
                    - self.feed_water_credit_m3;
                if moisture > room.max(0.) {
                    break;
                }
                let mut m = self.hopper.take_volume(self.parcel_volume());
                // Keep moisture at its own feed chute and source velocity.
                // Never dilute the suspended river-intake carrier with it.
                self.feed_water_credit_m3 += m.pore_water_m3;
                self.water_used_m3 += m.pore_water_m3;
                m.pore_water_m3 = 0.;
                self.processed_m3 += m.volume();
                self.feed_credit -= self.parcel_volume();
                self.emit(m);
            }
        }
    }
    fn emit(&mut self, m: Material) {
        self.sequence += 1;
        let z = ((self.sequence.wrapping_mul(1664525) % 1000) as f32 / 1000.0 - 0.5)
            * self.profile.width()
            * 0.70;
        let sand = Material {
            soil_m3: if self.feed_grading == FeedGrading::Legacy {
                m.soil_m3
            } else {
                0.
            },
            sand_m3: m.sand_m3,
            ..Material::default()
        };
        let fines = Material {
            soil_m3: if self.feed_grading == FeedGrading::Alluvial {
                m.soil_m3
            } else {
                0.
            },
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
        for (kind, mut material, phase) in [
            (GrainKind::Sand, fines, 0.0),
            (GrainKind::Sand, sand, 0.19),
            (GrainKind::Gravel, gravel, 0.37),
            (GrainKind::Gold, gold, 0.73),
        ] {
            let size_sample = ((self.sequence.wrapping_mul(1103515245).wrapping_add(12345) >> 8)
                % 1000) as f32
                / 999.0;
            let diameter_m = match kind {
                // Equal-volume parcels sample a log-spaced fine fraction.
                // This changes the physical drag diameter, not the parcel mass.
                GrainKind::Sand
                    if self.feed_grading == FeedGrading::Alluvial && material.soil_m3 > 0. =>
                {
                    0.00002 * 10.0_f32.powf(size_sample)
                }
                GrainKind::Gold => 0.00008 + size_sample * 0.00042,
                GrainKind::Sand => 0.0002 + size_sample * 0.0010,
                GrainKind::Gravel => 0.001 + size_sample * 0.005,
            };
            if kind == GrainKind::Gold && self.granular_packets {
                self.pending_gold_g += material.gold_g;
                let physical_mass =
                    std::f64::consts::PI / 6.0 * (diameter_m as f64).powi(3) * 19_300.0 * 1000.0;
                if self.pending_gold_g < physical_mass {
                    continue;
                }
                material.gold_g = physical_mass;
                self.pending_gold_g -= physical_mass;
            }
            if !material.is_empty() {
                let position = if self.granular_packets {
                    // A continuous chute footprint, without coincident sand,
                    // gravel and gold clumps or the old 40-lane repeating lattice.
                    let phase =
                        if self.feed_grading == FeedGrading::Legacy && kind == GrainKind::Sand {
                            0.0
                        } else {
                            phase
                        };
                    let sample = |factor: f64| {
                        ((self.sequence as f64 + 0.5) * factor + phase).fract() as f32
                    };
                    Vec3::new(
                        0.10 + 0.04 * sample(0.754877666),
                        (self.floor_cells() + 1) as f32 * self.profile.cell()
                            + 0.05
                            + 0.02 * sample(0.569840291),
                        (sample(0.618033989) - 0.5) * self.profile.width() * 0.70
                            + self.profile.width() * 0.5
                            + self.profile.cell(),
                    )
                } else {
                    Vec3::new(
                        0.12,
                        (self.floor_cells() + 1) as f32 * self.profile.cell() + 0.06,
                        z + self.profile.width() * 0.5 + self.profile.cell(),
                    )
                };
                self.grains.push(Grain {
                    volume_resolved: self.granular_packets,
                    kind,
                    position,
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
        let before = self.grains.len();
        self.grains.retain(|g| {
            let over_mats = g.position.x >= 0.40 && g.position.x < self.profile.length() - 0.12;
            // Collect the settled material supported by the mats. Moving
            // upstream/suspended grains remain in the physical domain.
            if !self.granular_packets || (over_mats && g.resting) {
                m.add(g.material);
                false
            } else {
                true
            }
        });
        if self.grains.len() == before {
            return Err("Let material settle on the mats before collecting".into());
        }
        self.contacts = Default::default();
        self.concentrate.add(m);
        self.cleanups += 1;
        Ok(m)
    }
    pub fn validate_relocation(
        &self,
        p: Placement,
        site: &Site,
        machine: Vec3,
    ) -> Result<(), String> {
        if self.pump_on
            || !self.grains.is_empty()
            || !self.water.is_empty()
            || !self.outflow.is_empty()
        {
            return Err(
                "Stop the pump, let water drain and clean the mats before moving the sluice".into(),
            );
        }
        p.validate_profile(site, machine, self.profile)
    }
    pub fn relocate(&mut self, p: Placement, site: &Site, machine: Vec3) -> Result<(), String> {
        self.validate_relocation(p, site, machine)?;
        self.placement = Some(p);
        Ok(())
    }
    pub fn valid(&self) -> bool {
        // Preserve the old contact-history allowance for old/small saves, then
        // scale it with the actual population. File input has its own byte cap.
        let contact_limit = self
            .grains
            .len()
            .max(INITIAL_GRAIN_STORAGE)
            .saturating_mul(20);
        self.hopper.valid()
            && self.inlet.valid()
            && self.tailings.valid()
            && self.concentrate.valid()
            && self.hopper.volume() <= 0.85001
            && self.water.len() + self.outflow.len() <= self.profile.capacity()
            && self.outflow.iter().all(|p| {
                p.position.is_finite()
                    && p.position.length() < 1000.
                    && p.velocity.is_finite()
                    && p.velocity.length() < 100.
            })
            && self.contacts.spheres.len() < contact_limit
            && self.contacts.planes.len() < contact_limit
            && self.contacts.sdf.len() < contact_limit
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
            && self.feed_water_credit_m3.is_finite()
            && (0.0..1.0).contains(&self.feed_water_credit_m3)
            && self.water_returned_m3.is_finite()
            && self.water_returned_m3 >= 0.0
            && self.grains.iter().all(|g| {
                g.affine.is_finite()
                    && g.rotation.is_finite()
                    && g.angular_velocity.is_finite()
                    && g.position.is_finite()
                    && g.velocity.is_finite()
                    && g.material.valid()
                    && g.material.pore_water_m3 == 0.
                    && g.solid_volume() > 0.
                    && g.position.length() < 100.0
                    && g.velocity.length() < 100.0
                    && (0.00002..=0.02).contains(&g.diameter_m)
            })
            && [self.flow_m3s, self.feed_m3s, self.time]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
            && (0.001..=0.120).contains(&self.flow_m3s)
            && (0.00001..=0.02).contains(&self.feed_m3s)
            && self
                .parcel_volume_m3
                .is_none_or(|v| v.is_finite() && (1e-9..=PARCEL_VOLUME).contains(&v))
            && self.processed_m3.is_finite()
            && self.processed_m3 >= 0.0
            && self.water_used_m3.is_finite()
            && self.water_used_m3 >= 0.0
            && self.fluid_clock.is_finite()
            && (0.0..=0.034).contains(&self.fluid_clock)
            && self.pending_gold_g.is_finite()
            && self.pending_gold_g >= 0.0
            && self.feed_credit.is_finite()
            && (0.0..=self.parcel_volume() * if self.granular_packets { 32.01 } else { 2.01 })
                .contains(&self.feed_credit)
    }
}

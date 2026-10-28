//! Finite moving screen with detailed wires by default. Its weighted parcels
//! follow physical grain diameter and wet mass in a deliberately dilute domain.
//! Optional bulk-deck loading/adhesion is an independently saved statistical
//! approximation; neither mode has a screen-water or explicit packing solver.
//! Wires, gutter and catch bin have the same geometry exposed to presentation.
#[path = "screen_bed.rs"]
mod bed;
pub use bed::{BulkBedCell, BulkBedField, BulkBedSample, BulkProjection};
#[path = "bulk_screen.rs"]
mod bulk;
use crate::{sluice::Grain, Material, Vec3};
pub use bulk::BulkDeck;
use bulk::BulkState;
use glam::DVec3;
use serde::{Deserialize, Serialize};

pub const MAX_SCREEN_PARTICLES: usize = 4096;
const G: f64 = 9.81;
const MAX_QUERIES: usize = 4_000_000;
const MAX_INTERVALS: usize = 1_000_000;
const MAX_EVENTS: usize = 24;
const MAX_MANIFOLD: usize = 96;
const MANIFOLD_SWEEPS: usize = 64;
const CONTACT_SKIN_M: f64 = 1e-10;

/// Deliberate first-screen approximation: coupled impacts are plastic.
/// Elastic multi-contact rebound/compliance still requires calibration.
/// The inelastic impulse/contact distinction is discussed in
/// https://arxiv.org/abs/1706.00221; this does not claim multi-contact uniqueness.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SimultaneousImpactPolicy {
    #[default]
    Plastic,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScreenConfig {
    pub length_m: f64,
    pub width_m: f64,
    pub aperture_m: f64,
    pub wire_radius_m: f64,
    pub slope_deg: f64,
    pub amplitude_downstream_m: f64,
    pub amplitude_normal_m: f64,
    pub frequency_hz: f64,
    pub friction_mu: f64,
    /// Normal restitution for isolated impacts; the simultaneous policy is
    /// saved explicitly and never selected by solver success or failure.
    pub restitution: f64,
    pub simultaneous_impact: SimultaneousImpactPolicy,
}
impl Default for ScreenConfig {
    fn default() -> Self {
        Self {
            length_m: 0.8,
            width_m: 0.38,
            aperture_m: 0.003,
            wire_radius_m: 0.001,
            slope_deg: 12.,
            amplitude_downstream_m: 0.002,
            amplitude_normal_m: 0.002,
            frequency_hz: 12.,
            friction_mu: 0.4,
            restitution: 0.1,
            simultaneous_impact: SimultaneousImpactPolicy::Plastic,
        }
    }
}
impl ScreenConfig {
    fn valid(&self) -> bool {
        (0.1..=2.).contains(&self.length_m)
            && (0.05..=1.).contains(&self.width_m)
            && (0.0005..=0.05).contains(&self.aperture_m)
            && (0.0001..=0.01).contains(&self.wire_radius_m)
            && (0.0..=25.).contains(&self.slope_deg)
            && (0.0..=0.01).contains(&self.amplitude_downstream_m)
            && (0.0..=0.01).contains(&self.amplitude_normal_m)
            && (0.0..=30.).contains(&self.frequency_hz)
            && (0.0..=1.).contains(&self.friction_mu)
            && (0.0..=1.).contains(&self.restitution)
            && self.length_m / self.pitch() < 1500.
            && self.width_m / self.pitch() < 1500.
    }
    fn pitch(&self) -> f64 {
        self.aperture_m + 2. * self.wire_radius_m
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScreenParticle {
    pub id: u64,
    pub grain: Grain,
    /// Double precision world state is authoritative while owned by the screen.
    pub position: DVec3,
    pub velocity: DVec3,
    /// Authoritative spin. Grain's f32 mirror is used at rendering/handoff;
    /// that boundary conversion is not an extra physical energy allowance.
    pub angular_velocity: DVec3,
    pub in_gutter: bool,
    passed_deck: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScreenStep {
    pub intervals: usize,
    pub contact_queries: usize,
    pub contacts: usize,
    /// Detailed-wire mode only: work supplied by wires, sum J dot v_wire.
    /// The statistical bulk law does not resolve this actuator-work audit.
    pub motor_work_j: f64,
    /// Explicit numerical gravitational work of the bounded clearance repair.
    /// This is not physical actuator work or an additional energy allowance.
    pub projection_work_j: f64,
    pub maximum_repair_m: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Screen {
    pub config: ScreenConfig,
    pub origin: Vec3,
    pub yaw: f32,
    pub motor_on: bool,
    /// Drive clock freezes when the motor is off; the deck does not snap back.
    pub time: f64,
    pub elapsed_time_s: f64,
    pub particles: Vec<ScreenParticle>,
    pub undersize: Vec<Grain>,
    pub oversize: Vec<Grain>,
    /// Escapes retain their actual airborne state for a world-flight owner.
    pub spills: Vec<Grain>,
    pub free_water_m3: f64,
    pub admitted: Material,
    pub delivered: Material,
    pub released: Material,
    pub free_water_admitted_m3: f64,
    pub free_water_delivered_m3: f64,
    pub last_step: ScreenStep,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bulk: Option<BulkState>,
    installed_config: ScreenConfig,
    next_id: u64,
    /// Transaction-local immutable geometry; never persisted or retained by
    /// a completed step. Configuration changes cannot leave a stale cache.
    #[serde(skip)]
    frame: Option<ScreenFrame>,
}
#[derive(Clone, Copy, Debug)]
struct ScreenFrame {
    downstream: DVec3,
    across: DVec3,
    normal: DVec3,
    tangent_amplitude: DVec3,
    normal_amplitude: DVec3,
    omega: f64,
    slope_tan: f64,
    slope_cos: f64,
}
impl ScreenFrame {
    fn new(screen: &Screen) -> Self {
        let downstream = screen.downstream();
        let across = screen.across();
        let normal = screen.normal();
        Self {
            downstream,
            across,
            normal,
            tangent_amplitude: downstream * screen.config.amplitude_downstream_m,
            normal_amplitude: normal * screen.config.amplitude_normal_m,
            omega: std::f64::consts::TAU * screen.config.frequency_hz,
            slope_tan: screen.config.slope_deg.to_radians().tan(),
            slope_cos: screen.config.slope_deg.to_radians().cos(),
        }
    }
}

#[derive(Clone, Copy)]
struct Wire {
    a: DVec3,
    b: DVec3,
    local_a: DVec3,
    local_b: DVec3,
    radius: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Surface {
    Wire,
    Gutter,
    Bin,
}
#[derive(Clone, Copy)]
struct Hit {
    u: f64,
    normal: DVec3,
    velocity: DVec3,
    surface: Surface,
    index: usize,
}

struct Contact {
    hit: Hit,
    target: f64,
}
#[derive(Clone)]
struct ContactEvaluation {
    velocity: DVec3,
    scaled_spin: DVec3,
    projection: Projection,
    explicit_normal: Option<Vec<f64>>,
    residual: Vec<f64>,
    normal_residual: f64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContactMode {
    Open,
    Stick,
    Slide,
}
struct ContactProblem<'a> {
    rows: &'a [Contact],
    bases: &'a [(DVec3, DVec3)],
    velocity: DVec3,
    spin: DVec3,
}
impl ContactEvaluation {
    fn normal_impulse(&self, index: usize) -> f64 {
        self.explicit_normal
            .as_ref()
            .map_or_else(|| self.projection.multiplier(index), |values| values[index])
    }
}
impl ContactProblem<'_> {
    fn evaluate(&self, x: &[f64], mu: f64) -> Option<ContactEvaluation> {
        evaluate_contacts(self.rows, self.bases, x, self.velocity, self.spin, mu)
    }
}
#[derive(Clone)]
struct ContactTrial {
    impulses: Vec<f64>,
    normal_impulses: Option<Vec<f64>>,
    converged: bool,
    iterations: usize,
    mode: Option<[ContactMode; 2]>,
    reason: &'static str,
    direction: &'static str,
    backtrack: usize,
}
struct ContactSolution {
    trial: ContactTrial,
    trial_mu: f64,
    iterations: usize,
    seed_attempts: usize,
}
#[derive(Clone, Copy, Default)]
struct ContactContext {
    time: f64,
    interval: f64,
    guard: f64,
    synthetic: bool,
}
/// Exact evaluated endpoints of one dyadic refinement interval. Sharing a
/// midpoint or an admitted endpoint avoids repeated trigonometry without
/// interpolating phase or changing any contact arithmetic or tolerance.
#[derive(Clone, Copy)]
struct WireBracket {
    lo: f64,
    hi: f64,
    lower: (DVec3, DVec3),
    upper: (DVec3, DVec3),
}
impl WireBracket {
    fn at(self, screen: &Screen, sweep: &WireSweep, u: f64) -> (DVec3, DVec3) {
        if u.to_bits() == self.lo.to_bits() {
            self.lower
        } else if u.to_bits() == self.hi.to_bits() {
            self.upper
        } else {
            screen.offset_at(sweep.time0 + (sweep.time1 - sweep.time0) * u)
        }
    }
}
struct WireSweep {
    start: DVec3,
    delta: DVec3,
    velocity: DVec3,
    time0: f64,
    time1: f64,
    wire: Wire,
    radius: f64,
    index: usize,
    acceleration: f64,
}
fn same_feature(a: Hit, b: Hit) -> bool {
    a.surface == b.surface && a.index == b.index
}

impl Screen {
    pub fn new(origin: Vec3, yaw: f32) -> Self {
        Self {
            bulk: None,
            installed_config: ScreenConfig::default(),
            config: ScreenConfig::default(),
            origin,
            yaw,
            motor_on: false,
            time: 0.,
            elapsed_time_s: 0.,
            particles: Vec::new(),
            undersize: Vec::new(),
            oversize: Vec::new(),
            spills: Vec::new(),
            free_water_m3: 0.,
            admitted: Material::default(),
            delivered: Material::default(),
            released: Material::default(),
            free_water_admitted_m3: 0.,
            free_water_delivered_m3: 0.,
            last_step: ScreenStep::default(),
            next_id: 0,
            frame: None,
        }
    }
    fn downstream(&self) -> DVec3 {
        self.frame.map_or_else(
            || DVec3::new((self.yaw as f64).sin(), 0., (self.yaw as f64).cos()),
            |frame| frame.downstream,
        )
    }
    fn across(&self) -> DVec3 {
        self.frame.map_or_else(
            || DVec3::new((self.yaw as f64).cos(), 0., -(self.yaw as f64).sin()),
            |frame| frame.across,
        )
    }
    fn normal(&self) -> DVec3 {
        self.frame.map_or_else(
            || {
                (DVec3::Y + self.downstream() * self.config.slope_deg.to_radians().tan())
                    .normalize()
            },
            |frame| frame.normal,
        )
    }
    pub fn to_world(&self, local: DVec3) -> DVec3 {
        self.origin.as_dvec3()
            + self.downstream() * local.x
            + DVec3::Y * local.y
            + self.across() * local.z
    }
    pub fn to_local(&self, world: DVec3) -> DVec3 {
        let q = world - self.origin.as_dvec3();
        DVec3::new(q.dot(self.downstream()), q.y, q.dot(self.across()))
    }
    /// World translation and velocity of the wire assembly. Gutter/bin are fixed.
    pub fn offset_at(&self, time: f64) -> (DVec3, DVec3) {
        let frame = self.frame.unwrap_or_else(|| ScreenFrame::new(self));
        let w = frame.omega;
        let phase = w * if self.motor_on { time } else { self.time };
        let tangent = frame.tangent_amplitude;
        let normal = frame.normal_amplitude;
        let position = tangent * phase.sin() + normal * (1. - phase.cos());
        let velocity = if self.motor_on {
            w * (tangent * phase.cos() + normal * phase.sin())
        } else {
            DVec3::ZERO
        };
        (position, velocity)
    }
    /// Same finite, round wires used by contact. Cross wires touch the lower
    /// side of longitudinal wires; their interiors do not overlap at crossings.
    pub fn wire_segments(&self, time: f64) -> Vec<(DVec3, DVec3, f64)> {
        let offset = self.offset_at(time).0;
        self.rest_wires()
            .into_iter()
            .map(|w| (w.a + offset, w.b + offset, w.radius))
            .collect()
    }
    fn rest_wires(&self) -> Vec<Wire> {
        let c = &self.config;
        let pitch = c.pitch();
        let slope = c.slope_deg.to_radians().tan();
        let n = self.normal() * c.wire_radius_m;
        let mut out = Vec::new();
        let mut add = |a: DVec3, b: DVec3, level: DVec3| {
            let shift = DVec3::new(
                level.dot(self.downstream()),
                level.y,
                level.dot(self.across()),
            );
            out.push(Wire {
                a: self.to_world(a) + level,
                b: self.to_world(b) + level,
                local_a: a + shift,
                local_b: b + shift,
                radius: c.wire_radius_m,
            });
        };
        for k in
            (-(c.width_m * 0.5 / pitch).floor() as i32)..=((c.width_m * 0.5 / pitch).floor() as i32)
        {
            let z = k as f64 * pitch;
            add(
                DVec3::new(0., 0., z),
                DVec3::new(c.length_m, -c.length_m * slope, z),
                n,
            );
        }
        let pitch_x = pitch * c.slope_deg.to_radians().cos();
        for k in 0..=(c.length_m / pitch_x).floor() as usize {
            let x = k as f64 * pitch_x;
            add(
                DVec3::new(x, -x * slope, -c.width_m * 0.5),
                DVec3::new(x, -x * slope, c.width_m * 0.5),
                -n,
            );
        }
        // Finite perimeter rails close the incomplete edge cells.
        for z in [-c.width_m * 0.5, c.width_m * 0.5] {
            add(
                DVec3::new(0., 0., z),
                DVec3::new(c.length_m, -c.length_m * slope, z),
                n,
            );
        }
        add(
            DVec3::new(c.length_m, -c.length_m * slope, -c.width_m * 0.5),
            DVec3::new(c.length_m, -c.length_m * slope, c.width_m * 0.5),
            -n,
        );
        out
    }
    fn wire_ranges(&self, low: DVec3, high: DVec3, total: usize) -> [std::ops::Range<usize>; 3] {
        let pitch = self.config.pitch();
        let pitch_x = pitch * self.config.slope_deg.to_radians().cos();
        let half = (self.config.width_m * 0.5 / pitch).floor() as i64;
        let count_z = (half * 2 + 1) as usize;
        let count_x = (self.config.length_m / pitch_x).floor() as usize + 1;
        let bounds = |a: f64, b: f64, p: f64, offset: i64, count: usize| {
            let lo = ((a / p).ceil() as i64 - offset).clamp(0, count as i64) as usize;
            let hi = ((b / p).floor() as i64 - offset + 1).clamp(0, count as i64) as usize;
            lo..hi.max(lo)
        };
        let z = bounds(low.z, high.z, pitch, -half, count_z);
        // Cross wires occupy the lower normal layer: its world-normal offset
        // also shifts local downstream x. Index their actual centre lines.
        let cross_shift = self.config.wire_radius_m * self.config.slope_deg.to_radians().sin();
        let x = bounds(
            low.x + cross_shift,
            high.x + cross_shift,
            pitch_x,
            0,
            count_x,
        );
        [
            z,
            (count_z + x.start)..(count_z + x.end),
            (count_z + count_x)..total,
        ]
    }
    /// Static receiving V-gutter: centre y=-.15 at inlet, -.55 at outlet.
    pub fn gutter_triangles(&self) -> Vec<[DVec3; 3]> {
        let c = &self.config;
        let mut out = Vec::new();
        for sign in [-1., 1.] {
            let point = |x: f64, z: f64| {
                self.to_world(DVec3::new(
                    x,
                    -0.15 - 0.40 * x / c.length_m + 0.08 * z.abs() / (c.width_m * 0.5),
                    z,
                ))
            };
            quad(
                &mut out,
                [
                    point(0., 0.),
                    point(c.length_m, 0.),
                    point(c.length_m, sign * c.width_m * 0.5),
                    point(0., sign * c.width_m * 0.5),
                ],
            );
        }
        out
    }
    /// Catch bin floor and three outer walls; upstream entry remains open.
    pub fn bin_triangles(&self) -> Vec<[DVec3; 3]> {
        let c = &self.config;
        let y = -c.length_m * c.slope_deg.to_radians().tan() - 0.10;
        let p = |x, dy, z| self.to_world(DVec3::new(x, y + dy, z));
        let (a, b, w) = (c.length_m, c.length_m + 0.18, c.width_m * 0.5);
        let mut out = Vec::new();
        quad(
            &mut out,
            [p(a, 0., -w), p(b, 0., -w), p(b, 0., w), p(a, 0., w)],
        );
        quad(
            &mut out,
            [p(b, 0., -w), p(b, 0.12, -w), p(b, 0.12, w), p(b, 0., w)],
        );
        for z in [-w, w] {
            quad(
                &mut out,
                [p(a, 0., z), p(a, 0.12, z), p(b, 0.12, z), p(b, 0., z)],
            );
        }
        out
    }
    pub fn set_config(&mut self, config: ScreenConfig) -> Result<(), String> {
        if self.motor_on || self.stock_count() != 0 || self.free_water_m3 != 0. {
            return Err(
                "Stop and empty the screen and its product stocks before reconfiguration".into(),
            );
        }
        if !config.valid() {
            return Err("Unsupported screen dimensions or drive settings".into());
        }
        self.installed_config = config.clone();
        self.config = config;
        self.time = 0.;
        Ok(())
    }
    /// Select the statistical bulk deck only while stopped and completely empty.
    /// Detailed wire contacts remain the default and an independent reference.
    pub fn set_bulk_deck(&mut self, config: Option<BulkDeck>) -> Result<(), String> {
        if !self.valid() || self.motor_on || self.stock_count() != 0 || self.free_water_m3 != 0. {
            return Err("Stop and empty screen before changing its processing model".into());
        }
        if config.as_ref().is_some_and(|c| !c.valid()) {
            return Err("Invalid bulk screen parameters".into());
        }
        self.bulk = config.map(BulkState::new);
        Ok(())
    }
    pub fn bulk_deck(&self) -> Option<&BulkDeck> {
        self.bulk.as_ref().map(|b| &b.config)
    }
    pub fn processing_model_version(&self) -> u32 {
        match self.bulk_deck().map(|c| c.projection) {
            None => 1,
            Some(BulkProjection::CellBinsV1) => 2,
            Some(BulkProjection::SmoothCellsV2) => 3,
        }
    }
    /// Derived diagnostics own no additional material or water.
    pub fn bulk_bed_field(&self) -> Option<BulkBedField> {
        self.bulk.as_ref().map(|b| b.field_at(self, self.time))
    }
    pub fn bulk_bed(&self) -> Option<Vec<(f64, f64)>> {
        self.bulk_bed_field()
            .map(|b| b.cells().iter().map(|c| (c.depth_m, c.blinded)).collect())
    }
    /// Supported IDs represented by the derived field; airborne/products retain
    /// their independent visual/physical state.
    pub fn bulk_bed_particles(&self) -> Vec<u64> {
        self.bulk
            .as_ref()
            .map_or_else(Vec::new, |b| b.supported_ids(self))
    }
    fn stock_count(&self) -> usize {
        self.particles.len() + self.undersize.len() + self.oversize.len() + self.spills.len()
    }
    pub fn try_reserve_feed(&mut self, count: usize) -> Result<(), String> {
        if !self.valid()
            || self
                .stock_count()
                .checked_add(count)
                .is_none_or(|n| n > MAX_SCREEN_PARTICLES)
        {
            return Err("Screen storage full or invalid; feed retained by donor".into());
        }
        self.particles
            .try_reserve(count)
            .map_err(|_| "Cannot reserve screen feed storage".to_string())
    }
    pub fn admit(&mut self, grains: &[Grain], moisture: f64) -> Result<(), String> {
        if !moisture.is_finite()
            || moisture < 0.
            || !(self.free_water_m3 + moisture).is_finite()
            || self.next_id.checked_add(grains.len() as u64).is_none()
        {
            return Err("Invalid screen water or identity; feed retained".into());
        }
        for g in grains {
            if !valid_grain(g) {
                return Err("Invalid screen grain; whole feed batch retained".into());
            }
            let p = self.to_local(g.position.as_dvec3() - self.offset_at(self.time).0);
            if p.x < 0.
                || p.x > self.config.length_m
                || p.z.abs() > self.config.width_m * 0.5
                || self.deck_distance(p)
                    < g.diameter_m as f64 * 0.5 + 2. * self.config.wire_radius_m
            {
                return Err("Screen feed must enter above the real deck footprint".into());
            }
        }
        let mut admitted = self.admitted;
        for g in grains {
            admitted.add(g.material);
        }
        if !admitted.valid() || !(self.free_water_admitted_m3 + moisture).is_finite() {
            return Err("Screen admission totals overflowed; feed retained".into());
        }
        self.try_reserve_feed(grains.len())?;
        for g in grains {
            self.next_id += 1;
            if let Some(bulk) = &mut self.bulk {
                bulk.admit(self.next_id);
            }
            self.admitted.add(g.material);
            self.particles.push(ScreenParticle {
                id: self.next_id,
                grain: g.clone(),
                position: g.position.as_dvec3(),
                velocity: g.velocity.as_dvec3(),
                angular_velocity: g.angular_velocity.as_dvec3(),
                in_gutter: false,
                passed_deck: false,
            });
        }
        self.free_water_m3 += moisture;
        self.free_water_admitted_m3 += moisture;
        Ok(())
    }
    pub fn take_undersize(&mut self) -> Vec<Grain> {
        let out = std::mem::take(&mut self.undersize);
        for g in &out {
            self.delivered.add(g.material);
        }
        out
    }
    pub fn take_oversize(&mut self) -> Vec<Grain> {
        let out = std::mem::take(&mut self.oversize);
        for g in &out {
            self.released.add(g.material);
        }
        out
    }
    pub fn take_spills(&mut self) -> Vec<Grain> {
        let out = std::mem::take(&mut self.spills);
        for g in &out {
            self.released.add(g.material);
        }
        out
    }
    pub fn take_free_water(&mut self) -> f64 {
        let v = self.free_water_m3;
        self.free_water_m3 = 0.;
        self.free_water_delivered_m3 += v;
        v
    }
    /// Explicit stopped-machine removal, not simulated passage or recovery.
    pub fn cleanout(&mut self) -> Result<Vec<Grain>, String> {
        if self.motor_on || !self.valid() {
            return Err("Stop screen before manual cleanout".into());
        }
        let mut out = Vec::new();
        out.try_reserve(self.stock_count())
            .map_err(|_| "Cannot reserve cleanout; screen stock retained")?;
        out.extend(self.particles.drain(..).map(|p| p.grain));
        if let Some(bulk) = &mut self.bulk {
            bulk.clear();
        }
        out.append(&mut self.undersize);
        out.append(&mut self.oversize);
        out.append(&mut self.spills);
        for g in &out {
            self.released.add(g.material);
        }
        Ok(out)
    }
    pub fn material(&self) -> Material {
        let mut out = Material::default();
        for g in self
            .particles
            .iter()
            .map(|p| &p.grain)
            .chain(&self.undersize)
            .chain(&self.oversize)
            .chain(&self.spills)
        {
            out.add(g.material);
        }
        out
    }
    pub fn valid(&self) -> bool {
        if !self.config.valid()
            || self.config != self.installed_config
            || !self.origin.is_finite()
            || !self.yaw.is_finite()
            || !self.time.is_finite()
            || self.time < 0.
            || !self.elapsed_time_s.is_finite()
            || self.elapsed_time_s < self.time
            || self.stock_count() > MAX_SCREEN_PARTICLES
            || !self.admitted.valid()
            || !self.delivered.valid()
            || !self.released.valid()
            || ![
                self.free_water_m3,
                self.free_water_admitted_m3,
                self.free_water_delivered_m3,
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.)
        {
            return false;
        }
        if self.bulk.as_ref().is_some_and(|b| !b.valid(self)) {
            return false;
        }
        let mut ids = std::collections::BTreeSet::new();
        for p in &self.particles {
            if p.id == 0
                || p.id > self.next_id
                || !ids.insert(p.id)
                || !valid_grain(&p.grain)
                || !p.position.is_finite()
                || !p.velocity.is_finite()
                || !p.angular_velocity.is_finite()
                || p.angular_velocity.length() >= 1e7
                || p.grain.angular_velocity != p.angular_velocity.as_vec3()
                || p.position.length() > 10000.
                || p.velocity.length() > 200.
                || p.grain.position != p.position.as_vec3()
                || p.grain.velocity != p.velocity.as_vec3()
                || (p.in_gutter && !p.passed_deck)
            {
                return false;
            }
        }
        if self
            .undersize
            .iter()
            .chain(&self.oversize)
            .chain(&self.spills)
            .any(|g| !valid_grain(g))
        {
            return false;
        }
        let mut accounted = self.material();
        accounted.add(self.delivered);
        accounted.add(self.released);
        same_material(accounted, self.admitted)
            && close(
                self.free_water_m3 + self.free_water_delivered_m3,
                self.free_water_admitted_m3,
            )
            && [
                self.last_step.motor_work_j,
                self.last_step.projection_work_j,
                self.last_step.maximum_repair_m,
            ]
            .iter()
            .all(|v| v.is_finite())
    }
    /// Commit only a complete bounded step. Geometry/work/capacity refusal leaves
    /// every input particle, output queue, clock and material counter unchanged.
    pub fn step(&mut self, dt: f32) -> Result<(), String> {
        if !self.valid() || !dt.is_finite() || !(0.0..=0.1).contains(&dt) {
            return Err("Invalid screen step; state retained".into());
        }
        if dt == 0. {
            return Ok(());
        }
        let mut next = self.clone();
        next.last_step = ScreenStep::default();
        next.frame = Some(ScreenFrame::new(&next));
        next.undersize
            .try_reserve(next.particles.len())
            .map_err(|_| "Cannot reserve screen underflow".to_string())?;
        next.oversize
            .try_reserve(next.particles.len())
            .map_err(|_| "Cannot reserve screen catch bin".to_string())?;
        next.spills
            .try_reserve(next.particles.len())
            .map_err(|_| "Cannot reserve screen spills".to_string())?;
        if let Some(mut bulk) = next.bulk.take() {
            bulk.step(&mut next, dt as f64)?;
            next.bulk = Some(bulk);
        } else {
            let wires = next.rest_wires();
            let gutter = next.gutter_triangles();
            let bin = next.bin_triangles();
            let mut particles = std::mem::take(&mut next.particles);
            for mut particle in particles.drain(..) {
                match next.advance(&mut particle, dt as f64, &wires, &gutter, &bin)? {
                    Some(Surface::Gutter) => next.undersize.push(particle.grain),
                    Some(Surface::Bin) => next.oversize.push(particle.grain),
                    Some(Surface::Wire) => next.spills.push(particle.grain),
                    None => next.particles.push(particle),
                }
            }
        }
        next.elapsed_time_s += dt as f64;
        if next.motor_on {
            next.time += dt as f64;
        }
        next.frame = None;
        if !next.valid() {
            return Err("Screen postcondition refused; state retained".into());
        }
        *self = next;
        Ok(())
    }
    fn deck_distance(&self, p: DVec3) -> f64 {
        let frame = self.frame.unwrap_or_else(|| ScreenFrame::new(self));
        (p.y + p.x * frame.slope_tan) * frame.slope_cos
    }
    fn drive_time(&self, elapsed: f64) -> f64 {
        self.time + if self.motor_on { elapsed } else { 0. }
    }
    fn passed(&self, p: &mut ScreenParticle, t: f64) {
        let local = self.to_local(p.position - self.offset_at(self.drive_time(t)).0);
        let r = p.grain.diameter_m as f64 * 0.5;
        if local.x >= r
            && local.x <= self.config.length_m - r
            && local.z.abs() <= self.config.width_m * 0.5 - r
            && self.deck_distance(local) < -2. * self.config.wire_radius_m - r
        {
            p.passed_deck = true;
        }
    }
    fn count_query(&mut self) -> Result<(), String> {
        self.last_step.contact_queries += 1;
        if self.last_step.contact_queries > MAX_QUERIES {
            return Err("Screen contact work budget exhausted; state retained".into());
        }
        Ok(())
    }
    fn refine_wire(
        &mut self,
        sweep: &WireSweep,
        bracket: WireBracket,
        depth: usize,
    ) -> Result<Option<Hit>, String> {
        self.count_query()?;
        if depth > 20 {
            return Err("Screen exact-phase contact refinement exhausted; state retained".into());
        }
        let WireBracket {
            lo,
            hi,
            lower,
            upper,
        } = bracket;
        let phase = |u: f64| sweep.time0 + (sweep.time1 - sweep.time0) * u;
        let (offset0, velocity) = lower;
        let q0 = sweep.start + sweep.delta * lo - offset0;
        let q1 = sweep.start + sweep.delta * hi - upper.0;
        let duration = (sweep.time1 - sweep.time0) * (hi - lo);
        let error = sweep.acceleration * duration * duration / 8.;
        let delta = q0 - segment_closest(q0, sweep.wire.a, sweep.wire.b);
        let gap = delta.length() - sweep.radius;
        if gap <= CONTACT_SKIN_M {
            let normal = delta
                .try_normalize()
                .ok_or("Screen refined contact normal undefined")?;
            if (sweep.velocity - velocity).dot(normal) < -1e-10 {
                return Ok(Some(Hit {
                    u: lo,
                    normal,
                    velocity,
                    surface: Surface::Wire,
                    index: sweep.index,
                }));
            }
        }
        // Being inside the fat bracket is not physical contact. Include that
        // entire uncertain interval even if its first chord points away.
        let candidate = if gap <= error + CONTACT_SKIN_M {
            Some(0.)
        } else {
            cast_capsule(
                q0,
                q1 - q0,
                sweep.wire.a,
                sweep.wire.b,
                sweep.radius + error + CONTACT_SKIN_M,
            )
        };
        let Some(candidate) = candidate else {
            return Ok(None);
        };
        if error <= CONTACT_SKIN_M * 0.25 {
            let u = lo + (hi - lo) * candidate;
            let actual = sweep.start + sweep.delta * u;
            let (offset, velocity) = bracket.at(self, sweep, u);
            let delta =
                actual - segment_closest(actual, sweep.wire.a + offset, sweep.wire.b + offset);
            let normal = delta
                .try_normalize()
                .ok_or("Screen refined contact normal undefined")?;
            if delta.length() - sweep.radius <= CONTACT_SKIN_M + 1e-12
                && (sweep.velocity - velocity).dot(normal) < -1e-10
            {
                return Ok(Some(Hit {
                    u,
                    normal,
                    velocity,
                    surface: Surface::Wire,
                    index: sweep.index,
                }));
            }
            // A separating bracket start must not hide a later true entry.
            if let Some(local) = cast_capsule(q0, q1 - q0, sweep.wire.a, sweep.wire.b, sweep.radius)
            {
                let u = lo + (hi - lo) * local;
                let actual = sweep.start + sweep.delta * u;
                let (offset, velocity) = bracket.at(self, sweep, u);
                let delta =
                    actual - segment_closest(actual, sweep.wire.a + offset, sweep.wire.b + offset);
                let normal = delta
                    .try_normalize()
                    .ok_or("Screen refined contact normal undefined")?;
                if delta.length() - sweep.radius <= CONTACT_SKIN_M + 1e-12
                    && (sweep.velocity - velocity).dot(normal) < -1e-10
                {
                    return Ok(Some(Hit {
                        u,
                        normal,
                        velocity,
                        surface: Surface::Wire,
                        index: sweep.index,
                    }));
                }
            }
            return Ok(None);
        }
        let mid = (lo + hi) * 0.5;
        let middle = self.offset_at(phase(mid));
        if let Some(hit) = self.refine_wire(
            sweep,
            WireBracket {
                lo,
                hi: mid,
                lower,
                upper: middle,
            },
            depth + 1,
        )? {
            return Ok(Some(hit));
        }
        self.refine_wire(
            sweep,
            WireBracket {
                lo: mid,
                hi,
                lower: middle,
                upper,
            },
            depth + 1,
        )
    }
    // Original uncached arithmetic is a test-only refinement reference.
    #[cfg(test)]
    fn refine_wire_reference(
        &mut self,
        sweep: &WireSweep,
        lo: f64,
        hi: f64,
        depth: usize,
    ) -> Result<Option<Hit>, String> {
        self.count_query()?;
        if depth > 20 {
            return Err("Screen exact-phase contact refinement exhausted; state retained".into());
        }
        let phase = |u: f64| sweep.time0 + (sweep.time1 - sweep.time0) * u;
        let (offset0, velocity) = self.offset_at(phase(lo));
        let q0 = sweep.start + sweep.delta * lo - offset0;
        let q1 = sweep.start + sweep.delta * hi - self.offset_at(phase(hi)).0;
        let duration = (sweep.time1 - sweep.time0) * (hi - lo);
        let error = sweep.acceleration * duration * duration / 8.;
        let delta = q0 - segment_closest(q0, sweep.wire.a, sweep.wire.b);
        let gap = delta.length() - sweep.radius;
        if gap <= CONTACT_SKIN_M {
            let normal = delta
                .try_normalize()
                .ok_or("Screen refined contact normal undefined")?;
            if (sweep.velocity - velocity).dot(normal) < -1e-10 {
                return Ok(Some(Hit {
                    u: lo,
                    normal,
                    velocity,
                    surface: Surface::Wire,
                    index: sweep.index,
                }));
            }
        }
        // Being inside the fat bracket is not physical contact. Include that
        // entire uncertain interval even if its first chord points away.
        let candidate = if gap <= error + CONTACT_SKIN_M {
            Some(0.)
        } else {
            cast_capsule(
                q0,
                q1 - q0,
                sweep.wire.a,
                sweep.wire.b,
                sweep.radius + error + CONTACT_SKIN_M,
            )
        };
        let Some(candidate) = candidate else {
            return Ok(None);
        };
        if error <= CONTACT_SKIN_M * 0.25 {
            let u = lo + (hi - lo) * candidate;
            let actual = sweep.start + sweep.delta * u;
            let (offset, velocity) = self.offset_at(phase(u));
            let delta =
                actual - segment_closest(actual, sweep.wire.a + offset, sweep.wire.b + offset);
            let normal = delta
                .try_normalize()
                .ok_or("Screen refined contact normal undefined")?;
            if delta.length() - sweep.radius <= CONTACT_SKIN_M + 1e-12
                && (sweep.velocity - velocity).dot(normal) < -1e-10
            {
                return Ok(Some(Hit {
                    u,
                    normal,
                    velocity,
                    surface: Surface::Wire,
                    index: sweep.index,
                }));
            }
            // A separating bracket start must not hide a later true entry.
            if let Some(local) = cast_capsule(q0, q1 - q0, sweep.wire.a, sweep.wire.b, sweep.radius)
            {
                let u = lo + (hi - lo) * local;
                let actual = sweep.start + sweep.delta * u;
                let (offset, velocity) = self.offset_at(phase(u));
                let delta =
                    actual - segment_closest(actual, sweep.wire.a + offset, sweep.wire.b + offset);
                let normal = delta
                    .try_normalize()
                    .ok_or("Screen refined contact normal undefined")?;
                if delta.length() - sweep.radius <= CONTACT_SKIN_M + 1e-12
                    && (sweep.velocity - velocity).dot(normal) < -1e-10
                {
                    return Ok(Some(Hit {
                        u,
                        normal,
                        velocity,
                        surface: Surface::Wire,
                        index: sweep.index,
                    }));
                }
            }
            return Ok(None);
        }
        let mid = (lo + hi) * 0.5;
        if let Some(hit) = self.refine_wire_reference(sweep, lo, mid, depth + 1)? {
            return Ok(Some(hit));
        }
        self.refine_wire_reference(sweep, mid, hi, depth + 1)
    }
    fn advance(
        &mut self,
        p: &mut ScreenParticle,
        dt: f64,
        wires: &[Wire],
        gutter: &[[DVec3; 3]],
        bin: &[[DVec3; 3]],
    ) -> Result<Option<Surface>, String> {
        let r = p.grain.diameter_m as f64 * 0.5;
        let omega = std::f64::consts::TAU * self.config.frequency_hz;
        let acceleration = G + if self.motor_on {
            omega * omega * (self.config.amplitude_downstream_m + self.config.amplitude_normal_m)
        } else {
            0.
        };
        // Kick-drift-kick approximates gravity with straight drift intervals.
        // Gravity and sinusoidal-wire displacement each differ from their
        // chord by at most |acceleration|*h^2/8. Inflate the continuous wire
        // casts by their summed bound (<=0.5% physical r), then check the exact
        // sinusoidal endpoint geometry. This is a convergent time discretization,
        // not an exact parabolic impact or exact contact-time motor solution.
        let hmax = (1. / 960_f64).min((8. * r * 0.005 / acceleration.max(1e-30)).sqrt());
        let mut elapsed = 0.;
        while elapsed < dt {
            self.last_step.intervals += 1;
            if self.last_step.intervals > MAX_INTERVALS {
                return Err("Screen interval budget exhausted; state retained".into());
            }
            let h = hmax.min(dt - elapsed);
            let guard = acceleration * h * h / 8. + 1e-10;
            p.velocity.y -= G * h * 0.5;
            let mut remaining = h;
            let mut at = elapsed;
            let mut events = 0;
            // A support manifold owns the remaining straight drift. Reapplying
            // restitution separately to the same adjacent wires creates Zeno
            // impacts; a genuinely new feature triggers another joint solve.
            let mut held: Vec<Hit> = Vec::new();
            while remaining > 1e-12 {
                let start = p.position;
                let end = start + p.velocity * remaining;
                let t0 = self.drive_time(at);
                let t1 = self.drive_time(at + remaining);
                let drive0 = self.offset_at(t0);
                let drive1 = self.offset_at(t1);
                let off0 = drive0.0;
                let off1 = drive1.0;
                // Preserve the uncached root's exact phase expressions,
                // including their floating-point evaluation order.
                let lower_time = t0 + (t1 - t0) * 0.;
                let upper_time = t0 + (t1 - t0) * 1.;
                let bracket = WireBracket {
                    lo: 0.,
                    hi: 1.,
                    lower: if lower_time.to_bits() == t0.to_bits() {
                        drive0
                    } else {
                        self.offset_at(lower_time)
                    },
                    upper: if upper_time.to_bits() == t1.to_bits() {
                        drive1
                    } else {
                        self.offset_at(upper_time)
                    },
                };
                let local0 = self.to_local(start - off0);
                let local1 = self.to_local(end - off1);
                let low = local0.min(local1) - DVec3::splat(r + self.config.wire_radius_m + guard);
                let high = local0.max(local1) + DVec3::splat(r + self.config.wire_radius_m + guard);
                let mut best: Option<Hit> = None;
                for index in self
                    .wire_ranges(low, high, wires.len())
                    .into_iter()
                    .flatten()
                {
                    let wire = &wires[index];
                    if held
                        .iter()
                        .any(|c| c.surface == Surface::Wire && c.index == index)
                    {
                        continue;
                    }
                    let a = wire.local_a;
                    let b = wire.local_b;
                    if a.min(b).cmpgt(high).any() || a.max(b).cmplt(low).any() {
                        continue;
                    }
                    let sweep = WireSweep {
                        start,
                        delta: end - start,
                        velocity: p.velocity,
                        time0: t0,
                        time1: t1,
                        wire: *wire,
                        radius: r + wire.radius,
                        index,
                        acceleration: acceleration - G,
                    };
                    if let Some(hit) = self.refine_wire(&sweep, bracket, 0)? {
                        choose(&mut best, hit);
                    }
                }
                for (triangles, surface) in [(gutter, Surface::Gutter), (bin, Surface::Bin)] {
                    for (index, tri) in triangles.iter().enumerate() {
                        if held
                            .iter()
                            .any(|c| c.surface == surface && c.index == index)
                        {
                            continue;
                        }
                        self.count_query()?;
                        if !triangle_bounds_overlap(start, end, *tri, r + 1e-12) {
                            continue;
                        }
                        if let Some((u, normal)) = cast_triangle(start, end - start, *tri, r) {
                            if u < 1e-9 && p.velocity.dot(normal) >= -1e-10 {
                                continue;
                            }
                            choose(
                                &mut best,
                                Hit {
                                    u,
                                    normal,
                                    velocity: DVec3::ZERO,
                                    surface,
                                    index,
                                },
                            );
                        }
                    }
                }
                let Some(hit) = best else {
                    p.position = end;
                    break;
                };
                let travel = remaining * hit.u;
                p.position += p.velocity * travel;
                at += travel;
                remaining -= travel;
                self.passed(p, at);
                if hit.surface == Surface::Gutter && p.passed_deck {
                    p.in_gutter = true;
                }
                let contacts = self.manifold(
                    p,
                    self.drive_time(at),
                    wires,
                    gutter,
                    bin,
                    CONTACT_SKIN_M + 1e-12,
                    Some(hit),
                )?;
                self.solve_manifold(
                    p,
                    &contacts,
                    &held,
                    true,
                    ContactContext {
                        time: self.drive_time(at),
                        interval: h,
                        guard,
                        synthetic: false,
                    },
                )?;
                held = contacts;
                self.last_step.contacts += held.len();
                let bin_local = self.to_local(p.position);
                let bin_floor =
                    -self.config.length_m * self.config.slope_deg.to_radians().tan() - 0.10;
                let on_bin_floor = (bin_local.y - bin_floor - r).abs() <= 1e-8
                    && bin_local.x >= self.config.length_m
                    && bin_local.x <= self.config.length_m + 0.18
                    && bin_local.z.abs() <= self.config.width_m * 0.5;
                if hit.surface == Surface::Bin
                    && !p.passed_deck
                    && hit.normal.y > 0.99
                    && on_bin_floor
                {
                    p.grain.position = p.position.as_vec3();
                    p.grain.velocity = p.velocity.as_vec3();
                    rotate_grain(p, at - elapsed);
                    p.grain.resting = p.velocity.length_squared() < 1e-8;
                    return Ok(Some(Surface::Bin));
                }
                events += 1;
                if events > MAX_EVENTS {
                    return Err(format!("Screen impact budget exhausted; state retained: particle={} r={r:.9e} at={at:.9e} remaining={remaining:.9e} local={:?} velocity={:?} held={:?}",p.id,self.to_local(p.position).to_array(),p.velocity.to_array(),held.iter().map(|c|(c.surface,c.index)).collect::<Vec<_>>()));
                }
            }
            p.velocity.y -= G * h * 0.5;
            elapsed += h;
            self.passed(p, elapsed);
            // Only the bounded curved-chord error may need a clearance repair.
            let (off, _) = self.offset_at(self.drive_time(elapsed));
            let local_point = self.to_local(p.position - off);
            let pad = DVec3::splat(r + self.config.wire_radius_m + 4. * guard + 1e-8);
            let mut correction_constraints = Vec::new();
            let mut needs_correction = false;
            for index in self
                .wire_ranges(local_point - pad, local_point + pad, wires.len())
                .into_iter()
                .flatten()
            {
                let wire = &wires[index];
                if wire
                    .local_a
                    .min(wire.local_b)
                    .cmpgt(local_point + pad)
                    .any()
                    || wire
                        .local_a
                        .max(wire.local_b)
                        .cmplt(local_point - pad)
                        .any()
                {
                    continue;
                }
                self.count_query()?;
                let delta = p.position - segment_closest(p.position, wire.a + off, wire.b + off);
                let distance = delta.length();
                let penetration = r + wire.radius - distance;
                if penetration > 0. {
                    needs_correction = true;
                    if penetration > 4. * guard + 1e-8 {
                        return Err(format!("Screen wire penetration exceeded swept bound; state retained: particle={} wire={} r={r:.9e} penetration={penetration:.9e} bound={:.9e} local={:?} velocity={:?} wire_a={:?} wire_b={:?} phase={:.9e} interval={h:.9e}",p.id,index,4.*guard+1e-8,local_point.to_array(),p.velocity.to_array(),wire.local_a.to_array(),wire.local_b.to_array(),self.drive_time(elapsed)));
                    }
                }
                if penetration >= -(4. * guard + 1e-8) {
                    let n = delta
                        .try_normalize()
                        .ok_or("Screen repair normal undefined")?;
                    correction_constraints.push((n, penetration + 1e-10));
                }
            }
            if needs_correction {
                let displacement=project_halfspaces(&correction_constraints).ok_or_else(||format!("Screen simultaneous clearance is infeasible; state retained: particle={} constraints={}",p.id,correction_constraints.len()))?;
                if displacement.length() > 4. * guard + 1e-8 {
                    return Err(format!("Screen simultaneous clearance exceeded swept bound; state retained: particle={} correction={:.9e} bound={:.9e}",p.id,displacement.length(),4.*guard+1e-8));
                }
                p.position += displacement;
                self.last_step.maximum_repair_m =
                    self.last_step.maximum_repair_m.max(displacement.length());
                self.last_step.projection_work_j +=
                    p.grain.material.wet_mass_kg() * G * displacement.y;
                let contacts = self.manifold(
                    p,
                    self.drive_time(elapsed),
                    wires,
                    gutter,
                    bin,
                    CONTACT_SKIN_M + 1e-12,
                    None,
                )?;
                self.solve_manifold(
                    p,
                    &contacts,
                    &held,
                    false,
                    ContactContext {
                        time: self.drive_time(elapsed),
                        interval: h,
                        guard,
                        synthetic: false,
                    },
                )?;
            }
            rotate_grain(p, h);
            p.grain.position = p.position.as_vec3();
            p.grain.velocity = p.velocity.as_vec3();
            p.grain.resting = false;
            if !valid_grain(&p.grain) {
                return Err("Nonfinite or excessive screen motion; state retained".into());
            }
            let local = self.to_local(p.position);
            let c = &self.config;
            let below =
                self.deck_distance(self.to_local(p.position - off)) < -2. * c.wire_radius_m - r;
            if p.in_gutter && below && local.x >= c.length_m + r {
                return Ok(Some(Surface::Gutter));
            }
            if local.x < -r - 0.02
                || local.x > c.length_m + 0.18 + r
                || local.z.abs() > c.width_m * 0.5 + r
                || local.y < -0.8
            {
                return Ok(Some(Surface::Wire));
            }
        }
        Ok(None)
    }
    #[allow(
        clippy::too_many_arguments,
        reason = "Join the three registered surface sets at one physical contact time"
    )]
    fn manifold(
        &mut self,
        p: &ScreenParticle,
        time: f64,
        wires: &[Wire],
        gutter: &[[DVec3; 3]],
        bin: &[[DVec3; 3]],
        skin: f64,
        primary: Option<Hit>,
    ) -> Result<Vec<Hit>, String> {
        let r = p.grain.diameter_m as f64 * 0.5;
        let (offset, velocity) = self.offset_at(time);
        let local = self.to_local(p.position - offset);
        let pad = DVec3::splat(r + self.config.wire_radius_m + skin);
        let mut out = Vec::new();
        for index in self
            .wire_ranges(local - pad, local + pad, wires.len())
            .into_iter()
            .flatten()
        {
            let w = &wires[index];
            if w.local_a.min(w.local_b).cmpgt(local + pad).any()
                || w.local_a.max(w.local_b).cmplt(local - pad).any()
            {
                continue;
            }
            self.count_query()?;
            let delta = p.position - segment_closest(p.position, w.a + offset, w.b + offset);
            if delta.length() <= r + w.radius + skin {
                let normal = delta
                    .try_normalize()
                    .ok_or("Screen manifold wire normal undefined")?;
                out.push(Hit {
                    u: 0.,
                    normal,
                    velocity,
                    surface: Surface::Wire,
                    index,
                });
            }
        }
        for (triangles, surface) in [(gutter, Surface::Gutter), (bin, Surface::Bin)] {
            for (index, t) in triangles.iter().enumerate() {
                self.count_query()?;
                if !triangle_bounds_overlap(p.position, p.position, *t, r + skin + 1e-12) {
                    continue;
                }
                let closest = triangle_closest(p.position, *t);
                let delta = p.position - closest;
                if delta.length() <= r + skin {
                    let normal = delta
                        .try_normalize()
                        .ok_or("Screen manifold triangle normal undefined")?;
                    out.push(Hit {
                        u: 0.,
                        normal,
                        velocity: DVec3::ZERO,
                        surface,
                        index,
                    });
                }
            }
        }
        if let Some(primary) = primary {
            if !out.iter().any(|h| same_feature(*h, primary)) {
                out.push(primary);
            }
        }
        if out.len() > MAX_MANIFOLD {
            return Err("Screen contact manifold storage exhausted; state retained".into());
        }
        Ok(out)
    }
    fn contact_trial(
        &mut self,
        problem: &ContactProblem<'_>,
        mu: f64,
        seed: &[f64],
        sweeps: usize,
    ) -> Result<ContactTrial, String> {
        let mut x = seed.to_vec();
        let mut result = ContactTrial {
            normal_impulses: None,
            impulses: x.clone(),
            converged: false,
            iterations: 0,
            mode: None,
            reason: "outer_iteration_budget",
            direction: "none",
            backtrack: 0,
        };
        for _ in 0..sweeps {
            result.iterations += 1;
            self.count_query()?;
            let state = problem
                .evaluate(&x, mu)
                .ok_or("Screen normal support is infeasible; state retained")?;
            if contact_residual(&state.residual) < 1e-10 && state.normal_residual < 1e-9 {
                result.impulses = x;
                result.converged = true;
                result.reason = "converged";
                return Ok(result);
            }
            let score = state.residual.iter().map(|v| v * v).sum::<f64>();
            for _ in &x {
                self.count_query()?;
            }
            let jacobian = projected_contact_jacobian(problem, &state, &x, mu);
            let rhs: Vec<_> = state.residual.iter().map(|v| -v).collect();
            let mut accepted = None;
            for (kind, direction) in [jacobian.and_then(|j| rank_solve(j, &rhs)), Some(rhs)]
                .into_iter()
                .enumerate()
            {
                let Some(direction) = direction else {
                    continue;
                };
                result.direction = if kind == 0 {
                    "newton"
                } else {
                    "projected_fixed_point"
                };
                for backtrack in 0..12 {
                    result.backtrack = backtrack;
                    let fraction = 2_f64.powi(-(backtrack as i32));
                    let trial: Vec<_> = x
                        .iter()
                        .zip(&direction)
                        .map(|(x, d)| x + fraction * d)
                        .collect();
                    self.count_query()?;
                    let Some(state) = problem.evaluate(&trial, mu) else {
                        continue;
                    };
                    if state.residual.iter().map(|v| v * v).sum::<f64>()
                        < score * (1. - 1e-4 * fraction)
                    {
                        accepted = Some(trial);
                        break;
                    }
                }
                if accepted.is_some() {
                    break;
                }
            }
            let Some(trial) = accepted else {
                result.reason = "no_residual_decrease";
                break;
            };
            x = trial;
        }
        result.impulses = x;
        Ok(result)
    }
    fn static_contact_trial(
        &mut self,
        problem: &ContactProblem<'_>,
        mu: f64,
    ) -> Result<Option<ContactTrial>, String> {
        let rows = problem.rows;
        let count = rows.len();
        if count < 3 || rows.iter().any(|row| row.target != 0.) {
            return Ok(None);
        }
        let boundary = rows[0].hit.velocity;
        if rows
            .iter()
            .any(|row| row.hit.velocity.distance_squared(boundary) > 1e-24)
        {
            return Ok(None);
        }
        let force = boundary - problem.velocity;
        let torque = -problem.spin / 2.5;
        let mut pairs = Vec::new();
        for (i, first) in rows.iter().enumerate() {
            for (j, second) in rows.iter().enumerate().skip(i + 1) {
                pairs.push((i, j, first.hit.normal.dot(second.hit.normal)));
            }
        }
        pairs.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        for (i, j, dot) in pairs {
            let ni = rows[i].hit.normal;
            let nj = rows[j].hit.normal;
            let d = ni - nj;
            let norm = d.length_squared();
            let gain = 1. - dot;
            let di = ni * dot - nj;
            let dj = nj * dot - ni;
            let margins = [mu * gain - di.length(), mu * gain - dj.length()];
            if norm == 0. || margins.iter().any(|m| *m <= 0.) {
                continue;
            }
            let kappa = d.cross(ni).dot(force) + d.dot(torque);
            for (k, third_row) in rows.iter().enumerate() {
                if k == i || k == j {
                    continue;
                }
                let nk = third_row.hit.normal;
                let a = d.cross(ni - nk);
                let tangent = a - nk * a.dot(nk);
                let length = tangent.length();
                for sign in [-1., 1.] {
                    self.count_query()?;
                    let denominator = a.dot(nk) + sign * mu * length;
                    let third = if kappa == 0. {
                        DVec3::ZERO
                    } else {
                        if denominator == 0. {
                            continue;
                        }
                        let magnitude = kappa / denominator;
                        if !magnitude.is_finite() || magnitude <= 0. {
                            continue;
                        }
                        (nk + tangent.normalize_or_zero() * (sign * mu)) * magnitude
                    };
                    let remaining = force - third;
                    let moment = torque + nk.cross(third) + nj.cross(remaining);
                    let mut first = d.cross(moment) / norm;
                    first += d * (remaining.dot(d) / (2. * norm));
                    let mut second = remaining - first;
                    let mut gamma = 0_f64;
                    for (normal, impulse, margin) in
                        [(ni, first, margins[0]), (nj, second, margins[1])]
                    {
                        let lambda = normal.dot(impulse);
                        let tangent = impulse - normal * lambda;
                        gamma = gamma
                            .max(-lambda / gain)
                            .max((tangent.length() - mu * lambda) / margin);
                    }
                    if !gamma.is_finite() {
                        continue;
                    }
                    first += d * gamma;
                    second -= d * gamma;
                    let mut impulses = vec![DVec3::ZERO; count];
                    impulses[i] = first;
                    impulses[j] = second;
                    impulses[k] = third;
                    let normals: Vec<_> = rows
                        .iter()
                        .zip(&impulses)
                        .map(|(row, j)| row.hit.normal.dot(*j))
                        .collect();
                    let mut tangents = Vec::with_capacity(count * 2);
                    for (index, j) in impulses.iter().enumerate() {
                        tangents.push(j.dot(problem.bases[index].0));
                        tangents.push(j.dot(problem.bases[index].1));
                    }
                    self.count_query()?;
                    let Some(state) = evaluate_contact_impulses(
                        rows,
                        problem.bases,
                        &tangents,
                        problem.velocity,
                        problem.spin,
                        mu,
                        Some(&normals),
                    ) else {
                        continue;
                    };
                    if state.velocity.distance(boundary) < 1e-10
                        && state.scaled_spin.length() < 1e-10
                        && state.normal_residual < 1e-9
                        && contact_residual(&state.residual) < 1e-10
                    {
                        return Ok(Some(ContactTrial {
                            impulses: tangents,
                            normal_impulses: Some(normals),
                            converged: true,
                            iterations: 1,
                            mode: None,
                            reason: "converged",
                            direction: "geometric_static_wrench",
                            backtrack: 0,
                        }));
                    }
                }
            }
        }
        Ok(None)
    }
    fn contact_solution(
        &mut self,
        problem: &ContactProblem<'_>,
        mu: f64,
    ) -> Result<ContactSolution, String> {
        let zero = vec![0.; problem.rows.len() * 2];
        self.count_query()?;
        let baseline = problem
            .evaluate(&zero, mu)
            .ok_or("Screen normal support is infeasible; state retained")?;
        if baseline.projection.count == 0
            && baseline.normal_residual < 1e-9
            && contact_residual(&baseline.residual) < 1e-10
        {
            let trial = ContactTrial {
                impulses: zero,
                normal_impulses: None,
                converged: true,
                iterations: 1,
                mode: None,
                reason: "converged",
                direction: "separating_no_impulse",
                backtrack: 0,
            };
            return Ok(ContactSolution {
                trial,
                trial_mu: mu,
                iterations: 1,
                seed_attempts: 0,
            });
        }
        if let Some(trial) = self.static_contact_trial(problem, mu)? {
            return Ok(ContactSolution {
                trial,
                trial_mu: mu,
                iterations: 1,
                seed_attempts: 0,
            });
        }
        if problem.rows.len() == 1 {
            self.count_query()?;
            let row = &problem.rows[0];
            let n = row.hit.normal;
            let incoming = problem.velocity + problem.spin.cross(-n) - row.hit.velocity;
            let normal = (row.target - incoming.dot(n)).max(0.);
            let slip = incoming - n * incoming.dot(n);
            let tangent = -slip.normalize_or_zero() * (slip.length() / 3.5).min(mu * normal);
            let impulses = vec![
                tangent.dot(problem.bases[0].0),
                tangent.dot(problem.bases[0].1),
            ];
            let state = problem
                .evaluate(&impulses, mu)
                .ok_or("Screen isolated impact is infeasible; state retained")?;
            let converged =
                contact_residual(&state.residual) < 1e-10 && state.normal_residual < 1e-9;
            let trial = ContactTrial {
                normal_impulses: None,
                impulses,
                converged,
                iterations: 1,
                mode: None,
                reason: if converged {
                    "converged"
                } else {
                    "isolated_residual"
                },
                direction: "isolated_analytic",
                backtrack: 0,
            };
            return Ok(ContactSolution {
                trial,
                trial_mu: mu,
                iterations: 1,
                seed_attempts: 0,
            });
        }
        if problem.rows.len() != 2 || mu == 0. {
            let trial = self.contact_trial(problem, mu, &zero, MANIFOLD_SWEEPS)?;
            return Ok(ContactSolution {
                iterations: trial.iterations,
                trial,
                trial_mu: mu,
                seed_attempts: 0,
            });
        }
        use ContactMode::{Open, Slide, Stick};
        let mut trial = ContactTrial {
            normal_impulses: None,
            impulses: vec![0.; 4],
            converged: false,
            iterations: 0,
            mode: None,
            reason: "reduced_seed_budget",
            direction: "reduced_alpha",
            backtrack: 0,
        };
        let mut seed_attempts = 0;
        for modes in [
            [Stick, Stick],
            [Slide, Stick],
            [Stick, Slide],
            [Slide, Slide],
            [Open, Open],
            [Stick, Open],
            [Open, Stick],
            [Slide, Open],
            [Open, Slide],
        ] {
            // Keep the original double-stick branch first. For the other
            // modes an exact-Jacobian proposal can avoid repeated seed
            // searches. It changes no law/tolerance and must certify every
            // original contact and the final boundary-work energy gate.
            if modes == [Slide, Stick] {
                let mut fast = self.contact_trial(problem, mu, &zero, 8)?;
                if fast.converged {
                    self.count_query()?;
                    if let Some(state) = problem.evaluate(&fast.impulses, mu) {
                        let inferred = std::array::from_fn(|i| {
                            let row = &problem.rows[i];
                            if state.normal_impulse(i) <= 1e-9
                                && fast.impulses[2 * i].hypot(fast.impulses[2 * i + 1]) < 1e-10
                            {
                                Open
                            } else {
                                let n = row.hit.normal;
                                let relative =
                                    state.velocity + state.scaled_spin.cross(-n) - row.hit.velocity;
                                if (relative - n * relative.dot(n)).length() < 3.5e-10 {
                                    Stick
                                } else {
                                    Slide
                                }
                            }
                        });
                        if contact_candidate(problem, inferred, &fast.impulses, mu) {
                            fast.mode = Some(inferred);
                            return Ok(ContactSolution {
                                iterations: fast.iterations,
                                trial: fast,
                                trial_mu: mu,
                                seed_attempts: 0,
                            });
                        }
                    }
                }
            }
            trial.mode = Some(modes);
            let Some(response) = contact_response(problem, modes.map(|m| m != Open)) else {
                continue;
            };
            let slides = modes.iter().filter(|m| **m == Slide).count();
            if slides == 0 {
                self.count_query()?;
                let Some(base) = reduced_contact(&response, modes, mu, &[]) else {
                    continue;
                };
                let gauges = if modes == [Stick, Stick] {
                    sticking_gauges(&response, problem, mu, &base.tangent)
                } else {
                    vec![0.]
                };
                for gamma in gauges {
                    let mut x = base.tangent.clone();
                    if gamma != 0. {
                        let n = [problem.rows[0].hit.normal, problem.rows[1].hit.normal];
                        let dot = n[0].dot(n[1]);
                        for (i, _) in problem.rows.iter().enumerate() {
                            let direction = n[i] * dot - n[1 - i];
                            x[2 * i] += gamma * direction.dot(problem.bases[i].0);
                            x[2 * i + 1] += gamma * direction.dot(problem.bases[i].1);
                        }
                    }
                    self.count_query()?;
                    if contact_candidate(problem, modes, &x, mu) {
                        trial.impulses = x;
                        trial.converged = true;
                        trial.reason = "converged";
                        return Ok(ContactSolution {
                            iterations: trial.iterations,
                            trial,
                            trial_mu: mu,
                            seed_attempts,
                        });
                    }
                }
                continue;
            }
            // A finite dimensionless ladder around the isolated-sphere
            // tangent mobility 3.5; solver seeds do not change physical data.
            let exponents = [0, -1, 1, -2, 2, -3, 3, -4, 4];
            let mut seeds = Vec::new();
            for exponent in exponents {
                seeds.push(vec![(3.5_f64 * 10_f64.powi(exponent)).ln(); slides]);
            }
            if slides == 2 {
                for a in exponents {
                    for b in exponents {
                        if a != b && seeds.len() < MANIFOLD_SWEEPS {
                            seeds.push(vec![
                                (3.5_f64 * 10_f64.powi(a)).ln(),
                                (3.5_f64 * 10_f64.powi(b)).ln(),
                            ]);
                        }
                    }
                }
            }
            for mut logs in seeds {
                seed_attempts += 1;
                for _ in 0..MANIFOLD_SWEEPS {
                    trial.iterations += 1;
                    self.count_query()?;
                    let Some(value) = reduced_contact(&response, modes, mu, &logs) else {
                        trial.reason = "reduced_linear_domain";
                        break;
                    };
                    trial.impulses = value.tangent.clone();
                    self.count_query()?;
                    if contact_candidate(problem, modes, &value.tangent, mu) {
                        trial.converged = true;
                        trial.reason = "converged";
                        return Ok(ContactSolution {
                            iterations: trial.iterations,
                            trial,
                            trial_mu: mu,
                            seed_attempts,
                        });
                    }
                    let score = value.residual.iter().map(|v| v * v).sum::<f64>();
                    let rhs: Vec<_> = value.residual.iter().map(|v| -v).collect();
                    let Some(direction) = small_solve(&value.jacobian, &rhs) else {
                        trial.reason = "reduced_rank";
                        break;
                    };
                    let mut accepted = None;
                    for backtrack in 0..12 {
                        trial.backtrack = backtrack;
                        let fraction = 2_f64.powi(-(backtrack as i32));
                        let candidate: Vec<_> = logs
                            .iter()
                            .zip(&direction)
                            .map(|(x, d)| x + fraction * d)
                            .collect();
                        self.count_query()?;
                        let Some(next) =
                            reduced_contact_value(&response, modes, mu, &candidate, false)
                        else {
                            continue;
                        };
                        if next.residual.iter().map(|v| v * v).sum::<f64>()
                            < score * (1. - 1e-4 * fraction)
                        {
                            accepted = Some(candidate);
                            break;
                        }
                    }
                    let Some(next) = accepted else {
                        trial.reason = "no_reduced_descent";
                        break;
                    };
                    logs = next;
                }
            }
        }
        Ok(ContactSolution {
            iterations: trial.iterations,
            trial,
            trial_mu: mu,
            seed_attempts,
        })
    }
    fn solve_manifold(
        &mut self,
        p: &mut ScreenParticle,
        hits: &[Hit],
        held: &[Hit],
        restitution: bool,
        context: ContactContext,
    ) -> Result<(), String> {
        let r = p.grain.diameter_m as f64 * 0.5;
        let mass = p.grain.material.wet_mass_kg();
        let spin = p.angular_velocity;
        let kinetic_before =
            mass * (0.5 * p.velocity.length_squared() + 0.2 * r * r * spin.length_squared());
        // Identical sphere constraints at a coplanar triangle seam are one
        // physical contact. Merge only f64-roundoff-equivalent normals/drive
        // velocities; the positional contact skin is not a merge tolerance.
        let mut unique: Vec<(Hit, bool)> = Vec::new();
        for hit in hits {
            let was_held = held
                .iter()
                .any(|old| same_feature(*old, *hit) || equivalent_constraint(*old, *hit));
            if let Some((_, held)) = unique
                .iter_mut()
                .find(|(old, _)| equivalent_constraint(*old, *hit))
            {
                *held |= was_held;
            } else {
                unique.push((*hit, was_held));
            }
        }
        // Plastic passivity: ΔK-W = sum J.c_after - .5 ΔqᵀMΔq <= 0,
        // by normal complementarity and friction opposing final slip. Unlike
        // isolated Newton restitution this law has no elastic strain reservoir.
        let simultaneous = unique.len() > 1;
        let rows: Vec<Contact> = unique
            .iter()
            .map(|(hit, was_held)| {
                let incoming =
                    (p.velocity + spin.cross(-hit.normal * r) - hit.velocity).dot(hit.normal);
                let bounce = if simultaneous {
                    match self.config.simultaneous_impact {
                        SimultaneousImpactPolicy::Plastic => 0.,
                    }
                } else if restitution && !was_held {
                    self.config.restitution
                } else {
                    0.
                };
                Contact {
                    hit: *hit,
                    target: -bounce * incoming.min(0.),
                }
            })
            .collect();
        if rows.len() > 12 {
            return Err("Screen coupled contact work dimension exhausted; state retained".into());
        }
        let bases: Vec<_> = rows
            .iter()
            .map(|row| tangent_basis(row.hit.normal))
            .collect();
        let problem = ContactProblem {
            rows: &rows,
            bases: &bases,
            velocity: p.velocity,
            spin: spin * r,
        };
        let configured_mu = self.config.friction_mu;
        let solution = self.contact_solution(&problem, configured_mu)?;
        let trial = solution.trial;
        let trial_mu = solution.trial_mu;
        let outer_iterations = solution.iterations;
        let seed_attempts = solution.seed_attempts;
        let impulses = trial.impulses;
        // Recompute the exact final coordinates with the original contact
        // map before either committing state or writing a bounded refusal.
        self.count_query()?;
        let evaluated = evaluate_contact_impulses(
            problem.rows,
            problem.bases,
            &impulses,
            problem.velocity,
            problem.spin,
            configured_mu,
            trial.normal_impulses.as_deref(),
        )
        .ok_or("Screen final contact support is infeasible; state retained")?;
        if trial.converged
            && trial_mu == configured_mu
            && contact_residual(&evaluated.residual) < 1e-10
            && evaluated.normal_residual < 1e-9
        {
            if impulses.iter().all(|v| *v == 0.)
                && (0..rows.len()).all(|i| evaluated.normal_impulse(i) == 0.)
            {
                // Preserve a true no-op bit for bit, including authoritative
                // spin; even multiplication/division by r is unnecessary.
                return Ok(());
            }
            let mut work = 0.;
            for (i, row) in rows.iter().enumerate() {
                let tangent = bases[i].0 * impulses[2 * i] + bases[i].1 * impulses[2 * i + 1];
                let impulse = (tangent + row.hit.normal * evaluated.normal_impulse(i)) * mass;
                work += impulse.dot(row.hit.velocity);
            }
            let kinetic_after = mass
                * (0.5 * evaluated.velocity.length_squared()
                    + 0.2 * evaluated.scaled_spin.length_squared());
            let excess = kinetic_after - kinetic_before - work;
            if excess > (kinetic_before.abs() + work.abs()) * 1e-10 + 1e-18 {
                return Err(format!("Screen manifold added unexplained kinetic energy; state retained: particle={} rows={} excess={excess:.9e}",p.id,rows.len()));
            }
            p.velocity = evaluated.velocity;
            p.angular_velocity = evaluated.scaled_spin / r;
            p.grain.angular_velocity = p.angular_velocity.as_vec3();
            self.last_step.motor_work_j += work;
            return Ok(());
        }
        let initial_velocity = problem.velocity;
        let stop_reason = if trial.converged && trial_mu != configured_mu {
            "configured_friction_not_reached"
        } else {
            trial.reason
        };
        let last_direction = trial.direction;
        let last_backtrack = trial.backtrack;
        let last_mode = format!("{:?}", trial.mode);
        // At most 12 supports/24 tangent coordinates; the bounded refusal
        // snapshot is sufficient for a tiny replay, never a particle trace.
        let geometry = self.rest_wires();
        let offset = self.offset_at(context.time).0;
        let contact_rows:Vec<_>=rows.iter().enumerate().map(|(i,row)| {
            let (gap,closest,endpoints,wire_radius)=match row.hit.surface {
                Surface::Wire=>{
                    let w=&geometry[row.hit.index];
                    let a=w.a+offset;let b=w.b+offset;
                    let closest=segment_closest(p.position,a,b);
                    (p.position.distance(closest)-r-w.radius,closest,Some([a.to_array(),b.to_array()]),Some(w.radius))
                },
                Surface::Gutter|Surface::Bin=>{
                    let triangles=if row.hit.surface==Surface::Gutter {self.gutter_triangles()} else {self.bin_triangles()};
                    let closest=triangle_closest(p.position,triangles[row.hit.index]);
                    (p.position.distance(closest)-r,closest,None,None)
                }
            };
            serde_json::json!({"surface":format!("{:?}",row.hit.surface),"index":row.hit.index,
                "actual_gap_m":gap,"actual_distance_m":p.position.distance(closest),"closest_world_m":closest.to_array(),"wire_endpoints_world_m":endpoints,"wire_radius_m":wire_radius,
                "normal_world":row.hit.normal.to_array(),"surface_velocity_world_m_s":row.hit.velocity.to_array(),"target_m_s":row.target,
                "normal_impulse_per_mass_m_s":evaluated.normal_impulse(i),"tangent_basis_world":[bases[i].0.to_array(),bases[i].1.to_array()],
                "tangent_impulse_per_mass_m_s":[impulses[2*i],impulses[2*i+1]],"fixed_point_residual_m_s":[evaluated.residual[2*i],evaluated.residual[2*i+1]]})
        }).collect();
        let local_velocity = DVec3::new(
            initial_velocity.dot(self.downstream()),
            initial_velocity.y,
            initial_velocity.dot(self.across()),
        );
        let snapshot = serde_json::json!({"schema":1,"particle_id":p.id,"synthetic_supports":context.synthetic,"radius_m":r,"mass_kg":mass,"kinetic_before_j":kinetic_before,
            "position_world_m":p.position.to_array(),"position_local_m":self.to_local(p.position).to_array(),"input_velocity_world_m_s":initial_velocity.to_array(),"input_velocity_local_m_s":local_velocity.to_array(),"input_spin_world_rad_s":spin.to_array(),
            "drive_time_s":context.time,"interval_s":context.interval,"guard_m":context.guard,"friction_mu":self.config.friction_mu,"restitution":self.config.restitution,"restitution_seeded":restitution,
            "normal_residual_m_s":evaluated.normal_residual,"tangent_residual_m_s":contact_residual(&evaluated.residual),"outer_iterations":outer_iterations,"stop_reason":stop_reason,"last_direction":last_direction,"last_backtrack":last_backtrack,"friction_mode":last_mode,"reduced_seed_attempts":seed_attempts,"last_trial_friction_mu":trial_mu,"contacts":contact_rows});
        Err(format!(
            "Screen contact manifold did not converge; state retained; contact_snapshot={snapshot}"
        ))
    }
}

fn equivalent_constraint(a: Hit, b: Hit) -> bool {
    a.normal.distance_squared(b.normal) <= 1e-20 && a.velocity.distance_squared(b.velocity) <= 1e-24
}
fn rotate_grain(p: &mut ScreenParticle, dt: f64) {
    if p.angular_velocity.length_squared() > 0. {
        p.grain.rotation = (glam::DQuat::from_scaled_axis(p.angular_velocity * dt)
            * p.grain.rotation.as_dquat())
        .normalize()
        .as_quat();
    }
}

fn tangent_basis(n: DVec3) -> (DVec3, DVec3) {
    let axis = if n.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
    let a = n.cross(axis).normalize();
    (a, n.cross(a))
}
fn contact_residual(values: &[f64]) -> f64 {
    values
        .chunks_exact(2)
        .map(|v| v[0].hypot(v[1]))
        .fold(0., f64::max)
}
fn evaluate_contacts(
    rows: &[Contact],
    bases: &[(DVec3, DVec3)],
    impulses: &[f64],
    initial_velocity: DVec3,
    initial_spin: DVec3,
    mu: f64,
) -> Option<ContactEvaluation> {
    evaluate_contact_impulses(
        rows,
        bases,
        impulses,
        initial_velocity,
        initial_spin,
        mu,
        None,
    )
}
fn evaluate_contact_impulses(
    rows: &[Contact],
    bases: &[(DVec3, DVec3)],
    impulses: &[f64],
    initial_velocity: DVec3,
    initial_spin: DVec3,
    mu: f64,
    normals: Option<&[f64]>,
) -> Option<ContactEvaluation> {
    let mut velocity = initial_velocity;
    let mut spin = initial_spin;
    let tangents: Vec<_> = bases
        .iter()
        .enumerate()
        .map(|(i, (a, b))| *a * impulses[2 * i] + *b * impulses[2 * i + 1])
        .collect();
    for (row, tangent) in rows.iter().zip(&tangents) {
        velocity += *tangent;
        spin += (-row.hit.normal).cross(*tangent) * 2.5;
    }
    let constraints: Vec<_> = rows
        .iter()
        .map(|row| {
            (
                row.hit.normal,
                row.target + row.hit.normal.dot(row.hit.velocity - velocity),
            )
        })
        .collect();
    let projection = if let Some(normals) = normals {
        if normals.len() != rows.len() || normals.iter().any(|v| !v.is_finite() || *v < 0.) {
            return None;
        }
        Projection {
            point: rows
                .iter()
                .zip(normals)
                .map(|(row, value)| row.hit.normal * *value)
                .sum(),
            active: [(0, 0.); 3],
            count: 0,
        }
    } else {
        halfspace_projection(&constraints)?
    };
    velocity += projection.point;
    let mut residual = Vec::with_capacity(rows.len() * 2);
    let mut normal_residual: f64 = 0.;
    for (i, row) in rows.iter().enumerate() {
        let n = row.hit.normal;
        let relative = velocity + spin.cross(-n) - row.hit.velocity;
        let normal = relative.dot(n) - row.target;
        let normal_impulse = normals.map_or_else(|| projection.multiplier(i), |values| values[i]);
        normal_residual = normal_residual.max(if normal_impulse > 1e-12 {
            normal.abs()
        } else {
            (-normal).max(0.)
        });
        let slip = relative - n * relative.dot(n);
        let mut projected = tangents[i] - slip / 3.5;
        projected -= n * projected.dot(n);
        let limit = mu * normal_impulse;
        if projected.length() > limit {
            projected = projected.normalize_or_zero() * limit;
        }
        let difference = tangents[i] - projected;
        residual.push(difference.dot(bases[i].0));
        residual.push(difference.dot(bases[i].1));
    }
    if !velocity.is_finite() || !spin.is_finite() || residual.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(ContactEvaluation {
        velocity,
        scaled_spin: spin,
        projection,
        explicit_normal: normals.map(|values| values.to_vec()),
        residual,
        normal_residual,
    })
}

/// Exact semismooth derivative of the SAME projected Coulomb map. The
/// normal active set has at most three independent planes, regardless of
/// stored manifold size. Actual nonlinear trials determine branch changes.
fn projected_contact_jacobian(
    problem: &ContactProblem<'_>,
    state: &ContactEvaluation,
    x: &[f64],
    mu: f64,
) -> Option<Vec<Vec<f64>>> {
    let rows = problem.rows;
    let size = x.len();
    let mut jacobian = vec![vec![0.; size]; size];
    let active = &state.projection.active[..state.projection.count];
    for (column, direction) in problem.bases.iter().flat_map(|(a, b)| [*a, *b]).enumerate() {
        let mut dl = vec![0.; rows.len()];
        match active {
            [] => {}
            [(i, _)] => dl[*i] = -rows[*i].hit.normal.dot(direction),
            [(i, _), (j, _)] => {
                let a = rows[*i].hit.normal;
                let b = rows[*j].hit.normal;
                let dot = a.dot(b);
                let determinant = 1. - dot * dot;
                if determinant <= 1e-14 {
                    return None;
                }
                let da = -a.dot(direction);
                let db = -b.dot(direction);
                dl[*i] = (da - dot * db) / determinant;
                dl[*j] = (db - dot * da) / determinant;
            }
            [(i, _), (j, _), (k, _)] => {
                let a = rows[*i].hit.normal;
                let b = rows[*j].hit.normal;
                let c = rows[*k].hit.normal;
                let determinant = a.dot(b.cross(c));
                if determinant.abs() < 1e-14 {
                    return None;
                }
                let p = (b.cross(c) * -a.dot(direction)
                    + c.cross(a) * -b.dot(direction)
                    + a.cross(b) * -c.dot(direction))
                    / determinant;
                dl[*i] = p.dot(b.cross(c)) / determinant;
                dl[*j] = p.dot(c.cross(a)) / determinant;
                dl[*k] = p.dot(a.cross(b)) / determinant;
            }
            _ => return None,
        }
        let dv = direction
            + rows
                .iter()
                .zip(&dl)
                .map(|(row, lambda)| row.hit.normal * *lambda)
                .sum::<DVec3>();
        let dw = (-rows[column / 2].hit.normal).cross(direction) * 2.5;
        for (i, row) in rows.iter().enumerate() {
            let n = row.hit.normal;
            let tangent = problem.bases[i].0 * x[2 * i] + problem.bases[i].1 * x[2 * i + 1];
            let relative = state.velocity + state.scaled_spin.cross(-n) - row.hit.velocity;
            let slip = relative - n * relative.dot(n);
            let dr = dv + dw.cross(-n);
            let ds = dr - n * dr.dot(n);
            let dt = if i == column / 2 {
                direction
            } else {
                DVec3::ZERO
            };
            let mut y = tangent - slip / 3.5;
            y -= n * y.dot(n);
            let mut dy = dt - ds / 3.5;
            dy -= n * dy.dot(n);
            let length = y.length();
            let limit = mu * state.normal_impulse(i);
            let dp = if length >= limit && length > 0. {
                let unit = y / length;
                (dy - unit * unit.dot(dy)) * (limit / length) + unit * (mu * dl[i])
            } else if limit > 0. {
                dy
            } else {
                DVec3::ZERO
            };
            let residual = dt - dp;
            jacobian[2 * i][column] = residual.dot(problem.bases[i].0);
            jacobian[2 * i + 1][column] = residual.dot(problem.bases[i].1);
        }
    }
    jacobian
        .iter()
        .flatten()
        .all(|v| v.is_finite())
        .then_some(jacobian)
}

/// Affine tangent mobility after eliminating a declared active normal set.
/// For closed contacts S is PSD: T.S.T = |P sum T|^2 + 2.5|sum n cross T|^2.
/// Adding positive slide multipliers regularizes its internal-force nullspace.
struct ContactResponse {
    offset: [f64; 4],
    mobility: [[f64; 4]; 4],
    lambda: [f64; 2],
    lambda_gradient: [[f64; 4]; 2],
}
fn normal_response(rows: &[Contact], closed: [bool; 2], rhs: [f64; 2]) -> Option<[f64; 2]> {
    match closed {
        [false, false] => Some([0.; 2]),
        [true, false] => Some([rhs[0], 0.]),
        [false, true] => Some([0., rhs[1]]),
        [true, true] => {
            let dot = rows[0].hit.normal.dot(rows[1].hit.normal);
            let determinant = 1. - dot * dot;
            (determinant > 1e-14).then(|| {
                [
                    (rhs[0] - dot * rhs[1]) / determinant,
                    (rhs[1] - dot * rhs[0]) / determinant,
                ]
            })
        }
    }
}
fn contact_response(problem: &ContactProblem<'_>, closed: [bool; 2]) -> Option<ContactResponse> {
    let rows = problem.rows;
    let lambda = normal_response(
        rows,
        closed,
        [
            rows[0].target
                + rows[0]
                    .hit
                    .normal
                    .dot(rows[0].hit.velocity - problem.velocity),
            rows[1].target
                + rows[1]
                    .hit
                    .normal
                    .dot(rows[1].hit.velocity - problem.velocity),
        ],
    )?;
    let velocity =
        problem.velocity + rows[0].hit.normal * lambda[0] + rows[1].hit.normal * lambda[1];
    let mut out = ContactResponse {
        offset: [0.; 4],
        mobility: [[0.; 4]; 4],
        lambda,
        lambda_gradient: [[0.; 4]; 2],
    };
    for (i, row) in rows.iter().enumerate() {
        let relative = velocity + problem.spin.cross(-row.hit.normal) - row.hit.velocity;
        out.offset[2 * i] = relative.dot(problem.bases[i].0);
        out.offset[2 * i + 1] = relative.dot(problem.bases[i].1);
    }
    for (column, direction) in problem.bases.iter().flat_map(|(a, b)| [*a, *b]).enumerate() {
        let source = column / 2;
        let dl = normal_response(
            rows,
            closed,
            [
                -rows[0].hit.normal.dot(direction),
                -rows[1].hit.normal.dot(direction),
            ],
        )?;
        for (i, value) in dl.iter().enumerate() {
            out.lambda_gradient[i][column] = *value;
        }
        let dv = direction + rows[0].hit.normal * dl[0] + rows[1].hit.normal * dl[1];
        let ds = (-rows[source].hit.normal).cross(direction) * 2.5;
        for (i, row) in rows.iter().enumerate() {
            let relative = dv + ds.cross(-row.hit.normal);
            out.mobility[2 * i][column] = relative.dot(problem.bases[i].0);
            out.mobility[2 * i + 1][column] = relative.dot(problem.bases[i].1);
        }
    }
    Some(out)
}
struct ReducedContact {
    tangent: Vec<f64>,
    residual: Vec<f64>,
    jacobian: Vec<Vec<f64>>,
}
fn reduced_contact(
    response: &ContactResponse,
    modes: [ContactMode; 2],
    mu: f64,
    logs: &[f64],
) -> Option<ReducedContact> {
    reduced_contact_value(response, modes, mu, logs, true)
}
fn reduced_contact_value(
    response: &ContactResponse,
    modes: [ContactMode; 2],
    mu: f64,
    logs: &[f64],
    derivative: bool,
) -> Option<ReducedContact> {
    let mut matrix = response.mobility;
    let mut rhs = response.offset.map(|v| -v);
    let mut alpha = [0.; 2];
    let mut index = 0;
    for (i, mode) in modes.iter().enumerate() {
        match mode {
            ContactMode::Open => {
                for coordinate in [2 * i, 2 * i + 1] {
                    matrix[coordinate].fill(0.);
                    matrix[coordinate][coordinate] = 1.;
                    rhs[coordinate] = 0.;
                }
            }
            ContactMode::Stick => {}
            ContactMode::Slide => {
                alpha[i] = logs[index].exp();
                index += 1;
                if !alpha[i].is_finite() || alpha[i] == 0. {
                    return None;
                }
                matrix[2 * i][2 * i] += alpha[i];
                matrix[2 * i + 1][2 * i + 1] += alpha[i];
            }
        }
    }
    let tangent = rank_solve_fixed(matrix, rhs)?;
    // A singular sticking face must satisfy every original linear row.
    if matrix.iter().zip(&rhs).any(|(row, b)| {
        (row.iter().zip(&tangent).map(|(a, x)| a * x).sum::<f64>() - b).abs() > 1e-10
    }) {
        return None;
    }
    let lambda: [f64; 2] = std::array::from_fn(|i| {
        response.lambda[i]
            + response.lambda_gradient[i]
                .iter()
                .zip(&tangent)
                .map(|(a, x)| a * x)
                .sum::<f64>()
    });
    let slides: Vec<_> = modes
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (*m == ContactMode::Slide).then_some(i))
        .collect();
    let mut residual = Vec::new();
    for i in &slides {
        if lambda[*i] <= 0. {
            return None;
        }
        residual.push(tangent[2 * i].hypot(tangent[2 * i + 1]) - mu * lambda[*i]);
    }
    let mut jacobian = vec![vec![0.; slides.len()]; slides.len()];
    for (column, source) in slides.iter().enumerate().filter(|_| derivative) {
        let mut rhs = [0.; 4];
        rhs[2 * source] = -alpha[*source] * tangent[2 * source];
        rhs[2 * source + 1] = -alpha[*source] * tangent[2 * source + 1];
        let delta = rank_solve_fixed(matrix, rhs)?;
        for (row, i) in slides.iter().enumerate() {
            let length = tangent[2 * i].hypot(tangent[2 * i + 1]);
            if length == 0. {
                return None;
            }
            let dl = response.lambda_gradient[*i]
                .iter()
                .zip(&delta)
                .map(|(a, x)| a * x)
                .sum::<f64>();
            jacobian[row][column] =
                (tangent[2 * i] * delta[2 * i] + tangent[2 * i + 1] * delta[2 * i + 1]) / length
                    - mu * dl;
        }
    }
    Some(ReducedContact {
        tangent: tangent.to_vec(),
        residual,
        jacobian,
    })
}

/// Intersect a bounded number of exact quadratic/linear cone intervals.
fn sticking_gauges(
    response: &ContactResponse,
    problem: &ContactProblem<'_>,
    mu: f64,
    tangent: &[f64],
) -> Vec<f64> {
    let n = [problem.rows[0].hit.normal, problem.rows[1].hit.normal];
    let dot = n[0].dot(n[1]);
    let c = 1. - dot;
    if c <= 0. {
        return vec![0.];
    }
    let mut intervals = vec![(f64::NEG_INFINITY, f64::INFINITY)];
    for (i, _) in problem.rows.iter().enumerate() {
        let lambda = response.lambda[i]
            + response.lambda_gradient[i]
                .iter()
                .zip(tangent)
                .map(|(a, x)| a * x)
                .sum::<f64>();
        let t = problem.bases[i].0 * tangent[2 * i] + problem.bases[i].1 * tangent[2 * i + 1];
        let direction = n[i] * dot - n[1 - i];
        let a = direction.length_squared() - mu * mu * c * c;
        let b = 2. * (t.dot(direction) - mu * mu * lambda * c);
        let constant = t.length_squared() - mu * mu * lambda * lambda;
        let mut allowed = Vec::new();
        if a == 0. {
            if b > 0. {
                allowed.push((f64::NEG_INFINITY, -constant / b));
            } else if b < 0. {
                allowed.push((-constant / b, f64::INFINITY));
            } else if constant <= 0. {
                allowed.push((f64::NEG_INFINITY, f64::INFINITY));
            }
        } else {
            let discriminant = b * b - 4. * a * constant;
            if discriminant < 0. {
                if a < 0. {
                    allowed.push((f64::NEG_INFINITY, f64::INFINITY));
                }
            } else {
                let q = -0.5 * (b + discriminant.sqrt().copysign(b));
                let mut roots = if q == 0. {
                    [-b / (2. * a); 2]
                } else {
                    [q / a, constant / q]
                };
                if roots[0] > roots[1] {
                    roots.swap(0, 1);
                }
                if a > 0. {
                    allowed.push((roots[0], roots[1]));
                } else {
                    allowed.push((f64::NEG_INFINITY, roots[0]));
                    allowed.push((roots[1], f64::INFINITY));
                }
            }
        }
        let lower = -lambda / c;
        intervals = intervals
            .iter()
            .flat_map(|(lo, hi)| {
                allowed.iter().filter_map(move |(a, b)| {
                    let lo = lo.max(*a).max(lower);
                    let hi = hi.min(*b);
                    (lo <= hi).then_some((lo, hi))
                })
            })
            .collect();
    }
    intervals
        .into_iter()
        .flat_map(|(lo, hi)| {
            let nearest = 0_f64.clamp(lo, hi);
            [nearest, nearest.next_up(), nearest.next_down()]
                .into_iter()
                .filter(move |x| x.is_finite() && *x >= lo && *x <= hi)
        })
        .collect()
}

fn small_solve(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    if rhs.len() == 1 {
        return (matrix[0][0] != 0.)
            .then(|| vec![rhs[0] / matrix[0][0]])
            .filter(|x| x[0].is_finite());
    }
    let scale = matrix.iter().flatten().map(|v| v.abs()).fold(0., f64::max);
    if scale == 0. || !scale.is_finite() {
        return None;
    }
    let a = matrix[0][0] / scale;
    let b = matrix[0][1] / scale;
    let c = matrix[1][0] / scale;
    let d = matrix[1][1] / scale;
    let determinant = a * d - b * c;
    if determinant.abs() <= 1e-14 {
        return None;
    }
    let x = rhs[0] / scale;
    let y = rhs[1] / scale;
    let out = vec![(d * x - b * y) / determinant, (a * y - c * x) / determinant];
    out.iter().all(|v| v.is_finite()).then_some(out)
}
fn contact_candidate(
    problem: &ContactProblem<'_>,
    modes: [ContactMode; 2],
    x: &[f64],
    mu: f64,
) -> bool {
    let Some(state) = problem.evaluate(x, mu) else {
        return false;
    };
    if contact_residual(&state.residual) >= 1e-10 || state.normal_residual >= 1e-9 {
        return false;
    }
    modes.iter().enumerate().all(|(i, mode)| match mode {
        ContactMode::Open => {
            state.normal_impulse(i) <= 1e-9 && x[2 * i].hypot(x[2 * i + 1]) < 1e-10
        }
        ContactMode::Stick => {
            let n = problem.rows[i].hit.normal;
            let relative =
                state.velocity + state.scaled_spin.cross(-n) - problem.rows[i].hit.velocity;
            (relative - n * relative.dot(n)).length() < 3.5e-10
        }
        ContactMode::Slide => true, // positive alpha is represented by exp(log).
    })
}

/// Complete pivoting retains the independent contact modes and sets the
/// internal-force nullspace increment to zero. Each smooth friction mode uses
/// its own trial merit; only the caller's original Coulomb/energy gate admits
/// state. This rank test never changes the contact law.
#[allow(
    clippy::needless_range_loop,
    reason = "Complete row/column pivoting needs explicit matrix indices"
)]
fn rank_solve(mut matrix: Vec<Vec<f64>>, rhs: &[f64]) -> Option<Vec<f64>> {
    let size = rhs.len();
    let mut b = rhs.to_vec();
    let mut order: Vec<_> = (0..size).collect();
    let scale = matrix.iter().flatten().map(|v| v.abs()).fold(0., f64::max);
    let mut rank = 0;
    for k in 0..size {
        let mut pivot = (k, k, 0.);
        for i in k..size {
            for j in k..size {
                if matrix[i][j].abs() > pivot.2 {
                    pivot = (i, j, matrix[i][j].abs());
                }
            }
        }
        if pivot.2 <= scale.max(1.) * 1e-11 {
            break;
        }
        matrix.swap(k, pivot.0);
        b.swap(k, pivot.0);
        for row in &mut matrix {
            row.swap(k, pivot.1);
        }
        order.swap(k, pivot.1);
        for i in k + 1..size {
            let factor = matrix[i][k] / matrix[k][k];
            for j in k..size {
                matrix[i][j] -= factor * matrix[k][j];
            }
            b[i] -= factor * b[k];
        }
        rank += 1;
    }
    let mut permuted = vec![0.; size];
    for i in (0..rank).rev() {
        let remainder = (i + 1..rank)
            .map(|j| matrix[i][j] * permuted[j])
            .sum::<f64>();
        permuted[i] = (b[i] - remainder) / matrix[i][i];
    }
    let mut result = vec![0.; size];
    for i in 0..size {
        result[order[i]] = permuted[i];
    }
    result.iter().all(|v| v.is_finite()).then_some(result)
}

/// Stack storage preserves complete-pivot order and rank thresholds.
#[allow(
    clippy::needless_range_loop,
    reason = "Complete pivoting uses explicit indices"
)]
fn rank_solve_fixed<const N: usize>(mut matrix: [[f64; N]; N], rhs: [f64; N]) -> Option<[f64; N]> {
    let size = N;
    let mut b = rhs;
    let mut order: [usize; N] = std::array::from_fn(|i| i);
    let scale = matrix.iter().flatten().map(|v| v.abs()).fold(0., f64::max);
    let mut rank = 0;
    for k in 0..size {
        let mut pivot = (k, k, 0.);
        for i in k..size {
            for j in k..size {
                if matrix[i][j].abs() > pivot.2 {
                    pivot = (i, j, matrix[i][j].abs());
                }
            }
        }
        if pivot.2 <= scale.max(1.) * 1e-11 {
            break;
        }
        matrix.swap(k, pivot.0);
        b.swap(k, pivot.0);
        for row in &mut matrix {
            row.swap(k, pivot.1);
        }
        order.swap(k, pivot.1);
        for i in k + 1..size {
            let factor = matrix[i][k] / matrix[k][k];
            for j in k..size {
                matrix[i][j] -= factor * matrix[k][j];
            }
            b[i] -= factor * b[k];
        }
        rank += 1;
    }
    let mut permuted = [0.; N];
    for i in (0..rank).rev() {
        let remainder = (i + 1..rank)
            .map(|j| matrix[i][j] * permuted[j])
            .sum::<f64>();
        permuted[i] = (b[i] - remainder) / matrix[i][i];
    }
    let mut result = [0.; N];
    for i in 0..size {
        result[order[i]] = permuted[i];
    }
    result.iter().all(|v| v.is_finite()).then_some(result)
}

fn quad(out: &mut Vec<[DVec3; 3]>, p: [DVec3; 4]) {
    out.push([p[0], p[1], p[2]]);
    out.push([p[0], p[2], p[3]]);
}
fn choose(best: &mut Option<Hit>, hit: Hit) {
    if best.is_none_or(|b| hit.u < b.u) {
        *best = Some(hit);
    }
}
fn segment_closest(p: DVec3, a: DVec3, b: DVec3) -> DVec3 {
    let d = b - a;
    a + d * ((p - a).dot(d) / d.length_squared()).clamp(0., 1.)
}
/// Conservative swept-sphere AABB rejection only; admitted triangles retain
/// the original exact narrowphase, contact ordering, skin and time policy.
fn triangle_bounds_overlap(start: DVec3, end: DVec3, t: [DVec3; 3], radius: f64) -> bool {
    let pad = DVec3::splat(radius);
    let low = start.min(end) - pad;
    let high = start.max(end) + pad;
    !t[0].min(t[1]).min(t[2]).cmpgt(high).any() && !t[0].max(t[1]).max(t[2]).cmplt(low).any()
}
fn triangle_closest(p: DVec3, t: [DVec3; 3]) -> DVec3 {
    let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize();
    let projected = p - n * (p - t[0]).dot(n);
    if inside_triangle(projected, t, n) {
        return projected;
    }
    (0..3)
        .map(|i| segment_closest(p, t[i], t[(i + 1) % 3]))
        .min_by(|a, b| a.distance_squared(p).total_cmp(&b.distance_squared(p)))
        .unwrap()
}

/// Minimum-norm simultaneous clearance in R3. The tangent halfspace of a
/// convex capsule is conservative: satisfying it also clears the real capsule.
/// At most three independent planes are active at a Euclidean projection.
fn project_halfspaces(rows: &[(DVec3, f64)]) -> Option<DVec3> {
    halfspace_projection(rows).map(|projection| projection.point)
}
#[derive(Clone, Copy)]
struct Projection {
    point: DVec3,
    active: [(usize, f64); 3],
    count: usize,
}
impl Projection {
    fn multiplier(&self, index: usize) -> f64 {
        self.active[..self.count]
            .iter()
            .find_map(|(i, v)| (*i == index).then_some(*v))
            .unwrap_or(0.)
    }
}
fn halfspace_projection(rows: &[(DVec3, f64)]) -> Option<Projection> {
    // The original empty active set is already the unique minimum-norm
    // correction when feasible. Enumerating nonzero faces cannot improve it.
    if rows.iter().all(|(n, b)| n.dot(DVec3::ZERO) >= *b - 1e-12) {
        return Some(Projection {
            point: DVec3::ZERO,
            active: [(0, 0.); 3],
            count: 0,
        });
    }
    let mut best: Option<Projection> = None;
    let mut consider = |active: &[(usize, f64)]| {
        let p = active
            .iter()
            .map(|(i, v)| rows[*i].0 * v.max(0.))
            .sum::<DVec3>();
        if p.is_finite()
            && rows.iter().all(|(n, b)| n.dot(p) >= *b - 1e-12)
            && best.is_none_or(|old| p.length_squared() < old.point.length_squared())
        {
            let mut weights = [(0, 0.); 3];
            for (dst, (index, value)) in weights.iter_mut().zip(active) {
                *dst = (*index, value.max(0.));
            }
            best = Some(Projection {
                point: p,
                active: weights,
                count: active.len(),
            });
        }
    };
    consider(&[]);
    for (i, (a, ba)) in rows.iter().enumerate() {
        if *ba >= 0. {
            consider(&[(i, *ba)]);
        }
        for (j, (b, bb)) in rows.iter().enumerate().skip(i + 1) {
            let dot = a.dot(*b);
            let determinant = 1. - dot * dot;
            if determinant > 1e-14 {
                let x = (*ba - dot * bb) / determinant;
                let y = (*bb - dot * ba) / determinant;
                if x >= -1e-12 && y >= -1e-12 {
                    consider(&[(i, x), (j, y)]);
                }
            }
            for (k, (c, bc)) in rows.iter().enumerate().skip(j + 1) {
                let determinant = a.dot(b.cross(*c));
                if determinant.abs() < 1e-14 {
                    continue;
                }
                let p = (b.cross(*c) * *ba + c.cross(*a) * *bb + a.cross(*b) * *bc) / determinant;
                let x = p.dot(b.cross(*c)) / determinant;
                let y = p.dot(c.cross(*a)) / determinant;
                let z = p.dot(a.cross(*b)) / determinant;
                if x >= -1e-12 && y >= -1e-12 && z >= -1e-12 {
                    consider(&[(i, x), (j, y), (k, z)]);
                }
            }
        }
    }
    best
}
fn roots(a: f64, b: f64, c: f64) -> [f64; 2] {
    if a <= 1e-30 {
        return [f64::INFINITY; 2];
    }
    let discriminant = b * b - 4. * a * c;
    if discriminant < 0. {
        return [f64::INFINITY; 2];
    }
    let q = -0.5 * (b + discriminant.sqrt().copysign(b));
    let mut out = [q / a, if q != 0. { c / q } else { -b / (2. * a) }];
    if out[1] < out[0] {
        out.swap(0, 1);
    }
    out
}
fn cast_capsule(p: DVec3, d: DVec3, a: DVec3, b: DVec3, r: f64) -> Option<f64> {
    let nearest = segment_closest(p, a, b);
    let delta = p - nearest;
    if delta.length_squared() <= r * r * (1. + 1e-12) {
        return (delta.dot(d) < -1e-18).then_some(0.);
    }
    let axis = b - a;
    let l2 = axis.length_squared();
    let q = p - a;
    let perpendicular = q - axis * (q.dot(axis) / l2);
    let speed = d - axis * (d.dot(axis) / l2);
    let mut hit = f64::INFINITY;
    for u in roots(
        speed.length_squared(),
        2. * perpendicular.dot(speed),
        perpendicular.length_squared() - r * r,
    ) {
        let along = (q + d * u).dot(axis) / l2;
        if (0.0..=1.).contains(&u) && (0.0..=1.).contains(&along) {
            hit = hit.min(u);
        }
    }
    for end in [a, b] {
        let q = p - end;
        for u in roots(
            d.length_squared(),
            2. * q.dot(d),
            q.length_squared() - r * r,
        ) {
            if (0.0..=1.).contains(&u) {
                hit = hit.min(u);
            }
        }
    }
    hit.is_finite().then_some(hit)
}
fn inside_triangle(p: DVec3, t: [DVec3; 3], n: DVec3) -> bool {
    (0..3).all(|i| (t[(i + 1) % 3] - t[i]).cross(p - t[i]).dot(n) >= -1e-12)
}
fn cast_triangle(p: DVec3, d: DVec3, t: [DVec3; 3], r: f64) -> Option<(f64, DVec3)> {
    let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize();
    let mut best: Option<(f64, DVec3)> = None;
    for normal in [n, -n] {
        let distance = (p - t[0]).dot(normal);
        let speed = d.dot(normal);
        if speed >= -1e-15 {
            continue;
        }
        let u = if distance <= r && distance >= -1e-10 {
            0.
        } else {
            (r - distance) / speed
        };
        if (0.0..=1.).contains(&u)
            && inside_triangle(p + d * u - normal * r, t, n)
            && best.is_none_or(|b| u < b.0)
        {
            best = Some((u, normal));
        }
    }
    for i in 0..3 {
        if let Some(u) = cast_capsule(p, d, t[i], t[(i + 1) % 3], r) {
            let at = p + d * u;
            let closest = segment_closest(at, t[i], t[(i + 1) % 3]);
            if let Some(normal) = (at - closest).try_normalize() {
                if best.is_none_or(|b| u < b.0) {
                    best = Some((u, normal));
                }
            }
        }
    }
    best
}
fn valid_grain(g: &Grain) -> bool {
    g.position.is_finite()
        && g.position.length() < 10000.
        && g.velocity.is_finite()
        && g.velocity.length() < 200.
        && g.affine.is_finite()
        && g.rotation.is_finite()
        && (g.rotation.length_squared() - 1.).abs() < 1e-3
        && g.angular_velocity.is_finite()
        && g.angular_velocity.length() < 1e7
        && g.material.valid()
        && g.material.wet_mass_kg().is_finite()
        && g.material.wet_mass_kg() > 0.
        && (0.00002..=0.02).contains(&g.diameter_m)
}
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= (a.abs().max(b.abs()) * 1e-10).max(1e-14)
}
fn same_material(a: Material, b: Material) -> bool {
    [a.soil_m3, a.sand_m3, a.gravel_m3, a.gold_g, a.pore_water_m3]
        .into_iter()
        .zip([b.soil_m3, b.sand_m3, b.gravel_m3, b.gold_g, b.pore_water_m3])
        .all(|(a, b)| close(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn captured_grain() -> Grain {
        Grain {
            volume_resolved: true,
            kind: crate::GrainKind::Gravel,
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            material: Material {
                gravel_m3: 2e-6,
                ..Default::default()
            },
            resting: false,
            diameter_m: 0.0042732735,
            affine: glam::Mat3::ZERO,
            rotation: Quat::IDENTITY,
            angular_velocity: Vec3::new(311.76086, -0.061976466, -0.013173505),
        }
    }
    #[test]
    fn shared_wire_brackets_match_uncached_phase_hits_and_query_work() {
        let mut hits = 0;
        let mut misses = 0;
        for case in 0..2048 {
            let mut screen = Screen::new(
                Vec3::new(14.55, 1.8, 17.2),
                [0., 0.37, -1.2, 3.1][(case / 5) % 4],
            );
            screen.motor_on = (case / 3) % 2 == 0;
            screen.time =
                [0., 0.00001, 0.123, 2.0333334393799305, 13.766667384654284][(case / 11) % 5];
            screen.config.slope_deg = [0., 12., 25.][(case / 17) % 3];
            screen.frame = Some(ScreenFrame::new(&screen));
            let wires = screen.rest_wires();
            let index = (case * 37) % wires.len();
            let wire = wires[index];
            let t0 = screen.time + (case % 17) as f64 * 1e-5;
            let t1 = t0 + [1. / 960., 0.0002, 0.0005][(case / 7) % 3];
            let radius = [4e-5, 0.00015, 0.0015, 0.003][(case / 7) % 4] + wire.radius;
            let n = (screen.normal() + screen.across() * (((case / 13) % 11) as f64 - 5.) * 0.25)
                .normalize();
            let gap = [-1e-6, -1e-10, 0., 1e-10, 5e-7, 1e-4][(case / 2) % 6];
            let point = wire.a.lerp(wire.b, (case % 7) as f64 / 6.);
            let start = point + screen.offset_at(t0).0 + n * (radius + gap);
            let velocity = screen.offset_at(t0).1
                + n * if case % 2 == 0 { -0.35 } else { 0.35 }
                + screen.downstream() * 0.1;
            let omega = std::f64::consts::TAU * screen.config.frequency_hz;
            let sweep = WireSweep {
                start,
                delta: velocity * (t1 - t0),
                velocity,
                time0: t0,
                time1: t1,
                wire,
                radius,
                index,
                acceleration: if screen.motor_on {
                    omega
                        * omega
                        * (screen.config.amplitude_downstream_m + screen.config.amplitude_normal_m)
                } else {
                    0.
                },
            };
            let phase = |u: f64| t0 + (t1 - t0) * u;
            let bracket = WireBracket {
                lo: 0.,
                hi: 1.,
                lower: screen.offset_at(phase(0.)),
                upper: screen.offset_at(phase(1.)),
            };
            screen.last_step = ScreenStep::default();
            let cached = screen.refine_wire(&sweep, bracket, 0);
            let cached_queries = screen.last_step.contact_queries;
            screen.last_step = ScreenStep::default();
            let reference = screen.refine_wire_reference(&sweep, 0., 1., 0);
            if matches!(reference, Ok(Some(_))) {
                hits += 1;
            }
            if matches!(reference, Ok(None)) {
                misses += 1;
            }
            let signature = |hit: Hit| {
                (
                    hit.u.to_bits(),
                    hit.normal.to_array().map(f64::to_bits),
                    hit.velocity.to_array().map(f64::to_bits),
                    hit.surface,
                    hit.index,
                )
            };
            assert_eq!(
                cached.map(|hit| hit.map(signature)),
                reference.map(|hit| hit.map(signature)),
                "case {case}"
            );
            assert_eq!(
                cached_queries, screen.last_step.contact_queries,
                "case {case}"
            );
        }
        assert!(
            hits > 100 && misses > 100,
            "hit/miss coverage {hits}/{misses}"
        );
    }
    #[test]
    fn transaction_frame_keeps_exact_drive_geometry_and_is_not_saved() {
        for yaw in [0., 0.37, -1.2, 3.1] {
            for slope in [0., 12., 25.] {
                let mut screen = Screen::new(Vec3::new(14., 2., 17.), yaw);
                screen.config.slope_deg = slope;
                screen.time = 0.321;
                for motor in [false, true] {
                    screen.motor_on = motor;
                    let frame = ScreenFrame::new(&screen);
                    let plain = serde_json::to_value(&screen).unwrap();
                    for time in [0., 0.01, 0.123, 1.2, 17.8] {
                        let drive = screen.offset_at(time);
                        let local = screen.to_local(DVec3::new(14.3, 1.8, 17.4));
                        let distance = screen.deck_distance(local);
                        screen.frame = Some(frame);
                        assert_eq!(drive, screen.offset_at(time));
                        assert_eq!(local, screen.to_local(DVec3::new(14.3, 1.8, 17.4)));
                        assert_eq!(distance, screen.deck_distance(local));
                        assert_eq!(plain, serde_json::to_value(&screen).unwrap());
                        screen.frame = None;
                    }
                }
            }
        }
    }
    #[test]
    fn triangle_broadphase_keeps_face_edge_corner_and_initial_overlap_hits() {
        let t = [
            DVec3::new(-1., 0., -1.),
            DVec3::new(1., 0., -1.),
            DVec3::new(0., 0., 1.),
        ];
        let mut hits = 0;
        for radius in [0.0001, 0.01, 0.2] {
            for x in -20..=20 {
                for z in -20..=20 {
                    let start = DVec3::new(x as f64 / 16., 0.3, z as f64 / 16.);
                    for delta in [DVec3::new(0., -0.8, 0.), DVec3::new(0.2, -0.3, -0.1)] {
                        if cast_triangle(start, delta, t, radius).is_some() {
                            hits += 1;
                            assert!(triangle_bounds_overlap(
                                start,
                                start + delta,
                                t,
                                radius + 1e-12
                            ));
                        }
                    }
                    let closest = triangle_closest(start - DVec3::Y * 0.3, t);
                    assert!(triangle_bounds_overlap(closest, closest, t, radius + 1e-12));
                }
            }
        }
        assert!(hits > 1000);
    }
    #[test]
    fn stack_contact_elimination_preserves_complete_pivoting() {
        for seed in 0..512 {
            let mut state = seed as u64 + 1;
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 32) % 2001) as f64 / 1000. - 1.
            };
            let mut matrix: [[f64; 4]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| next()));
            if seed % 3 == 0 {
                matrix[3] = matrix[1];
            }
            if seed % 7 == 0 {
                matrix[2] = [0.; 4];
            }
            let rhs = std::array::from_fn(|_| next());
            let dynamic = rank_solve(matrix.map(|row| row.to_vec()).to_vec(), &rhs);
            let fixed = rank_solve_fixed(matrix, rhs).map(|x| x.to_vec());
            assert_eq!(dynamic, fixed);
        }
    }
    #[test]
    fn residual_only_line_search_preserves_the_reduced_contact_map() {
        let normals = [DVec3::new(0.8, 0.6, 0.), DVec3::new(-0.8, 0.6, 0.)];
        let rows: Vec<_> = normals
            .into_iter()
            .enumerate()
            .map(|(index, normal)| Contact {
                hit: Hit {
                    u: 0.,
                    normal,
                    velocity: DVec3::ZERO,
                    surface: Surface::Wire,
                    index,
                },
                target: 0.,
            })
            .collect();
        let bases: Vec<_> = normals.into_iter().map(tangent_basis).collect();
        let problem = ContactProblem {
            rows: &rows,
            bases: &bases,
            velocity: DVec3::new(0.2, -0.5, 0.3),
            spin: DVec3::new(0.1, 0.2, 0.3),
        };
        for modes in [
            [ContactMode::Slide, ContactMode::Stick],
            [ContactMode::Stick, ContactMode::Slide],
            [ContactMode::Slide, ContactMode::Slide],
        ] {
            let response = contact_response(&problem, [true, true]).unwrap();
            for exponent in -6..=6 {
                let logs = vec![
                    (3.5_f64 * 10_f64.powi(exponent)).ln();
                    modes.iter().filter(|m| **m == ContactMode::Slide).count()
                ];
                let full = reduced_contact(&response, modes, 0.4, &logs);
                let residual = reduced_contact_value(&response, modes, 0.4, &logs, false);
                if let Some(full) = full {
                    let residual = residual.unwrap();
                    assert_eq!(full.tangent, residual.tangent);
                    assert_eq!(full.residual, residual.residual);
                }
            }
        }
    }
    #[test]
    fn captured_open_neighbor_is_not_a_restitution_support() {
        let time = 0.256250013038516;
        let mut screen = Screen::new(Vec3::ZERO, 0.);
        screen.motor_on = true;
        screen.time = time;
        screen.elapsed_time_s = time;
        let endpoint = DVec3::new(14.550000381469726, 1.8011914434131429, 17.251161216663434);
        screen.origin = (endpoint
            - screen.offset_at(time).0
            - screen.normal() * 0.001
            - screen.across() * -0.05)
            .as_vec3();
        let position = DVec3::new(14.552501436796987, 1.7699230942910296, 17.40738515821234);
        let mut grain = captured_grain();
        grain.position = position.as_vec3();
        let particle = ScreenParticle {
            id: 1,
            angular_velocity: grain.angular_velocity.as_dvec3(),
            grain,
            position,
            velocity: DVec3::new(0., -0.021825851429282013, 0.5422811811225781),
            in_gutter: false,
            passed_deck: false,
        };
        let wires = screen.rest_wires();
        let offset = screen.offset_at(time).0;
        let r = particle.grain.diameter_m as f64 * 0.5;
        let gap = position.distance(segment_closest(
            position,
            wires[28].a + offset,
            wires[28].b + offset,
        )) - r
            - wires[28].radius;
        assert!((gap - 1.6819083509919848e-6).abs() < 2e-12);
        let contacts = screen
            .manifold(
                &particle,
                time,
                &wires,
                &screen.gutter_triangles(),
                &screen.bin_triangles(),
                CONTACT_SKIN_M + 1e-12,
                None,
            )
            .unwrap();
        assert!(!contacts
            .iter()
            .any(|c| c.surface == Surface::Wire && c.index == 28));
        assert!(contacts
            .iter()
            .any(|c| c.surface == Surface::Wire && c.index == 29));
    }
    #[test]
    fn isolated_restitution_and_coplanar_seams_preserve_the_sphere_law() {
        for (duplicate, was_held) in [(false, false), (true, false), (true, true)] {
            let mut screen = Screen::new(Vec3::ZERO, 0.);
            let boundary = DVec3::new(0.2, 0.1, 0.3);
            let normal = DVec3::Y;
            let spin = DVec3::new(0.123456789012345, -0.234567890123456, 0.345678901234567);
            let mut grain = captured_grain();
            grain.angular_velocity = spin.as_vec3();
            let mut particle = ScreenParticle {
                id: 1,
                grain,
                position: DVec3::ZERO,
                velocity: boundary + DVec3::new(0.3, -0.6, -0.2),
                angular_velocity: spin,
                in_gutter: false,
                passed_deck: false,
            };
            let r = particle.grain.diameter_m as f64 * 0.5;
            let incoming = particle.velocity + spin.cross(-normal * r) - boundary;
            let coefficient = if was_held {
                0.
            } else {
                screen.config.restitution
            };
            let jn = -(1. + coefficient) * incoming.dot(normal);
            let slip = incoming - normal * incoming.dot(normal);
            let tangent =
                -slip.normalize() * (slip.length() / 3.5).min(screen.config.friction_mu * jn);
            let expected_velocity = particle.velocity + normal * jn + tangent;
            let expected_spin = spin + (-normal).cross(tangent) * (2.5 / r);
            let hit = Hit {
                u: 0.,
                normal,
                velocity: boundary,
                surface: Surface::Gutter,
                index: 0,
            };
            let second = Hit { index: 1, ..hit };
            let hits = if duplicate {
                vec![hit, second]
            } else {
                vec![hit]
            };
            let held = if was_held { vec![second] } else { Vec::new() };
            screen
                .solve_manifold(
                    &mut particle,
                    &hits,
                    &held,
                    true,
                    ContactContext {
                        synthetic: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert!(particle.velocity.distance(expected_velocity) < 1e-12);
            assert!(particle.angular_velocity.distance(expected_spin) < 1e-10);
            assert_eq!(
                particle.grain.angular_velocity,
                particle.angular_velocity.as_vec3()
            );
            assert!(((particle.velocity - boundary).dot(normal) - coefficient * 0.6).abs() < 1e-12);
        }
    }

    #[test]
    fn simultaneous_plastic_impact_captures_without_an_elastic_reservoir() {
        let mut screen = Screen::new(Vec3::ZERO, 0.);
        let grain = captured_grain();
        let r = grain.diameter_m as f64 * 0.5;
        let side = 0.0025 / (r + 0.001);
        let up = screen.normal() * (1. - side * side).sqrt();
        let velocity = DVec3::new(0., 0.0669642650724169, 0.14859424506977062);
        let normals = [up + DVec3::X * side, up - DVec3::X * side];
        let mut particle = ScreenParticle {
            id: 1,
            angular_velocity: grain.angular_velocity.as_dvec3(),
            grain,
            position: DVec3::ZERO,
            velocity: DVec3::new(0., -0.021825851429282013, 0.5422811811225781),
            in_gutter: false,
            passed_deck: false,
        };
        let rows = [
            Hit {
                u: 0.,
                normal: normals[0],
                velocity,
                surface: Surface::Wire,
                index: 0,
            },
            Hit {
                u: 0.,
                normal: normals[1],
                velocity,
                surface: Surface::Wire,
                index: 1,
            },
        ];
        let targets = [0.; 2];
        screen
            .solve_manifold(
                &mut particle,
                &rows,
                &[],
                true,
                ContactContext {
                    synthetic: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let slips = rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let point =
                    particle.velocity + particle.angular_velocity.cross(-row.normal * r) - velocity;
                assert!(point.dot(row.normal) - targets[i] >= -1e-9);
                (point - row.normal * point.dot(row.normal)).length()
            })
            .sum::<f64>();
        assert!(
            slips < 1e-8,
            "the declared plastic wedge law captures this state"
        );
    }

    #[test]
    fn captured_later_two_touch_uses_the_exact_near_stick_derivative() {
        let mut screen = Screen::new(Vec3::ZERO, 0.);
        let mut grain = captured_grain();
        grain.angular_velocity =
            DVec3::new(356.92474365234375, -0.3818053603172302, 0.2915396988391876).as_vec3();
        let velocity = DVec3::new(0., 0.14586712736058585, 0.053388909648294375);
        let normals = [
            DVec3::new(0.7970320419061024, 0.5907395471466922, 0.12556556698358845),
            DVec3::new(-0.7970320419063089, 0.5907395471464255, 0.12556556698353177),
        ];
        let mut particle = ScreenParticle {
            id: 1,
            angular_velocity: grain.angular_velocity.as_dvec3(),
            grain,
            position: DVec3::new(14.552500381469725, 1.770225619516059, 17.414168166205283),
            velocity: DVec3::new(
                5.5012418231914495e-15,
                0.0384494372821886,
                0.5162843060910425,
            ),
            in_gutter: false,
            passed_deck: false,
        };
        let r = particle.grain.diameter_m as f64 * 0.5;
        let hits = normals
            .into_iter()
            .enumerate()
            .map(|(index, normal)| Hit {
                u: 0.,
                normal,
                velocity,
                surface: Surface::Wire,
                index,
            })
            .collect::<Vec<_>>();
        let targets = [0.; 2];
        let rows: Vec<_> = hits
            .iter()
            .enumerate()
            .map(|(i, hit)| Contact {
                hit: *hit,
                target: targets[i],
            })
            .collect();
        let bases: Vec<_> = normals.into_iter().map(tangent_basis).collect();
        // Independent differences of the reduced cone equations check the
        // exact implicit linear response, including the spin contribution.
        let spin = particle.angular_velocity * r;
        let problem = ContactProblem {
            rows: &rows,
            bases: &bases,
            velocity: particle.velocity,
            spin,
        };
        let response = contact_response(&problem, [true, true]).unwrap();
        let modes = [ContactMode::Slide, ContactMode::Slide];
        let logs = vec![3.5_f64.ln(), 3.5_f64.ln()];
        let value = reduced_contact(&response, modes, screen.config.friction_mu, &logs).unwrap();
        let direction = [0.31, -0.27];
        let epsilon = 1e-6;
        let plus: Vec<_> = logs
            .iter()
            .zip(direction)
            .map(|(x, d)| x + epsilon * d)
            .collect();
        let minus: Vec<_> = logs
            .iter()
            .zip(direction)
            .map(|(x, d)| x - epsilon * d)
            .collect();
        let a = reduced_contact(&response, modes, screen.config.friction_mu, &plus).unwrap();
        let b = reduced_contact(&response, modes, screen.config.friction_mu, &minus).unwrap();
        for (i, row) in value.jacobian.iter().enumerate() {
            let analytic: f64 = row.iter().zip(direction).map(|(a, d)| a * d).sum();
            let numerical = (a.residual[i] - b.residual[i]) / (2. * epsilon);
            assert!((analytic - numerical).abs() < 1e-7 * analytic.abs().max(1.));
        }
        screen
            .solve_manifold(
                &mut particle,
                &hits,
                &[],
                true,
                ContactContext {
                    synthetic: true,
                    ..Default::default()
                },
            )
            .unwrap();
        for (i, n) in normals.iter().enumerate() {
            let relative = particle.velocity + particle.angular_velocity.cross(-*n * r) - velocity;
            assert!(relative.dot(*n) - targets[i] >= -1e-9);
        }
    }

    #[test]
    fn opening_neighbor_has_no_friction_or_normal_impulse() {
        let mut screen = Screen::new(Vec3::ZERO, 0.);
        let normals = [DVec3::Y, DVec3::X];
        let hits: Vec<_> = normals
            .into_iter()
            .enumerate()
            .map(|(index, normal)| Hit {
                u: 0.,
                normal,
                velocity: DVec3::ZERO,
                surface: Surface::Wire,
                index,
            })
            .collect();
        let rows: Vec<_> = hits
            .iter()
            .enumerate()
            .map(|(i, hit)| Contact {
                hit: *hit,
                target: if i == 0 { 0.02 } else { 0. },
            })
            .collect();
        let bases: Vec<_> = normals.into_iter().map(tangent_basis).collect();
        let problem = ContactProblem {
            rows: &rows,
            bases: &bases,
            velocity: DVec3::new(0.3, -0.2, 0.1),
            spin: DVec3::ZERO,
        };
        let solution = screen
            .contact_solution(&problem, screen.config.friction_mu)
            .unwrap();
        assert!(solution.trial.converged);
        assert_eq!(solution.trial_mu, screen.config.friction_mu);
        let state = problem
            .evaluate(&solution.trial.impulses, screen.config.friction_mu)
            .unwrap();
        assert!(state.normal_impulse(1) < 1e-12);
        assert!(
            solution.trial.impulses[2].abs() < 1e-10 && solution.trial.impulses[3].abs() < 1e-10
        );
        assert_eq!(solution.trial.mode.unwrap()[1], ContactMode::Open);
        assert!(state.velocity.x > 0.2 && state.velocity.y > 0.019999999);
    }

    #[test]
    fn reduced_solver_crosses_stick_slide_states_in_a_narrow_throat() {
        let fixtures = [
            (
                0.0017662663012742996,
                DVec3::new(375.13336181640625, 26.26544189453125, 5.407153129577637),
                DVec3::new(
                    -2.032879073410321e-18,
                    -0.3043187938719343,
                    0.33230402657765296,
                ),
                DVec3::new(0., -0.1460892534458535, -0.010237388154901028),
                [
                    DVec3::new(0.9037452320282449, 0.41871636401001117, 0.08900091064955255),
                    DVec3::new(-0.903745234552746, 0.41871635879673835, 0.0890009095414372),
                ],
            ),
            (
                0.0017662663012742996,
                DVec3::new(442.1387939453125, 0.09959723055362701, -0.2304864525794983),
                DVec3::new(0., 0.06697790866183728, 0.38070455844367757),
                DVec3::new(0., 0.1471704103944764, 0.04137525405091625),
                [
                    DVec3::new(0.903745239253102, 0.4187163490904185, 0.089000907477082),
                    DVec3::new(-0.903745239253102, 0.4187163490904185, 0.089000907477082),
                ],
            ),
            (
                0.0015535536222159863,
                DVec3::new(378.2216796875, -36.54425048828125, -13.576312065124512),
                DVec3::new(
                    0.012661016617138143,
                    0.01709558517231846,
                    0.3918149513659571,
                ),
                DVec3::new(0., 0.12738426545093895, 0.10310062145114711),
                [
                    DVec3::new(0.9790277995811089, 0.19927478126088127, 0.04235716233176943),
                    DVec3::new(
                        -0.9790277990098132,
                        0.19927478394630332,
                        0.04235716290257351,
                    ),
                ],
            ),
        ];
        for (radius, input_spin, input_velocity, boundary_velocity, input_normals) in fixtures {
            for yaw in [0., 0.61, 1.8] {
                let rotation = glam::DQuat::from_rotation_y(yaw);
                let mut screen = Screen::new(Vec3::ZERO, 0.);
                let mut grain = captured_grain();
                grain.diameter_m = (2. * radius) as f32;
                grain.angular_velocity = (rotation * input_spin).as_vec3();
                let mut particle = ScreenParticle {
                    id: 7,
                    angular_velocity: grain.angular_velocity.as_dvec3(),
                    grain,
                    position: DVec3::ZERO,
                    velocity: rotation * input_velocity,
                    in_gutter: false,
                    passed_deck: false,
                };
                let surface_velocity = rotation * boundary_velocity;
                let normals = input_normals.map(|n| rotation * n);
                let hits: Vec<_> = normals
                    .into_iter()
                    .enumerate()
                    .map(|(index, normal)| Hit {
                        u: 0.,
                        normal,
                        velocity: surface_velocity,
                        surface: Surface::Wire,
                        index,
                    })
                    .collect();
                let config_before = screen.config.clone();
                let targets = [0.; 2];
                screen
                    .solve_manifold(
                        &mut particle,
                        &hits,
                        &[],
                        true,
                        ContactContext {
                            synthetic: true,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(screen.config, config_before);
                let r = particle.grain.diameter_m as f64 * 0.5;
                for (i, n) in normals.iter().enumerate() {
                    let relative = particle.velocity + particle.angular_velocity.cross(-*n * r)
                        - surface_velocity;
                    assert!(relative.dot(*n) - targets[i] >= -1e-9);
                }
                assert!(screen.last_step.contact_queries < MAX_QUERIES);
            }
        }
    }

    #[test]
    fn captured_three_wire_contact_uses_exact_projected_mobility() {
        let mut screen = Screen::new(Vec3::ZERO, 0.);
        let boundary = DVec3::new(0., 0.05904362987980223, 0.15073807362156882);
        let normals = [
            DVec3::new(
                0.9986033370792957,
                0.051678928418395075,
                0.010984695335829887,
            ),
            DVec3::new(
                -0.9986033370792957,
                0.051678928418395075,
                0.010984695335829887,
            ),
            DVec3::new(0., 0.9420490417989443, -0.33547518961272055),
        ];
        let spin = DVec3::new(
            -2.694678430217397,
            -5.76893759543079e-16,
            -9.653956507353713e-15,
        );
        let mut grain = captured_grain();
        grain.diameter_m = (2. * 0.0015035034157335758) as f32;
        grain.angular_velocity = spin.as_vec3();
        let mut particle = ScreenParticle {
            id: 39,
            grain,
            position: DVec3::ZERO,
            velocity: DVec3::new(
                -2.642742795433417e-19,
                0.05738904828945132,
                0.15091318689034613,
            ),
            angular_velocity: spin,
            in_gutter: false,
            passed_deck: false,
        };
        let r = particle.grain.diameter_m as f64 * 0.5;
        let hits: Vec<_> = normals
            .into_iter()
            .enumerate()
            .map(|(index, normal)| Hit {
                u: 0.,
                normal,
                velocity: boundary,
                surface: Surface::Wire,
                index,
            })
            .collect();
        let rows: Vec<_> = hits
            .iter()
            .map(|hit| Contact {
                hit: *hit,
                target: 0.,
            })
            .collect();
        let bases: Vec<_> = normals.into_iter().map(tangent_basis).collect();
        let problem = ContactProblem {
            rows: &rows,
            bases: &bases,
            velocity: particle.velocity,
            spin: spin * r,
        };
        let x = [
            0.0004048553135014805,
            -0.00020032176381361787,
            -0.00040485531350056874,
            -0.0002003217638136587,
            0.0005662933327568566,
            1.7308707697601973e-16,
        ];
        let state = problem.evaluate(&x, screen.config.friction_mu).unwrap();
        let jacobian =
            projected_contact_jacobian(&problem, &state, &x, screen.config.friction_mu).unwrap();
        let direction = [0.31, -0.27, 0.19, 0.37, -0.13, 0.17];
        let epsilon = 1e-10;
        let plus: Vec<_> = x
            .iter()
            .zip(direction)
            .map(|(x, d)| x + epsilon * d)
            .collect();
        let minus: Vec<_> = x
            .iter()
            .zip(direction)
            .map(|(x, d)| x - epsilon * d)
            .collect();
        let a = problem.evaluate(&plus, screen.config.friction_mu).unwrap();
        let b = problem.evaluate(&minus, screen.config.friction_mu).unwrap();
        for (i, row) in jacobian.iter().enumerate() {
            let analytic: f64 = row.iter().zip(direction).map(|(a, d)| a * d).sum();
            let numerical = (a.residual[i] - b.residual[i]) / (2. * epsilon);
            assert!((analytic - numerical).abs() < 2e-6 * analytic.abs().max(1.));
        }
        let relative_energy = |p: &ScreenParticle| {
            0.5 * (p.velocity - boundary).length_squared()
                + 0.2 * r * r * p.angular_velocity.length_squared()
        };
        let before = relative_energy(&particle);
        screen
            .solve_manifold(
                &mut particle,
                &hits,
                &[],
                true,
                ContactContext {
                    synthetic: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(relative_energy(&particle) <= before + 1e-16);
        for n in normals {
            assert!((particle.velocity - boundary).dot(n) >= -1e-9);
        }
    }

    #[test]
    fn common_surface_static_capture_and_separating_contacts_are_frame_invariant() {
        let rotations = [
            glam::DQuat::IDENTITY,
            glam::DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.3, -0.8, 0.5),
        ];
        for rotation in rotations {
            for separating in [false, true] {
                for extra in [false, true] {
                    let mut screen = Screen::new(Vec3::ZERO, 0.);
                    let boost = DVec3::new(0.7, -0.4, 0.2);
                    let boundary =
                        rotation * DVec3::new(0., 0.04721547266646776, 0.15289794513504812) + boost;
                    let spin = rotation
                        * DVec3::new(
                            -4.829075055570884,
                            5.1920438358877114e-15,
                            -7.0252589663107384e-15,
                        );
                    let mut grain = captured_grain();
                    grain.diameter_m = (2. * 0.0015035034157335758) as f32;
                    grain.angular_velocity = spin.as_vec3();
                    let velocity = if separating {
                        boundary + rotation * DVec3::Y * 0.1
                    } else {
                        rotation
                            * DVec3::new(
                                -4.2690460541616737e-19,
                                0.045999263914443485,
                                0.15268774435165888,
                            )
                            + boost
                    };
                    let mut particle = ScreenParticle {
                        id: 39,
                        grain,
                        position: DVec3::ZERO,
                        velocity,
                        angular_velocity: spin,
                        in_gutter: false,
                        passed_deck: false,
                    };
                    let mut normals = vec![
                        DVec3::new(
                            0.9986021933697105,
                            0.05170006890588294,
                            0.010989188884090003,
                        ),
                        DVec3::new(
                            -0.9986021933697105,
                            0.05170006890588294,
                            0.010989188884090003,
                        ),
                        DVec3::new(0., 0.9420629166961121, -0.3354362249161742),
                    ];
                    if extra {
                        normals.push(DVec3::Y);
                    }
                    let hits: Vec<_> = normals
                        .into_iter()
                        .enumerate()
                        .map(|(index, n)| Hit {
                            u: 0.,
                            normal: rotation * n,
                            velocity: boundary,
                            surface: Surface::Wire,
                            index,
                        })
                        .collect();
                    screen
                        .solve_manifold(
                            &mut particle,
                            &hits,
                            &[],
                            true,
                            ContactContext {
                                synthetic: true,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    if separating {
                        assert_eq!(particle.velocity, velocity);
                        assert_eq!(particle.angular_velocity, spin);
                        assert_eq!(screen.last_step.motor_work_j, 0.);
                    } else {
                        assert!(particle.velocity.distance(boundary) < 1e-10);
                        assert!(
                            particle.angular_velocity.length()
                                * (particle.grain.diameter_m as f64 * 0.5)
                                < 1e-10
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn coupled_two_wire_friction_keeps_support_and_actuator_energy_at_a_narrow_throat() {
        for vertical in [0.8, 0.02] {
            let mut screen = Screen::new(Vec3::ZERO, 0.);
            let velocity = DVec3::new(0.1, 0.2, 0.3);
            let mut particle = ScreenParticle {
                id: 1,
                position: DVec3::ZERO,
                velocity: velocity + DVec3::X * 0.5 - DVec3::Y * 0.1,
                in_gutter: false,
                passed_deck: false,
                angular_velocity: DVec3::ZERO,
                grain: Grain {
                    volume_resolved: true,
                    kind: crate::GrainKind::Gravel,
                    position: Vec3::ZERO,
                    velocity: Vec3::ZERO,
                    material: Material {
                        gravel_m3: 1e-6,
                        ..Default::default()
                    },
                    resting: false,
                    diameter_m: 0.004,
                    affine: glam::Mat3::ZERO,
                    rotation: Quat::IDENTITY,
                    angular_velocity: Vec3::ZERO,
                },
            };
            let rows = [
                Hit {
                    u: 0.,
                    normal: (DVec3::Y * vertical + DVec3::Z).normalize(),
                    velocity,
                    surface: Surface::Wire,
                    index: 0,
                },
                Hit {
                    u: 0.,
                    normal: (DVec3::Y * vertical - DVec3::Z).normalize(),
                    velocity,
                    surface: Surface::Wire,
                    index: 1,
                },
            ];
            let mass = particle.grain.material.wet_mass_kg();
            let r = particle.grain.diameter_m as f64 * 0.5;
            let before = 0.5 * mass * particle.velocity.length_squared();
            screen
                .solve_manifold(
                    &mut particle,
                    &rows,
                    &rows,
                    false,
                    ContactContext {
                        synthetic: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            for row in rows {
                assert!((particle.velocity - velocity).dot(row.normal) >= -1e-9);
            }
            let after = mass
                * (0.5 * particle.velocity.length_squared()
                    + 0.2 * r * r * particle.angular_velocity.length_squared());
            assert!(after - before - screen.last_step.motor_work_j <= 1e-12);
        }
    }

    #[test]
    fn two_support_projection_matches_the_analytic_corner_and_refuses_contradictions() {
        let n1 = DVec3::Y;
        let n2 = DVec3::new(3_f64.sqrt() * 0.5, 0.5, 0.);
        let incoming = DVec3::new(-1., -1., 2.);
        let boundary = DVec3::new(0.2, 0.3, 0.4);
        let rows = [
            (n1, n1.dot(boundary - incoming)),
            (n2, n2.dot(boundary - incoming)),
        ];
        let projection = halfspace_projection(&rows).unwrap();
        // Both independent faces support the corner; the unconstrained Z
        // velocity must survive. This is an independently specified solution.
        let expected = DVec3::new(0.2, 0.3, 2.);
        assert!((incoming + projection.point - expected).length() < 1e-12);
        assert!(projection.multiplier(0) > 0. && projection.multiplier(1) > 0.);
        assert!(
            (projection.point - n1 * projection.multiplier(0) - n2 * projection.multiplier(1))
                .length()
                < 1e-12
        );
        assert!(halfspace_projection(&[(DVec3::Y, 0.1), (-DVec3::Y, 0.1)]).is_none());
    }
}

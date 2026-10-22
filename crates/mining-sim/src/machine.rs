use crate::{Material, Site, Vec3};
use serde::{Deserialize, Serialize};

pub const BUCKET_CAPACITY: f64 = 0.28;
pub const BOOM_LENGTH: f32 = 3.0;
pub const STICK_LENGTH: f32 = 2.3;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HydraulicLimits {
    pub crowd_force_n: f32,
    pub bucket_force_n: f32,
    pub tool_power_w: f32,
}
impl Default for HydraulicLimits {
    fn default() -> Self {
        // Generic 8-ton-class force envelope, not this linkage's measured
        // hydraulics. Tool power is an assumption; see the earthwork protocol.
        Self {
            crowd_force_n: 41_300.,
            bucket_force_n: 58_900.,
            tool_power_w: 30_000.,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub enum DigLimit {
    #[default]
    None,
    Ground,
    Bedrock,
    Capacity,
    Force,
    Power,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct DiggingFeedback {
    pub soil_force_n: f32,
    pub requested_force_n: f32,
    pub tool_power_w: f32,
    pub accepted_fraction: f32,
    pub excavated_m3: f64,
    pub limited: bool,
    pub reason: DigLimit,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MachineInput {
    pub drive: f32,
    pub steer: f32,
    pub swing: f32,
    pub boom: f32,
    pub stick: f32,
    pub curl: f32,
    pub cut: bool,
    pub dump: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Excavator {
    pub position: Vec3,
    pub heading: f32,
    pub swing: f32,
    pub boom: f32,
    pub stick: f32,
    pub curl: f32,
    pub bucket: Material,
    pub engine_on: bool,
    pub fuel_l: f32,
    pub distance_m: f32,
    #[serde(default)]
    pub hydraulics: HydraulicLimits,
    #[serde(default)]
    pub digging: DiggingFeedback,
}

#[derive(Clone, Copy, Debug)]
pub struct ArmPose {
    pub shoulder: Vec3,
    pub elbow: Vec3,
    pub wrist: Vec3,
    pub lip: Vec3,
    pub heel: Vec3,
    pub angle: f32,
    pub forward: Vec3,
}

impl Excavator {
    pub fn new(site: &Site) -> Self {
        let mut machine = Self {
            position: Vec3::new(19.0, site.height(19.0, 17.0), 17.0),
            heading: 0.0,
            swing: 0.0,
            boom: 0.48,
            stick: -1.30,
            curl: -0.20,
            bucket: Material::default(),
            engine_on: true,
            fuel_l: 45.0,
            distance_m: 0.0,
            hydraulics: HydraulicLimits::default(),
            digging: DiggingFeedback::default(),
        };
        // A fresh machine starts with its actual floor and teeth clear of the
        // bank. Old saves retain their pose and may retreat from old contact.
        for _ in 0..72 {
            let pose = machine.arm();
            let right = Vec3::new(pose.forward.z, 0., -pose.forward.x);
            let clear = [0., 0.5, 1.].into_iter().all(|along| {
                [-0.5, 0., 0.5].into_iter().all(|side| {
                    let p = pose.heel.lerp(pose.lip, along)
                        + right * (crate::earthworks::BUCKET_WIDTH_M * side);
                    p.y >= site.height(p.x, p.z) + 0.05
                })
            });
            if clear {
                break;
            }
            machine.boom = (machine.boom + 0.01).min(1.20);
        }
        machine
    }
    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.heading.sin(), 0.0, self.heading.cos())
    }
    pub fn arm(&self) -> ArmPose {
        let yaw = self.heading + self.swing;
        let f = Vec3::new(yaw.sin(), 0.0, yaw.cos());
        let shoulder = self.position + Vec3::Y * 1.65 + f * 0.35;
        let segment =
            |angle: f32, len: f32| f * (angle.cos() * len) + Vec3::Y * (angle.sin() * len);
        let elbow = shoulder + segment(self.boom, BOOM_LENGTH);
        let wrist = elbow + segment(self.boom + self.stick, STICK_LENGTH);
        let angle = self.boom + self.stick + self.curl;
        // Match the rendered bucket floor and tooth tips, including the
        // vertical offset; the old lip was an unrelated point on the arm axis.
        let bucket_point = |y: f32, z: f32| {
            wrist
                + f * (z * angle.cos() - y * angle.sin())
                + Vec3::Y * (y * angle.cos() + z * angle.sin())
        };
        let lip = bucket_point(-0.23, 1.065);
        let heel = bucket_point(-0.2375, -0.01);
        ArmPose {
            shoulder,
            elbow,
            wrist,
            lip,
            heel,
            angle,
            forward: f,
        }
    }
    pub fn step(
        &mut self,
        input: MachineInput,
        site: &mut Site,
        dt: f32,
    ) -> Option<(Vec3, Vec3, Material)> {
        self.digging = DiggingFeedback::default();
        if !self.engine_on
            || self.fuel_l <= 0.0
            || !dt.is_finite()
            || !(0. ..=0.05).contains(&dt)
            || [
                input.drive,
                input.steer,
                input.swing,
                input.boom,
                input.stick,
                input.curl,
            ]
            .iter()
            .any(|v| !v.is_finite())
        {
            return None;
        }
        let input = MachineInput {
            drive: input.drive.clamp(-1., 1.),
            steer: input.steer.clamp(-1., 1.),
            swing: input.swing.clamp(-1., 1.),
            boom: input.boom.clamp(-1., 1.),
            stick: input.stick.clamp(-1., 1.),
            curl: input.curl.clamp(-1., 1.),
            ..input
        };
        let before = self.clone();
        let from = self.arm();
        let activity = input.drive.abs()
            + input.steer.abs()
            + input.swing.abs()
            + input.boom.abs()
            + input.stick.abs()
            + input.curl.abs();
        self.fuel_l = (self.fuel_l - dt * (0.0003 + activity.min(3.0) * 0.001)).max(0.0);
        self.heading = (self.heading + input.steer * dt * 0.45).rem_euclid(std::f32::consts::TAU);
        let next = self.position + self.forward() * input.drive * dt * 1.6;
        if Site::bounds(next, 2.0) && site.water_depth(next.x, next.z) < 0.35 {
            let heights = [(-0.85, -1.15), (0.85, -1.15), (-0.85, 1.15), (0.85, 1.15)]
                .map(|(x, z)| site.height(next.x + x, next.z + z));
            let min = heights.iter().copied().fold(f32::INFINITY, f32::min);
            let max = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max - min < 0.72 && (max - self.position.y).abs() < 0.70 {
                self.distance_m += (next - self.position).length();
                self.position = Vec3::new(next.x, max, next.z);
            }
        }
        self.swing = (self.swing + input.swing * dt * 0.65).rem_euclid(std::f32::consts::TAU);
        self.boom = (self.boom + input.boom * dt * 0.42).clamp(-0.24, 1.20);
        self.stick = (self.stick + input.stick * dt * 0.55).clamp(-2.15, -0.15);
        self.curl = (self.curl + input.curl * dt * 0.90).clamp(-1.70, 1.45);
        if input.dump && !self.bucket.is_empty() {
            self.curl = (self.curl - dt * 1.7).max(-1.7);
        }
        {
            let cutting = input.cut || input.curl > 0.;
            let candidate = self.clone();
            let force_limit = if input.curl.abs() > input.stick.abs() + input.boom.abs() {
                self.hydraulics.bucket_force_n
            } else {
                self.hydraulics.crowd_force_n
            };
            let capacity = (BUCKET_CAPACITY - self.bucket.volume()).max(0.);
            let load_mass = self.bucket.wet_mass_kg() as f32;
            let tool_power = self.hydraulics.tool_power_w;
            let motion = |f: f32| {
                let mut m = candidate.clone();
                m.position = before.position.lerp(candidate.position, f);
                m.distance_m = before.distance_m + (candidate.distance_m - before.distance_m) * f;
                m.heading = before.heading
                    + (candidate.heading - before.heading)
                        .sin()
                        .atan2((candidate.heading - before.heading).cos())
                        * f;
                m.swing = before.swing
                    + (candidate.swing - before.swing)
                        .sin()
                        .atan2((candidate.swing - before.swing).cos())
                        * f;
                m.boom = before.boom + (candidate.boom - before.boom) * f;
                m.stick = before.stick + (candidate.stick - before.stick) * f;
                m.curl = before.curl + (candidate.curl - before.curl) * f;
                m
            };
            let check = |m: &Self| {
                let arm = m.arm();
                let mut plan = if cutting {
                    site.bucket_cut(from, arm)
                } else {
                    Default::default()
                };
                plan.rock_blocked |= site.bucket_blocked(from, arm, cutting);
                let floor_travel = from.lip.distance(arm.lip).max(from.heel.distance(arm.heel));
                let work = plan.soil_force_n * floor_travel
                    + load_mass * 9.81 * (arm.wrist.y - from.wrist.y).max(0.);
                let valid = !plan.rock_blocked
                    && plan.material.volume() <= capacity + 1e-9
                    && plan.soil_force_n <= force_limit
                    && work <= tool_power * dt;
                (valid, plan, work / dt.max(1e-6))
            };
            let (requested_valid, requested_plan, requested_power) = check(self);
            let requested_force = requested_plan.soil_force_n;
            let reason = if requested_valid {
                DigLimit::None
            } else if requested_plan.rock_blocked {
                if cutting {
                    DigLimit::Bedrock
                } else {
                    DigLimit::Ground
                }
            } else if requested_plan.material.volume() > capacity + 1e-9 {
                DigLimit::Capacity
            } else if requested_force > force_limit {
                DigLimit::Force
            } else {
                DigLimit::Power
            };
            let mut fraction = 1.;
            let (plan, power) = if requested_valid {
                (requested_plan, requested_power)
            } else {
                let (mut low, mut high) = (0., 1.);
                for _ in 0..12 {
                    let mid = (low + high) * 0.5;
                    if check(&motion(mid)).0 {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                fraction = low;
                *self = motion(fraction);
                let (_, plan, power) = check(self);
                (plan, power)
            };
            self.digging = DiggingFeedback {
                soil_force_n: plan.soil_force_n,
                requested_force_n: requested_force,
                tool_power_w: power,
                accepted_fraction: fraction,
                excavated_m3: plan.material.volume(),
                limited: fraction < 0.999,
                reason,
            };
            let batch = site.apply_bucket_cut(plan);
            self.bucket.add(batch);
        }
        let arm = self.arm();
        if input.dump && !self.bucket.is_empty() {
            // Opening the bucket releases its real contents at the bucket mouth.
            return Some((
                arm.wrist + arm.forward * 0.35 - Vec3::Y * 0.10,
                arm.forward * 0.35,
                self.bucket.take_volume((dt * 0.32) as f64),
            ));
        }
        None
    }
    pub fn valid(&self) -> bool {
        Site::bounds(self.position, 0.0)
            && self.position.y.abs() < 100.0
            && [
                self.heading,
                self.swing,
                self.boom,
                self.stick,
                self.curl,
                self.fuel_l,
                self.distance_m,
            ]
            .iter()
            .all(|v| v.is_finite())
            && (-0.24..=1.20).contains(&self.boom)
            && (-2.15..=-0.15).contains(&self.stick)
            && (-1.70..=1.45).contains(&self.curl)
            && (0.0..=45.0).contains(&self.fuel_l)
            && self.bucket.valid()
            && self.bucket.volume() <= BUCKET_CAPACITY + 1e-5
            && [
                self.hydraulics.crowd_force_n,
                self.hydraulics.bucket_force_n,
                self.hydraulics.tool_power_w,
            ]
            .iter()
            .all(|v| v.is_finite() && (1. ..=1_000_000.).contains(v))
            && [
                self.digging.soil_force_n,
                self.digging.requested_force_n,
                self.digging.tool_power_w,
                self.digging.accepted_fraction,
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.)
            && self.digging.accepted_fraction <= 1.
            && self.digging.excavated_m3.is_finite()
            && self.digging.excavated_m3 >= 0.
    }
}

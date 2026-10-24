use crate::{Material, Site, Vec3};
use serde::{Deserialize, Serialize};

pub const BUCKET_CAPACITY: f64 = 0.28;
pub const BOOM_LENGTH: f32 = 3.0;
pub const STICK_LENGTH: f32 = 2.3;
pub const STICK_VISUAL_WIDTH_M: f32 = 0.22;
pub const LINK_VISUAL_DEPTH_RATIO: f32 = 1.3;

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
    MechanismValidation,
    MechanicalFailure,
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
    #[serde(default)]
    pub tracks: crate::tracks::TrackMotion,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub articulation: Option<crate::articulation::Articulation>,
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
    pub right: Vec3,
    pub up: Vec3,
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
            tracks: crate::tracks::TrackMotion::default(),
            articulation: None,
        };
        // A fresh machine starts with its actual floor and teeth clear of the
        // bank. Old saves retain their pose and may retreat from old contact.
        for _ in 0..72 {
            let pose = machine.arm();
            let right = pose.right;
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
    pub fn bucket_model(&self) -> crate::bucket::Model {
        self.articulation
            .as_ref()
            .map_or(crate::bucket::Model::LegacyFloor, |a| a.bucket_model())
    }
    pub fn bucket_rotation(&self) -> glam::Quat {
        let shell = self.bucket_model().is_shell();
        let angle = if shell {
            self.boom + self.stick - self.curl + std::f32::consts::FRAC_PI_2
        } else {
            self.boom + self.stick + self.curl
        };
        let q = self.upper_rotation() * glam::Quat::from_rotation_x(-angle);
        if shell {
            q * glam::Quat::from_rotation_y(std::f32::consts::PI)
        } else {
            q
        }
    }
    pub fn forward(&self) -> Vec3 {
        self.chassis_rotation() * Vec3::Z
    }
    pub fn arm(&self) -> ArmPose {
        let rotation = self.upper_rotation();
        let f = rotation * Vec3::Z;
        let up = rotation * Vec3::Y;
        let right = rotation * Vec3::X;
        let shoulder = self.position + up * 1.65 + f * 0.35;
        let segment = |angle: f32, len: f32| f * (angle.cos() * len) + up * (angle.sin() * len);
        let elbow = shoulder + segment(self.boom, BOOM_LENGTH);
        let wrist = elbow + segment(self.boom + self.stick, STICK_LENGTH);
        let shell = self.bucket_model().is_shell();
        let angle = if shell {
            self.boom + self.stick - self.curl + std::f32::consts::FRAC_PI_2
        } else {
            self.boom + self.stick + self.curl
        };
        // Match the rendered bucket floor and tooth tips, including the
        // vertical offset; the old lip was an unrelated point on the arm axis.
        let bucket_point = |y: f32, z: f32| {
            if shell {
                return wrist + self.bucket_rotation() * Vec3::new(0., y, z);
            }
            wrist
                + f * (z * angle.cos() - y * angle.sin())
                + up * (y * angle.cos() + z * angle.sin())
        };
        let lip = if shell {
            wrist + self.bucket_rotation() * self.bucket_model().geometry().tooth_tip
        } else {
            bucket_point(-0.23, 1.065)
        };
        let heel = if self.bucket_model().is_shell() {
            let geometry = self.bucket_model().geometry();
            let point = geometry
                .parts
                .iter()
                .filter(|p| !p.tooth)
                .flat_map(|p| p.vertices.iter())
                .min_by(|a, b| a.y.total_cmp(&b.y).then(a.z.total_cmp(&b.z)))
                .unwrap();
            bucket_point(point.y, point.z)
        } else {
            bucket_point(-0.2375, -0.01)
        };
        ArmPose {
            shoulder,
            elbow,
            wrist,
            lip,
            heel,
            angle,
            forward: f,
            right,
            up,
        }
    }
    pub fn step(
        &mut self,
        input: MachineInput,
        site: &mut Site,
        dt: f32,
    ) -> Option<(Vec3, Vec3, Material)> {
        self.step_with_obstacles(input, site, dt, &[])
    }
    pub fn step_with_obstacles(
        &mut self,
        input: MachineInput,
        site: &mut Site,
        dt: f32,
        obstacles: &[crate::tracks::Obstacle],
    ) -> Option<(Vec3, Vec3, Material)> {
        self.step_with_soil(input, site, dt, obstacles, None)
    }
    pub fn step_with_soil(
        &mut self,
        input: MachineInput,
        site: &mut Site,
        dt: f32,
        obstacles: &[crate::tracks::Obstacle],
        mut soil: Option<&mut crate::soil::ActiveSoil>,
    ) -> Option<(Vec3, Vec3, Material)> {
        if !dt.is_finite()
            || !(0. ..=0.05).contains(&dt)
            || dt == 0.
            || [
                input.drive,
                input.steer,
                input.swing,
                input.boom,
                input.stick,
                input.curl,
            ]
            .iter()
            .any(|x| !x.is_finite())
        {
            return None;
        }
        // Contact, changing arm geometry and finite cuts share the mechanical
        // timestep. Updating soil only after several rigid steps creates a
        // frame-dependent resistance and can lift the chassis off an old cut.
        let steps = (dt * self.tracks.mechanical_hz as f32 - 0.00001)
            .ceil()
            .max(1.) as usize;
        let h = dt / steps as f32;
        let mut digging = DiggingFeedback {
            accepted_fraction: 1.,
            ..Default::default()
        };
        let mut normal = 0.;
        let mut friction = 0f32;
        let input = MachineInput {
            drive: input.drive.clamp(-1., 1.),
            steer: input.steer.clamp(-1., 1.),
            swing: input.swing.clamp(-1., 1.),
            boom: input.boom.clamp(-1., 1.),
            stick: input.stick.clamp(-1., 1.),
            curl: input.curl.clamp(-1., 1.),
            ..input
        };
        for _ in 0..steps {
            if self.articulation.is_some() {
                let env = crate::articulation::Environment {
                    gravity: Vec3::new(0., -9.81, 0.),
                    site: Some(site),
                    obstacles,
                    loads: &[],
                };
                let result = if let Some(soil) = soil.as_deref_mut() {
                    crate::articulation::step_soil(self, input, h, &env, soil)
                } else {
                    crate::articulation::step(self, input, h, &env)
                };
                if let Err(error) = result {
                    self.articulation.as_mut().unwrap().feedback.failure = Some(error);
                    self.digging = DiggingFeedback {
                        limited: true,
                        reason: DigLimit::MechanicalFailure,
                        ..Default::default()
                    };
                    return None;
                }
                let feedback = &self.articulation.as_ref().unwrap().feedback;
                self.digging = DiggingFeedback {
                    tool_power_w: (feedback.pump_work_j / h as f64) as f32,
                    accepted_fraction: feedback.command_fraction,
                    limited: feedback.unsupported_transfer || feedback.command_fraction < 0.999,
                    reason: if feedback.unsupported_transfer {
                        DigLimit::MechanismValidation
                    } else if feedback.command_fraction < 0.999 {
                        DigLimit::Power
                    } else {
                        DigLimit::None
                    },
                    ..Default::default()
                };
                if self.engine_on && self.fuel_l > 0. {
                    let activity = input.drive.abs()
                        + input.steer.abs()
                        + input.swing.abs()
                        + input.boom.abs()
                        + input.stick.abs()
                        + input.curl.abs();
                    self.fuel_l = (self.fuel_l - h * (0.0003 + activity.min(3.) * 0.001)).max(0.);
                }
            } else {
                self.step_mechanics(input, site, h, obstacles);
            }
            let f = self.digging;
            digging.soil_force_n = digging.soil_force_n.max(f.soil_force_n);
            digging.requested_force_n = digging.requested_force_n.max(f.requested_force_n);
            digging.tool_power_w = digging.tool_power_w.max(f.tool_power_w);
            digging.accepted_fraction = digging.accepted_fraction.min(f.accepted_fraction);
            digging.excavated_m3 += f.excavated_m3;
            if f.limited {
                digging.limited = true;
                digging.reason = f.reason;
            }
            normal += self.tracks.feedback.normal_force_n / steps as f32;
            friction = friction.max(self.tracks.feedback.maximum_friction_ratio);
        }
        self.digging = digging;
        self.tracks.feedback.normal_force_n = normal;
        self.tracks.feedback.maximum_friction_ratio = friction;
        if self.articulation.is_none()
            && input.dump
            && self.engine_on
            && self.fuel_l > 0.
            && !self.bucket.is_empty()
        {
            let arm = self.arm();
            return Some((
                arm.wrist + arm.forward * 0.35 - arm.up * 0.10,
                arm.forward * 0.35,
                self.bucket.take_volume((dt * 0.32) as f64),
            ));
        }
        None
    }
    fn step_mechanics(
        &mut self,
        input: MachineInput,
        site: &mut Site,
        dt: f32,
        obstacles: &[crate::tracks::Obstacle],
    ) -> Option<(Vec3, Vec3, Material)> {
        self.digging = DiggingFeedback::default();
        if !dt.is_finite()
            || !(0.0..=0.05).contains(&dt)
            || dt == 0.
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
        let input = if !self.engine_on || self.fuel_l <= 0. {
            MachineInput::default()
        } else {
            input
        };
        self.advance_tracks(input, site, dt, obstacles);
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
        if self.engine_on && self.fuel_l > 0. {
            self.fuel_l = (self.fuel_l - dt * (0.0003 + activity.min(3.0) * 0.001)).max(0.0);
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
            let reaction = self.tracks.feedback.bucket_reaction_n;
            let opposition = |arm: ArmPose| {
                let d = ((arm.heel + arm.lip) - (from.heel + from.lip)) * 0.5;
                if d.length_squared() > 1e-12 {
                    (-reaction.dot(d.normalize())).max(0.)
                } else {
                    0.
                }
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
                let displacement = ((arm.heel + arm.lip) - (from.heel + from.lip)) * 0.5;
                let opposing_force = opposition(arm);
                // The prescribed joints must also pay gravitational lift of
                // their links and work against the preceding contact reaction.
                // This remains explicit mechanical coupling, not full hydraulics.
                let gravity_work = 9.81
                    * (1300.
                        * ((arm.shoulder.y + arm.elbow.y) - (from.shoulder.y + from.elbow.y))
                        * 0.5
                        + 840.
                            * ((arm.elbow.y + arm.wrist.y) - (from.elbow.y + from.wrist.y))
                            * 0.5
                        + (250. + load_mass) * (arm.wrist.y - from.wrist.y))
                        .max(0.);
                let work = plan.soil_force_n * floor_travel
                    + gravity_work
                    + (-reaction.dot(displacement)).max(0.);
                let valid = !plan.rock_blocked
                    && plan.material.volume() <= capacity + 1e-9
                    && plan.soil_force_n + opposing_force <= force_limit
                    && work <= tool_power * dt;
                (valid, plan, work / dt.max(1e-6))
            };
            let (requested_valid, requested_plan, requested_power) = check(self);
            let requested_force = requested_plan.soil_force_n + opposition(self.arm());
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
        None
    }
    pub fn valid(&self) -> bool {
        // The dynamic solver may leave sub-milliradian limit error; keep the
        // original model's exact domain while validating real, unclamped poses.
        let epsilon = if self.articulation.is_some() {
            0.001
        } else {
            0.
        };
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
            && (-0.24 - epsilon..=1.20 + epsilon).contains(&self.boom)
            && (-2.15 - epsilon..=-0.15 + epsilon).contains(&self.stick)
            && (-1.70 - epsilon..=1.45 + epsilon).contains(&self.curl)
            && self
                .articulation
                .as_ref()
                .is_none_or(|state| state.valid_for(self))
            && (0.0..=45.0).contains(&self.fuel_l)
            && self.tracks.valid()
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

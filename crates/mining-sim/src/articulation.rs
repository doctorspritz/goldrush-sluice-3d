//! Floating-base, reduced-coordinate machine dynamics. The explicit model
//! is a prerequisite for soil reactions, not a calibrated hydraulic circuit.
use crate::{
    machine::{BOOM_LENGTH, STICK_LENGTH},
    tracks::{
        pose, rv, sq, sv, Belts, Obstacle, TRACK_GAUGE, TRACK_LENGTH, TRACK_WIDTH, TRAVEL_SPEED,
    },
    Excavator, MachineInput, Site, Vec3,
};
use glam::{DMat3, DQuat, DVec3, Quat};
use rapier3d::prelude as r;
use serde::{Deserialize, Serialize};

const ROOT_COM_Y: f32 = (2600. * 0.325 + 700. * 0.70) / 3300.;
const ROOT_COM: Vec3 = Vec3::new(0., ROOT_COM_Y, 0.);
const UPPER_COM: Vec3 = Vec3::new(0., 1.10, -0.1);
mod exchange;
pub use exchange::{capture_payload, payload_frame, release_payload, sample_payload};

const AXIS: usize = 3; // A revolute joint's own free angular X coordinate.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Articulation {
    #[serde(default, skip_serializing_if = "crate::bucket::Model::is_legacy")]
    bucket_model: crate::bucket::Model,
    /// Root COM linear/world angular velocity, then the four joint rates.
    pub velocity: [f32; 10],
    /// Swing, -boom, -stick, then +curl for ShellV1 or -curl for LegacyFloor.
    pub coordinates: [f32; 4],
    /// Finite elastic actuator references; not prescribed body positions.
    pub reference: [f32; 4],
    pub stiffness_nm_rad: [f32; 4],
    pub damping_nms_rad: [f32; 4],
    pub payload_kg: f32,
    /// Actual carried spatial moments after an exchange. Absent v7 state
    /// retains the original declared floor-distributed payload model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_moments: Option<crate::payload::MassMoments>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exchange: Option<crate::payload::ExchangeLedger>,
    pub pump_energy_j: f64,
    pub relief_heat_j: f64,
    pub feedback: Feedback,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Feedback {
    pub mass_kg: f64,
    pub center_of_mass: Vec3,
    pub linear_momentum_ns: Vec3,
    /// Angular momentum about the fixed world origin.
    pub angular_momentum_nms: Vec3,
    /// Intrinsic angular momentum, independently of the translation budget.
    pub angular_momentum_about_com_nms: Vec3,
    pub kinetic_j: f64,
    pub gravitational_j: f64,
    pub elastic_j: f64,
    pub pump_work_j: f64,
    pub command_heat_j: f64,
    pub relief_heat_j: f64,
    /// Signed change of total mechanical energy minus command supply,
    /// plus separately calculated relief/return heat. Damping/contact loss
    /// is not invented from this residual or claimed as a closed heat budget.
    pub passivity_residual_j: f64,
    pub command_fraction: f32,
    pub external_work_j: f64,
    /// False while the existing ideal belt drive supplies unmetered work.
    pub passivity_inputs_accounted: bool,
    /// Reduced-coordinate motor impulses are not written back by this pinned
    /// engine. An absent observation must never be rendered as zero load.
    pub motor_impulse_nms: Option<[f32; 4]>,
    pub ground_normal_n: f32,
    pub unsupported_transfer: bool,
    pub failure: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct Load {
    /// 0 chassis, 1 upper house, 2 boom, 3 stick, 4 bucket.
    pub part: usize,
    pub force_n: Vec3,
    pub point: Vec3,
    pub torque_nm: Vec3,
}

/// Explicit validation environment. Ordinary integration uses terrestrial
/// gravity, actual Site geometry and the existing finite equipment boxes.
pub struct Environment<'a> {
    pub gravity: Vec3,
    pub site: Option<&'a Site>,
    pub obstacles: &'a [Obstacle],
    pub loads: &'a [Load],
}
impl Environment<'_> {
    pub fn free(gravity: Vec3) -> Self {
        Self {
            gravity,
            site: None,
            obstacles: &[],
            loads: &[],
        }
    }
}

impl Articulation {
    pub fn bucket_model(&self) -> crate::bucket::Model {
        self.bucket_model
    }
    pub fn valid(&self) -> bool {
        self.velocity
            .iter()
            .all(|x| x.is_finite() && x.abs() < 100.)
            && self
                .coordinates
                .iter()
                .chain(&self.reference)
                .all(|x| x.is_finite() && x.abs() < 20.)
            && self
                .stiffness_nm_rad
                .iter()
                .all(|x| x.is_finite() && (0. ..=1e9).contains(x))
            && self
                .damping_nms_rad
                .iter()
                .all(|x| x.is_finite() && (0. ..=1e9).contains(x))
            && self.payload_kg.is_finite()
            && (0. ..=10_000.).contains(&self.payload_kg)
            && self.pump_energy_j.is_finite()
            && self.pump_energy_j >= 0.
            && self.relief_heat_j.is_finite()
            && self.relief_heat_j >= 0.
    }
    pub fn valid_for(&self, machine: &Excavator) -> bool {
        self.valid()
            && self.payload_kg == machine.bucket.wet_mass_kg() as f32
            && (self.coordinates[0] - machine.swing)
                .sin()
                .atan2((self.coordinates[0] - machine.swing).cos())
                .abs()
                < 0.00001
            && (self.coordinates[1] + machine.boom).abs() < 0.000001
            && (self.coordinates[2] + machine.stick).abs() < 0.000001
            && (self.coordinates[3]
                - if self.bucket_model.is_shell() {
                    machine.curl
                } else {
                    -machine.curl
                })
            .abs()
                < 0.000001
            && Vec3::from_slice(&self.velocity[3..6]).distance(machine.tracks.angular_velocity)
                < 0.00001
            && self.feedback.center_of_mass.is_finite()
            && self.feedback.linear_momentum_ns.is_finite()
            && self.feedback.angular_momentum_nms.is_finite()
            && self.feedback.angular_momentum_about_com_nms.is_finite()
            && [
                self.feedback.kinetic_j,
                self.feedback.gravitational_j,
                self.feedback.elastic_j,
                self.feedback.pump_work_j,
                self.feedback.command_heat_j,
                self.feedback.relief_heat_j,
                self.feedback.passivity_residual_j,
                self.feedback.external_work_j,
                self.feedback.mass_kg,
            ]
            .iter()
            .all(|x| x.is_finite())
            && (0. ..=1.).contains(&self.feedback.command_fraction)
            && self.exchange.as_ref().is_none_or(|x| x.valid())
            && self.payload_moments.is_none_or(|x| {
                x.valid()
                    && x.mass_kg > 0.
                    && x.center.length() < 4.
                    && (x.mass_kg - machine.bucket.wet_mass_kg()).abs() < 1e-9 * x.mass_kg.max(1.)
                    && self.exchange.is_some()
            })
            && torque_limits(machine).iter().enumerate().all(|(i, limit)| {
                self.stiffness_nm_rad[i] == 0.
                    || (self.coordinates[i] - self.reference[i]).abs()
                        <= *limit / self.stiffness_nm_rad[i] + 0.000002
            })
    }
}

fn parameters_valid(m: &Excavator) -> bool {
    let q = m.chassis_rotation();
    m.position.is_finite()
        && q.is_finite()
        && (q.length_squared() - 1.).abs() < 0.0001
        && [m.heading, m.swing, m.boom, m.stick, m.curl]
            .iter()
            .all(|x| x.is_finite())
        && m.tracks.friction.is_finite()
        && (0. ..=2.).contains(&m.tracks.friction)
        && (60..=960).contains(&m.tracks.mechanical_hz)
        && [
            m.hydraulics.crowd_force_n,
            m.hydraulics.bucket_force_n,
            m.hydraulics.tool_power_w,
        ]
        .iter()
        .all(|x| x.is_finite() && (0. ..=1_000_000.).contains(x))
        && m.bucket.valid()
        && m.bucket.volume() <= crate::machine::BUCKET_CAPACITY + 1e-5
}

impl Excavator {
    pub fn enable_articulation(&mut self) -> Result<(), String> {
        if !parameters_valid(self) {
            return Err("Invalid mechanism parameters".into());
        }
        if let Some(state) = &self.articulation {
            return if state.valid_for(self) {
                Ok(())
            } else {
                Err("Invalid articulated state".into())
            };
        }
        let coords = [
            self.swing.sin().atan2(self.swing.cos()),
            -self.boom,
            -self.stick,
            self.curl,
        ];
        self.articulation = Some(Articulation {
            bucket_model: crate::bucket::Model::MountedShellV2,
            velocity: [0.; 10],
            coordinates: coords,
            reference: coords,
            stiffness_nm_rad: [20e6; 4],
            damping_nms_rad: [100_000.; 4],
            payload_kg: self.bucket.wet_mass_kg() as f32,
            payload_moments: None,
            exchange: None,
            pump_energy_j: 0.,
            relief_heat_j: 0.,
            feedback: Feedback::default(),
        });
        let initial = self.articulation.as_ref().unwrap().clone();
        let mut model = Model::new(self, &Environment::free(Vec3::ZERO), &initial)?;
        let totals = model.totals();
        let root_com = sv(model.world.bodies[model.bodies[0]].center_of_mass());
        let v = self.tracks.velocity + self.tracks.angular_velocity.cross(root_com - totals.com);
        let state = self.articulation.as_mut().unwrap();
        state.velocity[..3].copy_from_slice(&v.to_array());
        state.velocity[3..6].copy_from_slice(&self.tracks.angular_velocity.to_array());
        model.set_velocity(&state.velocity);
        state.feedback = model.feedback(state, &self.hydraulics);
        Ok(())
    }
}

fn torque_limits(m: &Excavator) -> [f32; 4] {
    // Declared force-to-torque lever assumptions, not cylinder pin geometry.
    [
        m.hydraulics.crowd_force_n * TRACK_GAUGE,
        m.hydraulics.crowd_force_n * BOOM_LENGTH,
        m.hydraulics.crowd_force_n * STICK_LENGTH,
        m.hydraulics.bucket_force_n * 1.065,
    ]
}
fn potential(error: f32, stiffness: f32, limit: f32) -> f64 {
    if stiffness == 0. || limit == 0. {
        return 0.;
    }
    let k = stiffness as f64;
    let f = limit as f64;
    let x = (error as f64).abs();
    let elastic = x.min(f / k);
    0.5 * k * elastic * elastic + f * (x - elastic)
}
fn elastic(state: &Articulation, limits: [f32; 4]) -> f64 {
    (0..4)
        .map(|i| {
            potential(
                state.coordinates[i] - state.reference[i],
                state.stiffness_nm_rad[i],
                limits[i],
            )
        })
        .sum()
}
fn relieve(state: &mut Articulation, limits: [f32; 4]) -> f64 {
    let before = elastic(state, limits);
    for (i, limit) in limits.iter().enumerate() {
        let k = state.stiffness_nm_rad[i];
        if k > 0. {
            let error = state.coordinates[i] - state.reference[i];
            let cap = *limit / k;
            state.reference[i] = state.coordinates[i] - error.clamp(-cap, cap);
        } else {
            // With no elastic element its reference is a pure coordinate gauge.
            state.reference[i] = state.coordinates[i];
        }
    }
    (before - elastic(state, limits)).max(0.)
}
fn command(
    m: &Excavator,
    state: &mut Articulation,
    input: MachineInput,
    dt: f32,
) -> (f64, f64, f64, f32) {
    let limits = torque_limits(m);
    let enabled = m.engine_on && m.fuel_l > 0.;
    let rates = [
        input.swing * 0.65,
        -input.boom * 0.42,
        -input.stick * 0.55,
        if state.bucket_model.is_shell() {
            input.curl * 0.90
        } else {
            -input.curl * 0.90
        },
    ];
    let ranges = [
        (-20., 20.),
        (-1.20, 0.24),
        (0.15, 2.15),
        if state.bucket_model.is_shell() {
            (-1.70, 1.45)
        } else {
            (-1.45, 1.70)
        },
    ];
    let before = state.reference;
    let candidate = |fraction: f32| {
        let mut target = before;
        let mut pump = 0.;
        let mut returned = 0.;
        for i in 0..4 {
            if enabled {
                target[i] = (before[i] + rates[i] * dt * fraction).clamp(ranges[i].0, ranges[i].1);
            }
            let old = potential(
                state.coordinates[i] - before[i],
                state.stiffness_nm_rad[i],
                limits[i],
            );
            let new = potential(
                state.coordinates[i] - target[i],
                state.stiffness_nm_rad[i],
                limits[i],
            );
            pump += (new - old).max(0.);
            returned += (old - new).max(0.);
        }
        (target, pump, returned)
    };
    let available = if enabled {
        m.hydraulics.tool_power_w as f64 * dt as f64
    } else {
        0.
    };
    let mut fraction = 1.;
    if candidate(1.).1 > available {
        let (mut low, mut high) = (0., 1.);
        for _ in 0..24 {
            let mid = (low + high) * 0.5;
            if candidate(mid).1 <= available {
                low = mid;
            } else {
                high = mid;
            }
        }
        fraction = low;
    }
    let (target, pump, returned) = candidate(fraction);
    state.reference = target;
    let relief = relieve(state, limits);
    (pump, returned, relief, fraction)
}

struct Totals {
    mass: f64,
    com: Vec3,
    momentum: Vec3,
    angular: Vec3,
    angular_com: Vec3,
    kinetic: f64,
    gravity: f64,
}
struct Model {
    world: r::PhysicsWorld,
    bodies: [r::RigidBodyHandle; 5],
    joints: [r::MultibodyJointHandle; 4],
    gravity: Vec3,
}
impl Model {
    fn new(m: &mut Excavator, env: &Environment<'_>, s: &Articulation) -> Result<Self, String> {
        Self::with_ground(m, env, s, None)
    }
    fn with_ground(
        m: &mut Excavator,
        env: &Environment<'_>,
        s: &Articulation,
        ground_override: Option<r::SharedShape>,
    ) -> Result<Self, String> {
        let q = m.chassis_rotation();
        let upper = m.upper_rotation();
        let arm = m.arm();
        let boom = upper * Quat::from_rotation_x(-m.boom);
        let stick = boom * Quat::from_rotation_x(-m.stick);
        let bucket = if s.bucket_model.is_shell() {
            m.bucket_rotation()
        } else {
            stick * Quat::from_rotation_x(-m.curl)
        };
        let heel = Vec3::new(0., -0.2375, -0.01);
        let lip = Vec3::new(0., -0.23, 1.065);
        let floor_q = Quat::from_rotation_arc(Vec3::Z, (lip - heel).normalize());
        let floor_center = (heel + lip) * 0.5 + floor_q * Vec3::Y * 0.025;
        let shell = s.bucket_model.is_shell();
        let bucket_moments = if shell {
            Some(steel_moments(s.bucket_model).combine(carried_moments(m)))
        } else {
            None
        };
        let bucket_com = if let Some(moments) = bucket_moments {
            moments.center.as_vec3()
        } else if let Some(payload) = s.payload_moments {
            steel_moments(s.bucket_model)
                .combine(payload)
                .center
                .as_vec3()
        } else {
            floor_center
        };
        let mut world = r::PhysicsWorld::new();
        world.gravity = rv(env.gravity);
        world.integration_parameters = r::IntegrationParameters {
            dt: 1. / m.tracks.mechanical_hz as f32,
            num_solver_iterations: 4,
            num_internal_pgs_iterations: 16,
            friction_model: r::FrictionModel::Coulomb,
            contact_clustering: true,
            contact_recycling: false,
            normalized_allowed_linear_error: 0.0001,
            ..Default::default()
        };
        if let Some(site) = env.site {
            let (shape, c) = if let Some(shape) = ground_override {
                (
                    shape,
                    Vec3::new(
                        (crate::site::WIDTH - 1) as f32 * crate::site::CELL * 0.5,
                        0.,
                        (crate::site::WIDTH - 1) as f32 * crate::site::CELL * 0.5,
                    ),
                )
            } else {
                let ground = m.ground(site);
                (
                    ground.shape.clone(),
                    Vec3::new(
                        (ground.origin[0] as f32 + 32.) * crate::site::CELL,
                        0.,
                        (ground.origin[1] as f32 + 32.) * crate::site::CELL,
                    ),
                )
            };
            world.insert(
                r::RigidBodyBuilder::fixed(),
                r::ColliderBuilder::new(shape)
                    .translation(rv(c))
                    .friction(1.)
                    .user_data(10),
            );
            let edge = (crate::site::WIDTH - 1) as f32 * crate::site::CELL;
            for (c, h) in [
                (Vec3::new(-0.5, 10., edge * 0.5), Vec3::new(0.5, 20., edge)),
                (
                    Vec3::new(edge + 0.5, 10., edge * 0.5),
                    Vec3::new(0.5, 20., edge),
                ),
                (Vec3::new(edge * 0.5, 10., -0.5), Vec3::new(edge, 20., 0.5)),
                (
                    Vec3::new(edge * 0.5, 10., edge + 0.5),
                    Vec3::new(edge, 20., 0.5),
                ),
            ] {
                world.insert(
                    r::RigidBodyBuilder::fixed(),
                    r::ColliderBuilder::cuboid(h.x, h.y, h.z).translation(rv(c)),
                );
            }
        }
        for o in env.obstacles {
            world.insert(
                r::RigidBodyBuilder::fixed(),
                r::ColliderBuilder::cuboid(o.half_extents.x, o.half_extents.y, o.half_extents.z)
                    .position(pose(o.center, o.rotation))
                    .friction(0.65),
            );
        }
        let poses = [
            pose(m.position + q * ROOT_COM, q),
            pose(m.position + upper * UPPER_COM, upper),
            pose(arm.shoulder + boom * Vec3::Z * (BOOM_LENGTH * 0.5), boom),
            pose(arm.elbow + stick * Vec3::Z * (STICK_LENGTH * 0.5), stick),
            pose(arm.wrist + bucket * bucket_com, bucket),
        ];
        let bodies = std::array::from_fn(|i| {
            let mut builder = r::RigidBodyBuilder::dynamic()
                .pose(poses[i])
                .can_sleep(false)
                .ccd_enabled(true);
            if i == 4 {
                if let Some(moments) = bucket_moments {
                    builder =
                        builder.additional_mass_properties(r::MassProperties::with_inertia_matrix(
                            rv(moments.center.as_vec3() - bucket_com),
                            moments.mass_kg as f32,
                            r::Matrix::from_cols_array(
                                &moments.inertia().to_cols_array().map(|x| x as f32),
                            ),
                        ));
                } else if let Some(payload) = s.payload_moments {
                    builder =
                        builder.additional_mass_properties(r::MassProperties::with_inertia_matrix(
                            rv(payload.center.as_vec3() - bucket_com),
                            payload.mass_kg as f32,
                            r::Matrix::from_cols_array(
                                &payload.inertia().to_cols_array().map(|x| x as f32),
                            ),
                        ));
                }
            }
            world.insert_body(builder)
        });
        let mut attach = |part: usize, b: r::ColliderBuilder| {
            world.insert_collider(
                b.friction(m.tracks.friction)
                    .friction_combine_rule(r::CoefficientCombineRule::Min)
                    .restitution(0.),
                Some(bodies[part]),
            );
        };
        for (tag, side) in [(1, -1.), (2, 1.)] {
            attach(
                0,
                r::ColliderBuilder::cuboid(TRACK_WIDTH * 0.5, 0.325, TRACK_LENGTH * 0.5)
                    .translation(rv(Vec3::new(side * TRACK_GAUGE * 0.5, 0.325, 0.) - ROOT_COM))
                    .mass(1300.)
                    .user_data(tag)
                    .active_hooks(r::ActiveHooks::MODIFY_SOLVER_CONTACTS),
            );
        }
        attach(
            0,
            r::ColliderBuilder::cuboid(0.85, 0.175, 0.95)
                .translation(rv(Vec3::new(0., 0.70, 0.) - ROOT_COM))
                .mass(700.),
        );
        attach(1, r::ColliderBuilder::cuboid(1.025, 0.30, 1.15).mass(3000.));
        attach(
            1,
            r::ColliderBuilder::cuboid(0.53, 0.68, 0.775)
                .translation(rv(Vec3::new(-0.64, 2.10, 0.22) - UPPER_COM))
                .mass(0.),
        );
        attach(
            2,
            r::ColliderBuilder::capsule_from_endpoints(
                rv(-Vec3::Z * (BOOM_LENGTH * 0.5)),
                rv(Vec3::Z * (BOOM_LENGTH * 0.5)),
                0.17,
            )
            .mass(1300.),
        );
        attach(
            3,
            r::ColliderBuilder::capsule_from_endpoints(
                rv(-Vec3::Z * (STICK_LENGTH * 0.5)),
                rv(Vec3::Z * (STICK_LENGTH * 0.5)),
                0.13,
            )
            .mass(840.),
        );
        if shell {
            attach(
                4,
                r::ColliderBuilder::new(s.bucket_model.geometry().shape())
                    .translation(rv(-bucket_com))
                    .density(0.)
                    .user_data(3),
            );
        } else {
            attach(
                4,
                r::ColliderBuilder::cuboid(0.41, 0.025, (lip - heel).length() * 0.5)
                    .position(pose(floor_center - bucket_com, floor_q))
                    .mass(if s.payload_moments.is_some() {
                        250.
                    } else {
                        250. + s.payload_kg
                    })
                    .user_data(3),
            );
        }
        for b in bodies {
            world.bodies[b].recompute_mass_properties_from_colliders(&world.colliders);
        }
        let limits = torque_limits(m);
        let anchors = [
            (
                Vec3::new(0., 0.95, 0.) - ROOT_COM,
                Vec3::new(0., 0.95, 0.) - UPPER_COM,
            ),
            (
                Vec3::new(0., 1.65, 0.35) - UPPER_COM,
                -Vec3::Z * (BOOM_LENGTH * 0.5),
            ),
            (
                Vec3::Z * (BOOM_LENGTH * 0.5),
                -Vec3::Z * (STICK_LENGTH * 0.5),
            ),
            (Vec3::Z * (STICK_LENGTH * 0.5), -bucket_com),
        ];
        let ranges = [
            None,
            Some([-1.20, 0.24]),
            Some([0.15, 2.15]),
            Some(if shell { [-1.70, 1.45] } else { [-1.45, 1.70] }),
        ];
        let mut joints = Vec::new();
        for i in 0..4 {
            let mut j =
                r::RevoluteJointBuilder::new(if i == 0 { r::Vector::Y } else { r::Vector::X })
                    .local_anchor1(rv(anchors[i].0))
                    .local_anchor2(rv(anchors[i].1))
                    .contacts_enabled(false)
                    .motor_model(r::MotorModel::ForceBased)
                    .motor_position(s.reference[i], s.stiffness_nm_rad[i], s.damping_nms_rad[i])
                    .motor_max_force(limits[i]);
            if shell && i == 3 {
                let mount = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)
                    * Quat::from_rotation_y(std::f32::consts::PI);
                j.0.data.local_frame2.rotation = pose(Vec3::ZERO, mount.inverse()).rotation;
            }
            if let Some(range) = ranges[i] {
                j = j.limits(range);
            }
            joints.push(
                world
                    .insert_multibody_joint(bodies[i], bodies[i + 1], j)
                    .ok_or("Invalid articulated tree")?,
            );
        }
        let joints: [_; 4] = joints.try_into().unwrap();
        let (mb, _) = world.multibody_joints.get_mut(joints[0]).unwrap();
        for i in 0..4 {
            mb.link_mut(i + 1)
                .unwrap()
                .joint
                .apply_displacement(&[s.coordinates[i]]);
        }
        mb.forward_kinematics(&world.bodies, true);
        mb.damping_mut().fill(0.); // All modeled joint damping is explicit above.
        mb.set_self_contacts_enabled(true);
        mb.update_rigid_bodies(&mut world.bodies, true);
        let mut result = Self {
            world,
            bodies,
            joints,
            gravity: env.gravity,
        };
        result.set_velocity(&s.velocity);
        for load in env.loads {
            if load.part >= 5
                || !load.force_n.is_finite()
                || !load.point.is_finite()
                || !load.torque_nm.is_finite()
            {
                return Err("Invalid external articulated load".into());
            }
            let b = &mut result.world.bodies[bodies[load.part]];
            b.add_force_at_point(rv(load.force_n), rv(load.point), false);
            b.add_torque(rv(load.torque_nm), false);
        }
        Ok(result)
    }
    fn set_velocity(&mut self, velocity: &[f32; 10]) {
        let (mb, _) = self.world.multibody_joints.get_mut(self.joints[0]).unwrap();
        mb.generalized_velocity_mut().copy_from_slice(velocity);
        for (i, handle) in self.bodies.iter().enumerate() {
            let j = mb.body_jacobian(i);
            let mut v = [0.; 6];
            for row in 0..6 {
                for col in 0..10 {
                    v[row] += j[(row, col)] * velocity[col];
                }
            }
            self.world.bodies[*handle].set_linvel(r::Vector::new(v[0], v[1], v[2]), false);
            self.world.bodies[*handle].set_angvel(r::Vector::new(v[3], v[4], v[5]), false);
        }
    }
    fn totals(&self) -> Totals {
        let (mut mass, mut center, mut p, mut l, mut k, mut u) =
            (0., DVec3::ZERO, DVec3::ZERO, DVec3::ZERO, 0., 0.);
        for h in self.bodies {
            let b = &self.world.bodies[h];
            let m = b.mass() as f64;
            let c = sv(b.center_of_mass()).as_dvec3();
            let v = sv(b.linvel()).as_dvec3();
            let a = b.angvel();
            // This pinned implementation returns I, despite the older public
            // comment calling it a square root. Its kinetic-energy path uses
            // the same inverse of effective_world_inv_inertia directly.
            let inertia = b.mass_properties().effective_angular_inertia();
            let spin = sv(inertia * a).as_dvec3();
            mass += m;
            center += c * m;
            p += v * m;
            l += c.cross(v * m) + spin;
            k += b.kinetic_energy() as f64;
            u -= m * self.gravity.as_dvec3().dot(c);
        }
        Totals {
            mass,
            com: (center / mass).as_vec3(),
            momentum: p.as_vec3(),
            angular: l.as_vec3(),
            angular_com: (l - (center / mass).cross(p)).as_vec3(),
            kinetic: k,
            gravity: u,
        }
    }
    fn feedback(&self, s: &Articulation, hydraulics: &crate::machine::HydraulicLimits) -> Feedback {
        let t = self.totals();
        let limits = [
            hydraulics.crowd_force_n * TRACK_GAUGE,
            hydraulics.crowd_force_n * BOOM_LENGTH,
            hydraulics.crowd_force_n * STICK_LENGTH,
            hydraulics.bucket_force_n * 1.065,
        ];
        Feedback {
            mass_kg: t.mass,
            center_of_mass: t.com,
            linear_momentum_ns: t.momentum,
            angular_momentum_nms: t.angular,
            angular_momentum_about_com_nms: t.angular_com,
            kinetic_j: t.kinetic,
            gravitational_j: t.gravity,
            elastic_j: elastic(s, limits),
            ..Default::default()
        }
    }
}

/// One mechanical step. Active soil and payload exchange are deliberately not
/// admitted by this core: they need their own force and custody transactions.
pub fn step(
    m: &mut Excavator,
    input: MachineInput,
    dt: f32,
    env: &Environment<'_>,
) -> Result<(), String> {
    step_inner(m, input, dt, env, None)
}
/// Moving finite soil shares the very same constraint solve with all five
/// machine bodies. No prescribed earth pressure or post-step impulse kick.
pub fn step_soil(
    m: &mut Excavator,
    input: MachineInput,
    dt: f32,
    env: &Environment<'_>,
    soil: &mut crate::soil::ActiveSoil,
) -> Result<(), String> {
    if !soil.valid() {
        return Err("Invalid moving soil".into());
    }
    if env.site.is_some_and(|site| !soil.dry_contact(site)) {
        return Err("Wet moving-soil contact is not coupled yet".into());
    }
    let mut machine = m.clone();
    let mut candidate = soil.clone();
    step_inner(&mut machine, input, dt, env, Some(&mut candidate))?;
    if !machine.valid() || !candidate.valid() {
        let clod = candidate
            .clods
            .iter()
            .enumerate()
            .find(|(_, c)| !c.valid())
            .map(|(i, c)| {
                (
                    i,
                    c.center,
                    c.velocity.length(),
                    c.angular_velocity.length(),
                    c.rotation.length_squared(),
                )
            });
        return Err(format!("Moving-soil contact integration became invalid: machine={} articulation={} tracks={} angles={:?} invalid_clod={clod:?}",
            machine.valid(), machine.articulation.as_ref().is_some_and(|a| a.valid_for(&machine)), machine.tracks.valid(), [machine.boom, machine.stick, machine.curl]));
    }
    *m = machine;
    *soil = candidate;
    Ok(())
}
fn step_inner(
    m: &mut Excavator,
    input: MachineInput,
    dt: f32,
    env: &Environment<'_>,
    mut soil: Option<&mut crate::soil::ActiveSoil>,
) -> Result<(), String> {
    if !dt.is_finite() || dt <= 0. || dt > 0.05 || !env.gravity.is_finite() {
        return Err("Invalid articulated timestep/gravity".into());
    }
    if !parameters_valid(m) {
        return Err("Invalid mechanism parameters".into());
    }
    let mut s = m
        .articulation
        .as_ref()
        .ok_or("Articulation was not enabled")?
        .clone();
    if !s.valid_for(m) {
        return Err("Articulated state or constant payload changed outside custody".into());
    }
    let axes = [
        input.drive,
        input.steer,
        input.swing,
        input.boom,
        input.stick,
        input.curl,
    ];
    if axes.iter().any(|x| !x.is_finite() || x.abs() > 1.) {
        return Err("Invalid articulated input".into());
    }
    let ground = soil
        .as_deref_mut()
        .and_then(|soil| env.site.map(|site| soil.ground(site)));
    let mut model = Model::with_ground(m, env, &s, ground)?;
    model
        .world
        .integration_parameters
        .maximum_dissipation_friction = soil.is_some();
    if soil.is_some() && s.bucket_model == crate::bucket::Model::MountedShellV2 {
        crate::soil_contacts::configure(&mut model.world);
    }
    let mut soil_handles = Vec::new();
    if let Some(soil) = soil.as_deref_mut() {
        for clod in &mut soil.clods {
            let handle = if s.bucket_model == crate::bucket::Model::MountedShellV2 {
                clod.insert_precise(&mut model.world)?
            } else {
                clod.insert(&mut model.world)?
            };
            soil_handles.push(handle);
        }
    }
    let soil_before = soil.as_deref().map(|s| s.totals(env.gravity));
    let before = model.feedback(&s, &m.hydraulics);
    let before_frames = model.bodies.map(|h| {
        let b = &model.world.bodies[h];
        (sv(b.center_of_mass()), sq(*b.rotation()))
    });
    let (pump, returned, relief_before, fraction) = command(m, &mut s, input, dt);
    let (mb, _) = model
        .world
        .multibody_joints
        .get_mut(model.joints[0])
        .unwrap();
    for i in 0..4 {
        mb.link_mut(i + 1).unwrap().joint.data.set_motor_position(
            r::JointAxis::AngX,
            s.reference[i],
            s.stiffness_nm_rad[i],
            s.damping_nms_rad[i],
        );
    }
    model.world.integration_parameters.dt = dt;
    let powered = m.engine_on && m.fuel_l > 0.;
    let drive = if powered {
        input.drive * TRAVEL_SPEED
    } else {
        0.
    };
    let steer = if powered { input.steer * 0.42 } else { 0. };
    let speed = [
        (drive + steer).clamp(-TRAVEL_SPEED, TRAVEL_SPEED),
        (drive - steer).clamp(-TRAVEL_SPEED, TRAVEL_SPEED),
    ];
    model.world.step_with_events(
        &Belts {
            forward: *model.world.bodies[model.bodies[0]].rotation() * r::Vector::Z,
            speed,
        },
        &(),
    );
    let (mb, _) = model.world.multibody_joints.get(model.joints[0]).unwrap();
    s.velocity
        .copy_from_slice(mb.generalized_velocity().as_slice());
    for i in 0..4 {
        s.coordinates[i] = mb.link(i + 1).unwrap().joint.coords()[AXIS];
    }
    // Public rigid-body velocities lag the last generalized solver update.
    // Re-evaluate the final coordinate chart and its Jacobian before reading
    // physical momentum/energy. No generalized state is changed here.
    let (mb, _) = model
        .world
        .multibody_joints
        .get_mut(model.joints[0])
        .unwrap();
    mb.forward_kinematics(&model.world.bodies, false);
    mb.update_rigid_bodies(&mut model.world.bodies, true);
    model.set_velocity(&s.velocity);
    let relief_after = relieve(&mut s, torque_limits(m));
    // A chart change preserves the relative spring deformation without letting
    // many revolutions erode the precision of the unbounded swing coordinate.
    let wrapped = s.coordinates[0].sin().atan2(s.coordinates[0].cos());
    s.reference[0] += wrapped - s.coordinates[0];
    s.coordinates[0] = wrapped;
    let mut feedback = model.feedback(&s, &m.hydraulics);
    feedback.pump_work_j = pump;
    feedback.command_heat_j = returned;
    feedback.relief_heat_j = relief_before + relief_after;
    feedback.command_fraction = fraction;
    feedback.motor_impulse_nms = None;
    feedback.unsupported_transfer = input.cut || input.dump;
    let mut external_work = 0.;
    for load in env.loads {
        let (com, q) = before_frames[load.part];
        let body = &model.world.bodies[model.bodies[load.part]];
        let delta_com = sv(body.center_of_mass()) - com;
        let delta_rotation = (sq(*body.rotation()) * q.inverse()).to_scaled_axis();
        let torque = load.torque_nm + (load.point - com).cross(load.force_n);
        external_work += load.force_n.as_dvec3().dot(delta_com.as_dvec3())
            + torque.as_dvec3().dot(delta_rotation.as_dvec3());
    }
    feedback.external_work_j = external_work;
    feedback.passivity_residual_j =
        feedback.kinetic_j + feedback.gravitational_j + feedback.elastic_j
            - (before.kinetic_j + before.gravitational_j + before.elastic_j)
            - pump
            + returned
            + relief_before
            + relief_after
            - external_work;
    let mut normal = 0.;
    let mut count = 0;
    let mut bucket = r::Vector::ZERO;
    let mut ratio = 0f32;
    let mut belt_source = false;
    for pair in model.world.narrow_phase.contact_pairs() {
        let a = &model.world.colliders[pair.collider1];
        let b = &model.world.colliders[pair.collider2];
        let external = a.parent().is_none_or(|p| !model.bodies.contains(&p))
            || b.parent().is_none_or(|p| !model.bodies.contains(&p));
        for manifold in pair.solver_manifolds() {
            for p in &manifold.points {
                let n = p.data.impulse;
                if n > 0. {
                    for tag in [a.user_data, b.user_data] {
                        if (1..=2).contains(&tag) && speed[tag as usize - 1] != 0. {
                            belt_source = true;
                        }
                    }
                    if external {
                        normal += n;
                        count += 1;
                        ratio = ratio.max(p.data.tangent_impulse.norm() / n);
                    }
                    if a.user_data == 3 {
                        bucket -= manifold.data.normal * n;
                    } else if b.user_data == 3 {
                        bucket += manifold.data.normal * n;
                    }
                }
            }
        }
    }
    feedback.ground_normal_n = normal / dt;
    feedback.passivity_inputs_accounted = !belt_source;
    if !s.valid() || !feedback.kinetic_j.is_finite() || !feedback.elastic_j.is_finite() {
        return Err("Non-finite articulated result".into());
    }
    let root = &model.world.bodies[model.bodies[0]];
    let q = sq(*root.rotation());
    let old = m.position;
    m.position = sv(root.translation()) - q * ROOT_COM;
    let f = q * Vec3::Z;
    if f.x * f.x + f.z * f.z > 1e-8 {
        m.heading = f.x.atan2(f.z).rem_euclid(std::f32::consts::TAU);
    }
    m.tracks.tilt = (Quat::from_rotation_y(-m.heading) * q).normalize();
    m.swing = s.coordinates[0].rem_euclid(std::f32::consts::TAU);
    m.boom = -s.coordinates[1];
    m.stick = -s.coordinates[2];
    m.curl = if s.bucket_model.is_shell() {
        s.coordinates[3]
    } else {
        -s.coordinates[3]
    };
    m.tracks.velocity = feedback.linear_momentum_ns / feedback.mass_kg as f32;
    m.tracks.angular_velocity = sv(root.angvel());
    m.distance_m += Vec3::new(m.position.x - old.x, 0., m.position.z - old.z).length();
    m.tracks.feedback = crate::tracks::TrackFeedback {
        contacts: count,
        mass_kg: feedback.mass_kg as f32,
        kinetic_energy_j: feedback.kinetic_j as f32,
        center_of_mass: feedback.center_of_mass,
        normal_force_n: normal / dt,
        bucket_reaction_n: sv(bucket) / dt,
        maximum_friction_ratio: ratio,
        tilt_deg: (q * Vec3::Y)
            .dot(Vec3::Y)
            .clamp(-1., 1.)
            .acos()
            .to_degrees(),
    };
    if let Some(soil) = soil {
        for (clod, handle) in soil.clods.iter_mut().zip(soil_handles) {
            clod.read(&model.world.bodies[handle]);
        }
        let a = soil_before.unwrap();
        let b = soil.totals(env.gravity);
        let mut contacts = 0;
        let mut impulse = DVec3::ZERO;
        let mut fixed_contact = false;
        for pair in model.world.narrow_phase.contact_pairs() {
            let ca = &model.world.colliders[pair.collider1];
            let cb = &model.world.colliders[pair.collider2];
            for manifold in pair.solver_manifolds() {
                for point in &manifold.points {
                    if point.data.impulse <= 0. {
                        continue;
                    }
                    fixed_contact |= ca.parent().is_none_or(|h| model.world.bodies[h].is_fixed())
                        || cb.parent().is_none_or(|h| model.world.bodies[h].is_fixed());
                    if (ca.user_data == 3 && cb.user_data == 20)
                        || (cb.user_data == 3 && ca.user_data == 20)
                    {
                        contacts += 1;
                        // Normal impulse only. Tangential friction is still solved,
                        // but is not mislabeled as an observed total tool wrench.
                        impulse += sv(manifold.data.normal).as_dvec3()
                            * point.data.impulse as f64
                            * if ca.user_data == 3 { -1. } else { 1. };
                    }
                }
            }
        }
        let dp = feedback.linear_momentum_ns.as_dvec3() - before.linear_momentum_ns.as_dvec3();
        let dl = feedback.angular_momentum_nms.as_dvec3() - before.angular_momentum_nms.as_dvec3();
        let mass = a.mass_kg + feedback.mass_kg;
        let center = (before.center_of_mass.as_dvec3() * before.mass_kg
            + feedback.center_of_mass.as_dvec3() * feedback.mass_kg
            + a.first_kg_m
            + b.first_kg_m)
            / (2. * mass);
        soil.feedback = crate::soil::ContactFeedback {
            steps: soil.feedback.steps + 1,
            bucket_contacts: contacts,
            bucket_normal_impulse_ns: impulse,
            machine_momentum_change_ns: dp,
            soil_momentum_change_ns: b.momentum_ns - a.momentum_ns,
            coupled_passivity_residual_j: feedback.passivity_residual_j
                + b.kinetic_j
                + b.gravitational_j
                - a.kinetic_j
                - a.gravitational_j,
            closed_free_system: !fixed_contact && env.loads.is_empty() && !belt_source,
            free_linear_residual_ns: dp + b.momentum_ns
                - a.momentum_ns
                - env.gravity.as_dvec3() * (mass * dt as f64),
            free_angular_residual_nms: dl + b.angular_nms
                - a.angular_nms
                - center.cross(env.gravity.as_dvec3()) * (mass * dt as f64),
        };
    }
    s.pump_energy_j += pump;
    s.relief_heat_j += returned + relief_before + relief_after;
    s.feedback = feedback;
    m.articulation = Some(s);
    Ok(())
}

fn steel_moments(model: crate::bucket::Model) -> crate::payload::MassMoments {
    if model.is_shell() {
        return model.geometry().steel;
    }
    let heel = DVec3::new(0., -0.2375, -0.01);
    let lip = DVec3::new(0., -0.23, 1.065);
    let q = DQuat::from_rotation_arc(DVec3::Z, (lip - heel).normalize());
    crate::payload::MassMoments::cuboid(
        250.,
        (heel + lip) * 0.5 + q * DVec3::Y * 0.025,
        DVec3::new(0.41, 0.025, (lip - heel).length() * 0.5),
        q,
    )
}
fn carried_moments(m: &Excavator) -> crate::payload::MassMoments {
    let s = m.articulation.as_ref().unwrap();
    s.payload_moments.unwrap_or_else(|| {
        if s.bucket_model.is_shell() {
            s.bucket_model
                .geometry()
                .fill_moments(m.bucket.volume(), m.bucket.wet_mass_kg())
        } else {
            steel_moments(s.bucket_model).scaled(m.bucket.wet_mass_kg() / 250.)
        }
    })
}

#[cfg(test)]
mod shell_tests {
    use super::*;
    #[test]
    fn registered_shell_vertices_follow_the_same_hinged_frame_as_the_mesh() {
        for [boom, stick, curl, load] in [[0.8, -1.3, -0.2, 0.], [0.4, -1.8, 0.7, 0.1]] {
            let mut m = Excavator::new(&Site::new());
            m.position = Vec3::new(16., 10., 16.);
            m.heading = 0.7;
            m.swing = 0.4;
            m.boom = boom;
            m.stick = stick;
            m.curl = curl;
            m.tracks.tilt = Quat::from_rotation_x(0.2) * Quat::from_rotation_z(-0.15);
            m.bucket.soil_m3 = load as f64;
            m.enable_articulation().unwrap();
            let state = m.articulation.as_ref().unwrap().clone();
            let model = Model::new(&mut m, &Environment::free(Vec3::ZERO), &state).unwrap();
            let body = &model.world.bodies[model.bodies[4]];
            let collider = &model.world.colliders[body.colliders()[0]];
            let local = collider.position_wrt_parent().unwrap();
            let frame = body.position() * local;
            let compound = collider.shape().as_compound().unwrap();
            let g = m.bucket_model().geometry();
            assert_eq!(compound.shapes().len(), g.parts.len());
            for ((pose, shape), part) in compound.shapes().iter().zip(&g.parts) {
                let hull = shape.as_convex_polyhedron().unwrap();
                for vertex in &part.vertices {
                    let actual = frame * (*pose * rv(*vertex));
                    let expected = m.arm().wrist + m.bucket_rotation() * *vertex;
                    assert!(sv(actual).distance(expected) < 0.00001);
                    assert!(hull
                        .points()
                        .iter()
                        .any(|p| sv(*p).distance(*vertex) < 1e-7));
                }
            }
            let expected = g.steel.combine(carried_moments(&m));
            assert!((body.mass() as f64 - expected.mass_kg).abs() < 0.001);
            let com = m.arm().wrist + m.bucket_rotation() * expected.center.as_vec3();
            assert!(sv(body.center_of_mass()).distance(com) < 0.00001);
            let registered = body.mass_properties().effective_angular_inertia();
            let q = m.bucket_rotation().as_dquat();
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                let actual = sv(registered * rv(axis)).as_dvec3();
                let expected = q * (expected.inertia() * (q.inverse() * axis.as_dvec3()));
                assert!((actual - expected).length() < 0.0002);
            }
        }
    }
}

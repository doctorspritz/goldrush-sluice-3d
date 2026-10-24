//! Rigid machine/terrain contact, separate from the water and grain solvers.
//! Joints remain prescribed, force-limited actuators; this is not yet a full
//! multibody hydraulic or deformable-soil model. All lengths/masses use SI.
use crate::{
    machine::MachineInput,
    site::{CELL, WIDTH},
    Excavator, Site, Vec3,
};
use glam::Quat;
use rapier3d::prelude as r;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const TRACK_WIDTH: f32 = 0.45;
pub const TRACK_GAUGE: f32 = 1.85;
pub const TRACK_LENGTH: f32 = 2.88;
pub const TRAVEL_SPEED: f32 = 5.1 / 3.6;
const PATCH: usize = 65;

#[derive(Clone, Debug)]
pub(crate) struct Ground {
    pub(crate) origin: [usize; 2],
    heights: Vec<f32>,
    pub(crate) shape: r::SharedShape,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct TrackFeedback {
    pub contacts: usize,
    pub mass_kg: f32,
    pub kinetic_energy_j: f32,
    pub center_of_mass: Vec3,
    pub normal_force_n: f32,
    pub bucket_reaction_n: Vec3,
    pub maximum_friction_ratio: f32,
    pub tilt_deg: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrackMotion {
    /// Chassis tilt relative to its horizontal heading. The public heading
    /// remains compatible with operator steering, placement and old saves.
    pub tilt: Quat,
    /// World-space centre-of-mass velocity, not chassis-origin velocity.
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    /// Declared rigid-ground Coulomb coefficient; not calibrated wet soil.
    pub friction: f32,
    pub mechanical_hz: u16,
    pub feedback: TrackFeedback,
    // Immutable geometry is only a cache. Rebuilding it changes no physics.
    #[serde(skip)]
    ground: Option<Arc<Ground>>,
}
impl Default for TrackMotion {
    fn default() -> Self {
        Self {
            tilt: Quat::IDENTITY,
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            friction: 0.65,
            mechanical_hz: 240,
            feedback: TrackFeedback::default(),
            ground: None,
        }
    }
}
impl TrackMotion {
    pub fn valid(&self) -> bool {
        self.tilt.is_finite()
            && (self.tilt.length_squared() - 1.).abs() < 0.0001
            && self.velocity.is_finite()
            && self.velocity.length() < 100.
            && self.angular_velocity.is_finite()
            && self.angular_velocity.length() < 100.
            && self.friction.is_finite()
            && (0. ..=2.).contains(&self.friction)
            && (60..=960).contains(&self.mechanical_hz)
            && self.feedback.bucket_reaction_n.is_finite()
            && self.feedback.center_of_mass.is_finite()
            && [
                self.feedback.mass_kg,
                self.feedback.kinetic_energy_j,
                self.feedback.normal_force_n,
                self.feedback.maximum_friction_ratio,
                self.feedback.tilt_deg,
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.)
    }
    pub(crate) fn carries_motion(&self) -> bool {
        self.tilt != Quat::IDENTITY
            || self.velocity != Vec3::ZERO
            || self.angular_velocity != Vec3::ZERO
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Obstacle {
    pub center: Vec3,
    pub rotation: Quat,
    pub half_extents: Vec3,
}

pub(crate) fn rv(v: Vec3) -> r::Vector {
    r::Vector::from_array(v.to_array())
}
pub(crate) fn sv(v: r::Vector) -> Vec3 {
    Vec3::from_array(v.to_array())
}
pub(crate) fn rr(q: Quat) -> r::Rotation {
    r::Rotation::from_xyzw(q.x, q.y, q.z, q.w)
}
pub(crate) fn sq(q: r::Rotation) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}
pub(crate) fn pose(p: Vec3, q: Quat) -> r::Pose {
    r::Pose::from_parts(rv(p), rr(q))
}

pub(crate) struct Belts {
    pub(crate) forward: r::Vector,
    pub(crate) speed: [f32; 2],
}
impl r::PhysicsHooks for Belts {
    fn modify_solver_contacts(&self, ctx: &mut r::ContactModificationContext) {
        let a = ctx.colliders[ctx.collider1].user_data;
        let b = ctx.colliders[ctx.collider2].user_data;
        let (tag, sign) = if (1..=2).contains(&a) {
            (a, -1.)
        } else if (1..=2).contains(&b) {
            (b, 1.)
        } else {
            return;
        };
        if let Some(m) = ctx.rigid_mut() {
            let tangent = self.forward - *m.normal * self.forward.dot(*m.normal);
            // Keep projected belt speed: normal motion cannot supply traction.
            for c in m.solver_contacts.iter_mut() {
                c.tangent_velocity = tangent * (self.speed[tag as usize - 1] * sign);
            }
        }
    }
}

impl Excavator {
    pub fn chassis_rotation(&self) -> Quat {
        Quat::from_rotation_y(self.heading) * self.tracks.tilt
    }
    pub fn upper_rotation(&self) -> Quat {
        self.chassis_rotation() * Quat::from_rotation_y(self.swing)
    }
    pub fn up(&self) -> Vec3 {
        self.chassis_rotation() * Vec3::Y
    }

    pub(crate) fn ground(&mut self, site: &Site) -> Arc<Ground> {
        let origin = [self.position.x, self.position.z].map(|v| {
            ((v / CELL).floor() as i32 / 16 * 16 - 32).clamp(0, (WIDTH - PATCH) as i32) as usize
        });
        // Parry's Array2 is column-major: rows are Z, columns are X.
        let mut heights = Vec::with_capacity(PATCH * PATCH);
        for x in 0..PATCH {
            for z in 0..PATCH {
                heights.push(
                    site.height((origin[0] + x) as f32 * CELL, (origin[1] + z) as f32 * CELL),
                );
            }
        }
        if let Some(g) = &self.tracks.ground {
            if g.origin == origin && g.heights == heights {
                return g.clone();
            }
        }
        let scale = r::Vector::new((PATCH - 1) as f32 * CELL, 1., (PATCH - 1) as f32 * CELL);
        let shape = r::SharedShape::heightfield_with_flags(
            r::Array2::new(PATCH, PATCH, heights.clone()),
            scale,
            r::HeightFieldFlags::FIX_INTERNAL_EDGES,
        );
        let g = Arc::new(Ground {
            origin,
            heights,
            shape,
        });
        self.tracks.ground = Some(g.clone());
        g
    }

    /// Deterministic at mechanical-step boundaries: only pose and velocities
    /// carry dynamics between steps. Contact warm-starts are not hidden save
    /// state. The shared terrain shape avoids rebuilding unchanged geometry.
    pub(crate) fn advance_tracks(
        &mut self,
        input: MachineInput,
        site: &Site,
        dt: f32,
        obstacles: &[Obstacle],
    ) {
        let ground = self.ground(site);
        let q = self.chassis_rotation();
        let mut world = r::PhysicsWorld::new();
        let steps = 1;
        let sub_dt = dt / steps as f32;
        world.integration_parameters = r::IntegrationParameters {
            dt: sub_dt,
            // TGS stages integrate time; PGS iterations converge the two belts'
            // simultaneous friction constraints before each integration stage.
            num_solver_iterations: 4,
            num_internal_pgs_iterations: 16,
            friction_model: r::FrictionModel::Coulomb,
            contact_clustering: true,
            contact_recycling: false,
            normalized_allowed_linear_error: 0.0001,
            ..Default::default()
        };
        let terrain_center = Vec3::new(
            (ground.origin[0] as f32 + (PATCH - 1) as f32 * 0.5) * CELL,
            0.,
            (ground.origin[1] as f32 + (PATCH - 1) as f32 * 0.5) * CELL,
        );
        world.insert(
            r::RigidBodyBuilder::fixed(),
            r::ColliderBuilder::new(ground.shape.clone())
                .translation(rv(terrain_center))
                .friction(1.)
                .restitution(0.),
        );
        let chassis = world.insert_body(
            r::RigidBodyBuilder::dynamic()
                .pose(pose(self.position, q))
                .linvel(rv(self.tracks.velocity))
                .angvel(rv(self.tracks.angular_velocity))
                .can_sleep(false)
                .ccd_enabled(true),
        );
        let mut attach = |builder: r::ColliderBuilder| {
            world.insert_collider(
                builder
                    .friction(self.tracks.friction)
                    .friction_combine_rule(r::CoefficientCombineRule::Min)
                    .restitution(0.),
                Some(chassis),
            )
        };
        for (tag, side) in [(1, -1.), (2, 1.)] {
            attach(
                r::ColliderBuilder::cuboid(TRACK_WIDTH * 0.5, 0.325, TRACK_LENGTH * 0.5)
                    .translation(r::Vector::new(side * TRACK_GAUGE * 0.5, 0.325, 0.))
                    .mass(1300.)
                    .user_data(tag)
                    .active_hooks(r::ActiveHooks::MODIFY_SOLVER_CONTACTS),
            );
        }
        attach(
            r::ColliderBuilder::cuboid(0.85, 0.175, 0.95)
                .translation(r::Vector::new(0., 0.70, 0.))
                .mass(700.),
        );
        let upper = Quat::from_rotation_y(self.swing);
        attach(
            r::ColliderBuilder::cuboid(1.025, 0.30, 1.15)
                .position(pose(upper * Vec3::new(0., 1.10, -0.1), upper))
                .mass(3000.),
        );
        // Locked-link mass distribution is explicit, not manufacturer data.
        // Updating prescribed joints does not yet resolve their recoil/work.
        let arm = self.arm();
        let local = |v: Vec3| q.inverse() * (v - self.position);
        attach(
            r::ColliderBuilder::capsule_from_endpoints(
                rv(local(arm.shoulder)),
                rv(local(arm.elbow)),
                0.17,
            )
            .mass(1300.),
        );
        attach(
            r::ColliderBuilder::capsule_from_endpoints(
                rv(local(arm.elbow)),
                rv(local(arm.wrist)),
                0.13,
            )
            .mass(840.),
        );
        let bucket_q = upper * Quat::from_rotation_x(-arm.angle);
        let wrist = local(arm.wrist);
        // Match the actual sloped floor between heel (y=-.2375,z=-.01)
        // and tooth (y=-.23,z=1.065), not a box extending below that cut.
        let heel = Vec3::new(0., -0.2375, -0.01);
        let lip = Vec3::new(0., -0.23, 1.065);
        let floor_rotation = Quat::from_rotation_arc(Vec3::Z, (lip - heel).normalize());
        attach(
            r::ColliderBuilder::cuboid(0.41, 0.025, (lip - heel).length() * 0.5)
                .position(pose(
                    wrist + bucket_q * ((heel + lip) * 0.5 + floor_rotation * Vec3::Y * 0.025),
                    bucket_q * floor_rotation,
                ))
                .mass(250. + self.bucket.wet_mass_kg() as f32)
                .user_data(3),
        );
        // Inverted machines land on the physical cab, not their track samples.
        attach(
            r::ColliderBuilder::cuboid(0.53, 0.68, 0.775)
                .position(pose(upper * Vec3::new(-0.64, 2.10, 0.22), upper))
                .mass(0.),
        );
        for o in obstacles {
            world.insert(
                r::RigidBodyBuilder::fixed(),
                r::ColliderBuilder::cuboid(o.half_extents.x, o.half_extents.y, o.half_extents.z)
                    .position(pose(o.center, o.rotation))
                    .friction(0.65),
            );
        }
        // Finite claim boundary; no pose clamping or a ground-height teleport.
        let edge = (WIDTH - 1) as f32 * CELL;
        for (center, half) in [
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
                r::ColliderBuilder::cuboid(half.x, half.y, half.z).translation(rv(center)),
            );
        }
        world.bodies[chassis].recompute_mass_properties_from_colliders(&world.colliders);
        let powered = self.engine_on && self.fuel_l > 0.;
        let drive = if powered {
            input.drive.clamp(-1., 1.) * TRAVEL_SPEED
        } else {
            0.
        };
        let steer = if powered {
            input.steer.clamp(-1., 1.) * 0.42
        } else {
            0.
        };
        let speed = [
            (drive + steer).clamp(-TRAVEL_SPEED, TRAVEL_SPEED),
            (drive - steer).clamp(-TRAVEL_SPEED, TRAVEL_SPEED),
        ];
        let mut normal_impulse = 0.;
        let mut bucket_impulse = r::Vector::ZERO;
        let mut maximum_ratio = 0f32;
        let mut contacts = 0;
        for _ in 0..steps {
            let forward = *world.bodies[chassis].rotation() * r::Vector::Z;
            world.step_with_events(&Belts { forward, speed }, &());
            contacts = 0;
            for pair in world.narrow_phase.contact_pairs() {
                if pair.collider1 == pair.collider2 {
                    continue;
                }
                for manifold in pair.solver_manifolds() {
                    for p in &manifold.points {
                        let n = p.data.impulse;
                        if n > 0. {
                            contacts += 1;
                            normal_impulse += n;
                            if world.colliders[pair.collider1].user_data == 3 {
                                bucket_impulse -= manifold.data.normal * n;
                            } else if world.colliders[pair.collider2].user_data == 3 {
                                bucket_impulse += manifold.data.normal * n;
                            }
                            let ratio = p.data.tangent_impulse.norm() / n;
                            maximum_ratio = maximum_ratio.max(ratio);
                        }
                    }
                }
            }
        }
        let body = &world.bodies[chassis];
        let old = self.position;
        self.position = sv(body.translation());
        let rotation = sq(*body.rotation());
        let forward = rotation * Vec3::Z;
        if forward.x * forward.x + forward.z * forward.z > 1e-8 {
            self.heading = forward.x.atan2(forward.z).rem_euclid(std::f32::consts::TAU);
        }
        self.tracks.tilt = (Quat::from_rotation_y(-self.heading) * rotation).normalize();
        self.tracks.velocity = sv(body.linvel());
        self.tracks.angular_velocity = sv(body.angvel());
        self.distance_m += Vec3::new(self.position.x - old.x, 0., self.position.z - old.z).length();
        self.tracks.feedback = TrackFeedback {
            contacts,
            mass_kg: body.mass(),
            kinetic_energy_j: body.kinetic_energy(),
            center_of_mass: sv(body.center_of_mass()),
            normal_force_n: normal_impulse / dt,
            bucket_reaction_n: sv(bucket_impulse) / dt,
            maximum_friction_ratio: maximum_ratio,
            tilt_deg: (rotation * Vec3::Y)
                .dot(Vec3::Y)
                .clamp(-1., 1.)
                .acos()
                .to_degrees(),
        };
    }
}

//! Fixed-pose rigid Coulomb block work. Impulse rows are paired with the
//! common block midpoint, including normal/tangent cross terms; these are
//! net-block partitions, never sequential per-row energy claims.
use super::*;
use crate::dynamics::IntegrationParameters;
use crate::dynamics::solver::contact_constraint::{
    ContactWithCoulombFriction, ContactWithCoulombFrictionBuilder,
};
use crate::dynamics::solver::manifold_store::ManifoldStore;
use crate::geometry::{ColliderHandle, ColliderSet};
use crate::math::{Pose, Real, SIMD_WIDTH, SimdReal};
use crate::utils::{CrossProduct, DotProduct};
use simba::simd::SimdValue;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub(super) struct ColliderSnapshot {
    parent: Option<RigidBodyHandle>,
    local: Option<Pose>,
    world: Pose,
}
pub(super) fn capture_colliders(
    colliders: &ColliderSet,
) -> HashMap<ColliderHandle, ColliderSnapshot> {
    colliders
        .iter()
        .map(|(h, c)| {
            (h, ColliderSnapshot {
                parent: c.parent(),
                local: c.position_wrt_parent().copied(),
                world: *c.position(),
            })
        })
        .collect()
}

/// The coefficients are promoted from the actual f32 solver/composed pose,
/// without normalizing or silently replacing its quaternion.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct PoseWork {
    translation: [f64; 3],
    rotation_xyzw: [f64; 4],
}
fn pose_work(p: Pose) -> PoseWork {
    PoseWork {
        translation: p.translation.to_array().map(f64::from),
        rotation_xyzw: p.rotation.to_array().map(f64::from),
    }
}
fn vector(v: Vector) -> [f64; 3] {
    v.to_array().map(f64::from)
}
fn lane_vector(v: <SimdReal as crate::utils::ScalarType>::Vector, lane: usize) -> Vector {
    Vector::from_array(std::array::from_fn(|i| v[i].extract(lane)))
}
fn six(v: crate::dynamics::solver::SolverVel<Real>) -> VectorN<f64> {
    VectorN::from_iterator(
        6,
        v.linear
            .to_array()
            .into_iter()
            .chain(v.angular.to_array())
            .map(f64::from),
    )
}
fn six_array(v: &VectorN<f64>) -> [f64; 6] {
    std::array::from_fn(|i| v[i])
}
fn wrench(direction: Vector, torque: Vector) -> VectorN<f64> {
    VectorN::from_iterator(
        6,
        direction
            .to_array()
            .into_iter()
            .chain(torque.to_array())
            .map(f64::from),
    )
}
fn kinetic_change(before: &VectorN<f64>, after: &VectorN<f64>, mass: &DMatrix<f64>) -> f64 {
    0.5 * (after - before).dot(&(mass * (after + before)))
}
#[derive(Debug, serde::Serialize)]
struct RigidSideWork {
    body_handle: Option<(u32, u32)>,
    solver_body_id: u32,
    dynamic: bool,
    sleeping: bool,
    /// Sentinel poses are the identity actually used by the contact solver.
    solver_center_of_mass_pose: PoseWork,
    physical_center_of_mass_pose: Option<PoseWork>,
    physical_pose_source: &'static str,
    body_origin_pose: Option<PoseWork>,
    local_center_of_mass_m: [f64; 3],
    mass_kg: f64,
    local_principal_inertia_kg_m2: [f64; 3],
    local_principal_frame_xyzw: [f64; 4],
    collider_parent_handle: Option<(u32, u32)>,
    collider_local_pose: Option<PoseWork>,
    captured_collider_world_pose: Option<PoseWork>,
    collider_pose_source: &'static str,
    before_velocity: [f64; 6],
    after_velocity: [f64; 6],
    stored_response_delta_velocity: [f64; 6],
    actual_delta_velocity: [f64; 6],
    physical_kinetic_change_j: f64,
}
#[derive(Debug, serde::Serialize)]
struct RigidPointWork {
    contact_id: u8,
    /// Original localized manifold witnesses at the current solver poses.
    world_anchors_m: [[f64; 3]; 2],
    /// Substep tracking uses builder anchors, with conveyor movement on side 1.
    tracking_world_anchors_m: [[f64; 3]; 2],
    tracking_local_anchors_m: [[f64; 3]; 2],
    tracking_base_distance_m: f64,
    restitution_seed_m_s: f64,
    tracking_distance_m: f64,
    tangent_velocity_m_s: [f64; 3],
    tangent_tracking_offset_m: [f64; 3],
    frozen_world_arms_m: [[f64; 3]; 2],
    force_direction1: [f64; 3],
    tangent_directions1: [[f64; 3]; 2],
    normal_torque_directions: [[f64; 3]; 2],
    tangent_torque_directions: [[[f64; 3]; 2]; 2],
    normal_inverse_torque_directions: [[f64; 3]; 2],
    tangent_inverse_torque_directions: [[[f64; 3]; 2]; 2],
    inverse_mass_diagonal: [[f64; 3]; 2],
    normal_impulse_before_ns: f64,
    normal_impulse_after_ns: f64,
    tangent_impulse_before_ns: [f64; 2],
    tangent_impulse_after_ns: [f64; 2],
    normal_delta_ns: f64,
    tangent_delta_ns: [f64; 2],
    normal_midpoint_work_j: f64,
    tangent_midpoint_work_j: f64,
    prescribed_midpoint_work_j: f64,
    normal_bias_pairing_j: f64,
    tangent_bias_pairing_j: f64,
    normal_rhs_m_s: Option<f64>,
    normal_stored_rhs_without_bias_m_s: f64,
    tangent_rhs_m_s: Option<[f64; 2]>,
    tangent_rhs_without_bias_m_s: [f64; 2],
    normal_inverse_effective_mass: f64,
    normal_cfm_factor: f64,
    tangent_effective_mass_entries: [f64; 3],
    maximum_dissipation: bool,
}
/// One real (non-padding) lane/manifold of a fixed-pose Coulomb operation.
#[derive(Debug, serde::Serialize)]
pub struct RigidContactWork {
    record_kind: &'static str,
    manifold: [u32; 2],
    chunk: usize,
    lane: usize,
    clustered: bool,
    advisory_subshapes: [u32; 2],
    collider_handles: [(u32, u32); 2],
    collider_world_poses: [Option<PoseWork>; 2],
    relative_dominance: i16,
    stage: &'static str,
    unbiased: bool,
    solve_friction: bool,
    solved_time_s: f32,
    inverse_substep_dt_s: f64,
    normal_bias_gain_s_inv: f64,
    max_corrective_velocity_m_s: f64,
    friction_coefficient: f64,
    constraint_cfm_factor: f64,
    physical_kinetic_change_j: f64,
    normal_midpoint_work_j: f64,
    tangent_midpoint_work_j: f64,
    prescribed_midpoint_work_j: f64,
    physical_minus_impulse_work_j: f64,
    max_impulse_velocity_residual: f64,
    sides: Vec<RigidSideWork>,
    points: Vec<RigidPointWork>,
}
struct RigidSide {
    data: RigidSideWork,
    mass: DMatrix<f64>,
    velocity: VectorN<f64>,
}
struct Lane {
    row: RigidContactWork,
    sides: [RigidSide; 2],
}
pub(crate) struct Snapshot {
    old: ContactWithCoulombFriction<SimdReal>,
    lanes: Vec<Lane>,
}

fn physical_pose(
    handle: RigidBodyHandle,
    island: &[RigidBodyHandle],
    solver: &SolverBodies,
    bodies: &RigidBodySet,
    mb: &MultibodyJointSet,
) -> (Pose, &'static str) {
    let body = &bodies[handle];
    if let Some(link) = mb.rigid_body_link(handle) {
        let link = mb[link.multibody].link(link.id).unwrap();
        return (
            link.local_to_world
                .prepend_translation(body.local_center_of_mass()),
            "current_multibody_link",
        );
    }
    let id = body.effective_active_set_offset() as usize;
    if island.get(id) == Some(&handle) {
        assert_eq!(bodies[handle].ids.active_set_id as usize, id);
        return (solver.poses[id].pose(), "current_solver_com");
    }
    (
        body.position()
            .prepend_translation(body.local_center_of_mass()),
        "body_pose_outside_island",
    )
}
fn side(
    handle: Option<RigidBodyHandle>,
    collider: ColliderHandle,
    id: u32,
    island: &[RigidBodyHandle],
    solver: &SolverBodies,
    bodies: &RigidBodySet,
    mb: &MultibodyJointSet,
) -> (RigidSide, Option<PoseWork>) {
    if id != u32::MAX {
        let handle = handle.expect("rigid solver slot requires actual body handle");
        assert_eq!(
            island.get(id as usize),
            Some(&handle),
            "rigid audit solver/body mapping"
        );
        assert_eq!(bodies[handle].ids.active_set_id, id);
        assert!(
            mb.rigid_body_link(handle).is_none(),
            "Coulomb lane cannot use multibody solver slots"
        );
        assert!((id as usize) < solver.len());
    }
    let solver_pose = if id == u32::MAX {
        Pose::IDENTITY
    } else {
        solver.poses[id as usize].pose()
    };
    let body = handle.map(|h| &bodies[h]);
    let (com, source) = handle
        .map(|h| physical_pose(h, island, solver, bodies, mb))
        .map_or((None, "no_body"), |(p, s)| (Some(p), s));
    let local_com = body.map_or(Vector::ZERO, |b| b.local_center_of_mass());
    let origin = com.map(|p| Pose::from_parts(p.translation - p.rotation * local_com, p.rotation));
    let captured = COLLIDERS.with(|c| {
        c.borrow()
            .as_ref()
            .and_then(|all| all.get(&collider))
            .copied()
    });
    if let Some(c) = captured {
        assert_eq!(
            c.parent, handle,
            "collider snapshot parent differs from manifold"
        );
    }
    let (collider_world, collider_source) = match captured {
        Some(c) => match c.parent {
            Some(_) => (
                Some(origin.expect("parented collider body pose") * c.local.unwrap()),
                "current_solver_com_body_origin_local_composition",
            ),
            None => (Some(c.world), "captured_standalone_world"),
        },
        None => (None, "not_captured"),
    };
    let velocity = six(solver.get_vel(id));
    let dynamic = body.is_some_and(|b| b.is_dynamic());
    let mass = if dynamic && id != u32::MAX {
        spatial_mass(body.unwrap(), solver_pose.rotation)
    } else {
        DMatrix::zeros(6, 6)
    };
    let local = body.map(|b| &b.mass_properties().local_mprops);
    let data = RigidSideWork {
        body_handle: handle.map(|h| h.into_raw_parts()),
        solver_body_id: id,
        dynamic,
        sleeping: body.is_some_and(|b| b.is_sleeping()),
        solver_center_of_mass_pose: pose_work(solver_pose),
        physical_center_of_mass_pose: com.map(pose_work),
        physical_pose_source: source,
        body_origin_pose: origin.map(pose_work),
        local_center_of_mass_m: vector(local_com),
        mass_kg: body.map_or(0., |b| f64::from(b.mass())),
        local_principal_inertia_kg_m2: local
            .map_or([0.; 3], |p| p.principal_inertia().to_array().map(f64::from)),
        local_principal_frame_xyzw: local.map_or([0., 0., 0., 1.], |p| {
            p.principal_inertia_local_frame.to_array().map(f64::from)
        }),
        collider_parent_handle: captured.and_then(|c| c.parent).map(|h| h.into_raw_parts()),
        collider_local_pose: captured.and_then(|c| c.local).map(pose_work),
        captured_collider_world_pose: captured.map(|c| pose_work(c.world)),
        collider_pose_source: collider_source,
        before_velocity: six_array(&velocity),
        after_velocity: [0.; 6],
        stored_response_delta_velocity: [0.; 6],
        actual_delta_velocity: [0.; 6],
        physical_kinetic_change_j: 0.,
    };
    (
        RigidSide {
            data,
            mass,
            velocity,
        },
        collider_world.map(pose_work),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn before(
    c: &ContactWithCoulombFriction<SimdReal>,
    builder: &ContactWithCoulombFrictionBuilder,
    solver: &SolverBodies,
    island: &[RigidBodyHandle],
    bodies: &RigidBodySet,
    mb: &MultibodyJointSet,
    store: &ManifoldStore<'_>,
    params: &IntegrationParameters,
    stage: &'static str,
    unbiased: bool,
    time: f32,
    chunk: usize,
    solve_friction: bool,
) -> Option<Snapshot> {
    if !rigid_enabled() {
        return None;
    }
    let mut lanes = Vec::new();
    let mut writable_lanes = HashMap::new();
    let infos = builder.audit_infos();
    let poses = [
        solver.gather_transforms(c.solver_vel1),
        solver.gather_transforms(c.solver_vel2),
    ];
    for lane in 0..SIMD_WIDTH {
        let id = c.manifold_id[lane];
        if id.is_padding() {
            continue;
        }
        let manifold = store.get(id);
        let (colliders, clustered) = store.audit_pair_metadata(id);
        let handles = [manifold.data.rigid_body1, manifold.data.rigid_body2];
        let ids = [c.solver_vel1[lane], c.solver_vel2[lane]];
        let [(first, world1), (second, world2)] = std::array::from_fn(|i| {
            side(handles[i], colliders[i], ids[i], island, solver, bodies, mb)
        });
        // Same-color lanes may share prescribed kinematic/world sides, but
        // writable dynamic slots must be disjoint for lane-local energy deltas.
        for (side_index, observed) in [&first, &second].into_iter().enumerate() {
            if !observed.data.dynamic || ids[side_index] == u32::MAX {
                continue;
            }
            let im = lane_vector(if side_index == 0 { c.im1 } else { c.im2 }, lane);
            let has_response = im.length_squared() != 0.
                || (0..c.num_contacts as usize).any(|p| {
                    if c.manifold_contact_id[p][lane] == u8::MAX {
                        return false;
                    }
                    let np = c.normal_part[p];
                    let tp = c.tangent_part[p];
                    let normal = if side_index == 0 {
                        np.ii_torque_dir1
                    } else {
                        np.ii_torque_dir2
                    };
                    let tangent = if side_index == 0 {
                        tp.ii_torque_dir1
                    } else {
                        tp.ii_torque_dir2
                    };
                    lane_vector(normal, lane).length_squared() != 0.
                        || tangent
                            .into_iter()
                            .any(|v| lane_vector(v, lane).length_squared() != 0.)
                });
            if has_response {
                let previous = writable_lanes.insert(ids[side_index], lane);
                assert!(
                    previous.is_none_or(|old| old == lane),
                    "rigid audit writable body shared between SIMD lanes"
                );
            }
        }
        let mut points = Vec::new();
        let normal = lane_vector(c.dir1, lane);
        let tangent1 = lane_vector(c.tangent1, lane);
        let tangents = [tangent1, lane_vector(c.dir1.gcross(c.tangent1), lane)];
        for point in 0..c.num_contacts as usize {
            let cid = c.manifold_contact_id[point][lane];
            if cid == u8::MAX {
                continue;
            }
            let sc = manifold
                .data
                .solver_contacts
                .iter()
                .find(|sc| sc.contact_indices()[0] == u32::from(cid))
                .expect("active Coulomb point has original manifold solver contact");
            let world_anchor = [sc.anchor1, sc.anchor2];
            let world_anchor: [_; 2] =
                std::array::from_fn(|i| solver.get_pose(ids[i]).transform_point(world_anchor[i]));
            let info = infos[point];
            let local = [
                lane_vector(info.local_p1, lane),
                lane_vector(info.local_p2, lane),
            ];
            let tangent_vel = lane_vector(info.tangent_vel, lane);
            let offset_wide = info.tangent_vel * SimdReal::splat(time);
            let offset = lane_vector(offset_wide, lane);
            let tracked_wide = [
                poses[0].transform_point(info.local_p1) + offset_wide,
                poses[1].transform_point(info.local_p2),
            ];
            let tracked = tracked_wide.map(|p| lane_vector(p, lane));
            let tracking_distance = info.dist + (tracked_wide[0] - tracked_wide[1]).gdot(c.dir1);
            let np = c.normal_part[point];
            let tp = c.tangent_part[point];
            points.push(RigidPointWork {
                contact_id: cid,
                world_anchors_m: world_anchor.map(vector),
                tracking_world_anchors_m: tracked.map(vector),
                tracking_local_anchors_m: local.map(vector),
                tracking_base_distance_m: f64::from(info.dist.extract(lane)),
                restitution_seed_m_s: f64::from(info.restitution_seed.extract(lane)),
                tracking_distance_m: f64::from(tracking_distance.extract(lane)),
                tangent_velocity_m_s: vector(tangent_vel),
                tangent_tracking_offset_m: vector(offset),
                frozen_world_arms_m: [
                    manifold.points[cid as usize].data.solver_dp1,
                    manifold.points[cid as usize].data.solver_dp2,
                ]
                .map(vector),
                force_direction1: vector(normal),
                tangent_directions1: tangents.map(vector),
                normal_torque_directions: [
                    lane_vector(np.torque_dir1, lane),
                    lane_vector(np.torque_dir2, lane),
                ]
                .map(vector),
                tangent_torque_directions: [
                    [
                        lane_vector(tp.torque_dir1[0], lane),
                        lane_vector(tp.torque_dir1[1], lane),
                    ]
                    .map(vector),
                    [
                        lane_vector(tp.torque_dir2[0], lane),
                        lane_vector(tp.torque_dir2[1], lane),
                    ]
                    .map(vector),
                ],
                normal_inverse_torque_directions: [
                    lane_vector(np.ii_torque_dir1, lane),
                    lane_vector(np.ii_torque_dir2, lane),
                ]
                .map(vector),
                tangent_inverse_torque_directions: [
                    [
                        lane_vector(tp.ii_torque_dir1[0], lane),
                        lane_vector(tp.ii_torque_dir1[1], lane),
                    ]
                    .map(vector),
                    [
                        lane_vector(tp.ii_torque_dir2[0], lane),
                        lane_vector(tp.ii_torque_dir2[1], lane),
                    ]
                    .map(vector),
                ],
                inverse_mass_diagonal: [lane_vector(c.im1, lane), lane_vector(c.im2, lane)]
                    .map(vector),
                normal_impulse_before_ns: f64::from(np.impulse.extract(lane)),
                normal_impulse_after_ns: 0.,
                tangent_impulse_before_ns: std::array::from_fn(|i| {
                    f64::from(tp.impulse[i].extract(lane))
                }),
                tangent_impulse_after_ns: [0.; 2],
                normal_delta_ns: 0.,
                tangent_delta_ns: [0.; 2],
                normal_midpoint_work_j: 0.,
                tangent_midpoint_work_j: 0.,
                prescribed_midpoint_work_j: 0.,
                normal_bias_pairing_j: 0.,
                tangent_bias_pairing_j: 0.,
                normal_rhs_m_s: (stage != "warmstart").then(|| {
                    f64::from(
                        if stage == "projection" {
                            np.rhs_wo_bias
                        } else {
                            np.rhs
                        }
                        .extract(lane),
                    )
                }),
                normal_stored_rhs_without_bias_m_s: f64::from(np.rhs_wo_bias.extract(lane)),
                tangent_rhs_m_s: (stage != "warmstart").then(|| {
                    std::array::from_fn(|i| {
                        f64::from(
                            if stage == "projection" {
                                tp.rhs_wo_bias[i]
                            } else {
                                tp.rhs[i]
                            }
                            .extract(lane),
                        )
                    })
                }),
                tangent_rhs_without_bias_m_s: std::array::from_fn(|i| {
                    f64::from(tp.rhs_wo_bias[i].extract(lane))
                }),
                normal_inverse_effective_mass: f64::from(np.r.extract(lane)),
                normal_cfm_factor: if stage == "projection" {
                    1.
                } else {
                    f64::from(np.cfm_factor.extract(lane))
                },
                tangent_effective_mass_entries: std::array::from_fn(|i| {
                    f64::from(tp.r[i].extract(lane))
                }),
                maximum_dissipation: tp.maximum_dissipation,
            });
        }
        let row = RigidContactWork {
            record_kind: "rigid_coulomb",
            manifold: [id.edge, id.manifold],
            chunk,
            lane,
            clustered,
            advisory_subshapes: [manifold.subshape1, manifold.subshape2],
            collider_handles: colliders.map(|h| h.into_raw_parts()),
            collider_world_poses: [world1, world2],
            relative_dominance: manifold.data.relative_dominance,
            stage,
            unbiased,
            solve_friction,
            solved_time_s: time,
            inverse_substep_dt_s: f64::from(params.inv_dt()),
            normal_bias_gain_s_inv: f64::from(if ids.contains(&u32::MAX) {
                params.static_contact_softness.erp_inv_dt(params.dt)
            } else {
                params.contact_softness.erp_inv_dt(params.dt)
            }),
            max_corrective_velocity_m_s: f64::from(params.max_corrective_velocity()),
            friction_coefficient: f64::from(c.limit.extract(lane)),
            constraint_cfm_factor: if stage == "projection" {
                1.
            } else {
                f64::from(c.cfm_factor.extract(lane))
            },
            physical_kinetic_change_j: 0.,
            normal_midpoint_work_j: 0.,
            tangent_midpoint_work_j: 0.,
            prescribed_midpoint_work_j: 0.,
            physical_minus_impulse_work_j: 0.,
            max_impulse_velocity_residual: 0.,
            sides: Vec::new(),
            points,
        };
        lanes.push(Lane {
            row,
            sides: [first, second],
        });
    }
    Some(Snapshot { old: *c, lanes })
}

pub(crate) fn after(
    snapshot: Option<Snapshot>,
    c: &ContactWithCoulombFriction<SimdReal>,
    solver: &SolverBodies,
) {
    let Some(s) = snapshot else {
        return;
    };
    for mut lane in s.lanes {
        let k = lane.row.lane;
        let end: [_; 2] =
            std::array::from_fn(|i| six(solver.get_vel(lane.sides[i].data.solver_body_id)));
        let mid: [_; 2] = std::array::from_fn(|i| (&lane.sides[i].velocity + &end[i]) * 0.5);
        let mut response = [VectorN::zeros(6), VectorN::zeros(6)];
        let normal = lane_vector(c.dir1, k);
        let tangent1 = lane_vector(c.tangent1, k);
        let tangents = [tangent1, lane_vector(c.dir1.gcross(c.tangent1), k)];
        for p in &mut lane.row.points {
            let point = (0..c.num_contacts as usize)
                .find(|i| c.manifold_contact_id[*i][k] == p.contact_id)
                .unwrap();
            let np = &c.normal_part[point];
            let tp = &c.tangent_part[point];
            p.normal_impulse_after_ns = f64::from(np.impulse.extract(k));
            p.tangent_impulse_after_ns =
                std::array::from_fn(|i| f64::from(tp.impulse[i].extract(k)));
            p.normal_delta_ns = if lane.row.stage == "warmstart" {
                p.normal_impulse_after_ns
            } else {
                p.normal_impulse_after_ns - f64::from(s.old.normal_part[point].impulse.extract(k))
            };
            p.tangent_delta_ns = std::array::from_fn(|i| {
                if lane.row.stage == "warmstart" {
                    p.tangent_impulse_after_ns[i]
                } else {
                    p.tangent_impulse_after_ns[i]
                        - f64::from(s.old.tangent_part[point].impulse[i].extract(k))
                }
            });
            let deltas = [
                p.normal_delta_ns,
                p.tangent_delta_ns[0],
                p.tangent_delta_ns[1],
            ];
            for (row, &delta) in deltas.iter().enumerate() {
                let direction = if row == 0 { normal } else { tangents[row - 1] };
                for i in 0..2 {
                    if lane.sides[i].data.solver_body_id == u32::MAX {
                        continue;
                    }
                    let (force, torque, inverse_torque, im) = if i == 0 {
                        (
                            direction,
                            if row == 0 {
                                lane_vector(np.torque_dir1, k)
                            } else {
                                lane_vector(tp.torque_dir1[row - 1], k)
                            },
                            if row == 0 {
                                lane_vector(np.ii_torque_dir1, k)
                            } else {
                                lane_vector(tp.ii_torque_dir1[row - 1], k)
                            },
                            lane_vector(c.im1, k),
                        )
                    } else {
                        (
                            -direction,
                            if row == 0 {
                                lane_vector(np.torque_dir2, k)
                            } else {
                                lane_vector(tp.torque_dir2[row - 1], k)
                            },
                            if row == 0 {
                                lane_vector(np.ii_torque_dir2, k)
                            } else {
                                lane_vector(tp.ii_torque_dir2[row - 1], k)
                            },
                            lane_vector(c.im2, k),
                        )
                    };
                    let work = delta * wrench(force, torque).dot(&mid[i]);
                    response[i] += wrench(im * force, inverse_torque) * delta;
                    if lane.sides[i].data.dynamic {
                        if row == 0 {
                            p.normal_midpoint_work_j += work;
                        } else {
                            p.tangent_midpoint_work_j += work;
                        }
                    } else {
                        p.prescribed_midpoint_work_j += work;
                    }
                }
            }
            if lane.row.stage == "solve" && !lane.row.unbiased {
                p.normal_bias_pairing_j =
                    -p.normal_delta_ns * f64::from(np.rhs.extract(k) - np.rhs_wo_bias.extract(k));
                p.tangent_bias_pairing_j = -(0..2)
                    .map(|i| {
                        p.tangent_delta_ns[i]
                            * f64::from(tp.rhs[i].extract(k) - tp.rhs_wo_bias[i].extract(k))
                    })
                    .sum::<f64>();
            }
            lane.row.normal_midpoint_work_j += p.normal_midpoint_work_j;
            lane.row.tangent_midpoint_work_j += p.tangent_midpoint_work_j;
            lane.row.prescribed_midpoint_work_j += p.prescribed_midpoint_work_j;
        }
        let same = lane.sides[0].data.solver_body_id != u32::MAX
            && lane.sides[0].data.solver_body_id == lane.sides[1].data.solver_body_id;
        let total_response: [_; 2] = std::array::from_fn(|i| {
            if same {
                &response[0] + &response[1]
            } else {
                response[i].clone()
            }
        });
        for i in 0..2 {
            let actual = &end[i] - &lane.sides[i].velocity;
            let change = kinetic_change(&lane.sides[i].velocity, &end[i], &lane.sides[i].mass);
            lane.sides[i].data.after_velocity = six_array(&end[i]);
            lane.sides[i].data.actual_delta_velocity = six_array(&actual);
            lane.sides[i].data.stored_response_delta_velocity = six_array(&total_response[i]);
            lane.sides[i].data.physical_kinetic_change_j = change;
            if !(same && i == 1) {
                lane.row.physical_kinetic_change_j += change;
            }
            lane.row.max_impulse_velocity_residual = lane
                .row
                .max_impulse_velocity_residual
                .max((actual - &total_response[i]).amax());
        }
        lane.row.physical_minus_impulse_work_j = lane.row.physical_kinetic_change_j
            - lane.row.normal_midpoint_work_j
            - lane.row.tangent_midpoint_work_j;
        lane.row.sides = lane.sides.into_iter().map(|s| s.data).collect();
        RECORDING.with(|r| {
            r.borrow_mut()
                .as_mut()
                .unwrap()
                .rigid_contacts
                .push(lane.row)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::graph::Graph;
    use crate::dynamics::solver::manifold_store::ManifoldStoreParts;
    use crate::dynamics::solver::solver_contact_graph::ContactRef;
    use crate::dynamics::{MassProperties, RigidBodyBuilder};
    use crate::geometry::{
        ColliderBuilder, ContactManifold, ContactPair, PairContacts, SolverContact,
    };
    use parry::shape::PackedFeatureId;

    #[test]
    fn rigid_midpoint_ledger_closes_coupled_rows_at_current_anisotropic_pose() {
        let mut bodies = RigidBodySet::new();
        let local_com = Vector::new(0.2, -0.1, 0.3);
        let mut body = RigidBodyBuilder::dynamic()
            .rotation(Vector::Z * core::f32::consts::FRAC_PI_2)
            .additional_mass_properties(MassProperties::new(local_com, 3., Vector::new(1., 2., 4.)))
            .build();
        body.recompute_mass_properties_from_colliders(&ColliderSet::new());
        let handle = bodies.insert(body);
        let mut colliders = ColliderSet::new();
        let local_pose = Pose::translation(0.05, 0.02, 0.01);
        let collider = colliders.insert_with_parent(
            ColliderBuilder::ball(0.1)
                .density(0.)
                .position(local_pose)
                .build(),
            handle,
            &mut bodies,
        );
        let world = colliders.insert(ColliderBuilder::ball(0.1).density(0.).build());
        bodies[handle].recompute_mass_properties_from_colliders(&colliders);
        bodies[handle].ids.active_set_id = 0;
        let mut solver = SolverBodies::default();
        solver.resize(1);
        solver.copy_from(0, &bodies[handle]);
        // Cached body pose is deliberately stale; current solver inertia/pose
        // remains a 90-degree rotation with world principal diagonal (2,1,4).
        bodies[handle].pos.position.rotation = Rotation::IDENTITY;
        let arm = Vector::new(0.4, 0.2, 0.1);
        let anchor = solver.poses[0].rotation.inverse() * arm;
        let world_anchor = solver.poses[0].translation + arm;
        let mut manifold = ContactManifold::new();
        manifold.data.rigid_body1 = Some(handle);
        manifold.data.normal = -Vector::Y;
        manifold.data.solver_body_ids = [0, u32::MAX];
        manifold.data.friction = 1.;
        let mut point = parry::query::TrackedContact::<crate::geometry::ContactData>::new(
            anchor,
            world_anchor,
            PackedFeatureId::face(0),
            PackedFeatureId::face(0),
            0.,
        );
        point.data.solver_dp1 = arm;
        manifold.points.push(point);
        manifold.data.solver_contacts.push(SolverContact {
            anchor1: anchor,
            anchor2: world_anchor,
            dist: 0.,
            tangent_velocity: Vector::ZERO,
            contact_id: [0],
            padding: [0.],
        });
        let mut pair = ContactPair::new(collider, world);
        let PairContacts::Rigid(rigid) = &mut pair.contacts else {
            unreachable!()
        };
        rigid.manifolds.push(manifold);
        let mut graph = Graph::<(), ContactPair>::with_capacity(2, 1);
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n1, n2, pair);
        let store = unsafe {
            ManifoldStore::from_parts(ManifoldStoreParts::new(
                graph.edges.as_mut_ptr(),
                graph.edges.len(),
            ))
        };
        let id = ContactRef {
            edge: 0,
            manifold: 0,
        };
        let ids = std::array::from_fn(|lane| if lane == 0 { id } else { ContactRef::PADDING });
        let manifold = store.get(id);
        let mut builder = unsafe { core::mem::zeroed() };
        let mut c = unsafe { core::mem::zeroed() };
        ContactWithCoulombFrictionBuilder::generate(
            ids,
            [manifold; SIMD_WIDTH],
            &bodies,
            &solver,
            &mut builder,
            &mut c,
        );
        let params = IntegrationParameters {
            dt: 1. / 120.,
            ..IntegrationParameters::default()
        };
        builder.update(&params, 0., &solver, &MultibodyJointSet::new(), &mut c);
        c.normal_part[0].impulse = SimdReal::splat(0.7);
        c.tangent_part[0].impulse[0] = SimdReal::splat(0.4);
        c.tangent_part[0].impulse[1] = SimdReal::splat(0.);
        let direction = lane_vector(c.dir1, 0);
        let tangent = lane_vector(c.tangent1, 0);
        let impulse_n = wrench(direction, arm.cross(direction)) * 0.7;
        let impulse_t = wrench(tangent, arm.cross(tangent)) * 0.4;
        let diagonal = VectorN::from_vec(vec![3., 3., 3., 2., 1., 4.]);
        let cross = impulse_n.dot(&impulse_t.component_div(&diagonal));
        assert!(cross.abs() > 1e-4, "fixture lost coupled row work");
        // Legacy diagnostics must never retain rigid traces, even when given
        // a real manifold that explicit collider-context recording will retain.
        let (_, legacy) = super::super::record(|| {
            assert!(!rigid_enabled());
            super::super::note_rigid_policy(true);
            let skipped = before(
                &c,
                &builder,
                &solver,
                &[handle],
                &bodies,
                &MultibodyJointSet::new(),
                &store,
                &params,
                "warmstart",
                false,
                0.,
                0,
                true,
            );
            assert!(skipped.is_none());
            after(skipped, &c, &solver);
        });
        assert_eq!(legacy.rigid_len(), 0);
        assert!(!legacy.twist_selected_unobserved());
        assert!(legacy.rigid_scope.contains("rigid collection disabled"));
        assert!(!rigid_enabled());
        let (_, report) = super::super::record_with_colliders(&colliders, || {
            assert!(rigid_enabled());
            let s = before(
                &c,
                &builder,
                &solver,
                &[handle],
                &bodies,
                &MultibodyJointSet::new(),
                &store,
                &params,
                "warmstart",
                false,
                0.,
                0,
                true,
            );
            assert!(s.is_some());
            c.warmstart(&mut solver);
            after(s, &c, &solver);
        });
        assert_eq!(
            report.rigid_len(),
            1,
            "padding lanes must not duplicate the physical block"
        );
        assert_eq!(report.len(), 0, "legacy generic count changed");
        let row = &report.rigid_contacts[0];
        let end = six(solver.vels[0]);
        let independent_energy = 0.5 * end.dot(&end.component_mul(&diagonal));
        assert!((row.physical_kinetic_change_j - independent_energy).abs() < 2e-7);
        assert!(row.physical_minus_impulse_work_j.abs() < 2e-7);
        assert!(row.max_impulse_velocity_residual < 2e-7);
        let separate_rows = 0.5
            * (impulse_n.dot(&impulse_n.component_div(&diagonal))
                + impulse_t.dot(&impulse_t.component_div(&diagonal)));
        assert!((independent_energy - separate_rows - cross).abs() < 2e-7);
        let stale_diagonal = VectorN::from_vec(vec![3., 3., 3., 1., 2., 4.]);
        let stale_energy = 0.5 * end.dot(&end.component_mul(&stale_diagonal));
        assert!((independent_energy - stale_energy).abs() > 1e-4);
        let expected = solver.poses[0].pose();
        let expected_origin = Pose::from_parts(
            expected.translation - expected.rotation * local_com,
            expected.rotation,
        );
        let expected_collider = pose_work(expected_origin * local_pose);
        let actual_collider = row.collider_world_poses[0].unwrap();
        assert_eq!(actual_collider.translation, expected_collider.translation);
        assert_eq!(
            actual_collider.rotation_xyzw,
            expected_collider.rotation_xyzw
        );
        assert_eq!(row.sides[0].physical_pose_source, "current_solver_com");
        assert_eq!(row.sides[1].solver_body_id, u32::MAX);
        assert!(
            (Vector::from_array(row.points[0].world_anchors_m[0].map(|x| x as f32)) - world_anchor)
                .length()
                < 1e-7
        );
    }

    #[test]
    fn all_padding_rigid_chunk_never_resolves_body_or_manifold_sentinels() {
        let mut c: ContactWithCoulombFriction<SimdReal> = unsafe { core::mem::zeroed() };
        c.manifold_id = [ContactRef::PADDING; SIMD_WIDTH];
        c.solver_vel1 = [u32::MAX; SIMD_WIDTH];
        c.solver_vel2 = [u32::MAX; SIMD_WIDTH];
        let builder = unsafe { core::mem::zeroed() };
        let solver = SolverBodies::default();
        let store =
            unsafe { ManifoldStore::from_parts(ManifoldStoreParts::new(core::ptr::null_mut(), 0)) };
        let colliders = ColliderSet::new();
        let (_, report) = super::super::record_with_colliders(&colliders, || {
            assert!(rigid_enabled());
            let s = before(
                &c,
                &builder,
                &solver,
                &[],
                &RigidBodySet::new(),
                &MultibodyJointSet::new(),
                &store,
                &IntegrationParameters::default(),
                "solve",
                false,
                0.,
                0,
                true,
            );
            assert!(
                s.is_some(),
                "padding test must enter the selected collector"
            );
            after(s, &c, &solver);
        });
        assert_eq!(report.rigid_len(), 0);
        assert_eq!(report.len(), 0);
    }
}

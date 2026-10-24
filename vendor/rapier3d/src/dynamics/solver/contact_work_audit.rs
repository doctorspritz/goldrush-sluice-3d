//! Optional read-only work ledger for single-threaded contact blocks.
//!
//! Net impulses in one fixed-pose contact block are paired with its midpoint
//! velocity. This partitions block work by normal/tangent rows; it is not the
//! sequential work of each row. Physical kinetic energy is independently
//! reconstructed from current link Jacobians, local masses and inertia tensors.
use crate::dynamics::solver::contact_constraint::GenericContactConstraint;
use crate::dynamics::solver::solver_body::SolverBodies;
use crate::dynamics::{MultibodyJointSet, RigidBody, RigidBodyHandle, RigidBodySet};
use crate::math::{DVector, Rotation, Vector};
use na::{DMatrix, DVector as VectorN, Matrix3, Quaternion, UnitQuaternion};
use std::cell::RefCell;
#[cfg(test)]
use std::vec;
use std::{format, vec::Vec};

pub(crate) mod rigid_work;

std::thread_local! {
    static RECORDING: RefCell<Option<WorkReport>> = const { RefCell::new(None) };
    static COLLIDERS: RefCell<Option<std::collections::HashMap<crate::geometry::ColliderHandle, rigid_work::ColliderSnapshot>>> = const { RefCell::new(None) };
}

/// Contacts and fixed-stage energy observations from one diagnostic run.
#[derive(Debug, Default, serde::Serialize)]
pub struct WorkReport {
    contacts: Vec<ContactWork>,
    rigid_contacts: Vec<rigid_work::RigidContactWork>,
    rigid_scope: &'static str,
    twist_selected_unobserved: bool,
    phases: Vec<PhaseEnergy>,
    islands: Vec<Vec<(u32, u32)>>,
}
impl WorkReport {
    /// Number of observed generic contact blocks.
    pub fn len(&self) -> usize {
        self.contacts.len()
    }
    /// Number of observed non-padding rigid Coulomb lane/manifold blocks.
    pub fn rigid_len(&self) -> usize {
        self.rigid_contacts.len()
    }
    /// Iterate rigid lane/manifold records without cloning the full report.
    pub fn iter_rigid_contacts(&self) -> impl Iterator<Item = &rigid_work::RigidContactWork> {
        self.rigid_contacts.iter()
    }
    /// Iterate generic block records without cloning the full report.
    pub fn iter_generic_contacts(&self) -> impl Iterator<Item = &ContactWork> {
        self.contacts.iter()
    }
    /// Iterate the small fixed-stage energy observations.
    pub fn iter_phases(&self) -> impl Iterator<Item = &PhaseEnergy> {
        self.phases.iter()
    }
    /// Stable body-handle scope of each observed island.
    pub fn islands(&self) -> &[Vec<(u32, u32)>] {
        &self.islands
    }
    /// Whether the run selected twist contacts outside the rigid audit scope.
    pub fn twist_selected_unobserved(&self) -> bool {
        self.twist_selected_unobserved
    }
    /// Whether no contact blocks were observed.
    pub fn is_empty(&self) -> bool {
        self.contacts.is_empty()
    }
}
/// Physical kinetic energy and mass-weighted position at one named solver stage.
#[derive(Debug, serde::Serialize)]
pub struct PhaseEnergy {
    island: usize,
    label: &'static str,
    unbiased: bool,
    time_s: f32,
    dynamic_bodies: usize,
    kinetic_j: f64,
    generalized_kinetic_j: f64,
    mass_position_kg_m: [f64; 3],
}

pub(crate) fn enabled() -> bool {
    RECORDING.with(|r| r.borrow().is_some())
}

/// Rigid diagnostics are explicitly selected by the collider-context recorder.
pub(crate) fn rigid_enabled() -> bool {
    enabled() && COLLIDERS.with(|c| c.borrow().is_some())
}

/// Serialized diagnostic for one fixed-pose generic contact solve block.
#[derive(Debug, serde::Serialize)]
pub struct ContactWork {
    bodies: [std::string::String; 2],
    manifold: std::string::String,
    stage: &'static str,
    unbiased: bool,
    solved_time_s: f32,
    physical_kinetic_change_j: f64,
    normal_midpoint_work_j: f64,
    tangent_midpoint_work_j: f64,
    physical_minus_impulse_work_j: f64,
    max_impulse_velocity_residual: f64,
    points: Vec<PointWork>,
}
#[derive(Debug, serde::Serialize)]
struct PointWork {
    contact_id: u8,
    normal_delta_ns: f64,
    tangent_delta_ns: [f64; 2],
    normal_work_j: f64,
    tangent_work_j: f64,
    normal_bias_pairing_j: f64,
    tangent_bias_pairing_j: f64,
    normal_rhs_m_s: Option<f32>,
    normal_rhs_without_bias_m_s: f32,
}

/// Record generic contact blocks and fixed-stage energy while executing `run`.
/// Rigid collection is disabled; use [`record_with_colliders`] to select it.
///
/// Only single-threaded builds and non-nested recording are admitted. A panic
/// clears the collector. This is diagnostic overhead, never a timing mode.
pub fn record<R>(run: impl FnOnce() -> R) -> (R, WorkReport) {
    record_inner(None, run)
}

/// Record contacts with immutable collider parent/local-pose metadata captured
/// before the run. Current world collider poses are derived at each rigid block
/// from the live solver CoM, body local CoM and this collider-local transform.
/// Collider attachments/local poses must remain unchanged during the run.
pub fn record_with_colliders<R>(
    colliders: &crate::geometry::ColliderSet,
    run: impl FnOnce() -> R,
) -> (R, WorkReport) {
    record_inner(Some(rigid_work::capture_colliders(colliders)), run)
}

fn record_inner<R>(
    colliders: Option<
        std::collections::HashMap<crate::geometry::ColliderHandle, rigid_work::ColliderSnapshot>,
    >,
    run: impl FnOnce() -> R,
) -> (R, WorkReport) {
    const {
        assert!(
            !cfg!(feature = "parallel"),
            "work audit requires single-threaded Rapier"
        );
    }
    let rigid_scope = if colliders.is_some() {
        "rigid_coulomb warmstart/projection/solve only; twist/restitution blocks unobserved"
    } else {
        "generic contacts and phases only; rigid collection disabled"
    };
    RECORDING.with(|r| {
        let mut r = r.borrow_mut();
        assert!(r.is_none(), "nested work recording");
        *r = Some(WorkReport {
            rigid_scope,
            ..WorkReport::default()
        });
    });
    COLLIDERS.with(|c| *c.borrow_mut() = colliders);
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            RECORDING.with(|r| {
                r.borrow_mut().take();
            });
            COLLIDERS.with(|c| {
                c.borrow_mut().take();
            });
        }
    }
    let reset = Reset;
    let result = run();
    let rows = RECORDING.with(|r| r.borrow_mut().take().unwrap());
    drop(reset);
    (result, rows)
}

pub(crate) fn note_rigid_policy(twist: bool) {
    if !rigid_enabled() {
        return;
    }
    RECORDING.with(|r| {
        if let Some(report) = r.borrow_mut().as_mut() {
            report.twist_selected_unobserved |= twist;
        }
    });
}

struct Side {
    generic: bool,
    id: u32,
    mass: DMatrix<f64>,
    velocity: VectorN<f64>,
}
pub(crate) struct Snapshot {
    sides: [Side; 2],
    constraint: GenericContactConstraint,
    bodies: [std::string::String; 2],
    unbiased: bool,
    time: f32,
    stage: &'static str,
}
fn spatial_mass(body: &RigidBody, rotation: Rotation) -> DMatrix<f64> {
    let mut mass = DMatrix::zeros(6, 6);
    if !body.is_dynamic() {
        return mass;
    }
    for i in 0..3 {
        mass[(i, i)] = f64::from(body.mass());
    }
    let local = &body.mass_properties().local_mprops;
    let q = rotation * local.principal_inertia_local_frame;
    let [x, y, z, w] = q.to_array().map(f64::from);
    let q = UnitQuaternion::new_unchecked(Quaternion::new(w, x, y, z));
    let axes = q.to_rotation_matrix();
    let inertia = local.principal_inertia().to_array().map(f64::from);
    let tensor = axes.matrix()
        * Matrix3::from_diagonal(&na::Vector3::from(inertia))
        * axes.matrix().transpose();
    mass.view_mut((3, 3), (3, 3)).copy_from(&tensor);
    mass
}
fn velocity(
    generic: bool,
    id: u32,
    ndofs: usize,
    bodies: &SolverBodies,
    q: &DVector,
) -> VectorN<f64> {
    if id == u32::MAX {
        return VectorN::zeros(0);
    }
    if generic {
        return q.rows(id as usize, ndofs).map(f64::from).into_owned();
    }
    let v = bodies.vels[id as usize];
    VectorN::from_iterator(
        6,
        v.linear
            .to_array()
            .into_iter()
            .chain(v.angular.to_array())
            .map(f64::from),
    )
}
fn side(
    handle: RigidBodyHandle,
    generic: bool,
    id: u32,
    ndofs: usize,
    solver: &SolverBodies,
    q: &DVector,
    bodies: &RigidBodySet,
    multibodies: &MultibodyJointSet,
) -> Side {
    let velocity = velocity(generic, id, ndofs, solver, q);
    let mass = if id == u32::MAX {
        DMatrix::zeros(0, 0)
    } else if generic {
        let link = multibodies
            .rigid_body_link(handle)
            .expect("recorded generic link");
        let mb = &multibodies[link.multibody];
        assert_eq!(mb.ndofs(), ndofs);
        let mut matrix = DMatrix::zeros(ndofs, ndofs);
        for (i, link) in mb.links().enumerate() {
            let rb = &bodies[link.rigid_body_handle()];
            let j = mb.body_jacobian(i).map(f64::from);
            matrix += j.transpose() * spatial_mass(rb, link.local_to_world.rotation) * j;
        }
        matrix
    } else {
        spatial_mass(&bodies[handle], solver.poses[id as usize].rotation)
    };
    Side {
        generic,
        id,
        mass,
        velocity,
    }
}

pub(crate) fn before(
    c: &GenericContactConstraint,
    handles: [RigidBodyHandle; 2],
    solver: &SolverBodies,
    q: &DVector,
    bodies: &RigidBodySet,
    mb: &MultibodyJointSet,
    unbiased: bool,
    time: f32,
    stage: &'static str,
) -> Option<Snapshot> {
    if !RECORDING.with(|r| r.borrow().is_some()) {
        return None;
    }
    let sides = [
        side(
            handles[0],
            c.generic_constraint_mask & 1 != 0,
            c.solver_vel1,
            c.ndofs1,
            solver,
            q,
            bodies,
            mb,
        ),
        side(
            handles[1],
            c.generic_constraint_mask & 2 != 0,
            c.solver_vel2,
            c.ndofs2,
            solver,
            q,
            bodies,
            mb,
        ),
    ];
    Some(Snapshot {
        sides,
        constraint: *c,
        bodies: handles.map(|h| format!("{h:?}")),
        unbiased,
        time,
        stage,
    })
}
fn basis(
    side: &Side,
    jac: &DVector,
    index: usize,
    direction: Vector,
    torque: Vector,
    inverse_mass: Vector,
    inverse_torque: Vector,
) -> (VectorN<f64>, VectorN<f64>) {
    let n = side.velocity.len();
    if side.id == u32::MAX {
        return (VectorN::zeros(0), VectorN::zeros(0));
    }
    if side.generic {
        (
            jac.rows(index, n).map(f64::from).into_owned(),
            jac.rows(index + n, n).map(f64::from).into_owned(),
        )
    } else {
        let force = VectorN::from_iterator(
            6,
            direction
                .to_array()
                .into_iter()
                .chain(torque.to_array())
                .map(f64::from),
        );
        let linear = inverse_mass * direction;
        let response = VectorN::from_iterator(
            6,
            linear
                .to_array()
                .into_iter()
                .chain(inverse_torque.to_array())
                .map(f64::from),
        );
        (force, response)
    }
}
fn kinetic(velocity: &VectorN<f64>, mass: &DMatrix<f64>) -> f64 {
    0.5 * velocity.dot(&(mass * velocity))
}

pub(crate) fn after(
    snapshot: Option<Snapshot>,
    c: &GenericContactConstraint,
    jac: &DVector,
    solver: &SolverBodies,
    q: &DVector,
) {
    let Some(s) = snapshot else {
        return;
    };
    let old = &s.constraint;
    let ndofs = [c.ndofs1, c.ndofs2];
    let end: [_; 2] =
        std::array::from_fn(|i| velocity(s.sides[i].generic, s.sides[i].id, ndofs[i], solver, q));
    let mid: [_; 2] = std::array::from_fn(|i| (&s.sides[i].velocity + &end[i]) * 0.5);
    let mut response: [_; 2] = std::array::from_fn(|i| VectorN::zeros(end[i].len()));
    let mut physical = 0.;
    for i in 0..2 {
        if i == 1 && s.sides[0].generic == s.sides[1].generic && s.sides[0].id == s.sides[1].id {
            continue;
        }
        physical +=
            kinetic(&end[i], &s.sides[i].mass) - kinetic(&s.sides[i].velocity, &s.sides[i].mass);
    }
    let step = 2 * (c.ndofs1 + c.ndofs2);
    let tangents = [c.tangent1, c.dir1.cross(c.tangent1)];
    let mut points = Vec::new();
    let mut normal_work = 0.;
    let mut tangent_work = 0.;
    for point in 0..c.num_contacts as usize {
        let normal = &c.normal_part[point];
        let tangent = &c.tangent_part[point];
        let dn = f64::from(if s.stage == "warmstart" {
            normal.impulse
        } else {
            normal.impulse - old.normal_part[point].impulse
        });
        let dt: [f64; 2] = std::array::from_fn(|i| {
            f64::from(if s.stage == "warmstart" {
                tangent.impulse[i]
            } else {
                tangent.impulse[i] - old.tangent_part[point].impulse[i]
            })
        });
        let deltas = [dn, dt[0], dt[1]];
        let mut work = [0.; 3];
        for row in 0..3 {
            let first = c.j_id + point * 3 * step + row * step;
            let direction = if row == 0 { c.dir1 } else { tangents[row - 1] };
            for side_id in 0..2 {
                let (index, direction, torque, im, it) = if side_id == 0 {
                    (
                        first,
                        direction,
                        if row == 0 {
                            normal.torque_dir1
                        } else {
                            tangent.torque_dir1[row - 1]
                        },
                        c.im1,
                        if row == 0 {
                            normal.ii_torque_dir1
                        } else {
                            tangent.ii_torque_dir1[row - 1]
                        },
                    )
                } else {
                    (
                        first + c.ndofs1 * 2,
                        -direction,
                        if row == 0 {
                            normal.torque_dir2
                        } else {
                            tangent.torque_dir2[row - 1]
                        },
                        c.im2,
                        if row == 0 {
                            normal.ii_torque_dir2
                        } else {
                            tangent.ii_torque_dir2[row - 1]
                        },
                    )
                };
                let (force, map) = basis(&s.sides[side_id], jac, index, direction, torque, im, it);
                work[row] += deltas[row] * force.dot(&mid[side_id]);
                response[side_id] += map * deltas[row];
            }
        }
        normal_work += work[0];
        tangent_work += work[1] + work[2];
        points.push(PointWork {
            contact_id: c.manifold_contact_id[point],
            normal_delta_ns: dn,
            tangent_delta_ns: dt,
            normal_work_j: work[0],
            tangent_work_j: work[1] + work[2],
            normal_bias_pairing_j: if s.stage == "solve" {
                -dn * f64::from(normal.rhs - normal.rhs_wo_bias)
            } else {
                0.
            },
            tangent_bias_pairing_j: if s.stage == "solve" {
                -(0..2)
                    .map(|i| dt[i] * f64::from(tangent.rhs[i] - tangent.rhs_wo_bias[i]))
                    .sum::<f64>()
            } else {
                0.
            },
            normal_rhs_m_s: if s.stage == "warmstart" {
                None
            } else if s.stage == "projection" {
                Some(normal.rhs_wo_bias)
            } else {
                Some(normal.rhs)
            },
            normal_rhs_without_bias_m_s: normal.rhs_wo_bias,
        });
    }
    let mut residual: f64 = 0.;
    for i in 0..2 {
        let mut expected = response[i].clone();
        if s.sides[0].generic == s.sides[1].generic && s.sides[0].id == s.sides[1].id {
            expected += &response[1 - i];
        }
        let actual = &end[i] - &s.sides[i].velocity;
        residual = residual.max((actual - expected).amax());
    }
    let row = ContactWork {
        bodies: s.bodies,
        manifold: format!("{:?}", c.manifold_id),
        stage: s.stage,
        unbiased: s.unbiased,
        solved_time_s: s.time,
        physical_kinetic_change_j: physical,
        normal_midpoint_work_j: normal_work,
        tangent_midpoint_work_j: tangent_work,
        physical_minus_impulse_work_j: physical - normal_work - tangent_work,
        max_impulse_velocity_residual: residual,
        points,
    };
    RECORDING.with(|r| r.borrow_mut().as_mut().unwrap().contacts.push(row));
}

pub(crate) fn phase(
    label: &'static str,
    unbiased: bool,
    time: f32,
    handles: &[RigidBodyHandle],
    solver: &SolverBodies,
    q: &DVector,
    bodies: &RigidBodySet,
    mb: &MultibodyJointSet,
) {
    if !RECORDING.with(|r| r.borrow().is_some()) {
        return;
    }
    let scope: Vec<_> = handles.iter().map(|h| h.into_raw_parts()).collect();
    let island = RECORDING.with(|r| {
        let mut r = r.borrow_mut();
        let report = r.as_mut().unwrap();
        if let Some(index) = report.islands.iter().position(|old| old == &scope) {
            index
        } else {
            report.islands.push(scope);
            report.islands.len() - 1
        }
    });
    let mut row = PhaseEnergy {
        island,
        label,
        unbiased,
        time_s: time,
        dynamic_bodies: 0,
        kinetic_j: 0.,
        generalized_kinetic_j: 0.,
        mass_position_kg_m: [0.; 3],
    };
    for handle in handles {
        let body = &bodies[*handle];
        if !body.is_dynamic() {
            continue;
        }
        assert!(
            !body.is_sleeping(),
            "phase audit only covers awake rigid bodies"
        );
        let (v, rotation, center, generic) = if let Some(link) = mb.rigid_body_link(*handle) {
            let multibody = &mb[link.multibody];
            let current = multibody.link(link.id).unwrap();
            let j = multibody.body_jacobian(link.id).map(f64::from);
            let velocity = q
                .rows(multibody.solver_id as usize, multibody.ndofs())
                .map(f64::from);
            (
                VectorN::from_column_slice((j * velocity).as_slice()),
                current.local_to_world.rotation,
                current.local_to_world * body.local_center_of_mass(),
                true,
            )
        } else {
            let id = body.effective_active_set_offset();
            (
                velocity(false, id, 0, solver, q),
                solver.poses[id as usize].rotation,
                solver.poses[id as usize].translation,
                false,
            )
        };
        let energy = kinetic(&v, &spatial_mass(body, rotation));
        row.kinetic_j += energy;
        if generic {
            row.generalized_kinetic_j += energy;
        }
        for (i, position) in center.to_array().into_iter().enumerate() {
            row.mass_position_kg_m[i] += f64::from(body.mass()) * f64::from(position);
        }
        row.dynamic_bodies += 1;
    }
    RECORDING.with(|r| r.borrow_mut().as_mut().unwrap().phases.push(row));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn midpoint_work_matches_independent_rotated_rigid_energy() {
        let mut body = crate::dynamics::RigidBodyBuilder::dynamic()
            .additional_mass_properties(crate::dynamics::MassProperties::new(
                Vector::ZERO,
                3.,
                Vector::new(1., 2., 4.),
            ))
            .build();
        body.recompute_mass_properties_from_colliders(&crate::geometry::ColliderSet::new());
        let mass = spatial_mass(
            &body,
            Rotation::from_rotation_z(core::f32::consts::FRAC_PI_2),
        );
        let v = VectorN::from_vec(vec![1., -0.2, 0.3, -0.4, 0.5, 0.6]);
        let impulse = VectorN::from_vec(vec![0.2, -0.5, 0.1, 0.4, -0.1, 0.3]);
        let exact_diagonal = VectorN::from_vec(vec![3., 3., 3., 2., 1., 4.]);
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { exact_diagonal[i] } else { 0. };
                assert!((mass[(i, j)] - expected).abs() < 1e-6);
            }
        }
        let delta = impulse.component_div(&exact_diagonal);
        let end = &v + delta;
        let work = impulse.dot(&((&v + &end) * 0.5));
        assert!((kinetic(&end, &mass) - kinetic(&v, &mass) - work).abs() < 1e-6);
    }
    #[test]
    fn recording_is_scoped_and_recovers_after_panic() {
        let (_, rows) = record(|| ());
        assert!(rows.is_empty());
        assert!(std::panic::catch_unwind(|| record(|| panic!("observer control"))).is_err());
        let (_, rows) = record(|| ());
        assert!(rows.is_empty());
    }
}

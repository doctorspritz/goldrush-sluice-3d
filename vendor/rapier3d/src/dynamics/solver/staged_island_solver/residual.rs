//! Probe copies of the actual operators at the common global barrier.
use super::*;
use crate::contact_work_audit::residual::{self, Family, Frame, Point};
use crate::dynamics::solver::solver_body::SolverBodies;
use crate::math::{DVector, Vector};
use crate::utils::ScalarType;
use simba::simd::SimdValue;

fn vector(v: <SimdReal as ScalarType>::Vector, lane: usize) -> Vector {
    Vector::from_array(std::array::from_fn(|i| v[i].extract(lane)))
}
fn array(v: Vector) -> [f64; 3] {
    v.to_array().map(f64::from)
}
fn reset(scratch: &mut SolverBodies, actual: &SolverBodies, ids: impl IntoIterator<Item = u32>) {
    for id in ids {
        if id != u32::MAX {
            scratch.vels[id as usize] = actual.vels[id as usize];
        }
    }
}
fn velocity_change(
    f: &mut Family,
    scratch: &SolverBodies,
    actual: &SolverBodies,
    ids: impl IntoIterator<Item = u32>,
) {
    for id in ids {
        if id != u32::MAX {
            let a = actual.get_vel(id);
            let b = scratch.get_vel(id);
            f.maximum_linear_probe_m_s = f
                .maximum_linear_probe_m_s
                .max(f64::from((b.linear - a.linear).length()));
            f.maximum_angular_probe_rad_s = f
                .maximum_angular_probe_rad_s
                .max(f64::from((b.angular - a.angular).length()));
        }
    }
}
fn generic_change(f: &mut Family, a: &DVector, b: &DVector) {
    for (a, b) in a.iter().zip(b.iter()) {
        f.maximum_generalized_probe = f.maximum_generalized_probe.max(f64::from((b - a).abs()));
    }
}
fn contact_row(f: &mut Family, mass: f32, old: f32, new: f32, ta: [f32; 2], tb: [f32; 2]) {
    assert!(
        [mass, old, new, ta[0], ta[1], tb[0], tb[1]]
            .iter()
            .all(|x| x.is_finite()),
        "nonfinite contact probe"
    );
    f.rows += 1;
    let dn = f64::from(new) - f64::from(old);
    let dt = (f64::from(tb[0]) - f64::from(ta[0])).hypot(f64::from(tb[1]) - f64::from(ta[1]));
    f.maximum_impulse_correction = f.maximum_impulse_correction.max(dn.abs()).max(dt);
    if mass > 0. {
        f.maximum_normal_response_m_s = f
            .maximum_normal_response_m_s
            .max(dn.abs() / f64::from(mass));
    }
}
fn joint_row(
    f: &mut Family,
    old: f32,
    new: f32,
    kind: crate::dynamics::solver::joint_constraint::WritebackId,
) {
    use crate::dynamics::solver::joint_constraint::WritebackId::*;
    let k = match kind {
        Dof(_) => 0,
        Limit(_) => 1,
        Motor(_) => 2,
        Friction(_) => 3,
    };
    f.joint_kinds[k] += 1;
    f.rows += 1;
    f.maximum_impulse_correction = f
        .maximum_impulse_correction
        .max((f64::from(new) - f64::from(old)).abs());
}

/// Called only after a completed solve-pass barrier or integration barrier.
/// No worker-owned references are formed when this observer is disabled.
pub(super) unsafe fn observe(
    ctx: &SharedCtx<'_>,
    group_index: usize,
    params: &IntegrationParameters,
    time: Real,
    integrated: bool,
) {
    if !residual::enabled() {
        return;
    }
    assert_eq!(ctx.sync.num_workers, 1);
    let group = &ctx.groups[group_index];
    let vs = unsafe { &*ctx.velocity_solver };
    let actual = &vs.solver_bodies;
    let contacts = unsafe { &*ctx.contact_constraints };
    let joints = unsafe { &*ctx.joint_constraints };
    let pairs = residual::pairs();
    let mut frame = Frame {
        group: group_index,
        groups: ctx.groups.len(),
        time_s: time,
        dt_s: params.dt,
        pgs_iterations: group.num_pgs_iterations,
        biased_friction: params.friction_in_bias_pass
            || params.num_internal_stabilization_iterations == 0,
        twist_not_probed: ctx.use_twist,
        soft_slots_not_probed: group.soft_slots.len(),
        ..Frame::default()
    };
    let mut scratch = SolverBodies {
        vels: actual.vels.clone(),
        poses: actual.poses.clone(),
        flags: actual.flags.clone(),
    };
    let mut generic = vs.generic_solver_vels.clone();
    if !ctx.use_twist {
        for chunk in group.chunks.clone() {
            let c = unsafe { &*ctx.coulomb_constraints.add(chunk) };
            let ids = c.solver_vel1.into_iter().chain(c.solver_vel2);
            let mut probe = *c;
            if !integrated {
                reset(&mut scratch, actual, ids.clone());
                probe.solve(&mut scratch, true, frame.biased_friction);
                frame.rigid.blocks += 1;
                velocity_change(&mut frame.rigid, &scratch, actual, ids);
            }
            let builder = unsafe { &*ctx.coulomb_builders.add(chunk) };
            for lane in 0..SIMD_WIDTH {
                let mr = c.manifold_id[lane];
                if mr.is_padding() {
                    continue;
                }
                let manifold = ctx.store.get(mr);
                let handles = [manifold.data.rigid_body1, manifold.data.rigid_body2];
                let (colliders, clustered) = ctx.store.audit_pair_metadata(mr);
                let watched = pairs
                    .iter()
                    .any(|p| handles == p.map(Some) || handles == [Some(p[1]), Some(p[0])]);
                if integrated && !watched {
                    continue;
                }
                let n = vector(c.dir1, lane);
                let ids = [c.solver_vel1[lane], c.solver_vel2[lane]];
                let poses = ids.map(|id| actual.get_pose(id));
                let vels = ids.map(|id| actual.get_vel(id));
                for point in 0..c.num_contacts as usize {
                    let contact_id = c.manifold_contact_id[point][lane];
                    if contact_id == u8::MAX {
                        continue;
                    }
                    let a = &c.normal_part[point];
                    let b = &probe.normal_part[point];
                    let ta = [
                        c.tangent_part[point].impulse.x.extract(lane),
                        c.tangent_part[point].impulse.y.extract(lane),
                    ];
                    let tb = [
                        probe.tangent_part[point].impulse.x.extract(lane),
                        probe.tangent_part[point].impulse.y.extract(lane),
                    ];
                    let mass = a.r.extract(lane);
                    let old = a.impulse.extract(lane);
                    let new = b.impulse.extract(lane);
                    if !integrated {
                        contact_row(&mut frame.rigid, mass, old, new, ta, tb);
                    }
                    let info = &builder.audit_infos()[point];
                    let p1 = poses[0].transform_point(vector(info.local_p1, lane))
                        + vector(info.tangent_vel, lane)
                            * (time + if integrated { params.dt } else { 0. });
                    let p2 = poses[1].transform_point(vector(info.local_p2, lane));
                    let material1 =
                        poses[0].transform_point(vector(info.diagnostic_material_p1, lane));
                    let material2 =
                        poses[1].transform_point(vector(info.diagnostic_material_p2, lane));
                    let dn = f64::from(new) - f64::from(old);
                    let value = Point {
                        key_chunk_lane_point: [chunk, lane, point],
                        manifold: [mr.edge, mr.manifold],
                        bodies: handles.map(|h| h.map(|h| h.into_raw_parts())),
                        contact_id,
                        colliders: colliders.map(|h| h.into_raw_parts()),
                        clustered,
                        force_direction1: array(n),
                        rotation_before_xyzw: poses.map(|p| p.rotation.to_array().map(f64::from)),
                        normal_correction_ns: dn,
                        tangent_correction_ns: (f64::from(tb[0]) - f64::from(ta[0]))
                            .hypot(f64::from(tb[1]) - f64::from(ta[1])),
                        normal_response_m_s: if mass > 0. {
                            dn.abs() / f64::from(mass)
                        } else {
                            0.
                        },
                        normal_velocity_m_s: f64::from(
                            n.dot(vels[0].linear - vels[1].linear)
                                + vector(a.torque_dir1, lane).dot(vels[0].angular)
                                + vector(a.torque_dir2, lane).dot(vels[1].angular),
                        ),
                        rhs_m_s: f64::from(a.rhs.extract(lane)),
                        normal_impulse_ns: f64::from(old),
                        effective_mass_kg: f64::from(mass),
                        cfm_factor: f64::from(a.cfm_factor.extract(lane)),
                        tracking_gap_before_m: f64::from(
                            info.dist.extract(lane) + (p1 - p2).dot(n),
                        ),
                        material_gap_before_m: f64::from((material1 - material2).dot(n)),
                        com_before_m: poses.map(|p| array(p.translation)),
                        linear_before_m_s: vels.map(|v| array(v.linear)),
                        angular_before_rad_s: vels.map(|v| array(v.angular)),
                        ..Point::default()
                    };
                    if watched {
                        assert!(frame.watched.len() < 128, "watched contact bound exceeded");
                        frame.watched.push(value);
                    } else if !integrated {
                        frame.top_rigid.push(value);
                        frame.top_rigid.sort_unstable_by(|a, b| {
                            b.normal_response_m_s.total_cmp(&a.normal_response_m_s)
                        });
                        frame.top_rigid.truncate(16);
                    }
                }
            }
        }
    }
    if integrated {
        residual::finish(group_index, time, frame.watched);
        return;
    }
    // Generic constraints are solved only for single-group layouts, matching
    // solve_pass exactly. Each row/block probe is isolated from every other.
    if ctx.groups.len() == 1 {
        for c in &contacts.generic_velocity_constraints {
            scratch.vels.clone_from(&actual.vels);
            generic.copy_from(&vs.generic_solver_vels);
            let mut probe = *c;
            probe.solve(
                &contacts.generic_jacobians,
                &mut scratch,
                &mut generic,
                true,
                frame.biased_friction,
            );
            frame.generic_contacts.blocks += 1;
            velocity_change(
                &mut frame.generic_contacts,
                &scratch,
                actual,
                0..actual.len() as u32,
            );
            generic_change(
                &mut frame.generic_contacts,
                &vs.generic_solver_vels,
                &generic,
            );
            for i in 0..c.num_contacts as usize {
                if c.manifold_contact_id[i] == u8::MAX {
                    continue;
                }
                contact_row(
                    &mut frame.generic_contacts,
                    c.normal_part[i].r,
                    c.normal_part[i].impulse,
                    probe.normal_part[i].impulse,
                    [c.tangent_part[i].impulse.x, c.tangent_part[i].impulse.y],
                    [
                        probe.tangent_part[i].impulse.x,
                        probe.tangent_part[i].impulse.y,
                    ],
                );
            }
        }
        for c in &joints.generic_velocity_constraints {
            scratch.vels.clone_from(&actual.vels);
            generic.copy_from(&vs.generic_solver_vels);
            let mut probe = *c;
            probe.solve(&joints.generic_jacobians, &mut scratch, &mut generic);
            frame.generic_joints.blocks += 1;
            joint_row(
                &mut frame.generic_joints,
                c.impulse,
                probe.impulse,
                c.writeback_id,
            );
            velocity_change(
                &mut frame.generic_joints,
                &scratch,
                actual,
                0..actual.len() as u32,
            );
            generic_change(&mut frame.generic_joints, &vs.generic_solver_vels, &generic);
        }
    }
    for builder in group.joint_builders.clone() {
        for row in ctx.joint_rows[builder].clone() {
            let c = &joints.velocity_constraints[row];
            let mut probe = *c;
            let ids = c.solver_vel1.into_iter().chain(c.solver_vel2);
            reset(&mut scratch, actual, ids.clone());
            probe.solve(&mut scratch);
            frame.scalar_joints.blocks += 1;
            joint_row(
                &mut frame.scalar_joints,
                c.impulse,
                probe.impulse,
                c.writeback_id,
            );
            velocity_change(&mut frame.scalar_joints, &scratch, actual, ids);
        }
    }
    for chunk in group.joint_chunks.clone() {
        for row in ctx.joint_chunk_rows[chunk].clone() {
            let c = &joints.simd_velocity_constraints[row];
            let mut probe = *c;
            let ids = c.solver_vel1.into_iter().chain(c.solver_vel2);
            reset(&mut scratch, actual, ids.clone());
            probe.solve(&mut scratch);
            frame.simd_joints.blocks += 1;
            for lane in 0..SIMD_WIDTH {
                if c.solver_vel1[lane] != u32::MAX || c.solver_vel2[lane] != u32::MAX {
                    joint_row(
                        &mut frame.simd_joints,
                        c.impulse.extract(lane),
                        probe.impulse.extract(lane),
                        c.writeback_id,
                    );
                }
            }
            velocity_change(&mut frame.simd_joints, &scratch, actual, ids);
        }
    }
    residual::push(frame);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::solver::SolverVel;
    use crate::dynamics::solver::contact_constraint::ContactConstraintNormalPart;
    #[test]
    fn local_success_hides_a_coupled_residual_and_repeated_sweeps_converge() {
        let mut a = ContactConstraintNormalPart::<Real> {
            r: 0.5,
            cfm_factor: 1.,
            ..ContactConstraintNormalPart::zero()
        };
        let mut b = ContactConstraintNormalPart::<Real> {
            r: 1.,
            cfm_factor: 1.,
            ..ContactConstraintNormalPart::zero()
        };
        let mut v = [SolverVel::<Real>::default(); 3];
        v[0].linear.x = -1.;
        let step = |a: &mut ContactConstraintNormalPart<Real>,
                    b: &mut ContactConstraintNormalPart<Real>,
                    v: &mut [SolverVel<Real>; 3]| {
            let [v0, v1, wall] = v;
            a.solve(&Vector::X, &Vector::ONE, &Vector::ONE, v0, v1);
            assert!(
                (v0.linear.x - v1.linear.x).abs() < 1e-6,
                "first local row did not solve"
            );
            b.solve(&Vector::X, &Vector::ONE, &Vector::ZERO, v1, wall);
        };
        step(&mut a, &mut b, &mut v);
        assert_eq!(v[0].linear.x, -0.5);
        assert_eq!(v[1].linear.x, 0.);
        // Independently known global row defect: -0.5 m/s despite its local
        // zero immediately before the wall row. A scratch operator needs 0.25 Ns.
        let mut probe = a;
        let mut scratch = v;
        let [v0, v1, _] = &mut scratch;
        probe.solve(&Vector::X, &Vector::ONE, &Vector::ONE, v0, v1);
        assert_eq!(probe.impulse - a.impulse, 0.25);
        assert_eq!(v[0].linear.x, -0.5, "probe mutated actual state");
        for _ in 0..31 {
            step(&mut a, &mut b, &mut v);
        }
        // The independent exact solution is two stopped masses, with one Ns
        // transmitted through each interface to the immovable wall.
        assert!(v[0].linear.x.abs() < 1e-7 && v[1].linear.x == 0.);
        assert!((a.impulse - 1.).abs() < 1e-7 && (b.impulse - 1.).abs() < 1e-7);
    }
}

use crate::dynamics::solver::GenericRhs;
use crate::dynamics::{IntegrationParameters, MultibodyJointSet, RigidBodySet};
use crate::geometry::ContactManifold;
#[cfg(feature = "dim3")]
use crate::math::TangentImpulse;
use crate::math::{DIM, DVector, MAX_MANIFOLD_POINTS, Real};
use crate::utils::{self, AngularInertiaOps, CrossProduct, DotProduct};

use super::{ContactConstraintNormalPart, ContactConstraintTangentPart};
use crate::dynamics::solver::CoulombContactPointInfos;
use crate::dynamics::solver::manifold_store::ManifoldStore;
use crate::dynamics::solver::solver_body::SolverBodies;
use crate::dynamics::solver::solver_contact_graph::ContactRef;
use crate::prelude::RigidBodyHandle;
#[cfg(feature = "dim2")]
use crate::utils::OrthonormalBasis;
use parry::math::Vector;

#[derive(Copy, Clone)]
pub(crate) struct GenericContactConstraintBuilder {
    infos: [CoulombContactPointInfos<Real>; MAX_MANIFOLD_POINTS],
    handle1: RigidBodyHandle,
    handle2: RigidBodyHandle,
}

impl GenericContactConstraintBuilder {
    pub fn invalid() -> Self {
        Self {
            infos: [CoulombContactPointInfos::default(); MAX_MANIFOLD_POINTS],
            handle1: RigidBodyHandle::invalid(),
            handle2: RigidBodyHandle::invalid(),
        }
    }

    pub fn generate(
        manifold_id: ContactRef,
        manifold: &ContactManifold,
        bodies: &RigidBodySet,
        multibodies: &MultibodyJointSet,
        out_builder: &mut GenericContactConstraintBuilder,
        out_constraint: &mut GenericContactConstraint,
        jacobians: &mut DVector,
        jacobian_id: &mut usize,
    ) {
        // TODO PERF: we haven’t tried to optimized this codepath yet (since it relies
        //            on multibodies which are already much slower than regular bodies).
        let handle1 = manifold
            .data
            .rigid_body1
            .unwrap_or(RigidBodyHandle::invalid());
        let handle2 = manifold
            .data
            .rigid_body2
            .unwrap_or(RigidBodyHandle::invalid());

        let rb1 = &bodies.get(handle1).unwrap_or(&bodies.default_fixed);
        let rb2 = &bodies.get(handle2).unwrap_or(&bodies.default_fixed);

        // Frontier pairs (partial-island sleep): a sleeping body (or multibody link) acts as a
        // world-attached wall — same treatment as a fixed body, and its `active_set_id`/
        // `solver_id` belongs to another (sleeping) island so it must not be referenced.
        let effective_type = |rb: &crate::dynamics::RigidBody| {
            if rb.is_sleeping() {
                crate::dynamics::RigidBodyType::Fixed
            } else {
                rb.body_type
            }
        };
        let (vels1, mprops1, type1) = (&rb1.vels, &rb1.mprops, effective_type(rb1));
        let (vels2, mprops2, type2) = (&rb2.vels, &rb2.mprops, effective_type(rb2));

        // A multibody's fixed root behaves exactly like a regular fixed body.
        // Thus, a contact against a fixed root must not reference the multibody's
        // `solver_id`, because the multibody is only set up on the island owning
        // its *dynamic* links.
        let multibody1 = multibodies
            .rigid_body_link(handle1)
            .map(|m| (&multibodies[m.multibody], m.id))
            .filter(|(mb, link_id)| (*link_id != 0 || mb.root_is_dynamic) && !rb1.is_sleeping());
        let multibody2 = multibodies
            .rigid_body_link(handle2)
            .map(|m| (&multibodies[m.multibody], m.id))
            .filter(|(mb, link_id)| (*link_id != 0 || mb.root_is_dynamic) && !rb2.is_sleeping());
        let solver_vel1 =
            multibody1
                .map(|mb| mb.0.solver_id)
                .unwrap_or(if type1.is_dynamic_or_kinematic() {
                    rb1.ids.active_set_id
                } else {
                    u32::MAX
                });
        let solver_vel2 =
            multibody2
                .map(|mb| mb.0.solver_id)
                .unwrap_or(if type2.is_dynamic_or_kinematic() {
                    rb2.ids.active_set_id
                } else {
                    u32::MAX
                });
        let force_dir1 = -manifold.data.normal;

        #[cfg(feature = "dim2")]
        let tangents1 = force_dir1.orthonormal_basis();
        #[cfg(feature = "dim3")]
        let tangents1 = super::compute_tangent_contact_directions::<Real>(
            &force_dir1,
            &vels1.linvel,
            &vels2.linvel,
        );

        let multibodies_ndof = multibody1.map(|m| m.0.ndofs()).unwrap_or(0)
            + multibody2.map(|m| m.0.ndofs()).unwrap_or(0);
        // For each solver contact we generate DIM constraints, and each constraints appends
        // the multibodies jacobian and weighted jacobians
        let required_jacobian_len =
            *jacobian_id + manifold.data.solver_contacts.len() * multibodies_ndof * 2 * DIM;

        // Grow the jacobian buffer to fit this constraint: `generate` runs serially in the
        // staged solver's pre-phase (before any worker starts), so growing here is race-free
        // (the old `!parallel` guard served the deleted parallel solver's shared buffer).
        if jacobians.nrows() < required_jacobian_len {
            jacobians.resize_vertically_mut(required_jacobian_len, 0.0);
        }

        let chunk_j_id = *jacobian_id;

        let manifold_points = &manifold.data.solver_contacts;
        out_constraint.dir1 = force_dir1;
        out_constraint.im1 = if type1.is_dynamic_or_kinematic() {
            mprops1.effective_inv_mass
        } else {
            Vector::ZERO
        };
        out_constraint.im2 = if type2.is_dynamic_or_kinematic() {
            mprops2.effective_inv_mass
        } else {
            Vector::ZERO
        };
        out_constraint.solver_vel1 = solver_vel1;
        out_constraint.solver_vel2 = solver_vel2;
        out_constraint.manifold_id = manifold_id;
        out_constraint.num_contacts = manifold_points.len() as u8;
        #[cfg(feature = "dim3")]
        {
            out_constraint.tangent1 = tangents1[0];
        }

        for k in 0..manifold_points.len() {
            let manifold_point = &manifold_points[k];
            // Reconstruct the world contact points and separation from the
            // body-local anchors (see `SolverContactGeneric::anchor1`) and the
            // bodies' current poses.
            let (p1, p2) = manifold
                .data
                .solver_contact_world_points(manifold_point, bodies);
            let dist = (p1 - p2).dot(force_dir1);

            let cid = (manifold_point.contact_id[0] & !crate::geometry::NEW_CONTACT_BIT) as usize;
            let pt_data = &manifold.points[cid].data;

            // World-space lever arms frozen at the pair's last full narrow-phase update
            // (anchor freezing, `ContactData::solver_dp1`); a world-attached side's
            // arm holds the absolute frozen point (`solver_contact_world_points` convention).
            let dp1 = pt_data.solver_dp1;
            let dp2 = pt_data.solver_dp2;
            let point =
                if manifold.data.relative_dominance > 0 || manifold.data.rigid_body1.is_none() {
                    dp1
                } else {
                    mprops1.world_com + dp1
                };

            let vel1 = vels1.linvel + vels1.angvel.gcross(dp1);
            let vel2 = vels2.linvel + vels2.angvel.gcross(dp2);

            out_constraint.limit = manifold.data.friction;
            out_constraint.manifold_contact_id[k] =
                (manifold_point.contact_id[0] & !crate::geometry::NEW_CONTACT_BIT) as u8;

            // Normal part.
            let restitution_seed;
            {
                let torque_dir1 = dp1.gcross(force_dir1);
                let torque_dir2 = dp2.gcross(-force_dir1);

                let ii_torque_dir1 = if type1.is_dynamic_or_kinematic() {
                    mprops1
                        .effective_world_inv_inertia
                        .transform_vector(torque_dir1)
                } else {
                    Default::default()
                };
                let ii_torque_dir2 = if type2.is_dynamic_or_kinematic() {
                    mprops2
                        .effective_world_inv_inertia
                        .transform_vector(torque_dir2)
                } else {
                    Default::default()
                };

                let inv_r1 = if let Some((mb1, link_id1)) = multibody1.as_ref() {
                    mb1.fill_jacobians(*link_id1, force_dir1, torque_dir1, jacobian_id, jacobians)
                        .0
                } else if type1.is_dynamic_or_kinematic() {
                    force_dir1.dot(mprops1.effective_inv_mass * force_dir1)
                        + ii_torque_dir1.gdot(torque_dir1)
                } else {
                    0.0
                };

                let inv_r2 = if let Some((mb2, link_id2)) = multibody2.as_ref() {
                    mb2.fill_jacobians(*link_id2, -force_dir1, torque_dir2, jacobian_id, jacobians)
                        .0
                } else if type2.is_dynamic_or_kinematic() {
                    force_dir1.dot(mprops2.effective_inv_mass * force_dir1)
                        + ii_torque_dir2.gdot(torque_dir2)
                } else {
                    0.0
                };

                let r = crate::utils::inv(inv_r1 + inv_r2);

                // Warm-start impulses and contact newness come from the manifold
                // point (they are not duplicated on the solver contacts).
                let is_new = pt_data.impulse == 0.0;
                let is_bouncy = crate::geometry::is_bouncy(manifold.data.restitution, is_new);

                restitution_seed =
                    (is_bouncy * manifold.data.restitution) * (vel1 - vel2).dot(force_dir1);

                out_constraint.normal_part[k] = ContactConstraintNormalPart {
                    torque_dir1,
                    torque_dir2,
                    ii_torque_dir1,
                    ii_torque_dir2,
                    rhs: Default::default(),
                    rhs_wo_bias: Default::default(),
                    cfm_factor: Default::default(),
                    // Minus the warm-start impulse, so the first substep's `update` cancels it
                    // when it folds `impulse` in: that value belongs to the previous step (see
                    // the coulomb-friction `generate`).
                    impulse_accumulator: -pt_data.warmstart_impulse,
                    impulse: pt_data.warmstart_impulse,
                    r,
                    #[cfg(feature = "block-solver")]
                    r_mat_elts: [0.0; 2],
                };
            }

            // Tangent parts.
            {
                // 3D: project the world-space friction warm-start onto the
                // current tangent basis (see `ContactData::warmstart_tangent_world`).
                #[cfg(feature = "dim3")]
                {
                    let w = pt_data.warmstart_tangent_world;
                    out_constraint.tangent_part[k].impulse =
                        TangentImpulse::new(w.gdot(tangents1[0]), w.gdot(tangents1[1]));
                }
                #[cfg(feature = "dim2")]
                {
                    out_constraint.tangent_part[k].impulse = pt_data.warmstart_tangent_impulse;
                }

                // See the normal part.
                out_constraint.tangent_part[k].impulse_accumulator =
                    -out_constraint.tangent_part[k].impulse;

                for j in 0..DIM - 1 {
                    let torque_dir1 = dp1.gcross(tangents1[j]);
                    let ii_torque_dir1 = if type1.is_dynamic_or_kinematic() {
                        mprops1
                            .effective_world_inv_inertia
                            .transform_vector(torque_dir1)
                    } else {
                        Default::default()
                    };
                    out_constraint.tangent_part[k].torque_dir1[j] = torque_dir1;
                    out_constraint.tangent_part[k].ii_torque_dir1[j] = ii_torque_dir1;

                    let torque_dir2 = dp2.gcross(-tangents1[j]);
                    let ii_torque_dir2 = if type2.is_dynamic_or_kinematic() {
                        mprops2
                            .effective_world_inv_inertia
                            .transform_vector(torque_dir2)
                    } else {
                        Default::default()
                    };
                    out_constraint.tangent_part[k].torque_dir2[j] = torque_dir2;
                    out_constraint.tangent_part[k].ii_torque_dir2[j] = ii_torque_dir2;

                    let tangent_glam = tangents1[j];
                    let inv_r1 = if let Some((mb1, link_id1)) = multibody1.as_ref() {
                        mb1.fill_jacobians(
                            *link_id1,
                            tangent_glam,
                            torque_dir1,
                            jacobian_id,
                            jacobians,
                        )
                        .0
                    } else if type1.is_dynamic_or_kinematic() {
                        tangent_glam.dot(mprops1.effective_inv_mass * tangent_glam)
                            + ii_torque_dir1.gdot(torque_dir1)
                    } else {
                        0.0
                    };

                    let inv_r2 = if let Some((mb2, link_id2)) = multibody2.as_ref() {
                        mb2.fill_jacobians(
                            *link_id2,
                            -tangent_glam,
                            torque_dir2,
                            jacobian_id,
                            jacobians,
                        )
                        .0
                    } else if type2.is_dynamic_or_kinematic() {
                        tangent_glam.dot(mprops2.effective_inv_mass * tangent_glam)
                            + ii_torque_dir2.gdot(torque_dir2)
                    } else {
                        0.0
                    };

                    let r = crate::utils::inv(inv_r1 + inv_r2);
                    let rhs_wo_bias = manifold_point.tangent_velocity.gdot(tangents1[j]);

                    out_constraint.tangent_part[k].rhs_wo_bias[j] = rhs_wo_bias;
                    out_constraint.tangent_part[k].rhs[j] = rhs_wo_bias;

                    // TODO: in 3D, we should take into account gcross[0].dot(gcross[1])
                    // in lhs. See the corresponding code on the `velocity_constraint.rs`
                    // file.
                    out_constraint.tangent_part[k].r[j] = r;
                }
            }

            // Builder. The substep anchors are the frozen per-body arms (each body's frozen
            // contact point rides its own rigid motion); the base separation is rebased so the
            // per-substep tracking is a pure delta from build-time poses.
            let point2 =
                if manifold.data.relative_dominance < 0 || manifold.data.rigid_body2.is_none() {
                    dp2
                } else {
                    mprops2.world_com + dp2
                };
            // Each anchor must live in the exact frame `update()` re-resolves it
            // with: a multibody link's rigid-body pose, a regular solver body's
            // CoM-centered pose, or the identity for a world-attached side
            // (`solver_vel == u32::MAX`, whose "local" anchor is the world point).
            let local_point = |is_multibody: bool,
                               solver_vel: u32,
                               rb: &crate::dynamics::RigidBody,
                               world_point: Vector| {
                if is_multibody {
                    rb.pos.position.inverse_transform_point(world_point)
                } else if solver_vel != u32::MAX {
                    rb.pos.position.rotation.inverse() * (world_point - rb.mprops.world_com)
                } else {
                    world_point
                }
            };
            let infos = CoulombContactPointInfos {
                local_p1: local_point(multibody1.is_some(), solver_vel1, rb1, point),
                local_p2: local_point(multibody2.is_some(), solver_vel2, rb2, point2),
                tangent_vel: manifold_point.tangent_velocity,
                dist: dist - (point - point2).dot(force_dir1),
                restitution_seed,
            };

            out_builder.handle1 = handle1;
            out_builder.handle2 = handle2;
            out_builder.infos[k] = infos;
            out_constraint.manifold_contact_id[k] =
                (manifold_point.contact_id[0] & !crate::geometry::NEW_CONTACT_BIT) as u8;
        }

        let ndofs1 = multibody1.map(|mb| mb.0.ndofs()).unwrap_or(0);
        let ndofs2 = multibody2.map(|mb| mb.0.ndofs()).unwrap_or(0);

        // NOTE: we use the generic constraint for non-dynamic bodies because this will
        //       reduce all ops to nothing because its ndofs will be zero.
        let generic_constraint_mask = (multibody1.is_some() as u8)
            | ((multibody2.is_some() as u8) << 1)
            | (!type1.is_dynamic_or_kinematic() as u8)
            | ((!type2.is_dynamic_or_kinematic() as u8) << 1);

        out_constraint.j_id = chunk_j_id;
        out_constraint.ndofs1 = ndofs1;
        out_constraint.ndofs2 = ndofs2;
        out_constraint.generic_constraint_mask = generic_constraint_mask;
    }

    pub fn update(
        &self,
        params: &IntegrationParameters,
        solved_dt: Real,
        bodies: &SolverBodies,
        multibodies: &MultibodyJointSet,
        constraint: &mut GenericContactConstraint,
    ) {
        // Contacts touching a fixed body use a stiffer "static" softness.
        // A world-attached side has solver-vel id `u32::MAX`; exclude multibody links, whose
        // pose comes from the multibody path rather than a solver body.
        let side1_static = constraint.solver_vel1 == u32::MAX
            && multibodies.rigid_body_link(self.handle1).is_none();
        let side2_static = constraint.solver_vel2 == u32::MAX
            && multibodies.rigid_body_link(self.handle2).is_none();
        let softness = if side1_static || side2_static {
            &params.static_contact_softness
        } else {
            &params.contact_softness
        };
        let cfm_factor = softness.cfm_factor(params.dt);
        let inv_dt = params.inv_dt();
        let erp_inv_dt = softness.erp_inv_dt(params.dt);

        // We don't update jacobians so the update is mostly identical to the non-generic velocity constraint.
        // Anchor frames must match `generate` exactly: identity for a world-attached side
        // (including sleeping multibody links, whose anchors are stored as world points),
        // the link pose for a live multibody side, the solver-body (CoM-centered) pose otherwise.
        let pose_for = |handle: RigidBodyHandle, solver_vel: u32, is_multibody: bool| {
            if solver_vel == u32::MAX {
                crate::math::Pose::IDENTITY
            } else if is_multibody {
                multibodies
                    .rigid_body_link(handle)
                    .map(|m| multibodies[m.multibody].link(m.id).unwrap().local_to_world)
                    .unwrap_or(crate::math::Pose::IDENTITY)
            } else {
                bodies.get_pose(solver_vel).pose()
            }
        };
        let pose1 = pose_for(
            self.handle1,
            constraint.solver_vel1,
            constraint.generic_constraint_mask & 0b01 != 0,
        );
        let pose2 = pose_for(
            self.handle2,
            constraint.solver_vel2,
            constraint.generic_constraint_mask & 0b10 != 0,
        );
        let all_infos = &self.infos[..constraint.num_contacts as usize];
        let normal_parts = &mut constraint.normal_part[..constraint.num_contacts as usize];
        let tangent_parts = &mut constraint.tangent_part[..constraint.num_contacts as usize];
        for part in tangent_parts.iter_mut() {
            part.maximum_dissipation = params.maximum_dissipation_friction;
        }

        #[cfg(feature = "dim2")]
        let tangents1 = constraint.dir1.orthonormal_basis();
        #[cfg(feature = "dim3")]
        let tangents1 = [
            constraint.tangent1,
            constraint.dir1.gcross(constraint.tangent1),
        ];

        let dir1_na = constraint.dir1;
        for ((info, normal_part), tangent_part) in all_infos
            .iter()
            .zip(normal_parts.iter_mut())
            .zip(tangent_parts.iter_mut())
        {
            // Tangent velocity is equivalent to the first body's surface moving artificially.
            let p1 = pose1 * info.local_p1 + info.tangent_vel * solved_dt;
            let p2 = pose2 * info.local_p2;
            let dist = info.dist + (p1 - p2).gdot(dir1_na);

            // Normal part.
            {
                // NOTE: `info.restitution_seed` is deliberately NOT part of
                // the substep rhs (see the coulomb kernel); it is applied once, after all
                // substeps, by `Self::apply_restitution`.
                let rhs_wo_bias = dist.max(0.0) * inv_dt;
                // No slop deadzone on the bias (see the coulomb kernel).
                let rhs_bias = (erp_inv_dt * dist).clamp(-params.max_corrective_velocity(), 0.0);
                let new_rhs = rhs_wo_bias + rhs_bias;

                normal_part.rhs_wo_bias = rhs_wo_bias;
                normal_part.rhs = new_rhs;
                // Bank the previous substep's impulse before the warm-start scaling (see the
                // coulomb-friction `update`).
                normal_part.impulse_accumulator += normal_part.impulse;
                normal_part.impulse *= params.warmstart_coefficient;
            }

            // Tangent part.
            {
                tangent_part.impulse_accumulator += tangent_part.impulse;
                tangent_part.impulse *= params.warmstart_coefficient;

                for j in 0..DIM - 1 {
                    let bias = (p1 - p2).gdot(tangents1[j]) * inv_dt;
                    tangent_part.rhs[j] = tangent_part.rhs_wo_bias[j] + bias;
                }
            }
        }

        constraint.cfm_factor = cfm_factor;
    }

    /// Does any active point of this constraint hold a restitution seed (an approaching
    /// bouncy contact captured at prepare time)? Cheap pre-check gating
    /// [`Self::apply_restitution`].
    pub fn has_bouncy_seed(&self, num_contacts: u8) -> bool {
        let num = (num_contacts as usize).min(MAX_MANIFOLD_POINTS);
        self.infos[..num]
            .iter()
            .any(|info| info.restitution_seed < 0.0)
    }

    /// End-of-step restitution pass (box2d-style): after all substeps, drive each bouncy
    /// point's normal velocity to its prepare-time `restitution * approach_velocity`, gated
    /// on the point having applied an impulse. Implemented by re-running the normal solve
    /// with `rhs = seed` on those points and a zeroed effective mass on the others.
    pub fn apply_restitution(
        &self,
        constraint: &mut GenericContactConstraint,
        jacobians: &DVector,
        bodies: &mut SolverBodies,
        generic_solver_vels: &mut DVector,
    ) {
        if !self.has_bouncy_seed(constraint.num_contacts) {
            return;
        }

        let mut any_gated = false;
        for (info, normal_part) in self.infos[..constraint.num_contacts as usize]
            .iter()
            .zip(constraint.normal_part[..constraint.num_contacts as usize].iter_mut())
        {
            if info.restitution_seed < 0.0 && normal_part.total_impulse() > 0.0 {
                normal_part.rhs = info.restitution_seed;
                any_gated = true;
            } else {
                normal_part.r = 0.0;
            }
        }

        if any_gated {
            constraint.cfm_factor = 1.0;
            constraint.solve(jacobians, bodies, generic_solver_vels, true, false);
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct GenericContactConstraint {
    /*
     * Fields specific to multibodies.
     */
    pub j_id: usize,
    pub ndofs1: usize,
    pub ndofs2: usize,
    pub generic_constraint_mask: u8,

    /*
     * Fields similar to the rigid-body constraints.
     */
    pub dir1: Vector, // Non-penetration force direction for the first body.
    #[cfg(feature = "dim3")]
    pub tangent1: Vector, // One of the friction force directions.
    pub im1: Vector,
    pub im2: Vector,
    pub cfm_factor: Real,
    pub limit: Real,
    pub solver_vel1: u32,
    pub solver_vel2: u32,
    pub manifold_id: ContactRef,
    pub manifold_contact_id: [u8; MAX_MANIFOLD_POINTS],
    pub num_contacts: u8,
    pub normal_part: [ContactConstraintNormalPart<Real>; MAX_MANIFOLD_POINTS],
    pub tangent_part: [ContactConstraintTangentPart<Real>; MAX_MANIFOLD_POINTS],
}

impl GenericContactConstraint {
    pub fn invalid() -> Self {
        Self {
            j_id: usize::MAX,
            ndofs1: usize::MAX,
            ndofs2: usize::MAX,
            generic_constraint_mask: u8::MAX,
            dir1: Vector::ZERO,
            #[cfg(feature = "dim3")]
            tangent1: Vector::ZERO,
            im1: Vector::ZERO,
            im2: Vector::ZERO,
            cfm_factor: 0.0,
            limit: 0.0,
            solver_vel1: u32::MAX,
            solver_vel2: u32::MAX,
            manifold_id: ContactRef::PADDING,
            manifold_contact_id: [u8::MAX; MAX_MANIFOLD_POINTS],
            num_contacts: u8::MAX,
            normal_part: [ContactConstraintNormalPart::zero(); MAX_MANIFOLD_POINTS],
            tangent_part: [ContactConstraintTangentPart::zero(); MAX_MANIFOLD_POINTS],
        }
    }

    /// Improve the cached initial guess with one local unbiased contact solve.
    /// Restore the physical biased rows; retain the newly projected impulses.
    pub fn project_warmstart(
        &mut self,
        jacobians: &DVector,
        bodies: &mut SolverBodies,
        generic_solver_vels: &mut DVector,
    ) {
        let rhs = self.normal_part.map(|p| (p.rhs, p.cfm_factor));
        let tangent_rhs = self.tangent_part.map(|p| p.rhs);
        let cfm = self.cfm_factor;
        for p in &mut self.normal_part {
            p.rhs = p.rhs_wo_bias;
            p.cfm_factor = 1.0;
        }
        for p in &mut self.tangent_part {
            p.rhs = p.rhs_wo_bias;
        }
        self.cfm_factor = 1.0;
        self.solve(jacobians, bodies, generic_solver_vels, true, true);
        for (p, (rhs, cfm)) in self.normal_part.iter_mut().zip(rhs) {
            p.rhs = rhs;
            p.cfm_factor = cfm;
        }
        for (p, rhs) in self.tangent_part.iter_mut().zip(tangent_rhs) {
            p.rhs = rhs;
        }
        self.cfm_factor = cfm;
    }

    pub fn warmstart(
        &mut self,
        jacobians: &DVector,
        bodies: &mut SolverBodies,
        generic_solver_vels: &mut DVector,
    ) {
        let mut solver_vel1 = if self.solver_vel1 == u32::MAX {
            GenericRhs::Fixed
        } else if self.generic_constraint_mask & 0b01 == 0 {
            GenericRhs::SolverVel(bodies.vels[self.solver_vel1 as usize])
        } else {
            GenericRhs::GenericId(self.solver_vel1)
        };

        let mut solver_vel2 = if self.solver_vel2 == u32::MAX {
            GenericRhs::Fixed
        } else if self.generic_constraint_mask & 0b10 == 0 {
            GenericRhs::SolverVel(bodies.vels[self.solver_vel2 as usize])
        } else {
            GenericRhs::GenericId(self.solver_vel2)
        };

        let tangent_parts = &mut self.tangent_part[..self.num_contacts as usize];
        let normal_parts = &mut self.normal_part[..self.num_contacts as usize];
        Self::generic_warmstart_group(
            normal_parts,
            tangent_parts,
            jacobians,
            self.dir1,
            #[cfg(feature = "dim3")]
            self.tangent1,
            self.im1,
            self.im2,
            self.ndofs1,
            self.ndofs2,
            self.j_id,
            &mut solver_vel1,
            &mut solver_vel2,
            generic_solver_vels,
        );

        if let GenericRhs::SolverVel(solver_vel1) = solver_vel1 {
            bodies.vels[self.solver_vel1 as usize] = solver_vel1;
        }

        if let GenericRhs::SolverVel(solver_vel2) = solver_vel2 {
            bodies.vels[self.solver_vel2 as usize] = solver_vel2;
        }
    }

    pub fn solve(
        &mut self,
        jacobians: &DVector,
        bodies: &mut SolverBodies,
        generic_solver_vels: &mut DVector,
        solve_restitution: bool,
        solve_friction: bool,
    ) {
        let mut solver_vel1 = if self.solver_vel1 == u32::MAX {
            GenericRhs::Fixed
        } else if self.generic_constraint_mask & 0b01 == 0 {
            GenericRhs::SolverVel(bodies.vels[self.solver_vel1 as usize])
        } else {
            GenericRhs::GenericId(self.solver_vel1)
        };

        let mut solver_vel2 = if self.solver_vel2 == u32::MAX {
            GenericRhs::Fixed
        } else if self.generic_constraint_mask & 0b10 == 0 {
            GenericRhs::SolverVel(bodies.vels[self.solver_vel2 as usize])
        } else {
            GenericRhs::GenericId(self.solver_vel2)
        };

        let normal_parts = &mut self.normal_part[..self.num_contacts as usize];
        let tangent_parts = &mut self.tangent_part[..self.num_contacts as usize];
        Self::generic_solve_group(
            self.cfm_factor,
            normal_parts,
            tangent_parts,
            jacobians,
            self.dir1,
            #[cfg(feature = "dim3")]
            self.tangent1,
            self.im1,
            self.im2,
            self.limit,
            self.ndofs1,
            self.ndofs2,
            self.j_id,
            &mut solver_vel1,
            &mut solver_vel2,
            generic_solver_vels,
            solve_restitution,
            solve_friction,
        );

        if let GenericRhs::SolverVel(solver_vel1) = solver_vel1 {
            bodies.vels[self.solver_vel1 as usize] = solver_vel1;
        }

        if let GenericRhs::SolverVel(solver_vel2) = solver_vel2 {
            bodies.vels[self.solver_vel2 as usize] = solver_vel2;
        }
    }

    pub fn writeback_impulses(&self, manifolds_all: &ManifoldStore) {
        // SAFETY: each generic constraint owns its manifold exclusively during
        //         writeback (generic constraints are solved/written serially).
        let manifold = unsafe { manifolds_all.get_mut(self.manifold_id) };

        #[cfg(feature = "dim3")]
        let tangent2 = self.dir1.gcross(self.tangent1);
        for k in 0..self.num_contacts as usize {
            let contact_id = self.manifold_contact_id[k];
            let active_contact = &mut manifold.points[contact_id as usize];
            // The stored impulses are serialized state: canonicalize signed zeros (see
            // `utils::canonicalize_zero`) so snapshots stay cross-platform deterministic.
            active_contact.data.warmstart_impulse =
                utils::canonicalize_zero(self.normal_part[k].impulse);
            active_contact.data.warmstart_tangent_impulse =
                utils::canonicalize_zero(self.tangent_part[k].impulse);
            #[cfg(feature = "dim3")]
            {
                let imp = active_contact.data.warmstart_tangent_impulse;
                active_contact.data.warmstart_tangent_world =
                    utils::canonicalize_zero(self.tangent1 * imp.x + tangent2 * imp.y);
            }
            active_contact.data.impulse =
                utils::canonicalize_zero(self.normal_part[k].total_impulse());
            active_contact.data.tangent_impulse =
                utils::canonicalize_zero(self.tangent_part[k].total_impulse());
        }
    }

    pub fn remove_cfm_and_bias_from_rhs(&mut self) {
        self.cfm_factor = 1.0;
        for normal_part in &mut self.normal_part {
            normal_part.rhs = normal_part.rhs_wo_bias;
        }
        for tangent_part in &mut self.tangent_part {
            tangent_part.rhs = tangent_part.rhs_wo_bias;
        }
    }
}

#[cfg(all(test, feature = "dim3"))]
mod warmstart_projection_tests {
    use super::*;
    use crate::alloc_prelude::vec;
    fn contact(m1: Real, m2: Real) -> GenericContactConstraint {
        let mut c = GenericContactConstraint::invalid();
        c.j_id = 0;
        c.ndofs1 = 0;
        c.ndofs2 = 0;
        c.generic_constraint_mask = 0;
        c.solver_vel1 = 0;
        c.solver_vel2 = 1;
        c.num_contacts = 1;
        c.dir1 = Vector::Y;
        c.tangent1 = Vector::X;
        c.im1 = Vector::splat(1. / m1);
        c.im2 = Vector::splat(1. / m2);
        c.cfm_factor = 0.31;
        c.limit = 1.;
        let r = 1. / (1. / m1 + 1. / m2);
        c.normal_part[0].r = r;
        c.normal_part[0].rhs = -8.;
        c.normal_part[0].rhs_wo_bias = 0.;
        c.normal_part[0].cfm_factor = 0.4;
        c.normal_part[0].impulse_accumulator = 0.37;
        c.tangent_part[0].r = [r, r, 0.];
        c.tangent_part[0].maximum_dissipation = true;
        c.tangent_part[0].rhs = [3., -7.];
        c
    }
    #[test]
    fn warmstart_projection_matches_inelastic_momentum_and_restores_biased_rows() {
        for m1 in [1., 1000., 1_000_000.] {
            let m2 = 1.;
            for seed in [0., 7., 123.] {
                let mut c = contact(m1, m2);
                let r = c.normal_part[0].r;
                c.normal_part[0].impulse = seed * r;
                c.tangent_part[0].impulse = na::vector![seed * r * 0.2, -seed * r * 0.1];
                let saved = c;
                let mut bodies = SolverBodies::default();
                bodies.resize(2);
                let v = Vector::new(0.3, -0.8, 0.2);
                bodies.vels[0].linear = v;
                let expected = v * (m1 / (m1 + m2));
                let jac = DVector::zeros(0);
                let mut generic = DVector::zeros(0);
                c.warmstart(&jac, &mut bodies, &mut generic);
                c.project_warmstart(&jac, &mut bodies, &mut generic);
                for body in &bodies.vels {
                    assert!(
                        (body.linear - expected).length() < 0.00003,
                        "m={m1},seed={seed},v={:?},expected={expected:?}",
                        body.linear
                    );
                }
                assert_eq!(c.cfm_factor, saved.cfm_factor);
                assert_eq!(c.normal_part[0].rhs, saved.normal_part[0].rhs);
                assert_eq!(c.normal_part[0].cfm_factor, saved.normal_part[0].cfm_factor);
                assert_eq!(
                    c.normal_part[0].impulse_accumulator,
                    saved.normal_part[0].impulse_accumulator
                );
                assert_eq!(c.tangent_part[0].rhs, saved.tangent_part[0].rhs);
                assert!(
                    c.tangent_part[0].impulse.norm() <= c.limit * c.normal_part[0].impulse + 1e-5
                );
                let kinetic = 0.5 * m1 as f64 * bodies.vels[0].linear.as_dvec3().length_squared()
                    + 0.5 * m2 as f64 * bodies.vels[1].linear.as_dvec3().length_squared();
                let reference = 0.5 * (m1 + m2) as f64 * expected.as_dvec3().length_squared();
                assert!((kinetic - reference).abs() < 0.00005 * (reference + 1.));
            }
        }
    }
    #[test]
    fn warmstart_projection_uses_generalized_inverse_mass_at_the_real_ratio() {
        for m1 in [1., 1000., 1_000_000.] {
            let m2 = 1.;
            let mut c = contact(m1, m2);
            c.ndofs1 = 1;
            c.generic_constraint_mask = 1;
            c.im1 = Vector::ZERO;
            c.limit = 0.;
            c.normal_part[0].impulse = 123. * c.normal_part[0].r;
            c.tangent_part[0].r = [m2, m2, 0.];
            let jac = DVector::from_vec(vec![1., 1. / m1, 0., 0., 0., 0.]);
            let mut generic = DVector::from_vec(vec![-0.8]);
            let mut bodies = SolverBodies::default();
            bodies.resize(2);
            bodies.vels[1].linear.y = 0.2;
            let expected = (-0.8 * m1 + 0.2 * m2) / (m1 + m2);
            c.warmstart(&jac, &mut bodies, &mut generic);
            c.project_warmstart(&jac, &mut bodies, &mut generic);
            assert!(
                (generic[0] - expected).abs() < 0.00003,
                "generalized mass {m1}: {} vs {expected}",
                generic[0]
            );
            assert!((bodies.vels[1].linear.y - expected).abs() < 0.00003);
        }
    }
}

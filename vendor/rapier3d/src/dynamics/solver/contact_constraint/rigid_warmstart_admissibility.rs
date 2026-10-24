//! Experimental passive initializer for rigid Coulomb contact guesses only.
//! The physical biased/unbiased solves following this initializer are unchanged.
use super::*;
use crate::dynamics::solver::SolverVel;
use crate::dynamics::{RigidBody, RigidBodyHandle};
use crate::math::{Rotation, Vector};
use glamx::{DQuat, DVec3};

#[derive(Clone, Copy)]
struct PhysicalMass {
    mass: f64,
    principal: DVec3,
    frame: DQuat,
}
impl PhysicalMass {
    fn new(body: &RigidBody, rotation: Rotation) -> Self {
        let local = &body.mass_properties().local_mprops;
        Self {
            mass: f64::from(body.mass()),
            principal: DVec3::from_array(local.principal_inertia().to_array().map(f64::from)),
            // Neither the rigid body's pose nor its cached world inertia tracks
            // every solver substep. Reconstruct the tensor at the solver pose.
            frame: DQuat::from_array(rotation.to_array().map(f64::from)).normalize()
                * DQuat::from_array(
                    local
                        .principal_inertia_local_frame
                        .to_array()
                        .map(f64::from),
                )
                .normalize(),
        }
    }
    fn dot(self, a: SolverVel<Real>, b: SolverVel<Real>) -> f64 {
        let linear = |v: SolverVel<Real>| DVec3::from_array(v.linear.to_array().map(f64::from));
        let angular = |v: SolverVel<Real>| {
            self.frame.inverse() * DVec3::from_array(v.angular.to_array().map(f64::from))
        };
        self.mass * linear(a).dot(linear(b)) + (self.principal * angular(a)).dot(angular(b))
    }
    fn change(self, before: SolverVel<Real>, after: SolverVel<Real>) -> f64 {
        // Subtract promoted velocities, not large kinetic energies or rounded
        // f32 velocity differences. This is the actual rounded-state delta.
        let a = DVec3::from_array(before.linear.to_array().map(f64::from));
        let b = DVec3::from_array(after.linear.to_array().map(f64::from));
        let wa = self.frame.inverse() * DVec3::from_array(before.angular.to_array().map(f64::from));
        let wb = self.frame.inverse() * DVec3::from_array(after.angular.to_array().map(f64::from));
        0.5 * (self.mass * (b - a).dot(b + a) + (self.principal * (wb - wa)).dot(wb + wa))
    }
}
fn lane(v: SolverVel<SimdReal>, index: usize) -> SolverVel<Real> {
    let mut result = SolverVel::zero();
    result.linear = Vector::from_array(core::array::from_fn(|axis| v.linear[axis].extract(index)));
    result.angular =
        Vector::from_array(core::array::from_fn(|axis| v.angular[axis].extract(index)));
    result
}
fn restore_lane(target: &mut SolverVel<SimdReal>, source: SolverVel<SimdReal>, index: usize) {
    for axis in 0..3 {
        target.linear[axis].replace(index, source.linear[axis].extract(index));
        target.angular[axis].replace(index, source.angular[axis].extract(index));
    }
}
fn finite(v: SolverVel<Real>) -> bool {
    v.linear.is_finite() && v.angular.is_finite()
}
fn same_velocity(a: SolverVel<Real>, b: SolverVel<Real>) -> bool {
    a.linear == b.linear && a.angular == b.angular
}
impl ContactWithCoulombFriction<SimdReal> {
    fn apply_current_guess(&mut self, a: &mut SolverVel<SimdReal>, b: &mut SolverVel<SimdReal>) {
        for normal in &mut self.normal_part[..self.num_contacts as usize] {
            normal.warmstart(&self.dir1, &self.im1, &self.im2, a, b);
        }
        let second = self.dir1.gcross(self.tangent1);
        for tangent in &mut self.tangent_part[..self.num_contacts as usize] {
            tangent.warmstart([&self.tangent1, &second], &self.im1, &self.im2, a, b);
        }
    }
    fn guess_in_cone(&self, index: usize) -> bool {
        let mu = f64::from(self.limit.extract(index));
        if !mu.is_finite() || mu < 0. {
            return false;
        }
        for point in 0..self.num_contacts as usize {
            let normal = f64::from(self.normal_part[point].impulse.extract(index));
            let x = f64::from(self.tangent_part[point].impulse[0].extract(index));
            let y = f64::from(self.tangent_part[point].impulse[1].extract(index));
            if !normal.is_finite()
                || !x.is_finite()
                || !y.is_finite()
                || normal < 0.
                || x.hypot(y) > mu * normal
            {
                return false;
            }
            if self.manifold_contact_id[point][index] == u8::MAX
                && (normal != 0. || x != 0. || y != 0.)
            {
                return false;
            }
        }
        true
    }
    fn scale_guess(&mut self, index: usize, scale: Real) {
        for point in 0..self.num_contacts as usize {
            let normal = &mut self.normal_part[point].impulse;
            normal.replace(
                index,
                if scale == 0. {
                    0.
                } else {
                    normal.extract(index) * scale
                },
            );
            for tangent in self.tangent_part[point].impulse.iter_mut() {
                tangent.replace(
                    index,
                    if scale == 0. {
                        0.
                    } else {
                        tangent.extract(index) * scale
                    },
                );
            }
        }
    }
    /// Opt-in physical-energy certification of the COMBINED warm-start and
    /// local unbiased projection. It changes a numerical initial guess, never
    /// clips the physical state produced by the following constraint solves.
    ///
    /// One scale covers every point and normal/tangent component of a lane.
    /// Historical accumulators, RHS, contact geometry and cadence are retained.
    /// Kinematics and world-attached/dominance sides provide no kinetic budget;
    /// prescribed-motion work is intentionally not credited to this passive
    /// initializer. Their velocities must remain exactly unchanged.
    pub(crate) fn warmstart_project_admissible(
        &mut self,
        solver: &mut SolverBodies,
        bodies: &RigidBodySet,
        island: &[RigidBodyHandle],
        multibodies: &MultibodyJointSet,
    ) {
        let ids = [self.solver_vel1, self.solver_vel2];
        // SIMD gathers can be unchecked: certify every live mapping first.
        for index in 0..SIMD_WIDTH {
            if self.manifold_id[index].is_padding() {
                assert_eq!(
                    [ids[0][index], ids[1][index]],
                    [u32::MAX; 2],
                    "padding lane cannot own a rigid solver target"
                );
            }
            for side in 0..2 {
                let id = ids[side][index];
                if id == u32::MAX {
                    continue;
                }
                assert!(
                    (id as usize) < solver.vels.len() && (id as usize) < solver.poses.len(),
                    "rigid warmstart solver id out of range"
                );
                let handle = *island
                    .get(id as usize)
                    .expect("admissible rigid warmstart requires an awake rigid island target");
                assert_eq!(
                    bodies[handle].ids.active_set_id, id,
                    "solver/body mapping changed"
                );
            }
        }
        let baseline = [solver.gather_vels(ids[0]), solver.gather_vels(ids[1])];
        let masses: [[Option<PhysicalMass>; 2]; SIMD_WIDTH] = core::array::from_fn(|index| {
            core::array::from_fn(|side| {
                let id = ids[side][index];
                if id == u32::MAX {
                    return None;
                }
                // Do not infer body handles from padded/soft solver slots.
                let handle = *island
                    .get(id as usize)
                    .expect("admissible rigid warmstart requires an awake rigid island target");
                let body = &bodies[handle];
                assert_eq!(body.ids.active_set_id, id, "solver/body mapping changed");
                assert!(
                    !body.is_sleeping() && !body.is_soft_frame(),
                    "unsupported sleeping/soft warmstart target"
                );
                assert!(
                    multibodies.rigid_body_link(handle).is_none(),
                    "rigid initializer cannot target a multibody slot"
                );
                assert!(
                    finite(lane(baseline[side], index)),
                    "nonfinite baseline velocity"
                );
                // Coloring only promises disjoint WRITABLE bodies. Shared
                // kinematics are prescribed read-only targets; their identical
                // baseline velocity must survive every lane's certification.
                if body.is_dynamic() {
                    for other_index in 0..=index {
                        for other_side in 0..2 {
                            if other_index < index || other_side < side {
                                assert_ne!(
                                    id, ids[other_side][other_index],
                                    "rigid warmstart lanes must own disjoint dynamic bodies"
                                );
                            }
                        }
                    }
                }
                body.is_dynamic()
                    .then(|| PhysicalMass::new(body, solver.poses[id as usize].rotation))
            })
        });
        self.warmstart(solver);
        self.project_warmstart(solver);
        // Reconstruct the response of COMPLETE final p, including every cross
        // term. The prior warm seed and cancellation rounding do not define p.
        let mut response = [SolverVel::zero(), SolverVel::zero()];
        let [a, b] = &mut response;
        self.apply_current_guess(a, b);
        let mut scales = [0.; SIMD_WIDTH];
        for (index, metrics) in masses.iter().enumerate() {
            let mut a = 0.;
            let mut b = 0.;
            for side in 0..2 {
                if let Some(mass) = metrics[side] {
                    let v = lane(baseline[side], index);
                    let delta = lane(response[side], index);
                    a += mass.dot(v, delta);
                    b += mass.dot(delta, delta);
                }
            }
            let scale = if self.manifold_id[index].is_padding()
                || !self.guess_in_cone(index)
                || !a.is_finite()
                || !b.is_finite()
                || b < 0.
            {
                0.
            } else if a + 0.5 * b <= 0. {
                1.
            } else if b > 0. {
                // Interior energy MINIMUM, not the fragile zero-work boundary.
                (-a / b).clamp(0., 1.) as Real
            } else {
                0.
            };
            scales[index] = scale;
            self.scale_guess(index, scale);
        }
        let mut result = baseline;
        let [a, b] = &mut result;
        self.apply_current_guess(a, b);
        for (index, metrics) in masses.iter().enumerate() {
            let mut change = 0.;
            let mut valid = self.guess_in_cone(index);
            for side in 0..2 {
                let before = lane(baseline[side], index);
                let after = lane(result[side], index);
                valid &= finite(after);
                if let Some(mass) = metrics[side] {
                    change += mass.change(before, after);
                } else {
                    valid &= same_velocity(before, after);
                }
            }
            if scales[index] == 0. || !valid || !change.is_finite() || change > 0. {
                // Exact rejection of this numerical guess, not a velocity cap.
                self.scale_guess(index, 0.);
                for side in 0..2 {
                    restore_lane(&mut result[side], baseline[side], index);
                }
            }
        }
        solver.scatter_vels(ids[0], result[0]);
        solver.scatter_vels(ids[1], result[1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::solver::solver_contact_graph::ContactRef;
    use crate::dynamics::{MassProperties, RigidBodyBuilder};
    use crate::geometry::ColliderSet;
    fn fixture(
        vy: Real,
    ) -> (
        ContactWithCoulombFriction<SimdReal>,
        SolverBodies,
        RigidBodySet,
        [RigidBodyHandle; SIMD_WIDTH],
    ) {
        let mut bodies = RigidBodySet::new();
        let mut solver = SolverBodies::default();
        solver.resize(SIMD_WIDTH);
        let handles = core::array::from_fn(|index| {
            let mut body = RigidBodyBuilder::dynamic()
                .additional_mass_properties(MassProperties::new(Vector::ZERO, 1., Vector::ONE))
                .build();
            body.recompute_mass_properties_from_colliders(&ColliderSet::new());
            let handle = bodies.insert(body);
            // Insertion resets internal solver references.
            bodies[handle].ids.active_set_id = index as u32;
            solver.copy_from(index, &bodies[handle]);
            solver.vels[index].linear.y = vy;
            handle
        });
        let mut c: ContactWithCoulombFriction<SimdReal> = unsafe { core::mem::zeroed() };
        let wide = SimdReal::splat;
        c.dir1 = na::vector![wide(0.), wide(1.), wide(0.)];
        c.tangent1 = na::vector![wide(1.), wide(0.), wide(0.)];
        c.im1 = na::Vector3::repeat(wide(1.));
        c.limit = wide(1.);
        c.cfm_factor = wide(0.3);
        c.num_contacts = 1;
        c.solver_vel1 = core::array::from_fn(|index| index as u32);
        c.solver_vel2 = [u32::MAX; SIMD_WIDTH];
        c.manifold_id = core::array::from_fn(|index| ContactRef {
            edge: index as u32,
            manifold: 0,
        });
        c.normal_part[0].torque_dir1 = na::vector![wide(0.), wide(0.), wide(1.)];
        c.normal_part[0].ii_torque_dir1 = c.normal_part[0].torque_dir1;
        c.normal_part[0].impulse = wide(1.);
        c.normal_part[0].impulse_accumulator = wide(7.);
        c.normal_part[0].r = wide(0.5);
        c.normal_part[0].cfm_factor = wide(0.3);
        c.normal_part[0].rhs = wide(-3.);
        let t = &mut c.tangent_part[0];
        t.maximum_dissipation = true;
        t.torque_dir1[0] = na::vector![wide(0.), wide(0.), wide(-1.)];
        t.ii_torque_dir1[0] = t.torque_dir1[0];
        t.torque_dir1[1] = na::vector![wide(-1.), wide(1.), wide(0.)];
        t.ii_torque_dir1[1] = t.torque_dir1[1];
        t.r = [wide(2.), wide(3.), wide(0.)];
        t.impulse[0] = wide(1.);
        t.impulse_accumulator[0] = wide(-5.);
        t.rhs = [wide(2.), wide(-4.)];
        (c, solver, bodies, handles)
    }
    #[test]
    fn coupled_one_contact_projects_positive_energy_from_a_feasible_seed() {
        let (mut c, mut solver, _, _) = fixture(0.);
        c.warmstart(&mut solver);
        c.project_warmstart(&mut solver);
        for index in 0..SIMD_WIDTH {
            let v = solver.vels[index];
            let energy = 0.5 * (v.linear.length_squared() + v.angular.length_squared());
            assert_eq!(energy, 0.1875, "old one-pass counterexample");
            assert_eq!(c.normal_part[0].impulse.extract(index), 0.5);
            assert_eq!(c.tangent_part[0].impulse[0].extract(index), 0.25);
        }
    }
    #[test]
    fn initializer_rejects_rest_uses_interior_and_retains_dissipative_guess() {
        for vy in [0., -0.25, -1.] {
            let (mut c, mut solver, bodies, handles) = fixture(vy);
            c.warmstart_project_admissible(
                &mut solver,
                &bodies,
                &handles,
                &MultibodyJointSet::new(),
            );
            for index in 0..SIMD_WIDTH {
                let v = solver.vels[index];
                let energy = 0.5 * (v.linear.length_squared() + v.angular.length_squared());
                assert!(energy <= 0.5 * vy * vy);
                assert!(c.guess_in_cone(index));
                assert_eq!(c.normal_part[0].impulse_accumulator.extract(index), 7.);
                assert_eq!(c.tangent_part[0].impulse_accumulator[0].extract(index), -5.);
                assert_eq!(c.normal_part[0].rhs.extract(index), -3.);
                assert_eq!(c.normal_part[0].cfm_factor.extract(index), 0.3);
                assert_eq!(c.tangent_part[0].rhs[0].extract(index), 2.);
                let expected = if vy == 0. {
                    0.
                } else if vy == -0.25 {
                    1. / 6.
                } else {
                    1.
                };
                assert!((c.normal_part[0].impulse.extract(index) - expected).abs() < 1e-6);
            }
        }
    }
    #[test]
    fn mixed_lanes_and_multiple_points_use_a_complete_manifold_response() {
        let (mut c, mut solver, bodies, handles) = fixture(0.);
        let (_, mut trial_solver, _, _) = fixture(0.);
        c.num_contacts = 2;
        c.normal_part[1] = c.normal_part[0];
        c.tangent_part[1] = c.tangent_part[0];
        c.manifold_contact_id[1] = [1; SIMD_WIDTH];
        for index in 0..SIMD_WIDTH {
            let vy = [0., -0.25, -1.][index % 3];
            solver.vels[index].linear.y = vy;
            trial_solver.vels[index].linear.y = vy;
        }
        let mut trial = c;
        trial.warmstart(&mut trial_solver);
        trial.project_warmstart(&mut trial_solver);
        c.warmstart_project_admissible(&mut solver, &bodies, &handles, &MultibodyJointSet::new());
        for index in 0..SIMD_WIDTH {
            let vy = [0., -0.25, -1.][index % 3];
            let v = solver.vels[index];
            assert!(
                0.5 * (v.linear.length_squared() + v.angular.length_squared())
                    <= 0.5 * vy * vy + 1e-8
            );
            assert!(c.guess_in_cone(index));
            let mut ratio: Option<Real> = None;
            for point in 0..2 {
                let original = trial.normal_part[point].impulse.extract(index);
                if original > 0. {
                    let next = c.normal_part[point].impulse.extract(index) / original;
                    if let Some(ratio) = ratio {
                        assert!((next - ratio).abs() < 1e-6);
                    }
                    ratio = Some(next);
                }
            }
        }
    }
    #[test]
    fn padding_and_prescribed_kinematic_sides_keep_their_baseline_velocity() {
        let (mut c, mut solver, mut bodies, handles) = fixture(-0.25);
        bodies[handles[0]].body_type = crate::dynamics::RigidBodyType::KinematicVelocityBased;
        solver.vels[0].linear = Vector::new(2., 3., 4.);
        for axis in 0..3 {
            c.im1[axis].replace(0, 0.);
        }
        for point in 0..c.num_contacts as usize {
            for axis in 0..3 {
                c.normal_part[point].ii_torque_dir1[axis].replace(0, 0.);
                for tangent in 0..2 {
                    c.tangent_part[point].ii_torque_dir1[tangent][axis].replace(0, 0.);
                }
            }
        }
        if SIMD_WIDTH > 1 {
            c.manifold_id[SIMD_WIDTH - 1] = ContactRef::PADDING;
            c.solver_vel1[SIMD_WIDTH - 1] = u32::MAX;
        }
        c.warmstart_project_admissible(&mut solver, &bodies, &handles, &MultibodyJointSet::new());
        assert_eq!(solver.vels[0].linear, Vector::new(2., 3., 4.));
        if SIMD_WIDTH > 1 {
            assert_eq!(solver.vels[SIMD_WIDTH - 1].linear.y, -0.25);
        }
    }
    #[test]
    fn shared_moving_kinematic_slot_is_not_a_writable_lane_conflict() {
        let (mut c, mut solver, mut bodies, handles) = fixture(-0.25);
        let prescribed = Vector::new(2., 0., 0.);
        let body = RigidBodyBuilder::kinematic_velocity_based()
            .linvel(prescribed)
            .build();
        let handle = bodies.insert(body);
        bodies[handle].ids.active_set_id = SIMD_WIDTH as u32;
        let mut island = crate::alloc_prelude::Vec::from(handles);
        island.push(handle);
        solver.resize(SIMD_WIDTH + 1);
        solver.copy_from(SIMD_WIDTH, &bodies[handle]);
        c.solver_vel2 = [SIMD_WIDTH as u32; SIMD_WIDTH];
        c.warmstart_project_admissible(&mut solver, &bodies, &island, &MultibodyJointSet::new());
        assert_eq!(solver.vels[SIMD_WIDTH].linear, prescribed);
        assert_eq!(solver.vels[SIMD_WIDTH].angular, Vector::ZERO);
        for index in 0..SIMD_WIDTH {
            let velocity = solver.vels[index];
            assert!(
                0.5 * (velocity.linear.length_squared() + velocity.angular.length_squared())
                    <= 0.03125 + 1e-8
            );
            assert!(c.guess_in_cone(index));
        }
    }
    #[test]
    #[should_panic(expected = "solver/body mapping changed")]
    fn stale_solver_body_mapping_is_refused_before_applying_a_guess() {
        let (mut c, mut solver, mut bodies, handles) = fixture(0.);
        bodies[handles[0]].ids.active_set_id = u32::MAX;
        c.warmstart_project_admissible(&mut solver, &bodies, &handles, &MultibodyJointSet::new());
    }
    #[test]
    #[should_panic(expected = "rigid warmstart solver id out of range")]
    fn out_of_range_solver_id_is_refused_before_simd_gather() {
        let (mut c, mut solver, bodies, handles) = fixture(0.);
        c.solver_vel1[0] = solver.vels.len() as u32 + 100;
        c.warmstart_project_admissible(&mut solver, &bodies, &handles, &MultibodyJointSet::new());
    }
    #[test]
    fn physical_metric_uses_current_pose_not_stale_body_world_inertia() {
        let mut body = RigidBodyBuilder::dynamic()
            .additional_mass_properties(MassProperties::new(
                Vector::ZERO,
                3.,
                Vector::new(1., 2., 4.),
            ))
            .build();
        body.recompute_mass_properties_from_colliders(&ColliderSet::new());
        let mass = PhysicalMass::new(
            &body,
            Rotation::from_rotation_z(core::f32::consts::FRAC_PI_2),
        );
        let mut v = SolverVel::zero();
        v.angular = Vector::X;
        assert!((mass.dot(v, v) - 2.).abs() < 1e-6);
        v.angular = Vector::Y;
        assert!((mass.dot(v, v) - 1.).abs() < 1e-6);
        v.linear = Vector::Z;
        assert!((mass.dot(v, v) - 4.).abs() < 1e-6);
    }
}

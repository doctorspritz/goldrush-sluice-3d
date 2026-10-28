//! Bounded, read-only fixed-point probes at the common pre-integration state.
//! Each constraint block starts from the same actual velocities. Probe changes
//! are residuals, not an applied sweep or additive physical work/energy.
use crate::dynamics::RigidBodyHandle;
use std::{cell::RefCell, vec::Vec};

/// Bounded observations for one explicitly selected diagnostic run.
#[derive(Default, Debug, serde::Serialize)]
pub struct Report {
    /// Retained pre-integration frames in execution order.
    pub frames: Vec<Frame>,
    /// Frames beyond the fixed 256-frame storage bound.
    pub dropped_frames: usize,
}
/// Independent fixed-point corrections for one constraint family.
#[derive(Default, Debug, serde::Serialize)]
pub struct Family {
    /// Actual operator blocks probed independently.
    pub blocks: usize,
    /// Non-padding contact points or scalar joint rows.
    pub rows: usize,
    /// Joint rows by Dof, Limit, Motor and Friction writeback kind, respectively.
    pub joint_kinds: [usize; 4],
    /// Contact units are N s; joint rows can mix linear/angular units.
    pub maximum_impulse_correction: f64,
    /// Largest contact impulse correction divided by its effective mass.
    pub maximum_normal_response_m_s: f64,
    /// Largest rigid linear velocity change from an isolated probe.
    pub maximum_linear_probe_m_s: f64,
    /// Largest rigid angular velocity change from an isolated probe.
    pub maximum_angular_probe_rad_s: f64,
    /// Generalized coordinates mix linear and angular units; not a speed norm.
    pub maximum_generalized_probe: f64,
}
/// A rigid contact point at the common pre-integration velocity state.
#[derive(Default, Debug, serde::Serialize)]
pub struct Point {
    /// Stable chunk, SIMD lane and local point for this solver step.
    pub key_chunk_lane_point: [usize; 3],
    /// Contact-graph edge and manifold ordinal.
    pub manifold: [u32; 2],
    /// Body raw handles including generation, if attached.
    pub bodies: [Option<(u32, u32)>; 2],
    /// Original manifold point identity.
    pub contact_id: u8,
    /// Registered collider raw handles including generation.
    pub colliders: [(u32, u32); 2],
    /// Whether the source pair uses clustered contact manifolds.
    pub clustered: bool,
    /// Frozen world force direction on side one.
    pub force_direction1: [f64; 3],
    /// Actual global solver CoM rotation before integration.
    pub rotation_before_xyzw: [[f64; 4]; 2],
    /// Actual global solver CoM rotation after integration.
    pub rotation_after_xyzw: Option<[[f64; 4]; 2]>,
    /// Signed normal impulse correction from a copied actual block.
    pub normal_correction_ns: f64,
    /// Magnitude of tangential impulse correction from that block.
    pub tangent_correction_ns: f64,
    /// Absolute normal correction divided by effective mass.
    pub normal_response_m_s: f64,
    /// Frozen row Jacobian applied to the actual global velocities.
    pub normal_velocity_m_s: f64,
    /// Actual biased normal right-hand side.
    pub rhs_m_s: f64,
    /// Accumulated impulse before the isolated probe.
    pub normal_impulse_ns: f64,
    /// Actual normal row effective mass, not inverse mass.
    pub effective_mass_kg: f64,
    /// Actual stored float32 normal regularization factor.
    pub cfm_factor: f64,
    /// Builder tracking witness gap at the current pre-integration poses.
    pub tracking_gap_before_m: f64,
    /// Builder tracking witness gap immediately after integration.
    pub tracking_gap_after_m: Option<f64>,
    /// Original material-witness gap before integration.
    pub material_gap_before_m: f64,
    /// Original material-witness gap after integration.
    pub material_gap_after_m: Option<f64>,
    /// Actual solver center-of-mass translations before integration.
    pub com_before_m: [[f64; 3]; 2],
    /// Actual solver center-of-mass translations after integration.
    pub com_after_m: Option<[[f64; 3]; 2]>,
    /// Actual global solver linear velocities before integration.
    pub linear_before_m_s: [[f64; 3]; 2],
    /// Actual global solver angular velocities before integration.
    pub angular_before_rad_s: [[f64; 3]; 2],
}
/// One completed biased solve, followed by its actual integration geometry.
#[derive(Default, Debug, serde::Serialize)]
pub struct Frame {
    /// Current solve-group ordinal.
    pub group: usize,
    /// Total solve groups in the island.
    pub groups: usize,
    /// Substep start time within this outer step.
    pub time_s: f32,
    /// Actual physical substep duration.
    pub dt_s: f32,
    /// Actual biased passes completed at these positions.
    pub pgs_iterations: usize,
    /// Whether the actual biased passes solve friction.
    pub biased_friction: bool,
    /// Twist contacts were selected and remain outside probe coverage.
    pub twist_not_probed: bool,
    /// Number of soft slots in this group, outside probe coverage.
    pub soft_slots_not_probed: usize,
    /// Rigid Coulomb manifold probes.
    pub rigid: Family,
    /// Multibody/generalized contact probes.
    pub generic_contacts: Family,
    /// Scalar rigid joint/limit/motor/friction rows.
    pub scalar_joints: Family,
    /// SIMD rigid joint/limit/motor/friction rows.
    pub simd_joints: Family,
    /// Multibody/generalized joint/limit/motor/friction rows.
    pub generic_joints: Family,
    /// Up to sixteen un-watched points ranked by normal response.
    pub top_rigid: Vec<Point>,
    /// All points for selected body pairs, bounded to 128 per frame.
    pub watched: Vec<Point>,
}
struct State {
    report: Report,
    pairs: Vec<[RigidBodyHandle; 2]>,
}
std::thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}
pub(crate) fn enabled() -> bool {
    STATE.with(|s| s.borrow().is_some())
}
pub(crate) fn pairs() -> Vec<[RigidBodyHandle; 2]> {
    STATE.with(|s| s.borrow().as_ref().unwrap().pairs.clone())
}
pub(crate) fn push(frame: Frame) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let r = &mut s.as_mut().unwrap().report;
        if r.frames.len() < 256 {
            r.frames.push(frame);
        } else {
            r.dropped_frames += 1;
        }
    });
}
pub(crate) fn finish(group: usize, time: f32, points: Vec<Point>) {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        let r = &mut s.as_mut().unwrap().report;
        if r.dropped_frames != 0 {
            return;
        }
        let f = r
            .frames
            .last_mut()
            .expect("integration without residual frame");
        assert_eq!(f.group, group);
        assert_eq!(f.time_s, time);
        for p in points {
            let a = f
                .watched
                .iter_mut()
                .find(|a| {
                    a.key_chunk_lane_point == p.key_chunk_lane_point && a.manifold == p.manifold
                })
                .expect("watched contact changed during integration");
            a.tracking_gap_after_m = Some(p.tracking_gap_before_m);
            a.material_gap_after_m = Some(p.material_gap_before_m);
            a.com_after_m = Some(p.com_before_m);
            a.rotation_after_xyzw = Some(p.rotation_before_xyzw);
        }
        assert!(f.watched.iter().all(|p| p.tracking_gap_after_m.is_some()));
    });
}
/// Select a separate, bounded observer without enabling the per-operation work
/// trace. Parallel and nested collectors are refused; unwinding clears TLS.
/// At most 256 substep frames and 16 ranked rigid points per frame are retained.
/// Soft/twist blocks are explicitly outside scope. No physical settings change.
pub fn record<R>(pairs: &[[RigidBodyHandle; 2]], run: impl FnOnce() -> R) -> (R, Report) {
    const {
        assert!(
            !cfg!(feature = "parallel"),
            "residual audit requires single-threaded Rapier"
        );
    }
    assert!(
        !super::enabled(),
        "work and residual collectors are mutually exclusive"
    );
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        assert!(s.is_none(), "nested residual recording");
        *s = Some(State {
            report: Report::default(),
            pairs: pairs.to_vec(),
        });
    });
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            STATE.with(|s| *s.borrow_mut() = None);
        }
    }
    let guard = Reset;
    let result = run();
    let report = STATE.with(|s| s.borrow_mut().take().unwrap().report);
    drop(guard);
    (result, report)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_recording_is_refused_and_panic_clears_state() {
        let failed = std::panic::catch_unwind(|| record(&[], || record(&[], || ())));
        assert!(failed.is_err());
        assert!(!enabled());
        let (answer, r) = record(&[], || 42);
        assert_eq!(answer, 42);
        assert!(r.frames.is_empty());
    }
    #[test]
    fn bounded_frames_report_truncation() {
        let (_, r) = record(&[], || {
            for _ in 0..300 {
                push(Frame::default());
            }
        });
        assert_eq!(r.frames.len(), 256);
        assert_eq!(r.dropped_frames, 44);
    }
}

#[cfg(test)]
mod exclusivity_tests {
    use super::*;
    #[test]
    fn work_and_residual_collectors_cannot_be_nested_in_either_order() {
        assert!(std::panic::catch_unwind(|| record(&[], || super::super::record(|| ()))).is_err());
        assert!(std::panic::catch_unwind(|| super::super::record(|| record(&[], || ()))).is_err());
        assert!(!enabled() && !super::super::enabled());
    }
}

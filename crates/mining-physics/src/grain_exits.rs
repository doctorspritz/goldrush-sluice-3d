//! Bounded, opt-in observations of existing solver-domain grain removals.
//! Records do not transport material or alter the removal/transfer decision.
use glam::Vec3;
use mining_sim::sluice::Grain;

/// Maximum retained grain records in one public hydraulic step.
pub const MAX_CAPTURE_LIMIT: usize = 4096;
/// Grain centre is below the solver domain's inclusive lower X boundary.
pub const NEG_X: u8 = 1;
/// Grain centre reaches the solver domain's exclusive upper X boundary.
/// This includes the outlet buffer; it is not the physical sluice-lip plane.
pub const POS_X: u8 = 2;
/// Grain centre reaches the solver domain's exclusive lower Y boundary.
pub const NEG_Y: u8 = 4;
/// Grain centre reaches the solver domain's exclusive upper Y boundary.
pub const POS_Y: u8 = 8;
/// Grain centre reaches the solver domain's exclusive lower Z boundary.
pub const NEG_Z: u8 = 16;
/// Grain centre reaches the solver domain's exclusive upper Z boundary.
pub const POS_Z: u8 = 32;

/// A nonphysical diagnostic input cannot be assigned a reliable exit face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassificationError {
    /// At least one position coordinate is NaN or infinite.
    NonFinitePosition,
    /// At least one domain extent is nonpositive, NaN or infinite.
    InvalidDomain,
}

/// Classify the exact existing bounds: X >= 0, Y > 0, Z > 0 and all upper
/// coordinates strictly below the supplied extents. Finite interior points
/// return zero. Corners can set multiple bits; nonfinite inputs are refused.
pub fn classify_exit(position: Vec3, domain_max_m: [f32; 3]) -> Result<u8, ClassificationError> {
    if !domain_max_m.iter().all(|v| v.is_finite() && *v > 0.) {
        return Err(ClassificationError::InvalidDomain);
    }
    if !position.is_finite() {
        return Err(ClassificationError::NonFinitePosition);
    }
    let mut faces = 0;
    for (outside, face) in [
        (position.x < 0., NEG_X),
        (position.x >= domain_max_m[0], POS_X),
        (position.y <= 0., NEG_Y),
        (position.y >= domain_max_m[1], POS_Y),
        (position.z <= 0., NEG_Z),
        (position.z >= domain_max_m[2], POS_Z),
    ] {
        if outside {
            faces |= face;
        }
    }
    Ok(faces)
}

/// Status of one diagnostic snapshot, independent of the material ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Capture was not enabled.
    Disabled,
    /// Enabled, but no public hydraulic step has begun.
    AwaitingStep,
    /// The public hydraulic step has not returned yet.
    InProgress,
    /// A successful public step did not execute the grain-removal stage.
    NoRemovalStage,
    /// The whole public step succeeded and every removal was captured.
    Complete,
    /// The step succeeded, but overflow or invalid diagnostic input occurred.
    Incomplete,
    /// The public step returned an error, including errors after removals.
    Failed,
    /// The previous snapshot has already been taken or explicitly cleared.
    Taken,
}
impl Status {
    /// Stable names for opt-in JSON receipts.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::AwaitingStep => "awaiting_step",
            Self::InProgress => "in_progress",
            Self::NoRemovalStage => "no_removal_stage",
            Self::Complete => "complete",
            Self::Incomplete => "incomplete",
            Self::Failed => "failed",
            Self::Taken => "taken",
        }
    }
}

/// Exact grain data immediately before the existing swap-remove and transfer.
#[derive(Clone, Debug)]
pub struct GrainExit {
    /// Complete grain value; no physical or optical resampling is applied.
    pub grain: Grain,
    /// Index immediately before this removal, not a persistent grain identity.
    pub grain_index: usize,
    /// Bitwise combination of the six solver-domain face constants.
    pub faces: u8,
}

/// Reusable output buffer. Reuse one value across takes to avoid allocations
/// after its first bounded reserve. The previous contents are replaced.
#[derive(Debug)]
pub struct Snapshot {
    /// Completion/failure status for the entire public hydraulic step.
    pub status: Status,
    /// Configured record bound, or zero when disabled.
    pub limit: usize,
    /// Session clock at the public step's entry, absent before a valid step.
    pub sim_time: Option<f64>,
    /// Actual buffered solver-domain upper bounds, not the sluice lip.
    pub domain_max_m: Option<[f32; 3]>,
    /// Whether the removal loop finished. A later failed step still has status
    /// Failed and is never a complete snapshot, even if this stage finished.
    pub removal_stage_completed: bool,
    /// Actual removal callbacks, including dropped and invalid records.
    pub observed: usize,
    /// Valid removal records omitted because the configured bound was reached.
    pub dropped: usize,
    /// Removal records refused for nonfinite, invalid-domain or interior input.
    pub invalid: usize,
    /// Whether the recorded clock and domain metadata were valid.
    pub metadata_valid: bool,
    exits: Vec<GrainExit>,
}
impl Default for Snapshot {
    fn default() -> Self {
        Self {
            status: Status::Disabled,
            limit: 0,
            sim_time: None,
            domain_max_m: None,
            removal_stage_completed: false,
            observed: 0,
            dropped: 0,
            invalid: 0,
            metadata_valid: true,
            exits: Vec::new(),
        }
    }
}
impl Snapshot {
    /// Captured records in the original removal order; no full count is implied
    /// unless `is_complete()` succeeds.
    pub fn exits(&self) -> &[GrainExit] {
        &self.exits
    }
    /// True only after a successful whole step and complete removal stage,
    /// without overflow, invalid metadata or refused records.
    pub fn is_complete(&self) -> bool {
        self.status == Status::Complete
            && self.removal_stage_completed
            && self.metadata_valid
            && self.dropped == 0
            && self.invalid == 0
            && self.observed == self.exits.len()
    }
    /// Discard the snapshot while retaining its bounded storage for reuse.
    pub fn clear(&mut self) {
        self.reset(
            if self.limit == 0 {
                Status::Disabled
            } else {
                Status::Taken
            },
            self.limit,
        );
    }
    fn reset(&mut self, status: Status, limit: usize) {
        self.status = status;
        self.limit = limit;
        self.sim_time = None;
        self.domain_max_m = None;
        self.removal_stage_completed = false;
        self.observed = 0;
        self.dropped = 0;
        self.invalid = 0;
        self.metadata_valid = true;
        self.exits.clear();
    }
    fn reserve(&mut self, limit: usize) -> Result<(), String> {
        if self.exits.capacity() < limit {
            self.exits
                .try_reserve_exact(limit - self.exits.len())
                .map_err(|e| format!("Grain-exit diagnostic allocation refused: {e}"))?;
        }
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct Capture {
    limit: usize,
    snapshot: Snapshot,
}
impl Capture {
    pub(crate) fn enable(&mut self, limit: usize) -> Result<(), String> {
        if !(1..=MAX_CAPTURE_LIMIT).contains(&limit) {
            return Err(format!(
                "Grain-exit capture limit must be in 1..={MAX_CAPTURE_LIMIT}"
            ));
        }
        self.snapshot.reserve(limit)?;
        self.limit = limit;
        self.snapshot.reset(Status::AwaitingStep, limit);
        Ok(())
    }
    pub(crate) fn disable(&mut self) {
        self.limit = 0;
        self.snapshot.reset(Status::Disabled, 0);
    }
    pub(crate) fn enabled(&self) -> bool {
        self.limit != 0
    }
    pub(crate) fn begin_step(&mut self, sim_time: f64) {
        if !self.enabled() {
            return;
        }
        self.snapshot.reset(Status::InProgress, self.limit);
        self.snapshot.metadata_valid = sim_time.is_finite();
        self.snapshot.sim_time = sim_time.is_finite().then_some(sim_time);
    }
    pub(crate) fn begin_removal_stage(&mut self, domain_max_m: [f32; 3]) {
        if !self.enabled() {
            return;
        }
        if domain_max_m.iter().all(|v| v.is_finite() && *v > 0.) {
            self.snapshot.domain_max_m = Some(domain_max_m);
        } else {
            self.snapshot.metadata_valid = false;
        }
    }
    pub(crate) fn record(&mut self, grain: &Grain, grain_index: usize, domain_max_m: [f32; 3]) {
        if !self.enabled() {
            return;
        }
        self.snapshot.observed += 1;
        let Ok(faces) = classify_exit(grain.position, domain_max_m) else {
            self.snapshot.invalid += 1;
            return;
        };
        if faces == 0 {
            self.snapshot.invalid += 1;
        } else if self.snapshot.exits.len() >= self.limit {
            self.snapshot.dropped += 1;
        } else {
            // Enable/take reserve the complete bound outside the removal loop.
            // Disabled/overflow paths never clone the grain or grow a buffer.
            self.snapshot.exits.push(GrainExit {
                grain: grain.clone(),
                grain_index,
                faces,
            });
        }
    }
    pub(crate) fn complete_removal_stage(&mut self) {
        if !self.enabled() {
            return;
        }
        self.snapshot.removal_stage_completed = true;
        self.snapshot.metadata_valid &= self.snapshot.domain_max_m.is_some();
    }
    pub(crate) fn finish_step(&mut self, success: bool) {
        if !self.enabled() {
            return;
        }
        self.snapshot.status = if !success {
            Status::Failed
        } else if !self.snapshot.metadata_valid
            || self.snapshot.dropped > 0
            || self.snapshot.invalid > 0
        {
            Status::Incomplete
        } else if self.snapshot.removal_stage_completed {
            Status::Complete
        } else {
            Status::NoRemovalStage
        };
    }
    pub(crate) fn take_into(&mut self, out: &mut Snapshot) -> Result<(), String> {
        // Reserve before replacing either snapshot. Allocation failure retains
        // the pending observation, and can never alter physical custody.
        out.reserve(self.limit)?;
        out.exits.clear();
        std::mem::swap(out, &mut self.snapshot);
        self.snapshot.reset(
            if self.enabled() {
                Status::Taken
            } else {
                Status::Disabled
            },
            self.limit,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat3, Quat};
    use mining_sim::{GrainKind, Material};

    fn grain(position: Vec3) -> Grain {
        Grain {
            volume_resolved: true,
            kind: GrainKind::Sand,
            position,
            velocity: Vec3::new(0.1, -0.2, 0.3),
            material: Material {
                soil_m3: 2e-6,
                sand_m3: 3e-6,
                gravel_m3: 4e-6,
                gold_g: 0.0007,
                pore_water_m3: 1e-8,
            },
            resting: false,
            diameter_m: 0.00015,
            affine: Mat3::from_cols(Vec3::X, Vec3::Y * 2., Vec3::Z * 3.),
            rotation: Quat::from_rotation_y(0.37),
            angular_velocity: Vec3::new(0.4, 0.5, 0.6),
        }
    }
    #[test]
    fn exact_solver_bounds_and_corner_faces_are_preserved() {
        let domain = [1., 2., 3.];
        for x in [0., -0., f32::from_bits(1f32.to_bits() - 1)] {
            assert_eq!(classify_exit(Vec3::new(x, 1., 1.), domain), Ok(0));
        }
        for (p, mask) in [
            (Vec3::new(-f32::MIN_POSITIVE, 1., 1.), NEG_X),
            (Vec3::new(1., 1., 1.), POS_X),
            (Vec3::new(0.5, 0., 1.), NEG_Y),
            (Vec3::new(0.5, -0., 1.), NEG_Y),
            (Vec3::new(0.5, 2., 1.), POS_Y),
            (Vec3::new(0.5, 1., 0.), NEG_Z),
            (Vec3::new(0.5, 1., -0.), NEG_Z),
            (Vec3::new(0.5, 1., 3.), POS_Z),
            (Vec3::new(-1., 0., 3.), NEG_X | NEG_Y | POS_Z),
            (Vec3::new(1., 2., 3.), POS_X | POS_Y | POS_Z),
        ] {
            assert_eq!(classify_exit(p, domain), Ok(mask));
        }
        assert_eq!(
            classify_exit(Vec3::new(0., f32::MIN_POSITIVE, f32::MIN_POSITIVE), domain),
            Ok(0)
        );
    }
    #[test]
    fn nonfinite_points_and_invalid_domains_are_refused() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for axis in 0..3 {
                let mut p = Vec3::splat(0.5);
                p[axis] = bad;
                assert_eq!(
                    classify_exit(p, [1.; 3]),
                    Err(ClassificationError::NonFinitePosition)
                );
            }
        }
        for bad in [0., -1., f32::NAN, f32::INFINITY] {
            assert_eq!(
                classify_exit(Vec3::ONE, [bad, 2., 3.]),
                Err(ClassificationError::InvalidDomain)
            );
        }
    }
    #[test]
    fn default_disabled_capture_does_no_work_or_allocation() {
        let mut c = Capture::default();
        let mut out = Snapshot::default();
        c.begin_step(1.);
        c.begin_removal_stage([1.; 3]);
        c.record(&grain(Vec3::ONE), 0, [1.; 3]);
        c.complete_removal_stage();
        c.finish_step(true);
        c.take_into(&mut out).unwrap();
        assert_eq!(out.status, Status::Disabled);
        assert!(!out.is_complete());
        assert_eq!(out.observed, 0);
        assert_eq!(out.exits.capacity(), 0);
        assert_eq!(c.snapshot.exits.capacity(), 0);
        assert!(c.enable(0).is_err());
        assert!(c.enable(MAX_CAPTURE_LIMIT + 1).is_err());
        assert!(!c.enabled());
    }
    #[test]
    fn whole_step_failure_overrides_completed_removal_and_each_step_resets() {
        let mut c = Capture::default();
        let mut out = Snapshot::default();
        c.enable(2).unwrap();
        c.take_into(&mut out).unwrap();
        assert_eq!(out.status, Status::AwaitingStep);
        c.begin_step(1.);
        c.finish_step(true);
        c.take_into(&mut out).unwrap();
        assert_eq!(out.status, Status::NoRemovalStage);
        assert!(!out.is_complete());
        c.begin_step(2.);
        c.begin_removal_stage([1.; 3]);
        c.complete_removal_stage();
        c.finish_step(true);
        c.take_into(&mut out).unwrap();
        assert!(out.is_complete());
        assert!(out.exits().is_empty());
        c.begin_step(3.);
        c.begin_removal_stage([1.; 3]);
        c.record(&grain(Vec3::ONE), 7, [1.; 3]);
        c.complete_removal_stage();
        c.finish_step(false);
        c.take_into(&mut out).unwrap();
        assert_eq!(out.status, Status::Failed);
        assert!(!out.is_complete());
        assert!(out.removal_stage_completed);
        assert_eq!(out.exits().len(), 1);
        c.take_into(&mut out).unwrap();
        assert_eq!(out.status, Status::Taken);
        assert!(out.exits().is_empty());
        c.begin_step(4.);
        c.finish_step(false);
        c.take_into(&mut out).unwrap();
        assert_eq!(out.status, Status::Failed);
        assert!(!out.removal_stage_completed);
        assert_eq!(out.observed, 0);
        c.disable();
        c.take_into(&mut out).unwrap();
        assert_eq!(out.status, Status::Disabled);
    }
    #[test]
    fn overflow_and_invalid_records_are_counted_without_complete_status() {
        let mut c = Capture::default();
        let mut out = Snapshot::default();
        c.enable(2).unwrap();
        c.begin_step(1.);
        c.begin_removal_stage([1.; 3]);
        for i in 0..3 {
            c.record(&grain(Vec3::ONE), i, [1.; 3]);
        }
        c.record(&grain(Vec3::splat(0.5)), 3, [1.; 3]);
        c.record(&grain(Vec3::splat(f32::NAN)), 4, [1.; 3]);
        c.complete_removal_stage();
        c.finish_step(true);
        c.take_into(&mut out).unwrap();
        assert_eq!(out.status, Status::Incomplete);
        assert!(!out.is_complete());
        assert_eq!(
            (out.observed, out.exits().len(), out.dropped, out.invalid),
            (5, 2, 1, 2)
        );
        assert_eq!(out.exits()[0].grain_index, 0);
        assert_eq!(out.exits()[1].grain_index, 1);
        c.begin_step(f64::NAN);
        c.begin_removal_stage([1.; 3]);
        c.complete_removal_stage();
        c.finish_step(true);
        c.take_into(&mut out).unwrap();
        assert_eq!(out.status, Status::Incomplete);
        assert!(!out.metadata_valid);
        assert_eq!(out.sim_time, None);
    }
    #[test]
    fn copies_preserve_every_grain_field_and_take_clear_reuse_storage() {
        let mut c = Capture::default();
        let mut out = Snapshot::default();
        c.enable(2).unwrap();
        let mut g = grain(Vec3::new(1., -0., 0.5));
        let source = serde_json::to_vec(&g).unwrap();
        for tick in 0..4 {
            c.begin_step(tick as f64);
            c.begin_removal_stage([1.; 3]);
            c.record(&g, 17, [1.; 3]);
            c.complete_removal_stage();
            c.finish_step(true);
            let pointer = c.snapshot.exits.as_ptr();
            c.take_into(&mut out).unwrap();
            assert_eq!(out.exits.as_ptr(), pointer);
            assert!(out.is_complete());
            assert_eq!(out.exits()[0].faces, POS_X | NEG_Y);
            assert_eq!(serde_json::to_vec(&out.exits()[0].grain).unwrap(), source);
            assert_eq!(serde_json::to_vec(&g).unwrap(), source);
            assert!(
                c.snapshot.exits.capacity() >= 2
                    && c.snapshot.exits.capacity() <= MAX_CAPTURE_LIMIT
            );
            let capacity = out.exits.capacity();
            out.clear();
            assert_eq!(out.exits.capacity(), capacity);
        }
        c.begin_step(5.);
        c.begin_removal_stage([1.; 3]);
        c.record(&g, 0, [1.; 3]);
        c.complete_removal_stage();
        c.finish_step(true);
        c.take_into(&mut out).unwrap();
        g.material.soil_m3 = 0.;
        assert_ne!(serde_json::to_vec(&g).unwrap(), source);
        assert_eq!(serde_json::to_vec(&out.exits()[0].grain).unwrap(), source);
    }
}

//! A compound is one rigid solid's UNION, not a sum of penalty volumes. Claim
//! overlap with component-wise inclusion-exclusion. Internal steel seams
//! cannot add a second pressure response or fill a hollow cavity.
use super::*;
const MAX_DEPTH: usize = 32;
const MAX_VISITS: usize = 1_048_576;
pub(super) fn accumulate(
    result: &mut Response,
    m: Moments,
    p: Affine,
    am: RigidMotion,
    bm: RigidMotion,
    delta: DVec3,
    energy: Option<f64>,
) -> Result<(), String> {
    if let Some(energy) = energy {
        super::accumulate_energy(result, 0, m, p, am, DVec3::ZERO, energy)?;
    } else {
        super::accumulate(result, 0, m, p, am, DVec3::ZERO)?;
    }
    let arm_b = m.first - delta * m.volume;
    result.forces_n[1] -= p.gradient * m.volume;
    result.torques_nm[1] -= arm_b.cross(p.gradient);
    result.potential_rate_w += p
        .gradient
        .dot(bm.linear_velocity_m_s * m.volume + bm.angular_velocity_rad_s.cross(arm_b));
    Ok(())
}
mod exact;
pub(super) fn union(
    ac: &[WorldCell],
    bc: &[WorldCell],
    am: RigidMotion,
    bm: RigidMotion,
    delta: DVec3,
) -> Result<Response, String> {
    exact::union(ac, bc, am, bm, delta)
}

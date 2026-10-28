//! Finite pressure-field elastic contact on complete registered solid geometry.
//!
//! This component integrates a specified pressure potential over finite overlap.
//! It does not choose real-soil parameters, modify bodies, integrate motion or
//! select Session/retained-world/save policies. Friction, yield and damping are
//! outside this component. Positive-volume pressure ties report a nonsmooth
//! state and use the supplied velocity's one-sided material derivative.
//!
//! Model background: Elandt et al., https://arxiv.org/abs/1904.11433 .
use super::RigidMotion;
use glam::DVec3;
mod geometry;
mod poly;
mod rigid;
mod terrain;
pub use geometry::{PressureBody, RegisteredSolid};
use poly::{Affine, Moments, Poly, WorldCell};
pub use terrain::RegisteredTerrainLayer;

/// A compliant or rigid finite opposing solid; never an unbounded face plane.
pub enum Opposing<'a> {
    Rigid(&'a RegisteredSolid),
    /// Complete closed parts of ONE rigid body. Overlapping steel is integrated once.
    RigidCompound(&'a [RegisteredSolid]),
    /// Finite erodible support envelope between matching registered surfaces.
    /// Bedrock beneath this layer is a separate contact domain.
    TerrainLayer(&'a RegisteredTerrainLayer),
    Compliant(&'a PressureBody),
}
#[derive(Default, Debug)]
pub struct Response {
    pub elastic_energy_j: f64,
    pub forces_n: [DVec3; 2],
    pub torques_nm: [DVec3; 2],
    pub overlap_volume_m3: f64,
    pub weak_pressure_volume_m3: [f64; 2],
    /// Equal affine pressure over positive volume is nonsmooth. Its directional
    /// derivative is selected by pressure-rate ordering, with a midpoint only
    /// when those rates also coincide. This is not viscous dissipation.
    pub directional_tie_volume_m3: f64,
    pub potential_rate_w: f64,
    pub body_power_w: f64,
    pub power_residual_w: f64,
    pub force_residual_n: DVec3,
    pub moment_residual_nm: DVec3,
    pub tested_tetrahedron_pairs: usize,
    pub intersecting_tetrahedron_pairs: usize,
    pub union_intersection_visits: usize,
    pub union_maximum_depth: usize,
    pub union_energy_roundoff_estimate_j: f64,
    pub union_volume_roundoff_estimate_m3: f64,
    pub union_input_tetrahedra: usize,
    pub union_candidate_tetrahedra: usize,
}
fn cells(body: &PressureBody, motion: RigidMotion, shift: DVec3) -> Result<Vec<WorldCell>, String> {
    body.cells
        .iter()
        .map(|c| {
            WorldCell::new(
                c.vertices.map(|p| shift + motion.rotation * p),
                Some(Affine {
                    origin: shift + motion.rotation * c.field.origin,
                    value: c.field.value,
                    gradient: motion.rotation * c.field.gradient,
                    triangle: c
                        .field
                        .triangle
                        .map(|t| t.map(|p| shift + motion.rotation * p)),
                    triangle_scale: c.field.triangle_scale,
                }),
            )
        })
        .collect()
}
fn accumulate(
    result: &mut Response,
    body: usize,
    m: Moments,
    p: Affine,
    motion: RigidMotion,
    com: DVec3,
) -> Result<(), String> {
    accumulate_energy(result, body, m, p, motion, com, m.pressure(p))
}
fn accumulate_energy(
    result: &mut Response,
    body: usize,
    m: Moments,
    p: Affine,
    motion: RigidMotion,
    com: DVec3,
    energy: f64,
) -> Result<(), String> {
    if m.volume == 0. {
        return Ok(());
    }
    if !energy.is_finite() || energy < 0. {
        return Err(format!(
            "Pressure energy is not representable/nonnegative: {energy}"
        ));
    }
    let arm = m.first - com * m.volume;
    result.elastic_energy_j += energy;
    result.weak_pressure_volume_m3[body] += m.volume;
    result.forces_n[body] += p.gradient * m.volume;
    result.torques_nm[body] += arm.cross(p.gradient);
    result.potential_rate_w -= p
        .gradient
        .dot(motion.linear_velocity_m_s * m.volume + motion.angular_velocity_rad_s.cross(arm));
    Ok(())
}
/// Evaluate both finite solids in a nearby world-oriented frame. Registered
/// material coordinates must use the body's CoM frame. Current unit rotations
/// and twists are validated, never normalized or silently reconstructed.
/// U = integral_overlap min(p_A,p_B) dV; rigid B uses p_B = infinity.
pub fn evaluate(
    a: &PressureBody,
    am: RigidMotion,
    b: Opposing<'_>,
    bm: RigidMotion,
) -> Result<Response, String> {
    am.validate()?;
    bm.validate()?;
    let delta = bm.center_of_mass_m - am.center_of_mass_m;
    let ac = cells(a, am, DVec3::ZERO)?;
    let rigid = !matches!(b, Opposing::Compliant(_));
    let compound = matches!(b, Opposing::RigidCompound(_) | Opposing::TerrainLayer(_));
    let terrain_total = match &b {
        Opposing::TerrainLayer(layer) => Some(layer.tetrahedra()),
        _ => None,
    };
    let bc = match b {
        Opposing::TerrainLayer(layer) => {
            layer.cells(&ac, bm, delta, 1_048_576 / ac.len().max(1))?
        }
        Opposing::Compliant(b) => cells(b, bm, delta)?,
        Opposing::Rigid(b) => b
            .tets
            .iter()
            .map(|t| WorldCell::new(t.map(|p| delta + bm.rotation * p), None))
            .collect::<Result<Vec<_>, _>>()?,
        Opposing::RigidCompound(parts) => {
            if parts.is_empty() || parts.len() > 1024 {
                return Err("Invalid finite rigid compound part count".into());
            }
            let count = parts
                .iter()
                .try_fold(0usize, |sum, part| sum.checked_add(part.tets.len()))
                .ok_or("Finite pressure pair work bound exceeded")?;
            if ac.len().checked_mul(count).is_none_or(|n| n > 1_048_576) {
                return Err("Finite pressure pair work bound exceeded".into());
            }
            let mut cells = Vec::with_capacity(count);
            for (component, part) in parts.iter().enumerate() {
                for t in &part.tets {
                    let mut cell = WorldCell::new(t.map(|p| delta + bm.rotation * p), None)?;
                    cell.rigid_component = component;
                    cells.push(cell);
                }
            }
            cells
        }
    };
    if ac.len().checked_mul(bc.len()).is_none_or(|n| n > 1_048_576) {
        return Err("Finite pressure pair work bound exceeded".into());
    }
    let mut result = if compound {
        rigid::union(&ac, &bc, am, bm, delta)?
    } else {
        Response::default()
    };
    if let Some(total) = terrain_total {
        result.union_input_tetrahedra = total;
    }
    if !compound {
        for a in &ac {
            for b in &bc {
                result.tested_tetrahedron_pairs += 1;
                if !a.intersects(b) {
                    continue;
                }
                let mut overlap = Poly::tetrahedron(a.tet);
                for plane in b.planes {
                    overlap = overlap.clip(plane)?;
                }
                let whole = overlap
                    .moments()
                    .map_err(|e| format!("FULL_OVERLAP a={:?} b={:?}: {e}", a.tet, b.tet))?;
                if whole.volume == 0. {
                    continue;
                }
                result.intersecting_tetrahedron_pairs += 1;
                result.overlap_volume_m3 += whole.volume;
                let pa = a.pressure.unwrap();
                if rigid {
                    rigid::accumulate(&mut result, whole, pa, am, bm, delta, None)?;
                } else {
                    let pb = b.pressure.unwrap();
                    let origin = overlap.points[0];
                    let mut split = pa.difference(pb, origin);
                    if split.gradient == DVec3::ZERO && split.value == 0. {
                        result.directional_tie_volume_m3 += whole.volume;
                        split = pa
                            .rate(am, DVec3::ZERO, origin)
                            .difference(pb.rate(bm, delta, origin), origin);
                        if split.gradient == DVec3::ZERO && split.value == 0. {
                            accumulate(&mut result, 0, whole.scaled(0.5), pa, am, DVec3::ZERO)?;
                            accumulate(&mut result, 1, whole.scaled(0.5), pb, bm, delta)?;
                            continue;
                        }
                    }
                    let weak_a = overlap
                        .clone()
                        .clip(split)?
                        .moments()
                        .map_err(|e| format!("WEAK_A split={split:?}: {e}"))?;
                    let weak_b = overlap
                        .clip(split.negative())?
                        .moments()
                        .map_err(|e| format!("WEAK_B split={split:?}: {e}"))?;
                    accumulate(&mut result, 0, weak_a, pa, am, DVec3::ZERO)?;
                    accumulate(&mut result, 1, weak_b, pb, bm, delta)?;
                }
            }
        }
    }
    result.body_power_w = result.forces_n[0].dot(am.linear_velocity_m_s)
        + result.forces_n[1].dot(bm.linear_velocity_m_s)
        + result.torques_nm[0].dot(am.angular_velocity_rad_s)
        + result.torques_nm[1].dot(bm.angular_velocity_rad_s);
    result.power_residual_w = result.potential_rate_w + result.body_power_w;
    result.force_residual_n = result.forces_n[0] + result.forces_n[1];
    result.moment_residual_nm =
        result.torques_nm[0] + result.torques_nm[1] + delta.cross(result.forces_n[1]);
    if ![
        result.elastic_energy_j,
        result.overlap_volume_m3,
        result.potential_rate_w,
        result.body_power_w,
        result.power_residual_w,
        result.union_energy_roundoff_estimate_j,
        result.union_volume_roundoff_estimate_m3,
    ]
    .iter()
    .all(|v| v.is_finite())
        || result
            .forces_n
            .iter()
            .chain(&result.torques_nm)
            .any(|v| !v.is_finite())
    {
        return Err("Finite pressure response overflow".into());
    }
    Ok(result)
}
#[cfg(test)]
mod tests;

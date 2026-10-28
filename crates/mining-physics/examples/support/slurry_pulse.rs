//! Finite quartz pulses and read-only assays for transport qualification.
//! Quadrature refines parcel weight without changing the declared PSD or its
//! first two spatial moments. This module supplies no transport or optics law.
use glam::{DQuat, DVec3, EulerRot, Mat3, Quat, Vec3};
use mining_sim::{sluice::Grain, GrainKind, Material};
use serde_json::{json, Value};

pub const DIAMETERS_M: [f32; 3] = [47e-6, 97e-6, 173e-6];
pub const TOTAL_SOLID_M3: f64 = 3e-8;
const SCALE_M: DVec3 = DVec3::new(0.009, 0.003, 0.035);
const ROTATIONS_XYZ: [[f64; 3]; 3] = [
    [0.31, 0.47, 0.23],
    [0.61, -0.29, 0.83],
    [-0.43, 0.79, -0.37],
];

pub struct Pulse {
    pub grains: Vec<Grain>,
    pub report: Value,
}

fn rule(order: usize) -> Result<Vec<(f64, f64)>, String> {
    match order {
        2 => {
            let x = 1. / 3_f64.sqrt();
            Ok(vec![(-x, 0.5), (x, 0.5)])
        }
        4 => {
            let a = ((3. + 2. * (6_f64 / 5.).sqrt()) / 7.).sqrt();
            let b = ((3. - 2. * (6_f64 / 5.).sqrt()) / 7.).sqrt();
            let outer = (18. - 30_f64.sqrt()) / 72.;
            let inner = (18. + 30_f64.sqrt()) / 72.;
            Ok(vec![(-a, outer), (-b, inner), (b, inner), (a, outer)])
        }
        6 => Ok(vec![
            (-0.932_469_514_203_152, 0.1713244923791704 / 2.),
            (-0.6612093864662645, 0.3607615730481386 / 2.),
            (-0.2386191860831969, 0.467913934572691 / 2.),
            (0.2386191860831969, 0.467913934572691 / 2.),
            (0.6612093864662645, 0.3607615730481386 / 2.),
            (0.932_469_514_203_152, 0.1713244923791704 / 2.),
        ]),
        _ => Err("pulse quadrature order must be 2, 4 or 6; odd rules share a central node".into()),
    }
}

pub fn bin(diameter_m: f32) -> Option<usize> {
    DIAMETERS_M
        .iter()
        .position(|d| d.to_bits() == diameter_m.to_bits())
}

pub fn build(order: usize, center: Vec3, velocity: Vec3) -> Result<Pulse, String> {
    let nodes = rule(order)?;
    if !center.is_finite() || !velocity.is_finite() {
        return Err("pulse centre and velocity must be finite".into());
    }
    if center.length() >= 100. || velocity.length() >= 100. {
        return Err("pulse centre or velocity exceeds persistent grain bounds".into());
    }
    let mut grains = Vec::with_capacity(3 * order.pow(3));
    for (diameter_m, angles) in DIAMETERS_M.into_iter().zip(ROTATIONS_XYZ) {
        let rotation = DQuat::from_euler(EulerRot::XYZ, angles[0], angles[1], angles[2]);
        for &(x, wx) in &nodes {
            for &(y, wy) in &nodes {
                for &(z, wz) in &nodes {
                    // Rotate an isotropic quadrature BEFORE anisotropic scale.
                    // E[q]=0 and E[qq^T]=I/3 at every admitted order, hence
                    // each class has covariance diag(scale^2/3).
                    let offset = (rotation * DVec3::new(x, y, z)) * SCALE_M;
                    let position = (center.as_dvec3() + offset).as_vec3();
                    if !position.is_finite() || position.length() >= 100. {
                        return Err("pulse support exceeds persistent grain bounds".into());
                    }
                    grains.push(Grain {
                        volume_resolved: true,
                        kind: GrainKind::Sand,
                        position,
                        velocity,
                        material: Material::sand_from_solid_volume(
                            TOTAL_SOLID_M3 / 3. * wx * wy * wz,
                        ),
                        resting: false,
                        diameter_m,
                        affine: Mat3::ZERO,
                        rotation: Quat::IDENTITY,
                        angular_velocity: Vec3::ZERO,
                    });
                }
            }
        }
    }
    let all: Vec<_> = grains.iter().collect();
    let report = json!({
        "quadrature_order":order,"parcels":grains.len(),
        "diameters_m":DIAMETERS_M,"solid_volume_fraction_per_bin":[1./3.,1./3.,1./3.],
        "declared_total_solid_volume_m3":TOTAL_SOLID_M3,
        "center_m":center.to_array(),"velocity_mps":velocity.to_array(),
        "standardized_half_extent":[1.,1.,1.],
        "post_rotation_scale_m":SCALE_M.to_array(),"rotation_xyz_radians":ROTATIONS_XYZ,
        "enclosing_ellipsoid_half_axes_m":(SCALE_M*3_f64.sqrt()).to_array(),
        "analytic_covariance_m2":[[SCALE_M.x*SCALE_M.x/3.,0.,0.],
            [0.,SCALE_M.y*SCALE_M.y/3.,0.],[0.,0.,SCALE_M.z*SCALE_M.z/3.]],
        "actual":moments(&all),"support":support(&grains),
        "limits":["GL cubature approximates a uniform cube after rotation and scale",
            "physical positions are rounded to f32; actual moments are reported",
            "rigid contact envelopes may overlap; no mass or position is repaired",
            "no physical suspension, mixing or optical calibration is implied"]
    });
    Ok(Pulse { grains, report })
}

fn moments(grains: &[&Grain]) -> Value {
    let solid_volume: f64 = grains.iter().map(|g| g.solid_volume()).sum();
    let mass: f64 = grains.iter().map(|g| g.material.mass_kg()).sum();
    let (centroid, covariance) = if mass > 0. {
        let center = grains
            .iter()
            .map(|g| g.position.as_dvec3() * g.material.mass_kg())
            .sum::<DVec3>()
            / mass;
        // Centred second pass avoids subtracting nearly equal raw moments
        // when a small pulse is translated through the larger local domain.
        let mut covariance = [[0.; 3]; 3];
        for g in grains {
            let r = (g.position.as_dvec3() - center).to_array();
            let w = g.material.mass_kg() / mass;
            for (i, row) in covariance.iter_mut().enumerate() {
                for (j, entry) in row.iter_mut().enumerate() {
                    *entry += w * r[i] * r[j];
                }
            }
        }
        (Some(center.to_array()), Some(covariance))
    } else {
        (None, None)
    };
    json!({"parcels":grains.len(),"solid_volume_m3":solid_volume,"mass_kg":mass,
        "moment_weight":"dry_mass_kg","centroid_m":centroid,"covariance_m2":covariance})
}

fn support(grains: &[Grain]) -> Value {
    let mut center_min = DVec3::splat(f64::INFINITY);
    let mut center_max = DVec3::splat(f64::NEG_INFINITY);
    let mut contact_min = center_min;
    let mut contact_max = center_max;
    let mut radius_min = f64::INFINITY;
    let mut radius_max = 0_f64;
    let mut represented = 0.;
    let mut minimum_distance = f64::INFINITY;
    let mut minimum_gap = f64::INFINITY;
    let mut coincident = 0;
    let mut overlapping = 0;
    let mut inter_bin_overlaps = 0;
    for (i, g) in grains.iter().enumerate() {
        let p = g.position.as_dvec3();
        let r = g.collision_radius() as f64;
        center_min = center_min.min(p);
        center_max = center_max.max(p);
        contact_min = contact_min.min(p - DVec3::splat(r));
        contact_max = contact_max.max(p + DVec3::splat(r));
        radius_min = radius_min.min(r);
        radius_max = radius_max.max(r);
        represented += g.represented_grains();
        for other in &grains[..i] {
            let distance = p.distance(other.position.as_dvec3());
            let gap = distance - r - other.collision_radius() as f64;
            minimum_distance = minimum_distance.min(distance);
            minimum_gap = minimum_gap.min(gap);
            coincident += usize::from(distance == 0.);
            if gap < 0. {
                overlapping += 1;
                inter_bin_overlaps += usize::from(bin(g.diameter_m) != bin(other.diameter_m));
            }
        }
    }
    json!({"center_min_m":center_min.to_array(),"center_max_m":center_max.to_array(),
        "contact_envelope_min_m":contact_min.to_array(),"contact_envelope_max_m":contact_max.to_array(),
        "collision_radius_min_m":radius_min,"collision_radius_max_m":radius_max,
        "represented_grains":represented,"minimum_center_distance_m":minimum_distance,
        "minimum_contact_gap_m":minimum_gap,"coincident_pairs":coincident,
        "overlapping_contact_pairs":overlapping,"inter_bin_overlapping_contact_pairs":inter_bin_overlaps})
}

/// Assay only the declared Sand cohorts. Incoming fixed-D50 material remains
/// visible as untracked stock, rather than being silently labelled pulse mass.
pub fn resident(grains: &[Grain], wet: &[bool]) -> Value {
    assert_eq!(wet.len(), grains.len(), "complete grain wetness mask");
    let mut tracked = Vec::new();
    let mut bins = Vec::new();
    for (i, d) in DIAMETERS_M.into_iter().enumerate() {
        let selected: Vec<_> = grains
            .iter()
            .enumerate()
            .filter(|(_, g)| g.kind == GrainKind::Sand && bin(g.diameter_m) == Some(i))
            .collect();
        let all: Vec<_> = selected.iter().map(|(_, g)| *g).collect();
        tracked.extend_from_slice(&all);
        let phase = |resting: bool, carrier: Option<bool>| {
            moments(
                &selected
                    .iter()
                    .filter(|(index, g)| {
                        g.resting == resting && carrier.is_none_or(|value| wet[*index] == value)
                    })
                    .map(|(_, g)| *g)
                    .collect::<Vec<_>>(),
            )
        };
        let mut row = moments(&all);
        row["diameter_m"] = json!(d);
        row["wet_moving"] = phase(false, Some(true));
        row["dry_moving"] = phase(false, Some(false));
        row["resting"] = phase(true, None);
        row["collision_radius_min_m"] =
            json!(all.iter().map(|g| g.collision_radius()).reduce(f32::min));
        row["collision_radius_max_m"] =
            json!(all.iter().map(|g| g.collision_radius()).reduce(f32::max));
        row["represented_grains"] = json!(all.iter().map(|g| g.represented_grains()).sum::<f64>());
        bins.push(row);
    }
    json!({"total":moments(&tracked),"bins":bins,"untracked_parcels":grains.len()-tracked.len()})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn number(value: &Value) -> f64 {
        value.as_f64().unwrap()
    }

    #[test]
    fn every_rule_preserves_each_psd_mass_center_and_analytic_full_covariance() {
        for order in [2, 4, 6] {
            for center in [
                Vec3::ZERO,
                Vec3::new(0.31, 0.27, 0.2),
                Vec3::new(-0.71, 1.31, 0.43),
            ] {
                let velocity = Vec3::new(0.35, -0.01, 0.02);
                let pulse = build(order, center, velocity).unwrap();
                assert_eq!(pulse.grains.len(), 3 * order.pow(3));
                let assay = resident(&pulse.grains, &vec![true; pulse.grains.len()]);
                assert!((number(&assay["total"]["solid_volume_m3"]) - 3e-8).abs() < 1e-21);
                for (i, row) in assay["bins"].as_array().unwrap().iter().enumerate() {
                    assert_eq!(row["parcels"], order.pow(3));
                    assert!((number(&row["solid_volume_m3"]) - 1e-8).abs() < 3e-22);
                    assert!((number(&row["mass_kg"]) - 2650. * 1e-8).abs() < 1e-18);
                    let c = center.as_dvec3().to_array();
                    let scale = SCALE_M.to_array();
                    // Float-position rounding has a bounded contribution to
                    // COM/covariance. The oracle is the uniform-volume analytic
                    // moment, not another evaluation of the GL stencil.
                    let rounding = c.map(|v| v.abs());
                    let rounding: [f64; 3] = std::array::from_fn(|j| {
                        f32::EPSILON as f64 * (rounding[j] + 3_f64.sqrt() * scale[j])
                    });
                    for (a, coordinate) in c.into_iter().enumerate() {
                        assert!(
                            (number(&row["centroid_m"][a]) - coordinate).abs()
                                < rounding[a] + 1e-14
                        );
                        for (b, other_scale) in scale.into_iter().enumerate() {
                            let target = if a == b { scale[a] * scale[a] / 3. } else { 0. };
                            let tolerance = 2.
                                * (rounding[a] * other_scale / 3_f64.sqrt()
                                    + rounding[b] * scale[a] / 3_f64.sqrt()
                                    + 2. * rounding[a] * rounding[b])
                                + 1e-17;
                            assert!(
                                (number(&row["covariance_m2"][a][b]) - target).abs() < tolerance,
                                "order={order}, bin={i}, covariance={a},{b}: {row}"
                            );
                        }
                    }
                    let expected_population =
                        1e-8 / (std::f64::consts::PI / 6. * (DIAMETERS_M[i] as f64).powi(3));
                    assert!(
                        (number(&row["represented_grains"]) / expected_population - 1.).abs()
                            < 1e-13
                    );
                }
                for (i, g) in pulse.grains.iter().enumerate() {
                    assert!(g.volume_resolved && g.material.valid());
                    assert!(g.solid_volume() > 0. && g.collision_radius() >= g.diameter_m * 0.5);
                    assert_eq!(g.velocity, velocity);
                    assert_eq!(g.affine, Mat3::ZERO);
                    assert_eq!(g.rotation, Quat::IDENTITY);
                    assert_eq!(g.angular_velocity, Vec3::ZERO);
                    assert!(!g.resting);
                    assert!(
                        pulse.grains[..i]
                            .iter()
                            .all(|other| other.position != g.position),
                        "different PSD classes must not start at coincident centres"
                    );
                }
                assert_eq!(pulse.report["support"]["coincident_pairs"], 0);
            }
        }
    }

    #[test]
    fn support_reports_real_contact_overlap_without_repairing_the_fixture() {
        let mut pair = build(2, Vec3::new(0.31, 0.27, 0.2), Vec3::ZERO)
            .unwrap()
            .grains[..2]
            .to_vec();
        pair[1] = pair[0].clone();
        let radius = pair[0].collision_radius();
        pair[1].position.x += radius;
        let before = serde_json::to_string(&pair).unwrap();
        let observed = support(&pair);
        assert_eq!(observed["overlapping_contact_pairs"], 1);
        assert_eq!(observed["coincident_pairs"], 0);
        assert!(number(&observed["minimum_contact_gap_m"]) < 0.);
        let bound = number(&observed["contact_envelope_min_m"][0]);
        assert_eq!(bound, pair[0].position.x as f64 - radius as f64);
        assert_eq!(before, serde_json::to_string(&pair).unwrap());
        pair[1].position = pair[0].position;
        assert_eq!(support(&pair)["coincident_pairs"], 1);
    }

    #[test]
    fn resident_assay_partitions_actual_weight_and_excludes_other_cohorts() {
        let mut grains = build(2, Vec3::new(0.31, 0.27, 0.2), Vec3::ZERO)
            .unwrap()
            .grains;
        for (i, g) in grains.iter_mut().enumerate() {
            g.resting = i % 8 >= 2;
        }
        let mut untracked = grains[0].clone();
        untracked.diameter_m = 0.0001;
        grains.push(untracked);
        let mut other_kind = grains[0].clone();
        other_kind.kind = GrainKind::Gravel;
        grains.push(other_kind);
        let wet: Vec<_> = (0..grains.len()).map(|i| i % 8 == 0).collect();
        let before = serde_json::to_string(&grains).unwrap();
        let assay = resident(&grains, &wet);
        assert_eq!(assay["untracked_parcels"], 2);
        for (i, row) in assay["bins"].as_array().unwrap().iter().enumerate() {
            for (phase, fraction, count) in [
                ("wet_moving", 0.125, 1),
                ("dry_moving", 0.125, 1),
                ("resting", 0.75, 6),
            ] {
                assert_eq!(row[phase]["parcels"], count);
                assert!((number(&row[phase]["solid_volume_m3"]) - 1e-8 * fraction).abs() < 1e-22);
                assert!((number(&row[phase]["mass_kg"]) - 2650. * 1e-8 * fraction).abs() < 1e-18);
            }
            for (phase, offset) in [("wet_moving", 0), ("dry_moving", 1)] {
                let expected = grains[i * 8 + offset].position.to_array();
                for (a, coordinate) in expected.into_iter().enumerate() {
                    assert!(
                        (number(&row[phase]["centroid_m"][a]) - coordinate as f64).abs() < 1e-15
                    );
                    for b in 0..3 {
                        assert!(number(&row[phase]["covariance_m2"][a][b]).abs() < 1e-28);
                    }
                }
            }
        }
        assert_eq!(before, serde_json::to_string(&grains).unwrap());
        assert_eq!(bin(f32::from_bits(DIAMETERS_M[0].to_bits() + 1)), None);
        let empty = resident(&[], &[]);
        assert_eq!(empty["total"]["solid_volume_m3"], 0.);
        assert!(empty["total"]["centroid_m"].is_null());
        assert!(empty["total"]["covariance_m2"].is_null());
    }

    #[test]
    fn invalid_rules_and_nonfinite_or_unsaveable_inputs_are_rejected() {
        for order in [0, 1, 3, 5, 8] {
            assert!(build(order, Vec3::ZERO, Vec3::ZERO).is_err());
        }
        for (center, velocity) in [
            (Vec3::splat(f32::NAN), Vec3::ZERO),
            (Vec3::ZERO, Vec3::splat(f32::INFINITY)),
            (Vec3::new(100., 0., 0.), Vec3::ZERO),
            (Vec3::ZERO, Vec3::new(100., 0., 0.)),
        ] {
            assert!(build(2, center, velocity).is_err());
        }
    }

    #[test]
    #[should_panic(expected = "complete grain wetness mask")]
    fn resident_assay_requires_the_complete_snapshot_mask() {
        let pulse = build(2, Vec3::ZERO, Vec3::ZERO).unwrap();
        resident(&pulse.grains, &[]);
    }
}

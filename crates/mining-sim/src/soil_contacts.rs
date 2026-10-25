//! Rigid-soil convex contact policies. The original policy repairs coincident
//! clipping degeneracy at touching edges. The verified retained-world policy
//! also invalidates a negative cached manifold when a hull axis certifies
//! separation, then regenerates ordinary positive/speculative contacts.
//! Genuine penetrations keep the pinned manifold and physical coefficients.
#[cfg(all(test, feature = "contact-work-audit"))]
#[path = "soil_contacts/convex_sat.rs"]
mod convex_sat;
use q::details::NormalConstraints;
use q::{DefaultQueryDispatcher, PersistentQueryDispatcher, QueryDispatcher};
use rapier3d::parry::{
    query as q,
    shape::{ConvexPolyhedron, PackedFeatureId, Shape},
};
use rapier3d::{
    geometry::{ContactData, ContactManifoldData},
    prelude as r,
};
type Manifold = q::ContactManifold<ContactManifoldData, ContactData>;
const TOUCH_EPS: f32 = 1e-5;
struct StablePolyContacts {
    verify_separation: bool,
    // Diagnostic-only: rebuild cached features while retaining physical points,
    // coefficients, existing touch repair and the verified separation policy.
    #[cfg(test)]
    force_fresh: bool,
    #[cfg(all(test, feature = "contact-work-audit"))]
    sat_features: bool,
}
const TOUCH_ONLY: StablePolyContacts = StablePolyContacts {
    verify_separation: false,
    #[cfg(test)]
    force_fresh: false,
    #[cfg(all(test, feature = "contact-work-audit"))]
    sat_features: false,
};
#[cfg(any(test, feature = "contact-history"))]
const VERIFIED: StablePolyContacts = StablePolyContacts {
    verify_separation: true,
    #[cfg(test)]
    force_fresh: false,
    #[cfg(all(test, feature = "contact-work-audit"))]
    sat_features: false,
};
#[cfg(all(test, feature = "contact-work-audit"))]
const FRESH_FOR_TEST: StablePolyContacts = StablePolyContacts {
    verify_separation: true,
    force_fresh: true,
    sat_features: false,
};
#[cfg(all(test, feature = "contact-work-audit"))]
const SAT_FOR_TEST: StablePolyContacts = StablePolyContacts {
    verify_separation: true,
    force_fresh: true,
    sat_features: true,
};
impl StablePolyContacts {
    // Cached feature contacts can keep a negative gap while an independent
    // hull axis already separates the solids. Force feature regeneration;
    // do not remove speculative contact or alter actual penetration depths.
    fn invalidate_separated(
        &self,
        pos: &r::Pose,
        a: &ConvexPolyhedron,
        b: &ConvexPolyhedron,
        m: &mut Manifold,
    ) -> bool {
        if self.verify_separation
            && m.points.iter().any(|p| p.dist < -TOUCH_EPS)
            && touch_axis(a, b, pos).1 > TOUCH_EPS
        {
            m.mark_shapes_deformed();
            true
        } else {
            false
        }
    }
}
pub(crate) fn restore_dispatcher(world: &mut r::PhysicsWorld) {
    world.narrow_phase.set_query_dispatcher(TOUCH_ONLY);
}
#[cfg(feature = "contact-history")]
pub(crate) fn restore_verified_dispatcher(world: &mut r::PhysicsWorld) {
    world.narrow_phase.set_query_dispatcher(VERIFIED);
}
/// Unqualified diagnostic trial, never a persisted or Session contact policy.
/// The caller must also disable outer pair recycling; otherwise the narrow
/// phase can return before either persistent-dispatcher entrypoint runs.
#[cfg(all(test, feature = "contact-work-audit"))]
pub(crate) fn restore_fresh_dispatcher_for_test(world: &mut r::PhysicsWorld) {
    world.narrow_phase.set_query_dispatcher(FRESH_FOR_TEST);
}
#[cfg(all(test, feature = "contact-work-audit"))]
pub(crate) fn restore_sat_dispatcher_for_test(world: &mut r::PhysicsWorld) {
    convex_sat::reset_stats();
    world.narrow_phase.set_query_dispatcher(SAT_FOR_TEST);
}
#[cfg(all(test, feature = "contact-work-audit"))]
pub(crate) fn sat_diagnostics_for_test() -> serde_json::Value {
    convex_sat::stats()
}
pub(crate) fn configure(world: &mut r::PhysicsWorld) {
    restore_dispatcher(world);
    // New rigid clods and substrate approach the noncompliant unilateral
    // limit. Ordinary game pile springs admit centimetres of penetration under
    // excavator loads, enough to cross the thin terrain collision surface.
    let rigid = r::SpringCoefficients::new(1.0e6, 1.);
    world.integration_parameters.contact_softness = rigid;
    world.integration_parameters.static_contact_softness = rigid;
}
fn child(shape: &dyn Shape, id: u32) -> Option<(r::Pose, &ConvexPolyhedron)> {
    if let Some(h) = shape.as_convex_polyhedron() {
        return Some((r::Pose::IDENTITY, h));
    }
    let (pose, s) = shape.as_compound()?.shapes().get(id as usize)?;
    s.as_convex_polyhedron().map(|h| (*pose, h))
}
// Complete convex SAT: face normals and pairwise edge crosses. Both signs of
// every axis are tested, retaining the largest separation and its outward normal.
fn touch_axis(a: &ConvexPolyhedron, b: &ConvexPolyhedron, pos: &r::Pose) -> (r::Vector, f32) {
    let bp: Vec<_> = b.points().iter().map(|p| pos * *p).collect();
    let mut best = (r::Vector::Y, f32::NEG_INFINITY);
    let mut axis = |n: r::Vector| {
        let len = n.length_squared();
        if len < 1e-12 {
            return;
        }
        let n = n / len.sqrt();
        let amin = a
            .points()
            .iter()
            .map(|p| p.dot(n))
            .fold(f32::INFINITY, f32::min);
        let amax = a
            .points()
            .iter()
            .map(|p| p.dot(n))
            .fold(f32::NEG_INFINITY, f32::max);
        let bmin = bp.iter().map(|p| p.dot(n)).fold(f32::INFINITY, f32::min);
        let bmax = bp
            .iter()
            .map(|p| p.dot(n))
            .fold(f32::NEG_INFINITY, f32::max);
        for candidate in [(n, bmin - amax), (-n, amin - bmax)] {
            if candidate.1 > best.1 {
                best = candidate;
            }
        }
    };
    for f in a.faces() {
        axis(f.normal);
    }
    for f in b.faces() {
        axis(pos.rotation * f.normal);
    }
    for x in a.edges() {
        for y in b.edges() {
            axis(x.dir.cross(pos.rotation * y.dir));
        }
    }
    best
}
fn repair(pos: &r::Pose, a: &ConvexPolyhedron, b: &ConvexPolyhedron, m: &mut Manifold) {
    let Some(witness) = m
        .points
        .iter()
        .rev()
        .find(|p| p.fid1 == PackedFeatureId::UNKNOWN && p.fid2 == PackedFeatureId::UNKNOWN)
    else {
        return;
    };
    let gap = pos * witness.local_p2 - witness.local_p1;
    if gap.length_squared() > TOUCH_EPS * TOUCH_EPS
        || !m.points.iter().any(|p| p.dist < -8. * TOUCH_EPS)
    {
        return;
    }
    let (normal, separation) = touch_axis(a, b, pos);
    if separation.abs() > TOUCH_EPS {
        return;
    }
    // Only real coincident witnesses remain on the actual shared edge/face.
    // Extended clipped pairs span the hull and fabricate deep penetration.
    m.points
        .retain(|p| (pos * p.local_p2 - p.local_p1).length_squared() <= TOUCH_EPS * TOUCH_EPS);
    for p in &mut m.points {
        p.dist = (pos * p.local_p2 - p.local_p1).dot(normal);
        p.data = Default::default();
    }
    m.local_n1 = normal;
    m.local_n2 = pos.rotation.inverse() * -normal;
}
impl QueryDispatcher for StablePolyContacts {
    fn intersection_test(
        &self,
        p: &r::Pose,
        a: &dyn Shape,
        b: &dyn Shape,
    ) -> Result<q::ShapeIntersection, q::Unsupported> {
        DefaultQueryDispatcher.intersection_test(p, a, b)
    }
    fn distance(
        &self,
        p: &r::Pose,
        a: &dyn Shape,
        b: &dyn Shape,
    ) -> Result<q::ShapeDistance, q::Unsupported> {
        DefaultQueryDispatcher.distance(p, a, b)
    }
    fn contact(
        &self,
        p: &r::Pose,
        a: &dyn Shape,
        b: &dyn Shape,
        d: f32,
    ) -> Result<Option<q::Contact>, q::Unsupported> {
        DefaultQueryDispatcher.contact(p, a, b, d)
    }
    fn closest_points(
        &self,
        p: &r::Pose,
        a: &dyn Shape,
        b: &dyn Shape,
        d: f32,
    ) -> Result<q::ClosestPoints, q::Unsupported> {
        DefaultQueryDispatcher.closest_points(p, a, b, d)
    }
    fn cast_shapes(
        &self,
        p: &r::Pose,
        v: r::Vector,
        a: &dyn Shape,
        b: &dyn Shape,
        o: q::ShapeCastOptions,
    ) -> Result<Option<q::ShapeCastHit>, q::Unsupported> {
        DefaultQueryDispatcher.cast_shapes(p, v, a, b, o)
    }
    fn cast_shapes_nonlinear(
        &self,
        m1: &q::NonlinearRigidMotion,
        a: &dyn Shape,
        m2: &q::NonlinearRigidMotion,
        b: &dyn Shape,
        start: f32,
        end: f32,
        stop: bool,
    ) -> Result<Option<q::ShapeCastHit>, q::Unsupported> {
        DefaultQueryDispatcher.cast_shapes_nonlinear(m1, a, m2, b, start, end, stop)
    }
}
impl PersistentQueryDispatcher<ContactManifoldData, ContactData> for StablePolyContacts {
    fn contact_manifolds(
        &self,
        p: &r::Pose,
        a: &dyn Shape,
        b: &dyn Shape,
        d: f32,
        ms: &mut Vec<Manifold>,
        w: &mut Option<q::ContactManifoldsWorkspace>,
    ) -> Result<(), q::Unsupported> {
        #[cfg(all(test, feature = "contact-work-audit"))]
        let original_histories: Vec<_> = if self.sat_features {
            ms.iter()
                .filter(|m| child(a, m.subshape1).is_some() && child(b, m.subshape2).is_some())
                .map(|m| (m.subshape1, m.subshape2, m.points.clone()))
                .collect()
        } else {
            Vec::new()
        };
        #[cfg(test)]
        if self.force_fresh {
            for manifold in ms.iter_mut() {
                manifold.mark_shapes_deformed();
            }
        }
        DefaultQueryDispatcher.contact_manifolds(p, a, b, d, ms, w)?;
        for m in ms {
            if let (Some((pa, ha)), Some((pb, hb))) = (child(a, m.subshape1), child(b, m.subshape2))
            {
                let relative = pa.inverse() * *p * pb;
                if self.invalidate_separated(&relative, ha, hb, m) {
                    DefaultQueryDispatcher
                        .contact_manifold_convex_convex(&relative, ha, hb, None, None, d, m)?;
                }
                #[cfg(all(test, feature = "contact-work-audit"))]
                let separated = if self.sat_features {
                    let old = original_histories
                        .iter()
                        .find(|h| h.0 == m.subshape1 && h.1 == m.subshape2)
                        .map(|h| h.2.as_slice())
                        .unwrap_or(&[]);
                    convex_sat::rebuild_traced(&relative, ha, hb, m, old, Some(d))
                } else {
                    None
                };
                repair(&relative, ha, hb, m);
                #[cfg(all(test, feature = "contact-work-audit"))]
                if let Some(pre_negative) = separated {
                    convex_sat::observe_final_separated(&relative, ha, hb, m, d, pre_negative);
                }
            } else {
                #[cfg(all(test, feature = "contact-work-audit"))]
                if self.sat_features {
                    convex_sat::unsupported();
                }
            }
        }
        Ok(())
    }
    fn contact_manifold_convex_convex(
        &self,
        p: &r::Pose,
        a: &dyn Shape,
        b: &dyn Shape,
        n1: Option<&dyn NormalConstraints>,
        n2: Option<&dyn NormalConstraints>,
        d: f32,
        m: &mut Manifold,
    ) -> Result<(), q::Unsupported> {
        #[cfg(all(test, feature = "contact-work-audit"))]
        let original_history = if self.sat_features {
            m.points.clone()
        } else {
            Vec::new()
        };
        #[cfg(test)]
        if self.force_fresh {
            m.mark_shapes_deformed();
        }
        DefaultQueryDispatcher.contact_manifold_convex_convex(p, a, b, n1, n2, d, m)?;
        if n1.is_none() && n2.is_none() {
            if let (Some(ha), Some(hb)) = (a.as_convex_polyhedron(), b.as_convex_polyhedron()) {
                if self.invalidate_separated(p, ha, hb, m) {
                    DefaultQueryDispatcher
                        .contact_manifold_convex_convex(p, ha, hb, None, None, d, m)?;
                }
                #[cfg(all(test, feature = "contact-work-audit"))]
                let separated = if self.sat_features {
                    convex_sat::rebuild_traced(p, ha, hb, m, &original_history, Some(d))
                } else {
                    None
                };
                repair(p, ha, hb, m);
                #[cfg(all(test, feature = "contact-work-audit"))]
                if let Some(pre_negative) = separated {
                    convex_sat::observe_final_separated(p, ha, hb, m, d, pre_negative);
                }
            } else {
                #[cfg(all(test, feature = "contact-work-audit"))]
                if self.sat_features {
                    convex_sat::unsupported();
                }
            }
        }
        #[cfg(all(test, feature = "contact-work-audit"))]
        if self.sat_features && (n1.is_some() || n2.is_some()) {
            convex_sat::unsupported();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hull(points: &[[f32; 3]]) -> r::SharedShape {
        r::SharedShape::convex_hull(
            &points
                .iter()
                .map(|p| r::Vector::from_array(*p))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
    fn manifold(
        dispatcher: &dyn PersistentQueryDispatcher<ContactManifoldData, ContactData>,
        p: &r::Pose,
        a: &dyn Shape,
        b: &dyn Shape,
    ) -> Manifold {
        let mut m = Manifold::new();
        dispatcher
            .contact_manifold_convex_convex(p, a, b, None, None, 0.02, &mut m)
            .unwrap();
        m
    }
    #[test]
    fn separated_and_compound_contacts_respect_a_known_convex_gap() {
        // Reduced from a real moving-bank snapshot. Every B point lies more
        // than 13 mm to the right of A; a 0.9 mm contact is impossible.
        let a = hull(&[
            [-0.39729223, -0.9438363, 0.34115833],
            [-0.39729223, -0.9061247, 0.44650918],
            [-0.39729223, -0.89445525, 0.44140255],
            [-0.39729223, -0.931577, 0.33769938],
            [0.39729223, -0.9438363, 0.34115833],
            [0.39729223, -0.9061247, 0.44650918],
            [0.39729223, -0.89445525, 0.44140255],
            [0.39729223, -0.931577, 0.33769938],
        ]);
        let b = hull(&[
            [-0.0625, 0.001363337, -0.03125],
            [-0.0625, 0.033320904, 0.09375],
            [0.0625, -0.03984213, -0.03125],
            [0.0625, 0.005157888, -0.03125],
        ]);
        let p = r::Pose::from_parts(
            r::Vector::new(0.48034477, -0.9432144, 0.38268948),
            r::Rotation::from_xyzw(-0.54669476, -0.5035481, -0.66869843, -0.020167604).normalize(),
        );
        let min_b = b
            .as_convex_polyhedron()
            .unwrap()
            .points()
            .iter()
            .map(|v| (p * *v).x)
            .fold(f32::INFINITY, f32::min);
        let max_a = a
            .as_convex_polyhedron()
            .unwrap()
            .points()
            .iter()
            .map(|v| v.x)
            .fold(f32::NEG_INFINITY, f32::max);
        let lower_bound = min_b - max_a;
        assert!(lower_bound > 0.013);
        let old = manifold(&DefaultQueryDispatcher, &p, &*a, &*b);
        let new = manifold(&TOUCH_ONLY, &p, &*a, &*b);
        let old_min = old
            .points
            .iter()
            .map(|p| p.dist)
            .fold(f32::INFINITY, f32::min);
        let new_min = new
            .points
            .iter()
            .map(|p| p.dist)
            .fold(f32::INFINITY, f32::min);
        println!(
            "separated convex gap lower bound={lower_bound}, old={old_min}, repaired={new_min}"
        );
        assert!(old_min >= lower_bound - 0.0001);
        assert!(!new.points.is_empty());
        assert!(
            new_min >= lower_bound - 0.0001,
            "fabricated near contact survived"
        );
        // Compound-child transforms must keep the same physical gap.
        let child_pose = r::Pose::translation(0.3, 0.5, 0.7);
        let compound = r::SharedShape::compound(vec![(child_pose, a.clone())]);
        let mut ms = Vec::new();
        let mut workspace = None;
        TOUCH_ONLY
            .contact_manifolds(
                &(child_pose * p),
                &*compound,
                &*b,
                0.02,
                &mut ms,
                &mut workspace,
            )
            .unwrap();
        assert!(ms.iter().any(|m| !m.points.is_empty()));
        assert!(ms
            .iter()
            .flat_map(|m| &m.points)
            .all(|p| p.dist >= lower_bound - 0.0001));
    }
    #[test]
    fn ordinary_face_contacts_and_real_penetrations_retain_the_pinned_manifold() {
        let points: Vec<_> = [-0.05, 0.05]
            .into_iter()
            .flat_map(|x| {
                [-0.05, 0.05]
                    .into_iter()
                    .flat_map(move |y| [-0.05, 0.05].map(|z| [x, y, z]))
            })
            .collect();
        let h = hull(&points);
        for x in [0.04, 0.09, 0.0999, 0.09998, 0.10002, 0.1001, 0.105] {
            let p = r::Pose::translation(x, 0., 0.);
            let old = manifold(&DefaultQueryDispatcher, &p, &*h, &*h);
            for dispatcher in [&TOUCH_ONLY, &VERIFIED] {
                let new = manifold(dispatcher, &p, &*h, &*h);
                assert!(!new.points.is_empty());
                assert_eq!(old.local_n1, new.local_n1);
                assert_eq!(old.local_n2, new.local_n2);
                assert_eq!(old.points.len(), new.points.len());
                for (a, b) in old.points.iter().zip(&new.points) {
                    assert_eq!(
                        (a.local_p1, a.local_p2, a.dist, a.fid1, a.fid2),
                        (b.local_p1, b.local_p2, b.dist, b.fid1, b.fid2)
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "contact-history")]
    fn cached_negative_contacts_between_separated_hulls_are_regenerated_not_discarded() {
        #[derive(serde::Deserialize)]
        struct Case {
            a: Vec<[f32; 3]>,
            b: Vec<[f32; 3]>,
            translation: [f32; 3],
            rotation: [f32; 4],
            cached_manifold: Manifold,
        }
        let case: Case = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/playtests/fixtures/2026-10-24-separated-clods.json"
        )))
        .unwrap();
        let a = hull(&case.a);
        let b = hull(&case.b);
        let pos = r::Pose::from_parts(
            r::Vector::from_array(case.translation),
            r::Rotation::from_array(case.rotation),
        );
        let mut old = case.cached_manifold.clone();
        DefaultQueryDispatcher
            .contact_manifold_convex_convex(&pos, &*a, &*b, None, None, 0.02, &mut old)
            .unwrap();
        assert!(
            old.points.iter().any(|p| p.dist < -0.001),
            "old cached failure was not reproduced"
        );
        let mut corrected = case.cached_manifold.clone();
        VERIFIED
            .contact_manifold_convex_convex(&pos, &*a, &*b, None, None, 0.02, &mut corrected)
            .unwrap();
        assert!(
            !corrected.points.is_empty(),
            "positive speculative contact must remain"
        );
        assert!(
            corrected.points.iter().all(|p| p.dist > 0.0007),
            "separated solids retained fictitious penetration"
        );
        // The public manifold dispatcher must also preserve compound-child
        // coordinates when invalidating a cached feature manifold.
        let child_pose = r::Pose::translation(0.3, 0.5, 0.7);
        let compound = r::SharedShape::compound(vec![(child_pose, a.clone())]);
        let parent_relative = child_pose * pos;
        let mut manifolds = Vec::new();
        let mut workspace = None;
        DefaultQueryDispatcher
            .contact_manifolds(
                &parent_relative,
                &*compound,
                &*b,
                0.02,
                &mut manifolds,
                &mut workspace,
            )
            .unwrap();
        assert_eq!(manifolds.len(), 1);
        let mut cached = case.cached_manifold;
        cached.set_subshape_pos1(Some(child_pose));
        manifolds[0] = cached;
        VERIFIED
            .contact_manifolds(
                &parent_relative,
                &*compound,
                &*b,
                0.02,
                &mut manifolds,
                &mut workspace,
            )
            .unwrap();
        assert!(manifolds.iter().any(|m| !m.points.is_empty()));
        assert!(manifolds
            .iter()
            .flat_map(|m| &m.points)
            .all(|p| p.dist > 0.0007));
        println!(
            "cached min={} corrected min={}",
            old.points
                .iter()
                .map(|p| p.dist)
                .fold(f32::INFINITY, f32::min),
            corrected
                .points
                .iter()
                .map(|p| p.dist)
                .fold(f32::INFINITY, f32::min)
        );
    }

    #[test]
    #[cfg(feature = "contact-work-audit")]
    fn cached_oblique_clod_manifold_requires_geometric_refresh() {
        #[derive(serde::Deserialize)]
        struct CapturedPose {
            translation: [f32; 3],
            rotation_xyzw: [f32; 4],
        }
        #[derive(serde::Deserialize)]
        struct Case {
            registered_local_vertices: [Vec<[f32; 3]>; 2],
            registered_shared_shapes: [r::SharedShape; 2],
            entry_captured_poses: [CapturedPose; 2],
            saved_original_manifold: Manifold,
            prediction_m: f32,
            observed_cached_min_gap_m: f32,
            observed_fresh_min_gap_m: f32,
            observed_fresh_max_gap_m: f32,
            observed_fresh_point_count: usize,
        }
        let case: Case =
            serde_json::from_str(include_str!("soil_contacts_pair2242_fixture.json")).unwrap();
        let poses = case.entry_captured_poses.map(|pose| {
            // Captured engine coefficients, with no quaternion normalization.
            r::Pose::from_parts(
                r::Vector::from_array(pose.translation),
                r::Rotation::from_array(pose.rotation_xyzw),
            )
        });
        let relative = poses[0].inverse() * poses[1];
        let [a, b] = &case.registered_shared_shapes;
        let ha = a.as_convex_polyhedron().unwrap();
        let hb = b.as_convex_polyhedron().unwrap();
        // Deserialize the actual registered topology; rebuilding a convex hull
        // can renumber features and invalidate this cached-manifold regression.
        for (hull, expected) in [ha, hb].into_iter().zip(&case.registered_local_vertices) {
            assert_eq!(
                hull.points()
                    .iter()
                    .map(|p| p.to_array())
                    .collect::<Vec<_>>(),
                *expected
            );
        }
        // This is the policy's own SAT calculation, not the independent f64
        // registered-geometry oracle used in the source observation.
        let (_, overlap) = touch_axis(ha, hb, &relative);
        assert!(
            (-0.00115..-0.00112).contains(&overlap),
            "actual registered pair should overlap by about 1.137 mm: {overlap}"
        );
        let minimum = |m: &Manifold| {
            m.points
                .iter()
                .map(|p| p.dist)
                .fold(f32::INFINITY, f32::min)
        };
        let maximum = |m: &Manifold| {
            m.points
                .iter()
                .map(|p| p.dist)
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let default = DefaultQueryDispatcher;
        let legacy: [(
            &str,
            &dyn PersistentQueryDispatcher<ContactManifoldData, ContactData>,
        ); 3] = [
            ("Default", &default),
            ("TOUCH_ONLY", &TOUCH_ONLY),
            ("VERIFIED", &VERIFIED),
        ];
        for (name, dispatcher) in legacy {
            let mut cached = case.saved_original_manifold.clone();
            dispatcher
                .contact_manifold_convex_convex(
                    &relative,
                    &**a,
                    &**b,
                    None,
                    None,
                    case.prediction_m,
                    &mut cached,
                )
                .unwrap();
            assert!(
                minimum(&cached) < -0.09,
                "{name} negative control disappeared"
            );
            assert!(
                (minimum(&cached) - case.observed_cached_min_gap_m).abs() <= TOUCH_EPS,
                "{name} no longer reproduces the captured cached patch"
            );
        }
        let mut refreshed = case.saved_original_manifold.clone();
        FRESH_FOR_TEST
            .contact_manifold_convex_convex(
                &relative,
                &**a,
                &**b,
                None,
                None,
                case.prediction_m,
                &mut refreshed,
            )
            .unwrap();
        assert_eq!(refreshed.points.len(), case.observed_fresh_point_count);
        assert_eq!(
            refreshed.points.len(),
            5,
            "the physical contact patch must remain"
        );
        assert!(refreshed.points.iter().all(|p| p.dist.abs() < 0.002));
        assert!((minimum(&refreshed) - case.observed_fresh_min_gap_m).abs() <= TOUCH_EPS);
        assert!((maximum(&refreshed) - case.observed_fresh_max_gap_m).abs() <= TOUCH_EPS);
        assert!(
            refreshed
                .local_n1
                .dot(case.saved_original_manifold.local_n1)
                < 0.
        );
        let mut cold = Manifold::new();
        VERIFIED
            .contact_manifold_convex_convex(
                &relative,
                &**a,
                &**b,
                None,
                None,
                case.prediction_m,
                &mut cold,
            )
            .unwrap();
        assert_eq!(cold.points.len(), refreshed.points.len());
        assert!((minimum(&cold) - minimum(&refreshed)).abs() <= TOUCH_EPS);
        assert!((maximum(&cold) - maximum(&refreshed)).abs() <= TOUCH_EPS);
        assert!(cold.local_n1.dot(refreshed.local_n1) > 0.9999);
    }
}

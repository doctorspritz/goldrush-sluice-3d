//! Explicit test-only convex SAT/feature-clipping experiment.
//! Complete registered faces/edges are retained; no four-vertex face truncation.
use super::{r, Manifold, TOUCH_EPS};
use glam::{DMat3, DQuat, DVec2, DVec3};
use rapier3d::parry::{
    query::TrackedContact,
    shape::{ConvexPolyhedron, PackedFeatureId},
};
use serde::Serialize;

const NUMERIC_M: f64 = 1e-9;
const CERTIFY_M: f64 = TOUCH_EPS as f64;

fn promoted(p: r::Vector) -> DVec3 {
    DVec3::from_array(p.to_array().map(f64::from))
}
fn rounded(p: DVec3) -> r::Vector {
    r::Vector::from_array(p.to_array().map(|x| x as f32))
}
struct Face {
    id: usize,
    vertices: Vec<usize>,
    normal: DVec3,
}
struct Hull {
    vertices: Vec<DVec3>,
    faces: Vec<Face>,
    edges: Vec<[usize; 2]>,
}
impl Hull {
    fn new(shape: &ConvexPolyhedron, linear: DMat3, shift: DVec3) -> Result<Self, &'static str> {
        let vertices: Vec<_> = shape
            .points()
            .iter()
            .map(|p| linear * promoted(*p) + shift)
            .collect();
        if vertices.len() < 4 || vertices.iter().any(|p| !p.is_finite()) {
            return Err("nonfinite or degenerate registered hull");
        }
        let center = vertices.iter().copied().sum::<DVec3>() / vertices.len() as f64;
        let mut faces = Vec::new();
        for (id, face) in shape.faces().iter().enumerate() {
            let first = face.first_vertex_or_edge as usize;
            let mut ids: Vec<_> = shape.vertices_adj_to_face()
                [first..first + face.num_vertices_or_edges as usize]
                .iter()
                .map(|x| *x as usize)
                .collect();
            if ids.len() < 3 {
                return Err("degenerate registered face");
            }
            let origin = vertices[ids[0]];
            let cross = (1..ids.len() - 1)
                .map(|i| (vertices[ids[i]] - origin).cross(vertices[ids[i + 1]] - origin))
                .max_by(|a, b| a.length_squared().total_cmp(&b.length_squared()))
                .unwrap();
            if cross.length_squared() <= 1e-30 {
                return Err("degenerate registered face normal");
            }
            let mut normal = cross.normalize();
            if normal.dot(center - origin) > 0. {
                normal = -normal;
                ids.reverse();
            }
            if ids
                .iter()
                .any(|i| normal.dot(vertices[*i] - origin).abs() > CERTIFY_M)
                || vertices.iter().any(|p| normal.dot(*p - origin) > CERTIFY_M)
            {
                return Err("registered face is not a certified convex support plane");
            }
            faces.push(Face {
                id,
                vertices: ids,
                normal,
            });
        }
        let edges = shape
            .edges()
            .iter()
            .map(|e| e.vertices.map(|v| v as usize))
            .collect();
        Ok(Self {
            vertices,
            faces,
            edges,
        })
    }
    fn interval(&self, n: DVec3) -> (f64, f64) {
        self.vertices
            .iter()
            .map(|p| p.dot(n))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            })
    }
    fn surface(&self, p: DVec3) -> bool {
        let mut nearest = f64::INFINITY;
        for face in &self.faces {
            let d = face.normal.dot(p - self.vertices[face.vertices[0]]);
            if d > CERTIFY_M {
                return false;
            }
            nearest = nearest.min(d.abs());
        }
        nearest <= CERTIFY_M
    }
    fn feature(&self, p: DVec3, face: usize) -> PackedFeatureId {
        for (i, v) in self.vertices.iter().enumerate() {
            if p.distance_squared(*v) <= NUMERIC_M * NUMERIC_M {
                return PackedFeatureId::vertex(i as u32);
            }
        }
        for (i, edge) in self.edges.iter().enumerate() {
            let a = self.vertices[edge[0]];
            let d = self.vertices[edge[1]] - a;
            let t = (p - a).dot(d) / d.length_squared();
            if (-1e-10..=1. + 1e-10).contains(&t)
                && p.distance_squared(a + d * t) <= NUMERIC_M * NUMERIC_M
            {
                return PackedFeatureId::edge(i as u32);
            }
        }
        PackedFeatureId::face(face as u32)
    }
}
#[derive(Clone, Serialize)]
pub(super) struct Proof {
    pub separation_m: f64,
    pub registered_sat_separation_m: f64,
    pub separated_query: bool,
    pub closest_witness_checked: bool,
    pub beyond_prediction_lower_bound_m: Option<f64>,
    pub axis_source: String,
    pub normal_a: [f64; 3],
    pub sat_axis_a: [f64; 3],
    pub clipped_points: usize,
    pub total_points: usize,
    pub duplicate_feature_pairs: usize,
    pub history: Option<HistoryTransfer>,
    pub closest_support_deficits_m: [f64; 2],
    pub closest_tangent_residual_m: f64,
    pub closest_combined_certificate_bound_m: f64,
    pub max_point_depth_bound_error_m: f64,
    pub complete_registered_vertices: [usize; 2],
    pub complete_registered_faces: [usize; 2],
    pub complete_registered_edges: [usize; 2],
}
#[derive(Clone, Copy)]
enum AxisSource {
    FaceA(usize),
    FaceB(usize),
    Edge(usize, usize),
    ClosestFeatures,
}
impl AxisSource {
    fn label(self) -> String {
        match self {
            Self::FaceA(i) => format!("face_a:{i}"),
            Self::FaceB(i) => format!("face_b:{i}"),
            Self::Edge(i, j) => format!("edge_cross:{i}:{j}"),
            Self::ClosestFeatures => "registered_closest_features".into(),
        }
    }
}
#[derive(Clone, Copy)]
struct Axis {
    normal: DVec3,
    separation: f64,
    source: AxisSource,
}
// An arbitrary direction can prove separation. A penetration direction must
// instead be generated by exposed Minkowski support features. A hidden edge
// nearly parallel to an exposed one can otherwise win by a float64 depth tie,
// even though its projected support features do not intersect.
fn axis_features_exposed(
    a: &Hull,
    b: &Hull,
    n: DVec3,
    amax: f64,
    bmin: f64,
    source: AxisSource,
) -> bool {
    let on_a = |id: usize| (a.vertices[id].dot(n) - amax).abs() <= NUMERIC_M;
    let on_b = |id: usize| (b.vertices[id].dot(n) - bmin).abs() <= NUMERIC_M;
    match source {
        AxisSource::FaceA(i) => a.faces[i].vertices.iter().copied().all(on_a),
        AxisSource::FaceB(i) => b.faces[i].vertices.iter().copied().all(on_b),
        AxisSource::Edge(i, j) => {
            a.edges[i].iter().copied().all(on_a) && b.edges[j].iter().copied().all(on_b)
        }
        AxisSource::ClosestFeatures => false,
    }
}
fn sat(a: &Hull, b: &Hull) -> Result<Axis, &'static str> {
    let mut best = Axis {
        normal: DVec3::ZERO,
        separation: f64::NEG_INFINITY,
        source: AxisSource::FaceA(0),
    };
    let mut test = |v: DVec3, source: AxisSource| {
        if !v.is_finite() || v.length_squared() < 1e-30 {
            return;
        }
        let n = v.normalize();
        let (amin, amax) = a.interval(n);
        let (bmin, bmax) = b.interval(n);
        for (normal, separation, amax, bmin) in [
            (n, bmin - amax, amax, bmin),
            (-n, amin - bmax, -amin, -bmax),
        ] {
            if separation > best.separation {
                // Validate only prospective winners, avoiding a support scan
                // for every irrelevant cross-axis. Positive separating planes
                // retain their original selection and prediction-exit role.
                if separation <= 0. && !axis_features_exposed(a, b, normal, amax, bmin, source) {
                    continue;
                }
                best = Axis {
                    normal,
                    separation,
                    source,
                };
            }
        }
    };
    for face in &a.faces {
        test(face.normal, AxisSource::FaceA(face.id));
    }
    for face in &b.faces {
        test(face.normal, AxisSource::FaceB(face.id));
    }
    for (i, ea) in a.edges.iter().enumerate() {
        let da = a.vertices[ea[1]] - a.vertices[ea[0]];
        for (j, eb) in b.edges.iter().enumerate() {
            let db = b.vertices[eb[1]] - b.vertices[eb[0]];
            let cross = da.cross(db);
            if cross.length_squared() > 1e-24 * da.length_squared() * db.length_squared() {
                test(cross, AxisSource::Edge(i, j));
            }
        }
    }
    if best.normal == DVec3::ZERO {
        Err("no nondegenerate SAT axis")
    } else {
        Ok(best)
    }
}
fn clip(mut polygon: Vec<DVec3>, reference: &Hull, face: &Face) -> Vec<DVec3> {
    for i in 0..face.vertices.len() {
        if polygon.is_empty() {
            break;
        }
        let origin = reference.vertices[face.vertices[i]];
        let end = reference.vertices[face.vertices[(i + 1) % face.vertices.len()]];
        let edge = end - origin;
        // Form the oriented area before taking its normal component. At an
        // exact registered endpoint edge.cross(edge) is exactly zero; rotating
        // the edge first introduced false negative orthogonality roundoff.
        let side = |p: DVec3| face.normal.dot(edge.cross(p - origin));
        let mut result = Vec::new();
        let mut previous = *polygon.last().unwrap();
        let mut previous_distance = side(previous);
        for current in polygon {
            let distance = side(current);
            if (previous_distance >= 0.) != (distance >= 0.) {
                result.push(
                    previous
                        + (current - previous)
                            * (previous_distance / (previous_distance - distance)),
                );
            }
            if distance >= 0. {
                result.push(current);
            }
            previous = current;
            previous_distance = distance;
        }
        polygon = result;
    }
    polygon
}
fn face_patch(a: &Hull, b: &Hull, n: DVec3, reference_a: bool, face: &Face) -> Vec<RawContact> {
    let (reference, incident, direction) = if reference_a { (a, b, n) } else { (b, a, -n) };
    let incident_face = incident
        .faces
        .iter()
        .max_by(|x, y| {
            x.normal
                .dot(-direction)
                .total_cmp(&y.normal.dot(-direction))
        })
        .unwrap();
    let polygon = incident_face
        .vertices
        .iter()
        .map(|i| incident.vertices[*i])
        .collect();
    let polygon = clip(polygon, reference, face);
    let plane = reference.vertices[face.vertices[0]];
    polygon
        .into_iter()
        .map(|p| {
            let projected =
                p - direction * (face.normal.dot(p - plane) / face.normal.dot(direction));
            let (pa, pb) = if reference_a {
                (projected, p)
            } else {
                (p, projected)
            };
            let (fa, fb) = if reference_a {
                (face.id, incident_face.id)
            } else {
                (incident_face.id, face.id)
            };
            (pa, pb, a.feature(pa, fa), b.feature(pb, fb))
        })
        .collect()
}
type RawContact = (DVec3, DVec3, PackedFeatureId, PackedFeatureId);
fn cross2(a: DVec2, b: DVec2) -> f64 {
    a.x * b.y - a.y * b.x
}
fn convex_loop(mut points: Vec<DVec2>) -> Vec<DVec2> {
    let mut unique = Vec::new();
    for p in points.drain(..) {
        if !unique
            .iter()
            .any(|q: &DVec2| p.distance_squared(*q) <= NUMERIC_M * NUMERIC_M)
        {
            unique.push(p);
        }
    }
    if unique.len() <= 1 {
        return unique;
    }
    let origin = unique[0];
    let end = *unique
        .iter()
        .max_by(|a, b| {
            a.distance_squared(origin)
                .total_cmp(&b.distance_squared(origin))
        })
        .unwrap();
    let direction = (end - origin).normalize();
    if unique
        .iter()
        .all(|p| cross2(*p - origin, direction).abs() <= NUMERIC_M)
    {
        let lo = *unique
            .iter()
            .min_by(|a, b| a.dot(direction).total_cmp(&b.dot(direction)))
            .unwrap();
        let hi = *unique
            .iter()
            .max_by(|a, b| a.dot(direction).total_cmp(&b.dot(direction)))
            .unwrap();
        return if lo.distance(hi) <= NUMERIC_M {
            vec![lo]
        } else {
            vec![lo, hi]
        };
    }
    unique.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    let mut lower = Vec::<DVec2>::new();
    for p in unique.iter().copied() {
        while lower.len() >= 2
            && cross2(
                lower[lower.len() - 1] - lower[lower.len() - 2],
                p - lower[lower.len() - 1],
            ) <= 0.
        {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper = Vec::<DVec2>::new();
    for p in unique.iter().rev().copied() {
        while upper.len() >= 2
            && cross2(
                upper[upper.len() - 1] - upper[upper.len() - 2],
                p - upper[upper.len() - 1],
            ) <= 0.
        {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}
fn inside_feature(p: DVec2, polygon: &[DVec2]) -> bool {
    match polygon.len() {
        0 => false,
        1 => p.distance(polygon[0]) <= NUMERIC_M,
        2 => {
            let d = polygon[1] - polygon[0];
            let t = ((p - polygon[0]).dot(d) / d.length_squared()).clamp(0., 1.);
            p.distance(polygon[0] + d * t) <= NUMERIC_M
        }
        n => {
            let strict =
                (0..n).all(|i| cross2(polygon[(i + 1) % n] - polygon[i], p - polygon[i]) >= 0.);
            // Per-halfplane slack is unbounded along a sliver polygon's weak
            // direction. Use actual point/segment distance at its boundary.
            strict
                || (0..n).any(|i| {
                    let edge = polygon[(i + 1) % n] - polygon[i];
                    let t = ((p - polygon[i]).dot(edge) / edge.length_squared()).clamp(0., 1.);
                    p.distance_squared(polygon[i] + edge * t) <= NUMERIC_M * NUMERIC_M
                })
        }
    }
}
fn feature_edges(polygon: &[DVec2]) -> Vec<(DVec2, DVec2)> {
    match polygon.len() {
        0 | 1 => Vec::new(),
        2 => vec![(polygon[0], polygon[1])],
        n => (0..n).map(|i| (polygon[i], polygon[(i + 1) % n])).collect(),
    }
}
/// Intersect the exposed support features in their common tangent plane.
/// This handles vertices, skew segments, collinear overlapping segments and
/// polygons uniformly. A zero-depth touch is retained as real contact geometry.
fn boundary_patch(a: &Hull, b: &Hull, n: DVec3) -> Option<Vec<RawContact>> {
    let u = if n.x.abs() < 0.8 {
        n.cross(DVec3::X).normalize()
    } else {
        n.cross(DVec3::Y).normalize()
    };
    let v = n.cross(u);
    let project = |p: DVec3| DVec2::new(p.dot(u), p.dot(v));
    let (_, amax) = a.interval(n);
    let (bmin, _) = b.interval(n);
    let af = convex_loop(
        a.vertices
            .iter()
            .filter(|p| (p.dot(n) - amax).abs() <= NUMERIC_M)
            .map(|p| project(*p))
            .collect(),
    );
    let bf = convex_loop(
        b.vertices
            .iter()
            .filter(|p| (p.dot(n) - bmin).abs() <= NUMERIC_M)
            .map(|p| project(*p))
            .collect(),
    );
    let mut intersection: Vec<_> = af
        .iter()
        .filter(|p| inside_feature(**p, &bf))
        .copied()
        .collect();
    intersection.extend(bf.iter().filter(|p| inside_feature(**p, &af)).copied());
    for (a0, a1) in feature_edges(&af) {
        let da = a1 - a0;
        for (b0, b1) in feature_edges(&bf) {
            let db = b1 - b0;
            let area = cross2(da, db);
            // Parallel overlaps are already represented by contained endpoints.
            if area.abs() <= 1e-24 * da.length() * db.length() {
                continue;
            }
            let r = b0 - a0;
            let s = cross2(r, db) / area;
            let t = cross2(r, da) / area;
            if (-1e-10..=1. + 1e-10).contains(&s) && (-1e-10..=1. + 1e-10).contains(&t) {
                let point = a0 + da * s;
                if point.distance(b0 + db * t) <= NUMERIC_M {
                    intersection.push(point);
                }
            }
        }
    }
    let intersection = convex_loop(intersection);
    if intersection.is_empty() {
        return None;
    }
    Some(
        intersection
            .into_iter()
            .map(|p| {
                let tangent = u * p.x + v * p.y;
                let pa = tangent + n * amax;
                let pb = tangent + n * bmin;
                let face_a = a
                    .faces
                    .iter()
                    .min_by(|x, y| {
                        x.normal
                            .dot(pa - a.vertices[x.vertices[0]])
                            .abs()
                            .total_cmp(&y.normal.dot(pa - a.vertices[y.vertices[0]]).abs())
                    })
                    .unwrap()
                    .id;
                let face_b = b
                    .faces
                    .iter()
                    .min_by(|x, y| {
                        x.normal
                            .dot(pb - b.vertices[x.vertices[0]])
                            .abs()
                            .total_cmp(&y.normal.dot(pb - b.vertices[y.vertices[0]]).abs())
                    })
                    .unwrap()
                    .id;
                (pa, pb, a.feature(pa, face_a), b.feature(pb, face_b))
            })
            .collect(),
    )
}
// Complete disjoint-polyhedron distance: vertex/face interior projections
// plus every edge/edge pair. Endpoint/segment candidates cover clamped and
// parallel cases. Cross-product line parameters avoid normal-equation
// cancellation on nearly parallel registered edges.
fn closest_registered_points(
    a: &Hull,
    b: &Hull,
    preferred_normal: DVec3,
) -> Result<(DVec3, DVec3, DVec3), &'static str> {
    let mut best = None;
    let mut distance2 = f64::INFINITY;
    let mut consider = |x: DVec3, y: DVec3, hint: DVec3| {
        let delta = y - x;
        let d = delta.length_squared();
        if !d.is_finite() || d >= distance2 {
            return;
        }
        let mut n = if hint.is_finite() && hint.length_squared() > 0. {
            hint.normalize()
        } else {
            preferred_normal
        };
        if delta.dot(n) < 0. {
            n = -n;
        }
        let defect = (a.interval(n).1 - x.dot(n)).abs()
            + (y.dot(n) - b.interval(n).0).abs()
            + (delta - n * delta.dot(n)).length();
        // A nearly-on-edge projection may be a closer invalid candidate.
        // Require the convex optimality conditions before choosing it, using
        // construction precision, not the much wider physical certificate.
        if defect <= NUMERIC_M {
            distance2 = d;
            best = Some((x, y, n));
        }
    };
    let on_segment = |p: DVec3, x: DVec3, y: DVec3| {
        let edge = y - x;
        let len = edge.length_squared();
        if len <= 1e-30 {
            return (x, x - p);
        }
        let t = ((p - x).dot(edge) / len).clamp(0., 1.);
        let q = x + edge * t;
        // Transport a line's exact perpendicular instead of normalizing the
        // subtraction of almost equal projected positions.
        let hint = if t > 0. && t < 1. {
            edge.cross((x - p).cross(edge))
        } else {
            q - p
        };
        (q, hint)
    };
    let project_face = |p: DVec3, h: &Hull, f: &Face| {
        let signed = f.normal.dot(p - h.vertices[f.vertices[0]]);
        if signed < 0. {
            return None;
        }
        let x = p - f.normal * signed;
        let strict = (0..f.vertices.len()).all(|i| {
            let start = h.vertices[f.vertices[i]];
            let edge = h.vertices[f.vertices[(i + 1) % f.vertices.len()]] - start;
            edge.cross(x - start).dot(f.normal) >= 0.
        });
        let near_boundary = (0..f.vertices.len()).any(|i| {
            let (q, _) = on_segment(
                x,
                h.vertices[f.vertices[i]],
                h.vertices[f.vertices[(i + 1) % f.vertices.len()]],
            );
            q.distance_squared(x) <= NUMERIC_M * NUMERIC_M
        });
        (strict || near_boundary).then_some(x)
    };
    for point in &a.vertices {
        for face in &b.faces {
            if let Some(x) = project_face(*point, b, face) {
                consider(*point, x, -face.normal);
            }
        }
    }
    for point in &b.vertices {
        for face in &a.faces {
            if let Some(x) = project_face(*point, a, face) {
                consider(x, *point, face.normal);
            }
        }
    }
    for ea in &a.edges {
        for eb in &b.edges {
            let [a0, a1] = ea.map(|i| a.vertices[i]);
            let [b0, b1] = eb.map(|i| b.vertices[i]);
            for point in [a0, a1] {
                let (q, hint) = on_segment(point, b0, b1);
                consider(point, q, hint);
            }
            for point in [b0, b1] {
                let (q, hint) = on_segment(point, a0, a1);
                consider(q, point, -hint);
            }
            let u = a1 - a0;
            let v = b1 - b0;
            let cross = u.cross(v);
            let denominator = cross.length_squared();
            if denominator > 1e-30 {
                let delta = b0 - a0;
                let ta = delta.cross(v).dot(cross) / denominator;
                let tb = delta.cross(u).dot(cross) / denominator;
                if (0. ..=1.).contains(&ta) && (0. ..=1.).contains(&tb) {
                    consider(a0 + u * ta, b0 + v * tb, cross);
                }
            }
        }
    }
    best.ok_or("no registered closest-feature candidate meets convex optimality")
}
fn checked_empty_prediction(
    ha: &Hull,
    hb: &Hull,
    pos: &r::Pose,
    axis: &Axis,
    registered_sat: &Axis,
    prediction: f32,
    m: &mut Manifold,
) -> Option<Proof> {
    let normal = rounded(axis.normal).normalize();
    let n = promoted(normal).normalize();
    let bound = hb.interval(n).0 - ha.interval(n).1;
    if !bound.is_finite() || bound <= f64::from(prediction) {
        return None;
    }
    // A registered separating plane proves no surface pair is in prediction.
    // This is a distance lower bound, not an unmeasured closest certificate.
    m.points.clear();
    m.local_n1 = normal;
    m.local_n2 = pos.rotation.inverse() * -normal;
    Some(Proof {
        separation_m: axis.separation,
        registered_sat_separation_m: registered_sat.separation,
        separated_query: true,
        closest_witness_checked: false,
        beyond_prediction_lower_bound_m: Some(bound),
        axis_source: axis.source.label(),
        normal_a: promoted(normal).to_array(),
        sat_axis_a: registered_sat.normal.to_array(),
        clipped_points: 0,
        total_points: 0,
        duplicate_feature_pairs: 0,
        history: None,
        closest_support_deficits_m: [0.; 2],
        closest_tangent_residual_m: 0.,
        closest_combined_certificate_bound_m: 0.,
        max_point_depth_bound_error_m: 0.,
        complete_registered_vertices: [ha.vertices.len(), hb.vertices.len()],
        complete_registered_faces: [ha.faces.len(), hb.faces.len()],
        complete_registered_edges: [ha.edges.len(), hb.edges.len()],
    })
}

#[derive(Clone, Default, Serialize)]
pub(super) struct HistoryTransfer {
    pub old_records: usize,
    pub new_records: usize,
    pub matched_records: usize,
    pub reused_old_records: usize,
    pub maximum_body_local_shift_m: f64,
}
type HistoryPoint = TrackedContact<rapier3d::geometry::ContactData>;
fn match_history_once(points: &mut [HistoryPoint], old: &[HistoryPoint]) -> HistoryTransfer {
    let mut candidates = Vec::new();
    for (i, p) in points.iter().enumerate() {
        for (j, q) in old.iter().enumerate() {
            if p.fid1 == q.fid1 && p.fid2 == q.fid2 {
                let distance = promoted(p.local_p1).distance_squared(promoted(q.local_p1))
                    + promoted(p.local_p2).distance_squared(promoted(q.local_p2));
                assert!(distance.is_finite(), "nonfinite cached witness distance");
                candidates.push((distance, i, j));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut used_old = vec![false; old.len()];
    let mut used_new = vec![false; points.len()];
    for point in points.iter_mut() {
        point.data = Default::default();
    }
    let mut report = HistoryTransfer {
        old_records: old.len(),
        new_records: points.len(),
        ..HistoryTransfer::default()
    };
    for (distance, i, j) in candidates {
        if used_new[i] || used_old[j] {
            continue;
        }
        points[i].data = old[j].data;
        used_new[i] = true;
        used_old[j] = true;
        report.matched_records += 1;
        report.maximum_body_local_shift_m = report.maximum_body_local_shift_m.max(distance.sqrt());
    }
    report.reused_old_records =
        report.matched_records - used_old.iter().filter(|used| **used).count();
    assert_eq!(
        report.reused_old_records, 0,
        "cached impulse record was cloned"
    );
    report
}
pub(super) fn rebuild(
    pos: &r::Pose,
    a: &ConvexPolyhedron,
    b: &ConvexPolyhedron,
    m: &mut Manifold,
) -> Result<Option<Proof>, &'static str> {
    let old = m.points.clone();
    let mut result = rebuild_geometry(pos, a, b, m, None)?;
    if let Some(proof) = result.as_mut() {
        proof.history = Some(match_history_once(&mut m.points, &old));
    }
    Ok(result)
}
/// Rebuild registered convex overlap and (with prediction) separated features.
/// A missing prediction retains ordinary positive queries only for observer
/// controls. Failure never certifies or deletes existing physical contacts.
fn rebuild_geometry(
    pos: &r::Pose,
    a: &ConvexPolyhedron,
    b: &ConvexPolyhedron,
    m: &mut Manifold,
    prediction: Option<f32>,
) -> Result<Option<Proof>, &'static str> {
    let rotation = DQuat::from_array(pos.rotation.to_array().map(f64::from));
    // Preserve captured coefficients: no quaternion normalization. Invert the
    // actual affine map for witnesses, rather than assuming perfect orthogonality.
    let linear = DMat3::from_cols(
        rotation * DVec3::X,
        rotation * DVec3::Y,
        rotation * DVec3::Z,
    );
    if !linear.is_finite() || linear.determinant().abs() < 1e-12 {
        return Err("invalid captured affine pose");
    }
    let inverse = linear.inverse();
    let shift = promoted(pos.translation);
    let ha = Hull::new(a, DMat3::IDENTITY, DVec3::ZERO)?;
    let hb = Hull::new(b, linear, shift)?;
    let mut axis = sat(&ha, &hb)?;
    let registered_sat = axis;
    let sat_separation = axis.separation;
    let sat_normal = axis.normal;
    let separated = sat_separation > 0.;
    if separated {
        let Some(prediction) = prediction else {
            return Ok(None);
        };
        if sat_separation > f64::from(prediction) + CERTIFY_M {
            if let Some(proof) =
                checked_empty_prediction(&ha, &hb, pos, &axis, &registered_sat, prediction, m)
            {
                return Ok(Some(proof));
            }
        }
        let (x, y, n) = closest_registered_points(&ha, &hb, sat_normal)?;
        let delta = y - x;
        let gap = hb.interval(n).0 - ha.interval(n).1;
        let defect = (ha.interval(n).1 - x.dot(n)).abs()
            + (y.dot(n) - hb.interval(n).0).abs()
            + (delta - n * delta.dot(n)).length();
        if defect > CERTIFY_M || gap < -CERTIFY_M {
            return Err("registered separated closest candidates fail combined support bound");
        }
        axis = Axis {
            normal: n,
            separation: gap,
            source: AxisSource::ClosestFeatures,
        };
        if gap > f64::from(prediction) + CERTIFY_M {
            if let Some(proof) =
                checked_empty_prediction(&ha, &hb, pos, &axis, &registered_sat, prediction, m)
            {
                return Ok(Some(proof));
            }
        }
    }
    let normal = rounded(axis.normal).normalize();
    // Construct exposed features with the unrounded SAT axis. Premature f32
    // rounding changes edge-plane membership before witnesses are constructed.
    let n = axis.normal;
    let solver_n = promoted(normal).normalize();
    let reference_a = ha.faces.iter().find(|f| f.normal.dot(n) > 1. - 1e-12);
    let reference_b = hb.faces.iter().find(|f| f.normal.dot(-n) > 1. - 1e-12);
    let mut patch = if separated {
        // The complete closest-feature intersection retains an entire parallel
        // n-gon/edge/vertex support set, not the first four face vertices.
        boundary_patch(&ha, &hb, n)
            .ok_or("registered separated support features have no checked tangent intersection")?
    } else if let Some(face) = reference_a {
        face_patch(&ha, &hb, n, true, face)
    } else if let Some(face) = reference_b {
        face_patch(&ha, &hb, n, false, face)
    } else {
        boundary_patch(&ha, &hb, n)
            .ok_or("SAT boundary support features have no checked tangent-plane intersection")?
    };
    let mut unique = Vec::<RawContact>::new();
    for p in patch.drain(..) {
        if !unique.iter().any(|q| {
            p.0.distance_squared(q.0) <= NUMERIC_M * NUMERIC_M
                && p.1.distance_squared(q.1) <= NUMERIC_M * NUMERIC_M
        }) {
            unique.push(p);
        }
    }
    if unique.is_empty() {
        return Err("empty clipped convex intersection");
    }
    let minimum = unique
        .iter()
        .map(|p| (p.1 - p.0).dot(n))
        .fold(f64::INFINITY, f64::min);
    if (minimum - axis.separation).abs() > CERTIFY_M {
        return Err("clipped patch does not reach complete SAT support depth");
    }
    let deepest: Vec<_> = unique
        .iter()
        .filter(|p| (p.1 - p.0).dot(n) <= minimum + NUMERIC_M)
        .collect();
    let closest_a = deepest.iter().map(|p| p.0).sum::<DVec3>() / deepest.len() as f64;
    let closest_b = deepest.iter().map(|p| p.1).sum::<DVec3>() / deepest.len() as f64;
    let (_, amax) = ha.interval(n);
    let (bmin, _) = hb.interval(n);
    let mut deficits = [amax - closest_a.dot(n), closest_b.dot(n) - bmin];
    let difference = closest_b - closest_a;
    let mut tangent = (difference - n * difference.dot(n)).length();
    if deficits.iter().map(|d| d.abs()).sum::<f64>() + tangent > CERTIFY_M {
        return Err("closest patch witnesses fail combined support/tangent certificate");
    }
    let clipped_points = unique.len();
    // Retain the ordinary extra closest-witness contact, but derive it from
    // the certified support patch rather than an unqualified EPA fallback.
    unique.push((
        closest_a,
        closest_b,
        PackedFeatureId::UNKNOWN,
        PackedFeatureId::UNKNOWN,
    ));
    let mut points = Vec::new();
    let (_, solver_amax) = ha.interval(solver_n);
    let (solver_bmin, _) = hb.interval(solver_n);
    let mut depth_error: f64 = 0.;
    let mut combined_bound = 0.;
    for (pa, pb, fa, fb) in unique {
        let local_a = rounded(pa);
        let local_b = rounded(inverse * (pb - shift));
        let actual_a = promoted(local_a);
        let actual_b = promoted(pos.transform_point(local_b));
        if !ha.surface(actual_a) || !hb.surface(actual_b) {
            return Err("rounded contact witness is not on registered hull surfaces");
        }
        let delta = actual_b - actual_a;
        let gap = delta.dot(promoted(normal));
        if (delta - solver_n * delta.dot(solver_n)).length() > CERTIFY_M {
            return Err("rounded witness pair is not normal aligned");
        }
        if fa == PackedFeatureId::UNKNOWN && fb == PackedFeatureId::UNKNOWN {
            deficits = [
                solver_amax - actual_a.dot(solver_n),
                actual_b.dot(solver_n) - solver_bmin,
            ];
            tangent = (delta - solver_n * delta.dot(solver_n)).length();
            combined_bound = deficits.iter().map(|d| d.abs()).sum::<f64>()
                + tangent
                + (gap - axis.separation).abs();
            if combined_bound > CERTIFY_M {
                return Err(
                    "rounded closest witnesses fail combined support/tangent/depth certificate",
                );
            }
        }
        depth_error = depth_error.max((axis.separation - gap).max(0.));
        if gap < axis.separation - CERTIFY_M {
            return Err("contact exceeds registered SAT penetration depth");
        }
        points.push(TrackedContact::new(
            local_a,
            local_b,
            fa,
            fb,
            (pos.transform_point(local_b) - local_a).dot(normal),
        ));
    }
    let mut duplicate_feature_pairs = 0;
    for (i, p) in points.iter().enumerate() {
        for q in &points[..i] {
            if p.fid1 == q.fid1 && p.fid2 == q.fid2 {
                duplicate_feature_pairs += 1;
            }
        }
    }
    // Only replace after every point and the closest/support witness certify.
    m.points.clear();
    m.points.extend(points);
    m.local_n1 = normal;
    m.local_n2 = pos.rotation.inverse() * -normal;
    Ok(Some(Proof {
        separation_m: axis.separation,
        registered_sat_separation_m: sat_separation,
        separated_query: separated,
        closest_witness_checked: true,
        beyond_prediction_lower_bound_m: None,
        axis_source: axis.source.label(),
        normal_a: promoted(normal).to_array(),
        sat_axis_a: sat_normal.to_array(),
        clipped_points,
        total_points: m.points.len(),
        duplicate_feature_pairs,
        history: None,
        closest_support_deficits_m: deficits,
        closest_tangent_residual_m: tangent,
        closest_combined_certificate_bound_m: combined_bound,
        max_point_depth_bound_error_m: depth_error,
        complete_registered_vertices: [ha.vertices.len(), hb.vertices.len()],
        complete_registered_faces: [ha.faces.len(), hb.faces.len()],
        complete_registered_edges: [ha.edges.len(), hb.edges.len()],
    }))
}

#[derive(Default, Serialize)]
struct PositiveProof {
    final_points: usize,
    closest_combined_bound_m: Option<f64>,
    outside_prediction_lower_bound_m: Option<f64>,
}
// Observe the actual final patch AFTER the existing parent repair. This never
// mutates or re-runs repair, drops contacts, or changes stored warm impulses.
fn check_final_separated(
    pos: &r::Pose,
    a: &ConvexPolyhedron,
    b: &ConvexPolyhedron,
    m: &Manifold,
    prediction: f32,
) -> Result<PositiveProof, &'static str> {
    let rotation = DQuat::from_array(pos.rotation.to_array().map(f64::from));
    let linear = DMat3::from_cols(
        rotation * DVec3::X,
        rotation * DVec3::Y,
        rotation * DVec3::Z,
    );
    let ha = Hull::new(a, DMat3::IDENTITY, DVec3::ZERO)?;
    let hb = Hull::new(b, linear, promoted(pos.translation))?;
    let mut proof = PositiveProof {
        final_points: m.points.len(),
        ..Default::default()
    };
    let normal = promoted(m.local_n1);
    if m.points.is_empty() {
        // Any unit separating plane supplies a valid distance lower bound.
        // The cached normal is checked geometrically, not assumed current.
        let mut bound = f64::NEG_INFINITY;
        if normal.is_finite() && normal.length_squared() > 1e-20 {
            let n = normal.normalize();
            bound = hb.interval(n).0 - ha.interval(n).1;
        }
        if bound <= f64::from(prediction) + CERTIFY_M {
            bound = bound.max(sat(&ha, &hb)?.separation);
        }
        if bound > f64::from(prediction) - CERTIFY_M {
            proof.outside_prediction_lower_bound_m = Some(bound);
            return Ok(proof);
        }
        return Err("empty separated patch not proved beyond prediction distance");
    }
    if !normal.is_finite() || (normal.length() - 1.).abs() > 1e-5 {
        return Err("final separated normal is not finite and unit");
    }
    let normal = normal.normalize();
    let back = linear * promoted(m.local_n2);
    if !back.is_finite() || (back + normal).length() > 1e-5 {
        return Err("final separated local normals disagree");
    }
    let mut closest = false;
    for point in &m.points {
        let x = promoted(point.local_p1);
        let y = linear * promoted(point.local_p2) + promoted(pos.translation);
        if !ha.surface(x) || !hb.surface(y) {
            return Err("final separated point is not on both registered surfaces");
        }
        let delta = y - x;
        let gap = delta.dot(normal);
        if !point.dist.is_finite() || (gap - f64::from(point.dist)).abs() > CERTIFY_M {
            return Err("final separated point depth disagrees with witnesses");
        }
        if point.dist < -TOUCH_EPS {
            return Err("final separated patch retains a negative contact");
        }
        if (delta - normal * gap).length() > CERTIFY_M {
            return Err("final separated contact witnesses have tangential residual");
        }
        if point.fid1 == PackedFeatureId::UNKNOWN && point.fid2 == PackedFeatureId::UNKNOWN {
            closest = true;
            let support_a = ha.interval(normal).1;
            let support_b = hb.interval(normal).0;
            let combined = (support_a - x.dot(normal)).abs()
                + (y.dot(normal) - support_b).abs()
                + (delta - normal * gap).length()
                + (gap - f64::from(point.dist)).abs();
            if combined > CERTIFY_M {
                return Err("final separated closest witness exceeds combined support bound");
            }
            proof.closest_combined_bound_m =
                Some(proof.closest_combined_bound_m.unwrap_or(0.).max(combined));
        }
    }
    if !closest {
        return Err("final separated patch has no independent closest witness");
    }
    Ok(proof)
}
pub(super) fn observe_final_separated(
    pos: &r::Pose,
    a: &ConvexPolyhedron,
    b: &ConvexPolyhedron,
    m: &Manifold,
    prediction: f32,
    pre_negative_points: usize,
) {
    let post_negative_points = m.points.iter().filter(|p| p.dist < -TOUCH_EPS).count();
    let outcome = check_final_separated(pos, a, b, m, prediction);
    STATS.with(|state| {
        let mut stats=state.borrow_mut();
        stats.positive_post_repair_queries+=1;
        stats.positive_post_repair_negative_queries+=usize::from(post_negative_points>0);
        stats.positive_negative_queries_removed_by_existing_repair+=usize::from(pre_negative_points>0 && post_negative_points==0);
        match outcome {
            Ok(proof)=>{
                stats.positive_post_repair_checked_queries+=1;
                if let Some(bound)=proof.closest_combined_bound_m {
                    stats.max_positive_closest_combined_bound_m=stats.max_positive_closest_combined_bound_m.max(bound);
                }
            },
            Err(reason)=>{
                stats.positive_post_repair_unproved_queries+=1;
                *stats.positive_post_repair_failures.entry(reason).or_default()+=1;
                if stats.positive_post_repair_examples.len()<4 && a.points().len()+b.points().len()<=256 && m.points.len()<=64 {
                    stats.positive_post_repair_examples.push(serde_json::json!({
                        "reason":reason,"relative_translation":pos.translation.to_array(),"relative_rotation_xyzw":pos.rotation.to_array(),
                        "registered_shared_shapes":[r::SharedShape::new(a.clone()),r::SharedShape::new(b.clone())],
                        "final_manifold_after_existing_repair":m,"prediction_m":prediction,
                        "pre_negative_points":pre_negative_points,"post_negative_points":post_negative_points}));
                }
            }
        }
    });
}

#[derive(Default, Serialize)]
struct Stats {
    admitted_queries: usize,
    certified_overlapping_patches: usize,
    certified_positive_closest_patches: usize,
    certified_separated_outside_prediction: usize,
    negative_ordinary_positive_queries_rebuilt: usize,
    positive_sat_gap_forwarded_to_ordinary_query: usize,
    positive_sat_gap_with_negative_ordinary_contacts: usize,
    positive_post_repair_queries: usize,
    positive_post_repair_checked_queries: usize,
    positive_post_repair_unproved_queries: usize,
    positive_post_repair_negative_queries: usize,
    positive_negative_queries_removed_by_existing_repair: usize,
    positive_post_repair_failures: std::collections::BTreeMap<&'static str, usize>,
    positive_post_repair_examples: Vec<serde_json::Value>,
    max_positive_closest_combined_bound_m: f64,
    failed_certifications: usize,
    duplicate_identity_patches: usize,
    matched_history_records: usize,
    reused_history_records: usize,
    maximum_history_shift_m: f64,
    duplicate_identity_examples: Vec<serde_json::Value>,
    failures: std::collections::BTreeMap<&'static str, usize>,
    unsupported_shape_or_normal_constraint_manifolds: usize,
    max_abs_closest_support_deficit_m: f64,
    max_closest_tangent_residual_m: f64,
    max_closest_combined_certificate_bound_m: f64,
    max_depth_bound_error_m: f64,
    complete_clipped_points: usize,
    total_points_with_closest_witness: usize,
    first_proofs: Vec<Proof>,
    failure_examples: Vec<serde_json::Value>,
}
std::thread_local! {static STATS:std::cell::RefCell<Stats>=std::cell::RefCell::new(Stats::default());}
pub(super) fn reset_stats() {
    STATS.with(|s| *s.borrow_mut() = Stats::default());
}
pub(super) fn stats() -> serde_json::Value {
    STATS.with(|s| serde_json::to_value(&*s.borrow()).unwrap())
}
pub(super) fn unsupported() {
    STATS.with(|s| {
        s.borrow_mut()
            .unsupported_shape_or_normal_constraint_manifolds += 1
    });
}
pub(super) fn rebuild_traced(
    pos: &r::Pose,
    a: &ConvexPolyhedron,
    b: &ConvexPolyhedron,
    m: &mut Manifold,
    old: &[HistoryPoint],
    prediction: Option<f32>,
) -> Option<usize> {
    let original_negative = m.points.iter().filter(|p| p.dist < -TOUCH_EPS).count();
    let mut outcome = rebuild_geometry(pos, a, b, m, prediction);
    let separated = matches!(&outcome, Ok(None))
        || matches!(&outcome,Ok(Some(proof)) if proof.separated_query)
        || (outcome.is_err() && {
            let rotation = DQuat::from_array(pos.rotation.to_array().map(f64::from));
            let linear = DMat3::from_cols(
                rotation * DVec3::X,
                rotation * DVec3::Y,
                rotation * DVec3::Z,
            );
            Hull::new(a, DMat3::IDENTITY, DVec3::ZERO)
                .and_then(|ha| {
                    Hull::new(b, linear, promoted(pos.translation)).and_then(|hb| sat(&ha, &hb))
                })
                .is_ok_and(|axis| axis.separation > 0.)
        });
    let separated_pre_negative =
        separated.then(|| m.points.iter().filter(|p| p.dist < -TOUCH_EPS).count());
    // Use the original pre-query records. The ordinary dispatcher may already
    // have copied one old feature record into multiple fresh points.
    let history = match_history_once(&mut m.points, old);
    if let Ok(Some(proof)) = outcome.as_mut() {
        proof.history = Some(history.clone());
    }
    STATS.with(|state|{
        let mut stats=state.borrow_mut();stats.admitted_queries+=1;
        stats.matched_history_records+=history.matched_records;
        stats.reused_history_records+=history.reused_old_records;
        stats.maximum_history_shift_m=stats.maximum_history_shift_m.max(history.maximum_body_local_shift_m);
        match outcome {
            Ok(None)=>{
                stats.positive_sat_gap_forwarded_to_ordinary_query+=1;
                if m.points.iter().any(|p|p.dist < -TOUCH_EPS){stats.positive_sat_gap_with_negative_ordinary_contacts+=1;}
            },
            Ok(Some(proof))=>{
                if proof.separated_query {
                    stats.negative_ordinary_positive_queries_rebuilt+=usize::from(original_negative>0);
                    if proof.closest_witness_checked {stats.certified_positive_closest_patches+=1;}
                    else {stats.certified_separated_outside_prediction+=1;}
                } else {stats.certified_overlapping_patches+=1;}
                if proof.duplicate_feature_pairs>0 {
                    stats.duplicate_identity_patches+=1;
                    if stats.duplicate_identity_examples.len()<2 && a.points().len()+b.points().len()<=256 && m.points.len()<=64 {
                        stats.duplicate_identity_examples.push(serde_json::json!({"relative_translation":pos.translation.to_array(),"relative_rotation_xyzw":pos.rotation.to_array(),"registered_shared_shapes":[r::SharedShape::new(a.clone()),r::SharedShape::new(b.clone())],"sat_manifold_with_duplicate_ids":&*m,"proof":&proof}));
                    }
                }
                stats.max_abs_closest_support_deficit_m=stats.max_abs_closest_support_deficit_m.max(proof.closest_support_deficits_m.iter().map(|d|d.abs()).fold(0.,f64::max));
                stats.max_closest_tangent_residual_m=stats.max_closest_tangent_residual_m.max(proof.closest_tangent_residual_m);
                stats.max_depth_bound_error_m=stats.max_depth_bound_error_m.max(proof.max_point_depth_bound_error_m);
                stats.max_closest_combined_certificate_bound_m=stats.max_closest_combined_certificate_bound_m.max(proof.closest_combined_certificate_bound_m);
                stats.complete_clipped_points+=proof.clipped_points;stats.total_points_with_closest_witness+=proof.total_points;
                if stats.first_proofs.len()<8 {stats.first_proofs.push(proof);}
            },
            Err(reason)=>{
                stats.failed_certifications+=1;
                let count=stats.failures.entry(reason).or_default();*count+=1;
                if *count==1 && stats.failure_examples.len()<8 {
                    let capture=a.points().len()+b.points().len()<=256 && m.points.len()<=64;
                    let mut example=serde_json::json!({"reason":reason,"relative_translation":pos.translation.to_array(),"relative_rotation_xyzw":pos.rotation.to_array(),"vertices":[a.points().len(),b.points().len()],"full_fixture_captured":capture});
                    if capture {
                        example["registered_shared_shapes"]=serde_json::to_value([r::SharedShape::new(a.clone()),r::SharedShape::new(b.clone())]).unwrap();
                        example["ordinary_manifold_before_sat"]=serde_json::to_value(&*m).unwrap();
                    }
                    stats.failure_examples.push(example);
                }
            },
        }
    });
    separated_pre_negative
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    fn cuboid(x: f32, y: f32, z: f32) -> r::SharedShape {
        let mut points = Vec::new();
        for a in [-x, x] {
            for b in [-y, y] {
                for c in [-z, z] {
                    points.push(r::Vector::new(a, b, c));
                }
            }
        }
        r::SharedShape::convex_hull(&points).unwrap()
    }
    fn minimum(m: &Manifold) -> f32 {
        m.points
            .iter()
            .map(|p| p.dist)
            .fold(f32::INFINITY, f32::min)
    }
    fn check_closest(proof: &Proof) {
        assert!(proof
            .closest_support_deficits_m
            .iter()
            .all(|d| d.abs() <= CERTIFY_M));
        assert!(proof.closest_tangent_residual_m <= CERTIFY_M);
        assert!(proof.closest_combined_certificate_bound_m <= CERTIFY_M);
        assert_eq!(proof.history.as_ref().unwrap().reused_old_records, 0);
        assert!(proof.max_point_depth_bound_error_m <= CERTIFY_M);
    }
    #[test]
    fn sat_face_patch_has_correct_depth_and_separation_preserves_existing_contacts() {
        let shape = cuboid(0.05, 0.04, 0.03);
        let h = shape.as_convex_polyhedron().unwrap();
        let mut m = Manifold::new();
        let proof = rebuild(&r::Pose::translation(0.099, 0., 0.), h, h, &mut m)
            .unwrap()
            .unwrap();
        assert!((proof.separation_m + 0.001).abs() < 1e-7);
        check_closest(&proof);
        assert_eq!(proof.clipped_points, 4);
        assert_eq!(m.points.len(), 5);
        assert!((minimum(&m) + 0.001).abs() < 1e-7);
        assert!((m.local_n1 - r::Vector::X).length() < 1e-6);
        let before = serde_json::to_string(&m).unwrap();
        assert!(rebuild(&r::Pose::translation(0.105, 0., 0.), h, h, &mut m)
            .unwrap()
            .is_none());
        assert_eq!(
            serde_json::to_string(&m).unwrap(),
            before,
            "separated query must leave ordinary contacts intact"
        );
    }
    #[derive(Deserialize)]
    struct Pose {
        translation: [f32; 3],
        rotation_xyzw: [f32; 4],
    }
    #[derive(Deserialize)]
    struct Case {
        registered_shared_shapes: [r::SharedShape; 2],
        entry_captured_poses: [Pose; 2],
        saved_original_manifold: Manifold,
    }
    fn captured() -> Case {
        serde_json::from_str(include_str!("../soil_contacts_pair2242_fixture.json")).unwrap()
    }
    fn pose(p: &Pose) -> r::Pose {
        r::Pose::from_parts(
            r::Vector::from_array(p.translation),
            r::Rotation::from_array(p.rotation_xyzw),
        )
    }
    #[test]
    fn registered_bad_cached_patch_becomes_a_certified_full_support_patch() {
        let case = captured();
        let [a, b] = &case.registered_shared_shapes;
        let pos =
            pose(&case.entry_captured_poses[0]).inverse() * pose(&case.entry_captured_poses[1]);
        let mut m = case.saved_original_manifold;
        assert!(minimum(&m) < -0.09, "captured old negative control changed");
        let old_normal = m.local_n1;
        let proof = rebuild(
            &pos,
            a.as_convex_polyhedron().unwrap(),
            b.as_convex_polyhedron().unwrap(),
            &mut m,
        )
        .unwrap()
        .unwrap();
        check_closest(&proof);
        assert_eq!(proof.complete_registered_vertices, [6, 6]);
        assert!((-0.00115..-0.00112).contains(&proof.separation_m));
        assert!((f64::from(minimum(&m)) - proof.separation_m).abs() < CERTIFY_M);
        assert!(m.local_n1.dot(old_normal) < 0.);
        assert!(
            m.points.len() >= 4,
            "keep a real contact patch and closest witness"
        );
        assert_eq!(m.points.iter().filter(|p|p.fid1==PackedFeatureId::UNKNOWN&&p.fid2==PackedFeatureId::UNKNOWN).count(),1);
        println!(
            "registered SAT patch {}",
            serde_json::to_string(&proof).unwrap()
        );
    }
    #[test]
    fn registered_patch_transports_under_common_motion_and_pair_swap() {
        let case = captured();
        let [a, b] = &case.registered_shared_shapes;
        let originals = case.entry_captured_poses.each_ref().map(pose);
        let mut baseline = Manifold::new();
        let p = rebuild(
            &(originals[0].inverse() * originals[1]),
            a.as_convex_polyhedron().unwrap(),
            b.as_convex_polyhedron().unwrap(),
            &mut baseline,
        )
        .unwrap()
        .unwrap();
        let normal = originals[0].rotation * baseline.local_n1;
        for q in [
            r::Rotation::IDENTITY,
            r::Rotation::from_rotation_x(0.7),
            r::Rotation::from_rotation_y(-1.1),
            r::Rotation::from_rotation_z(0.4),
        ] {
            let shift = r::Vector::new(0.125, -0.25, 0.5);
            let moved =
                originals.map(|p| r::Pose::from_parts(q * p.translation + shift, q * p.rotation));
            let mut m = Manifold::new();
            let proof = rebuild(
                &(moved[0].inverse() * moved[1]),
                a.as_convex_polyhedron().unwrap(),
                b.as_convex_polyhedron().unwrap(),
                &mut m,
            )
            .unwrap()
            .unwrap();
            check_closest(&proof);
            assert!((proof.separation_m - p.separation_m).abs() < CERTIFY_M);
            assert!((moved[0].rotation * m.local_n1 - q * normal).length() < 0.00001);
            for point in &m.points {
                let target = moved[0] * point.local_p1;
                let nearest = baseline
                    .points
                    .iter()
                    .map(|old| target.distance(q * (originals[0] * old.local_p1) + shift))
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    nearest <= TOUCH_EPS,
                    "actual witness did not transport: {nearest}m"
                );
            }
            let mut swapped = Manifold::new();
            let proof = rebuild(
                &(moved[1].inverse() * moved[0]),
                b.as_convex_polyhedron().unwrap(),
                a.as_convex_polyhedron().unwrap(),
                &mut swapped,
            )
            .unwrap()
            .unwrap();
            check_closest(&proof);
            assert!((proof.separation_m - p.separation_m).abs() < CERTIFY_M);
            assert!((moved[1].rotation * swapped.local_n1 + q * normal).length() < 0.00001);
        }
    }
    #[test]
    fn skew_edge_contact_has_exposed_normal_aligned_witnesses() {
        let a = cuboid(0.1, 0.003, 0.005);
        let b = cuboid(0.1, 0.004, 0.006);
        let rotation = r::Rotation::from_rotation_z(0.7) * r::Rotation::from_rotation_y(0.4);
        let pos = r::Pose::from_parts(r::Vector::ZERO, rotation);
        let mut m = Manifold::new();
        let proof = rebuild(
            &pos,
            a.as_convex_polyhedron().unwrap(),
            b.as_convex_polyhedron().unwrap(),
            &mut m,
        )
        .unwrap()
        .unwrap();
        assert!(
            proof.axis_source.starts_with("edge_cross"),
            "independent skew-rod control requires edge axis: {}",
            proof.axis_source
        );
        check_closest(&proof);
        let n = r::Vector::X.cross(rotation * r::Vector::X).normalize();
        let expected = 0.003 * n.y.abs()
            + 0.005 * n.z.abs()
            + 0.004 * (rotation * r::Vector::Y).dot(n).abs()
            + 0.006 * (rotation * r::Vector::Z).dot(n).abs();
        assert!(
            (proof.separation_m + f64::from(expected)).abs() < 1e-7,
            "independent rod-width penetration changed"
        );
        assert_eq!(proof.clipped_points, 1);
        assert_eq!(m.points.len(), 2);
        println!("edge SAT patch {}", serde_json::to_string(&proof).unwrap());
    }
    #[test]
    fn registered_nearly_parallel_edges_keep_the_combined_witness_certificate() {
        #[derive(Deserialize)]
        struct Captured {
            relative_translation: [f32; 3],
            relative_rotation_xyzw: [f32; 4],
            registered_shared_shapes: [r::SharedShape; 2],
            ordinary_manifold_before_sat: Manifold,
        }
        let case: Captured =
            serde_json::from_str(include_str!("fixtures/nearly_parallel_edges.json")).unwrap();
        let [a, b] = &case.registered_shared_shapes;
        let pos = r::Pose::from_parts(
            r::Vector::from_array(case.relative_translation),
            r::Rotation::from_array(case.relative_rotation_xyzw),
        );
        let mut m = case.ordinary_manifold_before_sat;
        let proof = rebuild(
            &pos,
            a.as_convex_polyhedron().unwrap(),
            b.as_convex_polyhedron().unwrap(),
            &mut m,
        )
        .unwrap()
        .unwrap();
        check_closest(&proof);
        // Independent promoted-coefficient SAT/line calculation is retained
        // with the source observation; this is an actual complete-bank case.
        assert!((proof.separation_m - (-8.810345970763911e-7)).abs() < 1e-9);
        assert!(proof.axis_source.starts_with("edge_cross"));
        assert!(proof.closest_tangent_residual_m < 1e-8);
        assert_eq!(proof.complete_registered_vertices, [4, 5]);
        println!(
            "captured near-parallel SAT patch {}",
            serde_json::to_string(&proof).unwrap()
        );
    }
    #[test]
    fn captured_cadence_overlap_uses_exposed_edge_generators() {
        #[derive(Deserialize)]
        struct Captured {
            relative_translation: [f32; 3],
            relative_rotation_xyzw: [f32; 4],
            registered_shared_shapes: [r::SharedShape; 2],
            ordinary_manifold_before_sat: Manifold,
        }
        let case: Captured =
            serde_json::from_str(include_str!("fixtures/cadence_overlap_boundary.json")).unwrap();
        let [a, b] = &case.registered_shared_shapes;
        let pos = r::Pose::from_parts(
            r::Vector::from_array(case.relative_translation),
            r::Rotation::from_array(case.relative_rotation_xyzw),
        );
        let rotation = DQuat::from_array(case.relative_rotation_xyzw.map(f64::from));
        let linear = DMat3::from_cols(
            rotation * DVec3::X,
            rotation * DVec3::Y,
            rotation * DVec3::Z,
        );
        let ha = Hull::new(
            a.as_convex_polyhedron().unwrap(),
            DMat3::IDENTITY,
            DVec3::ZERO,
        )
        .unwrap();
        let hb = Hull::new(
            b.as_convex_polyhedron().unwrap(),
            linear,
            DVec3::from_array(case.relative_translation.map(f64::from)),
        )
        .unwrap();
        // Negative control: the lower parallel edge is not a support feature
        // for this upward direction. Its projected features still fail the
        // unchanged 1 nm intersection rule after fixing axis selection.
        let da = ha.vertices[ha.edges[2][1]] - ha.vertices[ha.edges[2][0]];
        let db = hb.vertices[hb.edges[0][1]] - hb.vertices[hb.edges[0][0]];
        let mut hidden = da.cross(db).normalize();
        if hidden.y < 0. {
            hidden = -hidden;
        }
        let (_, amax) = ha.interval(hidden);
        let (bmin, _) = hb.interval(hidden);
        assert!(amax - ha.vertices[ha.edges[2][0]].dot(hidden) > 0.08);
        assert!(!axis_features_exposed(
            &ha,
            &hb,
            hidden,
            amax,
            bmin,
            AxisSource::Edge(2, 0)
        ));
        assert!(boundary_patch(&ha, &hb, hidden).is_none());
        let before = sat(&ha, &hb).unwrap();
        let (_, amax) = ha.interval(before.normal);
        let (bmin, _) = hb.interval(before.normal);
        assert!(axis_features_exposed(
            &ha,
            &hb,
            before.normal,
            amax,
            bmin,
            before.source
        ));
        println!(
            "captured overlap axis={} gap_m={:.17} normal={:?}",
            before.source.label(),
            before.separation,
            before.normal
        );
        let mut m = case.ordinary_manifold_before_sat;
        let proof = rebuild(
            &pos,
            a.as_convex_polyhedron().unwrap(),
            b.as_convex_polyhedron().unwrap(),
            &mut m,
        )
        .expect("valid captured overlap must construct a checked support intersection")
        .expect("captured overlap must retain contact");
        check_closest(&proof);
        assert!(
            (proof.registered_sat_separation_m - (-6.764458824262687e-7)).abs() < 1e-12,
            "independent promoted-coordinate support distance changed"
        );
        assert_eq!(proof.complete_registered_vertices, [6, 6]);
        assert!(proof.clipped_points > 0 && !m.points.is_empty());
        println!(
            "captured overlap proof {}",
            serde_json::to_string(&proof).unwrap()
        );
    }
    #[test]
    fn captured_sloping_face_clip_keeps_its_exact_shared_vertex() {
        #[derive(Deserialize)]
        struct Captured {
            relative_translation: [f32; 3],
            relative_rotation_xyzw: [f32; 4],
            registered_shared_shapes: [r::SharedShape; 2],
            ordinary_manifold_before_sat: Manifold,
        }
        let case: Captured =
            serde_json::from_str(include_str!("fixtures/sloping_face_vertex_touch.json")).unwrap();
        let [a, b] = &case.registered_shared_shapes;
        let pos = r::Pose::from_parts(
            r::Vector::from_array(case.relative_translation),
            r::Rotation::from_array(case.relative_rotation_xyzw),
        );
        let a = a.as_convex_polyhedron().unwrap();
        let b = b.as_convex_polyhedron().unwrap();
        // This is an exact common registered vertex, not a near-touch inferred
        // from a distance-query tolerance or a normalized replacement pose.
        assert_eq!(a.points()[1], pos * b.points()[3]);
        let mut m = case.ordinary_manifold_before_sat;
        let proof = rebuild(&pos, a, b, &mut m)
            .expect("an exact shared vertex must survive face clipping")
            .expect("a real zero-depth touch must retain contact");
        check_closest(&proof);
        assert_eq!(proof.registered_sat_separation_m, 0.);
        assert_eq!(proof.complete_registered_vertices, [4, 6]);
        assert!(m
            .points
            .iter()
            .any(|c| c.local_p1 == a.points()[1] && pos * c.local_p2 == a.points()[1]));
        println!(
            "exact sloping-face touch {}",
            serde_json::to_string(&proof).unwrap()
        );
    }
    #[test]
    fn exact_registered_vertex_touch_is_retained_as_a_real_contact() {
        #[derive(Deserialize)]
        struct Captured {
            relative_translation: [f32; 3],
            relative_rotation_xyzw: [f32; 4],
            registered_shared_shapes: [r::SharedShape; 2],
            ordinary_manifold_before_sat: Manifold,
        }
        let case: Captured =
            serde_json::from_str(include_str!("fixtures/vertex_touch.json")).unwrap();
        let [a, b] = &case.registered_shared_shapes;
        let pos = r::Pose::from_parts(
            r::Vector::from_array(case.relative_translation),
            r::Rotation::from_array(case.relative_rotation_xyzw),
        );
        let mut m = case.ordinary_manifold_before_sat;
        let proof = rebuild(
            &pos,
            a.as_convex_polyhedron().unwrap(),
            b.as_convex_polyhedron().unwrap(),
            &mut m,
        )
        .unwrap()
        .unwrap();
        check_closest(&proof);
        assert!(proof.separation_m.abs() < 1e-10);
        assert_eq!(proof.clipped_points, 1);
        assert_eq!(m.points.len(), 2);
        assert!(m
            .points
            .iter()
            .all(|p| (pos * p.local_p2 - p.local_p1).length() <= TOUCH_EPS));
        println!(
            "captured touching support patch {}",
            serde_json::to_string(&proof).unwrap()
        );
    }
    #[test]
    fn collinear_exposed_edges_preserve_the_entire_touching_segment() {
        let a = r::SharedShape::convex_hull(&[
            r::Vector::new(-0.05, 0., 0.),
            r::Vector::new(0.05, 0., 0.),
            r::Vector::new(0., -0.04, 0.03),
            r::Vector::new(0., -0.04, -0.03),
        ])
        .unwrap();
        let b = r::SharedShape::convex_hull(&[
            r::Vector::new(-0.03, 0., 0.),
            r::Vector::new(0.07, 0., 0.),
            r::Vector::new(0.02, 0.04, 0.03),
            r::Vector::new(0.02, 0.04, -0.03),
        ])
        .unwrap();
        let ha = Hull::new(
            a.as_convex_polyhedron().unwrap(),
            DMat3::IDENTITY,
            DVec3::ZERO,
        )
        .unwrap();
        let hb = Hull::new(
            b.as_convex_polyhedron().unwrap(),
            DMat3::IDENTITY,
            DVec3::ZERO,
        )
        .unwrap();
        let patch = boundary_patch(&ha, &hb, DVec3::Y).unwrap();
        assert_eq!(patch.len(), 2);
        let lo = patch.iter().map(|p| p.0.x).fold(f64::INFINITY, f64::min);
        let hi = patch
            .iter()
            .map(|p| p.0.x)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((lo + 0.03).abs() < 1e-8 && (hi - 0.05).abs() < 1e-8);
        assert!(patch.iter().all(|p| p.0.distance(p.1) <= NUMERIC_M));
        assert!(
            (hi - lo - 0.08).abs() < 1e-8,
            "actual shared line-contact extent was lost"
        );
    }
    #[test]
    fn captured_degenerate_boundary_keeps_all_support_intersections() {
        #[derive(Deserialize)]
        struct Captured {
            relative_translation: [f32; 3],
            relative_rotation_xyzw: [f32; 4],
            registered_shared_shapes: [r::SharedShape; 2],
            ordinary_manifold_before_sat: Manifold,
        }
        let case: Captured =
            serde_json::from_str(include_str!("fixtures/collinear_touch.json")).unwrap();
        let [a, b] = &case.registered_shared_shapes;
        let pos = r::Pose::from_parts(
            r::Vector::from_array(case.relative_translation),
            r::Rotation::from_array(case.relative_rotation_xyzw),
        );
        let mut m = case.ordinary_manifold_before_sat;
        let proof = rebuild(
            &pos,
            a.as_convex_polyhedron().unwrap(),
            b.as_convex_polyhedron().unwrap(),
            &mut m,
        )
        .unwrap()
        .unwrap();
        check_closest(&proof);
        assert!(proof.separation_m.abs() < 1e-8);
        assert!(!m.points.is_empty());
        println!(
            "captured degenerate support patch {}",
            serde_json::to_string(&proof).unwrap()
        );
    }
    #[test]
    fn duplicate_feature_histories_are_matched_once_using_local_geometry() {
        let point = |x: f32, impulse: f32| {
            let mut p: HistoryPoint = TrackedContact::new(
                r::Vector::new(x, 0., 0.),
                r::Vector::new(x, 0., 0.),
                PackedFeatureId::edge(0),
                PackedFeatureId::edge(0),
                0.,
            );
            p.data.impulse = impulse;
            p.data.warmstart_impulse = impulse;
            p
        };
        let old = vec![point(0., 2.), point(0.1, 5.)];
        let mut new = vec![point(0.1, 0.), point(0., 0.), point(0.2, 0.)];
        let report = match_history_once(&mut new, &old);
        assert_eq!(report.matched_records, 2);
        assert_eq!(report.reused_old_records, 0);
        assert_eq!(new[0].data.warmstart_impulse, 5.);
        assert_eq!(new[1].data.warmstart_impulse, 2.);
        assert_eq!(new[2].data.warmstart_impulse, 0.);
        assert_eq!(
            new.iter().map(|p| p.data.warmstart_impulse).sum::<f32>(),
            7.
        );
    }
    #[test]
    fn captured_duplicate_ids_use_pre_dispatcher_history_without_cloning() {
        use rapier3d::parry::query::PersistentQueryDispatcher;
        #[derive(Deserialize)]
        struct Captured {
            relative_translation: [f32; 3],
            relative_rotation_xyzw: [f32; 4],
            registered_shared_shapes: [r::SharedShape; 2],
            sat_manifold_with_duplicate_ids: Manifold,
        }
        let case: Captured =
            serde_json::from_str(include_str!("fixtures/duplicate_feature_ids.json")).unwrap();
        let [a, b] = &case.registered_shared_shapes;
        let pos = r::Pose::from_parts(
            r::Vector::from_array(case.relative_translation),
            r::Rotation::from_array(case.relative_rotation_xyzw),
        );
        let mut m = case.sat_manifold_with_duplicate_ids;
        for point in &mut m.points {
            point.data = Default::default();
        }
        let mut pair = None;
        for (i, p) in m.points.iter().enumerate() {
            for (j, q) in m.points[..i].iter().enumerate() {
                if p.fid1 == q.fid1 && p.fid2 == q.fid2 {
                    pair = Some((i, j));
                }
            }
        }
        let (i, j) = pair.expect("real duplicated feature fixture");
        m.points[i].data.warmstart_impulse = 2.;
        m.points[j].data.warmstart_impulse = 5.;
        let original_count = m.points.len();
        reset_stats();
        super::super::SAT_FOR_TEST
            .contact_manifold_convex_convex(&pos, &**a, &**b, None, None, 0.01, &mut m)
            .unwrap();
        assert_eq!(
            m.points.len(),
            original_count,
            "keep physical samples instead of deleting the ambiguous ones"
        );
        assert_eq!(
            m.points
                .iter()
                .map(|p| p.data.warmstart_impulse)
                .sum::<f32>(),
            7.
        );
        assert_eq!(stats()["reused_history_records"], 0);
    }
    #[test]
    fn separated_negative_contact_counter_is_an_active_negative_control() {
        let shape = cuboid(0.05, 0.04, 0.03);
        let h = shape.as_convex_polyhedron().unwrap();
        let mut m = Manifold::new();
        m.points.push(TrackedContact::new(
            r::Vector::ZERO,
            r::Vector::ZERO,
            PackedFeatureId::UNKNOWN,
            PackedFeatureId::UNKNOWN,
            -0.01,
        ));
        let old = m.points.clone();
        reset_stats();
        let pos = r::Pose::translation(0.105, 0., 0.);
        let pre = rebuild_traced(&pos, h, h, &mut m, &old, None).unwrap();
        super::super::repair(&pos, h, h, &mut m);
        observe_final_separated(&pos, h, h, &m, 0.02, pre);
        assert_eq!(stats()["positive_post_repair_negative_queries"], 1);
        assert_eq!(stats()["positive_post_repair_unproved_queries"], 1);
        assert_eq!(
            stats()["positive_sat_gap_with_negative_ordinary_contacts"],
            1
        );
        assert_eq!(
            m.points[0].dist, -0.01,
            "do not silently remove unqualified fallback contacts"
        );
    }
    fn hexagonal_prism(radius: f32) -> r::SharedShape {
        let mut points = Vec::new();
        for y in [-0.02, 0.02] {
            for i in 0..6 {
                let angle = std::f32::consts::TAU * i as f32 / 6.;
                points.push(r::Vector::new(
                    radius * angle.cos(),
                    y,
                    radius * angle.sin(),
                ));
            }
        }
        r::SharedShape::convex_hull(&points).unwrap()
    }
    #[test]
    fn complete_hexagonal_faces_are_clipped_without_a_four_vertex_limit() {
        let radius = 0.04f32;
        let shape = hexagonal_prism(radius);
        let h = shape.as_convex_polyhedron().unwrap();
        assert!(h.faces().iter().any(|f| f.num_vertices_or_edges == 6));
        let mut m = Manifold::new();
        let proof = rebuild(&r::Pose::translation(0., 0.039, 0.), h, h, &mut m)
            .unwrap()
            .unwrap();
        check_closest(&proof);
        assert_eq!(proof.clipped_points, 6);
        assert_eq!(m.points.len(), 7);
        let polygon = &m.points[..proof.clipped_points];
        let twice_area: f64 = (0..polygon.len())
            .map(|i| {
                let p = polygon[i].local_p1;
                let q = polygon[(i + 1) % polygon.len()].local_p1;
                f64::from(p.x) * f64::from(q.z) - f64::from(q.x) * f64::from(p.z)
            })
            .sum();
        let exact = 3. * 3f64.sqrt() / 2. * f64::from(radius).powi(2);
        assert!(
            (twice_area.abs() / 2. - exact).abs() / exact < 0.00001,
            "full physical face area lost"
        );
    }

    #[test]
    fn separated_observation_uses_the_actual_existing_touch_repair() {
        let shape = cuboid(0.05, 0.04, 0.03);
        let h = shape.as_convex_polyhedron().unwrap();
        let pos = r::Pose::translation(f32::from_bits(0.1_f32.to_bits() + 1), 0., 0.);
        let mut m = Manifold::new();
        m.local_n1 = r::Vector::X;
        m.local_n2 = -r::Vector::X;
        m.points.push(TrackedContact::new(
            -r::Vector::X * 0.05,
            -r::Vector::X * 0.05,
            PackedFeatureId::face(0),
            PackedFeatureId::face(0),
            -0.09,
        ));
        m.points.push(TrackedContact::new(
            r::Vector::X * 0.05,
            -r::Vector::X * 0.05,
            PackedFeatureId::UNKNOWN,
            PackedFeatureId::UNKNOWN,
            0.,
        ));
        let old = m.points.clone();
        reset_stats();
        let pre = rebuild_traced(&pos, h, h, &mut m, &old, None).unwrap();
        assert_eq!(pre, 1);
        assert_eq!(
            m.points.len(),
            2,
            "SAT audit must retain ordinary query before repair"
        );
        super::super::repair(&pos, h, h, &mut m);
        assert_eq!(
            m.points.len(),
            1,
            "existing repair must remove the extended noncoincident point"
        );
        let before = serde_json::to_string(&m).unwrap();
        observe_final_separated(&pos, h, h, &m, 0.02, pre);
        assert_eq!(
            before,
            serde_json::to_string(&m).unwrap(),
            "observation changed the solver patch"
        );
        let report = stats();
        assert_eq!(
            report["positive_sat_gap_with_negative_ordinary_contacts"],
            1
        );
        assert_eq!(
            report["positive_negative_queries_removed_by_existing_repair"],
            1
        );
        assert_eq!(report["positive_post_repair_negative_queries"], 0);
        assert_eq!(report["positive_post_repair_checked_queries"], 1);
        assert_eq!(report["positive_post_repair_unproved_queries"], 0);
    }
    #[test]
    fn final_separated_plain_dispatcher_checks_speculation_and_prediction_exit() {
        use rapier3d::parry::query::PersistentQueryDispatcher;
        let shape = cuboid(0.05, 0.04, 0.03);
        let h = shape.as_convex_polyhedron().unwrap();
        let mut m = Manifold::new();
        reset_stats();
        for offset in [0.105, 0.2] {
            super::super::SAT_FOR_TEST
                .contact_manifold_convex_convex(
                    &r::Pose::translation(offset, 0., 0.),
                    h,
                    h,
                    None,
                    None,
                    0.02,
                    &mut m,
                )
                .unwrap();
            if offset < 0.15 {
                assert!(!m.points.is_empty());
            } else {
                assert!(m.points.is_empty());
            }
        }
        let report = stats();
        assert_eq!(report["positive_post_repair_queries"], 2);
        assert_eq!(report["positive_post_repair_checked_queries"], 2);
        assert_eq!(report["positive_post_repair_unproved_queries"], 0);
        assert_eq!(report["positive_post_repair_negative_queries"], 0);
    }

    fn full_query(
        pos: &r::Pose,
        a: &ConvexPolyhedron,
        b: &ConvexPolyhedron,
        m: &mut Manifold,
        prediction: f32,
    ) -> Proof {
        let old = m.points.clone();
        let mut proof = rebuild_geometry(pos, a, b, m, Some(prediction))
            .unwrap()
            .unwrap();
        proof.history = Some(match_history_once(&mut m.points, &old));
        proof
    }
    #[test]
    fn positive_diagonal_distance_is_euclidean_and_can_prove_prediction_exit() {
        let shape = cuboid(0.05, 0.04, 0.03);
        let h = shape.as_convex_polyhedron().unwrap();
        let mut m = Manifold::new();
        let margin = 0.02;
        let pos = r::Pose::translation(0.105, 0.087, 0.065);
        let proof = full_query(&pos, h, h, &mut m, margin);
        check_closest(&proof);
        assert!(proof.separated_query && proof.closest_witness_checked);
        let expected = (0.005_f64.powi(2) + 0.007_f64.powi(2) + 0.005_f64.powi(2)).sqrt();
        assert!((proof.separation_m - expected).abs() < 1e-7);
        // The registered edge graph also contains facet diagonals. An
        // independent enumeration of ALL vertex-pair crosses bounds even
        // that overcomplete SAT set above by 9.736843 mm in this direction.
        assert!(proof.registered_sat_separation_m < 0.009737);
        assert!(proof.separation_m > proof.registered_sat_separation_m + 0.0002);
        check_final_separated(&pos, h, h, &m, margin).unwrap();
        let prediction_exit = 0.00984;
        let proof = full_query(&pos, h, h, &mut m, prediction_exit);
        assert!(proof.registered_sat_separation_m < f64::from(prediction_exit));
        assert!(proof.beyond_prediction_lower_bound_m.unwrap() > f64::from(prediction_exit));
        assert!(m.points.is_empty() && !proof.closest_witness_checked);
        check_final_separated(&pos, h, h, &m, prediction_exit).unwrap();
    }
    #[test]
    fn positive_parallel_hexagons_retain_the_complete_closest_face_area() {
        let shape = hexagonal_prism(0.04);
        let h = shape.as_convex_polyhedron().unwrap();
        let mut m = Manifold::new();
        let pos = r::Pose::translation(0., 0.045, 0.);
        let proof = full_query(&pos, h, h, &mut m, 0.02);
        check_closest(&proof);
        assert_eq!(proof.clipped_points, 6);
        assert_eq!(m.points.len(), 7);
        assert!((proof.separation_m - 0.005).abs() < 1e-7);
        let polygon = &m.points[..proof.clipped_points];
        let twice_area: f64 = (0..polygon.len())
            .map(|i| {
                let a = polygon[i].local_p1;
                let b = polygon[(i + 1) % polygon.len()].local_p1;
                f64::from(a.x) * f64::from(b.z) - f64::from(b.x) * f64::from(a.z)
            })
            .sum();
        let area = 3. * 3_f64.sqrt() * 0.04_f64.powi(2) / 2.;
        assert!((twice_area.abs() * 0.5 - area).abs() < 1e-9);
        check_final_separated(&pos, h, h, &m, 0.02).unwrap();
    }
    #[test]
    fn positive_skew_rod_witnesses_match_independent_line_normal_and_width() {
        let a = cuboid(0.1, 0.003, 0.005);
        let b = cuboid(0.1, 0.004, 0.006);
        let rotation = r::Rotation::from_rotation_z(0.7) * r::Rotation::from_rotation_y(0.4);
        let n = r::Vector::X.cross(rotation * r::Vector::X).normalize();
        let width = 0.003 * n.y.abs()
            + 0.005 * n.z.abs()
            + 0.004 * (rotation * r::Vector::Y).dot(n).abs()
            + 0.006 * (rotation * r::Vector::Z).dot(n).abs();
        let pos = r::Pose::from_parts(n * (width + 0.003), rotation);
        let mut m = Manifold::new();
        let proof = full_query(
            &pos,
            a.as_convex_polyhedron().unwrap(),
            b.as_convex_polyhedron().unwrap(),
            &mut m,
            0.02,
        );
        check_closest(&proof);
        assert!((proof.separation_m - 0.003).abs() < 1e-7);
        assert!(m.local_n1.dot(n) > 0.999999);
        assert_eq!(proof.clipped_points, 1);
    }
    #[test]
    fn captured_positive_normal_and_closest_failures_get_checked_registered_geometry() {
        #[derive(Deserialize)]
        struct Captured {
            relative_translation: [f32; 3],
            relative_rotation_xyzw: [f32; 4],
            registered_shared_shapes: [r::SharedShape; 2],
            final_manifold_after_existing_repair: Manifold,
            prediction_m: f32,
        }
        for text in [
            include_str!("fixtures/positive_zero_normal.json"),
            include_str!("fixtures/positive_bad_closest.json"),
        ] {
            let case: Captured = serde_json::from_str(text).unwrap();
            let [a, b] = &case.registered_shared_shapes;
            let a = a.as_convex_polyhedron().unwrap();
            let b = b.as_convex_polyhedron().unwrap();
            let pos = r::Pose::from_parts(
                r::Vector::from_array(case.relative_translation),
                r::Rotation::from_array(case.relative_rotation_xyzw),
            );
            let mut m = case.final_manifold_after_existing_repair;
            assert!(
                check_final_separated(&pos, a, b, &m, case.prediction_m).is_err(),
                "ordinary-query negative control stopped failing"
            );
            let proof = full_query(&pos, a, b, &mut m, case.prediction_m);
            assert!(proof.separated_query);
            if proof.closest_witness_checked {
                check_closest(&proof);
            }
            check_final_separated(&pos, a, b, &m, case.prediction_m).unwrap();
        }
    }

    #[test]
    fn sliver_feature_boundary_uses_euclidean_distance_not_inflated_halfplanes() {
        let triangle = [
            DVec2::new(0., 0.),
            DVec2::new(1., 0.),
            DVec2::new(0.5, 1e-10),
        ];
        let far = DVec2::new(2.5, 0.);
        let old = (0..3).all(|i| {
            let edge = triangle[(i + 1) % 3] - triangle[i];
            cross2(edge, far - triangle[i]) >= -NUMERIC_M * edge.length()
        });
        assert!(
            old,
            "negative control no longer reproduces the unbounded sliver tolerance"
        );
        assert!(!inside_feature(far, &triangle));
        assert!(inside_feature(DVec2::new(0.5, 5e-11), &triangle));
        assert!(inside_feature(DVec2::new(0.5, -0.5e-9), &triangle));
    }
    #[test]
    fn captured_inward_tiny_and_sliver_cases_keep_checked_separated_geometry() {
        #[derive(Deserialize)]
        struct Captured {
            relative_translation: [f32; 3],
            relative_rotation_xyzw: [f32; 4],
            registered_shared_shapes: [r::SharedShape; 2],
            ordinary_manifold_before_sat: Manifold,
        }
        for text in [
            include_str!("fixtures/positive_inward_projection.json"),
            include_str!("fixtures/positive_tiny_direction.json"),
            include_str!("fixtures/positive_sliver_boundary.json"),
        ] {
            let case: Captured = serde_json::from_str(text).unwrap();
            let [a, b] = &case.registered_shared_shapes;
            let a = a.as_convex_polyhedron().unwrap();
            let b = b.as_convex_polyhedron().unwrap();
            let pos = r::Pose::from_parts(
                r::Vector::from_array(case.relative_translation),
                r::Rotation::from_array(case.relative_rotation_xyzw),
            );
            let mut m = case.ordinary_manifold_before_sat;
            let proof = full_query(&pos, a, b, &mut m, 0.02);
            assert!(proof.separated_query);
            check_closest(&proof);
            check_final_separated(&pos, a, b, &m, 0.02).unwrap();
        }
    }
}

//! Rational construction for overlapping rigid unions. Predicates alone cannot
//! repair a rounded vertex that crossed a later cut plane. Float coordinates
//! are promoted exactly; clipping, cap ordering and volume moments stay rational
//! until positive tetrahedral integration. Moments use compensated f64 sums;
//! no rounded vertex is used to classify a later cut or decide volume sign. Work/precision exhaustion returns an error.
use super::*;
use num_rational::BigRational as Q;
use num_traits::{Signed, ToPrimitive, Zero};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};
type Point = [Q; 3];
fn q(x: f64) -> Q {
    Q::from_float(x).expect("validated finite pressure geometry")
}
fn point(p: DVec3) -> Point {
    p.to_array().map(q)
}
fn sub(a: &Point, b: &Point) -> Point {
    std::array::from_fn(|i| &a[i] - &b[i])
}
fn cross(a: &Point, b: &Point) -> Point {
    [
        &a[1] * &b[2] - &a[2] * &b[1],
        &a[2] * &b[0] - &a[0] * &b[2],
        &a[0] * &b[1] - &a[1] * &b[0],
    ]
}
fn dot(a: &Point, b: &Point) -> Q {
    &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
}
fn float(x: &Q) -> Result<f64, String> {
    let value = x
        .to_f64()
        .ok_or("Exact pressure value cannot be represented in f64")?;
    if !value.is_finite() || (value == 0. && !x.is_zero()) {
        return Err("Exact pressure value overflow/underflow".into());
    }
    Ok(value)
}
fn vector(p: &Point) -> Result<DVec3, String> {
    Ok(DVec3::new(float(&p[0])?, float(&p[1])?, float(&p[2])?))
}
#[derive(Clone)]
struct Halfspace {
    normal: Point,
    constant: Q,
}
impl Halfspace {
    fn face(t: [DVec3; 3]) -> Self {
        let [a, b, c] = t.map(point);
        let normal = cross(&sub(&b, &a), &sub(&c, &a));
        let constant = -dot(&normal, &a);
        Self { normal, constant }
    }
    fn at(&self, p: &Point) -> Q {
        dot(&self.normal, p) + &self.constant
    }
}
#[derive(Clone)]
struct ExactPoly {
    points: Vec<Point>,
    faces: Vec<Vec<usize>>,
}
impl ExactPoly {
    fn tet(t: [DVec3; 4]) -> Self {
        Self {
            points: t.map(point).into(),
            faces: geometry::FACES.map(|f| f.into()).into(),
        }
    }
    fn compact(self) -> Self {
        let used: BTreeSet<_> = self.faces.iter().flatten().copied().collect();
        let mut table = BTreeMap::new();
        let mut indices = vec![usize::MAX; self.points.len()];
        let mut points = Vec::new();
        for i in used {
            let p = &self.points[i];
            indices[i] = *table.entry(p.clone()).or_insert_with(|| {
                let n = points.len();
                points.push(p.clone());
                n
            });
        }
        let mut faces = Vec::new();
        for face in self.faces {
            let mut out = Vec::new();
            for i in face {
                if out.last() != Some(&indices[i]) {
                    out.push(indices[i]);
                }
            }
            if out.len() > 1 && out.first() == out.last() {
                out.pop();
            }
            if out.len() >= 3 {
                faces.push(out);
            }
        }
        Self { points, faces }
    }
    fn clip(self, plane: &Halfspace) -> Result<Self, String> {
        if self.faces.is_empty() {
            return Ok(self);
        }
        let values: Vec<_> = self.points.iter().map(|p| plane.at(p)).collect();
        let used: BTreeSet<_> = self.faces.iter().flatten().copied().collect();
        if used.iter().all(|i| !values[*i].is_positive()) {
            return Ok(self);
        }
        if used.iter().all(|i| !values[*i].is_negative()) {
            return Ok(Self {
                points: Vec::new(),
                faces: Vec::new(),
            });
        }
        let mut points = self.points;
        let mut faces = Vec::new();
        let mut edges = BTreeMap::new();
        let mut cuts = BTreeSet::new();
        for face in self.faces {
            let mut out = Vec::new();
            let mut a = *face.last().ok_or("Empty exact pressure face")?;
            for b in face {
                let sa = &values[a];
                let sb = &values[b];
                if sa.is_positive() != sb.is_positive() {
                    let id = if sa.is_zero() {
                        a
                    } else if sb.is_zero() {
                        b
                    } else {
                        let key = (a.min(b), a.max(b));
                        if let Some(id) = edges.get(&key) {
                            *id
                        } else {
                            let t = sa / (sa - sb);
                            let p: Point = std::array::from_fn(|i| {
                                &points[a][i] + (&points[b][i] - &points[a][i]) * &t
                            });
                            if p.iter().any(|x| x.numer().bits() + x.denom().bits() > 8192) {
                                return Err(
                                    "Exact pressure coordinate precision bound exceeded".into()
                                );
                            }
                            if points.len() >= 128 {
                                return Err("Exact pressure vertex bound exceeded".into());
                            }
                            let id = points.len();
                            points.push(p);
                            edges.insert(key, id);
                            id
                        }
                    };
                    if out.last() != Some(&id) {
                        out.push(id);
                    }
                    cuts.insert(id);
                }
                if !sb.is_positive() {
                    if out.last() != Some(&b) {
                        out.push(b);
                    }
                    if sb.is_zero() {
                        cuts.insert(b);
                    }
                }
                a = b;
            }
            if out.len() > 1 && out.first() == out.last() {
                out.pop();
            }
            if out.len() >= 3 {
                faces.push(out);
            }
        }
        if cuts.len() >= 3 {
            let mut center: [Q; 3] = std::array::from_fn(|_| Q::zero());
            for i in &cuts {
                for a in 0..3 {
                    center[a] += &points[*i][a];
                }
            }
            for x in &mut center {
                *x /= Q::from_integer(cuts.len().into());
            }
            let drop = (0..3).max_by_key(|i| plane.normal[*i].abs()).unwrap();
            if plane.normal[drop].is_zero() {
                return Err("Exact pressure cut has no normal".into());
            }
            let (x, y) = ((drop + 1) % 3, (drop + 2) % 3);
            let half = |p: &Point| p[y].is_negative() || (p[y].is_zero() && p[x].is_negative());
            let mut cap: Vec<_> = cuts.into_iter().collect();
            cap.sort_by(|a, b| {
                let a = sub(&points[*a], &center);
                let b = sub(&points[*b], &center);
                let h = half(&a).cmp(&half(&b));
                if h != Ordering::Equal {
                    return h;
                }
                let area = &a[x] * &b[y] - &a[y] * &b[x];
                if !area.is_zero() {
                    return if area.is_positive() {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    };
                }
                (&a[x] * &a[x] + &a[y] * &a[y]).cmp(&(&b[x] * &b[x] + &b[y] * &b[y]))
            });
            if plane.normal[drop].is_negative() {
                cap.reverse();
            }
            faces.push(cap);
        }
        if faces.len() > 32 {
            return Err("Exact pressure face bound exceeded".into());
        }
        Ok(Self { points, faces }.compact())
    }
    fn moments(&self, field: &Halfspace) -> Result<RawMoments, String> {
        if self.faces.is_empty() {
            return Ok(RawMoments::zero());
        }
        let origin = &self.points[*self.faces.iter().flatten().min().unwrap()];
        let mut result = RawMoments::zero();
        let six = Q::from_integer(6.into());
        let four = Q::from_integer(4.into());
        // Cone from a boundary vertex: exact convexity gives nonnegative
        // tetrahedra, without forming a high-denominator rational centroid.
        for face in &self.faces {
            let a = &self.points[face[0]];
            for i in 1..face.len() - 1 {
                let b = &self.points[face[i]];
                let c = &self.points[face[i + 1]];
                let dv = dot(&sub(a, origin), &cross(&sub(b, origin), &sub(c, origin))) / &six;
                if dv.is_negative() {
                    return Err("Exact pressure boundary has an inward triangle".into());
                }
                if dv.is_zero() {
                    continue;
                }
                let pressure = (field.at(origin) + field.at(a) + field.at(b) + field.at(c)) / &four;
                let energy = &pressure * &dv;
                if energy.is_negative() {
                    return Err("Exact pressure integral is negative".into());
                }
                let v = float(&dv)?;
                result.volume.add(v, v.abs());
                let centroid: Point = std::array::from_fn(|axis| {
                    (&origin[axis] + &a[axis] + &b[axis] + &c[axis]) / &four
                });
                let centroid = vector(&centroid)?;
                for axis in 0..3 {
                    let moment = centroid[axis] * v;
                    result.first[axis].add(moment, moment.abs());
                }
                let e = float(&energy)?;
                result.energy.add(e, e.abs());
            }
        }
        Ok(result)
    }
    fn bounds(&self) -> (Point, Point) {
        (
            std::array::from_fn(|i| self.points.iter().map(|p| &p[i]).min().unwrap().clone()),
            std::array::from_fn(|i| self.points.iter().map(|p| &p[i]).max().unwrap().clone()),
        )
    }
}
#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
    absolute: f64,
}
impl Sum {
    fn add(&mut self, value: f64, absolute: f64) {
        let total = self.value + value;
        self.correction += if self.value.abs() >= value.abs() {
            (self.value - total) + value
        } else {
            (value - total) + self.value
        };
        self.value = total;
        self.absolute += absolute;
    }
    fn result(&self) -> f64 {
        self.value + self.correction
    }
}
struct RawMoments {
    volume: Sum,
    first: [Sum; 3],
    energy: Sum,
}
impl RawMoments {
    fn zero() -> Self {
        Self {
            volume: Sum::default(),
            first: std::array::from_fn(|_| Sum::default()),
            energy: Sum::default(),
        }
    }
    fn add(&mut self, b: Self, positive: bool) {
        let sign = if positive { 1. } else { -1. };
        self.volume.add(sign * b.volume.result(), b.volume.absolute);
        self.energy.add(sign * b.energy.result(), b.energy.absolute);
        for (a, b) in self.first.iter_mut().zip(b.first) {
            a.add(sign * b.result(), b.absolute);
        }
    }
    fn converted(self) -> Result<(Moments, f64, f64, f64), String> {
        let volume = self.volume.result();
        let energy = self.energy.result();
        if volume < 0. || energy < 0. || !volume.is_finite() || !energy.is_finite() {
            return Err("Nonrepresentable/nonnegative rigid union moment".into());
        }
        if volume == 0. {
            if self.volume.absolute != 0. {
                return Err("Rigid union cancellation needs higher-precision moments".into());
            }
            return Ok((Moments::default(), 0., 0., 0.));
        }
        let first = DVec3::from_array(self.first.map(|s| s.result()));
        let origin = first / volume;
        let m = Moments {
            volume,
            first,
            origin,
            relative_first: first - origin * volume,
        };
        Ok((
            m,
            energy,
            self.volume.absolute * 16. * f64::EPSILON,
            self.energy.absolute * 16. * f64::EPSILON,
        ))
    }
}
struct Cell {
    planes: [Halfspace; 4],
    low: Point,
    high: Point,
}
struct Rule {
    cells: Vec<Cell>,
    groups: Vec<Vec<usize>>,
    visits: usize,
    tested: usize,
    intersecting: usize,
    depth: usize,
}
impl Rule {
    fn integrate(
        &mut self,
        poly: &ExactPoly,
        start: usize,
        depth: usize,
        field: &Halfspace,
    ) -> Result<RawMoments, String> {
        if depth > MAX_DEPTH {
            return Err("Exact union overlapping-component depth bound exceeded".into());
        }
        self.depth = self.depth.max(depth);
        let mut total = RawMoments::zero();
        let (low, high) = poly.bounds();
        for group in start..self.groups.len() {
            for member in 0..self.groups[group].len() {
                let id = self.groups[group][member];
                self.visits += 1;
                if self.visits > MAX_VISITS {
                    return Err("Exact union subset-work bound exceeded".into());
                }
                if depth == 0 {
                    self.tested += 1;
                }
                let cell = &self.cells[id];
                if (0..3).any(|i| high[i] <= cell.low[i] || cell.high[i] <= low[i]) {
                    continue;
                }
                let mut overlap = poly.clone();
                for plane in &cell.planes {
                    overlap = overlap.clip(plane)?;
                }
                if overlap.faces.is_empty() {
                    continue;
                }
                let moments = overlap.moments(field)?;
                if moments.volume.result() == 0. {
                    continue;
                }
                if depth == 0 {
                    self.intersecting += 1;
                }
                total.add(moments, true);
                // Tetrahedra within one registered star mesh are disjoint.
                // Only different rigid PARTS need inclusion-exclusion terms.
                if group + 1 < self.groups.len() {
                    total.add(
                        self.integrate(&overlap, group + 1, depth + 1, field)?,
                        false,
                    );
                }
            }
        }
        Ok(total)
    }
}
pub(super) fn union(
    ac: &[WorldCell],
    bc: &[WorldCell],
    am: RigidMotion,
    bm: RigidMotion,
    delta: DVec3,
) -> Result<Response, String> {
    let total = bc.len();
    let minimum = ac
        .iter()
        .fold(DVec3::splat(f64::INFINITY), |m, a| m.min(a.minimum));
    let maximum = ac
        .iter()
        .fold(DVec3::splat(f64::NEG_INFINITY), |m, a| m.max(a.maximum));
    // Bounds enclose every source tetrahedron. A disjoint interval proves
    // zero volume; no skin, distance epsilon or geometry reduction is used.
    let bc: Vec<_> = bc
        .iter()
        .filter(|b| !((maximum.cmple(b.minimum)) | (b.maximum.cmple(minimum))).any())
        .collect();
    if bc.is_empty() {
        return Ok(Response {
            union_input_tetrahedra: total,
            ..Default::default()
        });
    }
    let cells = bc
        .iter()
        .map(|b| Cell {
            planes: geometry::FACES.map(|f| Halfspace::face(f.map(|i| b.tet[i]))),
            low: point(b.minimum),
            high: point(b.maximum),
        })
        .collect();
    let mut group_ids = BTreeMap::<usize, Vec<usize>>::new();
    for (i, b) in bc.iter().enumerate() {
        group_ids.entry(b.rigid_component).or_default().push(i);
    }
    let mut rule = Rule {
        cells,
        groups: group_ids.into_values().collect(),
        visits: 0,
        tested: 0,
        intersecting: 0,
        depth: 0,
    };
    let mut result = Response {
        union_input_tetrahedra: total,
        union_candidate_tetrahedra: bc.len(),
        ..Default::default()
    };
    for a in ac {
        let whole = ExactPoly::tet(a.tet);
        let p = a.pressure.unwrap();
        let field = if let Some(t) = p.triangle {
            let mut f = Halfspace::face(t);
            let scale = q(p.triangle_scale);
            f.normal = f.normal.map(|v| v * &scale);
            f.constant *= scale;
            f
        } else {
            let normal = point(p.gradient);
            let constant = q(p.value) - dot(&normal, &point(p.origin));
            Halfspace { normal, constant }
        };
        let (moments, energy, volume_error, energy_error) =
            rule.integrate(&whole, 0, 0, &field)?.converted()?;
        if moments.volume > 0. {
            result.overlap_volume_m3 += moments.volume;
            result.union_volume_roundoff_estimate_m3 += volume_error;
            result.union_energy_roundoff_estimate_j += energy_error;
            super::accumulate(&mut result, moments, p, am, bm, delta, Some(energy))?;
        }
    }
    result.tested_tetrahedron_pairs = rule.tested;
    result.intersecting_tetrahedron_pairs = rule.intersecting;
    result.union_intersection_visits = rule.visits;
    result.union_maximum_depth = rule.depth;
    Ok(result)
}

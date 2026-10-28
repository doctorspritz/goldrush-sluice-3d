use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug)]
pub(super) struct Affine {
    pub origin: DVec3,
    pub value: f64,
    pub gradient: DVec3,
    pub triangle: Option<[DVec3; 3]>,
    pub triangle_scale: f64,
}
fn orient(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    let coordinate = |p: DVec3| robust::Coord3D {
        x: p.x,
        y: p.y,
        z: p.z,
    };
    robust::orient3d(coordinate(a), coordinate(b), coordinate(c), coordinate(d))
}
impl Affine {
    pub fn at(self, p: DVec3) -> f64 {
        if let Some([a, b, c]) = self.triangle {
            -orient(a, b, c, p) * self.triangle_scale
        } else {
            self.value + self.gradient.dot(p - self.origin)
        }
    }
    pub fn negative(self) -> Self {
        Self {
            value: -self.value,
            gradient: -self.gradient,
            triangle_scale: -self.triangle_scale,
            ..self
        }
    }
    pub fn difference(self, b: Self, origin: DVec3) -> Self {
        Self {
            origin,
            value: self.at(origin) - b.at(origin),
            gradient: self.gradient - b.gradient,
            triangle: None,
            triangle_scale: 0.,
        }
    }
    pub fn rate(self, motion: RigidMotion, com: DVec3, origin: DVec3) -> Self {
        Self {
            origin,
            value: -self.gradient.dot(
                motion.linear_velocity_m_s + motion.angular_velocity_rad_s.cross(origin - com),
            ),
            gradient: motion.angular_velocity_rad_s.cross(self.gradient),
            triangle: None,
            triangle_scale: 0.,
        }
    }
}
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Moments {
    pub volume: f64,
    pub first: DVec3,
    pub origin: DVec3,
    pub relative_first: DVec3,
}
impl Moments {
    pub fn scaled(self, w: f64) -> Self {
        Self {
            volume: self.volume * w,
            first: self.first * w,
            relative_first: self.relative_first * w,
            ..self
        }
    }
    pub fn pressure(self, p: Affine) -> f64 {
        p.at(self.origin) * self.volume + p.gradient.dot(self.relative_first)
    }
}
#[derive(Clone, Debug)]
pub(super) struct Poly {
    pub points: Vec<DVec3>,
    pub faces: Vec<Vec<usize>>,
}
impl Poly {
    pub fn tetrahedron(t: [DVec3; 4]) -> Self {
        Self {
            points: t.into(),
            faces: geometry::FACES.into_iter().map(|f| f.into()).collect(),
        }
    }
    /// Exact half-space boundary, without a skin, snapped points or area cutoffs.
    /// An edge cache shares the same intersection vertex between adjacent faces.
    pub fn clip(self, plane: Affine) -> Result<Self, String> {
        if self.faces.is_empty() {
            return Ok(self);
        }
        let values: Vec<_> = self.points.iter().map(|p| plane.at(*p)).collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite pressure clip".into());
        }
        let used: BTreeSet<_> = self.faces.iter().flatten().copied().collect();
        let lo = used
            .iter()
            .map(|i| values[*i])
            .fold(f64::INFINITY, f64::min);
        let hi = used
            .iter()
            .map(|i| values[*i])
            .fold(f64::NEG_INFINITY, f64::max);
        if hi <= 0. {
            return Ok(self);
        }
        if lo >= 0. {
            return Ok(Self {
                points: Vec::new(),
                faces: Vec::new(),
            });
        }
        let mut points = self.points;
        let mut faces = Vec::new();
        let mut cuts = BTreeSet::new();
        let mut edges = BTreeMap::new();
        for face in self.faces {
            let mut output = Vec::new();
            let mut a = *face.last().ok_or("Empty pressure polyhedron face")?;
            for b in face {
                let sa = values[a];
                let sb = values[b];
                if (sa <= 0.) != (sb <= 0.) {
                    let index = if sa == 0. {
                        a
                    } else if sb == 0. {
                        b
                    } else {
                        let key = (a.min(b), a.max(b));
                        if let Some(i) = edges.get(&key) {
                            *i
                        } else {
                            let scale = sa.abs().max(sb.abs());
                            let t = (sa / scale) / (sa / scale - sb / scale);
                            // Preserve equal endpoint coordinates and use the nearer endpoint.
                            let p = if t <= 0.5 {
                                points[a] + (points[b] - points[a]) * t
                            } else {
                                points[b] + (points[a] - points[b]) * (1. - t)
                            };
                            if !p.is_finite() || points.len() >= 128 {
                                return Err("Pressure intersection vertex bound or overflow".into());
                            }
                            let i = points.len();
                            points.push(p);
                            edges.insert(key, i);
                            i
                        }
                    };
                    push(&mut output, index);
                    cuts.insert(index);
                }
                if sb <= 0. {
                    push(&mut output, b);
                    if sb == 0. {
                        cuts.insert(b);
                    }
                }
                a = b;
            }
            close(&mut output);
            if output.len() >= 3 {
                faces.push(output);
            }
        }
        if cuts.len() >= 3 {
            let normal = plane.gradient.normalize();
            if !normal.is_finite() {
                return Err("Unresolved pressure cut plane".into());
            }
            let center = cuts.iter().map(|i| points[*i]).sum::<DVec3>() / cuts.len() as f64;
            let axis = if normal.x.abs() <= normal.y.abs() && normal.x.abs() <= normal.z.abs() {
                DVec3::X
            } else if normal.y.abs() <= normal.z.abs() {
                DVec3::Y
            } else {
                DVec3::Z
            };
            let u = normal.cross(axis).normalize();
            let v = normal.cross(u);
            let mut cap: Vec<_> = cuts.into_iter().collect();
            cap.sort_by(|a, b| {
                let a = points[*a] - center;
                let b = points[*b] - center;
                a.dot(v)
                    .atan2(a.dot(u))
                    .total_cmp(&b.dot(v).atan2(b.dot(u)))
            });
            faces.push(cap);
        }
        if faces.len() > 32 {
            return Err("Pressure intersection face bound exceeded".into());
        }
        Ok(Self { points, faces }.canonical())
    }
    fn canonical(self) -> Self {
        // Only derived intersection vertices with EXACTLY equal coordinates
        // share an index. Original registered solid geometry is never changed.
        let used: BTreeSet<_> = self.faces.iter().flatten().copied().collect();
        let mut by_coordinate = BTreeMap::new();
        let mut points = Vec::new();
        let mut indices = vec![usize::MAX; self.points.len()];
        for i in used {
            let point = self.points[i];
            let key = point
                .to_array()
                .map(|x| if x == 0. { 0 } else { x.to_bits() });
            indices[i] = *by_coordinate.entry(key).or_insert_with(|| {
                let id = points.len();
                points.push(point);
                id
            });
        }
        let mut faces = Vec::new();
        for original in self.faces {
            let mut face = Vec::new();
            for i in original {
                push(&mut face, indices[i]);
            }
            close(&mut face);
            if face.len() < 3 {
                continue;
            }
            // A face and its reversed copy must choose the same fan diagonal,
            // including when rounded intersection vertices are not coplanar.
            let start = face.iter().enumerate().min_by_key(|(_, i)| **i).unwrap().0;
            face.rotate_left(start);
            faces.push(face);
        }
        Self { points, faces }
    }
    fn has_volume(&self, used: &BTreeSet<usize>) -> bool {
        let Some(&first) = used.first() else {
            return false;
        };
        let a = self.points[first];
        let Some(b) = used.iter().map(|i| self.points[*i]).find(|p| *p != a) else {
            return false;
        };
        let collinear = |c: DVec3| {
            [(0, 1), (0, 2), (1, 2)].into_iter().all(|(x, y)| {
                robust::orient2d(
                    robust::Coord { x: a[x], y: a[y] },
                    robust::Coord { x: b[x], y: b[y] },
                    robust::Coord { x: c[x], y: c[y] },
                ) == 0.
            })
        };
        let Some(c) = used
            .iter()
            .map(|i| self.points[*i])
            .find(|p| !collinear(*p))
        else {
            return false;
        };
        used.iter().any(|i| orient(a, b, c, self.points[*i]) != 0.)
    }
    pub fn moments(&self) -> Result<Moments, String> {
        if self.faces.is_empty() {
            return Ok(Moments::default());
        }
        let used: BTreeSet<_> = self.faces.iter().flatten().copied().collect();
        if !self.has_volume(&used) {
            return Ok(Moments::default());
        }
        let anchor = self.points[*used.first().unwrap()];
        let center = anchor
            + used.iter().map(|i| self.points[*i] - anchor).sum::<DVec3>() / used.len() as f64;
        let mut result = Moments::default();
        // Cancel exact opposite oriented triangles BEFORE floating arithmetic.
        // This gives a doubled, zero-volume clipping face zero measure without
        // a volume epsilon; genuinely thin four-point solids remain admitted.
        let mut triangles = BTreeMap::<[usize; 3], i32>::new();
        for face in &self.faces {
            for i in 1..face.len() - 1 {
                let mut ids = [face[0], face[i], face[i + 1]];
                let mut sign = 1;
                for (a, b) in [(0, 1), (1, 2), (0, 1)] {
                    if ids[a] > ids[b] {
                        ids.swap(a, b);
                        sign = -sign;
                    }
                }
                if ids[0] != ids[1] && ids[1] != ids[2] {
                    *triangles.entry(ids).or_default() += sign;
                }
            }
        }
        for (ids, count) in triangles {
            if count == 0 {
                continue;
            }
            let [a, b, c] = ids.map(|i| self.points[i]);
            let volume = orient(a, b, c, center) * (f64::from(count) / 6.);
            result.volume += volume;
            result.relative_first += ((a - center) + (b - center) + (c - center)) * (volume / 4.);
        }
        result.origin = center;
        result.first = center * result.volume + result.relative_first;
        if !result.volume.is_finite() || result.volume < 0. || !result.first.is_finite() {
            return Err(format!(
                "Invalid pressure intersection moments: volume={} poly={self:?}",
                result.volume
            ));
        }
        Ok(result)
    }
}
fn push(face: &mut Vec<usize>, index: usize) {
    if face.last() != Some(&index) {
        face.push(index);
    }
}
fn close(face: &mut Vec<usize>) {
    if face.len() > 1 && face.first() == face.last() {
        face.pop();
    }
}
#[derive(Clone)]
pub(super) struct WorldCell {
    pub tet: [DVec3; 4],
    pub planes: [Affine; 4],
    pub pressure: Option<Affine>,
    pub minimum: DVec3,
    pub maximum: DVec3,
    pub rigid_component: usize,
}
impl WorldCell {
    pub fn new(tet: [DVec3; 4], pressure: Option<Affine>) -> Result<Self, String> {
        let matrix = glam::DMat3::from_cols(tet[1] - tet[0], tet[2] - tet[0], tet[3] - tet[0]);
        if !matrix.is_finite() || matrix.determinant() <= 0. {
            return Err("Invalid transformed pressure tetrahedron".into());
        }
        let planes = geometry::FACES.map(|f| {
            let origin = tet[f[0]];
            let gradient = (tet[f[1]] - origin).cross(tet[f[2]] - origin).normalize();
            Affine {
                origin,
                value: 0.,
                gradient,
                triangle: Some(f.map(|i| tet[i])),
                triangle_scale: 1. / (tet[f[1]] - origin).cross(tet[f[2]] - origin).length(),
            }
        });

        if planes.iter().any(|p| !p.gradient.is_finite())
            || pressure.is_some_and(|p| !p.gradient.is_finite() || !p.value.is_finite())
        {
            return Err("Pressure gradient overflow".into());
        }
        Ok(Self {
            tet,
            planes,
            pressure,
            rigid_component: 0,
            minimum: tet
                .into_iter()
                .fold(DVec3::splat(f64::INFINITY), DVec3::min),
            maximum: tet
                .into_iter()
                .fold(DVec3::splat(f64::NEG_INFINITY), DVec3::max),
        })
    }
    pub fn intersects(&self, b: &Self) -> bool {
        self.maximum.cmpgt(b.minimum).all() && b.maximum.cmpgt(self.minimum).all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_construction_vertices_and_opposite_faces_have_zero_measure() {
        let points = vec![
            DVec3::new(
                0.22211018094608345,
                -0.23196208943498517,
                -0.6769841644038268,
            ),
            DVec3::new(0.2147180999600663, 0.43621953652155326, -0.7506584071393952),
            DVec3::new(
                0.22211018094608345,
                -0.23196208943498517,
                -0.6769841644038268,
            ),
            DVec3::new(0.2147180999600663, 0.43621953652155326, -0.7506584071393952),
            DVec3::new(
                0.2407866007592468,
                -0.22604949330827132,
                -0.4908429290265144,
            ),
            DVec3::new(
                0.22003178999915515,
                0.40687853839719895,
                -0.6976987479443824,
            ),
        ];
        let faces = vec![
            vec![2, 0, 1, 3],
            vec![4, 0, 2],
            vec![5, 3, 1],
            vec![5, 1, 0, 4],
            vec![3, 5, 4, 2],
        ];
        let poly = Poly { points, faces }.canonical();
        assert_eq!(poly.points.len(), 4);
        assert_eq!(poly.moments().unwrap().volume, 0.);
    }
    #[test]
    fn truly_thin_tetrahedra_keep_positive_volume_without_a_cutoff() {
        for thickness in [1e-10, 1e-18, 1e-30] {
            let tet = [
                DVec3::ZERO,
                DVec3::X,
                DVec3::Y,
                DVec3::new(0.2, 0.2, thickness),
            ];
            let poly = Poly::tetrahedron(tet).canonical();
            assert_eq!(poly.points.len(), 4);
            let volume = poly.moments().unwrap().volume;
            assert!(
                (volume / (thickness / 6.) - 1.).abs() < 1e-12,
                "thin volume changed: {volume}"
            );
        }
    }
}

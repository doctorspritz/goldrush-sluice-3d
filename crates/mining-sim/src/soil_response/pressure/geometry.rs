use super::*;
use std::collections::BTreeMap;

pub(super) const FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];
const MAX_TETS: usize = 16_384;
fn volume(t: [DVec3; 4]) -> f64 {
    (t[1] - t[0]).dot((t[2] - t[0]).cross(t[3] - t[0])) / 6.
}

/// All registered vertices and faces are retained. The declared boundary is
/// the complete ordered fan triangulation of each registered convex face.
/// There is no hull regeneration, vertex reduction or collision margin.
#[derive(Clone, Debug)]
pub struct RegisteredSolid {
    pub(super) points: Vec<DVec3>,
    pub(super) faces: Vec<Vec<usize>>,
    pub(super) tets: Vec<[DVec3; 4]>,
    pub(super) volume: f64,
}
impl RegisteredSolid {
    pub fn from_hull(hull: &rapier3d::parry::shape::ConvexPolyhedron) -> Result<Self, String> {
        let points: Vec<_> = hull
            .points()
            .iter()
            .map(|p| DVec3::from_array(p.to_array().map(f64::from)))
            .collect();
        if !(4..=4096).contains(&points.len()) || points.iter().any(|p| !p.is_finite()) {
            return Err("Unsupported registered solid vertex count or coordinates".into());
        }
        let interior = points.iter().copied().sum::<DVec3>() / points.len() as f64;
        let mut faces = Vec::new();
        let mut triangles = Vec::new();
        let mut edges = BTreeMap::<(usize, usize), (usize, i32)>::new();
        for face in hull.faces() {
            let start = face.first_vertex_or_edge as usize;
            let mut ids: Vec<_> = hull.vertices_adj_to_face()
                [start..start + face.num_vertices_or_edges as usize]
                .iter()
                .map(|i| *i as usize)
                .collect();
            if ids.len() < 3 || ids.iter().any(|i| *i >= points.len()) {
                return Err("Invalid registered face indices".into());
            }
            let origin = points[ids[0]];
            let normal = (1..ids.len() - 1)
                .map(|i| (points[ids[i]] - origin).cross(points[ids[i + 1]] - origin))
                .sum::<DVec3>();
            if !normal.is_finite() || normal.length_squared() == 0. {
                return Err("Degenerate registered face".into());
            }
            if normal.dot(origin - interior) < 0. {
                ids.reverse();
            }
            for i in 0..ids.len() {
                let a = ids[i];
                let b = ids[(i + 1) % ids.len()];
                if a == b {
                    return Err("Duplicate registered face edge".into());
                }
                let edge = edges.entry((a.min(b), a.max(b))).or_default();
                edge.0 += 1;
                edge.1 += if a < b { 1 } else { -1 };
            }
            for i in 1..ids.len() - 1 {
                let t = [points[ids[0]], points[ids[i]], points[ids[i + 1]]];
                if (t[1] - t[0]).cross(t[2] - t[0]).length_squared() > 0. {
                    triangles.push(t);
                }
            }
            faces.push(ids);
        }
        if edges.values().any(|v| *v != (2, 0)) || triangles.len() > MAX_TETS || faces.len() < 4 {
            return Err("Registered solid is not a bounded closed oriented mesh".into());
        }
        let mut total = 0.;
        let mut moment = DVec3::ZERO;
        for t in &triangles {
            let cell = [interior, t[0], t[1], t[2]];
            let v = volume(cell);
            if !v.is_finite() || v <= 0. {
                return Err("Registered solid is not strictly star-shaped at its interior".into());
            }
            total += v;
            moment += cell.into_iter().sum::<DVec3>() * (v / 4.);
        }
        let center = moment / total;
        if !total.is_finite() || total <= 0. || !center.is_finite() {
            return Err("Registered solid volume overflow".into());
        }
        let mut tets = Vec::with_capacity(triangles.len());
        for t in triangles {
            let cell = [center, t[0], t[1], t[2]];
            if volume(cell) <= 0. {
                return Err("Registered volume centroid left its source mesh".into());
            }
            tets.push(cell);
        }
        Ok(Self {
            points,
            faces,
            tets,
            volume: total,
        })
    }
    /// Explicit frozen-frame mapping of registered coefficients. Captured f32
    /// rotations are promoted and applied as supplied, never normalized. This
    /// constructs a reference mesh for observation; it is not a state update or
    /// permission to reset a material field between integration steps.
    pub fn captured_frame(
        &self,
        translation: DVec3,
        rotation: glam::DQuat,
    ) -> Result<Self, String> {
        if !translation.is_finite()
            || !rotation.is_finite()
            || (rotation.length_squared() - 1.).abs() > 0.0001
        {
            return Err("Invalid captured solid frame".into());
        }
        let map = |p| rotation * p + translation;
        let points: Vec<_> = self.points.iter().copied().map(map).collect();
        let tets: Vec<_> = self.tets.iter().map(|t| t.map(map)).collect();
        let volumes: Vec<_> = tets.iter().copied().map(volume).collect();
        if points.iter().any(|p| !p.is_finite())
            || volumes.iter().any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err("Captured solid frame changed orientation or overflowed".into());
        }
        Ok(Self {
            points,
            faces: self.faces.clone(),
            tets,
            volume: volumes.into_iter().sum(),
        })
    }
    pub fn vertices(&self) -> &[DVec3] {
        &self.points
    }
    pub fn faces(&self) -> &[Vec<usize>] {
        &self.faces
    }
    pub fn volume_m3(&self) -> f64 {
        self.volume
    }
    pub fn tetrahedra(&self) -> usize {
        self.tets.len()
    }
    /// Complete reference cells for read-only geometric qualification.
    pub fn reference_tetrahedra(&self) -> &[[DVec3; 4]] {
        &self.tets
    }
}
#[derive(Clone, Copy, Debug)]
pub(super) struct ReferenceCell {
    pub vertices: [DVec3; 4],
    pub field: Affine,
}
/// A continuous piecewise-affine pressure field on the complete finite solid.
/// Pressure is zero on the registered boundary and has a specified positive
/// value at the volume centroid. The parameter is a model pressure scale in Pa,
/// not silently a measured Young's modulus or a soil default.
#[derive(Clone, Debug)]
pub struct PressureBody {
    pub(super) solid: RegisteredSolid,
    pub(super) cells: Vec<ReferenceCell>,
    pressure_pa: f64,
}
impl PressureBody {
    pub fn new(solid: RegisteredSolid, centroid_pressure_pa: f64) -> Result<Self, String> {
        if !centroid_pressure_pa.is_finite() || centroid_pressure_pa <= 0. {
            return Err("Invalid pressure-field scale".into());
        }
        let cells = solid
            .tets
            .iter()
            .map(|t| {
                // A root field is exactly zero on its real boundary triangle.
                // Fit that plane directly: inverting four nodal values introduced
                // cancellation and false near-ties between identical face fields.
                let normal = (t[2] - t[1]).cross(t[3] - t[1]);
                let scale = centroid_pressure_pa / normal.dot(t[0] - t[1]);
                let field = Affine {
                    origin: t[1],
                    value: 0.,
                    gradient: normal * scale,
                    triangle: Some([t[1], t[2], t[3]]),
                    triangle_scale: scale,
                };
                if !field.gradient.is_finite() {
                    return Err("Pressure-field gradient overflow".to_string());
                }
                Ok(ReferenceCell {
                    vertices: *t,
                    field,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            solid,
            cells,
            pressure_pa: centroid_pressure_pa,
        })
    }
    pub fn solid(&self) -> &RegisteredSolid {
        &self.solid
    }
    pub fn pressure_scale_pa(&self) -> f64 {
        self.pressure_pa
    }
    pub fn tetrahedra(&self) -> usize {
        self.cells.len()
    }
    /// Refine the same field/domain, interpolating pressure at new interior
    /// vertices. This is a numerical partition control, not clod-size convergence.
    pub fn refined(&self, levels: u32) -> Result<Self, String> {
        let multiplier = 4usize
            .checked_pow(levels)
            .ok_or("Pressure refinement overflow")?;
        let count = self
            .cells
            .len()
            .checked_mul(multiplier)
            .ok_or("Pressure refinement overflow")?;
        if count > MAX_TETS {
            return Err("Pressure refinement cell bound exceeded".into());
        }
        let mut cells = self.cells.clone();
        for _ in 0..levels {
            let mut next = Vec::with_capacity(cells.len() * 4);
            for cell in cells {
                let center = cell.vertices.into_iter().sum::<DVec3>() / 4.;
                for face in FACES {
                    next.push(ReferenceCell {
                        vertices: [
                            center,
                            cell.vertices[face[0]],
                            cell.vertices[face[1]],
                            cell.vertices[face[2]],
                        ],
                        field: cell.field,
                    });
                }
            }
            cells = next;
        }
        Ok(Self {
            solid: self.solid.clone(),
            cells,
            pressure_pa: self.pressure_pa,
        })
    }
}

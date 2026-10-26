//! A finite mechanical support envelope, not another soil inventory. Its upper
//! surface is the actual registered heightfield; its lower surface is explicit.
//! Matching heightfield triangles give disjoint vertical wedges. No hull,
//! unbounded plane, thickness default or collision skin is substituted.
use super::*;
use glam::DQuat;
use rapier3d::parry::shape::HeightField;

const MAX_TRIANGLES: usize = 65_536;
#[derive(Clone, Debug)]
struct Cell {
    triangle: u32,
    vertices: [DVec3; 4],
    minimum: DVec3,
    maximum: DVec3,
}
impl Cell {
    fn new(triangle: u32, vertices: [DVec3; 4]) -> Result<Option<Self>, String> {
        let mut vertices = vertices;
        let v = volume(vertices);
        if !v.is_finite() {
            return Err("Terrain layer volume overflow".into());
        }
        if v == 0. {
            if (0..4).any(|i| (i + 1..4).any(|j| vertices[i] == vertices[j])) {
                return Ok(None);
            }
            return Err("Positive terrain wedge has unrepresentable volume".into());
        }
        if v < 0. {
            vertices.swap(1, 2);
        }
        Ok(Some(Self {
            triangle,
            vertices,
            minimum: vertices
                .into_iter()
                .fold(DVec3::splat(f64::INFINITY), DVec3::min),
            maximum: vertices
                .into_iter()
                .fold(DVec3::splat(f64::NEG_INFINITY), DVec3::max),
        }))
    }
}
fn volume(t: [DVec3; 4]) -> f64 {
    (t[1] - t[0]).dot((t[2] - t[0]).cross(t[3] - t[0])) / 6.
}
/// Closed volume between two explicitly supplied, matching heightfields.
/// This is a frozen geometry component. Construction/evaluation neither edits
/// Site stocks nor chooses bedrock, yield, friction, wet contact or save policy.
#[derive(Clone, Debug)]
pub struct RegisteredTerrainLayer {
    cells: Vec<Cell>,
    triangles: usize,
    volume: f64,
}
impl RegisteredTerrainLayer {
    pub fn from_heightfields(top: &HeightField, bottom: &HeightField) -> Result<Self, String> {
        if top.nrows() != bottom.nrows() || top.ncols() != bottom.ncols() {
            return Err("Terrain layer grids differ".into());
        }
        let triangles = top
            .nrows()
            .checked_mul(top.ncols())
            .and_then(|n| n.checked_mul(2))
            .filter(|n| (1..=MAX_TRIANGLES).contains(n))
            .ok_or("Terrain layer triangle bound exceeded")?;
        let mut cells = Vec::with_capacity(triangles * 3);
        for id in 0..triangles as u32 {
            let upper = top
                .triangle_at_id(id)
                .ok_or("Terrain layer holes are not admitted")?;
            let lower = bottom
                .triangle_at_id(id)
                .ok_or("Terrain layer lower boundary has holes")?;
            let promote =
                |p: rapier3d::prelude::Vector| DVec3::from_array(p.to_array().map(f64::from));
            let ids = top
                .triangle_vids_at_id(id)
                .ok_or("Missing terrain vertex ownership")?;
            if bottom.triangle_vids_at_id(id) != Some(ids) {
                return Err("Terrain layer projected triangulations differ".into());
            }
            let mut order = [0, 1, 2];
            order.sort_unstable_by_key(|i| ids[*i]);
            let upper = upper.vertices().map(promote);
            let lower = lower.vertices().map(promote);
            let a = order.map(|i| upper[i]);
            let b = order.map(|i| lower[i]);
            for (upper, lower) in a.iter().zip(&b) {
                if !upper.is_finite()
                    || !lower.is_finite()
                    || upper.x != lower.x
                    || upper.z != lower.z
                    || upper.y < lower.y
                {
                    return Err(
                        "Terrain layer boundaries cross or have different projected topology"
                            .into(),
                    );
                }
            }
            // Global vertex order gives every shared vertical quad the same
            // diagonal, including after captured floating-point frame mapping.
            // A vertical triangular wedge splits into three disjoint cells.
            // Zero-thickness corners create duplicate-vertex zero-volume cells;
            // every strictly positive thickness is retained without a cutoff.
            for tet in [
                [a[0], a[1], a[2], b[0]],
                [a[1], a[2], b[0], b[1]],
                [a[2], b[0], b[1], b[2]],
            ] {
                if let Some(cell) = Cell::new(id, tet)? {
                    cells.push(cell);
                }
            }
        }
        let volume: f64 = cells.iter().map(|c| volume(c.vertices)).sum();
        if !volume.is_finite() {
            return Err("Terrain layer total volume overflow".into());
        }
        Ok(Self {
            cells,
            triangles,
            volume,
        })
    }
    pub fn triangles(&self) -> usize {
        self.triangles
    }
    pub fn tetrahedra(&self) -> usize {
        self.cells.len()
    }
    pub fn volume_m3(&self) -> f64 {
        self.volume
    }
    pub fn reference_cells(&self) -> impl Iterator<Item = (u32, [DVec3; 4])> + '_ {
        self.cells.iter().map(|c| (c.triangle, c.vertices))
    }
    /// Explicit frozen mapping of represented coefficients, without quaternion
    /// normalization. Consuming the envelope reuses its bounded storage.
    pub fn captured_frame(mut self, translation: DVec3, rotation: DQuat) -> Result<Self, String> {
        if !translation.is_finite()
            || !rotation.is_finite()
            || (rotation.length_squared() - 1.).abs() > 0.0001
        {
            return Err("Invalid captured terrain layer frame".into());
        }
        for cell in &mut self.cells {
            let t = cell.vertices.map(|p| rotation * p + translation);
            if t.iter().any(|p| !p.is_finite()) || volume(t) <= 0. || !volume(t).is_finite() {
                return Err("Captured terrain layer changed orientation or overflowed".into());
            }
            *cell = Cell::new(cell.triangle, t)?.ok_or("Captured terrain cell collapsed")?;
        }
        self.volume = self.cells.iter().map(|c| volume(c.vertices)).sum();
        if !self.volume.is_finite() {
            return Err("Captured terrain total volume overflow".into());
        }
        Ok(self)
    }
    pub(super) fn cells(
        &self,
        source: &[WorldCell],
        motion: RigidMotion,
        delta: DVec3,
        limit: usize,
    ) -> Result<Vec<WorldCell>, String> {
        if !delta.is_finite() {
            return Err("Nonfinite terrain relative frame".into());
        }
        let low = source
            .iter()
            .fold(DVec3::splat(f64::INFINITY), |v, c| v.min(c.minimum));
        let high = source
            .iter()
            .fold(DVec3::splat(f64::NEG_INFINITY), |v, c| v.max(c.maximum));
        let mut selected = Vec::new();
        for (id, cell) in self.cells.iter().enumerate() {
            // For identity rotation, monotone addition gives the exact bounds
            // of the same represented vertices used by WorldCell::new below.
            let (minimum, maximum) = if motion.rotation == DQuat::IDENTITY {
                (delta + cell.minimum, delta + cell.maximum)
            } else {
                let t = cell.vertices.map(|p| delta + motion.rotation * p);
                if t.iter().any(|p| !p.is_finite()) {
                    return Err("Terrain contact frame overflow".into());
                }
                (
                    t.into_iter().fold(DVec3::splat(f64::INFINITY), DVec3::min),
                    t.into_iter()
                        .fold(DVec3::splat(f64::NEG_INFINITY), DVec3::max),
                )
            };
            if !minimum.is_finite() || !maximum.is_finite() {
                return Err("Terrain contact bounds overflow".into());
            }
            if high.cmple(minimum).any() || maximum.cmple(low).any() {
                continue;
            }
            if selected.len() >= limit {
                return Err("Finite terrain contact work bound exceeded".into());
            }
            selected.push(id);
        }
        selected
            .into_iter()
            .map(|id| {
                // One component: the source heightfield wedges are disjoint.
                WorldCell::new(
                    self.cells[id].vertices.map(|p| delta + motion.rotation * p),
                    None,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;

//! Shared render detail; simulation/collision grids remain authoritative.
use crate::app::GameCamera;
use bevy::prelude::*;
use mining_sim::site::{CELL, WIDTH};

pub(crate) const CHUNK: usize = 16;
pub(crate) const CHUNKS: usize = (WIDTH - 1).div_ceil(CHUNK);
pub(crate) const FINE_SUBDIVISIONS: usize = 4;
pub(crate) const MAX_FINE_CHUNKS: usize = 16;

#[derive(Resource, Default)]
pub(crate) struct Detail {
    mask: u64,
    anchor: Option<(usize, usize)>,
    pub revision: u64,
}
impl Detail {
    pub fn select(&mut self, position: Vec3) {
        if !position.is_finite() {
            return;
        }
        let coordinate = |v: f32| {
            ((v / (CELL * CHUNK as f32)).floor() as i32).clamp(0, CHUNKS as i32 - 1) as usize
        };
        // Avoid repeated mesh rebuilds when the camera jitters across a 4 m
        // boundary. Both terrain and water retain the same selected mask.
        if let Some((x, z)) = self.anchor {
            let stable = |value: f32, chunk: usize| {
                let start = chunk as f32 * CELL * CHUNK as f32;
                (start - 0.75..=start + CELL * CHUNK as f32 + 0.75).contains(&value)
            };
            if stable(position.x, x) && stable(position.z, z) {
                return;
            }
        }
        let anchor = (coordinate(position.x), coordinate(position.z));
        if self.anchor == Some(anchor) {
            return;
        }
        self.anchor = Some(anchor);
        let mut chunks: Vec<_> = (0..CHUNKS * CHUNKS).collect();
        chunks.sort_by_key(|&i| {
            let dx = (i % CHUNKS) as i32 - anchor.0 as i32;
            let dz = (i / CHUNKS) as i32 - anchor.1 as i32;
            (dx * dx + dz * dz, i)
        });
        let mask = chunks
            .into_iter()
            .take(MAX_FINE_CHUNKS)
            .fold(0, |bits, i| bits | (1u64 << i));
        if mask != self.mask {
            self.mask = mask;
            self.revision = self.revision.wrapping_add(1);
        }
    }
    pub fn subdivisions(&self, x: usize, z: usize) -> usize {
        let i = (z / CHUNK) * CHUNKS + x / CHUNK;
        if self.mask & (1u64 << i) != 0 {
            FINE_SUBDIVISIONS
        } else {
            1
        }
    }
}
pub(crate) fn update_detail(
    camera: Query<&Transform, With<GameCamera>>,
    mut detail: ResMut<Detail>,
) {
    if let Ok(camera) = camera.single() {
        detail.select(camera.translation);
    }
}
// Same corner order and bilinear rule as quad_value in water_depth.wgsl.
pub(crate) fn interpolate(v: [f32; 4], u: f32, w: f32) -> f32 {
    (v[0] * (1. - u) + v[1] * u) * (1. - w) + (v[2] * (1. - u) + v[3] * u) * w
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detail_has_a_fixed_budget_and_stays_cached_within_a_chunk() {
        let mut detail = Detail::default();
        detail.select(Vec3::new(14., 2., 18.));
        assert_eq!(detail.mask.count_ones() as usize, MAX_FINE_CHUNKS);
        let first = detail.revision;
        detail.select(Vec3::new(14.1, 2., 18.1));
        assert_eq!(detail.revision, first);
        detail.select(Vec3::new(16.1, 2., 18.1));
        detail.select(Vec3::new(15.9, 2., 18.1));
        assert_eq!(
            detail.revision, first,
            "boundary jitter must not rebuild meshes"
        );
        detail.select(Vec3::new(26., 2., 18.));
        assert_ne!(detail.revision, first);
        assert_eq!(detail.mask.count_ones() as usize, MAX_FINE_CHUNKS);
        for z in 0..WIDTH - 1 {
            for x in 0..WIDTH - 1 {
                assert_eq!(
                    detail.subdivisions(x, z),
                    detail.subdivisions(x / CHUNK * CHUNK, z / CHUNK * CHUNK)
                );
            }
        }
    }
}

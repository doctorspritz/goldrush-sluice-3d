//! A volume-derived display of the saved bulk hopper inventory.
//! This surface is not a dense-particle, compaction, bridging or collision model.
use bevy::{
    asset::RenderAssetUsages, math::DVec3, mesh::Indices, prelude::*,
    render::render_resource::PrimitiveTopology,
};
use mining_sim::sluice::{HOPPER_X, HOPPER_Z};

use crate::app::Game;

#[derive(Clone, Copy)]
pub(crate) struct Bowl {
    pub bottom: f64,
    pub top: f64,
    pub lower_half: f64,
    pub upper_half: f64,
    angle: f64,
}
impl Bowl {
    pub fn screen() -> Self {
        Self {
            bottom: 1.62,
            top: 2.10,
            lower_half: 0.175,
            upper_half: 0.8,
            angle: 0.,
        }
    }
    fn legacy() -> Self {
        Self {
            bottom: 0.73,
            top: 1.55,
            lower_half: 0.28 / 2_f64.sqrt(),
            upper_half: 1.06 / 2_f64.sqrt(),
            angle: std::f64::consts::FRAC_PI_4,
        }
    }
    fn capacity(self) -> f64 {
        let a = self.lower_half;
        let b = self.upper_half;
        4. * (self.top - self.bottom) / 3. * (a * a + a * b + b * b)
    }
    fn corner(self, i: usize, y: f64, half: f64) -> DVec3 {
        let (x, z) = [(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)][i % 4];
        let (s, c) = self.angle.sin_cos();
        DVec3::new((x * c - z * s) * half, y, (x * s + z * c) * half)
    }
    fn level(self, volume: f64) -> (f64, f64, f64) {
        let capacity = self.capacity();
        if volume >= capacity {
            // The inherited 0.85 m³ handling limit exceeds the bowl's volume.
            // Display remaining bulk as a square pyramid above the rim rather
            // than silently clipping stock or stretching a hidden sphere.
            let heap = 3. * (volume - capacity) / (4. * self.upper_half * self.upper_half);
            return (self.top, self.upper_half, heap);
        }
        let k = (self.upper_half - self.lower_half) / (self.top - self.bottom);
        let radius = (self.lower_half.powi(3) + 3. * k * volume / 4.).cbrt();
        (self.bottom + (radius - self.lower_half) / k, radius, 0.)
    }
}

#[derive(Component, Default)]
pub(crate) struct Fill {
    key: Option<(bool, u64)>,
}

/// Closed geometry permits an independent signed-volume check. The outer bowl
/// hides its buried faces; UVs use metres and the existing mip-filtered soil.
fn stock_mesh(bowl: Bowl, volume: f64) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    if volume.is_finite() && volume > 1e-12 {
        let (y, half, heap) = bowl.level(volume);
        let low = |i| bowl.corner(i, bowl.bottom, bowl.lower_half);
        let high = |i| bowl.corner(i, y, half);
        let mut triangle = |a: DVec3, b: DVec3, c: DVec3| {
            let normal = (b - a).cross(c - a).normalize().as_vec3().to_array();
            let base = positions.len() as u32;
            for p in [a, b, c] {
                positions.push(p.as_vec3().to_array());
                normals.push(normal);
                uvs.push([p.x as f32, p.z as f32]);
            }
            indices.extend([base, base + 1, base + 2]);
        };
        triangle(low(0), low(1), low(2));
        triangle(low(0), low(2), low(3));
        for i in 0..4 {
            triangle(low(i), high(i), high(i + 1));
            triangle(low(i), high(i + 1), low(i + 1));
            if heap > 0. {
                triangle(high(i), DVec3::new(0., y + heap, 0.), high(i + 1));
            }
        }
        if heap == 0. {
            triangle(high(0), high(3), high(2));
            triangle(high(0), high(2), high(1));
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    if mesh.count_vertices() > 0 {
        mesh.generate_tangents()
            .expect("tapered stock surface has metre UVs");
    }
    mesh
}

pub(crate) fn empty_mesh() -> Mesh {
    stock_mesh(Bowl::screen(), 0.)
}

pub(crate) fn sync(
    game: Res<Game>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut fills: Query<(&Mesh3d, &mut Fill, &mut Transform, &mut Visibility)>,
) {
    let session = &game.session;
    let screen = session.plant.is_some();
    let volume = session.sluice.hopper.handling_volume_m3();
    let key = (screen, volume.to_bits());
    for (handle, mut fill, mut transform, mut visible) in &mut fills {
        if fill.key != Some(key) {
            if let Some(mesh) = meshes.get_mut(&handle.0) {
                *mesh = stock_mesh(
                    if screen {
                        Bowl::screen()
                    } else {
                        Bowl::legacy()
                    },
                    volume,
                );
                fill.key = Some(key);
            }
        }
        *visible = if volume > 1e-12 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        *transform =
            Transform::from_xyz(HOPPER_X, session.site.height(HOPPER_X, HOPPER_Z), HOPPER_Z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    fn measured_volume(mesh: &Mesh) -> f64 {
        let VertexAttributeValues::Float32x3(p) = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap()
        else {
            panic!("positions")
        };
        mesh.indices()
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
            .chunks_exact(3)
            .map(|i| {
                let point = |j: usize| DVec3::from_array(p[i[j]].map(f64::from));
                point(0).dot(point(1).cross(point(2))) / 6.
            })
            .sum()
    }
    #[test]
    fn finite_inventory_mesh_encloses_actual_bulk_through_bowl_and_above_rim() {
        for bowl in [Bowl::screen(), Bowl::legacy()] {
            for volume in [0.000001, 0.005, 0.059, bowl.capacity(), 0.85] {
                let mesh = stock_mesh(bowl, volume);
                let measured = measured_volume(&mesh);
                assert!(
                    (measured - volume).abs() < 2e-7,
                    "volume {volume} measured {measured}"
                );
                let VertexAttributeValues::Float32x3(normals) =
                    mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap()
                else {
                    panic!("normals")
                };
                assert!(normals.iter().flatten().all(|n| n.is_finite()));
                assert!(mesh.attribute(Mesh::ATTRIBUTE_TANGENT).is_some());
            }
        }
    }
    #[test]
    fn empty_stock_is_hidden_geometry_and_lower_stock_has_lower_level() {
        for bowl in [Bowl::screen(), Bowl::legacy()] {
            assert_eq!(stock_mesh(bowl, 0.).count_vertices(), 0);
            let low = bowl.level(0.005);
            let high = bowl.level(0.059);
            assert!(low.0 < high.0 && low.1 < high.1 && high.0 < bowl.top);
            assert!(
                bowl.level(0.85).2 > 0.,
                "full handling inventory includes above-rim heap"
            );
        }
    }
}

//! Physical-size grain proxies sampled inside each volume-resolved DEM parcel.
//! One mesh per material keeps thousands of grains from becoming thousands of
//! entities/draw calls. Display sampling never changes authoritative material.
use bevy::{
    asset::RenderAssetUsages, camera::visibility::NoFrustumCulling, mesh::Indices, prelude::*,
    render::render_resource::PrimitiveTopology,
};
use mining_sim::{GrainKind, Sluice};
const MAX_POINTS: usize = 32_768;
struct Geometry {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
}
fn geometry(s: &Sluice) -> [Geometry; 3] {
    let mut out = std::array::from_fn(|_| Geometry {
        positions: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
    });
    let Some(placement) = s.placement else {
        return out;
    };
    let allowance = (MAX_POINTS / s.grains.len().max(1)).clamp(1, 12);
    let axes = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    let triangles = [
        [0, 2, 4],
        [4, 2, 1],
        [1, 2, 5],
        [5, 2, 0],
        [0, 4, 3],
        [4, 1, 3],
        [1, 5, 3],
        [5, 0, 3],
    ];
    for (i, g) in s.grains.iter().enumerate() {
        let kind = match g.kind {
            GrainKind::Sand => 0,
            GrainKind::Gravel => 1,
            GrainKind::Gold => 2,
        };
        let physical_volume = std::f64::consts::PI * (g.diameter_m as f64).powi(3) / 6.0;
        let count = if g.volume_resolved {
            (g.solid_volume() / physical_volume)
                .ceil()
                .clamp(1.0, allowance as f64) as usize
        } else {
            1
        };
        let cloud = (g.collision_radius() - g.diameter_m * 0.5).max(0.0);
        for j in 0..count {
            let z = 1.0 - 2.0 * (j as f32 + 0.5) / count as f32;
            let angle = j as f32 * 2.3999631 + i as f32 * 1.618034;
            let r = (1.0 - z * z).sqrt();
            let fraction = ((j as f32 + 0.5) * 0.7548777).fract().cbrt();
            let offset = g.rotation
                * mining_sim::Vec3::new(r * angle.cos(), z, r * angle.sin())
                * cloud
                * fraction;
            let position = placement.to_world(s.grid_to_local(g.position + offset));
            let mesh = &mut out[kind];
            let base = mesh.positions.len() as u32;
            for axis in axes {
                let shape = if kind == 2 {
                    Vec3::new(axis.x, axis.y * 0.25, axis.z)
                } else {
                    axis
                };
                let local = g.rotation
                    * mining_sim::Vec3::from_array(shape.to_array())
                    * (g.diameter_m * 0.5);
                let point = position
                    + placement.forward() * local.x
                    + mining_sim::Vec3::Y * local.y
                    + placement.right() * local.z;
                mesh.positions.push(point.to_array());
                let normal = g.rotation * mining_sim::Vec3::from_array(axis.to_array());
                mesh.normals.push(
                    (placement.forward() * normal.x
                        + mining_sim::Vec3::Y * normal.y
                        + placement.right() * normal.z)
                        .to_array(),
                );
            }
            // Placement's downstream/up/right frame reflects handedness.
            for t in triangles {
                mesh.indices.extend([base + t[0], base + t[2], base + t[1]]);
            }
        }
    }
    out
}
pub fn sync(
    mut commands: Commands,
    game: Res<crate::app::Game>,
    palette: Res<crate::scene::Palette>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut handles: Local<Option<[Handle<Mesh>; 3]>>,
    mut revision: Local<Option<crate::app::FluidRevision>>,
) {
    let key = game.fluid_revision();
    if handles.is_some() && *revision == Some(key) {
        return;
    }
    if handles.is_none() {
        let materials = [
            palette.dirt.clone(),
            palette.sand.clone(),
            palette.gold.clone(),
        ];
        let created = std::array::from_fn(|i| {
            let handle = meshes.add(Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            ));
            commands.spawn((
                Mesh3d(handle.clone()),
                MeshMaterial3d(materials[i].clone()),
                Transform::default(),
                NoFrustumCulling,
            ));
            handle
        });
        *handles = Some(created);
    }
    *revision = Some(key);
    for (handle, data) in handles
        .as_ref()
        .unwrap()
        .iter()
        .zip(geometry(&game.session.sluice))
    {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals);
        mesh.insert_indices(Indices::U32(data.indices));
        *meshes.get_mut(handle).unwrap() = mesh;
    }
}

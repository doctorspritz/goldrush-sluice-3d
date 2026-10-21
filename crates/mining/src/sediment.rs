//! Physical-size grain proxies sampled inside each volume-resolved DEM parcel.
//! One mesh per material keeps thousands of grains from becoming thousands of
//! entities/draw calls. Display sampling never changes authoritative material.
use bevy::{
    asset::RenderAssetUsages, camera::visibility::NoFrustumCulling, mesh::Indices, prelude::*,
    render::render_resource::PrimitiveTopology,
};
use mining_sim::{GrainKind, Sluice};
pub(crate) const MAX_POINTS: usize = 32_768;
fn sample_counts(s: &Sluice) -> Vec<usize> {
    let wants: Vec<_> = s
        .grains
        .iter()
        .take(MAX_POINTS)
        .map(|g| {
            // Subpixel fines are mainly represented by the local suspension field.
            // Spend geometry on coarse grains that can actually cover the bed.
            let cap = if g.diameter_m < 0.0002 { 8. } else { 512. };
            let population = g.represented_grains();
            if population <= 1.00001 {
                1
            } else {
                population.ceil().clamp(1., cap) as usize
            }
        })
        .collect();
    let free = MAX_POINTS.saturating_sub(wants.len());
    let extra: usize = wants.iter().map(|n| n - 1).sum();
    wants
        .into_iter()
        .map(|n| 1 + (n - 1) * free / extra.max(free).max(1))
        .collect()
}
pub(crate) fn visible_grains(s: &Sluice) -> usize {
    sample_counts(s).iter().sum()
}
struct Geometry {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
}
fn geometry(s: &Sluice) -> [Geometry; 3] {
    let counts = sample_counts(s);
    let kind_index = |kind| match kind {
        GrainKind::Sand => 0,
        GrainKind::Gravel => 1,
        GrainKind::Gold => 2,
    };
    let mut per_kind = [0; 3];
    for (g, count) in s.grains.iter().zip(&counts) {
        per_kind[kind_index(g.kind)] += count;
    }
    let mut out = std::array::from_fn(|i| Geometry {
        positions: Vec::with_capacity(per_kind[i] * 6),
        normals: Vec::with_capacity(per_kind[i] * 6),
        indices: Vec::with_capacity(per_kind[i] * 24),
    });
    let Some(placement) = s.placement else {
        return out;
    };
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
    for (i, (g, count)) in s.grains.iter().zip(counts).enumerate() {
        let kind = kind_index(g.kind);
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
                bevy::light::NotShadowCaster,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_sampling_respects_global_budget_and_keeps_individual_gold() {
        let mut s = crate::demo::session().unwrap();
        let gold = s
            .sluice
            .grains
            .iter()
            .find(|g| g.kind == GrainKind::Gold)
            .unwrap()
            .clone();
        let mut gravel = s
            .sluice
            .grains
            .iter()
            .find(|g| g.kind == GrainKind::Gravel)
            .unwrap()
            .clone();
        gravel.diameter_m = 0.0003;
        s.sluice.grains = vec![gravel; mining_sim::sluice::MAX_GRAINS - 1];
        s.sluice.grains.push(gold);
        let samples = sample_counts(&s.sluice);
        let count: usize = samples.iter().sum();
        assert_eq!(*samples.last().unwrap(), 1);
        assert!(count > MAX_POINTS - mining_sim::sluice::MAX_GRAINS && count <= MAX_POINTS);
        let meshes = geometry(&s.sluice);
        assert_eq!(meshes[2].positions.len(), 6);
        assert_eq!(
            meshes.iter().map(|m| m.positions.len()).sum::<usize>(),
            count * 6
        );
        assert!(meshes
            .iter()
            .all(|m| m.indices.iter().all(|i| (*i as usize) < m.positions.len())));
    }
}

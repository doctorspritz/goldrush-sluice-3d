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
    let wet = mining_physics::geometry::grain_wetness(s);
    let wants: Vec<_> = s
        .grains
        .iter()
        .take(MAX_POINTS)
        .map(|g| {
            let cap = if g.diameter_m < 0.0002 { 8. } else { 512. };
            let population = g.represented_grains();
            if population <= 1.00001 {
                1
            } else {
                population.ceil().clamp(1., cap) as usize
            }
        })
        .collect();
    // Reserve the original shares before suppressing fine geometry. Reusing
    // their quota would turn neighbouring gravel into denser opaque clusters.
    let free = MAX_POINTS.saturating_sub(wants.len());
    let extra: usize = wants.iter().map(|n| n - 1).sum();
    wants
        .into_iter()
        .zip(&s.grains)
        .enumerate()
        .map(|(i, (n, g))| {
            // The physical adapter's nearby-water indicator also selects
            // channel optics. Keep moving sand without that carrier visible;
            // wet moving sand has one optical representation. This coarse
            // indicator does not establish exact particle submergence.
            if g.kind == GrainKind::Sand && !g.resting && wet[i] {
                0
            } else {
                1 + (n - 1) * free / extra.max(free).max(1)
            }
        })
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
    for (g, count) in s.grains.iter().zip(counts) {
        let kind = kind_index(g.kind);
        let cloud = (g.collision_radius() - g.diameter_m * 0.5).max(0.0);
        for j in 0..count {
            // A nested sequence keeps existing proxies fixed when the global
            // sampling budget changes or an earlier DEM parcel leaves the box.
            // Array index and total sample count are not particle identities.
            let z = 1.0 - 2.0 * ((j as f32 + 0.5) * 0.5698403).fract();
            let angle = j as f32 * 2.3999631;
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
    capture: Res<crate::motion_capture::MotionCapture>,
    palette: Res<crate::scene::Palette>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut handles: Local<Option<[Handle<Mesh>; 3]>>,
    mut revision: Local<Option<crate::app::FluidRevision>>,
) {
    if capture.water_only {
        return;
    }
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

    fn carrier_at(s: &mut Sluice, position: mining_sim::Vec3) {
        let mut water = s.water[0].clone();
        water.position = position;
        s.water = vec![water];
    }

    #[test]
    fn suspended_fines_use_optics_without_changing_material_or_hiding_gold_and_bed() {
        let mut s = crate::demo::session().unwrap().sluice;
        let mut fine = s
            .grains
            .iter()
            .find(|g| g.kind == GrainKind::Sand && !g.resting)
            .unwrap()
            .clone();
        fine.position = mining_sim::Vec3::new(0.6, 0.3, 0.2);
        carrier_at(&mut s, fine.position);
        let gold = s
            .grains
            .iter()
            .find(|g| g.kind == GrainKind::Gold)
            .unwrap()
            .clone();
        let mut deposited = fine.clone();
        deposited.resting = true;
        s.grains = vec![fine, gold, deposited];
        let before = serde_json::to_string(&s).unwrap();
        let field = mining_sim::slurry::SlurryField::new(&s);
        let projected = field.projected_volumes();
        let counts = sample_counts(&s);
        assert_eq!(counts[0], 0);
        assert_eq!(counts[1], 1);
        assert!(counts[2] > 0);
        assert!((projected.1 - s.grains[0].solid_volume()).abs() < 1e-15);
        assert_eq!(geometry(&s)[2].positions.len(), 6);
        assert_eq!(before, serde_json::to_string(&s).unwrap());
    }
    #[test]
    fn coarse_suspended_sand_has_one_optical_representation() {
        let mut s = crate::demo::session().unwrap().sluice;
        let mut sand = s
            .grains
            .iter()
            .find(|g| g.kind == GrainKind::Sand)
            .unwrap()
            .clone();
        sand.resting = false;
        sand.diameter_m = 0.002;
        sand.position = mining_sim::Vec3::new(0.6, 0.3, 0.2);
        carrier_at(&mut s, sand.position);
        s.grains = vec![sand];
        let before = serde_json::to_string(&s).unwrap();
        let field = mining_sim::slurry::SlurryField::new(&s);
        assert!((field.projected_volumes().1 - s.grains[0].solid_volume()).abs() < 1e-15);
        assert_eq!(sample_counts(&s), vec![0]);
        assert!(geometry(&s)[0].positions.is_empty());
        assert_eq!(before, serde_json::to_string(&s).unwrap());
        s.grains[0].resting = true;
        assert!(sample_counts(&s)[0] > 0);
    }

    #[test]
    fn local_water_selects_moving_sand_without_reallocating_dry_bed_or_gold_proxies() {
        let mut s = crate::demo::session().unwrap().sluice;
        let mut wet_sand = s
            .grains
            .iter()
            .find(|g| g.kind == GrainKind::Sand && g.diameter_m < 0.0002)
            .unwrap()
            .clone();
        wet_sand.position = mining_sim::Vec3::new(0.6, 0.3, 0.2);
        wet_sand.resting = false;
        let mut dry_sand = wet_sand.clone();
        dry_sand.position.y += 0.1;
        let mut bed = wet_sand.clone();
        bed.resting = true;
        let mut gold = s
            .grains
            .iter()
            .find(|g| g.kind == GrainKind::Gold)
            .unwrap()
            .clone();
        gold.position = wet_sand.position;
        carrier_at(&mut s, wet_sand.position);
        s.grains = vec![wet_sand, dry_sand, bed, gold];
        assert_eq!(
            mining_physics::geometry::grain_wetness(&s),
            vec![true, false, true, true]
        );
        let before = serde_json::to_string(&s).unwrap();
        let counts = sample_counts(&s);
        let meshes = geometry(&s);
        assert_eq!(counts[0], 0, "wet moving sand uses channel optics");
        assert!(counts[1] > 0, "dry moving sand remains visible");
        assert!(counts[2] > 0, "wet resting bed remains visible");
        assert_eq!(counts[3], 1, "wet individual gold remains visible");
        assert_eq!(meshes[0].positions.len(), (counts[1] + counts[2]) * 6);
        assert_eq!(meshes[2].positions.len(), 6);
        assert_eq!(before, serde_json::to_string(&s).unwrap());

        // Loss of local carrier restores the formerly wet parcel's physical
        // proxies without changing anyone else's reserved quota or seeds.
        s.water.clear();
        let dry_counts = sample_counts(&s);
        let dry_meshes = geometry(&s);
        assert!(dry_counts[0] > 0);
        assert_eq!(dry_counts[1..], counts[1..]);
        assert_eq!(
            dry_meshes[0].positions[dry_counts[0] * 6..],
            meshes[0].positions
        );
        assert_eq!(dry_meshes[2].positions, meshes[2].positions);
    }
    #[test]
    fn removing_a_parcel_does_not_reseed_surviving_proxy_clouds() {
        let mut s = crate::demo::session().unwrap().sluice;
        let gravel = s
            .grains
            .iter()
            .find(|g| g.kind == GrainKind::Gravel)
            .unwrap()
            .clone();
        let gold = s
            .grains
            .iter()
            .find(|g| g.kind == GrainKind::Gold)
            .unwrap()
            .clone();
        s.grains = vec![gold, gravel];
        let before = geometry(&s)[1].positions.clone();
        s.grains.remove(0);
        assert_eq!(before, geometry(&s)[1].positions);
    }
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

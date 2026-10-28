//! Finite mean-bed envelope. Geometry uses the same nonnegative stock field as
//! the opt-in solver; detail changes normals/colour, never its height or volume.
use crate::sediment::Geometry;
use bevy::prelude::*;
use mining_sim::screen::{BulkBedField, Screen};

pub(crate) struct BedMesh {
    pub geometry: Geometry,
    pub colours: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub tangents: Vec<[f32; 4]>,
}
#[derive(Clone, Copy)]
struct Vertex {
    position: Vec3,
    normal: Vec3,
    colour: [f32; 4],
}
impl BedMesh {
    fn triangle(&mut self, vertices: [Vertex; 3]) {
        let base = self.geometry.positions.len() as u32;
        for v in vertices {
            // A right-handed render frame uses -across for its third axis.
            self.geometry
                .positions
                .push([v.position.x, v.position.y, -v.position.z]);
            self.geometry
                .normals
                .push([v.normal.x, v.normal.y, -v.normal.z]);
            self.colours.push(v.colour);
            let p = Vec3::new(v.position.x, v.position.y, -v.position.z);
            let n = Vec3::new(v.normal.x, v.normal.y, -v.normal.z);
            let (u, w) = if n.y.abs() > 0.5 {
                (Vec3::X, -Vec3::Z)
            } else if n.x.abs() > 0.5 {
                (-Vec3::Z, Vec3::Y)
            } else {
                (Vec3::X, Vec3::Y)
            };
            // Reuse existing mip-filtered soil assets at quarter-metre repeat.
            // Millimetre-scale lighting detail adds no geometric/owned volume.
            self.uvs.push([p.dot(u) * 4., p.dot(w) * 4.]);
            let tangent = (u - n * n.dot(u)).normalize();
            self.tangents
                .push(tangent.extend(n.cross(tangent).dot(w).signum()).to_array());
        }
        self.geometry.indices.extend([base, base + 2, base + 1]);
    }
    fn quad(&mut self, v: [Vertex; 4]) {
        self.triangle([v[0], v[1], v[2]]);
        self.triangle([v[0], v[2], v[3]]);
    }
}
pub(crate) fn transform(screen: &Screen, time: f64) -> Transform {
    let yaw = screen.yaw;
    let slope = screen.config.slope_deg.to_radians() as f32;
    let down = Vec3::new(yaw.sin(), 0., yaw.cos());
    let across = Vec3::new(yaw.cos(), 0., -yaw.sin());
    let tangent = down * slope.cos() - Vec3::Y * slope.sin();
    let normal = down * slope.sin() + Vec3::Y * slope.cos();
    let origin = Vec3::from_array(screen.origin.to_array())
        + Vec3::from_array(screen.offset_at(time).0.as_vec3().to_array());
    Transform::from_matrix(Mat4::from_cols(
        tangent.extend(0.),
        normal.extend(0.),
        (-across).extend(0.),
        origin.extend(1.),
    ))
}
fn vertex(field: &BulkBedField, x: f64, z: f64, fallback: mining_sim::Material) -> Vertex {
    let s = field.sample(x, z);
    let m = if s.material.mass_kg() > 1e-30 {
        s.material
    } else {
        fallback
    };
    let mineral = m.mineral_solid_volume_m3();
    let gold = m.gold_g * 0.001 / mining_sim::material::GOLD_DENSITY_KG_M3;
    let total = (mineral + gold).max(1e-30);
    let bulk = m.volume().max(1e-30);
    let soil = m.soil_m3 / bulk;
    let sand = m.sand_m3 / bulk;
    let gravel = m.gravel_m3 / bulk;
    let dry = (Vec3::new(0.31, 0.18, 0.085) * soil as f32
        + Vec3::new(0.54, 0.40, 0.23) * sand as f32
        + Vec3::new(0.39, 0.35, 0.29) * gravel as f32)
        * (mineral / total) as f32
        + Vec3::new(0.58, 0.34, 0.06) * (gold / total) as f32;
    let saturation = m.saturation() as f32;
    let mut bits = (x.to_bits() >> 20) ^ (z.to_bits().rotate_left(21));
    bits ^= bits >> 30;
    bits = bits.wrapping_mul(0xbf58476d1ce4e5b9);
    bits ^= bits >> 27;
    let grit = ((bits >> 40) as f32 / (1_u32 << 24) as f32 - 0.5) * 0.14;
    let linear = Color::srgb(dry.x, dry.y, dry.z).to_linear();
    // Apply absorption to linear reflectance, once. Applying wet darkening
    // before sRGB conversion crushed shadowed soil into an unreadable patch.
    let colour =
        (Vec3::new(linear.red, linear.green, linear.blue) * (1. - 0.4 * saturation) * (1. + grit))
            .clamp(Vec3::ZERO, Vec3::ONE);
    let normal = Vec3::new(
        (-s.gradient.x * field.slope_cosine()) as f32 + grit * 0.18,
        1.,
        -s.gradient.y as f32 - grit * 0.12,
    )
    .normalize();
    Vertex {
        position: Vec3::new(
            (x / field.slope_cosine()) as f32,
            (2. * field.wire_radius_m() + s.depth_m) as f32,
            z as f32,
        ),
        normal,
        colour: colour.extend(1.).to_array(),
    }
}
pub(crate) fn roughness(field: Option<&BulkBedField>) -> f32 {
    let saturation = field.map_or(0., |f| f.material().saturation()) as f32;
    0.94 + (0.58 - 0.94) * saturation
}
pub(crate) fn geometry(field: Option<&BulkBedField>) -> BedMesh {
    let mut out = BedMesh {
        geometry: Geometry {
            positions: vec![],
            normals: vec![],
            indices: vec![],
        },
        colours: vec![],
        uvs: vec![],
        tangents: vec![],
    };
    let Some(field) = field else {
        return out;
    };
    if field.integrated_volume_m3() <= 0. {
        return out;
    }
    let fallback = field.material();
    let (nx, nz) = field.dimensions();
    let subdivisions = ((4096. / ((nx + 1) * (nz + 1)) as f64).sqrt().floor() as usize).clamp(1, 4);
    let (x, z) = field.mesh_axes(subdivisions);
    let base = (2. * field.wire_radius_m()) as f32;
    for a in x.windows(2) {
        for b in z.windows(2) {
            let v = [
                vertex(field, a[0], b[0], fallback),
                vertex(field, a[0], b[1], fallback),
                vertex(field, a[1], b[1], fallback),
                vertex(field, a[1], b[0], fallback),
            ];
            if v.iter().all(|v| v.position.y <= base) {
                continue;
            }
            let centre = vertex(field, (a[0] + a[1]) * 0.5, (b[0] + b[1]) * 0.5, fallback);
            for k in 0..4 {
                out.triangle([v[k], v[(k + 1) % 4], centre]);
            }
            let bottom = v.map(|mut v| {
                v.position.y = base;
                v.normal = -Vec3::Y;
                v
            });
            out.quad([bottom[0], bottom[3], bottom[2], bottom[1]]);
        }
    }
    let mut wall = |a: (f64, f64), b: (f64, f64), normal: Vec3| {
        let mut v0 = vertex(field, a.0, a.1, fallback);
        let mut v1 = vertex(field, b.0, b.1, fallback);
        if v0.position.y <= base && v1.position.y <= base {
            return;
        }
        v0.normal = normal;
        v1.normal = normal;
        let mut b0 = v0;
        let mut b1 = v1;
        b0.position.y = base;
        b1.position.y = base;
        out.quad([b0, v0, v1, b1]);
    };
    let l = field.length_m();
    let h = field.width_m() * 0.5;
    for b in z.windows(2) {
        wall((0., b[1]), (0., b[0]), -Vec3::X);
        wall((l, b[0]), (l, b[1]), Vec3::X);
    }
    for a in x.windows(2) {
        wall((a[0], -h), (a[1], -h), -Vec3::Z);
        wall((a[1], h), (a[0], h), Vec3::Z);
    }
    out
}

#[cfg(test)]
pub(crate) fn loaded_screen(wet: bool) -> Screen {
    use mining_sim::{screen::BulkDeck, sluice::Grain, GrainKind, Material};
    let mut s = Screen::new(mining_sim::Vec3::new(17., 2., 18.), 0.7);
    s.set_bulk_deck(Some(BulkDeck::default())).unwrap();
    let dry = Material {
        soil_m3: 8e-6,
        gold_g: 1e-8,
        ..Default::default()
    };
    let m = if wet {
        Material {
            pore_water_m3: dry.pore_volume_m3() * 0.7,
            ..dry
        }
    } else {
        dry
    };
    let grains: Vec<_> = (0..128)
        .map(|i| {
            let x = 0.05 + (i % 16) as f64 * 0.04;
            let z = -0.14 + (i / 16 % 8) as f64 * 0.04;
            Grain {
                kind: GrainKind::Sand,
                material: m,
                diameter_m: 0.0005,
                position: s
                    .to_world([x, 0.03 - x * s.config.slope_deg.to_radians().tan(), z].into())
                    .as_vec3(),
                velocity: Default::default(),
                affine: Default::default(),
                rotation: Default::default(),
                angular_velocity: Default::default(),
                resting: false,
                volume_resolved: true,
            }
        })
        .collect();
    s.admit(&grains, 0.).unwrap();
    s.motor_on = true;
    for _ in 0..15 {
        s.step(1. / 60.).unwrap();
    }
    assert!(s.valid());
    s
}
#[cfg(test)]
mod tests {
    use super::*;
    fn volume(mesh: &BedMesh) -> f64 {
        mesh.geometry
            .indices
            .chunks_exact(3)
            .map(|tri| {
                let a = Vec3::from_array(mesh.geometry.positions[tri[0] as usize]).as_dvec3();
                let b = Vec3::from_array(mesh.geometry.positions[tri[1] as usize]).as_dvec3();
                let c = Vec3::from_array(mesh.geometry.positions[tri[2] as usize]).as_dvec3();
                a.dot(b.cross(c)) / 6.
            })
            .sum()
    }
    #[test]
    fn closed_mean_envelope_volume_matches_actual_supported_stock() {
        let s = loaded_screen(true);
        let f = s.bulk_bed_field().unwrap();
        let mesh = geometry(Some(&f));
        assert!(f.integrated_volume_m3() > 0.0005);
        let measured = volume(&mesh);
        let expected = f.integrated_volume_m3();
        assert!(
            (measured / expected - 1.).abs() < 2e-5,
            "mesh{measured} vs field{expected}"
        );
        assert_eq!(mesh.geometry.positions.len(), mesh.colours.len());
        assert_eq!(mesh.geometry.positions.len(), mesh.uvs.len());
        assert_eq!(mesh.geometry.positions.len(), mesh.tangents.len());
        for (n, tangent) in mesh.geometry.normals.iter().zip(&mesh.tangents) {
            let t = Vec3::from_array([tangent[0], tangent[1], tangent[2]]);
            assert!((t.length() - 1.).abs() < 1e-5);
            assert!(t.dot(Vec3::from_array(*n)).abs() < 1e-5);
            assert_eq!(tangent[3].abs(), 1.);
        }
        assert!(mesh.uvs.iter().flatten().all(|v| v.is_finite()));
        assert!(mesh.geometry.indices.len() / 3 < 30000);
        assert!(mesh
            .geometry
            .normals
            .iter()
            .all(|n| (Vec3::from_array(*n).length() - 1.).abs() < 1e-5));
        assert!(mesh
            .colours
            .iter()
            .flatten()
            .all(|v| (0.0..=1.).contains(v)));
        println!(
            "BED_MESH volume={measured} expected={expected} triangles={}",
            mesh.geometry.indices.len() / 3
        );
    }
    #[test]
    fn right_handed_local_mesh_follows_actual_deck_frame_and_recorded_oscillator() {
        let s = loaded_screen(true);
        let f = s.bulk_bed_field().unwrap();
        let x = 0.23;
        let z = -0.09;
        let v = vertex(&f, x, z, f.material());
        let t = transform(&s, s.time);
        let actual = t.transform_point(Vec3::new(v.position.x, v.position.y, -v.position.z));
        let down = s.to_world([1., 0., 0.].into()) - s.origin.as_dvec3();
        let angle = s.config.slope_deg.to_radians();
        let up = s.to_world([0., 1., 0.].into()) - s.origin.as_dvec3();
        let normal = down * angle.sin() + up * angle.cos();
        let expected = s.to_world([x, -x * angle.tan(), z].into())
            + normal * (2. * s.config.wire_radius_m + f.sample(x, z).depth_m)
            + s.offset_at(s.time).0;
        assert!(
            mining_sim::Vec3::from_array(actual.to_array())
                .as_dvec3()
                .distance(expected)
                < 4e-6
        );
        assert!((t.scale - Vec3::ONE).length() < 1e-5);
    }
    #[test]
    fn moisture_darkens_the_same_material_without_minting_depth_or_stock() {
        let wet = loaded_screen(true);
        let dry = loaded_screen(false);
        let a = wet.bulk_bed_field().unwrap();
        let b = dry.bulk_bed_field().unwrap();
        let ca = vertex(&a, 0.23, 0., a.material()).colour;
        let cb = vertex(&b, 0.23, 0., b.material()).colour;
        assert!(ca[..3].iter().sum::<f32>() < cb[..3].iter().sum::<f32>());
        let before = serde_json::to_string(&wet).unwrap();
        let _ = geometry(Some(&a));
        assert_eq!(serde_json::to_string(&wet).unwrap(), before);
    }
    #[test]
    fn cleaned_or_reference_decks_have_no_phantom_bulk_geometry() {
        let mut s = loaded_screen(true);
        s.motor_on = false;
        let out = s.cleanout().unwrap();
        assert!(!out.is_empty());
        let f = s.bulk_bed_field().unwrap();
        assert_eq!(f.integrated_volume_m3(), 0.);
        assert!(geometry(Some(&f)).geometry.indices.is_empty());
        assert!(geometry(None).geometry.indices.is_empty());
        assert!(s.valid());
    }
    #[test]
    fn wet_reflectance_keeps_a_finite_dielectric_roughness() {
        let wet = loaded_screen(true).bulk_bed_field().unwrap();
        let dry = loaded_screen(false).bulk_bed_field().unwrap();
        assert!(roughness(Some(&wet)) >= 0.58 && roughness(Some(&wet)) < roughness(Some(&dry)));
        assert_eq!(roughness(None), 0.94);
    }
}

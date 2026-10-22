//! Bounded, deterministic conifer art. Crossed needle branches share one cutout
//! texture; the same apertures are used by the opaque and shadow passes.
use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

const WIDTH: usize = 512;
const HEIGHT: usize = 256;
const MIPS: u32 = 8; // Stop at 4x2: one binary texel cannot retain branch coverage.

fn hash(a: u32, b: u32) -> f32 {
    let n = a
        .wrapping_mul(374761393)
        .wrapping_add(b.wrapping_mul(668265263));
    let n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n ^ (n >> 16)) as f32 / u32::MAX as f32
}

fn colour(r: f32, g: f32, b: f32) -> Vec3 {
    Vec4::from_array(Color::srgb(r, g, b).to_linear().to_f32_array()).truncate()
}

fn stroke(pixels: &mut [Vec4], a: Vec2, b: Vec2, radius: f32, rgb: Vec3) {
    let size = Vec2::new(WIDTH as f32, HEIGHT as f32);
    let a = a * size;
    let b = b * size;
    let lo = (a.min(b) - Vec2::splat(radius + 1.))
        .floor()
        .max(Vec2::ZERO);
    let hi = (a.max(b) + Vec2::splat(radius + 1.))
        .ceil()
        .min(size - Vec2::ONE);
    let delta = b - a;
    for y in lo.y as usize..=hi.y as usize {
        for x in lo.x as usize..=hi.x as usize {
            let q = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - a;
            let t = (q.dot(delta) / delta.length_squared().max(1e-8)).clamp(0., 1.);
            let alpha = (radius + 0.5 - (q - delta * t).length()).clamp(0., 1.);
            let pixel = &mut pixels[y * WIDTH + x];
            if alpha > pixel.w {
                *pixel = rgb.extend(alpha);
            }
        }
    }
}

fn branch_pixels() -> Vec<Vec4> {
    let mut pixels = vec![colour(0.13, 0.25, 0.12).extend(0.); WIDTH * HEIGHT];
    let stem = colour(0.22, 0.16, 0.075);
    stroke(
        &mut pixels,
        Vec2::new(0.01, 0.5),
        Vec2::new(0.985, 0.5),
        1.15,
        stem,
    );
    for side in 0..2 {
        let sign = if side == 0 { -1. } else { 1. };
        for i in 0..25 {
            let u = 0.045 + i as f32 * 0.035 + hash(i, side + 13) * 0.012;
            let spread = (1. - u).powf(0.65) * (0.32 + hash(i, side + 37) * 0.10);
            let base = Vec2::new(u, 0.5);
            let tip = Vec2::new(
                (u + 0.09 + hash(i, 29) * 0.08).min(0.98),
                0.5 + sign * spread,
            );
            stroke(&mut pixels, base, tip, 0.65, stem);
            let axis = (tip - base).normalize();
            let across = Vec2::new(-axis.y, axis.x);
            for j in 1..=12 {
                let t = j as f32 / 13.;
                let anchor = base.lerp(tip, t);
                for leaf_side in [-1., 1.] {
                    let n = i * 59 + j * 3 + side;
                    let length = 0.024 + hash(n, 79) * 0.026;
                    let end = anchor + axis * (length * 0.65) + across * (length * leaf_side);
                    let tint = 0.78 + hash(n, 97) * 0.37;
                    stroke(
                        &mut pixels,
                        anchor,
                        end,
                        0.48 + hash(n, 41) * 0.18,
                        colour(0.17 * tint, 0.30 * tint, 0.13 * tint),
                    );
                }
            }
        }
    }
    pixels
}

fn coverage(pixels: &[Vec4], scale: f32) -> f32 {
    pixels.iter().filter(|p| p.w * scale >= 0.5).count() as f32 / pixels.len() as f32
}

/// Preserve the masked silhouette while reducing needles below a texel. Plain
/// alpha averaging erases a branch from both the scene and its shadow mip.
fn preserve_coverage(pixels: &mut [Vec4], target: f32) {
    let (mut lo, mut hi) = (0., 16.);
    let (mut best, mut error) = (1., (coverage(pixels, 1.) - target).abs());
    for _ in 0..16 {
        let scale = (lo + hi) * 0.5;
        let value = coverage(pixels, scale);
        if (value - target).abs() < error {
            best = scale;
            error = (value - target).abs();
        }
        if value < target {
            lo = scale;
        } else {
            hi = scale;
        }
    }
    for pixel in pixels {
        pixel.w = (pixel.w * best).clamp(0., 1.);
    }
}

pub(crate) fn material(images: &mut Assets<Image>) -> StandardMaterial {
    let mut pixels = branch_pixels();
    let target = coverage(&pixels, 1.);
    let (mut width, mut height) = (WIDTH, HEIGHT);
    let mut bytes = Vec::new();
    for level in 0..MIPS {
        if level > 0 {
            preserve_coverage(&mut pixels, target);
        }
        for p in &pixels {
            for linear in p.truncate().to_array() {
                let srgb = if linear <= 0.003_130_8 {
                    linear * 12.92
                } else {
                    1.055 * linear.powf(1. / 2.4) - 0.055
                };
                bytes.push((srgb.clamp(0., 1.) * 255.).round() as u8);
            }
            bytes.push((p.w * 255.).round() as u8);
        }
        if level + 1 == MIPS {
            break;
        }
        let mut next = Vec::with_capacity(width * height / 4);
        for y in 0..height / 2 {
            for x in 0..width / 2 {
                let i = y * 2 * width + x * 2;
                next.push(
                    (pixels[i] + pixels[i + 1] + pixels[i + width] + pixels[i + width + 1]) * 0.25,
                );
            }
        }
        pixels = next;
        width /= 2;
        height /= 2;
    }
    let mut image = Image::new_uninit(
        Extent3d {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = MIPS;
    image.data = Some(bytes);
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::ClampToEdge,
        address_mode_v: ImageAddressMode::ClampToEdge,
        anisotropy_clamp: 4,
        ..ImageSamplerDescriptor::linear()
    });
    StandardMaterial {
        base_color_texture: Some(images.add(image)),
        alpha_mode: AlphaMode::Mask(0.5),
        double_sided: true,
        cull_mode: None,
        perceptual_roughness: 0.92,
        reflectance: 0.15,
        ..default()
    }
}

#[derive(Default)]
struct BranchMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}
impl BranchMesh {
    fn card(&mut self, origin: Vec3, direction: Vec3, across: Vec3, length: f32, width: f32) {
        let start = self.positions.len() as u32;
        let normal = (across.cross(direction) * 0.4 + Vec3::Y * 0.8).normalize();
        for t in [0., 0.5, 1.] {
            let center = origin + direction * (t * length)
                - Vec3::Y * (std::f32::consts::PI * t).sin() * length * 0.07;
            for side in [-1., 1.] {
                self.positions
                    .push((center + across * width * side * 0.5).to_array());
                self.normals.push(normal.to_array());
                self.uvs.push([t, (side + 1.) * 0.5]);
            }
        }
        for row in 0..2 {
            let i = start + row * 2;
            self.indices
                .extend_from_slice(&[i, i + 1, i + 2, i + 1, i + 3, i + 2]);
        }
    }
}

pub(crate) fn canopy(seed: u32) -> Mesh {
    let mut data = BranchMesh::default();
    for level in 0..9 {
        let q = level as f32 / 9.;
        let height = 1.65 + q * 2.7;
        let count = 7 + (hash(level, seed + 5) * 3.) as u32;
        for j in 0..count {
            let angle = j as f32 / count as f32 * std::f32::consts::TAU
                + level as f32 * 2.399_963
                + hash(seed, j + level * 13) * 0.35;
            let radial = Vec3::new(angle.cos(), 0., angle.sin());
            let across = Vec3::new(-radial.z, 0., radial.x);
            let length = (1. - q).powf(0.72) * (1.02 + hash(j + level * 11, seed + 23) * 0.30);
            let origin = Vec3::Y * (height + hash(j, level + seed + 31) * 0.10);
            let direction =
                (radial - Vec3::Y * (0.08 + hash(j, seed + level + 51) * 0.18)).normalize();
            data.card(origin, direction, across, length, length * 0.68);
            data.card(
                origin,
                direction,
                (across * 0.35 + Vec3::Y * 0.94).normalize(),
                length,
                length * 0.56,
            );
        }
    }
    for across in [Vec3::X, Vec3::Z] {
        data.card(Vec3::Y * 3.92, Vec3::Y, across, 0.53, 0.20);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, data.uvs);
    mesh.insert_indices(Indices::U32(data.indices));
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;
    #[test]
    fn branch_mips_keep_openings_and_masked_coverage_without_repeating_edges() {
        let mut images = Assets::<Image>::default();
        let material = material(&mut images);
        assert_eq!(material.alpha_mode, AlphaMode::Mask(0.5));
        assert!(material.double_sided && material.cull_mode.is_none());
        let image = images.get(&material.base_color_texture.unwrap()).unwrap();
        let mut bytes = image.data.as_ref().unwrap().as_slice();
        let (mut width, mut height) = (WIDTH, HEIGHT);
        let mut reference = 0.;
        for level in 0..MIPS {
            let (pixels, next) = bytes.split_at(width * height * 4);
            let cover = pixels.chunks_exact(4).filter(|p| p[3] >= 128).count() as f32
                / (width * height) as f32;
            if level == 0 {
                reference = cover;
                assert!((0.05..0.6).contains(&cover));
            }
            assert!(
                (cover - reference).abs() < 0.04 + 1. / (width * height) as f32,
                "mip {level}: {cover} / {reference}"
            );
            bytes = next;
            width /= 2;
            height /= 2;
        }
        assert!(bytes.is_empty());
        let ImageSampler::Descriptor(sampler) = &image.sampler else {
            panic!("foliage sampler");
        };
        assert_eq!(sampler.address_mode_u, ImageAddressMode::ClampToEdge);
        assert_eq!(sampler.address_mode_v, ImageAddressMode::ClampToEdge);
    }
    #[test]
    fn canopy_geometry_has_bounded_cost_and_finite_three_dimensional_branches() {
        for seed in 0..8 {
            let mesh = canopy(seed);
            let Some(VertexAttributeValues::Float32x3(points)) =
                mesh.attribute(Mesh::ATTRIBUTE_POSITION)
            else {
                panic!("positions");
            };
            let Some(VertexAttributeValues::Float32x3(normals)) =
                mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
            else {
                panic!("normals");
            };
            assert_eq!(points.len(), normals.len());
            assert!(mesh.indices().unwrap().len() / 3 < 1024);
            assert!(points.iter().all(|p| {
                let p = Vec3::from_array(*p);
                p.is_finite() && (0.9..4.6).contains(&p.y) && p.xz().length() < 1.6
            }));
            assert!(normals
                .iter()
                .all(|n| (Vec3::from_array(*n).length() - 1.).abs() < 1e-5));
        }
    }
}

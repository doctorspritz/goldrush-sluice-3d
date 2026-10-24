//! Metre-scale visual soil detail. Textures add no terrain or collision geometry.
use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

const SIZE: usize = 256;

/// Multiply the geological vertex colours by neutral soil/gravel detail.
/// Mesh UVs are measured in metres, with one repeat per metre; tangents are required.
pub(crate) fn material(images: &mut Assets<Image>) -> StandardMaterial {
    let samples: Vec<_> = (0..SIZE * SIZE)
        .map(|i| {
            detail(
                (i % SIZE) as f32 / SIZE as f32,
                (i / SIZE) as f32 / SIZE as f32,
            )
        })
        .collect();
    let albedo = samples.iter().map(|sample| Vec3::splat(sample.0)).collect();
    let normals = (0..SIZE * SIZE)
        .map(|i| {
            let x = i % SIZE;
            let y = i / SIZE;
            let height = |x: usize, y: usize| samples[(y % SIZE) * SIZE + x % SIZE].1;
            // The scalar height is in metres, so central differences have physical slope.
            let dx = (height(x + 1, y) - height(x + SIZE - 1, y)) * SIZE as f32 * 0.5;
            let dy = (height(x, y + 1) - height(x, y + SIZE - 1)) * SIZE as f32 * 0.5;
            Vec3::new(-dx, -dy, 1.0).normalize()
        })
        .collect();
    StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(images.add(mip_image(albedo, false))),
        normal_map_texture: Some(images.add(mip_image(normals, true))),
        perceptual_roughness: 0.92,
        ..default()
    }
}

/// A periodic jittered gravel bed over fine soil noise. Heights stay a few mm;
/// these normals describe surface grain, not another large-scale terrain shape.
fn detail(u: f32, v: f32) -> (f32, f32) {
    let coarse = noise(u, v, 8, 2);
    let soil = noise(u, v, 64, 7);
    let fine = noise(u, v, 128, 11);
    let x = u * 16.0;
    let y = v * 16.0;
    let mut pebble = 0.0_f32;
    let mut stone_tone = 0.0_f32;
    for oy in -1..=1 {
        for ox in -1..=1 {
            let cx = x.floor() as i32 + ox;
            let cy = y.floor() as i32 + oy;
            let px = cx as f32 + 0.15 + hash(cx, cy, 16, 17) * 0.7;
            let py = cy as f32 + 0.15 + hash(cx, cy, 16, 23) * 0.7;
            let radius = 0.18 + hash(cx, cy, 16, 29) * 0.28;
            let distance = ((x - px).powi(2) + (y - py).powi(2)).sqrt() / radius;
            let dome = (1.0 - distance * distance).max(0.0).powi(2);
            if dome > pebble {
                pebble = dome;
                stone_tone = hash(cx, cy, 16, 31) - 0.5;
            }
        }
    }
    let albedo = (0.82
        + 0.075 * (coarse - 0.5)
        + 0.09 * (soil - 0.5)
        + 0.04 * (fine - 0.5)
        + pebble * (0.07 + 0.12 * stone_tone))
        .clamp(0.68, 0.98);
    let height = 0.0008 * (soil - 0.5) + 0.00035 * (fine - 0.5) + 0.003 * pebble;
    (albedo, height)
}

fn hash(x: i32, y: i32, period: i32, seed: u32) -> f32 {
    let mut bits = (x.rem_euclid(period) as u32).wrapping_mul(0x9e37_79b9)
        ^ (y.rem_euclid(period) as u32).wrapping_mul(0x85eb_ca6b)
        ^ seed.wrapping_mul(0xc2b2_ae35);
    bits = (bits ^ (bits >> 16)).wrapping_mul(0x7feb_352d);
    bits = (bits ^ (bits >> 15)).wrapping_mul(0x846c_a68b);
    ((bits ^ (bits >> 16)) >> 8) as f32 / 16_777_215.0
}

fn noise(u: f32, v: f32, period: i32, seed: u32) -> f32 {
    let x = u * period as f32;
    let y = v * period as f32;
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let tx = fade(x - x.floor());
    let ty = fade(y - y.floor());
    let a = hash(ix, iy, period, seed);
    let b = hash(ix + 1, iy, period, seed);
    let c = hash(ix, iy + 1, period, seed);
    let d = hash(ix + 1, iy + 1, period, seed);
    (a + (b - a) * tx) * (1.0 - ty) + (c + (d - c) * tx) * ty
}

/// Generate every mip on the CPU once. Colour mips average in linear space;
/// normal mips average vectors and renormalize instead of averaging RGB colours.
fn mip_image(mut pixels: Vec<Vec3>, normals: bool) -> Image {
    let mut data = Vec::with_capacity((SIZE * SIZE * 4 - 1) / 3 * 4);
    let mut width = SIZE;
    loop {
        for &pixel in &pixels {
            let encoded = if normals {
                pixel * 0.5 + Vec3::splat(0.5)
            } else {
                pixel.map(|linear| {
                    if linear <= 0.003_130_8 {
                        linear * 12.92
                    } else {
                        1.055 * linear.powf(1.0 / 2.4) - 0.055
                    }
                })
            };
            for channel in encoded.to_array() {
                data.push((channel.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            data.push(255);
        }
        if width == 1 {
            break;
        }
        let next_width = width / 2;
        let mut next = Vec::with_capacity(next_width * next_width);
        for y in 0..next_width {
            for x in 0..next_width {
                let index = y * 2 * width + x * 2;
                let average = (pixels[index]
                    + pixels[index + 1]
                    + pixels[index + width]
                    + pixels[index + width + 1])
                    * 0.25;
                next.push(if normals {
                    average.normalize()
                } else {
                    average
                });
            }
        }
        pixels = next;
        width = next_width;
    }
    let mut image = Image::new_uninit(
        Extent3d {
            width: SIZE as u32,
            height: SIZE as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        if normals {
            TextureFormat::Rgba8Unorm
        } else {
            TextureFormat::Rgba8UnormSrgb
        },
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = SIZE.ilog2() + 1;
    image.data = Some(data);
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        anisotropy_clamp: 4,
        ..ImageSamplerDescriptor::linear()
    });
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_mips_support_filtered_distant_terrain() {
        let mut images = Assets::<Image>::default();
        let material = material(&mut images);
        for handle in [material.base_color_texture, material.normal_map_texture] {
            let image = images.get(&handle.unwrap()).unwrap();
            let mut width = image.texture_descriptor.size.width as usize;
            let mut expected_bytes = 0;
            for _ in 0..image.texture_descriptor.mip_level_count {
                expected_bytes += width * width * 4;
                width /= 2;
            }
            assert_eq!(width, 0, "mip chain must reach 1x1");
            assert_eq!(image.data.as_ref().unwrap().len(), expected_bytes);
            let ImageSampler::Descriptor(sampler) = &image.sampler else {
                panic!("ground textures require their repeat sampler");
            };
            assert_eq!(sampler.address_mode_u, ImageAddressMode::Repeat);
            assert_eq!(sampler.address_mode_v, ImageAddressMode::Repeat);
            assert_eq!(sampler.mipmap_filter, bevy::image::ImageFilterMode::Linear);
        }
    }

    #[test]
    fn periodic_shallow_relief_has_valid_normals_at_every_mip() {
        for i in 0..65 {
            let u = i as f32 / 64.0;
            let a = detail(u, 0.375);
            let b = detail(u + 1.0, 1.375);
            assert!((a.0 - b.0).abs() < 1e-5);
            assert!((a.1 - b.1).abs() < 1e-5);
            assert!(a.1.abs() < 0.004, "detail must remain millimetre-scale");
        }
        let mut images = Assets::<Image>::default();
        let material = material(&mut images);
        let image = images.get(&material.normal_map_texture.unwrap()).unwrap();
        assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba8Unorm);
        for pixel in image.data.as_ref().unwrap().as_chunks::<4>().0 {
            let normal =
                Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32) / 127.5 - Vec3::ONE;
            assert!(normal.is_finite());
            assert!((normal.length() - 1.0).abs() < 0.012);
            assert!(
                normal.z > 0.65,
                "relief must not produce steep or inverted faces"
            );
            assert_eq!(pixel[3], 255);
        }
    }
}

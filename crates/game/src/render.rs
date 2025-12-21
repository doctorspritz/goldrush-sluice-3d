//! GPU-accelerated particle rendering
//!
//! Renders particles as smooth circles with velocity-based coloring.
//! Uses macroquad's material system for custom GLSL shaders.
//! Supports metaball-style rendering for cohesive fluid appearance.
//!
//! ## GPU Batching Strategy
//! Instead of one draw call per particle, we batch all particles of each
//! material type into a single Mesh and draw with one call per material.
//! This reduces draw calls from N to 5 (one per material type).

use macroquad::prelude::*;
use macroquad::miniquad::{BlendState, Equation, BlendFactor, BlendValue};
use macroquad::models::{Mesh, Vertex, draw_mesh};
use sim::Particles;

/// Vertex shader for particle circles
const VERTEX_SHADER: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;

varying lowp vec2 v_uv;

uniform mat4 Model;
uniform mat4 Projection;

void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    v_uv = texcoord;
}
"#;

/// Fragment shader for smooth circles
const FRAGMENT_SHADER: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;

uniform lowp vec4 particleColor;

void main() {
    // Calculate distance from center (UV is 0-1)
    vec2 center = vec2(0.5, 0.5);
    float dist = length(v_uv - center) * 2.0;

    // Smooth circle with soft edge
    float alpha = 1.0 - smoothstep(0.7, 1.0, dist);

    // Discard pixels outside circle
    if (alpha < 0.01) {
        discard;
    }

    // Use uniform color
    vec3 finalColor = particleColor.rgb;

    // Subtle rim darkening for depth
    float rim = smoothstep(0.5, 0.9, dist);
    finalColor = mix(finalColor, finalColor * 0.8, rim);

    gl_FragColor = vec4(finalColor, alpha * 0.85);
}
"#;

/// Metaball density pass - soft gaussian falloff with additive blending
const METABALL_DENSITY_FRAG: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;

uniform lowp vec4 particleColor;
uniform lowp float densityMult;

void main() {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(v_uv - center) * 2.0;

    // Gaussian-like falloff: strong in center, soft fade at edges
    float density = exp(-dist * dist * 4.0) * densityMult;

    // Output: RGB is color weighted by density, A is density itself
    gl_FragColor = vec4(particleColor.rgb * density, density);
}
"#;

/// Metaball threshold pass - thresholds the accumulated density
const METABALL_THRESHOLD_FRAG: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;

uniform sampler2D Texture;
uniform lowp float threshold;

void main() {
    // Flip Y when sampling to correct for render target coordinate mismatch
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec4 texSample = texture2D(Texture, uv);

    float density = texSample.a;

    // Discard below threshold
    if (density < threshold) {
        discard;
    }

    // Recover color (was pre-multiplied by density)
    vec3 color = texSample.rgb / max(density, 0.001);

    // Very sharp edge transition - minimal alpha variation
    float edge = smoothstep(threshold, threshold + 0.03, density);

    // Subtle rim darkening for depth (reduced effect)
    float rim = 1.0 - smoothstep(threshold, threshold + 0.1, density);
    color = mix(color, color * 0.9, rim * 0.3);

    // High consistent alpha with minimal variation
    gl_FragColor = vec4(color, 0.85 + edge * 0.1);
}
"#;

/// Particle renderer with GPU-accelerated instanced circles
pub struct ParticleRenderer {
    material: Material,
    /// 1x1 white texture for UV-mapped quad rendering
    white_texture: Texture2D,
    velocity_max: f32,
    particle_scale: f32,
}

impl ParticleRenderer {
    /// Create a new particle renderer
    pub fn new() -> Self {
        let material = load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: FRAGMENT_SHADER,
            },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("particleColor", UniformType::Float4),
                ],
                pipeline_params: PipelineParams {
                    color_blend: Some(BlendState::new(
                        Equation::Add,
                        BlendFactor::Value(BlendValue::SourceAlpha),
                        BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                    )),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("Failed to load particle shader");

        // Create 1x1 white texture for UV-mapped rendering
        let white_img = Image::gen_image_color(1, 1, WHITE);
        let white_texture = Texture2D::from_image(&white_img);

        Self {
            material,
            white_texture,
            velocity_max: 100.0,
            particle_scale: 2.5, // Smaller particles for denser flow
        }
    }

    /// Set maximum velocity for color scaling
    pub fn set_velocity_max(&mut self, max: f32) {
        self.velocity_max = max;
    }

    /// Set particle visual scale
    pub fn set_particle_scale(&mut self, scale: f32) {
        self.particle_scale = scale;
    }

    /// Render all particles as smooth circles
    pub fn draw(&self, particles: &Particles, screen_scale: f32) {
        gl_use_material(&self.material);

        let size = self.particle_scale * screen_scale;

        // Draw each particle as a textured quad (circle rendered in shader)
        for particle in particles.iter() {
            let x = particle.position.x * screen_scale;
            let y = particle.position.y * screen_scale;

            let [r, g, b, _] = particle.material.color();

            // Set particle color uniform
            self.material.set_uniform("particleColor", [
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                1.0f32,
            ]);

            // Draw textured quad (shader makes it circular, texture provides UVs)
            draw_texture_ex(
                &self.white_texture,
                x - size / 2.0,
                y - size / 2.0,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(size, size)),
                    ..Default::default()
                },
            );
        }

        gl_use_default_material();
    }

    /// Draw particles batched by material (one uniform per material type)
    pub fn draw_sorted(&self, particles: &Particles, screen_scale: f32) {
        gl_use_material(&self.material);

        let size = self.particle_scale * screen_scale;

        // Draw order: water, mud, sand, magnetite, gold (lightest to heaviest)
        let materials = [
            sim::ParticleMaterial::Water,
            sim::ParticleMaterial::Mud,
            sim::ParticleMaterial::Sand,
            sim::ParticleMaterial::Magnetite,
            sim::ParticleMaterial::Gold,
        ];

        for mat in materials {
            let [r, g, b, _] = mat.color();
            self.material.set_uniform("particleColor", [
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                1.0f32,
            ]);

            for particle in particles.iter() {
                if particle.material == mat {
                    let x = particle.position.x * screen_scale;
                    let y = particle.position.y * screen_scale;
                    draw_texture_ex(
                        &self.white_texture,
                        x - size / 2.0,
                        y - size / 2.0,
                        WHITE,
                        DrawTextureParams {
                            dest_size: Some(vec2(size, size)),
                            ..Default::default()
                        },
                    );
                }
            }
        }

        gl_use_default_material();
    }
}

impl Default for ParticleRenderer {
    fn default() -> Self {
        Self::new()
    }
}

/// Metaball-style fluid renderer with GPU batched rendering
/// Uses two-pass rendering: density accumulation + threshold
/// Batches all particles into meshes for minimal draw calls
pub struct MetaballRenderer {
    density_material: Material,
    threshold_material: Material,
    render_target: RenderTarget,
    white_texture: Texture2D,
    particle_scale: f32,
    threshold: f32,
}

impl MetaballRenderer {
    /// Create a new metaball renderer
    pub fn new(width: u32, height: u32) -> Self {
        // Density accumulation pass - additive blending
        let density_material = load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: METABALL_DENSITY_FRAG,
            },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("particleColor", UniformType::Float4),
                    UniformDesc::new("densityMult", UniformType::Float1),
                ],
                pipeline_params: PipelineParams {
                    // Additive blending: src + dst
                    color_blend: Some(BlendState::new(
                        Equation::Add,
                        BlendFactor::One,
                        BlendFactor::One,
                    )),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("Failed to load density shader");

        // Threshold pass - standard alpha blending
        let threshold_material = load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: METABALL_THRESHOLD_FRAG,
            },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("threshold", UniformType::Float1),
                ],
                pipeline_params: PipelineParams {
                    color_blend: Some(BlendState::new(
                        Equation::Add,
                        BlendFactor::Value(BlendValue::SourceAlpha),
                        BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                    )),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .expect("Failed to load threshold shader");

        // Create render target for density accumulation
        let render_target = render_target(width, height);
        render_target.texture.set_filter(FilterMode::Linear);

        // Create 1x1 white texture for mesh UV mapping
        let white_img = Image::gen_image_color(1, 1, WHITE);
        let white_texture = Texture2D::from_image(&white_img);

        Self {
            density_material,
            threshold_material,
            render_target,
            white_texture,
            particle_scale: 6.0,
            threshold: 0.08,
        }
    }

    /// Set particle visual scale (larger = more overlap = more blobby)
    pub fn set_particle_scale(&mut self, scale: f32) {
        self.particle_scale = scale;
    }

    /// Set threshold for metaball surface (lower = more fluid, higher = more droplets)
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Render particles with metaball effect
    /// Uses per-particle draw calls with material batching (one uniform set per material)
    pub fn draw(&self, particles: &Particles, screen_scale: f32) {
        let base_size = self.particle_scale * screen_scale;
        let rt_w = self.render_target.texture.width();
        let rt_h = self.render_target.texture.height();

        // Pass 1: Render density to render target
        set_camera(&Camera2D {
            zoom: vec2(2.0 / rt_w, -2.0 / rt_h),
            target: vec2(rt_w / 2.0, rt_h / 2.0),
            render_target: Some(self.render_target.clone()),
            ..Default::default()
        });

        clear_background(Color::new(0.0, 0.0, 0.0, 0.0));

        gl_use_material(&self.density_material);

        // Draw order: water, mud, sand, magnetite, gold
        // Batch by material to minimize uniform changes
        let materials = [
            sim::ParticleMaterial::Water,
            sim::ParticleMaterial::Mud,
            sim::ParticleMaterial::Sand,
            sim::ParticleMaterial::Magnetite,
            sim::ParticleMaterial::Gold,
        ];

        for mat in materials {
            let [r, g, b, _] = mat.color();
            let size = base_size * mat.render_scale();
            let density_mult = mat.density_contribution();

            self.density_material.set_uniform("particleColor", [
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                1.0f32,
            ]);
            self.density_material.set_uniform("densityMult", density_mult);

            for particle in particles.iter() {
                if particle.material == mat {
                    let x = particle.position.x * screen_scale;
                    let y = particle.position.y * screen_scale;
                    draw_texture_ex(
                        &self.white_texture,
                        x - size / 2.0,
                        y - size / 2.0,
                        WHITE,
                        DrawTextureParams {
                            dest_size: Some(vec2(size, size)),
                            ..Default::default()
                        },
                    );
                }
            }
        }

        gl_use_default_material();

        // Pass 2: Draw render target to screen with threshold
        set_default_camera();

        gl_use_material(&self.threshold_material);
        self.threshold_material.set_uniform("threshold", self.threshold);

        draw_texture_ex(
            &self.render_target.texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(screen_width(), screen_height())),
                ..Default::default()
            },
        );

        gl_use_default_material();
    }
}

/// Fast batched circle renderer using macroquad's internal batching
/// Renders particles as simple filled circles with no shader overhead
pub fn draw_particles_fast(particles: &Particles, screen_scale: f32, base_size: f32) {
    // Draw all particles using macroquad's draw_circle which batches internally
    // This avoids custom shader overhead and uniform changes
    for particle in particles.iter() {
        let x = particle.position.x * screen_scale;
        let y = particle.position.y * screen_scale;
        let [r, g, b, a] = particle.material.color();
        let color = Color::from_rgba(r, g, b, a);
        let size = base_size * particle.material.render_scale();

        draw_circle(x, y, size, color);
    }
}

/// Ultra-fast rectangle renderer - rectangles batch better than circles
pub fn draw_particles_rect(particles: &Particles, screen_scale: f32, base_size: f32) {
    for particle in particles.iter() {
        let x = particle.position.x * screen_scale;
        let y = particle.position.y * screen_scale;
        let [r, g, b, a] = particle.material.color();
        let color = Color::from_rgba(r, g, b, a);
        let size = base_size * particle.material.render_scale();

        draw_rectangle(x - size/2.0, y - size/2.0, size, size, color);
    }
}

/// Single-mesh batched renderer - builds meshes with all particles
/// Uses vertex colors and default material for maximum batching
/// Batches in chunks of ~8000 particles to stay within u16 index limits
pub fn draw_particles_mesh(particles: &Particles, screen_scale: f32, base_size: f32) {
    let count = particles.len();
    if count == 0 {
        return;
    }

    // Max particles per batch: 65536 indices / 6 indices per quad = 10922
    // Use 8000 for safety margin
    const MAX_PER_BATCH: usize = 8000;

    let mut batch_start = 0;
    while batch_start < count {
        let batch_end = (batch_start + MAX_PER_BATCH).min(count);
        let batch_size = batch_end - batch_start;

        // Pre-allocate vertex and index buffers for this batch
        let mut vertices: Vec<Vertex> = Vec::with_capacity(batch_size * 4);
        let mut indices: Vec<u16> = Vec::with_capacity(batch_size * 6);

        for (local_i, particle) in particles.iter().skip(batch_start).take(batch_size).enumerate() {
            let x = particle.position.x * screen_scale;
            let y = particle.position.y * screen_scale;
            let [r, g, b, a] = particle.material.color();
            let color = Color::from_rgba(r, g, b, a);
            let size = base_size * particle.material.render_scale();

            let half = size / 2.0;
            let base_idx = (local_i * 4) as u16;

            // Quad vertices (top-left, top-right, bottom-right, bottom-left)
            vertices.push(Vertex {
                position: vec3(x - half, y - half, 0.0),
                uv: vec2(0.0, 0.0),
                color: color.into(),
                normal: vec4(0.0, 0.0, 1.0, 0.0),
            });
            vertices.push(Vertex {
                position: vec3(x + half, y - half, 0.0),
                uv: vec2(1.0, 0.0),
                color: color.into(),
                normal: vec4(0.0, 0.0, 1.0, 0.0),
            });
            vertices.push(Vertex {
                position: vec3(x + half, y + half, 0.0),
                uv: vec2(1.0, 1.0),
                color: color.into(),
                normal: vec4(0.0, 0.0, 1.0, 0.0),
            });
            vertices.push(Vertex {
                position: vec3(x - half, y + half, 0.0),
                uv: vec2(0.0, 1.0),
                color: color.into(),
                normal: vec4(0.0, 0.0, 1.0, 0.0),
            });

            // Two triangles per quad
            indices.push(base_idx);
            indices.push(base_idx + 1);
            indices.push(base_idx + 2);
            indices.push(base_idx);
            indices.push(base_idx + 2);
            indices.push(base_idx + 3);
        }

        // Build and draw mesh for this batch
        let mesh = Mesh {
            vertices,
            indices,
            texture: None,
        };
        draw_mesh(&mesh);

        batch_start = batch_end;
    }
}

/// Generate a gradient texture for velocity coloring (alternative approach)
pub fn create_velocity_gradient(resolution: u32) -> Image {
    let mut img = Image::gen_image_color(resolution as u16, 1, WHITE);
    let pixels = img.get_image_data_mut();

    for i in 0..resolution {
        let t = i as f32 / (resolution - 1) as f32;

        // Blue (slow) -> Cyan -> White (fast)
        let r = (t * 255.0) as u8;
        let g = ((0.5 + t * 0.5) * 255.0) as u8;
        let b = 255;

        pixels[i as usize] = [r, g, b, 255];
    }

    img
}

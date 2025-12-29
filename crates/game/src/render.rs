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
//!
//! ## FastMetaballRenderer
//! Optimized metaball rendering with:
//! - Half-resolution density buffer
//! - Mesh batching for density pass (not per-particle draw calls)
//! - Separable Gaussian blur for smooth edges
//! - Gradient-based surface normals for lighting

use macroquad::prelude::*;
use macroquad::miniquad::{BlendState, Equation, BlendFactor, BlendValue};
use macroquad::models::{Mesh, Vertex, draw_mesh};
use sim::Particles;
use sim::particle::ParticleState;

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

    // Simplified color output without rim darkening
    gl_FragColor = vec4(finalColor, alpha);
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

        // Draw order: water, sand, magnetite, gold
        let materials = [
            sim::ParticleMaterial::Water,
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

    /// Draw particles, optionally filtering for only water or only solids
    pub fn draw_filtered(&self, particles: &Particles, screen_scale: f32, draw_water: bool) {
        gl_use_material(&self.material);

        let size = self.particle_scale * screen_scale;

        // Draw order: water, sand, magnetite, gold
        let materials = [
            sim::ParticleMaterial::Water,
            sim::ParticleMaterial::Sand,
            sim::ParticleMaterial::Magnetite,
            sim::ParticleMaterial::Gold,
        ];

        for mat in materials {
            let is_water = mat == sim::ParticleMaterial::Water;
            if draw_water != is_water {
                continue;
            }

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

        // Draw order: water, sand, magnetite, gold
        let materials = [
            sim::ParticleMaterial::Water,
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

    /// Render particles with metaball effect, filtered by type (water vs solids)
    pub fn draw_filtered(&self, particles: &Particles, screen_scale: f32, draw_water: bool) {
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

        // Draw order: water, sand, magnetite, gold
        let materials = [
            sim::ParticleMaterial::Water,
            sim::ParticleMaterial::Sand,
            sim::ParticleMaterial::Magnetite,
            sim::ParticleMaterial::Gold,
        ];

        for mat in materials {
            let is_water = mat == sim::ParticleMaterial::Water;
            if draw_water != is_water {
                continue;
            }

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

// ============================================================================
// OPTIMIZED METABALL RENDERER
// ============================================================================

/// Horizontal Gaussian blur shader
const BLUR_H_FRAG: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;

uniform sampler2D Texture;
uniform lowp vec2 texelSize;

void main() {
    // Flip Y for render target coordinate system
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);

    // 9-tap Gaussian kernel (sigma ~2.0)
    // Weights: 0.05, 0.09, 0.12, 0.15, 0.18, 0.15, 0.12, 0.09, 0.05
    vec4 sum = vec4(0.0);
    sum += texture2D(Texture, uv + vec2(-4.0 * texelSize.x, 0.0)) * 0.05;
    sum += texture2D(Texture, uv + vec2(-3.0 * texelSize.x, 0.0)) * 0.09;
    sum += texture2D(Texture, uv + vec2(-2.0 * texelSize.x, 0.0)) * 0.12;
    sum += texture2D(Texture, uv + vec2(-1.0 * texelSize.x, 0.0)) * 0.15;
    sum += texture2D(Texture, uv) * 0.18;
    sum += texture2D(Texture, uv + vec2(1.0 * texelSize.x, 0.0)) * 0.15;
    sum += texture2D(Texture, uv + vec2(2.0 * texelSize.x, 0.0)) * 0.12;
    sum += texture2D(Texture, uv + vec2(3.0 * texelSize.x, 0.0)) * 0.09;
    sum += texture2D(Texture, uv + vec2(4.0 * texelSize.x, 0.0)) * 0.05;

    gl_FragColor = sum;
}
"#;

/// Vertical Gaussian blur shader
const BLUR_V_FRAG: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;

uniform sampler2D Texture;
uniform lowp vec2 texelSize;

void main() {
    // Flip Y for render target coordinate system
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);

    // 9-tap Gaussian kernel (sigma ~2.0)
    vec4 sum = vec4(0.0);
    sum += texture2D(Texture, uv + vec2(0.0, -4.0 * texelSize.y)) * 0.05;
    sum += texture2D(Texture, uv + vec2(0.0, -3.0 * texelSize.y)) * 0.09;
    sum += texture2D(Texture, uv + vec2(0.0, -2.0 * texelSize.y)) * 0.12;
    sum += texture2D(Texture, uv + vec2(0.0, -1.0 * texelSize.y)) * 0.15;
    sum += texture2D(Texture, uv) * 0.18;
    sum += texture2D(Texture, uv + vec2(0.0, 1.0 * texelSize.y)) * 0.15;
    sum += texture2D(Texture, uv + vec2(0.0, 2.0 * texelSize.y)) * 0.12;
    sum += texture2D(Texture, uv + vec2(0.0, 3.0 * texelSize.y)) * 0.09;
    sum += texture2D(Texture, uv + vec2(0.0, 4.0 * texelSize.y)) * 0.05;

    gl_FragColor = sum;
}
"#;

/// Density accumulation shader with soft gaussian falloff
/// Uses uniform for water color (vertex color not reliably passed through custom materials)
const FAST_DENSITY_FRAG: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;

uniform lowp vec4 waterColor;

void main() {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(v_uv - center) * 2.0;

    // Gaussian-like falloff
    float density = exp(-dist * dist * 4.0);

    // Output: RGB weighted by density, A is density
    gl_FragColor = vec4(waterColor.rgb * density, density);
}
"#;

/// Vertex shader that passes vertex color to fragment shader
const FAST_DENSITY_VERT: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color;

varying lowp vec2 v_uv;
varying lowp vec4 v_color;

uniform mat4 Model;
uniform mat4 Projection;

void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    v_uv = texcoord;
    v_color = color;
}
"#;

/// Surface shader with gradient-based lighting and translucency
const SURFACE_FRAG: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;

uniform sampler2D Texture;
uniform lowp float threshold;
uniform lowp vec2 texelSize;

void main() {
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec4 center = texture2D(Texture, uv);
    float density = center.a;

    // Discard below threshold
    if (density < threshold) {
        discard;
    }

    // Recover base color
    vec3 baseColor = center.rgb / max(density, 0.001);

    // Calculate gradient for pseudo-normal (Sobel-like)
    float left = texture2D(Texture, uv - vec2(texelSize.x, 0.0)).a;
    float right = texture2D(Texture, uv + vec2(texelSize.x, 0.0)).a;
    float up = texture2D(Texture, uv - vec2(0.0, texelSize.y)).a;
    float down = texture2D(Texture, uv + vec2(0.0, texelSize.y)).a;

    vec2 gradient = vec2(right - left, down - up);
    float gradMag = length(gradient);

    // Simple directional light from top-left
    vec3 lightDir = normalize(vec3(-0.5, -0.7, 0.5));
    vec3 normal = normalize(vec3(-gradient.x, -gradient.y, 0.3));
    float diffuse = max(dot(normal, lightDir), 0.0);

    // Specular highlight
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVec), 0.0), 32.0);

    // Edge detection for rim
    float edge = smoothstep(threshold, threshold + 0.05, density);
    float rim = 1.0 - edge;

    // Combine lighting
    vec3 ambient = baseColor * 0.4;
    vec3 diffuseColor = baseColor * diffuse * 0.5;
    vec3 specColor = vec3(1.0) * spec * 0.3;
    vec3 rimColor = baseColor * rim * 0.2;

    vec3 finalColor = ambient + diffuseColor + specColor + rimColor;

    // Translucency based on density (thicker = more opaque)
    float alpha = smoothstep(threshold, threshold + 0.15, density) * 0.85 + 0.1;

    gl_FragColor = vec4(finalColor, alpha);
}
"#;

/// Optimized metaball renderer with mesh batching and blur
/// Uses half-resolution for density/blur, upscales to screen
pub struct FastMetaballRenderer {
    density_material: Material,
    blur_h_material: Material,
    blur_v_material: Material,
    surface_material: Material,
    /// Half-res render target for density
    density_rt: RenderTarget,
    /// Half-res render target for horizontal blur output
    blur_h_rt: RenderTarget,
    /// Half-res render target for vertical blur output (final density)
    blur_v_rt: RenderTarget,
    /// 1x1 white texture for UV mapping
    white_texture: Texture2D,
    /// Full screen dimensions
    screen_w: f32,
    screen_h: f32,
    /// Half-res dimensions
    half_w: f32,
    half_h: f32,
    /// Particle visual scale
    particle_scale: f32,
    /// Density threshold for surface
    threshold: f32,
}

impl FastMetaballRenderer {
    /// Create a new optimized metaball renderer
    /// Uses half-resolution for density/blur passes
    pub fn new(width: u32, height: u32) -> Self {
        let half_w = width / 2;
        let half_h = height / 2;

        // Density pass - additive blending with water color uniform
        let density_material = load_material(
            ShaderSource::Glsl {
                vertex: FAST_DENSITY_VERT,
                fragment: FAST_DENSITY_FRAG,
            },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("waterColor", UniformType::Float4),
                ],
                pipeline_params: PipelineParams {
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
        .expect("Failed to load fast density shader");

        // Horizontal blur
        let blur_h_material = load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: BLUR_H_FRAG,
            },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("texelSize", UniformType::Float2),
                ],
                ..Default::default()
            },
        )
        .expect("Failed to load blur H shader");

        // Vertical blur
        let blur_v_material = load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: BLUR_V_FRAG,
            },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("texelSize", UniformType::Float2),
                ],
                ..Default::default()
            },
        )
        .expect("Failed to load blur V shader");

        // Surface shader with lighting
        let surface_material = load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: SURFACE_FRAG,
            },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("threshold", UniformType::Float1),
                    UniformDesc::new("texelSize", UniformType::Float2),
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
        .expect("Failed to load surface shader");

        // Create half-res render targets
        let density_rt = render_target(half_w, half_h);
        density_rt.texture.set_filter(FilterMode::Linear);

        let blur_h_rt = render_target(half_w, half_h);
        blur_h_rt.texture.set_filter(FilterMode::Linear);

        let blur_v_rt = render_target(half_w, half_h);
        blur_v_rt.texture.set_filter(FilterMode::Linear);

        // White texture for UV mapping
        let white_img = Image::gen_image_color(1, 1, WHITE);
        let white_texture = Texture2D::from_image(&white_img);

        Self {
            density_material,
            blur_h_material,
            blur_v_material,
            surface_material,
            density_rt,
            blur_h_rt,
            blur_v_rt,
            white_texture,
            screen_w: width as f32,
            screen_h: height as f32,
            half_w: half_w as f32,
            half_h: half_h as f32,
            particle_scale: 3.5,  // Smaller metaballs (half-res so effective 7.0)
            threshold: 0.06,
        }
    }

    /// Set particle visual scale
    pub fn set_particle_scale(&mut self, scale: f32) {
        self.particle_scale = scale;
    }

    /// Set density threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Draw water particles as batched meshes for density pass
    fn draw_water_density_meshes(&self, particles: &Particles, screen_scale: f32) {
        const MAX_PER_BATCH: usize = 8000;

        let half_scale = screen_scale * 0.5; // Half-res coordinates
        let base_size = self.particle_scale * half_scale;

        let mut vertices: Vec<Vertex> = Vec::with_capacity(MAX_PER_BATCH * 4);
        let mut indices: Vec<u16> = Vec::with_capacity(MAX_PER_BATCH * 6);
        let mut count = 0usize;

        for particle in particles.iter() {
            if particle.material != sim::ParticleMaterial::Water {
                continue;
            }

            let x = particle.position.x * half_scale;
            let y = particle.position.y * half_scale;
            let [r, g, b, _] = particle.material.color();
            let color = Color::from_rgba(r, g, b, 255);
            let size = base_size * particle.material.render_scale();
            let half = size / 2.0;

            let base_idx = (count * 4) as u16;

            // Quad vertices with UVs for gaussian falloff
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

            indices.push(base_idx);
            indices.push(base_idx + 1);
            indices.push(base_idx + 2);
            indices.push(base_idx);
            indices.push(base_idx + 2);
            indices.push(base_idx + 3);

            count += 1;

            // Flush batch if full
            if count >= MAX_PER_BATCH {
                let mesh = Mesh {
                    vertices: std::mem::take(&mut vertices),
                    indices: std::mem::take(&mut indices),
                    texture: Some(self.white_texture.clone()),
                };
                draw_mesh(&mesh);
                vertices.reserve(MAX_PER_BATCH * 4);
                indices.reserve(MAX_PER_BATCH * 6);
                count = 0;
            }
        }

        // Draw remaining
        if !vertices.is_empty() {
            let mesh = Mesh {
                vertices,
                indices,
                texture: Some(self.white_texture.clone()),
            };
            draw_mesh(&mesh);
        }
    }

    /// Render water particles with metaball effect
    pub fn draw_water(&self, particles: &Particles, screen_scale: f32) {
        // Pass 1: Render density to half-res target
        set_camera(&Camera2D {
            zoom: vec2(2.0 / self.half_w, -2.0 / self.half_h),
            target: vec2(self.half_w / 2.0, self.half_h / 2.0),
            render_target: Some(self.density_rt.clone()),
            ..Default::default()
        });

        clear_background(Color::new(0.0, 0.0, 0.0, 0.0));

        gl_use_material(&self.density_material);

        // Set water color uniform (steel blue)
        let [r, g, b, _] = sim::ParticleMaterial::Water.color();
        self.density_material.set_uniform("waterColor", [
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            1.0f32,
        ]);

        self.draw_water_density_meshes(particles, screen_scale);

        gl_use_default_material();

        // Pass 2: Horizontal blur
        set_camera(&Camera2D {
            zoom: vec2(2.0 / self.half_w, -2.0 / self.half_h),
            target: vec2(self.half_w / 2.0, self.half_h / 2.0),
            render_target: Some(self.blur_h_rt.clone()),
            ..Default::default()
        });

        clear_background(Color::new(0.0, 0.0, 0.0, 0.0));

        gl_use_material(&self.blur_h_material);
        self.blur_h_material.set_uniform("texelSize", [1.0 / self.half_w, 1.0 / self.half_h]);

        draw_texture_ex(
            &self.density_rt.texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(self.half_w, self.half_h)),
                ..Default::default()
            },
        );

        gl_use_default_material();

        // Pass 3: Vertical blur
        set_camera(&Camera2D {
            zoom: vec2(2.0 / self.half_w, -2.0 / self.half_h),
            target: vec2(self.half_w / 2.0, self.half_h / 2.0),
            render_target: Some(self.blur_v_rt.clone()),
            ..Default::default()
        });

        clear_background(Color::new(0.0, 0.0, 0.0, 0.0));

        gl_use_material(&self.blur_v_material);
        self.blur_v_material.set_uniform("texelSize", [1.0 / self.half_w, 1.0 / self.half_h]);

        draw_texture_ex(
            &self.blur_h_rt.texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(self.half_w, self.half_h)),
                ..Default::default()
            },
        );

        gl_use_default_material();

        // Pass 4: Render to screen with surface shader
        set_default_camera();

        gl_use_material(&self.surface_material);
        self.surface_material.set_uniform("threshold", self.threshold);
        self.surface_material.set_uniform("texelSize", [1.0 / self.half_w, 1.0 / self.half_h]);

        draw_texture_ex(
            &self.blur_v_rt.texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(self.screen_w, self.screen_h)),
                ..Default::default()
            },
        );

        gl_use_default_material();
    }

    /// Render all particles (water as metaballs, sediment as mesh-batched)
    pub fn draw(&self, particles: &Particles, screen_scale: f32) {
        // Draw water with metaball effect
        self.draw_water(particles, screen_scale);

        // Draw sediment on top as batched mesh quads
        self.draw_sediment(particles, screen_scale);
    }

    /// Draw sediment particles as batched mesh quads
    fn draw_sediment(&self, particles: &Particles, screen_scale: f32) {
        const MAX_PER_BATCH: usize = 8000;

        let base_size = self.particle_scale * screen_scale * 0.5;

        let mut vertices: Vec<Vertex> = Vec::with_capacity(MAX_PER_BATCH * 4);
        let mut indices: Vec<u16> = Vec::with_capacity(MAX_PER_BATCH * 6);
        let mut count = 0usize;

        for particle in particles.iter() {
            if !particle.is_sediment() {
                continue;
            }

            let x = particle.position.x * screen_scale;
            let y = particle.position.y * screen_scale;
            let [r, g, b, a] = particle.material.color();
            let color = Color::from_rgba(r, g, b, a);
            let size = base_size * particle.material.render_scale();
            let half = size / 2.0;

            let base_idx = (count * 4) as u16;

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

            indices.push(base_idx);
            indices.push(base_idx + 1);
            indices.push(base_idx + 2);
            indices.push(base_idx);
            indices.push(base_idx + 2);
            indices.push(base_idx + 3);

            count += 1;

            // Flush batch if full
            if count >= MAX_PER_BATCH {
                let mesh = Mesh {
                    vertices: std::mem::take(&mut vertices),
                    indices: std::mem::take(&mut indices),
                    texture: Some(self.white_texture.clone()),
                };
                draw_mesh(&mesh);
                vertices.reserve(MAX_PER_BATCH * 4);
                indices.reserve(MAX_PER_BATCH * 6);
                count = 0;
            }
        }

        // Draw remaining
        if !vertices.is_empty() {
            let mesh = Mesh {
                vertices,
                indices,
                texture: Some(self.white_texture.clone()),
            };
            draw_mesh(&mesh);
        }
    }
}

/// Fast batched circle renderer using macroquad's internal batching
/// Renders particles as simple filled circles with no shader overhead
/// If debug_state is true, Bedload=red, Suspended=blue (for sediment only)
pub fn draw_particles_fast(particles: &Particles, screen_scale: f32, base_size: f32) {
    draw_particles_fast_debug(particles, screen_scale, base_size, false);
}

/// Fast renderer with optional debug state coloring
/// Two-pass: water first (background), then sediment on top (foreground)
pub fn draw_particles_fast_debug(particles: &Particles, screen_scale: f32, base_size: f32, debug_state: bool) {
    // Pass 1: Draw water particles first (background)
    for particle in particles.iter() {
        if particle.is_sediment() {
            continue; // Skip sediment in first pass
        }
        let x = particle.position.x * screen_scale;
        let y = particle.position.y * screen_scale;
        let size = base_size * particle.material.render_scale();
        let [r, g, b, a] = particle.material.color();
        let color = Color::from_rgba(r, g, b, a);
        draw_circle(x, y, size, color);
    }

    // Pass 2: Draw sediment particles on top (foreground)
    for particle in particles.iter() {
        if !particle.is_sediment() {
            continue; // Skip water in second pass
        }
        let x = particle.position.x * screen_scale;
        let y = particle.position.y * screen_scale;
        let size = base_size * particle.material.render_scale();

        let color = if debug_state {
            // Debug mode: Bedload = red, Suspended = blue
            match particle.state {
                ParticleState::Bedload => Color::from_rgba(255, 50, 50, 255),
                ParticleState::Suspended => Color::from_rgba(50, 100, 255, 255),
            }
        } else {
            let [r, g, b, a] = particle.material.color();
            Color::from_rgba(r, g, b, a)
        };

        draw_circle(x, y, size, color);
    }
}

/// Ultra-fast rectangle renderer - rectangles batch better than circles
/// Two-pass: water first (background), then sediment on top (foreground)
pub fn draw_particles_rect(particles: &Particles, screen_scale: f32, base_size: f32) {
    // Pass 1: Draw water first
    for particle in particles.iter() {
        if particle.is_sediment() {
            continue;
        }
        let x = particle.position.x * screen_scale;
        let y = particle.position.y * screen_scale;
        let [r, g, b, a] = particle.material.color();
        let color = Color::from_rgba(r, g, b, a);
        let size = base_size * particle.material.render_scale();
        draw_rectangle(x - size/2.0, y - size/2.0, size, size, color);
    }

    // Pass 2: Draw sediment on top
    for particle in particles.iter() {
        if !particle.is_sediment() {
            continue;
        }
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
/// Two-pass: water first (background), then sediment on top (foreground)
pub fn draw_particles_mesh(particles: &Particles, screen_scale: f32, base_size: f32) {
    // Max particles per batch: 65536 indices / 6 indices per quad = 10922
    // Use 8000 for safety margin
    const MAX_PER_BATCH: usize = 8000;

    // Helper to draw a filtered subset of particles
    let draw_filtered = |draw_sediment: bool| {
        let mut vertices: Vec<Vertex> = Vec::with_capacity(MAX_PER_BATCH * 4);
        let mut indices: Vec<u16> = Vec::with_capacity(MAX_PER_BATCH * 6);
        let mut local_i: usize = 0;

        for particle in particles.iter() {
            // Filter: first pass = water only, second pass = sediment only
            if particle.is_sediment() != draw_sediment {
                continue;
            }

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

            local_i += 1;

            // Flush batch if full
            if local_i >= MAX_PER_BATCH {
                let mesh = Mesh {
                    vertices: std::mem::take(&mut vertices),
                    indices: std::mem::take(&mut indices),
                    texture: None,
                };
                draw_mesh(&mesh);
                vertices.reserve(MAX_PER_BATCH * 4);
                indices.reserve(MAX_PER_BATCH * 6);
                local_i = 0;
            }
        }

        // Draw remaining particles in last batch
        if !vertices.is_empty() {
            let mesh = Mesh {
                vertices,
                indices,
                texture: None,
            };
            draw_mesh(&mesh);
        }
    };

    // Pass 1: Draw water (background)
    draw_filtered(false);
    // Pass 2: Draw sediment (foreground)
    draw_filtered(true);
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

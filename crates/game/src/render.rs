//! GPU-accelerated water particle rendering
//!
//! WATER-ONLY VERSION - Simplified for pure water simulation

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
    vec2 center = vec2(0.5, 0.5);
    float dist = length(v_uv - center) * 2.0;
    float alpha = 1.0 - smoothstep(0.7, 1.0, dist);

    if (alpha < 0.01) {
        discard;
    }

    gl_FragColor = vec4(particleColor.rgb, alpha);
}
"#;

/// Metaball density pass
const METABALL_DENSITY_FRAG: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;

uniform lowp vec4 particleColor;
uniform lowp float densityMult;

void main() {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(v_uv - center) * 2.0;
    float density = exp(-dist * dist * 4.0) * densityMult;
    gl_FragColor = vec4(particleColor.rgb * density, density);
}
"#;

/// Metaball threshold pass
const METABALL_THRESHOLD_FRAG: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;

uniform sampler2D Texture;
uniform lowp float threshold;

void main() {
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec4 texSample = texture2D(Texture, uv);
    float density = texSample.a;
    if (density < threshold) {
        discard;
    }
    vec3 color = texSample.rgb / max(density, 0.001);
    float edgeFactor = smoothstep(threshold, threshold + 0.08, density);
    gl_FragColor = vec4(color, edgeFactor * 0.85);
}
"#;

/// Metaball renderer (simplified stub)
pub struct MetaballRenderer {
    particle_renderer: ParticleRenderer,
}

impl MetaballRenderer {
    pub fn new(_width: u32, _height: u32) -> Self {
        Self {
            particle_renderer: ParticleRenderer::new(),
        }
    }

    pub fn draw(&mut self, particles: &Particles, _scale: f32, base_size: f32) {
        self.particle_renderer.draw_particles_metaball(particles, base_size);
    }

    pub fn draw_water(&mut self, particles: &Particles, _scale: f32, base_size: f32) {
        self.particle_renderer.draw_particles_metaball(particles, base_size);
    }

    pub fn set_threshold(&mut self, _threshold: f32) {}
    pub fn set_scale(&mut self, _scale: f32) {}
}

/// Fast particle rendering (simple colored rectangles)
pub fn draw_particles_fast(particles: &Particles, scale: f32, size: f32) {
    let color = Color::from_rgba(50, 140, 240, 180);
    for p in particles.iter() {
        draw_rectangle(p.position.x * scale, p.position.y * scale, size, size, color);
    }
}

/// Fast particle rendering with debug colors
pub fn draw_particles_fast_debug(particles: &Particles, scale: f32, size: f32, _debug: bool) {
    draw_particles_fast(particles, scale, size);
}

/// Fast rectangle rendering
pub fn draw_particles_rect(particles: &Particles, scale: f32, size: f32) {
    draw_particles_fast(particles, scale, size);
}

/// Mesh batch rendering
pub fn draw_particles_mesh(particles: &Particles, scale: f32, size: f32) {
    draw_particles_fast(particles, scale, size);
}

pub struct ParticleRenderer {
    pub material: Material,
    pub metaball_density_material: Material,
    pub metaball_threshold_material: Material,
    pub render_target: Option<RenderTarget>,
    pub base_size: f32,
    mesh_vertices: Vec<Vertex>,
    mesh_indices: Vec<u16>,
}

impl ParticleRenderer {
    pub fn new() -> Self {
        let material = load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: FRAGMENT_SHADER,
            },
            MaterialParams {
                uniforms: vec![
                    ("particleColor".to_string(), UniformType::Float4),
                ],
                ..Default::default()
            },
        ).unwrap();

        let metaball_density_material = load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: METABALL_DENSITY_FRAG,
            },
            MaterialParams {
                uniforms: vec![
                    ("particleColor".to_string(), UniformType::Float4),
                    ("densityMult".to_string(), UniformType::Float1),
                ],
                pipeline_params: PipelineParams {
                    color_blend: Some(BlendState::new(
                        Equation::Add,
                        BlendFactor::Value(BlendValue::SourceAlpha),
                        BlendFactor::One,
                    )),
                    ..Default::default()
                },
                ..Default::default()
            },
        ).unwrap();

        let metaball_threshold_material = load_material(
            ShaderSource::Glsl {
                vertex: VERTEX_SHADER,
                fragment: METABALL_THRESHOLD_FRAG,
            },
            MaterialParams {
                uniforms: vec![
                    ("threshold".to_string(), UniformType::Float1),
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
        ).unwrap();

        Self {
            material,
            metaball_density_material,
            metaball_threshold_material,
            render_target: None,
            base_size: 3.0,
            mesh_vertices: Vec::with_capacity(4000),
            mesh_indices: Vec::with_capacity(6000),
        }
    }

    fn ensure_render_target(&mut self) {
        let sw = screen_width() as u32;
        let sh = screen_height() as u32;

        let needs_new = match &self.render_target {
            None => true,
            Some(rt) => rt.texture.width() != sw as f32 || rt.texture.height() != sh as f32,
        };

        if needs_new {
            self.render_target = Some(render_target(sw, sh));
        }
    }

    /// Simple particle rendering - one draw call for all water
    pub fn draw_particles(&mut self, particles: &Particles, base_size: f32) {
        if particles.is_empty() {
            return;
        }

        // Water color: blue
        let color = [50.0 / 255.0, 140.0 / 255.0, 240.0 / 255.0, 0.7];

        self.mesh_vertices.clear();
        self.mesh_indices.clear();

        for particle in particles.iter() {
            let x = particle.position.x;
            let y = particle.position.y;
            let half = base_size * 0.5;

            let base_idx = self.mesh_vertices.len() as u16;

            self.mesh_vertices.push(Vertex {
                position: Vec3::new(x - half, y - half, 0.0),
                uv: Vec2::new(0.0, 0.0),
                color: WHITE,
            });
            self.mesh_vertices.push(Vertex {
                position: Vec3::new(x + half, y - half, 0.0),
                uv: Vec2::new(1.0, 0.0),
                color: WHITE,
            });
            self.mesh_vertices.push(Vertex {
                position: Vec3::new(x + half, y + half, 0.0),
                uv: Vec2::new(1.0, 1.0),
                color: WHITE,
            });
            self.mesh_vertices.push(Vertex {
                position: Vec3::new(x - half, y + half, 0.0),
                uv: Vec2::new(0.0, 1.0),
                color: WHITE,
            });

            self.mesh_indices.push(base_idx);
            self.mesh_indices.push(base_idx + 1);
            self.mesh_indices.push(base_idx + 2);
            self.mesh_indices.push(base_idx);
            self.mesh_indices.push(base_idx + 2);
            self.mesh_indices.push(base_idx + 3);
        }

        let mesh = Mesh {
            vertices: self.mesh_vertices.clone(),
            indices: self.mesh_indices.clone(),
            texture: None,
        };

        gl_use_material(&self.material);
        self.material.set_uniform("particleColor", color);
        draw_mesh(&mesh);
        gl_use_default_material();
    }

    /// Metaball rendering for cohesive water appearance
    pub fn draw_particles_metaball(&mut self, particles: &Particles, base_size: f32) {
        if particles.is_empty() {
            return;
        }

        self.ensure_render_target();
        let rt = self.render_target.as_ref().unwrap();

        // Pass 1: Accumulate density to render target
        set_camera(&Camera2D {
            render_target: Some(rt.clone()),
            ..Camera2D::from_display_rect(Rect::new(0.0, 0.0, screen_width(), screen_height()))
        });

        clear_background(Color::new(0.0, 0.0, 0.0, 0.0));

        // Water color
        let color = [50.0 / 255.0, 140.0 / 255.0, 240.0 / 255.0, 1.0];
        let density_mult = 0.035;
        let particle_size = base_size * 2.5;

        self.mesh_vertices.clear();
        self.mesh_indices.clear();

        for particle in particles.iter() {
            let x = particle.position.x;
            let y = particle.position.y;
            let half = particle_size * 0.5;

            let base_idx = self.mesh_vertices.len() as u16;

            self.mesh_vertices.push(Vertex {
                position: Vec3::new(x - half, y - half, 0.0),
                uv: Vec2::new(0.0, 0.0),
                color: WHITE,
            });
            self.mesh_vertices.push(Vertex {
                position: Vec3::new(x + half, y - half, 0.0),
                uv: Vec2::new(1.0, 0.0),
                color: WHITE,
            });
            self.mesh_vertices.push(Vertex {
                position: Vec3::new(x + half, y + half, 0.0),
                uv: Vec2::new(1.0, 1.0),
                color: WHITE,
            });
            self.mesh_vertices.push(Vertex {
                position: Vec3::new(x - half, y + half, 0.0),
                uv: Vec2::new(0.0, 1.0),
                color: WHITE,
            });

            self.mesh_indices.push(base_idx);
            self.mesh_indices.push(base_idx + 1);
            self.mesh_indices.push(base_idx + 2);
            self.mesh_indices.push(base_idx);
            self.mesh_indices.push(base_idx + 2);
            self.mesh_indices.push(base_idx + 3);
        }

        let mesh = Mesh {
            vertices: self.mesh_vertices.clone(),
            indices: self.mesh_indices.clone(),
            texture: None,
        };

        gl_use_material(&self.metaball_density_material);
        self.metaball_density_material.set_uniform("particleColor", color);
        self.metaball_density_material.set_uniform("densityMult", density_mult);
        draw_mesh(&mesh);
        gl_use_default_material();

        // Pass 2: Threshold and render to screen
        set_default_camera();

        gl_use_material(&self.metaball_threshold_material);
        self.metaball_threshold_material.set_uniform("threshold", 0.5f32);
        draw_texture_ex(
            &rt.texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(Vec2::new(screen_width(), screen_height())),
                flip_y: true,
                ..Default::default()
            },
        );
        gl_use_default_material();
    }
}

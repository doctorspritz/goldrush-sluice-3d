//! GPU-accelerated particle rendering
//!
//! Renders particles as smooth circles with velocity-based coloring.
//! Uses macroquad's material system for custom GLSL shaders.

use macroquad::prelude::*;
use macroquad::miniquad::{BlendState, Equation, BlendFactor, BlendValue};
use sim::Particles;

/// Vertex shader for instanced particle circles
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

// Custom uniform for particle color
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
    
    // Use the particle color uniform
    vec3 finalColor = particleColor.rgb;
    
    // Subtle rim darkening for depth
    float rim = smoothstep(0.5, 0.9, dist);
    finalColor = mix(finalColor, finalColor * 0.8, rim);
    
    gl_FragColor = vec4(finalColor, alpha * 0.85);
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
            particle_scale: 4.0, // Visible particle size
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

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
attribute vec4 color0;

// Per-quad instance data (we'll encode in color0)
varying lowp vec2 v_uv;
varying lowp vec4 v_color;
varying lowp float v_velocity;

uniform mat4 Model;
uniform mat4 Projection;

void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    v_uv = texcoord;
    v_color = color0;
    // Velocity encoded in alpha (we'll decode in fragment)
    v_velocity = color0.a;
}
"#;

/// Fragment shader for smooth circles with velocity coloring
const FRAGMENT_SHADER: &str = r#"#version 100
precision mediump float;

varying lowp vec2 v_uv;
varying lowp vec4 v_color;
varying lowp float v_velocity;

uniform float velocityMax;
uniform float particleScale;

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
    
    // Base color from material (RGB stored in v_color)
    vec3 baseColor = v_color.rgb;
    
    // Velocity-based color shift (faster = brighter/whiter)
    float velFactor = clamp(v_velocity / velocityMax, 0.0, 1.0);
    vec3 velocityColor = mix(baseColor, vec3(1.0), velFactor * 0.4);
    
    // Add subtle rim lighting for depth
    float rim = smoothstep(0.5, 0.9, dist);
    vec3 finalColor = mix(velocityColor, velocityColor * 0.7, rim);
    
    gl_FragColor = vec4(finalColor, alpha * 0.9);
}
"#;

/// Particle renderer with GPU-accelerated instanced circles
pub struct ParticleRenderer {
    material: Material,
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
                    UniformDesc::new("velocityMax", UniformType::Float1),
                    UniformDesc::new("particleScale", UniformType::Float1),
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

        Self {
            material,
            velocity_max: 100.0,
            particle_scale: 4.0,
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
        // Set uniforms
        self.material.set_uniform("velocityMax", self.velocity_max);
        self.material.set_uniform("particleScale", self.particle_scale);

        gl_use_material(&self.material);

        let size = self.particle_scale * screen_scale;

        // Draw each particle as a textured quad (circle rendered in shader)
        for particle in particles.iter() {
            let x = particle.position.x * screen_scale;
            let y = particle.position.y * screen_scale;
            
            // Encode velocity magnitude in alpha for shader
            let vel_mag = particle.velocity.length();
            let [r, g, b, _] = particle.material.color();
            
            // Normalize color to 0-1 and encode velocity in alpha
            let color = Color::new(
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                vel_mag, // Raw velocity, will be normalized in shader
            );

            // Draw quad (shader makes it circular)
            draw_rectangle(x - size / 2.0, y - size / 2.0, size, size, color);
        }

        gl_use_default_material();
    }

    /// Draw particles sorted by density (heaviest on top)
    pub fn draw_sorted(&self, particles: &Particles, screen_scale: f32) {
        // Set uniforms
        self.material.set_uniform("velocityMax", self.velocity_max);
        self.material.set_uniform("particleScale", self.particle_scale);

        gl_use_material(&self.material);

        let size = self.particle_scale * screen_scale;

        // Sort by density (lighter first, heavier on top)
        let mut indices: Vec<usize> = (0..particles.list.len()).collect();
        indices.sort_by(|&a, &b| {
            let da = particles.list[a].density();
            let db = particles.list[b].density();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        for idx in indices {
            let particle = &particles.list[idx];
            let x = particle.position.x * screen_scale;
            let y = particle.position.y * screen_scale;
            
            let vel_mag = particle.velocity.length();
            let [r, g, b, _] = particle.material.color();
            
            let color = Color::new(
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                vel_mag,
            );

            draw_rectangle(x - size / 2.0, y - size / 2.0, size, size, color);
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

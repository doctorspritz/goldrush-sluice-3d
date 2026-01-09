# Screen-Space Fluid Rendering

> **Created:** 2026-01-09
> **Priority:** Low (exploration/enhancement)
> **Status:** TODO

---

## Motivation

Current heightfield water can't render:
- Splashes and spray
- Overhanging water (waterfalls)
- 3D FLIP particles from detail zones

Screen-space fluid rendering provides smooth water surfaces directly from particles.

---

## Performance Analysis

| Technique | Vertex Cost | Pixel Cost | Quality | Splashes |
|-----------|-------------|------------|---------|----------|
| Heightfield (current) | O(grid²) | Simple | Good | ❌ |
| Screen-space fluid | O(particles) | 3-4 passes | Good | ✅ |
| Marching cubes | O(grid³) | Simple | Great | ✅ |

**Verdict:** Comparable to heightfield. Main cost is fullscreen blur passes.

---

## Algorithm Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Pass 1: DEPTH SPLAT                                         │
│                                                             │
│ For each particle:                                          │
│   - Render as point sprite (size = particle_radius * 2)     │
│   - Write sphere depth: z_out = z_center - sqrt(r² - uv²)   │
│   - Use depth test, write nearest                           │
│                                                             │
│ Output: depth_texture (R32F)                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 2: BILATERAL BLUR (Horizontal)                         │
│                                                             │
│ Gaussian blur with depth-aware rejection:                   │
│   weight = gaussian(offset) * edge_stop(depth_diff)         │
│                                                             │
│ Preserves edges, smooths surface                            │
│                                                             │
│ Output: blurred_depth_h (R32F)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 3: BILATERAL BLUR (Vertical)                           │
│                                                             │
│ Same as horizontal, on blurred_depth_h                      │
│                                                             │
│ Output: blurred_depth (R32F)                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Pass 4: COMPOSE                                             │
│                                                             │
│ 1. Reconstruct view-space position from depth               │
│ 2. Compute normal from depth gradient (Sobel)               │
│ 3. Apply Fresnel, reflection, refraction                    │
│ 4. Blend with scene                                         │
│                                                             │
│ Output: final_color (RGBA8)                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Shader Specifications

### Pass 1: `fluid_depth_3d.wgsl`

**Purpose:** Render particles as spherical splats to depth buffer

```wgsl
// Vertex: billboard quad per particle
@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32,
           @builtin(instance_index) i_idx: u32) -> VertexOutput {
    let pos = positions[i_idx].xyz;
    let radius = uniforms.particle_radius;
    
    // Billboard corners
    let uv = billboard_offsets[v_idx];  // [-1,1] quad
    let view_pos = (uniforms.view * vec4(pos, 1.0)).xyz;
    let offset_view = view_pos + vec3(uv * radius, 0.0);
    
    var out: VertexOutput;
    out.clip_position = uniforms.proj * vec4(offset_view, 1.0);
    out.uv = uv;
    out.view_z = offset_view.z;
    out.radius = radius;
    return out;
}

// Fragment: spherical depth
@fragment
fn fs_main(in: VertexOutput) -> @builtin(frag_depth) f32 {
    let dist_sq = dot(in.uv, in.uv);
    if (dist_sq > 1.0) { discard; }
    
    // Sphere depth offset
    let sphere_z = sqrt(1.0 - dist_sq) * in.radius;
    let depth = in.view_z - sphere_z;
    
    // Convert to NDC depth
    return (uniforms.proj[2][2] * depth + uniforms.proj[3][2]) / -depth;
}
```

### Pass 2-3: `fluid_blur_3d.wgsl`

**Purpose:** Bilateral Gaussian blur preserving edges

```wgsl
const KERNEL_SIZE: i32 = 7;
const SIGMA_SPACE: f32 = 3.0;
const SIGMA_DEPTH: f32 = 0.1;  // Depth threshold for edge preservation

@fragment
fn fs_blur(in: FullscreenInput) -> @location(0) f32 {
    let center_depth = textureSample(depth_tex, samp, in.uv).r;
    if (center_depth >= 1.0) { return 1.0; }  // Sky
    
    var sum = 0.0;
    var weight_sum = 0.0;
    
    for (var i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
        let offset = vec2(f32(i) * uniforms.texel_size.x, 0.0);  // Horizontal
        // let offset = vec2(0.0, f32(i) * uniforms.texel_size.y);  // Vertical
        
        let sample_uv = in.uv + offset;
        let sample_depth = textureSample(depth_tex, samp, sample_uv).r;
        
        // Gaussian spatial weight
        let spatial_weight = exp(-f32(i * i) / (2.0 * SIGMA_SPACE * SIGMA_SPACE));
        
        // Depth-aware edge preservation
        let depth_diff = abs(sample_depth - center_depth);
        let depth_weight = exp(-depth_diff * depth_diff / (2.0 * SIGMA_DEPTH * SIGMA_DEPTH));
        
        let weight = spatial_weight * depth_weight;
        sum += sample_depth * weight;
        weight_sum += weight;
    }
    
    return sum / weight_sum;
}
```

### Pass 4: `fluid_compose_3d.wgsl`

**Purpose:** Reconstruct surface, compute normals, shade

```wgsl
fn reconstruct_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
    let view_pos = uniforms.inv_proj * ndc;
    return view_pos.xyz / view_pos.w;
}

fn compute_normal(uv: vec2<f32>) -> vec3<f32> {
    let texel = uniforms.texel_size;
    
    let depth_c = textureSample(depth_tex, samp, uv).r;
    let depth_l = textureSample(depth_tex, samp, uv - vec2(texel.x, 0.0)).r;
    let depth_r = textureSample(depth_tex, samp, uv + vec2(texel.x, 0.0)).r;
    let depth_u = textureSample(depth_tex, samp, uv - vec2(0.0, texel.y)).r;
    let depth_d = textureSample(depth_tex, samp, uv + vec2(0.0, texel.y)).r;
    
    let pos_c = reconstruct_view_pos(uv, depth_c);
    let pos_l = reconstruct_view_pos(uv - vec2(texel.x, 0.0), depth_l);
    let pos_r = reconstruct_view_pos(uv + vec2(texel.x, 0.0), depth_r);
    let pos_u = reconstruct_view_pos(uv - vec2(0.0, texel.y), depth_u);
    let pos_d = reconstruct_view_pos(uv + vec2(0.0, texel.y), depth_d);
    
    let ddx = pos_r - pos_l;
    let ddy = pos_d - pos_u;
    
    return normalize(cross(ddy, ddx));
}

@fragment
fn fs_compose(in: FullscreenInput) -> @location(0) vec4<f32> {
    let depth = textureSample(depth_tex, samp, in.uv).r;
    if (depth >= 1.0) { discard; }
    
    let normal = compute_normal(in.uv);
    let view_pos = reconstruct_view_pos(in.uv, depth);
    let view_dir = normalize(-view_pos);
    
    // Fresnel
    let NdotV = max(dot(normal, view_dir), 0.0);
    let fresnel = pow(1.0 - NdotV, 4.0);
    
    // Lighting
    let light_dir = normalize(vec3(0.5, 1.0, 0.3));
    let half_dir = normalize(light_dir + view_dir);
    let specular = pow(max(dot(normal, half_dir), 0.0), 64.0);
    
    // Color
    let water_color = vec3(0.1, 0.3, 0.6);
    let sky_color = vec3(0.8, 0.9, 1.0);
    let final_color = mix(water_color, sky_color, fresnel * 0.6) + specular * 0.5;
    
    return vec4(final_color, 0.8 + fresnel * 0.2);
}
```

---

## Rust Integration

### New Structs

```rust
pub struct ScreenSpaceFluidRenderer {
    depth_texture: wgpu::Texture,
    blur_texture_h: wgpu::Texture,
    blur_texture_v: wgpu::Texture,
    
    depth_pipeline: wgpu::RenderPipeline,
    blur_h_pipeline: wgpu::RenderPipeline,
    blur_v_pipeline: wgpu::RenderPipeline,
    compose_pipeline: wgpu::RenderPipeline,
    
    uniforms: FluidRenderUniforms,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FluidRenderUniforms {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    texel_size: [f32; 2],
    particle_radius: f32,
    blur_depth_falloff: f32,
}
```

### Render Loop

```rust
impl ScreenSpaceFluidRenderer {
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, 
                  particle_positions: &wgpu::Buffer,
                  particle_count: u32,
                  output: &wgpu::TextureView) {
        // Pass 1: Depth splat
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                depth_stencil_attachment: Some(/* depth_texture */),
                ..
            });
            pass.set_pipeline(&self.depth_pipeline);
            pass.draw(0..4, 0..particle_count);  // 4 verts per billboard
        }
        
        // Pass 2: Horizontal blur
        {
            let mut pass = encoder.begin_render_pass(/* blur_texture_h */);
            pass.set_pipeline(&self.blur_h_pipeline);
            pass.draw(0..3, 0..1);  // Fullscreen triangle
        }
        
        // Pass 3: Vertical blur
        {
            let mut pass = encoder.begin_render_pass(/* blur_texture_v */);
            pass.set_pipeline(&self.blur_v_pipeline);
            pass.draw(0..3, 0..1);
        }
        
        // Pass 4: Compose
        {
            let mut pass = encoder.begin_render_pass(/* output */);
            pass.set_pipeline(&self.compose_pipeline);
            pass.draw(0..3, 0..1);
        }
    }
}
```

---

## Implementation Checklist

### Phase 1: Basic Pipeline
- [ ] Create `fluid_depth_3d.wgsl` with sphere splat
- [ ] Create depth-only render target (R32F)
- [ ] Test: particles render as smooth depth blobs

### Phase 2: Blur
- [ ] Create `fluid_blur_3d.wgsl` with bilateral filter
- [ ] Create intermediate blur textures
- [ ] Test: edges preserved, surface smoothed

### Phase 3: Shading  
- [ ] Create `fluid_compose_3d.wgsl` with normal reconstruction
- [ ] Add Fresnel, specular, color
- [ ] Test: looks like water surface

### Phase 4: Integration
- [ ] Create `ScreenSpaceFluidRenderer` struct
- [ ] Add toggle to switch heightfield ↔ screen-space
- [ ] Profile performance comparison

---

## References

- [GPU Gems 3 Ch.30: Real-Time Fluid Rendering](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids)
- [Simon Green - Screen Space Fluid Rendering (GDC 2010)](https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf)
- [blub fluid renderer (Rust reference)](https://github.com/Wumpf/blub)

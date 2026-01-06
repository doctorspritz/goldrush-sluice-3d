# Dig Test MVP - Implementation Plan

## Goal

Create a minimal 3D excavation demo: click on terrain to dig, spawning particles that fall into water.

## Success Criteria

- [ ] Heightfield terrain renders as colored grid
- [ ] Mouse click digs a hole (lowers height)
- [ ] Particles spawn from dug material
- [ ] Particles fall and interact with 3D FLIP water
- [ ] Can see the dig → fall → splash loop working

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  dig_test.rs (example)                                  │
│  ├── Heightfield (terrain storage + dig logic)          │
│  ├── FlipSimulation3D (existing 3D water)               │
│  ├── HeightfieldRenderer (new - renders terrain grid)   │
│  └── ParticleRenderer (existing - renders particles)    │
└─────────────────────────────────────────────────────────┘
```

## Files to Create/Modify

### 1. NEW: `crates/sim3d/src/heightfield.rs`

Simple heightfield terrain with dig operation.

```rust
use glam::Vec3;

/// Simple heightfield terrain for excavation
pub struct Heightfield {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,
    pub heights: Vec<f32>,
}

impl Heightfield {
    /// Create flat terrain at given height
    pub fn new(width: usize, depth: usize, cell_size: f32, initial_height: f32) -> Self {
        Self {
            width,
            depth,
            cell_size,
            heights: vec![initial_height; width * depth],
        }
    }

    /// Get height at grid position
    pub fn get_height(&self, x: usize, z: usize) -> f32 {
        if x < self.width && z < self.depth {
            self.heights[z * self.width + x]
        } else {
            0.0
        }
    }

    /// Set height at grid position
    pub fn set_height(&mut self, x: usize, z: usize, height: f32) {
        if x < self.width && z < self.depth {
            self.heights[z * self.width + x] = height;
        }
    }

    /// World bounds
    pub fn world_size(&self) -> Vec3 {
        Vec3::new(
            self.width as f32 * self.cell_size,
            1.0, // arbitrary Y
            self.depth as f32 * self.cell_size,
        )
    }

    /// Dig at world position, returns positions to spawn particles
    ///
    /// Parameters:
    /// - world_x, world_z: world position to dig at
    /// - radius: dig radius in world units
    /// - dig_depth: how much to lower the terrain
    ///
    /// Returns: Vec of world positions where particles should spawn
    pub fn dig(&mut self, world_x: f32, world_z: f32, radius: f32, dig_depth: f32) -> Vec<Vec3> {
        let cx = (world_x / self.cell_size) as i32;
        let cz = (world_z / self.cell_size) as i32;
        let r = (radius / self.cell_size).ceil() as i32;

        let mut spawn_positions = Vec::new();

        for dz in -r..=r {
            for dx in -r..=r {
                let x = cx + dx;
                let z = cz + dz;

                // Bounds check
                if x < 0 || x >= self.width as i32 || z < 0 || z >= self.depth as i32 {
                    continue;
                }

                // Distance check (circular dig)
                let dist_sq = (dx * dx + dz * dz) as f32;
                let r_sq = (radius / self.cell_size).powi(2);
                if dist_sq > r_sq {
                    continue;
                }

                let idx = z as usize * self.width + x as usize;
                let old_h = self.heights[idx];
                let new_h = (old_h - dig_depth).max(0.0);

                if new_h < old_h {
                    self.heights[idx] = new_h;

                    // Spawn particle at center of cell, slightly above old surface
                    let world_pos = Vec3::new(
                        (x as f32 + 0.5) * self.cell_size,
                        old_h + 0.05,
                        (z as f32 + 0.5) * self.cell_size,
                    );
                    spawn_positions.push(world_pos);
                }
            }
        }

        spawn_positions
    }

    /// Simple raycast - intersect ray with heightfield
    /// Returns hit position if ray intersects terrain
    pub fn raycast(&self, origin: Vec3, direction: Vec3) -> Option<Vec3> {
        // Normalize direction
        let dir = direction.normalize();

        // Step along ray, checking height at each XZ position
        // Simple approach: march in small steps
        let step_size = self.cell_size * 0.5;
        let max_dist = 100.0;

        let mut t = 0.0;
        while t < max_dist {
            let p = origin + dir * t;

            // Check if we're within bounds
            if p.x >= 0.0 && p.x < self.world_size().x
               && p.z >= 0.0 && p.z < self.world_size().z {

                // Get terrain height at this XZ
                let gx = (p.x / self.cell_size) as usize;
                let gz = (p.z / self.cell_size) as usize;
                let terrain_h = self.get_height(gx, gz);

                // Check if ray is at or below terrain
                if p.y <= terrain_h {
                    return Some(p);
                }
            }

            t += step_size;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heightfield_creation() {
        let hf = Heightfield::new(10, 10, 0.1, 0.5);
        assert_eq!(hf.width, 10);
        assert_eq!(hf.depth, 10);
        assert_eq!(hf.get_height(5, 5), 0.5);
    }

    #[test]
    fn test_dig() {
        let mut hf = Heightfield::new(10, 10, 0.1, 0.5);
        let spawns = hf.dig(0.5, 0.5, 0.15, 0.1);

        // Should have spawned some particles
        assert!(!spawns.is_empty());

        // Height should have decreased
        assert!(hf.get_height(5, 5) < 0.5);
    }
}
```

### 2. MODIFY: `crates/sim3d/src/lib.rs`

Add heightfield module export.

```rust
// Add near top of file:
pub mod heightfield;
pub use heightfield::Heightfield;
```

### 3. NEW: `crates/game/examples/dig_test.rs`

Main example file - combines heightfield, water, and rendering.

**Structure:**
```rust
//! Dig Test - 3D Excavation MVP
//!
//! Click on terrain to dig holes. Particles fall into water.
//!
//! Controls:
//! - LEFT CLICK: Dig at cursor
//! - LEFT/RIGHT: Rotate camera
//! - UP/DOWN: Zoom
//! - SPACE: Pause
//! - R: Reset
//!
//! Run: cargo run --example dig_test --release

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use sim3d::{FlipSimulation3D, Heightfield};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Configuration
const GRID_SIZE: usize = 32;
const CELL_SIZE: f32 = 0.1;
const TERRAIN_HEIGHT: f32 = 0.3;  // Initial terrain height (above water)
const WATER_LEVEL: f32 = 0.15;    // Water fills bottom portion
const MAX_PARTICLES: usize = 50000;
const DIG_RADIUS: f32 = 0.15;
const DIG_DEPTH: f32 = 0.05;
const SEDIMENT_DENSITY: f32 = 2.5;  // Heavier than water (1.0)

// ... (implement App struct similar to dam_break_3d_visual.rs)
```

**Key Components to Implement:**

1. **App struct** - holds simulation state:
   ```rust
   struct App {
       window: Option<Arc<Window>>,
       gpu: Option<GpuState>,
       sim: FlipSimulation3D,
       heightfield: Heightfield,
       paused: bool,
       camera_angle: f32,
       camera_distance: f32,
       camera_height: f32,
       mouse_pos: (f32, f32),
       window_size: (u32, u32),
   }
   ```

2. **Initialization** - set up water and terrain:
   ```rust
   fn new() -> Self {
       // Create 3D FLIP simulation
       let mut sim = FlipSimulation3D::new(GRID_SIZE, GRID_SIZE, GRID_SIZE, CELL_SIZE);
       sim.gravity = Vec3::new(0.0, -9.8, 0.0);

       // Create heightfield terrain (sits on top of water)
       let heightfield = Heightfield::new(GRID_SIZE, GRID_SIZE, CELL_SIZE, TERRAIN_HEIGHT);

       // Spawn water below terrain level
       // Fill cells from y=1 to y=water_level_cells
       let water_top = (WATER_LEVEL / CELL_SIZE) as usize;
       for i in 1..(GRID_SIZE - 1) {
           for j in 1..water_top {
               for k in 1..(GRID_SIZE - 1) {
                   // 2 particles per cell
                   for pi in 0..2 {
                       for pk in 0..2 {
                           let pos = Vec3::new(
                               (i as f32 + 0.25 + pi as f32 * 0.5) * CELL_SIZE,
                               (j as f32 + 0.25) * CELL_SIZE,
                               (k as f32 + 0.25 + pk as f32 * 0.5) * CELL_SIZE,
                           );
                           sim.spawn_particle(pos);
                       }
                   }
               }
           }
       }

       // ... rest of init
   }
   ```

3. **Mouse click handling** - dig and spawn:
   ```rust
   fn handle_click(&mut self, x: f32, y: f32) {
       // Convert screen coords to ray
       let ray_origin = self.camera_position();
       let ray_dir = self.screen_to_world_ray(x, y);

       // Raycast to heightfield
       if let Some(hit) = self.heightfield.raycast(ray_origin, ray_dir) {
           // Dig and get spawn positions
           let spawns = self.heightfield.dig(hit.x, hit.z, DIG_RADIUS, DIG_DEPTH);

           // Spawn sediment particles in FLIP simulation
           for pos in spawns {
               // Add small random velocity for visual variety
               let vel = Vec3::new(
                   (rand::random::<f32>() - 0.5) * 0.5,
                   -0.5,  // slight downward
                   (rand::random::<f32>() - 0.5) * 0.5,
               );
               self.sim.spawn_sediment(pos, vel, SEDIMENT_DENSITY);
           }

           println!("Dug at ({:.2}, {:.2}), spawned {} particles",
                    hit.x, hit.z, spawns.len());
       }
   }
   ```

4. **Camera ray calculation**:
   ```rust
   fn camera_position(&self) -> Vec3 {
       let center = Vec3::splat(GRID_SIZE as f32 * CELL_SIZE * 0.5);
       Vec3::new(
           center.x + self.camera_angle.cos() * self.camera_distance,
           self.camera_height,
           center.z + self.camera_angle.sin() * self.camera_distance,
       )
   }

   fn screen_to_world_ray(&self, screen_x: f32, screen_y: f32) -> Vec3 {
       // Normalized device coordinates
       let ndc_x = (2.0 * screen_x / self.window_size.0 as f32) - 1.0;
       let ndc_y = 1.0 - (2.0 * screen_y / self.window_size.1 as f32);

       // Inverse view-projection
       let view = self.view_matrix();
       let proj = self.projection_matrix();
       let inv_vp = (proj * view).inverse();

       // Unproject
       let near = inv_vp * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
       let far = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

       let near = near.truncate() / near.w;
       let far = far.truncate() / far.w;

       (far - near).normalize()
   }
   ```

5. **Heightfield rendering** - simple colored quads:
   - Create vertex buffer with quad per cell
   - Color based on height (brown gradient)
   - Update buffer when heights change

### 4. Heightfield Renderer (in dig_test.rs or separate file)

Render heightfield as grid of colored quads:

```rust
struct HeightfieldRenderer {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    num_indices: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TerrainVertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl HeightfieldRenderer {
    fn new(device: &wgpu::Device, heightfield: &Heightfield, ...) -> Self {
        // Generate vertices for each cell as a quad
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for z in 0..heightfield.depth {
            for x in 0..heightfield.width {
                let h = heightfield.get_height(x, z);
                let base_idx = vertices.len() as u32;

                // Height-based color (darker = lower)
                let color_factor = h / TERRAIN_HEIGHT;
                let color = [
                    0.4 + 0.2 * color_factor,  // R (brown)
                    0.25 + 0.15 * color_factor, // G
                    0.1 + 0.1 * color_factor,   // B
                ];

                let x0 = x as f32 * heightfield.cell_size;
                let x1 = (x + 1) as f32 * heightfield.cell_size;
                let z0 = z as f32 * heightfield.cell_size;
                let z1 = (z + 1) as f32 * heightfield.cell_size;

                // Four corners of quad (all at same height for now)
                vertices.push(TerrainVertex { position: [x0, h, z0], color });
                vertices.push(TerrainVertex { position: [x1, h, z0], color });
                vertices.push(TerrainVertex { position: [x1, h, z1], color });
                vertices.push(TerrainVertex { position: [x0, h, z1], color });

                // Two triangles
                indices.extend_from_slice(&[
                    base_idx, base_idx + 1, base_idx + 2,
                    base_idx, base_idx + 2, base_idx + 3,
                ]);
            }
        }

        // Create buffers...
    }

    fn update(&mut self, queue: &wgpu::Queue, heightfield: &Heightfield) {
        // Regenerate vertex data and upload
        // (Could optimize to only update changed cells)
    }
}
```

### 5. Shader for Terrain (terrain.wgsl)

Simple shader for colored terrain quads:

```wgsl
struct Uniforms {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
```

## Implementation Steps

### Step 1: Heightfield Module
1. Create `crates/sim3d/src/heightfield.rs` with the Heightfield struct
2. Add `pub mod heightfield;` to `crates/sim3d/src/lib.rs`
3. Run `cargo test -p sim3d` to verify tests pass

### Step 2: Basic Example Structure
1. Create `crates/game/examples/dig_test.rs`
2. Copy structure from `dam_break_3d_visual.rs`
3. Add Heightfield to App struct
4. Modify init to create terrain instead of just water
5. Verify it compiles: `cargo build --example dig_test`

### Step 3: Water Setup
1. Spawn water particles below terrain level
2. Verify water simulation runs: `cargo run --example dig_test --release`
3. Should see water sitting at bottom of simulation box

### Step 4: Heightfield Rendering
1. Add HeightfieldRenderer struct
2. Create terrain shader (terrain.wgsl or inline)
3. Add terrain rendering to draw loop
4. Verify terrain appears as brown grid above water

### Step 5: Mouse Input
1. Track mouse position in WindowEvent handler
2. Handle MouseButton::Left click
3. Implement screen_to_world_ray()
4. Implement raycast to heightfield

### Step 6: Dig Action
1. On click, call heightfield.dig()
2. Spawn sediment particles at returned positions
3. Update heightfield renderer with new heights
4. Verify digging works and particles fall

### Step 7: Polish
1. Add visual feedback (cursor indicator on terrain)
2. Add particle count display
3. Test dig → fall → splash loop
4. Adjust parameters (dig radius, sediment density) for good feel

## Testing Checklist

- [ ] `cargo test -p sim3d` passes
- [ ] `cargo build --example dig_test` compiles
- [ ] Example runs without crash
- [ ] Water renders and simulates correctly
- [ ] Terrain renders as colored grid
- [ ] Mouse click is detected
- [ ] Clicking on terrain creates hole
- [ ] Particles spawn from dig
- [ ] Particles fall due to gravity
- [ ] Particles interact with water (splash/settle)
- [ ] Can dig multiple holes
- [ ] Performance is acceptable (~60 FPS)

## Notes

- Keep it simple - no collapse physics yet, just height lowering
- Sediment particles use existing FLIP, no separate DEM needed for MVP
- Heightfield renderer can be basic - polish later
- If raycast is tricky, start with flat Y=0 intersection

## References

- `crates/game/examples/dam_break_3d_visual.rs` - 3D rendering template
- `crates/sim3d/src/lib.rs` - FlipSimulation3D API
- `crates/sim3d/src/particle.rs` - Particle3D, spawn_sediment

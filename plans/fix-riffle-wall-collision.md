# Fix Water Flowing Over Riffle Walls

**Type:** fix
**Priority:** P1
**Branch:** feature/sluice-3d-walls

## Problem Statement

Water particles "magically" flow over riffle walls without properly colliding with the vertical faces. Instead of building up volume behind riffles (pooling), particles slide along the bottom and climb the sides unrealistically.

### Current Behavior
- Particles flow downstream along floor
- When reaching a riffle, particles climb the vertical face
- No pooling or bounce-back occurs
- Water depth never builds

### Expected Behavior
- Particles collide with vertical riffle faces
- Horizontal velocity is zeroed/reflected at riffle walls
- Water pools behind riffles, building volume
- Particles can only flow over riffles when water depth exceeds riffle height

## Root Cause Analysis

### 1. GPU Boundary Conditions Don't Know About Riffles

**File:** `crates/game/src/gpu/shaders/enforce_bc_3d.wgsl`

The GPU boundary condition shader only handles:
- Domain boundaries (floor at y=0, walls at x/z limits)
- Solid cells marked in `cell_types` buffer

It does NOT receive or process `floor_heights` data, so it cannot enforce riffle walls.

```wgsl
// Current: Only checks domain boundaries
if (pos.y < 0.0) {
    vel.y = max(0.0, vel.y);  // Floor bounce
}
// Missing: Check floor_heights[i,k] for riffles
```

### 2. CPU Collision Happens Too Late

**File:** `crates/game/examples/sluice_riffles.rs`

The simulation pipeline is:
1. GPU: P2G transfer
2. GPU: Pressure solve
3. GPU: Enforce BC (no riffle awareness)
4. GPU: G2P transfer
5. CPU: Advection + collision ← **TOO LATE**

By the time CPU collision runs, velocities have already been interpolated from grid cells that include "inside riffle" velocities.

### 3. G2P Samples Across Solid Boundaries

**File:** `crates/game/src/gpu/shaders/g2p_3d.wgsl`

When a particle is near a riffle wall, G2P interpolates velocities from grid cells on BOTH sides of the wall. This causes particles to "see" velocity from inside the riffle, pulling them through.

### 4. Position Calculation Bug

**File:** `crates/game/examples/sluice_riffles.rs:~505`

```rust
// Bug: Pushes to wrong position
pos.x = (i_clamped as f32) * CELL_SIZE - 0.01;
```

This uses the NEW cell index, not the transition point. Particles should be pushed back to just before the riffle face.

## Proposed Solutions

### Solution A: GPU Floor Heights Buffer (Recommended)

**Effort:** Medium
**Risk:** Low
**Pros:** Proper physics, efficient, catches collisions at right time
**Cons:** Requires shader changes, buffer upload

#### Implementation Steps:

1. **Create floor_heights GPU buffer**
   ```rust
   // In GpuFlip3D or similar
   floor_heights_buffer: wgpu::Buffer,  // [u32; GRID_WIDTH * GRID_DEPTH]
   ```

2. **Upload floor_heights to GPU**
   ```rust
   fn upload_floor_heights(&self, queue: &Queue, heights: &[u32]) {
       queue.write_buffer(&self.floor_heights_buffer, 0, bytemuck::cast_slice(heights));
   }
   ```

3. **Modify enforce_bc_3d.wgsl**
   ```wgsl
   @group(0) @binding(N) var<storage, read> floor_heights: array<u32>;

   fn get_floor_height(i: u32, k: u32) -> f32 {
       let idx = k * grid_width + i;
       return f32(floor_heights[idx]) * cell_size;
   }

   @compute @workgroup_size(64)
   fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
       // ... existing code ...

       let i = u32(pos.x / cell_size);
       let k = u32(pos.z / cell_size);
       let floor_y = get_floor_height(i, k);

       // Floor collision with riffle awareness
       if (pos.y < floor_y) {
           pos.y = floor_y + 0.001;
           vel.y = max(0.0, vel.y);
       }

       // Riffle wall collision (check if moving into higher floor)
       if (vel.x > 0.0) {
           let next_i = min(i + 1, grid_width - 1);
           let next_floor = get_floor_height(next_i, k);
           if (next_floor > floor_y && pos.y < next_floor) {
               // Hit riffle wall
               let wall_x = f32(next_i) * cell_size;
               if (pos.x > wall_x - cell_size * 0.5) {
                   pos.x = wall_x - 0.01;
                   vel.x = 0.0;
               }
           }
       }
   }
   ```

4. **Fix G2P boundary sampling**
   ```wgsl
   // In g2p_3d.wgsl - clamp sample positions to valid fluid cells
   fn sample_velocity_safe(pos: vec3<f32>, floor_y: f32) -> vec3<f32> {
       var sample_pos = pos;
       sample_pos.y = max(sample_pos.y, floor_y + 0.5 * cell_size);
       return sample_velocity(sample_pos);
   }
   ```

### Solution B: Mark Riffle Cells as Solid

**Effort:** Low
**Risk:** Medium - may lose internal riffle detail
**Pros:** Uses existing solid cell handling
**Cons:** Coarse approximation, riffles become blocky

Fill `cell_types` buffer with SOLID for all cells where `y < floor_heights[i,k]`.

### Solution C: SDF-Based Collision

**Effort:** High
**Risk:** Medium
**Pros:** Smooth collisions, handles complex geometry
**Cons:** Requires SDF generation, more complex

Pre-compute signed distance field for riffle geometry, use gradient for collision response.

## Recommended Action

**Solution A: GPU Floor Heights Buffer**

This is the proper fix that addresses the root cause. The simulation already computes `floor_heights` on CPU - we just need to upload it to GPU and use it in boundary enforcement.

## Technical Details

### Affected Files

| File | Change |
|------|--------|
| `crates/game/src/gpu/flip_3d.rs` | Add floor_heights buffer, binding |
| `crates/game/src/gpu/shaders/enforce_bc_3d.wgsl` | Add riffle collision logic |
| `crates/game/src/gpu/shaders/g2p_3d.wgsl` | Clamp samples to valid cells |
| `crates/game/examples/sluice_riffles.rs` | Upload floor_heights, remove CPU hack |

### Buffer Layout

```rust
// floor_heights: [u32; GRID_WIDTH * GRID_DEPTH]
// Indexed as: floor_heights[k * GRID_WIDTH + i]
// Value: Number of cells from y=0 to floor surface
```

### Shader Binding

```wgsl
@group(0) @binding(5) var<storage, read> floor_heights: array<u32>;

struct Uniforms {
    grid_width: u32,
    grid_depth: u32,
    cell_size: f32,
    // ...
}
```

## Acceptance Criteria

- [ ] Particles collide with vertical riffle faces (X velocity zeroed)
- [ ] Water pools behind riffles, building depth
- [ ] Water only flows over riffle when depth > riffle height
- [ ] No particles penetrate riffle geometry
- [ ] GPU simulation runs at same performance (no regression)
- [ ] Visual demo shows realistic riffle pooling behavior

## Work Log

| Date | Action | Result |
|------|--------|--------|
| 2025-01-02 | Initial investigation | Found particles climbing walls |
| 2025-01-02 | Added CPU collision hack | Partial fix, timing wrong |
| 2025-01-02 | Research & root cause | GPU BC missing floor_heights |

## Resources

- FLIP fluid paper: https://www.cs.ubc.ca/~rbridson/docs/zhu-siggraph05-sandfluid.pdf
- MAC grid conventions: Staggered grid velocity storage
- Houdini FLIP: SDF-based collision for production solvers
- CFL condition: dt < cell_size / max_velocity for stability

# Plan: GPU SDF Collision

## Goal
Move SDF collision from CPU to GPU to eliminate velocity/position downloads.

## Current Problem
After G2P, we download velocities and positions to CPU, run SDF collision, then sync back. This blocking download kills performance.

## Solution
Create `sdf_collision_3d.wgsl` shader that:
1. Reads particle positions and velocities
2. Evaluates SDF at each particle position
3. Applies collision response (push out + velocity reflection)
4. Writes corrected positions and velocities back

## Files to Create

### `crates/game/src/gpu/shaders/sdf_collision_3d.wgsl`

```wgsl
// SDF Collision 3D Shader
//
// Evaluates signed distance field at particle positions and applies
// collision response: pushes particles out of solids and reflects velocity.

struct Params {
    particle_count: u32,
    // SDF box bounds
    box_min_x: f32,
    box_min_y: f32,
    box_min_z: f32,
    box_max_x: f32,
    box_max_y: f32,
    box_max_z: f32,
    // Collision params
    restitution: f32,  // velocity reflection coefficient (0-1)
    friction: f32,     // tangential velocity reduction (0-1)
    push_out_factor: f32,  // how far to push out of surface
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec3<f32>>;

// Simple box SDF (can be extended for more complex geometry)
fn sdf_box(p: vec3<f32>) -> f32 {
    let box_min = vec3<f32>(params.box_min_x, params.box_min_y, params.box_min_z);
    let box_max = vec3<f32>(params.box_max_x, params.box_max_y, params.box_max_z);

    // Signed distance to interior of box (negative inside, positive outside)
    let center = (box_min + box_max) * 0.5;
    let half_extents = (box_max - box_min) * 0.5;

    let q = abs(p - center) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Gradient of SDF (surface normal pointing outward from solid)
fn sdf_gradient(p: vec3<f32>) -> vec3<f32> {
    let eps = 0.001;
    return normalize(vec3<f32>(
        sdf_box(p + vec3<f32>(eps, 0.0, 0.0)) - sdf_box(p - vec3<f32>(eps, 0.0, 0.0)),
        sdf_box(p + vec3<f32>(0.0, eps, 0.0)) - sdf_box(p - vec3<f32>(0.0, eps, 0.0)),
        sdf_box(p + vec3<f32>(0.0, 0.0, eps)) - sdf_box(p - vec3<f32>(0.0, 0.0, eps))
    ));
}

@compute @workgroup_size(256)
fn sdf_collision(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.particle_count) {
        return;
    }

    let idx = id.x;
    var pos = positions[idx];
    var vel = velocities[idx];

    // Get signed distance (we want particles INSIDE the box, so negate)
    let d = -sdf_box(pos);  // Negative means inside solid (box walls)

    if (d < 0.0) {
        // Particle is inside wall, push it out
        let n = -sdf_gradient(pos);  // Normal pointing into box interior

        // Push out
        pos = pos - n * (d - params.push_out_factor);

        // Velocity reflection
        let vn = dot(vel, n);  // Normal component
        if (vn < 0.0) {
            // Moving into wall
            let vt = vel - n * vn;  // Tangential component
            vel = vt * (1.0 - params.friction) - n * vn * params.restitution;
        }
    }

    positions[idx] = pos;
    velocities[idx] = vel;
}
```

## Files to Modify

### `crates/game/src/gpu/flip_3d.rs`

Add SDF collision pipeline after advection:

1. Add SdfCollisionParams3D struct (matches shader)
2. Add pipeline, bind_group, params_buffer fields
3. Create pipeline in new()
4. Dispatch after advection in step()
5. Remove CPU collision code from example

### `crates/game/examples/box_3d_test.rs`

Remove the CPU SDF collision loop entirely. The GPU handles it now.

## Testing

1. Run with 5k particles, verify particles stay in box
2. Run with 100k particles, measure FPS improvement
3. Verify no visual regression in collision behavior

## Expected Improvement

Eliminating velocity download should give 2-3x speedup on top of current 31 FPS.
Combined with Phase 4 (direct rendering), should hit 60+ FPS at 50k particles.

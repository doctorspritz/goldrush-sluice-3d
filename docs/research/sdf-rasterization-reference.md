# SDF Rasterization for Static Particle Collision

## Problem Statement

When gold particles fall onto a settled sand pile, they sink through instead of sitting on top. This happens because:
1. PBD collision correction is applied per-particle-pair, not as a unified resistance
2. Static particles (is_static=1) can be pushed individually even when part of a solid pile
3. The force threshold for waking isn't high enough to prevent slow interpenetration

## Solution: Sleep-to-Grid / SDF Rasterization

Convert static particle clusters into a unified SDF representation that dynamic particles collide against as a solid boundary.

## Key Research Sources

### 1. NVIDIA Warp - Granular SDF Collision
**Source**: [github.com/NVIDIA/warp](https://github.com/NVIDIA/warp)
**File**: `warp/examples/sim/example_granular_collision_sdf.py`

Key features:
- `wp.sim.ModelBuilder()` for particle-based granular simulation
- SDF collision geometry from VDB files or procedural generation
- `wp.Volume.load_from_numpy()` for creating SDF from particle data
- Sparse volumes using NanoVDB format for memory efficiency
- Interactive rates for 100K+ particles

```python
# Example structure from Warp
builder = wp.sim.ModelBuilder()
builder.default_particle_radius = 0.1
# Add SDF collision shape to body
# SDF shapes support: is_solid, thickness, collision groups
```

### 2. MPM P2G Transfer (Material Point Method)
**Source**: [nialltl.neocities.org/articles/mpm_guide](https://nialltl.neocities.org/articles/mpm_guide)

Key technique: **Quadratic B-spline interpolation** for particle-to-grid transfer

Process:
1. Calculate which grid cells neighbor the particle's position
2. Evaluate quadratic kernel weight for each neighboring cell (3x3)
3. Weight particle properties (mass, momentum) by kernel values
4. Accumulate weighted contributions into each cell

```
For each particle:
  For each cell in 3x3 neighborhood:
    weight = quadratic_bspline(particle_pos - cell_pos)
    cell.mass += weight * particle.mass
    cell.momentum += weight * particle.momentum
```

Benefits:
- Efficient parallel computation ("each thread is dedicated to either 1 cell or 1 particle")
- Good balance between performance and stability

### 3. GPU Gems 3 - Rigid Body as Particles
**Source**: [developer.nvidia.com/gpugems/gpugems3/.../chapter-29](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-29-real-time-rigid-body-simulation-gpus)

Key technique: **Particle-based rigid body representation**

- Rigid bodies represented as "a set of particles that are spheres of identical size"
- Space discretized using 3D grid aligned with object
- Voxels marked as "inside" rigid body through ray-casting
- Optimal voxel size matches particle diameter

Grid-based collision detection:
- 3D uniform grid reduces O(n²) to O(n)
- Check 27 cells (self + 26 neighbors) for collision candidates

### 4. SDF Framework for Unified DEM
**Source**: [ResearchGate - SDF framework for unified DEM modeling](https://www.researchgate.net/publication/362490391_Signed_distance_field_framework_for_unified_DEM_modeling_of_granular_media_with_arbitrary_particle_shapes)

Key concept: **Pre-cached SDF lookup table**

- SDF function and surface projection function define interface
- Signed distance: positive inside particles, negative outside
- Zeroth isosurface represents particle surface
- Pre-cache potential values within a particle at lattice grid
- Contact detection becomes SDF query with lookup-table algorithm

### 5. Global SDF for Particle Collision (Flax Engine)
**Source**: [docs.flaxengine.com/manual/graphics/models/sdf.html](https://docs.flaxengine.com/manual/graphics/models/sdf.html)

Key technique: **Multi-cascade volumetric SDF**

- Rasterize all scene geometry into single Global Volume texture
- 4 cascades for precision: ~10cm near camera, covers 200m radius
- Three collision modes:
  - **Position (Global SDF)**: Snap particles to SDF surface
  - **Collision (Global SDF)**: Prevent particles passing through
  - **Conform to Global SDF**: Apply forces to keep particles on surface

GPU access via shader header (`GlobalSignDistanceField.hlsl`):
- Constant buffer data
- 3D texture resources
- Raycasting through SDF volume

### 6. Storm Granular Simulation Tool
**Source**: [CGPress - Storm granular simulation tool](https://cgpress.org/archives/storm-new-standalone-granular-simulation-tool-2.html)

Key feature: **Sleep/awake particles with geometry**

- Particles in persistent contact in slowly-moving collection stay in same voxel
- Significant compute time reduction when using voxel-based optimization
- Particle resolution affects simulation accuracy

### 7. Intersection Distance Field Collision
**Source**: [Eurographics Digital Library](https://diglib.eg.org/items/5167e4c5-50bd-4e0b-b60f-9fd8b03189ef)

Key technique: **Particle-based SDF sampling**

- Particles sample region where intersections can occur
- Distance field projects particles onto intersection surface
- Extracts collision data: normals, penetration depth
- Well suited to GPU parallelization
- Handles various object types uniformly

---

## Implementation Plan for This Project

### Phase 1: SDF Grid Buffer

Add a 2D SDF grid that represents static particles as a unified distance field.

```wgsl
// New buffer
@group(0) @binding(N) var<storage, read_write> sdf_grid: array<f32>;

// Grid parameters
const SDF_CELL_SIZE: f32 = CELL_SIZE;  // Match existing grid
const SDF_WIDTH: u32 = GRID_WIDTH;
const SDF_HEIGHT: u32 = GRID_HEIGHT;
```

### Phase 2: Rasterize Static Particles to SDF

Each frame, compute pass to splat static particles to SDF grid:

```wgsl
@compute @workgroup_size(256)
fn rasterize_static_to_sdf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let particle_idx = gid.x;
    if (particle_idx >= particle_count) { return; }

    // Only static particles contribute to SDF
    if (static_states[particle_idx] != 1u) { return; }

    let pos = positions[particle_idx];
    let radius = get_radius(particle_idx);

    // Splat negative distance (inside = negative)
    let cell_x = i32(pos.x / SDF_CELL_SIZE);
    let cell_y = i32(pos.y / SDF_CELL_SIZE);

    // Write to 3x3 neighborhood
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let cx = cell_x + dx;
            let cy = cell_y + dy;
            if (cx < 0 || cy < 0 || cx >= i32(SDF_WIDTH) || cy >= i32(SDF_HEIGHT)) {
                continue;
            }

            let cell_center = vec2<f32>(
                (f32(cx) + 0.5) * SDF_CELL_SIZE,
                (f32(cy) + 0.5) * SDF_CELL_SIZE
            );
            let dist = length(cell_center - pos) - radius;

            // Atomically take minimum (most negative = deepest inside)
            let cell_idx = u32(cy) * SDF_WIDTH + u32(cx);
            atomicMin(&sdf_grid_i32[cell_idx], i32(dist * 1000.0));  // Scale for i32 atomics
        }
    }
}
```

### Phase 3: Dynamic Particles Collide with SDF

During collision detection, dynamic particles sample SDF:

```wgsl
fn sample_sdf(pos: vec2<f32>) -> f32 {
    let cell_x = pos.x / SDF_CELL_SIZE;
    let cell_y = pos.y / SDF_CELL_SIZE;

    // Bilinear interpolation from 4 corners
    let x0 = u32(floor(cell_x));
    let y0 = u32(floor(cell_y));
    let x1 = x0 + 1u;
    let y1 = y0 + 1u;

    let fx = fract(cell_x);
    let fy = fract(cell_y);

    let d00 = f32(sdf_grid_i32[y0 * SDF_WIDTH + x0]) / 1000.0;
    let d10 = f32(sdf_grid_i32[y0 * SDF_WIDTH + x1]) / 1000.0;
    let d01 = f32(sdf_grid_i32[y1 * SDF_WIDTH + x0]) / 1000.0;
    let d11 = f32(sdf_grid_i32[y1 * SDF_WIDTH + x1]) / 1000.0;

    return mix(mix(d00, d10, fx), mix(d01, d11, fx), fy);
}

fn sdf_gradient(pos: vec2<f32>) -> vec2<f32> {
    let eps = SDF_CELL_SIZE * 0.5;
    let dx = sample_sdf(pos + vec2(eps, 0.0)) - sample_sdf(pos - vec2(eps, 0.0));
    let dy = sample_sdf(pos + vec2(0.0, eps)) - sample_sdf(pos - vec2(0.0, eps));
    return normalize(vec2(dx, dy));
}

// In collision loop for dynamic particles:
let sdf_dist = sample_sdf(pos);
if (sdf_dist < radius) {
    // Collision with static pile!
    let sdf_normal = sdf_gradient(pos);
    let penetration = radius - sdf_dist;

    // Push out along SDF gradient
    pos += sdf_normal * penetration;

    // Apply friction/damping
    let tangent = vec2(-sdf_normal.y, sdf_normal.x);
    let v_normal = dot(vel, sdf_normal);
    let v_tangent = dot(vel, tangent);

    if (v_normal < 0.0) {
        vel -= sdf_normal * v_normal * 1.5;  // Bounce with restitution
    }
    vel -= tangent * v_tangent * 0.3;  // Friction
}
```

### Phase 4: Clear SDF Each Frame

Before rasterization:

```wgsl
@compute @workgroup_size(256)
fn clear_sdf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < SDF_WIDTH * SDF_HEIGHT) {
        sdf_grid_i32[idx] = i32(SDF_CELL_SIZE * 10.0 * 1000.0);  // Large positive = far from surface
    }
}
```

### Pipeline Order

1. Clear SDF grid
2. Rasterize static particles → SDF
3. Run DEM forces (dynamic particles sample SDF for pile collision)
4. Integrate velocities
5. Check static transitions

### Expected Benefits

1. **Gold sits on sand**: Dynamic gold samples SDF, sees "inside" pile, gets pushed to surface
2. **No individual penetration**: Can't push between static particles because SDF is unified
3. **Performance**: O(1) SDF sample vs O(N) neighbor checks for pile collision
4. **Correct physics**: Normal from SDF gradient, not from individual particle contacts

### Potential Issues

1. **SDF resolution**: Grid too coarse → particles slip through gaps
   - Solution: Use finer grid (2-4x physics grid resolution)

2. **Atomic contention**: Many particles writing to same cell
   - Solution: Use hierarchical rasterization or tile-based approach

3. **Transition artifacts**: Particle becoming static appears suddenly in SDF
   - Solution: Smooth transition with alpha blending over several frames

---

## Alternative Approaches Considered

### 1. Infinite Mass (Rejected by User)
Static particles with infinite mass would not move when pushed. Rejected because:
- Not physically accurate
- Creates numerical instability
- Doesn't help dynamic-static transition

### 2. Position Locking Only
Lock positions of static particles. Problem:
- Individual particles still have separate collision boundaries
- Dynamic particles can slip between gaps

### 3. Force Multiplication
Multiply static particle resistance force. Problem:
- Still individual collisions
- Requires very high multiplier → instability

### 4. Constraint-Based (Jacobi/Gauss-Seidel)
More solver iterations to enforce non-penetration. Problem:
- Slow convergence
- Still works on individual particles
- Doesn't create unified surface

---

## Files to Modify

1. `crates/game/src/gpu/dem.rs`
   - Add SDF grid buffer
   - Add clear/rasterize pipelines
   - Bind SDF to forces shader

2. `crates/game/src/gpu/shaders/dem_forces.wgsl`
   - Add SDF sampling functions
   - Add SDF collision in dynamic particle loop

3. `crates/game/src/gpu/shaders/sdf_rasterize.wgsl` (NEW)
   - Static particle → SDF rasterization compute shader

4. `crates/game/src/gpu/shaders/sdf_clear.wgsl` (NEW)
   - Clear SDF grid compute shader

---

## Test Cases

1. **Gold on settled sand**: Drop gold, verify it stays on surface
2. **Heavy impact**: Drop heavy rock on pile, verify it creates crater but doesn't fall through
3. **Pile angle**: Verify angle of repose still works (SDF shouldn't affect static-static)
4. **Wake cascade**: Remove support, verify particles wake and SDF updates correctly
5. **Performance**: Measure frame time with 100K particles, ensure SDF doesn't hurt performance

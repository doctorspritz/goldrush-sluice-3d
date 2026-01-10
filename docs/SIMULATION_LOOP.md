# Simulation Loop

**Date:** 2026-01-10
**File:** `crates/game/src/gpu/flip_3d.rs` → `step_internal()` → `run_gpu_passes()`

---

## Per-Frame Order of Operations

### 0. Pre-Step (CPU)
```
upload_bc_and_cell_types()
apply_gravel_obstacles()  // optional
```

### 1. P2G Transfer (Particle → Grid)
```
p2g.upload_particles()    // CPU → GPU positions, velocities, densities, C matrices
p2g.encode()              // Atomic scatter to grid + divide by weights
water_p2g.encode()        // Water-only P2G (for flow measurement)
```
**Shader:** `p2g_scatter_3d.wgsl`, `p2g_divide_3d.wgsl`

### 2. Sediment Fraction Computation
```
sediment_fraction_pipeline  // Compute per-cell sediment/water ratio
gravel_porosity_pipeline    // Optional: reduce porosity near gravel obstacles
```
**Shader:** `sediment_fraction_3d.wgsl`, `gravel_porosity_3d.wgsl`

### 3. Sediment Pressure (Granular Column)
```
sediment_pressure_pipeline  // Computes overburden pressure from sediment column
```
**Shader:** `sediment_pressure_3d.wgsl`

### 4. Boundary Conditions (BEFORE saving old velocities!)
```
bc_u_pipeline  // Enforce BC on U grid faces
bc_v_pipeline  // Enforce BC on V grid faces
bc_w_pipeline  // Enforce BC on W grid faces
```
**Shader:** `bc_u_3d.wgsl`, `bc_v_3d.wgsl`, `bc_w_3d.wgsl`

### 5. Save Old Grid Velocities (for FLIP delta)
```
copy_buffer(grid_u → grid_u_old)
copy_buffer(grid_v → grid_v_old)
copy_buffer(grid_w → grid_w_old)
```
**Note:** This is a buffer copy, not a shader dispatch.

### 6. Apply Gravity
```
gravity_pipeline  // Add gravity*dt to V grid
```
**Shader:** `gravity_3d.wgsl`

### 7. Apply Flow Acceleration (Sluice Downstream)
```
if flow_accel > 0.0001:
    flow_pipeline  // Add flow_accel*dt to U grid
```
**Shader:** `flow_3d.wgsl`

### 8. Vorticity Confinement (Optional)
```
if vorticity_epsilon > 0:
    vorticity_compute_pipeline      // Compute curl of velocity field
    vorticity_confine_u_pipeline    // Apply confinement force to U
    vorticity_confine_v_pipeline    // Apply confinement force to V
    vorticity_confine_w_pipeline    // Apply confinement force to W
```
**Shader:** `vorticity_3d.wgsl`, `vorticity_confine_*_3d.wgsl`

### 9. Pressure Solve (MGPCG)
```
pressure.encode()  // Divergence → Jacobi/MGPCG iterations → Gradient subtraction
```
**Shader:** `divergence_3d.wgsl`, `mgpcg_*.wgsl` (multiple shaders)
**Iterations:** Configurable (default 120 for friction_sluice)

### 10. Porosity Drag (Optional)
```
if sediment_porosity_drag > 0:
    porosity_drag_u_pipeline  // Reduce U velocity in high-sediment cells
    porosity_drag_v_pipeline  // Reduce V velocity in high-sediment cells
    porosity_drag_w_pipeline  // Reduce W velocity in high-sediment cells
```
**Shader:** `porosity_drag_*_3d.wgsl`

### 11. G2P Transfer (Grid → Particle)
```
g2p.upload_params()  // Upload sediment physics params
g2p.encode()         // Sample grid, apply FLIP/PIC blend, sediment physics
```
**Shader:** `g2p_3d.wgsl`

### 12. Density Projection (Volume Conservation)
```
density_error_pipeline         // Compute density error from particle counts
pressure.clear_pressure()      // Reset pressure buffer
pressure.encode_iterations_only()  // 40 Jacobi iterations
density_position_grid_pipeline // Compute position deltas on grid
density_correct_pipeline       // Apply position correction to particles
```
**Shader:** `density_error_3d.wgsl`, `density_position_grid_3d.wgsl`, `density_correct_3d.wgsl`

### 13. Sediment Density Projection (DISABLED)
```
// DISABLED: jamming causes infinite compression
// Code exists but is commented out (lines 3339-3420)
```

### 14. Readback (GPU → CPU)
```
g2p.download()  // Velocities + C matrices
read_positions()  // Position buffer readback
```

---

## Key Buffers

| Buffer | Size | Purpose |
|--------|------|---------|
| `positions_buffer` | max_particles × Vec4 | Particle positions |
| `velocities_buffer` | max_particles × Vec4 | Particle velocities |
| `densities_buffer` | max_particles × f32 | Particle densities (1.0=water, >1.0=sediment) |
| `c_col0/1/2_buffer` | max_particles × Vec4 | APIC C matrix columns |
| `grid_u/v/w_buffer` | (w+1)×h×d, w×(h+1)×d, w×h×(d+1) | MAC grid velocities |
| `grid_u/v/w_old_buffer` | Same | Saved for FLIP delta |
| `cell_type_buffer` | w×h×d | 0=air, 1=fluid, 2=solid |
| `sdf_buffer` | w×h×d | Signed distance field (static geometry) |
| `bed_height_buffer` | w×d | Bed surface height per column |
| `particle_count_buffer` | w×h×d | Particle count per cell |
| `sediment_count_buffer` | w×h×d | Sediment particle count per cell |
| `sediment_fraction_buffer` | w×h×d | Sediment/total ratio per cell |

---

## Shader Files

Located in `crates/game/src/gpu/shaders/`:

```
# P2G
p2g_scatter_3d.wgsl
p2g_divide_3d.wgsl

# Boundary Conditions
bc_u_3d.wgsl
bc_v_3d.wgsl
bc_w_3d.wgsl

# Forces
gravity_3d.wgsl
flow_3d.wgsl

# Vorticity
vorticity_3d.wgsl
vorticity_confine_u_3d.wgsl
vorticity_confine_v_3d.wgsl
vorticity_confine_w_3d.wgsl

# Pressure (MGPCG)
divergence_3d.wgsl
mgpcg_*.wgsl (multiple)

# Sediment
sediment_fraction_3d.wgsl
sediment_pressure_3d.wgsl
sediment_cell_type_3d.wgsl (DISABLED)
gravel_porosity_3d.wgsl
porosity_drag_u_3d.wgsl
porosity_drag_v_3d.wgsl
porosity_drag_w_3d.wgsl

# G2P
g2p_3d.wgsl

# Density Projection
density_error_3d.wgsl
density_position_grid_3d.wgsl
density_correct_3d.wgsl
```

---

## Critical Notes

1. **BC enforcement BEFORE saving old velocities** - This is essential for correct FLIP delta computation. The grid must be in a valid state before we store the "before pressure" velocities.

2. **Sediment jamming is DISABLED** - The voxel-based collision system exists but is commented out because it caused infinite compression when sediment was marked SOLID.

3. **Density projection runs AFTER G2P** - This is the "implicit density projection" that pushes particles from crowded regions. It uses 40 Jacobi iterations (separate from the main pressure solve).

4. **P2G sediment counting** - The `p2g_scatter_3d.wgsl` shader counts sediment particles per cell using `atomicAdd`, but this data is currently unused because jamming is disabled.

# Shader Organization

**Location:** `crates/game/src/gpu/shaders/`
**Count:** 64 WGSL compute/render shaders

---

## FLIP/APIC Core Pipeline

### P2G (Particle-to-Grid Transfer)
| Shader | Purpose |
|--------|---------|
| `p2g_scatter_3d.wgsl` | Atomic scatter: particles write momentum to 3x3x3 grid neighborhood |
| `p2g_scatter_tiled_3d.wgsl` | Tiled variant for better memory access patterns |
| `p2g_cell_centric_3d.wgsl` | Cell-centric P2G (alternative to particle-centric) |
| `p2g_divide_3d.wgsl` | Normalize grid velocities by accumulated weights |

### G2P (Grid-to-Particle Transfer)
| Shader | Purpose |
|--------|---------|
| `g2p_3d.wgsl` | Gather grid velocities to particles, compute FLIP delta, apply sediment physics |

### Pressure Solve
| Shader | Purpose |
|--------|---------|
| `divergence_3d.wgsl` | Compute velocity divergence (RHS for pressure) |
| `pressure_3d.wgsl` | Jacobi pressure iteration |
| `pressure_gradient_3d.wgsl` | Subtract pressure gradient from velocity |

### Multigrid PCG (MGPCG)
| Shader | Purpose |
|--------|---------|
| `mg_smooth.wgsl` | Red-black Gauss-Seidel smoother |
| `mg_restrict.wgsl` | Downsample residual to coarser level |
| `mg_prolongate.wgsl` | Interpolate correction to finer level |
| `mg_residual.wgsl` | Compute residual at a level |
| `pcg_ops.wgsl` | PCG vector operations (dot, axpy, etc.) |

---

## Forces & Physics

### External Forces
| Shader | Purpose |
|--------|---------|
| `gravity_3d.wgsl` | Apply gravity to V (vertical) velocity grid |
| `flow_3d.wgsl` | Apply downstream flow acceleration to U grid |

### Vorticity Confinement
| Shader | Purpose |
|--------|---------|
| `vorticity_3d.wgsl` | Compute curl of velocity field (vorticity vector) |
| `vorticity_confine_3d.wgsl` | Apply confinement force to U/V/W grids |

### Boundary Conditions
| Shader | Purpose |
|--------|---------|
| `enforce_bc_3d.wgsl` | Enforce boundary conditions on MAC grid faces |

---

## Sediment Physics

### Fraction & Pressure
| Shader | Purpose |
|--------|---------|
| `sediment_fraction_3d.wgsl` | Compute sediment/water ratio per cell |
| `sediment_pressure_3d.wgsl` | Compute overburden pressure from sediment column (Drucker-Prager) |
| `sediment_cell_type_3d.wgsl` | Mark high-sediment cells as SOLID (disabled) |

### Porosity & Drag
| Shader | Purpose |
|--------|---------|
| `gravel_porosity_3d.wgsl` | Reduce porosity near gravel obstacles |
| `porosity_drag_3d.wgsl` | Reduce water velocity in high-sediment regions |

### Density Projection
| Shader | Purpose |
|--------|---------|
| `density_error_3d.wgsl` | Compute density error from particle counts |
| `density_position_grid_3d.wgsl` | Compute position correction field on grid |
| `density_correct_3d.wgsl` | Apply position correction to particles |
| `sediment_density_error_3d.wgsl` | Sediment-specific density error |
| `sediment_density_correct_3d.wgsl` | Sediment-specific position correction |

---

## Collision & Advection

### SDF Collision
| Shader | Purpose |
|--------|---------|
| `sdf_collision_3d.wgsl` | Push particles out of SDF geometry, apply friction |
| `gravel_obstacle_3d.wgsl` | Write gravel clumps to SDF as obstacles |

### Advection & Velocity
| Shader | Purpose |
|--------|---------|
| `advect_3d.wgsl` | Advect particles by their velocities |
| `velocity_extrapolate_3d.wgsl` | Extrapolate velocity into air cells |
| `kinematic_water_3d.wgsl` | Kinematic water surface constraint |

---

## Bed Exchange (Deposition/Entrainment)

| Shader | Purpose |
|--------|---------|
| `bed_flux_3d.wgsl` | Compute deposition/entrainment flux at bed surface |
| `bed_update_3d.wgsl` | Update bed height from flux |
| `bed_stats_3d.wgsl` | Gather particle statistics near bed (GPU probes) |

---

## Particle Sorting (GPU Counting Sort)

| Shader | Purpose |
|--------|---------|
| `particle_sort_keys.wgsl` | Compute cell keys from particle positions |
| `particle_sort_count.wgsl` | Count particles per cell |
| `particle_sort_prefix_sum.wgsl` | Parallel prefix sum for cell offsets |
| `particle_sort_scatter.wgsl` | Scatter particles to sorted order |

---

## SPH Solvers (Alternative to FLIP)

### Core SPH
| Shader | Purpose |
|--------|---------|
| `sph_simple.wgsl` | Basic SPH with viscosity |
| `sph_density.wgsl` | SPH density computation |
| `sph_pressure.wgsl` | SPH pressure forces |
| `sph_force.wgsl` | SPH force accumulation |

### DFSPH (Divergence-Free SPH)
| Shader | Purpose |
|--------|---------|
| `sph_dfsph.wgsl` | Divergence-free SPH solver |
| `sph_predict_hash.wgsl` | Predict positions and update spatial hash |

### IISPH (Implicit Incompressible SPH)
| Shader | Purpose |
|--------|---------|
| `sph_iisph.wgsl` | Implicit incompressible SPH solver |

### SPH Utilities
| Shader | Purpose |
|--------|---------|
| `sph_bitonic_sort.wgsl` | Bitonic sort for SPH neighbor finding |
| `sph_bruteforce.wgsl` | Brute-force neighbor search (debugging) |

---

## Heightfield (2.5D World Layer)

| Shader | Purpose |
|--------|---------|
| `heightfield_erosion.wgsl` | Shear-stress based erosion |
| `heightfield_collapse.wgsl` | Angle-of-repose collapse |
| `heightfield_water.wgsl` | Shallow water equations |
| `heightfield_render.wgsl` | Heightfield terrain rendering |
| `heightfield_emitter.wgsl` | Spawn particles from heightfield |
| `heightfield_material_tool.wgsl` | Modify heightfield material |
| `heightfield_bridge_merge.wgsl` | Merge 3D particles back to heightfield |

---

## Particle Emission & Absorption

| Shader | Purpose |
|--------|---------|
| `particle_emitter_3d.wgsl` | Spawn new particles in simulation |
| `particle_absorption_3d.wgsl` | Remove particles at absorbers |

---

## Screen-Space Fluid Rendering

| Shader | Purpose |
|--------|---------|
| `particle_3d.wgsl` | Particle point sprite rendering |
| `fluid_depth_3d.wgsl` | Render fluid depth buffer |
| `fluid_blur_3d.wgsl` | Bilateral blur for smooth surface |
| `fluid_cell_expand_3d.wgsl` | Expand fluid cells for rendering |
| `fluid_compose_3d.wgsl` | Final compositing with scene |

---

## Shader Naming Conventions

- `*_3d.wgsl` - 3D FLIP/particle simulation
- `*_u/v/w_*.wgsl` - Per-component (staggered MAC grid)
- `sph_*.wgsl` - SPH-based solvers
- `mg_*.wgsl` - Multigrid operations
- `heightfield_*.wgsl` - 2.5D world layer
- `fluid_*.wgsl` - Screen-space rendering

## Buffer Binding Conventions

- `@group(0) @binding(0)` - Uniform params struct
- `@group(0) @binding(1+)` - Storage buffers (positions, velocities, grid, etc.)
- MAC grid stored as separate U/V/W buffers with staggered dimensions

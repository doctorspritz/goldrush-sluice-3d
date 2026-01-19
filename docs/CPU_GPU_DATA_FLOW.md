# CPU ↔ GPU Data Flow

This document describes when and how data moves between CPU and GPU memory.

---

## Buffer Residency

### GPU-Resident (Hot Path)
These buffers stay on GPU during simulation. No per-frame transfers.

| Buffer | Size (1M particles) | Purpose |
|--------|---------------------|---------|
| `positions_buffer` | 12 MB | Particle positions (Vec3) |
| `velocities_buffer` | 12 MB | Particle velocities (Vec3) |
| `densities_buffer` | 4 MB | Particle densities (f32) |
| `c_col0/1/2_buffer` | 36 MB | APIC C matrix (3x Vec3) |
| `grid_u/v/w_buffer` | ~15 MB | MAC grid velocities |
| `grid_u/v/w_old_buffer` | ~15 MB | Saved for FLIP delta |
| `cell_type_buffer` | ~1 MB | 0=air, 1=fluid, 2=solid |
| `sdf_buffer` | ~1 MB | Signed distance field |
| `particle_count_buffer` | ~1 MB | Particles per cell |
| `sediment_fraction_buffer` | ~1 MB | Sediment ratio per cell |
| `pressure_buffer` | ~1 MB | Pressure field |

### CPU → GPU (Per-Frame Uploads)

| Data | When | Size | Purpose |
|------|------|------|---------|
| Uniform params | Every frame | ~200 bytes | Grid dims, dt, physics params |
| Boundary conditions | When changed | ~1 MB | Cell types, BC flags |
| SDF | When geometry changes | ~1 MB | Static collision geometry |
| Emitter params | When spawning | ~64 bytes | Spawn position, rate, velocity |

### GPU → CPU (Readback)

| Data | When | Size | Purpose |
|------|------|------|---------|
| Positions | Optional (debug/save) | 12 MB | Particle visualization on CPU |
| Velocities | Optional (debug) | 12 MB | Velocity analysis |
| Probe stats | Every frame | 256 bytes | Bed height, throughput counters |

---

## Per-Frame Timeline

```
Frame N
├── CPU: Update uniform params (dt, frame count)
│
├── GPU: P2G Scatter ──────────────────────────────────────────┐
│         particles → grid momentum (atomics)                  │
│                                                              │
├── GPU: P2G Divide                                            │
│         normalize by weights                                 │
│                                                              │
├── GPU: Sediment Fraction                                     │
│         count sediment per cell                              │
│                                                              │
├── GPU: Sediment Pressure                                     │
│         column scan for overburden                           │
│                                                              │
├── GPU: Boundary Conditions                                   │
│         enforce BC on grid faces                             │
│                                                              │
├── GPU: Copy grid_* → grid_*_old ─────────────────────────────┤
│         (buffer copy, no shader)                             │
│                                                              │
├── GPU: Gravity + Flow                                        │
│         external forces on grid                              │
│                                                              │
├── GPU: Vorticity Confinement (optional)                      │
│         compute curl, apply confinement                      │
│                                                              │
├── GPU: Pressure Solve (MGPCG) ───────────────────────────────┤
│         divergence → iterations → gradient                   │
│         (most expensive: 60-120 iterations)                  │
│                                                              │
├── GPU: Porosity Drag (optional)                              │
│         damp velocity in high-sediment cells                 │
│                                                              │
├── GPU: G2P ──────────────────────────────────────────────────┤
│         grid → particles (FLIP delta + sediment physics)     │
│                                                              │
├── GPU: Density Projection                                    │
│         push particles from crowded regions                  │
│                                                              │
├── GPU: Advection                                             │
│         move particles by velocity                           │
│                                                              │
├── GPU: SDF Collision                                         │
│         push out of geometry, friction                       │
│                                                              │
├── GPU: Bed Stats (optional) ─────────────────────────────────┤
│         gather probe statistics                              │
│                                                              │
└── GPU → CPU: Readback probe stats (async) ───────────────────┘
          ~256 bytes, non-blocking
```

---

## Async Readback Pattern

Probe stats use triple-buffered async readback to avoid stalls:

```rust
// Frame N: Request readback
encoder.copy_buffer_to_buffer(&gpu_stats, &staging[frame % 3]);

// Frame N+2: Read result (2-frame latency)
staging[(frame + 1) % 3].slice(..).map_async(MapMode::Read, ..);
```

Position/velocity readback (for saving or debug) uses similar pattern but is optional.

---

## Staging Buffer Usage

| Buffer | Direction | Purpose |
|--------|-----------|---------|
| `positions_staging` | GPU → CPU | Position readback (optional) |
| `velocities_staging` | GPU → CPU | Velocity readback (optional) |
| `c_col*_staging` | GPU → CPU | C matrix readback (optional) |
| `probe_stats_staging[3]` | GPU → CPU | Triple-buffered probe stats |
| `params_staging` | CPU → GPU | Uniform updates |

---

## Zero-Copy Optimizations

1. **Grid velocities**: P2G output is G2P input - no staging, direct buffer sharing
2. **Particle buffers**: Positions/velocities stay GPU-resident across frames
3. **Cell types**: Only re-uploaded when geometry changes (not every frame)
4. **Pressure**: Cleared and reused each frame, never read back

---

## Memory Map

```
┌─────────────────────────────────────────────────────────────────┐
│                         GPU VRAM (~120 MB)                      │
├─────────────────────────────────────────────────────────────────┤
│ Particle Buffers                                                │
│   positions[max_particles]     12 MB                            │
│   velocities[max_particles]    12 MB                            │
│   densities[max_particles]      4 MB                            │
│   c_col0/1/2[max_particles]    36 MB                            │
├─────────────────────────────────────────────────────────────────┤
│ Grid Buffers (100³ example)                                     │
│   grid_u[101×100×100]          ~4 MB                            │
│   grid_v[100×101×100]          ~4 MB                            │
│   grid_w[100×100×101]          ~4 MB                            │
│   grid_u/v/w_old               ~12 MB                           │
│   cell_type[100×100×100]       ~1 MB                            │
│   pressure[100×100×100]        ~4 MB                            │
│   sdf[100×100×100]             ~4 MB                            │
├─────────────────────────────────────────────────────────────────┤
│ Auxiliary                                                       │
│   particle_count[cells]         1 MB                            │
│   sediment_count[cells]         1 MB                            │
│   sediment_fraction[cells]      4 MB                            │
│   vorticity[cells]              12 MB                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       CPU RAM (Staging)                         │
├─────────────────────────────────────────────────────────────────┤
│ Uniform Params                 ~1 KB                            │
│ Probe Stats Staging (×3)       ~1 KB                            │
│ Position/Velocity Staging     ~24 MB (optional, debug only)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## When Readback Happens

| Scenario | Data | Latency |
|----------|------|---------|
| Normal simulation | Probe stats only | 2 frames (async) |
| Debug visualization | + positions | Blocking or async |
| Save state | + all particle data | Blocking |
| Headless benchmark | None | N/A |

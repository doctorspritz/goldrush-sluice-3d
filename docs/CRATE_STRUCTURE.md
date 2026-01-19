# Crate Structure

The project is organized into two main crates with distinct responsibilities.

---

## Crate Overview

```
crates/
├── game/          # GPU simulation, rendering, application
│   ├── src/
│   │   ├── gpu/   # GPU compute pipelines and shaders
│   │   └── ...    # App, rendering, input
│   └── tests/     # GPU tests, visual regression
│
└── sim3d/         # CPU simulation, shared types, algorithms
    └── src/       # Pure Rust, no GPU dependency
```

---

## `crates/sim3d` - CPU Simulation Core

**Purpose**: Platform-independent simulation logic, algorithms, and types.

**Dependencies**: `glam`, `rand`, `serde` (no GPU dependencies)

### Modules

| File | Purpose |
|------|---------|
| `lib.rs` | Crate entry, re-exports |
| `particle.rs` | `Particle3D` struct, material types |
| `grid.rs` | MAC grid structure, cell types |
| `transfer.rs` | CPU P2G/G2P with B-spline kernels |
| `kernels.rs` | B-spline weight functions |
| `pressure.rs` | CPU pressure solver (Jacobi) |
| `advection.rs` | Particle advection, CFL checking |
| `clump.rs` | DEM rigid body clumps (gravel, rocks) |
| `heightfield.rs` | 2.5D terrain heightfield |
| `world.rs` | World stack (terrain + water layers) |
| `constants.rs` | Physical constants, default parameters |
| `terrain_generator.rs` | Procedural terrain generation |
| `test_geometry.rs` | Test scene geometry builders |
| `serde_utils.rs` | Serialization helpers |

### Key Types

```rust
// Particle with position, velocity, density, APIC C matrix
pub struct Particle3D { ... }

// Staggered MAC grid for incompressible flow
pub struct Grid3D { ... }

// DEM rigid body for gravel simulation
pub struct Clump { ... }

// 2.5D terrain with base height + sediment layer
pub struct Heightfield { ... }
```

### When to Use sim3d

- CPU-only simulation (tests, reference implementation)
- Shared type definitions
- Algorithm prototyping before GPU port
- Headless benchmarks without GPU

---

## `crates/game` - GPU Application

**Purpose**: GPU-accelerated simulation, rendering, and application.

**Dependencies**: `wgpu`, `winit`, `egui`, `sim3d`

### Top-Level Modules

| File | Purpose |
|------|---------|
| `main.rs` | Application entry point |
| `app.rs` | Main application state |
| `camera.rs` | 3D camera controls |
| `input.rs` | Input handling |

### GPU Module (`src/gpu/`)

| File | Purpose |
|------|---------|
| `mod.rs` | GPU module entry |
| `flip_3d.rs` | Main FLIP solver orchestration (3400+ lines) |
| `p2g_3d.rs` | GPU Particle-to-Grid transfer |
| `g2p_3d.rs` | GPU Grid-to-Particle transfer |
| `pressure_3d.rs` | GPU pressure solver |
| `mgpcg.rs` | Multigrid PCG solver |
| `particle_sort.rs` | GPU counting sort |
| `bed_3d.rs` | Bed exchange, probe stats |
| `bridge_3d.rs` | Particle emission/absorption |
| `heightfield.rs` | GPU heightfield operations |
| `sph_3d.rs` | SPH solver (alternative) |
| `sph_dfsph.rs` | DFSPH solver |
| `fluid_renderer.rs` | Screen-space fluid rendering |
| `p2g_cell_centric_3d.rs` | Alternative P2G approach |

### Shader Directory (`src/gpu/shaders/`)

64 WGSL compute shaders. See [SHADERS.md](SHADERS.md) for full list.

---

## Dependency Flow

```
┌─────────────────────────────────────────────────────────┐
│                      crates/game                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │                   GPU Layer                      │   │
│  │  flip_3d.rs → p2g_3d.rs → g2p_3d.rs            │   │
│  │       ↓                                         │   │
│  │  shaders/*.wgsl                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                               │
│                         ▼ uses types from               │
│  ┌─────────────────────────────────────────────────┐   │
│  │               crates/sim3d                       │   │
│  │  Particle3D, Grid3D, Clump, Heightfield         │   │
│  │  kernels.rs, constants.rs                       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Data Flow Between Crates

### Initialization
```
sim3d::Particle3D → game uploads to GPU → GPU-resident
sim3d::Grid3D → game creates GPU buffers → GPU-resident
```

### Per-Frame
```
GPU simulation runs entirely in game/
No round-trip to sim3d after initialization
```

### Readback (Optional)
```
GPU positions → game staging buffer → CPU Vec<Particle3D>
(Only for save/debug, not hot path)
```

---

## Test Organization

### `crates/sim3d/tests/`
- Pure CPU tests
- Algorithm correctness
- No GPU required

### `crates/game/tests/`
- GPU integration tests
- Shader validation (via Naga)
- Visual regression tests
- Property-based tests

---

## Adding New Features

### Pure Algorithm (no GPU)
1. Add to `crates/sim3d/src/`
2. Write CPU tests
3. Port to GPU later if needed

### GPU Feature
1. Add shader to `crates/game/src/gpu/shaders/`
2. Add Rust pipeline code to `crates/game/src/gpu/`
3. Wire into `flip_3d.rs` simulation loop
4. Add shader validation test

### Shared Types
1. Define in `crates/sim3d/src/`
2. Use `bytemuck` derive for GPU upload
3. Import in `crates/game`

---

## Key Files by Feature

| Feature | sim3d | game |
|---------|-------|------|
| FLIP core | `transfer.rs`, `grid.rs` | `flip_3d.rs`, `p2g_3d.rs`, `g2p_3d.rs` |
| Pressure solve | `pressure.rs` | `pressure_3d.rs`, `mgpcg.rs` |
| Sediment | `particle.rs` (density) | `g2p_3d.rs`, sediment shaders |
| DEM/Gravel | `clump.rs` | `gravel_obstacle_3d.wgsl` |
| Heightfield | `heightfield.rs` | `heightfield.rs`, heightfield shaders |
| Rendering | - | `fluid_renderer.rs`, fluid shaders |

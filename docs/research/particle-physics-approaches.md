# Particle Physics Simulation Approaches - Research Documentation

*Last updated: December 2025*

This document captures research on particle physics simulation approaches for falling sand / fluid games. Reference this instead of repeating web searches.

---

## Table of Contents
1. [What Noita Does](#what-noita-does)
2. [Simulation Approaches Comparison](#simulation-approaches-comparison)
3. [Rust Libraries](#rust-libraries)
4. [Alternative Languages/Frameworks](#alternative-languagesframeworks)
5. [GPU Compute Options](#gpu-compute-options)
6. [Key Tutorials & Resources](#key-tutorials--resources)
7. [Performance Benchmarks](#performance-benchmarks)

---

## What Noita Does

Source: [GDC Talk - Exploring the Tech and Design of Noita](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design/) (2019)

### Core Architecture
- **Chunk System:** 64x64 pixel chunks
- **Dirty Rects:** Each chunk tracks which pixels need updating (critical for performance)
- **Simple Rules:** Liquids and gases use basic cellular automata - not complex fluid dynamics
- **Pressure/Temperature:** Additional simulation layers that influence particle state and motion
- **Rigid Bodies:** Marching squares algorithm for integrating with pixel simulation

### Key Insights
- World is divided into chunks for **memory/loading**, but simulation crosses chunk boundaries
- Multi-threading is "really hard" for falling sand - they acknowledge limitations
- The hard part is mixing rigid body physics with pixel simulation
- Performance comes from dirty rects and sleeping chunks, not fancy algorithms

### What They DON'T Do
- No SPH (Smoothed Particle Hydrodynamics)
- No continuous fluid simulation
- No per-particle velocity vectors stored
- Pressure is calculated, not stored per-cell

---

## Simulation Approaches Comparison

### 1. Cellular Automata (CA)
**What it is:** Grid of cells, each cell updates based on neighbor states

**Pros:**
- Simple to implement
- Excellent performance (O(n) per active cell)
- Good for falling sand, powder-like materials
- Easy to parallelize per-chunk

**Cons:**
- Information locality problem (cells can't "see" far)
- Water leveling is hard (no pressure concept)
- No continuous motion

**Best for:** Falling sand games, Noita-style, powder toy

**Example rule:**
```
if cell_below is empty: fall down
else if cell_below_left is empty: slide left
else if cell_below_right is empty: slide right
```

### 2. CA with Pressure Field
**What it is:** CA + separate pressure calculation pass

**Pros:**
- Fixes water leveling problem
- Still relatively simple
- Can be added incrementally to existing CA

**Cons:**
- Two passes per frame (pressure calc + CA update)
- Pressure propagation can be slow for large bodies
- Cross-chunk pressure is tricky

**Best for:** Games needing realistic water behavior without full fluid sim

**Pressure calculation:**
```
pressure[x,y] = count of liquid cells above
flow_direction = pressure gradient (high → low)
```

### 3. SPH (Smoothed Particle Hydrodynamics)
**What it is:** Continuous fluid simulation using particles with influence radius

**Pros:**
- Physically accurate fluid behavior
- Handles splashing, waves, mixing
- Well-studied algorithm

**Cons:**
- Computationally expensive (O(n²) naive, O(n log n) with spatial hashing)
- Requires tuning many parameters
- Overkill for pixel-art games

**Best for:** Realistic fluid simulation, scientific visualization

**Key papers:**
- "Particle-based fluid simulation for interactive applications" (Müller et al., 2003)
- "Divergence-Free SPH" (DFSPH) for incompressibility

### 4. PBD (Position Based Dynamics)
**What it is:** Game-friendly physics using position constraints

**Pros:**
- Stable and controllable
- Good for real-time games
- Handles fluid + soft body + cloth

**Cons:**
- Less physically accurate than SPH
- Still more complex than CA

**Best for:** Games needing believable (not accurate) physics

### 5. GPU Compute Shaders
**What it is:** Run simulation on GPU for massive parallelism

**Pros:**
- Handle millions of particles
- 10-100x speedup for embarrassingly parallel problems

**Cons:**
- CPU↔GPU data transfer overhead
- Complex debugging
- Not all algorithms parallelize well (CA has dependencies)

**Best for:** Large-scale simulations, scientific computing

---

## Rust Libraries

### Salva (dimforge)
**URL:** https://github.com/dimforge/salva

**What it is:** 2D/3D particle-based fluid simulation

**Features:**
- SPH and PBF (Position Based Fluids)
- Pressure solvers: DFSPH, IISPH
- Viscosity: DFSPH, Artificial, XSPH
- Surface tension support
- Rapier integration for rigid-fluid coupling
- WASM/WebGL support

**Status:** Active (updated Feb 2025)

**Use case:** When you need proper fluid simulation, not pixel-based

**Example:**
```rust
use salva2d::prelude::*;

let mut fluid = Fluid::new(Vec::new(), particle_radius, density);
let mut world = LiquidWorld::new(SolverParameters::default());
world.add_fluid(fluid);
world.step();
```

### Rapier (dimforge)
**URL:** https://rapier.rs/

**What it is:** Rigid body + collision detection physics

**Features:**
- 2D and 3D
- Continuous collision detection
- Joints, motors, constraints
- WASM support

**Status:** Very active, production-ready

**Use case:** Rigid body physics (crates, vehicles, characters), NOT fluids

### WGPU
**URL:** https://wgpu.rs/

**What it is:** WebGPU implementation in Rust

**Use case:** GPU compute shaders for particle simulation

**Compute shader pattern:**
```rust
// Ping-pong buffers for particle state
let buffer_a = device.create_buffer(...);
let buffer_b = device.create_buffer(...);

// Each frame: read from A, write to B, swap
compute_pass.dispatch_workgroups(width/8, height/8, 1);
std::mem::swap(&mut buffer_a, &mut buffer_b);
```

### bevy-sph
**URL:** https://github.com/AOS55/bevy-sph

**What it is:** SPH fluid simulation in Bevy game engine

**Status:** Experimental

**Use case:** If using Bevy and need fluid simulation

---

## Alternative Languages/Frameworks

### Taichi Lang (Python)
**URL:** https://www.taichi-lang.org/

**What it is:** Python-like language that compiles to GPU

**Pros:**
- Python syntax
- Auto-parallelization to GPU
- Billion-particle demos
- Great for prototyping physics

**Cons:**
- Different deployment story than Rust
- Learning curve for Taichi-specific patterns

**Example:**
```python
import taichi as ti
ti.init(arch=ti.gpu)

@ti.kernel
def update_particles():
    for i in particles:
        particles[i].velocity += gravity * dt
        particles[i].position += particles[i].velocity * dt
```

**Best resources:**
- [Taichi Fluid Collection](https://github.com/houkensjtu/taichi-fluid)
- [GeoTaichi](https://www.sciencedirect.com/science/article/abs/pii/S0010465524001425) - 2024 geophysics simulation

### LiquidFun (C++)
**URL:** https://google.github.io/liquidfun/

**What it is:** Box2D extension for particle fluids

**Pros:**
- Battle-tested (Google)
- Box2D integration
- JavaScript bindings

**Cons:**
- C++ (no Rust bindings)
- Last major update: 2014
- May be unmaintained

### Unity DOTS / ECS
**Use case:** If switching to Unity makes sense for your project

**Pros:**
- Visual editor
- Large ecosystem
- Good performance with Jobs system

**Cons:**
- C# (not Rust)
- Licensing costs
- Different development model

### Godot 4 with Compute Shaders
**Use case:** If Godot fits your project better

**Pros:**
- Open source
- GDScript or C#
- Built-in compute shader support

**Cons:**
- Less mature than Unity
- Different ecosystem

---

## GPU Compute Options

### WebGPU / WGPU
**Browser support:** Chrome, Edge, Firefox (2024+)

**Rust crate:** `wgpu`

**Key concepts:**
- **Compute shaders:** Programs that run on GPU for arbitrary computation
- **Workgroups:** Threads that share memory (typical size: 64)
- **Ping-pong buffers:** Read from texture A, write to texture B, swap

**Falling sand pattern:**
```wgsl
@compute @workgroup_size(8, 8)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let current = textureLoad(input, id.xy);
    let below = textureLoad(input, id.xy + vec2(0, 1));

    // Simple CA rule
    if (current.material == SAND && below.material == AIR) {
        textureStore(output, id.xy, AIR);
        textureStore(output, id.xy + vec2(0, 1), current);
    }
}
```

**Challenges:**
- Race conditions (two particles wanting same destination)
- Chunk boundaries
- Data transfer overhead

**Resources:**
- [WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html)
- [Reaction-Diffusion in WebGPU](https://tympanus.net/codrops/2024/05/01/reaction-diffusion-compute-shader-in-webgpu/)

### CUDA (NVIDIA only)
**Use case:** Maximum GPU performance, desktop only

**Not recommended for:** Cross-platform games

---

## Key Tutorials & Resources

### Falling Sand Specific
1. **[W-Shadow: Falling Sand Water Simulation](https://w-shadow.com/blog/2009/09/29/falling-sand-style-water-simulation/)**
   - Basic CA rules for water
   - Mentions pressure as extension
   - Source code available

2. **[Making Sandspiel](https://maxbittker.com/making-sandspiel/)**
   - WebGL + WASM implementation
   - Navier-Stokes for wind
   - Cross-system (GPU↔CPU) challenges

3. **[LeoTheLegion Series](https://leothelegion.net/2023/10/31/adding-water-to-my-falling-sand-simulator/)**
   - Adding water to CA
   - Density-based interactions
   - Practical tutorial

4. **[Winter.dev Falling Sand](https://winter.dev/articles/falling-sand)**
   - Game-focused approach
   - Simple rules

### Fluid Simulation Theory
1. **[GPU Gems: Fast Fluid Dynamics](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)**
   - Navier-Stokes on GPU
   - Pressure, advection, diffusion

2. **[Red Blob Games: Water Levels](https://www.redblobgames.com/x/1718-water-levels/)**
   - The fundamental limitation of CA for water
   - Why pressure is needed

### Noita Specific
1. **[GDC Talk (YouTube)](https://www.youtube.com/watch?v=prXuyMCgbTc)**
   - Official deep-dive
   - Chunk system, dirty rects, rigid bodies

2. **[80.lv Article](https://80.lv/articles/noita-a-game-based-on-falling-sand-simulation)**
   - Summary of techniques

---

## Performance Benchmarks

### Typical CA Performance (CPU, single-threaded)
| Grid Size | Cells | Expected FPS | Notes |
|-----------|-------|--------------|-------|
| 256×256 | 65K | 30-60 | Easily achievable |
| 512×512 | 262K | 15-30 | Needs optimization |
| 1024×768 | 786K | 5-15 | Needs dirty rects + sleeping |
| 1920×1080 | 2M | <5 | Needs GPU or heavy optimization |

### With Dirty Rects (only update changed regions)
| Active % | Effective Cells | FPS Multiplier |
|----------|-----------------|----------------|
| 100% | All | 1x |
| 10% | 1/10 | ~8x |
| 1% | 1/100 | ~50x |

**Key insight:** Most frames, <10% of cells are active. Dirty rects are essential.

### GPU Compute
| Grid Size | Cells | GPU FPS | CPU FPS |
|-----------|-------|---------|---------|
| 1024×1024 | 1M | 60+ | 5-10 |
| 2048×2048 | 4M | 30-60 | <5 |
| 4096×4096 | 16M | 15-30 | <1 |

**Note:** Data transfer overhead can negate GPU advantage for small grids.

---

## Recommendations by Use Case

### "I want Noita-like physics"
**Approach:** CA with dirty rects + pressure field
**Libraries:** Custom Rust implementation
**Effort:** Medium

### "I want realistic water"
**Approach:** Salva (SPH/PBF)
**Libraries:** `salva2d` + `rapier2d`
**Effort:** Low (library does heavy lifting)

### "I want maximum scale (millions of particles)"
**Approach:** GPU compute shaders
**Libraries:** `wgpu` with custom shaders
**Effort:** High

### "I want to prototype quickly"
**Approach:** Taichi Lang
**Libraries:** `taichi` Python package
**Effort:** Low (but different ecosystem)

### "I just want gold to sink in water"
**Approach:** Simple CA rules
**Libraries:** None needed
**Effort:** Very low

---

## Quick Reference: The Simplest Water That Works

```rust
fn update_water(world: &mut World, x: i32, y: i32) {
    // 1. Fall if space below
    if world.get(x, y+1) == Air {
        world.swap(x, y, x, y+1);
        return;
    }

    // 2. Sink through lighter liquids
    let below = world.get(x, y+1);
    if below.is_liquid() && below.density() < WATER_DENSITY {
        world.swap(x, y, x, y+1);
        return;
    }

    // 3. Spread sideways (random direction to prevent bias)
    let dirs = if random() { [-1, 1] } else { [1, -1] };
    for dx in dirs {
        if world.get(x+dx, y) == Air {
            world.swap(x, y, x+dx, y);
            return;
        }
    }
}
```

This handles 90% of water behavior. Add pressure only if leveling doesn't work.

# Pressure-Based Particle Physics Engine

## Overview

Replace the current ad-hoc cellular automata simulation with a proper pressure-based particle physics system. This will be the foundation for all game mechanics including sluice/trommel gold panning.

**Core Insight:** The current approach fights against physics. A pressure-based system naturally produces:
- Water leveling (pressure equilibrium)
- Material-specific flow (viscosity)
- Erosion mechanics (velocity-based particle detachment)
- Density-based settling (gold sinks, soil washes away)

## Problem Statement

Current issues with the cellular automata approach:
1. Water forms "mountains" - no pressure equalization
2. Chunk boundaries break information flow
3. Ad-hoc fixes (auto-leveling) create artifacts
4. No velocity concept = no erosion mechanics
5. Can't simulate sluice/trommel core gameplay

## Research Summary

### What Noita Does
From the [GDC Talk](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design/):
- 64x64 chunks with dirty rects for performance
- Liquids use simple CA rules + pressure/temperature simulation
- World processed in one pass (chunks are for loading/memory, not simulation boundaries)
- Rigid body integration via marching squares
- Multi-threading is "hard" - they divide into chunks but acknowledge limitations

### Available Approaches

| Approach | Performance | Accuracy | Complexity | Best For |
|----------|-------------|----------|------------|----------|
| **Cellular Automata + Pressure** | Excellent | Good | Medium | Falling sand games |
| **SPH (Smoothed Particle Hydrodynamics)** | Medium | Excellent | High | Realistic fluids |
| **PBD (Position Based Dynamics)** | Good | Good | Medium | Games |
| **GPU Compute (WebGPU/WGPU)** | Excellent | Varies | High | Large scale |
| **Taichi Lang** | Excellent | Excellent | Medium | Research/prototyping |

### Existing Libraries

#### Rust Options
1. **[Salva](https://github.com/dimforge/salva)** - 2D/3D fluid simulation (SPH/PBF)
   - Pros: Mature, integrates with Rapier, WASM support
   - Cons: Designed for realistic fluids, may be overkill

2. **[Rapier](https://rapier.rs/)** - Rigid body physics
   - Pros: Excellent, well-maintained
   - Cons: No fluid simulation built-in

3. **WGPU Compute Shaders** - GPU-based simulation
   - Pros: Massive parallelism, cross-platform
   - Cons: Complex to implement, debugging difficult

#### Alternative Languages
1. **[Taichi Lang](https://www.taichi-lang.org/)** (Python)
   - Pros: Python syntax, auto GPU compilation, billion-particle demos
   - Cons: Different ecosystem, deployment complexity

2. **[LiquidFun](https://google.github.io/liquidfun/)** (C++)
   - Pros: Box2D-based, particle fluids
   - Cons: C++, last major update 2014

3. **Unity DOTS** / **Godot 4 Compute**
   - Pros: Full game engine, visual tools
   - Cons: Major ecosystem change

## Recommended Approach

### Option A: Enhanced CA with Pressure Field (Recommended)

Stay in Rust, enhance current system with proper physics:

```
┌─────────────────────────────────────────────────┐
│                 Simulation Loop                  │
├─────────────────────────────────────────────────┤
│ 1. Calculate pressure field (water depth)        │
│ 2. Calculate velocity from pressure gradient     │
│ 3. Move particles based on velocity + gravity    │
│ 4. Apply erosion (velocity vs cohesion)          │
│ 5. Handle material interactions                  │
└─────────────────────────────────────────────────┘
```

**Why this approach:**
- Builds on existing code
- Good enough for game (not scientific simulation)
- Performant on CPU
- Can add GPU acceleration later

### Option B: GPU Compute (WGPU)

Move simulation to GPU compute shaders:

**Pros:**
- Massive parallelism (millions of particles)
- Cross-platform (WebGPU works in browsers)
- Future-proof

**Cons:**
- Significant rewrite
- Complex debugging
- Data transfer overhead

### Option C: Hybrid with Salva

Use Salva for water, keep CA for solids:

**Pros:**
- Realistic fluid behavior out of the box
- Rapier integration for rigid bodies

**Cons:**
- Two simulation systems to synchronize
- May be overkill

## Technical Design (Option A)

### Core Data Structures

```rust
// sim/src/particle.rs
#[derive(Clone, Copy)]
pub struct Particle {
    pub material: Material,
    pub velocity: Vec2,      // NEW: velocity vector
    pub pressure: f32,       // NEW: local pressure
}

// Material properties
impl Material {
    pub fn density(&self) -> f32;        // Mass per unit
    pub fn viscosity(&self) -> f32;      // Flow resistance (0=water, 1=rock)
    pub fn cohesion(&self) -> f32;       // Erosion resistance
    pub fn friction(&self) -> f32;       // Angle of repose
}
```

### Material Properties

| Material | Density | Viscosity | Cohesion | Behavior |
|----------|---------|-----------|----------|----------|
| Air      | 0       | 0         | 0        | Empty space |
| Water    | 1.0     | 0.01      | 0        | Flows freely, levels flat |
| Mud      | 1.5     | 0.3       | 0.2      | Viscous flow, erodes easily |
| Soil     | 2.0     | 0.8       | 0.5      | Slumps, doesn't level |
| Rock     | 3.0     | 1.0       | 1.0      | Static, doesn't move |
| Gold     | 5.0     | 0.9       | 0.8      | Sinks through everything |

### Pressure Calculation

```rust
// For each water cell, pressure = depth (cells above)
fn calculate_pressure(world: &World, x: i32, y: i32) -> f32 {
    let mut pressure = 0.0;
    let mut check_y = y - 1;

    while world.get_material(x, check_y).is_liquid() {
        pressure += world.get_material(x, check_y).density();
        check_y -= 1;
    }

    pressure
}

// Velocity from pressure gradient
fn calculate_velocity(world: &World, x: i32, y: i32) -> Vec2 {
    let p_left = get_pressure(world, x - 1, y);
    let p_right = get_pressure(world, x + 1, y);
    let p_up = get_pressure(world, x, y - 1);
    let p_down = get_pressure(world, x, y + 1);

    Vec2::new(
        (p_left - p_right) * FLOW_RATE,
        (p_up - p_down) * FLOW_RATE + GRAVITY,
    )
}
```

### Erosion Mechanics

```rust
fn apply_erosion(world: &mut World, x: i32, y: i32, velocity: Vec2) {
    let speed = velocity.length();
    let material = world.get_material(x, y);

    // Check neighbors for erodible materials
    for (nx, ny) in neighbors(x, y) {
        let neighbor = world.get_material(nx, ny);

        // Erosion chance based on velocity vs cohesion
        if speed > neighbor.cohesion() * EROSION_THRESHOLD {
            let erosion_chance = (speed - neighbor.cohesion()) / speed;
            if random() < erosion_chance {
                // Detach particle - becomes suspended in flow
                world.set_material(nx, ny, Material::Air);
                // Add suspended particle to water (simplified: just becomes mud)
                if neighbor == Material::Soil {
                    world.set_material(x, y, Material::Mud);
                }
            }
        }
    }
}
```

### Sluice Mechanics (Core Gameplay)

```
Water flow direction →
┌─────────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  Water surface
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │  Mixed material input
├─────────────────────────────────────┤
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  Riffles (barriers)
│     ████░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│         ████░░░░░░░░░░░░░░░░░░░░░░ │
│             ████                    │
├─────────────────────────────────────┤
│ ●●●● Gold settles first (high density)
│      ▓▓▓▓ Soil settles later
│           ░░░░ Mud washes away
└─────────────────────────────────────┘
```

**How it works:**
1. Water enters with suspended particles
2. Riffles create turbulence zones (low velocity)
3. Heavy particles (gold) settle in low-velocity zones
4. Light particles (mud) wash over and exit

## Implementation Phases

### Phase 1: Pressure Field
- Add pressure calculation to World
- Pressure = sum of liquid density above
- Update every frame for liquid cells

### Phase 2: Velocity-Based Movement
- Calculate velocity from pressure gradient
- Move liquids based on velocity, not just rules
- Viscosity dampens velocity

### Phase 3: Material Behaviors
- Implement angle of repose for solids
- Soil slumps but doesn't flow flat
- Mud flows slowly
- Water flows freely

### Phase 4: Erosion
- Velocity-based particle detachment
- Soil → Mud when wet and flowing
- Suspended particles in water

### Phase 5: Sluice Mechanics
- Riffle placement
- Turbulence zones
- Gold collection

### Phase 6: Optimization
- GPU acceleration (WGPU compute)
- Spatial partitioning
- Sleep detection for static regions

## Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| World Size | 256x256 | 1024x768+ |
| FPS | 21 | 60 |
| Particles | ~65K | ~500K |
| Fluid Physics | Broken | Realistic |

## Alternative: Full GPU Rewrite

If Option A proves insufficient, move to WGPU compute:

```rust
// Compute shader pseudocode
@compute @workgroup_size(8, 8)
fn update_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    // Read from input texture
    let particle = textureLoad(input, vec2<i32>(x, y));

    // Calculate pressure from neighbors
    let pressure = calculate_pressure(x, y);

    // Calculate velocity
    let velocity = pressure_gradient(x, y) + gravity;

    // Apply movement
    let new_pos = apply_velocity(particle, velocity);

    // Write to output texture
    textureStore(output, new_pos, particle);
}
```

## Decision Points

1. **Start with Option A** - enhanced CA with pressure
2. **If performance insufficient** → Add WGPU compute shaders
3. **If accuracy insufficient** → Consider Salva for water physics
4. **If complete rewrite needed** → Evaluate Taichi Lang for rapid prototyping

## Acceptance Criteria

- [ ] Water naturally levels to flat surface (pressure equilibrium)
- [ ] Soil slumps at angle of repose, doesn't flow flat
- [ ] Mud flows slowly, more viscous than water
- [ ] Gold sinks through water and mud
- [ ] Water flow velocity affects particle movement
- [ ] Erosion: fast water detaches mud/soil particles
- [ ] 60 FPS at 1024x768 resolution
- [ ] Sluice mechanics work (gold settles in riffles)

## References

### Tutorials & Articles
- [W-Shadow: Falling Sand Water Simulation](https://w-shadow.com/blog/2009/09/29/falling-sand-style-water-simulation/)
- [Making Sandspiel](https://maxbittker.com/making-sandspiel/)
- [LeoTheLegion: Adding Water to Falling Sand](https://leothelegion.net/2023/10/31/adding-water-to-my-falling-sand-simulator/)

### Libraries
- [Salva - Rust Fluid Simulation](https://github.com/dimforge/salva)
- [LiquidFun - Box2D Fluids](https://google.github.io/liquidfun/)
- [Taichi Lang](https://www.taichi-lang.org/)

### Talks & Papers
- [Noita GDC Talk](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design/)
- [GPU Gems: Fast Fluid Dynamics](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)

### GPU Compute
- [WebGPU Fundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html)
- [WGPU-rs](https://wgpu.rs/)

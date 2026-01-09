<!-- TODO: Review - part of future planning -->

# 3D Excavation System - Design Plan

## Overview

Realistic 3D dirt/sand/rock excavation that integrates with the existing FLIP fluid sluice simulation. This plan extends `plans/mining-materials-system.md` with 3D-specific mechanics, focusing on practical implementation paths.

**Goal**: Dig terrain in 3D, see realistic collapse physics, transport excavated material through water to the sluice for gold separation.

---

## Architecture Options

### Option A: Extend Existing 2D FLIP (Recommended for MVP)

Use the mature 2D simulation (`crates/sim/`) with pseudo-3D rendering:

```
┌─────────────────────────────────────────────────┐
│  2D FLIP Simulation (proven, fast)              │
│  + Heightfield terrain (2.5D)                   │
│  + Isometric/oblique rendering                  │
│  + Material layers per column                   │
└─────────────────────────────────────────────────┘
```

**Pros**: Faster to implement, known performance characteristics
**Cons**: No true caves/tunnels, limited vertical interaction

### Option B: Full 3D FLIP Simulation

Use the 3D simulation (`crates/sim3d/`) with true voxel terrain:

```
┌─────────────────────────────────────────────────┐
│  3D FLIP Simulation (newer, less mature)        │
│  + Voxel terrain (full 3D)                      │
│  + 3D perspective rendering                     │
│  + True underground spaces possible             │
└─────────────────────────────────────────────────┘
```

**Pros**: True 3D excavation, caves, tunnels
**Cons**: Higher development cost, performance challenges at scale

### Option C: Hybrid Approach

Heightfield terrain + 3D fluid + DEM particles:

```
┌─────────────────────────────────────────────────┐
│  Terrain: Heightfield (efficient storage)       │
│  Fluid: 3D FLIP (water flows in 3D)             │
│  Granular: 3D DEM (excavated particles)         │
│  Coupling: One-way drag + volume exclusion      │
└─────────────────────────────────────────────────┘
```

---

## Implementation Approaches

### Approach 1: wgpu Compute + Custom DEM (Most Practical)

Build on existing wgpu infrastructure with custom granular physics:

**Architecture**:
```rust
// crates/sim/src/excavation.rs

pub struct ExcavationSystem {
    /// Terrain as heightfield per column
    terrain_heights: Vec<f32>,        // [x * depth + z] -> height
    terrain_materials: Vec<Vec<TerrainLayer>>,  // layers per column

    /// Active granular particles (excavated, not yet settled)
    active_particles: ParticleBuffer, // GPU buffer

    /// Spatial hash for DEM collision
    spatial_hash: SpatialHash3D,
}

struct TerrainLayer {
    material: TerrainMaterial,
    thickness: f32,
    gold_concentration: f32,
}
```

**GPU Kernels** (WGSL):
1. `excavate.wgsl` - Remove terrain, spawn particles
2. `dem_contacts.wgsl` - Spatial hash + contact detection
3. `dem_forces.wgsl` - Spring-dashpot + friction forces
4. `particle_integrate.wgsl` - Velocity/position update
5. `terrain_collapse.wgsl` - Angle of repose stability

**Performance Target**: 100k-500k granular particles at 60 FPS (GPU)

### Approach 2: MPM for Everything (Better Physics, Higher Cost)

Replace FLIP with unified MPM that handles both fluid and granular:

**From Research**: MPM naturally handles:
- Fluid-solid coupling
- Material transitions (dry sand → wet mud)
- Yield stress fluids (mud slumping)

**Reference**: Niall's MPM guide (nialltl.neocities.org/articles/mpm_guide)

**Cons**: Requires rewriting simulation core, different parameter tuning

### Approach 3: Taichi Integration (Production Quality, Complex Setup)

Use Taichi for simulation, Rust for game logic:

**Workflow**:
1. Write simulation kernels in Taichi Python
2. Compile to AOT modules
3. Load via `taichi-runtime` crate
4. Share buffers between Taichi and wgpu

**Cons**: Two-language development, buffer sharing complexity

---

## Recommended Implementation Path

### Phase 1: Heightfield Excavation (MVP)

**Goal**: Dig 2.5D terrain, see realistic collapse

**Scope**:
- Heightfield terrain with material layers
- Shovel tool removes material from top layer
- Angle of repose collapse
- Excavated material → backpack (particles removed from sim)

**Files to Create/Modify**:
```
crates/sim/src/
├── terrain.rs      # NEW: Heightfield terrain system
├── excavation.rs   # NEW: Tool interaction, particle spawning
└── collapse.rs     # NEW: Angle of repose, cascade logic

crates/game/src/
├── tools/
│   ├── mod.rs      # NEW: Tool system
│   └── shovel.rs   # NEW: Shovel implementation
└── terrain_render.rs  # NEW: Heightfield rendering
```

**Key Types**:
```rust
// crates/sim/src/terrain.rs

pub struct HeightfieldTerrain {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,

    /// Height at each (x, z) column
    heights: Vec<f32>,

    /// Material stack per column (bottom to top)
    layers: Vec<Vec<TerrainLayer>>,
}

pub struct TerrainLayer {
    pub material: TerrainMaterial,
    pub height: f32,  // thickness of this layer
    pub gold_ppm: f32, // parts per million
}

#[derive(Clone, Copy)]
pub enum TerrainMaterial {
    Soil,
    Sand,
    Gravel,
    Clay,
    Bedrock,
    Permafrost,
}

impl HeightfieldTerrain {
    /// Excavate terrain at position, return spawned particles
    pub fn excavate(
        &mut self,
        position: Vec3,
        radius: f32,
        depth: f32,
        tool: &Tool,
    ) -> Vec<ExcavatedParticle>;

    /// Check and apply collapse based on angle of repose
    pub fn apply_collapse(&mut self) -> CollapseResult;
}
```

### Phase 2: Water Flow Integration

**Goal**: Water transports excavated material

**Scope**:
- 2D FLIP water simulation (existing)
- Excavated particles enter water
- Drag force coupling (water → particles)
- Particles carried toward sluice

**Coupling Model** (simplified):
```rust
// One-way coupling: fluid affects particles, particles don't affect fluid

fn apply_fluid_drag(
    particle: &mut Particle3D,
    fluid_grid: &Grid,
    dt: f32,
) {
    // Sample fluid velocity at particle position
    let fluid_vel = fluid_grid.sample_velocity(particle.position.xz());

    // Compute relative velocity
    let rel_vel = Vec3::new(fluid_vel.x, 0.0, fluid_vel.y) - particle.velocity;

    // Stokes drag (small particles) or Ergun (dense packing)
    let drag_coeff = compute_drag_coefficient(particle, fluid_grid);
    let drag_force = drag_coeff * rel_vel;

    particle.velocity += drag_force * dt / particle.mass();
}
```

### Phase 3: Granular DEM on GPU

**Goal**: Realistic pile formation and friction

**Scope**:
- GPU spatial hashing (existing pattern from 2D)
- Contact detection via hash grid
- Spring-dashpot normal force
- Coulomb friction model

**Contact Model**:
```wgsl
// dem_contacts.wgsl

fn compute_contact_force(
    p1: Particle,
    p2: Particle,
    overlap: f32,
) -> vec3<f32> {
    let stiffness = 3000.0;
    let damping = 50.0;
    let friction_coeff = 0.5;

    let normal = normalize(p2.position - p1.position);
    let rel_vel = p2.velocity - p1.velocity;

    // Normal force (spring + damping)
    let vn = dot(rel_vel, normal);
    let fn_mag = stiffness * overlap - damping * vn;
    let fn = max(fn_mag, 0.0) * normal;

    // Tangential force (friction)
    let vt = rel_vel - vn * normal;
    let ft_mag = min(length(vt) * damping, friction_coeff * fn_mag);
    let ft = -normalize(vt) * ft_mag;

    return fn + ft;
}
```

### Phase 4: Sluice Integration

**Goal**: Feed excavated material to sluice, separate gold

**Scope**:
- Backpack → Sluice material transfer
- Material spawns as particles in sluice sim
- Density-based separation (existing sediment physics)
- Gold collection tracking

**Interface**:
```rust
// crates/inventory/src/transfer.rs

pub struct MaterialPacket {
    pub contents: HashMap<TerrainMaterial, f32>,  // material → volume
    pub gold_particles: Vec<GoldParticle>,
}

impl Backpack {
    pub fn dump_to_sluice(&mut self, amount: f32) -> MaterialPacket {
        // Remove from backpack, return as packet
        // Sluice simulation will spawn particles from packet
    }
}
```

---

## User Interaction Design

### Camera System

**Recommendation**: Isometric perspective with rotation

```
     Y (up)
     │
     │   ╱ X
     │ ╱
     └─────── Z

    Camera looks down at ~45° angle
    Player can rotate in 90° increments
```

**Controls**:
- `Q`/`E` - Rotate camera 90°
- Mouse scroll - Zoom in/out
- Mouse drag (middle button) - Pan

### Tool Targeting

**Method**: Raycast to heightfield surface

```rust
fn get_dig_target(
    camera: &Camera,
    mouse_pos: Vec2,
    terrain: &HeightfieldTerrain,
) -> Option<Vec3> {
    let ray = camera.screen_to_ray(mouse_pos);
    terrain.raycast(ray.origin, ray.direction)
}
```

**Visual Feedback**:
- Circular cursor projected onto terrain surface
- Cursor color: Green (can dig), Red (too hard), Yellow (partial)
- Cursor size matches tool radius

### Excavation Input

**Click-to-dig** (discrete scoops):
- Left click → One scoop at cursor location
- Material removed → Backpack
- 0.3s cooldown between scoops

**Hold-to-dig** (continuous, optional enhancement):
- Hold left click → Continuous scooping at 3 scoops/second
- Visual feedback: Dirt particles flying into backpack
- Audio: Rhythmic digging sounds

---

## Material Properties (3D Extension)

Extends `mining-materials-system.md` with 3D-specific properties:

```rust
impl TerrainMaterial {
    pub fn angle_of_repose(&self) -> f32 {
        match self {
            Self::Sand => 30.0_f32.to_radians(),
            Self::Soil => 40.0_f32.to_radians(),
            Self::Gravel => 35.0_f32.to_radians(),
            Self::Clay => 45.0_f32.to_radians(),
            Self::Bedrock => 90.0_f32.to_radians(), // vertical cliffs ok
            Self::Permafrost => 70.0_f32.to_radians(),
        }
    }

    pub fn dig_hardness(&self) -> f32 {
        match self {
            Self::Sand => 1.0,
            Self::Soil => 2.0,
            Self::Gravel => 3.0,
            Self::Clay => 2.5,
            Self::Bedrock => 10.0,  // requires pickaxe
            Self::Permafrost => 8.0,
        }
    }

    pub fn gold_probability(&self) -> f32 {
        match self {
            Self::Sand => 0.0,
            Self::Soil => 0.001,
            Self::Gravel => 0.05,   // primary gold host
            Self::Clay => 0.002,
            Self::Bedrock => 0.02,  // gold veins
            Self::Permafrost => 0.01,
        }
    }
}
```

---

## Structural Collapse System

### Angle of Repose Algorithm

```rust
impl HeightfieldTerrain {
    pub fn apply_collapse(&mut self) -> Vec<CollapseEvent> {
        let mut events = Vec::new();
        let mut changed = true;

        while changed {
            changed = false;

            for x in 0..self.width {
                for z in 0..self.depth {
                    let height = self.heights[x * self.depth + z];
                    let material = self.top_material(x, z);
                    let max_slope = material.angle_of_repose();

                    // Check 4-connected neighbors
                    for (nx, nz) in self.neighbors(x, z) {
                        let neighbor_height = self.heights[nx * self.depth + nz];
                        let slope = (height - neighbor_height).atan2(self.cell_size);

                        if slope > max_slope {
                            // Collapse: move material to neighbor
                            let transfer = self.cell_size * (slope - max_slope).tan() * 0.5;
                            self.transfer_material(x, z, nx, nz, transfer);

                            events.push(CollapseEvent {
                                from: (x, z),
                                to: (nx, nz),
                                amount: transfer,
                                material,
                            });

                            changed = true;
                        }
                    }
                }
            }
        }

        events
    }
}
```

### Cascade Collapse Visualization

```
Before dig:          After dig:           After collapse:

  ████████            ████████             ███████
 ██████████          ███  ████            ████████
████████████        ████  ████           ██████████
████████████        ████  ████           ████████████
                         ↑ dig             ↑ material flows down
```

---

## Performance Considerations

### Particle Budget

From benchmarks in existing plan:
- 15,000 particles @ 60 FPS (CPU, 256x192 grid)
- Target: 100k-500k with GPU compute

**Budget Allocation**:
| System | Particles | Notes |
|--------|-----------|-------|
| Excavated (active) | 10,000 | Falling, rolling |
| Water-borne | 5,000 | Being transported |
| Sluice | 10,000 | High-res separation |
| **Total active** | **25,000** | CPU-manageable |

### Zone System (from existing plan)

```
┌─────────────────────────────────────────────────┐
│  DORMANT - Heightfield storage only             │
│  ┌─────────────────────────────────────────┐    │
│  │  HYBRID - Coarse collapse checks        │    │
│  │  ┌─────────────────────────────────┐    │    │
│  │  │  ACTIVE - Full particle sim     │    │    │
│  │  │  ~50m radius around player      │    │    │
│  │  └─────────────────────────────────┘    │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

**Transition Rules**:
- Active → Hybrid: Particles settle to heightfield
- Hybrid → Dormant: Remove particle buffer, keep heightfield
- Dormant → Hybrid: Allocate particle buffer, no particles yet
- Hybrid → Active: Can spawn particles from terrain

---

## Critical Decisions Needed

### Decision 1: Full 3D vs Heightfield

**Question**: Should terrain support caves/tunnels, or only surface excavation?

| Option | Pros | Cons |
|--------|------|------|
| **Heightfield (2.5D)** | Simpler, faster, predictable | No caves, limited vertical play |
| **Full 3D voxels** | True underground mining | Complex, memory-heavy |

**Recommendation**: Start with heightfield, add cave system later as enhancement.

### Decision 2: Particle Destination

**Question**: Where do excavated particles go?

| Option | Pros | Cons |
|--------|------|------|
| **Instant to backpack** | Simple, predictable | Less visual satisfaction |
| **Physics pile nearby** | More realistic | Clutter, performance |
| **Hybrid** | Best of both | More complex |

**Recommendation**: Hybrid - small particle burst animation, then snap to backpack.

### Decision 3: Fluid Coupling Fidelity

**Question**: How tightly coupled should water and granular be?

| Option | Pros | Cons |
|--------|------|------|
| **One-way (fluid→particles)** | Simple, fast | Water doesn't respond to sediment |
| **Two-way** | More realistic | Complex, slower |
| **Drift-flux (existing)** | Proven in sluice | Requires shared grid |

**Recommendation**: One-way for mining, two-way in sluice (existing behavior).

---

## File Structure

```
crates/
├── sim/src/
│   ├── terrain/
│   │   ├── mod.rs           # Heightfield terrain
│   │   ├── materials.rs     # TerrainMaterial enum
│   │   ├── collapse.rs      # Angle of repose logic
│   │   └── heightfield.rs   # Height storage, raycast
│   ├── excavation/
│   │   ├── mod.rs           # Excavation system
│   │   ├── tools.rs         # Tool types and effects
│   │   └── spawning.rs      # Particle spawning from terrain
│   └── coupling/
│       └── fluid_particle.rs # Water-granular interaction
│
├── game/src/
│   ├── input/
│   │   └── tool_input.rs    # Mouse/keyboard handling
│   ├── camera/
│   │   └── isometric.rs     # Isometric camera controller
│   └── render/
│       └── terrain_render.rs # Heightfield rendering
│
└── inventory/src/
    ├── backpack.rs          # Material storage
    └── transfer.rs          # Mining ↔ Sluice transfer
```

---

## Development Phases

### Phase 1: Basic Heightfield + Shovel (1-2 weeks)
- [ ] `HeightfieldTerrain` struct with material layers
- [ ] Raycast from camera to terrain
- [ ] Shovel tool removes top layer
- [ ] Basic heightfield rendering
- [ ] Simple backpack storage

**Success**: Can dig holes in terrain, material goes to backpack

### Phase 2: Collapse Physics (1 week)
- [ ] Angle of repose calculation
- [ ] Cascade collapse algorithm
- [ ] Collapse visualization (material slides down)
- [ ] Performance optimization (limit iterations)

**Success**: Realistic avalanche when digging steep slopes

### Phase 3: Water Integration (1-2 weeks)
- [ ] Connect excavation zone to water simulation
- [ ] Water fills excavated areas
- [ ] Drag force on excavated particles
- [ ] Particles flow toward sluice output

**Success**: Dig near water, see material wash away

### Phase 4: Sluice Integration (1 week)
- [ ] Backpack dump to sluice input
- [ ] Material spawns in sluice simulation
- [ ] Gold separation using existing sediment physics
- [ ] Collection tracking and display

**Success**: Complete loop - dig → transport → sluice → gold

---

## References

### Internal
- `plans/mining-materials-system.md` - Material properties, zone system
- `crates/sim/src/dem.rs` - Existing DEM implementation
- `crates/sim/src/flip/sediment.rs` - Sediment transport physics
- `crates/sim3d/` - 3D FLIP framework (if pursuing full 3D)

### External
- [Niall's MPM Guide](https://nialltl.neocities.org/articles/mpm_guide) - MPM fundamentals
- [WebGPU Particle Collision](https://lvngd.com/blog/webgpu-building-particle-simulation-collision-detection/) - GPU spatial hashing
- [Granule-In-Cell Method](https://arxiv.org/html/2504.00745v1) - Coupled granular-fluid

### Similar Games/Systems
- **Minecraft**: Block-based excavation, simple collapse
- **Deep Rock Galactic**: Terrain destruction, cave systems
- **Teardown**: Voxel destruction, physics debris
- **Terraria**: 2D tile-based mining with gravity

---

## Changelog

- v0.1 (2025-01-06): Initial 3D excavation plan, extends mining-materials-system

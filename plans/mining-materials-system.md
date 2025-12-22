# Mining Materials System - Design Plan

## Overview

A unified APIC-MPM simulation for mining materials including granular solids, fluids,
rigid bodies, and static structures. Designed for large-world real-time simulation
with active/dormant zone management.

---

## Architecture

### Simulation Zones

```
┌─────────────────────────────────────────────────────┐
│  DORMANT - Voxel/heightmap storage, no simulation   │
│  ┌─────────────────────────────────────────────┐    │
│  │  HYBRID - Coarse stability checks, fluid    │    │
│  │           pressure propagation only         │    │
│  │  ┌─────────────────────────────────────┐    │    │
│  │  │  ACTIVE - Full APIC-MPM simulation  │    │    │
│  │  │  ~2-3 screen widths around player   │    │    │
│  │  └─────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Physics Framework: APIC-MPM

- **Fluids**: Standard MPM with low viscosity
- **Viscoplastic (mud)**: Bingham/Herschel-Bulkley yield stress model
- **Granular (sand, soil, gravel)**: Drucker-Prager yield criterion
- **Rigid bodies (rocks)**: High-stiffness particle clusters with constraints
- **Static (bedrock, permafrost)**: Grid-based obstacles, not particles

---

## Material Definitions

### Material Type Enum

```rust
enum MaterialType {
    // Fluids
    Water,

    // Granular
    Sand,
    Soil,
    Gravel,
    Mud,  // wet soil, yield-stress behavior

    // Rigid Bodies
    Rock,

    // Static/Structural
    Bedrock,
    Permafrost,

    // Precious
    Gold,
}
```

### Unified Property Structure

```rust
struct MaterialProperties {
    // === Base Properties (all materials) ===
    density: f32,              // kg/m³
    cohesion: f32,             // Pa - inter-particle bonding
    friction_angle: f32,       // degrees - internal friction

    // === Flow Behavior ===
    is_fluid: bool,            // true = flows freely
    viscosity: f32,            // Pa·s - resistance to flow
    yield_stress: f32,         // Pa - stress before flow (0 for Newtonian fluids)

    // === Granular Properties ===
    angle_of_repose: f32,      // degrees - max stable slope
    permeability: f32,         // m/s - water flow rate through material
    saturation_threshold: f32, // 0-1 - water fraction before becoming "wet"

    // === Structural Properties ===
    hardness: f32,             // 0-10 - mining difficulty
    break_threshold: f32,      // J - energy to break/fragment
    is_static: bool,           // true = immovable until broken

    // === Gold Properties ===
    gold_concentration: f32,   // 0-1 - probability of containing gold
    releases_gold_on: GoldReleaseTrigger,

    // === Transformation ===
    wet_variant: Option<MaterialType>,      // what it becomes when saturated
    fragments_into: Vec<(MaterialType, f32)>, // (type, proportion) when broken
}

enum GoldReleaseTrigger {
    None,
    WaterAgitation { velocity_threshold: f32 },
    Breaking,
    Crushing,
}
```

---

## Material Specifications

### 1. Water

| Property | Value | Notes |
|----------|-------|-------|
| density | 1000 kg/m³ | |
| is_fluid | true | |
| viscosity | 0.001 Pa·s | Very low |
| yield_stress | 0 | Newtonian fluid |

**Behaviors:**
- Flows freely, seeks lowest point
- Erodes granular materials (erosion rate: sand > soil > mud > gravel)
- Transports particles based on `water_velocity` vs `particle_weight`
- Seeps through permeable materials
- Accumulates in depressions

**Erosion Power Formula (draft):**
```
erosion_rate = water_velocity² × water_density / (particle_cohesion + 1)
```

---

### 2. Sand

| Property | Value | Notes |
|----------|-------|-------|
| density | 1600 kg/m³ | |
| cohesion | 0-100 Pa | Very low, dry sand has almost none |
| friction_angle | 30° | |
| angle_of_repose | 30-35° | |
| permeability | 0.01 m/s | High - water drains quickly |
| saturation_threshold | 0.3 | |
| wet_variant | None | Wet sand = same but heavier, slightly more cohesive |
| gold_concentration | 0.0 | Sand doesn't contain gold |

**Behaviors:**
- Easily disturbed, low stability
- Collapses quickly when support removed
- Water drains through rapidly
- Easily eroded and transported by water

---

### 3. Soil/Dirt

| Property | Value | Notes |
|----------|-------|-------|
| density | 1400 kg/m³ | |
| cohesion | 5000-10000 Pa | Medium bonding |
| friction_angle | 35° | |
| angle_of_repose | 40-45° | |
| permeability | 0.001 m/s | Medium |
| saturation_threshold | 0.4 | |
| wet_variant | Some(Mud) | Becomes mud when saturated |
| gold_concentration | 0.001-0.01 | Low gold content possible |

**Behaviors:**
- More stable than sand
- Transforms to mud when water content > saturation_threshold
- Medium erosion resistance
- Can contain small gold particles (paydirt component)

---

### 4. Mud

| Property | Value | Notes |
|----------|-------|-------|
| density | 1800 kg/m³ | Heavier due to water content |
| cohesion | 2000-5000 Pa | Sticky |
| friction_angle | 15° | Low internal friction when wet |
| is_fluid | false | Not a free-flowing fluid |
| viscosity | 100-1000 Pa·s | Very high |
| yield_stress | 500-2000 Pa | Must overcome to flow (slump behavior) |
| angle_of_repose | 15-25° | Low due to wetness |
| permeability | 0.0001 m/s | Very low |

**Behaviors:**
- Yield-stress fluid: stays put until force exceeds yield_stress
- "Slumps" rather than flows
- Slow movement when disturbed
- Resists water erosion better than dry soil (already saturated)
- Created from soil + water saturation

**Later Phase:**
- Dries back to soil over time without water source

---

### 5. Gravel

| Property | Value | Notes |
|----------|-------|-------|
| density | 1800 kg/m³ | |
| cohesion | 0 Pa | No bonding between pieces |
| friction_angle | 40° | High friction |
| angle_of_repose | 35-40° | |
| permeability | 0.1 m/s | Very high |
| saturation_threshold | 0.2 | |
| gold_concentration | 0.01-0.05 | Common gold host |
| releases_gold_on | WaterAgitation | Releases in sluice |

**Behaviors:**
- Coarse granular material
- High friction keeps it stable
- Water flows through easily
- Heavy, resists water transport
- Primary gold-bearing material (paydirt component)
- Represented as small rigid clusters (2-5 particles)

---

### 6. Rocks (Movable)

| Property | Value | Notes |
|----------|-------|-------|
| density | 2600 kg/m³ | |
| cohesion | 50000+ Pa | Internal bonding (doesn't break easily) |
| friction_angle | 45° | |
| is_static | false | Can move, roll, tumble |
| hardness | 6-8 | Hard to break |
| break_threshold | 10000 J | Requires significant force |
| fragments_into | [(Gravel, 0.8), (Sand, 0.2)] | |
| gold_concentration | 0.001-0.02 | Can contain gold veins |

**Behaviors:**
- Multi-particle rigid body clusters (5-20 particles)
- Maintains shape, moves as unit
- Can roll, tumble, stack
- Part of mixed substrate
- Can be broken by mining tools into gravel + sand
- May release gold when broken

**Implementation:**
- Rigid body constraints between particles
- Collision detection as compound shape
- Break when accumulated stress > break_threshold

---

### 7. Bedrock

| Property | Value | Notes |
|----------|-------|-------|
| density | 2800 kg/m³ | |
| is_static | true | Immovable |
| hardness | 9-10 | Very hard to mine |
| break_threshold | 50000 J | Requires heavy tools |
| fragments_into | [(Rock, 0.5), (Gravel, 0.4), (Sand, 0.1)] | |
| gold_concentration | 0.02-0.1 | Often gold-bearing |

**Behaviors:**
- NOT a particle - grid-based static obstacle
- Defines the "floor" of the mine
- Must be actively mined/broken to remove
- Fragments into rocks + gravel when broken
- Water cannot penetrate
- Often contains gold (why you dig down to it!)

**Implementation:**
- Static grid cells, not particles
- Health/damage system for mining
- Spawns particles when broken

---

### 8. Permafrost

| Property | Value | Notes |
|----------|-------|-------|
| density | 1600 kg/m³ | Frozen soil |
| cohesion | 100000 Pa | Ice bonding |
| is_static | true | Until melted |
| hardness | 7-8 | Hard when frozen |

**Behaviors:**
- Frozen mixture of soil, gravel, rocks, ice
- Static/immovable until melted
- Blocks water flow
- When melted → releases mud + gravel + rocks
- Contains ice that becomes water when melted

**Melting (Later Phase):**
- Temperature system OR
- Proximity to heat source OR
- Water contact over time

**Contents (defined per-block):**
```rust
struct PermafrostBlock {
    frozen_contents: Vec<(MaterialType, f32)>, // what's inside
    ice_fraction: f32,  // becomes water
    melt_progress: f32, // 0-1
}
```

---

### 9. Gold

| Property | Value | Notes |
|----------|-------|-------|
| density | 19300 kg/m³ | Very heavy! |
| cohesion | 0 | Individual particles |

**Behaviors:**
- Two states: **Bound** and **Free**
- Bound: attached to host particle (gravel, rock, soil)
- Free: independent particle, settles quickly due to high density

**Binding System:**
```rust
struct GoldBinding {
    gold_particle_id: ParticleId,
    host_particle_id: ParticleId,
    release_threshold: f32, // agitation/velocity to release
}
```

**Release Triggers:**
- Host particle broken/crushed
- Sufficient water agitation (velocity > threshold)
- Sluice processing

**Separation Mechanics (Sluice):**
- Gold's high density causes rapid settling
- Riffles in sluice trap heavy particles
- Lighter materials wash over

---

## Structural Mechanics

### Angle of Repose Collapse

When material is removed, check stability of neighbors:

```rust
fn check_stability(particle: &Particle, neighbors: &[Particle]) -> bool {
    let material = particle.material_properties();
    let max_slope = material.angle_of_repose;

    for neighbor in neighbors {
        let slope = calculate_slope(particle.position, neighbor.position);
        if slope > max_slope {
            return false; // unstable, will slide
        }
    }

    // Also check: is there support below?
    let has_support = neighbors.iter().any(|n|
        n.position.y < particle.position.y - SUPPORT_THRESHOLD
    );

    has_support
}
```

### Cascade Collapse

When particles become unstable:
1. Mark as "sliding"
2. Apply gravity + friction forces
3. Check if slide disturbs more particles
4. Repeat until stable

### Mixed Substrate Behavior

When different materials are mixed:
- Use weighted average of properties
- Or use the weakest link (minimum angle of repose)
- Gravel in soil = soil angle of repose (gravel locked in matrix)
- Gravel on soil = gravel can roll on soil surface

---

## Paydirt Concept

"Paydirt" is not a distinct material type but a **mixture** with gold content:

```rust
struct PaydirtComposition {
    soil_fraction: f32,    // 0.3-0.5
    gravel_fraction: f32,  // 0.3-0.5
    sand_fraction: f32,    // 0.1-0.2
    clay_fraction: f32,    // 0.0-0.1
    gold_concentration: f32, // g/m³ or particles/m³
}
```

**Implementation Options:**

1. **Explicit Mixture Particles**: Each particle knows its composition
2. **Statistical**: Region has composition, individual particles sampled from distribution
3. **Layered**: Different materials in layers, gold probability per layer

**Recommendation**: Option 2 (Statistical) for performance, with gold particles explicitly spawned.

---

## Mining Tool Interaction

### Phase 1: Shovel

```rust
struct ShovelTool {
    dig_radius: f32,        // area of effect
    dig_depth: f32,         // how deep per action
    force: f32,             // applied to particles
    particles_per_scoop: u32, // max particles removed
}
```

**Shovel Action:**
1. Player clicks/drags
2. Find particles in dig_radius at cursor
3. Apply upward + outward force
4. Remove particles that exceed removal velocity
5. Removed particles go to... inventory? Bucket? Pile nearby?
6. Check structural stability of remaining particles

**Material Hardness Effect:**
```rust
fn dig_effectiveness(tool: &Tool, material: &Material) -> f32 {
    if material.is_static {
        return 0.0; // can't dig bedrock with shovel
    }
    tool.force / (material.hardness + 1.0)
}
```

---

## Water-Material Interactions

### Erosion

```rust
fn calculate_erosion(
    water_velocity: Vec2,
    material: &MaterialProperties,
) -> f32 {
    let erosion_resistance = material.cohesion +
                             material.density * DENSITY_FACTOR;
    let water_force = water_velocity.length_squared() * WATER_DENSITY;

    (water_force / erosion_resistance).max(0.0)
}
```

**Erosion Rates (relative):**
| Material | Erosion Resistance |
|----------|-------------------|
| Sand | 1.0 (baseline) |
| Soil | 3.0 |
| Mud | 5.0 |
| Gravel | 8.0 |
| Rock | 100.0 |

### Transport

Particles are picked up when:
```
water_velocity > sqrt(particle_mass × gravity / drag_coefficient)
```

Particles settle when velocity drops below threshold.

### Saturation

```rust
fn update_saturation(particle: &mut Particle, water_neighbors: &[Particle]) {
    let water_contact = water_neighbors.len() as f32 / MAX_NEIGHBORS as f32;

    particle.saturation += water_contact * SATURATION_RATE * dt;
    particle.saturation = particle.saturation.clamp(0.0, 1.0);

    // Check for material transformation
    if particle.saturation > particle.material.saturation_threshold {
        if let Some(wet_variant) = particle.material.wet_variant {
            particle.transform_to(wet_variant);
        }
    }
}
```

---

## Implementation Phases

### Phase 1: Core Granular Materials ✦ MVP
**Goal**: Diggable terrain with structural collapse

**Materials**: Sand, Soil, Gravel (as particles, not rigid clusters yet)
**Mechanics**:
- Basic material properties (density, cohesion, friction_angle)
- Angle of repose stability checking
- Cascade collapse when support removed
- Shovel tool: dig and displace particles
- Single active simulation zone

**Deliverables**:
- [ ] Material enum and properties struct
- [ ] Particle spawning with material types
- [ ] Angle of repose collapse system
- [ ] Basic shovel interaction
- [ ] Visual distinction between materials

**Success Criteria**: Can dig a hole, see realistic collapse of walls

---

### Phase 2: Water Integration
**Goal**: Water that flows and interacts with terrain

**Materials**: Add Water
**Mechanics**:
- Water as low-viscosity fluid (existing MPM)
- Water finds lowest point, pools
- Basic erosion: water velocity moves loose particles
- Particle transport in flowing water
- Water drains through permeable materials

**Deliverables**:
- [ ] Water material with fluid properties
- [ ] Erosion calculation and application
- [ ] Particle pickup and transport
- [ ] Permeability-based water seepage

**Success Criteria**: Pour water, it flows and carries sand with it

---

### Phase 3: Wet Materials & Mud
**Goal**: Saturation transforms materials

**Materials**: Add Mud (as wet soil transformation)
**Mechanics**:
- Saturation tracking (per-particle or grid-based)
- Soil → Mud transformation when saturated
- Mud yield-stress behavior (slump, not flow)
- Mud viscosity and slow movement

**Deliverables**:
- [ ] Saturation system
- [ ] Material transformation logic
- [ ] Bingham/yield-stress fluid model for mud
- [ ] Visual feedback for wet vs dry

**Success Criteria**: Add water to soil, it becomes mud that slumps

---

### Phase 4: Rigid Bodies (Rocks & Gravel Clusters)
**Goal**: Multi-particle rigid objects

**Materials**: Rocks, Gravel as clusters
**Mechanics**:
- Rigid body constraints between particles
- Compound collision shapes
- Rolling, tumbling, stacking
- Part of mixed substrate

**Deliverables**:
- [ ] Rigid body constraint system
- [ ] Rock spawning as particle clusters
- [ ] Gravel as small clusters (2-5 particles)
- [ ] Collision and stacking behavior

**Success Criteria**: Rocks roll down slopes, stack realistically

---

### Phase 5: Static Structures & Mining
**Goal**: Breakable bedrock and permafrost

**Materials**: Add Bedrock, Permafrost (basic)
**Mechanics**:
- Grid-based static obstacles
- Health/damage system
- Fragmentation: static → particles
- Tool effectiveness based on hardness
- Upgraded tools (pickaxe) for hard materials

**Deliverables**:
- [ ] Static material grid system
- [ ] Damage and breaking mechanics
- [ ] Fragmentation spawning
- [ ] Pickaxe tool for hard materials

**Success Criteria**: Mine bedrock, it breaks into rocks and gravel

---

### Phase 6: Gold System
**Goal**: Gold binding, tracking, and basic release

**Materials**: Add Gold particles
**Mechanics**:
- Gold binding to host particles
- Gold concentration per region/material
- Release on host destruction
- Gold high-density settling
- Basic gold collection tracking

**Deliverables**:
- [ ] Gold particle type
- [ ] Binding/release system
- [ ] Gold spawning based on concentration
- [ ] Collection/inventory tracking

**Success Criteria**: Break gold-bearing rock, gold particles released and settle

---

### Phase 7: Sluice & Separation (Separate Simulation)
**Goal**: Gold extraction mechanic

**Mechanics**:
- Higher-resolution fluid simulation
- Riffle/trap mechanics
- Density-based separation
- Gold collection from sluice

**Deliverables**:
- [ ] Sluice mini-simulation
- [ ] Material input from main world
- [ ] Separation physics
- [ ] Gold output tracking

**Success Criteria**: Feed paydirt into sluice, gold separates and collects

---

### Phase 8: Large World & Zones
**Goal**: Scalable world with zone management

**Mechanics**:
- Chunk-based world storage
- Active/Hybrid/Dormant zone transitions
- Particle-to-voxel serialization
- Procedural terrain generation

**Deliverables**:
- [ ] Chunk system
- [ ] Zone transition logic
- [ ] Dormant storage format
- [ ] Basic procedural generation

**Success Criteria**: Large world with good performance, seamless zone transitions

---

### Phase 9: Advanced Interactions
**Goal**: Temperature, drying, complex erosion

**Materials**: Permafrost melting
**Mechanics**:
- Temperature system (or simplified triggers)
- Permafrost melting → mud + contents
- Mud drying → soil
- Complex erosion (undercutting, etc.)

**Deliverables**:
- [ ] Temperature/heat system
- [ ] Permafrost melt mechanics
- [ ] Drying mechanics
- [ ] Advanced erosion patterns

---

### Phase 10: Equipment & Vehicles
**Goal**: Advanced mining tools

**Mechanics**:
- Bucket/scoop vehicles
- Water jets/hoses
- Explosives
- Conveyor systems?

---

## Module Structure (Parallel Development)

To enable parallel development of Mining and Sluice features:

```
src/
├── materials/              # SHARED - Material definitions
│   ├── mod.rs
│   ├── types.rs            # MaterialType enum
│   ├── properties.rs       # MaterialProperties struct
│   └── catalog.rs          # Default properties for each material
│
├── inventory/              # SHARED - Backpack/storage system
│   ├── mod.rs
│   ├── backpack.rs         # Player inventory
│   └── transfer.rs         # Material transfer between systems
│
├── mining/                 # MINING MODULE - Can develop independently
│   ├── mod.rs
│   ├── simulation.rs       # APIC-MPM for mining world
│   ├── terrain.rs          # Terrain generation, chunks
│   ├── tools/
│   │   ├── mod.rs
│   │   ├── shovel.rs
│   │   └── pickaxe.rs
│   ├── stability.rs        # Angle of repose, collapse
│   └── zones.rs            # Active/Hybrid/Dormant management
│
├── sluice/                 # SLUICE MODULE - Can develop independently
│   ├── mod.rs
│   ├── simulation.rs       # High-res fluid sim for sluice
│   ├── separation.rs       # Density-based gold separation
│   ├── riffles.rs          # Riffle trap mechanics
│   └── output.rs           # Gold collection
│
└── game/                   # INTEGRATION
    ├── mod.rs
    ├── player.rs           # Player state, position, current tool
    ├── world.rs            # Connects mining + sluice + inventory
    └── ui.rs               # HUD, inventory display
```

### Interface Between Mining ↔ Sluice

```rust
// In inventory/transfer.rs

/// Material packet for transfer between simulations
pub struct MaterialPacket {
    pub contents: HashMap<MaterialType, f32>,  // material → volume/mass
    pub gold_particles: u32,                    // bound gold count
    pub source_region: Option<WorldCoord>,      // where it came from
}

impl Backpack {
    /// Add material from mining (shovel scoop)
    pub fn add_from_mining(&mut self, packet: MaterialPacket);

    /// Dump material into sluice
    pub fn dump_to_sluice(&mut self, amount: f32) -> MaterialPacket;

    /// Dump material back into mining world
    pub fn dump_to_world(&mut self, amount: f32) -> MaterialPacket;
}
```

### Development Order

**Team A (Mining):**
1. `materials/` - shared types
2. `mining/simulation.rs` - basic APIC-MPM setup
3. `mining/tools/shovel.rs` - dig interaction
4. `mining/stability.rs` - collapse mechanics
5. `inventory/backpack.rs` - simple storage

**Team B (Sluice):** (can start after `materials/` is done)
1. `sluice/simulation.rs` - high-res fluid
2. `sluice/separation.rs` - density physics
3. `sluice/riffles.rs` - trap mechanics

**Integration Point:**
- Both connect via `inventory/transfer.rs`
- Player can shovel material → backpack → dump in sluice

---

## Resolved Decisions

1. **Particle Count Budget**: Benchmarked on M-series Mac (2024-12-21)

   | Grid Size | Pixel Size | Particles @ 60 FPS | Notes |
   |-----------|------------|-------------------|-------|
   | 128×128 | 256×256 px | ~6,000 | Sluice-sized |
   | 256×192 | 512×384 px | ~15,000 | **Recommended active zone** |
   | 512×384 | 1024×768 px | ~5,000 | Too large for dense sim |
   | 256×256 (cell=4) | 1024×1024 | ~20,000 | Coarse grid option |

   **Recommendation**: Use 256×192 grid with cell_size=2.0 for mining active zone.
   This gives ~15,000 particle budget at 60 FPS.

2. **Visual Style**: Pixel art for now
   - Simple colored particles/sprites per material
   - Can upgrade later

3. **Inventory System**: Virtual "backpack"
   - Dug materials go into backpack (removed from simulation)
   - Can be dumped out later (re-spawned as particles)
   - Simple quantity tracking per material type

4. **Parallel Development**: Mining and Sluice as separate modules
   - Shared material definitions
   - Independent simulations
   - Clean interface for material transfer

## Open Questions

1. **World Persistence**: Save/load requirements?
   - Affects dormant zone storage format
   - Particle vs voxel serialization

2. **Multiplayer Considerations**: Any future plans?
   - Affects zone ownership and synchronization

---

## Technical Risks

1. **Performance at Scale**: Large particle counts + rigid bodies + fluids
   - Mitigation: Zone system, LOD, GPU acceleration

2. **Rigid Body Stability**: MPM + rigid constraints can be tricky
   - Mitigation: Start simple, iterate

3. **Material Transitions**: Smooth granular ↔ fluid transitions
   - Mitigation: Yield-stress model handles this naturally

4. **Zone Boundaries**: Particles crossing active/dormant boundaries
   - Mitigation: Hybrid zone as buffer, careful serialization

---

## References

- Drucker-Prager yield criterion for granular materials
- Bingham plastic model for mud/yield-stress fluids
- APIC transfer for momentum conservation
- Position-based rigid body constraints

---

## Development Setup

### Worktree for Parallel Development
```bash
# Mining development (separate from sluice work)
.worktrees/feat-mining-materials/  # branch: feat/mining-materials

# Switch to mining worktree:
cd .worktrees/feat-mining-materials
```

### Quick Start (Phase 1)
1. Copy shared material types from existing `particle.rs`
2. Create `materials/` module with expanded properties
3. Set up basic mining scene (granular terrain + shovel)
4. Add angle of repose collapse

---

## Changelog

- v0.1 (2024-12-21): Initial design document
- v0.1.1 (2024-12-21): Added benchmark results, worktree setup

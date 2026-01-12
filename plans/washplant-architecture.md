# Washplant Architecture Plan

**Date:** 2026-01-12
**Status:** PLANNING
**Depends on:** Equipment geometry system, FLIP 3D, DEM clump physics

---

## Goal

Build a complete placer gold washplant simulation using the existing unified equipment framework. Material flows through multiple processing stages, with size classification and gravity separation.

---

## Real-World Washplant Flow

```
[Loader/Excavator]
       |
       v
+-------------+
|   HOPPER    |  <-- Surge bin, buffers raw feed
+-------------+
       |
       v
+-------------+
|   GRIZZLY   |  <-- Coarse screening, removes boulders (>4")
+-------------+
       |
       v
+-------------+
|   TROMMEL   |  <-- Rotating scrubber, breaks up clay, screens fines
+-------------+
       |
       v
+-------------+
| SHAKER DECK |  <-- Vibrating screen, size classification
+-------------+
       |
       v
+-------------+
|   SLUICE    |  <-- Gravity separation, captures gold
+-------------+
       |
       v
+-------------+
|  TAILINGS   |  <-- Waste discharge
+-------------+
```

---

## Current Assets Available

### Equipment Geometry (all implemented)
| Component | Config Type | Key Features |
|-----------|-------------|--------------|
| Hopper | `HopperConfig` | Tapered walls, open top |
| Grate/Screen | `GrateConfig` | Parallel bars with gaps |
| Gutter | `GutterConfig` | U-shaped channel |
| Chute | `ChuteConfig` | Angled slide with walls |
| Box | `BoxConfig` | Hollow container |
| Baffle | `BaffleConfig` | Deflector plate |
| Frame | `FrameConfig` | Structural support |
| Sluice | `SluiceConfig` | Riffled trough with slope |

### Particle Physics
| System | Status | Notes |
|--------|--------|-------|
| GPU FLIP 3D | WORKING | Water, pressure, advection |
| DEM Clumps | WORKING | Gravel/nuggets via ClusterSimulation3D |
| Screen-space fluid | WORKING | Visual water rendering |
| FLIP-DEM coupling | PARTIAL | One-way drag only |

### Existing Examples
- `shaker_deck_flip.rs` - Vibrating deck with perforations
- `friction_sluice.rs` - Sluice with riffles and gravel
- `industrial_sluice.rs` - Larger scale setup

---

## Architecture: Multi-Grid Simulation

### Approach: Separate Grids Per Equipment Stage

Each equipment piece has its **own simulation grid**. This gives us:
- **Memory efficiency** - No wasted cells between equipment
- **Variable resolution** - Fine grid for sluice (0.01m), coarse for hopper (0.05m)
- **Parallel potential** - Zones can tick independently
- **Isolated testing** - Debug each stage separately

```
+----------+      +----------+      +-----------+      +------------+
|  HOPPER  |----->| GRIZZLY  |----->|  SHAKER   |----->|   SLUICE   |
|  Grid A  |      |  Grid B  |      |  Grid C   |      |   Grid D   |
| (coarse) |      | (coarse) |      | (medium)  |      |  (fine)    |
+----------+      +----------+      +-----------+      +------------+
     |                 |                  |                  |
     v                 v                  v                  v
 [Transfer]        [Transfer]        [Transfer]         [Tailings]
   Zone              Zone              Zone               Exit
```

### Multi-Grid Benefits

| Aspect | Single Grid | Multi-Grid |
|--------|-------------|------------|
| Memory | 1 huge grid, mostly empty | Small grids, fully utilized |
| Resolution | Uniform everywhere | Variable per stage |
| Testing | All-or-nothing | Isolate each stage |
| Particles | 300k shared | 300k+ per stage possible |
| Complexity | Simple | Transfer logic needed |

### Grid Transfer Zones

Particles exit one grid and enter the next via **transfer volumes**:

```rust
pub struct TransferZone {
    /// Source grid identifier
    pub from_grid: GridId,
    /// Destination grid identifier
    pub to_grid: GridId,

    /// AABB in source grid where particles are captured
    pub capture_aabb: AABB,
    /// Position offset to apply when entering destination
    pub position_offset: Vec3,

    /// Velocity transformation (rotation for angled transfers)
    pub velocity_transform: Mat3,

    /// Optional: delay particles (simulates travel time through chute)
    pub transit_time: f32,
}
```

### Transfer Mechanics

1. **Capture**: Particles in `capture_aabb` with velocity pointing "out" are flagged
2. **Remove**: Flagged particles removed from source grid's simulation
3. **Queue**: Particle data stored in transfer buffer with transit timer
4. **Inject**: After `transit_time`, particle spawned in destination grid

```rust
impl TransferZone {
    pub fn process(&mut self,
        source: &mut FlipSimulation3D,
        dest: &mut FlipSimulation3D,
        dt: f32
    ) {
        // 1. Find particles exiting source
        let mut to_transfer = Vec::new();
        for (idx, particle) in source.particles.iter().enumerate() {
            if self.capture_aabb.contains(particle.position) {
                // Check velocity points toward exit
                let exit_dir = self.get_exit_direction();
                if particle.velocity.dot(exit_dir) > 0.0 {
                    to_transfer.push(idx);
                }
            }
        }

        // 2. Remove from source, queue for destination
        for idx in to_transfer.into_iter().rev() {
            let p = source.remove_particle(idx);
            self.transit_queue.push(TransitParticle {
                position: p.position + self.position_offset,
                velocity: self.velocity_transform * p.velocity,
                remaining_time: self.transit_time,
                density: p.density,
                // ... other particle data
            });
        }

        // 3. Inject particles that have completed transit
        self.transit_queue.retain(|tp| {
            tp.remaining_time -= dt;
            if tp.remaining_time <= 0.0 {
                dest.spawn_particle_with_velocity(tp.position, tp.velocity);
                false // Remove from queue
            } else {
                true // Keep in queue
            }
        });
    }
}

---

## Phase 1: Single-Grid Washplant Demo

### Step 1.1: Define WashplantConfig

```rust
pub struct WashplantConfig {
    // Grid dimensions
    pub grid_width: usize,    // X: flow direction
    pub grid_height: usize,   // Y: vertical
    pub grid_depth: usize,    // Z: cross-flow
    pub cell_size: f32,

    // Equipment placement
    pub hopper: HopperPlacement,
    pub grizzly: GrizzlyPlacement,
    pub shaker: ShakerPlacement,
    pub sluice: SluicePlacement,

    // Water system
    pub water_inlet_rate: f32,  // m3/s
    pub recirculation: bool,
}

pub struct HopperPlacement {
    pub origin: [f32; 3],       // World position
    pub config: HopperConfig,
    pub feed_rate: f32,         // kg/s raw material
}

pub struct GrizzlyPlacement {
    pub origin: [f32; 3],
    pub config: GrateConfig,
    pub vibration_freq: f32,    // Hz
    pub vibration_amp: f32,     // m
}

pub struct ShakerPlacement {
    pub origin: [f32; 3],
    pub config: GrateConfig,    // Perforated deck
    pub angle_deg: f32,         // Deck slope
    pub stroke_freq: f32,       // Hz
    pub stroke_amp: f32,        // m
    pub hole_sizes: Vec<f32>,   // Multiple decks with different apertures
}

pub struct SluicePlacement {
    pub origin: [f32; 3],
    pub config: SluiceConfig,   // Already has riffles
}
```

### Step 1.2: Equipment Builder Composition

```rust
pub struct WashplantBuilder {
    config: WashplantConfig,

    // Individual equipment builders
    hopper_builder: HopperGeometryBuilder,
    grizzly_builder: GrateGeometryBuilder,
    shaker_builders: Vec<GrateGeometryBuilder>,  // Multiple decks
    sluice_builder: SluiceGeometryBuilder,

    // Connecting geometry
    chute_builders: Vec<ChuteGeometryBuilder>,

    // Combined mesh for rendering
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
}

impl WashplantBuilder {
    pub fn build_all(&mut self) -> impl Iterator<Item = (usize, usize, usize)> {
        // Returns iterator of all solid cells across all equipment
        let hopper_cells = self.hopper_builder.solid_cells();
        let grizzly_cells = self.grizzly_builder.solid_cells();
        // ... combine all
    }

    pub fn build_combined_mesh(&mut self, grid: &Grid3D) {
        // Build single mesh from all equipment for efficient rendering
    }
}
```

### Step 1.3: Material Tracking System

```rust
pub struct MaterialZone {
    pub name: &'static str,
    pub bounds: AABB,
    pub equipment_type: EquipmentType,

    // Metrics
    pub particle_count: usize,
    pub total_mass: f32,
    pub avg_velocity: Vec3,

    // Recovery tracking (for sluice)
    pub gold_captured: f32,
    pub gangue_captured: f32,
}

pub struct WashplantStats {
    pub zones: Vec<MaterialZone>,
    pub throughput_tph: f32,     // Tons per hour
    pub gold_recovery_pct: f32,
    pub water_usage_m3h: f32,
}
```

---

## Phase 2: Dynamic Equipment (Vibration)

### Step 2.1: Vibrating Equipment Trait

```rust
pub trait Vibrating {
    fn update_position(&mut self, time: f32);
    fn get_current_offset(&self) -> Vec3;
    fn get_current_velocity(&self) -> Vec3;
}

pub struct VibratingEquipment<T> {
    pub base_position: Vec3,
    pub equipment: T,

    // Vibration parameters
    pub frequency: f32,      // Hz
    pub amplitude: Vec3,     // Directional amplitude
    pub phase: f32,

    // Precomputed
    omega: f32,              // 2*PI*frequency
}

impl<T> Vibrating for VibratingEquipment<T> {
    fn update_position(&mut self, time: f32) {
        let offset = self.amplitude * (self.omega * time + self.phase).sin();
        // Update solid cells in grid (expensive - see optimization below)
    }

    fn get_current_velocity(&self) -> Vec3 {
        // Derivative of position for particle coupling
        self.amplitude * self.omega * (self.omega * time + self.phase).cos()
    }
}
```

### Step 2.2: GPU-Accelerated Dynamic SDF

For vibrating equipment, regenerating solid cells every frame is expensive. Instead:

1. **Precompute SDF texture** for equipment in rest position
2. **Apply transform in shader** to SDF sample coordinates
3. **Only update when amplitude changes**, not every frame

```wgsl
// In collision shader
fn sample_vibrating_equipment_sdf(pos: vec3<f32>, time: f32) -> f32 {
    // Transform pos by inverse of current vibration offset
    let offset = equipment_amplitude * sin(equipment_omega * time);
    let local_pos = pos - equipment_origin - offset;

    // Sample precomputed SDF
    return textureSample(equipment_sdf, equipment_sampler, local_pos / sdf_scale).r;
}
```

### Step 2.3: Particle-Equipment Velocity Transfer

When particles contact vibrating equipment, they should receive impulse:

```rust
// In G2P or dedicated contact resolution
fn apply_vibrating_contact(
    particle: &mut Particle,
    equipment: &VibratingEquipment,
    contact_normal: Vec3,
    penetration: f32,
) {
    // Separate particle from equipment
    particle.position += contact_normal * penetration;

    // Transfer equipment velocity (makes particles "ride" the shaker)
    let equipment_vel = equipment.get_current_velocity();
    let normal_vel = contact_normal.dot(equipment_vel);

    // Only add equipment velocity component in contact direction
    particle.velocity += contact_normal * normal_vel * friction_coeff;
}
```

---

## Phase 3: Size Classification (Screening)

### Step 3.1: Screen Aperture System

Screens pass particles smaller than aperture, retain larger ones.

```rust
pub struct ScreenConfig {
    pub grate: GrateConfig,
    pub aperture_size: f32,     // Hole diameter (meters)
    pub open_area_pct: f32,     // % of deck that's holes
}

// In particle physics loop
fn check_screen_passage(
    particle: &Particle,
    screen: &ScreenConfig,
    particle_radius: f32,
) -> bool {
    if particle_radius > screen.aperture_size * 0.5 {
        return false;  // Too big, retained on deck
    }

    // Check if particle is over a hole (existing in_hole check)
    let local_pos = particle.position - screen.origin;
    screen.in_hole(local_pos)
}
```

### Step 3.2: Multi-Deck Classification

Real shakers have multiple decks stacked:
- Top deck: 2" holes (remove large rocks)
- Middle deck: 1/2" holes (remove gravel)
- Bottom deck: 1/8" holes (remove sand)

```rust
pub struct MultiDeckShaker {
    pub decks: Vec<ShakerDeck>,
    pub spacing: f32,  // Vertical distance between decks
}

pub struct ShakerDeck {
    pub config: ScreenConfig,
    pub height_offset: f32,
}
```

---

## Phase 4: Material Properties

### Step 4.1: Extended Particle Types

```rust
pub struct MaterialProperties {
    pub name: &'static str,
    pub density: f32,           // kg/m3
    pub radius_range: (f32, f32),  // Min-max size (meters)
    pub color: [f32; 4],

    // Classification
    pub is_valuable: bool,      // Track for recovery metrics
    pub settling_velocity: f32, // Stokes settling
}

pub const MATERIALS: &[MaterialProperties] = &[
    // Oversize (removed by grizzly)
    MaterialProperties { name: "boulder", density: 2700.0, radius_range: (0.10, 0.30), .. },
    MaterialProperties { name: "cobble", density: 2700.0, radius_range: (0.05, 0.10), .. },

    // Gravel (DEM clumps)
    MaterialProperties { name: "gravel", density: 2700.0, radius_range: (0.01, 0.05), .. },

    // Sand/Silt (FLIP particles)
    MaterialProperties { name: "sand", density: 2650.0, radius_range: (0.001, 0.01), .. },
    MaterialProperties { name: "silt", density: 2650.0, radius_range: (0.0001, 0.001), .. },

    // Heavies
    MaterialProperties { name: "magnetite", density: 5200.0, radius_range: (0.0005, 0.002), .. },
    MaterialProperties { name: "gold", density: 19300.0, radius_range: (0.0001, 0.005), .. },
];
```

### Step 4.2: Realistic Feed Composition

```rust
pub struct FeedComposition {
    pub material_fractions: HashMap<&'static str, f32>,  // Mass fractions
}

impl Default for FeedComposition {
    fn default() -> Self {
        // Typical placer gravel from Klondike
        Self {
            material_fractions: [
                ("boulder", 0.05),
                ("cobble", 0.10),
                ("gravel", 0.35),
                ("sand", 0.40),
                ("silt", 0.09),
                ("magnetite", 0.008),
                ("gold", 0.002),  // 2g/m3 - good ground!
            ].into()
        }
    }
}
```

---

## Phase 5: Water Management

### Step 5.1: Water Circuit

Real washplants recirculate water:

```
+-------------+
| Header Tank |<---+
+------+------+    |
       |           | (pump)
       v           |
  [Spray Bars]     |
       |           |
       v           |
  [EQUIPMENT]      |
       |           |
       v           |
+-------------+    |
| Settling    |----+
| Pond        |
+-------------+
       |
       v
   (Tailings)
```

### Step 5.2: Water Balance Tracking

```rust
pub struct WaterCircuit {
    pub header_tank_volume: f32,
    pub flow_rate: f32,  // m3/s into plant

    // Spray bar allocation
    pub spray_bars: Vec<SprayBar>,

    // Return flow
    pub return_rate: f32,  // m3/s back to header
    pub losses: f32,       // Evap, tailings moisture
}

pub struct SprayBar {
    pub position: Vec3,
    pub direction: Vec3,
    pub flow_rate: f32,
}
```

---

## Implementation Roadmap

### Milestone 1: Multi-Grid Washplant Layout (2-3 sessions)

**Goal:** Full plant visible with all stages, using multi-grid architecture.

#### 1.1 Core Infrastructure
- [ ] `WashplantStage` struct - wraps one FLIP simulation + equipment geometry
- [ ] `TransferZone` struct - particle handoff between stages
- [ ] `Washplant` orchestrator - manages all stages + transfers

#### 1.2 Stage Definitions
- [ ] **Hopper Stage** - coarse grid (0.05m), simple tapered box
- [ ] **Grizzly Stage** - coarse grid (0.05m), grate with bars
- [ ] **Shaker Stage** - medium grid (0.02m), perforated deck (static initially)
- [ ] **Sluice Stage** - fine grid (0.01m), riffles (existing SluiceConfig)

#### 1.3 Visual Layout
- [ ] Camera that can view entire plant or zoom to single stage
- [ ] Stage selection (keyboard 1-4 to focus)
- [ ] Combined mesh rendering for all equipment
- [ ] Stage status HUD (particle counts per zone)

#### 1.4 Particle Flow
- [ ] Emitter at hopper top
- [ ] Transfer zones between each stage
- [ ] Exit drain at sluice end
- [ ] Basic throughput counter

### Milestone 2: Individual Stage Testing (1-2 sessions)
- [ ] Run each stage in isolation with controlled input
- [ ] Tune grid resolutions and particle budgets per stage
- [ ] Validate transfer zone positions and velocities

### Milestone 3: Vibrating Shaker (2-3 sessions)
- [ ] Port shaker_deck_flip vibration logic to `VibratingEquipment` trait
- [ ] GPU dynamic SDF for vibrating deck
- [ ] Particle-deck velocity coupling
- [ ] Test: gravel walks down deck, fines fall through

### Milestone 4: Size Classification (1-2 sessions)
- [ ] Screen aperture checking for DEM clumps
- [ ] Oversize rejection path (grizzly rejects boulders)
- [ ] Undersize pass-through (shaker drops fines)
- [ ] Multiple size fractions tracked

### Milestone 5: Metrics & Water (1-2 sessions)
- [ ] Mass balance per stage
- [ ] Gold recovery tracking
- [ ] Throughput (TPH) display
- [ ] Water recirculation (header tank → spray bars → return)

---

## File Structure

```
crates/game/src/
  washplant/
    mod.rs            # Module exports
    stage.rs          # WashplantStage - wraps simulation + geometry
    transfer.rs       # TransferZone - particle handoff
    plant.rs          # Washplant - orchestrates all stages
    config.rs         # StageConfig, PlantConfig
    metrics.rs        # Throughput, recovery tracking

crates/game/examples/
  washplant_full.rs   # Full 4-stage plant demo
  washplant_hopper.rs # Isolated hopper testing
  washplant_sluice.rs # Isolated sluice testing
```

## Core Types (Milestone 1)

```rust
// stage.rs
pub struct WashplantStage {
    pub name: &'static str,
    pub config: StageConfig,

    // Simulation
    pub sim: FlipSimulation3D,
    pub gpu_flip: GpuFlip3D,
    pub dem: Option<ClusterSimulation3D>,

    // Geometry
    pub equipment_builder: Box<dyn EquipmentBuilder>,
    pub world_offset: Vec3,  // Position in plant layout

    // GPU resources
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,

    // Stats
    pub particle_count: usize,
    pub water_count: usize,
    pub sediment_count: usize,
}

// transfer.rs
pub struct TransferZone {
    pub from_stage: usize,
    pub to_stage: usize,
    pub capture_aabb: AABB,      // In source stage's local coords
    pub inject_position: Vec3,   // In dest stage's local coords
    pub inject_velocity: Vec3,   // Initial velocity in dest
    pub transit_time: f32,
    transit_queue: Vec<TransitParticle>,
}

// plant.rs
pub struct Washplant {
    pub stages: Vec<WashplantStage>,
    pub transfers: Vec<TransferZone>,
    pub metrics: PlantMetrics,

    // Rendering
    pub camera_target: CameraTarget,
    pub focused_stage: Option<usize>,
}

impl Washplant {
    pub fn tick(&mut self, dt: f32) {
        // 1. Tick all stages (can be parallel)
        for stage in &mut self.stages {
            stage.tick(dt);
        }

        // 2. Process transfers
        for transfer in &mut self.transfers {
            transfer.process(&mut self.stages, dt);
        }

        // 3. Update metrics
        self.metrics.update(&self.stages);
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        // Render all stages with world offsets applied
    }
}

// config.rs
pub struct StageConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,
    pub max_particles: usize,
    pub equipment_type: EquipmentType,
}

pub enum EquipmentType {
    Hopper(HopperConfig),
    Grizzly(GrateConfig),
    Shaker(GrateConfig),
    Sluice(SluiceConfig),
}
```

---

## Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Grid sizing | **Multi-grid** | Memory efficiency, variable resolution, isolated testing |
| Trommel | **Skip** | Rotating cylinder too complex; grizzly + shaker sufficient |
| Particle budget | **300k+ per stage** | Multi-grid means each stage has own budget |
| Two-way coupling | **Defer** | Start one-way, add if settling is wrong |

## Remaining Open Questions

1. **GPU per stage or shared?**
   - Option A: Each stage has own GpuFlip3D instance
   - Option B: One GPU instance, swap grids per stage
   - Leaning toward A for isolation, but more VRAM

2. **DEM gravel transfer**
   - ClusterSimulation3D clumps need to transfer between stages too
   - Same TransferZone concept, but for DEM particles

3. **Water conservation**
   - Multi-grid means water particles "disappear" at transfer
   - Need injection rate at each stage to match removal rate
   - Or: track total water mass across all stages, balance at end of frame

---

## References

- `equipment_geometry.rs` - All equipment config types
- `sluice_geometry.rs` - Sluice with riffles
- `shaker_deck_flip.rs` - Vibration implementation
- `friction_sluice.rs` - Current best physics reference

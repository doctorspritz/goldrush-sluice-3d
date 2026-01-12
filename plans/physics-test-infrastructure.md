# Physics Test Infrastructure Plan

## Problem
- Headless tests pass but don't reflect real visual behavior
- No way to verify physics visually with meaningful diagnostics
- Test scenarios don't use realistic equipment geometry

## Goals
1. Test equipment that matches real physics scenarios
2. Saved layouts for each test (loadable in washplant-editor)
3. Real-time diagnostics overlay in visual mode
4. Automated test runner with pass/fail criteria

---

## 1. Test Equipment Geometry

Add new equipment pieces specifically for physics testing:

### PieceKind::TestFloor
- Flat horizontal surface at configurable Y
- Configurable size (width x depth)
- Use: drop tests, settling tests

### PieceKind::TestRamp
- Angled surface (configurable angle 0-90°)
- Configurable length and width
- Use: rolling friction, sliding tests, flow angle tests

### PieceKind::TestWall
- Vertical wall at configurable position
- Configurable height and width
- Use: collision/bounce tests, containment

### PieceKind::TestBox
- Fully enclosed container (4 walls + floor)
- Configurable dimensions (width x height x depth)
- No ceiling (open top for spawning)
- Use: static water tests, volume tests, settling tests

### PieceKind::TestPool
- Like TestBox but with configurable initial water level
- Can pre-fill with water at start
- Use: buoyancy tests, hydrostatic pressure tests

### PieceKind::TestChute
- Angled channel with side walls
- Configurable angle, length, width
- Use: flow tests, transport tests

### PieceKind::TestFunnel
- Converging walls (wide top, narrow bottom)
- Use: concentration tests, clogging tests

### PieceKind::TestObstacle
- Simple solid block in flow path
- Configurable size and position
- Use: flow-around tests, wake tests

### PieceKind::TestRiffle
- Single riffle bar (simplified from full sluice)
- Configurable height, angle, spacing
- Use: isolated capture behavior tests

### Equipment Parameters (common)
```rust
struct TestEquipment {
    kind: TestEquipmentKind,
    position: Vec3,
    rotation: f32,          // Y-axis rotation
    dimensions: Vec3,       // width, height, depth
    // Type-specific
    angle: Option<f32>,     // For ramps, chutes
    water_level: Option<f32>, // For pools
}
```

---

## 2. Saved Test Layouts

Each test gets a `.json` layout file in `assets/test_layouts/`:

```
assets/test_layouts/
├── # DEM Tests (dry, no water)
├── dem_floor_drop.json           # TestFloor, drop clumps from height
├── dem_wall_bounce.json          # TestFloor + TestWall, lateral collision
├── dem_ramp_roll.json            # TestRamp, rolling/sliding behavior
├── dem_box_pile.json             # TestBox, clumps pile up
├── dem_settling.json             # TestFloor, many clumps settle
│
├── # Water Tests (SWE, no sediment)
├── swe_box_static.json           # TestBox pre-filled, water at rest
├── swe_box_fill.json             # TestBox empty, fill with emitter
├── swe_chute_flow.json           # TestChute angled, steady flow
├── swe_obstacle_flow.json        # TestChute + TestObstacle, flow around
├── swe_pool_drain.json           # TestPool with outlet, draining
│
├── # Combined Tests (water + sediment)
├── combined_buoyancy.json        # TestPool with water, drop gold+sand
├── combined_settling.json        # TestBox with still water, sediment settles
├── combined_transport.json       # TestChute with flow, sediment moves
├── combined_riffle_single.json   # TestChute + TestRiffle, capture test
│
├── # Integration Tests (full equipment)
├── integration_gutter.json       # Real gutter piece
├── integration_sluice.json       # Real sluice piece
├── integration_full.json         # Gutter → Sluice → capture
```

Layout files include:
- Equipment pieces with positions
- Emitter configs (rate, position, type: water/gold/sand)
- Camera position for best view
- Test-specific parameters

---

## 3. Diagnostics System

### DiagnosticsOverlay struct
```rust
struct DiagnosticsOverlay {
    enabled: bool,
    show_velocities: bool,      // Arrow vectors on particles
    show_forces: bool,          // Force vectors
    show_penetration: bool,     // Red highlight when inside solid
    show_grid: bool,            // SDF grid visualization

    // Measurements
    measurements: TestMeasurements,
}

struct TestMeasurements {
    // DEM
    dem_count: usize,
    dem_min_y: f32,
    dem_max_y: f32,
    dem_avg_velocity: f32,
    dem_max_velocity: f32,
    dem_penetration_count: usize,  // Particles inside solids
    dem_kinetic_energy: f32,

    // Fluid
    fluid_count: usize,
    fluid_volume: f32,
    fluid_flow_rate: f32,         // Particles/sec through checkpoint
    fluid_avg_velocity: f32,

    // Volume measurements (per region)
    region_volumes: HashMap<String, RegionVolume>,

    // Combined
    gold_captured: usize,         // Gold in capture zone
    sand_captured: usize,
    separation_ratio: f32,        // gold_captured / sand_captured

    frame: u32,
    elapsed_time: f32,
}

struct RegionVolume {
    name: String,
    bounds: (Vec3, Vec3),         // AABB min/max
    water_volume: f32,            // m³ of water in region
    sediment_volume: f32,         // m³ of sediment in region
    particle_count: usize,
}

// Individual particle tracking
struct ParticleTracker {
    tracked_particles: Vec<TrackedParticle>,
    checkpoints: Vec<Checkpoint>,
}

struct TrackedParticle {
    id: usize,
    particle_type: ParticleType,  // Water, Gold, Sand
    spawn_time: f32,
    spawn_position: Vec3,

    // Journey tracking
    path: Vec<(f32, Vec3)>,       // (time, position) samples
    checkpoint_times: Vec<(usize, f32)>,  // (checkpoint_id, time)

    // Current state
    current_position: Vec3,
    current_velocity: Vec3,

    // Final state
    exit_time: Option<f32>,
    exit_position: Option<Vec3>,
    captured: bool,               // Did it get captured in riffle?
}

struct Checkpoint {
    id: usize,
    name: String,
    plane: (Vec3, Vec3),          // Point + normal defining plane
    // Counts particles crossing in each direction
    forward_count: usize,
    backward_count: usize,
    flow_rate: f32,               // Particles/sec (rolling average)
}

enum ParticleType {
    Water,
    Gold,
    Sand,
}
```

### Diagnostic Display (on-screen)
```
╔═══════════════════════════════════════════════════════╗
║ TEST: Integration Riffle Capture                      ║
╠═══════════════════════════════════════════════════════╣
║ Frame: 1200   Time: 20.0s                             ║
╠───────────────────────────────────────────────────────╣
║ PARTICLES                                             ║
║   Water: 850    Gold: 45    Sand: 120                 ║
║   Spawned: 1200  Exited: 185  Captured: 32            ║
╠───────────────────────────────────────────────────────╣
║ VOLUME (by region)                                    ║
║   Gutter:  0.0012 m³ water, 0.0001 m³ sediment        ║
║   Sluice:  0.0008 m³ water, 0.0002 m³ sediment        ║
║   Riffle:  0.0002 m³ water, 0.0004 m³ sediment        ║
╠───────────────────────────────────────────────────────╣
║ FLOW (checkpoints)                                    ║
║   Inlet:     12.5 particles/s                         ║
║   Mid-sluice: 11.2 particles/s                        ║
║   Outlet:     8.3 particles/s                         ║
╠───────────────────────────────────────────────────────╣
║ TRACKED PARTICLE #42 (Gold)                           ║
║   Journey: Spawn → Gutter → Sluice → Riffle (CAUGHT) ║
║   Time: 4.2s spawn-to-capture                         ║
║   Path length: 1.8m                                   ║
╠───────────────────────────────────════════════════════╣
║ STATUS: ✓ SEPARATING (gold_capture > sand_capture)   ║
╚═══════════════════════════════════════════════════════╝
```

### Particle Journey Visualization
- Click on particle to track it
- Path drawn as colored line (fades over time)
- Checkpoint crossings marked with dots
- Color indicates speed (blue=slow, red=fast)

### Volume Regions
- Defined per-piece (gutter, sluice, riffle zones)
- Visual: semi-transparent box overlays
- Toggle with V key

---

## 4. Test Runner

### Automated Mode
```bash
cargo run --example washplant_editor -- --run-tests
```

Runs all tests sequentially:
1. Load test layout
2. Run for N frames or until condition met
3. Check pass/fail criteria
4. Log results
5. Continue to next test

Output:
```
Physics Test Suite
==================
[PASS] DEM Floor Collision     - All particles settled (min_y > 0.07)
[PASS] DEM Wall Collision      - Bounce detected (v_x sign change)
[FAIL] DEM Density Separation  - Gold not below sand (gold_avg_y > sand_avg_y)
[PASS] DEM Settling Time       - Settled in 3.2s (< 5s limit)
...
==================
Results: 8/10 passed
```

### Visual Mode
```bash
cargo run --example washplant_editor -- --test-mode
```

- Press T to enter test mode
- Press 1-0 to select test
- Diagnostics overlay shown
- Press N for next test
- Press R to restart current test

---

## 5. Pass/Fail Criteria

Each test has specific criteria:

### DEM Floor Collision
- **PASS**: All particles min_y > floor_y after 5s
- **PASS**: Max velocity < 0.1 m/s (settled)
- **FAIL**: Any particle min_y < floor_y - radius (penetration)
- **FAIL**: Any particle y > spawn_y (explosion)

### DEM Wall Collision
- **PASS**: Particle bounces (v_x changes sign)
- **PASS**: Particle stays in bounds
- **FAIL**: Particle passes through wall

### DEM Density Separation
- **PASS**: avg(gold_y) < avg(sand_y) after settling
- **PASS**: Separation > 0.02m
- **FAIL**: Gold floats above sand

### SWE Flow Downhill
- **PASS**: Flow rate > 0 at outlet
- **PASS**: Water level decreasing at inlet
- **FAIL**: Water flows uphill

### Integration Riffle Capture
- **PASS**: Gold capture rate > sand capture rate
- **PASS**: Total capture < 100% (some flows through)
- **FAIL**: No separation effect

---

## 6. Implementation Order

### Phase 1: Core Diagnostics
1. [ ] Add TestMeasurements struct to washplant_editor
2. [ ] Add RegionVolume tracking per piece
3. [ ] Implement basic measurement collection (counts, positions, velocities)
4. [ ] Add on-screen diagnostics display (text overlay)

### Phase 2: Particle Tracking
5. [ ] Add ParticleTracker struct
6. [ ] Implement particle ID assignment at spawn
7. [ ] Track particle positions each frame (sampled, not every frame)
8. [ ] Add Checkpoint system for flow measurement
9. [ ] Implement journey visualization (path lines)

### Phase 3: Test Infrastructure
10. [ ] Create test layout JSON format
11. [ ] Create saved layouts for all 10 tests
12. [ ] Add test equipment pieces (TestFloor, TestWall, etc.)
13. [ ] Define pass/fail criteria per test

### Phase 4: Test Runner
14. [ ] Implement --run-tests CLI mode (headless)
15. [ ] Implement visual test mode (T key, diagnostics overlay)
16. [ ] Add N=next, R=restart, P=pause controls
17. [ ] Generate test report output

---

## 7. Files to Modify

- `crates/game/examples/washplant_editor.rs` - Main changes
- `assets/test_layouts/*.json` - New test layout files
- `crates/sim3d/src/clump.rs` - Add measurement helpers
- `crates/sim3d/src/lib.rs` - Add measurement helpers for fluid


# Goldrush Fluid Miner - Physics Engine Plan

## Game Vision

A 2D mining simulation where **physics IS the gameplay**. Players run a gold mining operation using real fluid mechanics and material physics to extract gold from terrain.

**Core Loop:** Dig → Transport → Process → Optimize → Scale

**Inspirations:**
- Noita (falling sand physics)
- Factorio (engineering/optimization)
- Gold Rush: The Game (mining mechanics)
- Terraria (2D mining progression)

---

## World Generation

### Terrain Layers (top to bottom)
```
═══════════════════════════════════════════════════
~~~~~~~~~~ Water (rivers/lakes) ~~~~~~~~~~
▒▒▒▒▒▒▒▒▒▒ Topsoil (loose, easy to dig) ▒▒▒▒▒▒▒▒▒▒
▓▓▓▓▓▓▓▓▓▓ Subsoil (denser, some rocks) ▓▓▓▓▓▓▓▓▓▓
████░░████ Pay dirt (gold-bearing gravel) ████░░████
██████████ Clay/hardpan layer ██████████████████████
▓▓▓▓▓▓▓▓▓▓ Fractured rock (loose boulders) ▓▓▓▓▓▓▓▓
██████████████████████████████████████████████████
         ~~~~ Wavy Bedrock (static) ~~~~
```

### Materials

| Material | Density | Cohesion | Angle of Repose | Behavior |
|----------|---------|----------|-----------------|----------|
| Air | 0 | 0 | - | Empty |
| Water | 1.0 | 0 | 0° | Flows, seeks level |
| Mud | 1.3 | 0.2 | 5° | Viscous flow, erodes easily |
| Topsoil | 1.8 | 0.3 | 30° | Loose, easy to dig |
| Sand | 2.0 | 0.1 | 25° | Flows easily when disturbed |
| Gravel | 2.2 | 0.4 | 35° | Contains gold, medium dig |
| Clay | 2.5 | 0.7 | 45° | Sticky, hard to wash |
| Rock (loose) | 2.8 | 0.5 | 40° | Falls, can crush |
| Bedrock | 3.0 | 1.0 | 90° | Static, unbreakable |
| Gold | 5.0 | 0.8 | 35° | Heavy, sinks fast |

---

## Core Physics Systems

### 1. Gravity & Falling
Everything falls. Heavier things fall through lighter things.

```rust
// Priority: down > diagonal-down > stay
if can_move_to(x, y+1) { fall_down(); }
else if can_move_to(x-1, y+1) || can_move_to(x+1, y+1) { slide_diagonal(); }
else { stay(); }

// Density displacement: heavy sinks through light
if below.density() < self.density() { swap(); }
```

### 2. Angle of Repose (CRITICAL for rockfall)
Materials won't stack steeper than their angle of repose. Exceeding it triggers collapse.

```
Stable (30° slope):        Unstable (60° slope) → Collapse!
    ▒                          ▒
   ▒▒                          ▒▒  ← This will fall
  ▒▒▒                         ▒▒▒
 ▒▒▒▒                        ▒▒▒▒
▒▒▒▒▒                       ▒▒▒▒▒
```

**Implementation:**
```rust
fn check_stability(world: &World, x: i32, y: i32) -> bool {
    let material = world.get(x, y);
    let max_angle = material.angle_of_repose();

    // Check if supported adequately
    let support_left = count_support(world, x, y, -1);
    let support_right = count_support(world, x, y, 1);

    // Collapse if slope too steep
    let slope_angle = calculate_slope(support_left, support_right);
    slope_angle <= max_angle
}
```

**Gameplay:**
- Dig too steep → rockfall damages equipment, buries player
- Need to "bench" your pit (stepped walls)
- Or use shoring/supports (later upgrade)

### 3. Water Flow & Velocity

Water has **velocity** based on pressure differential. Velocity determines:
- Erosion rate (fast water removes material)
- Sluice effectiveness (gold separation)
- Flooding danger

```rust
struct WaterCell {
    material: Material,
    velocity: Vec2,  // Flow direction and speed
}

fn update_water_velocity(world: &mut World, x: i32, y: i32) {
    // Pressure = depth (water above)
    let pressure = count_water_above(world, x, y);

    // Velocity from pressure gradient
    let p_left = get_pressure(world, x-1, y);
    let p_right = get_pressure(world, x+1, y);

    let flow_x = (p_left - p_right) * FLOW_COEFFICIENT;
    let flow_y = GRAVITY + (pressure * PRESSURE_COEFFICIENT);

    world.set_velocity(x, y, Vec2::new(flow_x, flow_y));
}
```

### 4. Erosion

Water velocity vs material cohesion determines erosion:

```rust
fn apply_erosion(world: &mut World, x: i32, y: i32) {
    let velocity = world.get_velocity(x, y);
    let speed = velocity.length();

    // Check adjacent solid materials
    for (nx, ny) in neighbors(x, y) {
        let neighbor = world.get(nx, ny);
        if neighbor.is_solid() {
            // Erosion chance = speed vs cohesion
            let erosion_power = speed / neighbor.cohesion();
            if erosion_power > 1.0 && random() < (erosion_power - 1.0) * 0.1 {
                // Erode: solid becomes suspended in water
                world.set(nx, ny, Material::Water);
                // Suspended material carried by flow
                add_suspended_particle(world, x, y, neighbor);
            }
        }
    }
}
```

**Gameplay:**
- Fast water erodes banks → channel changes
- Can intentionally erode to shape terrain
- Uncontrolled erosion = problems

### 5. Sluice Physics (THE CORE MECHANIC)

```
Water + Material IN →
┌─────────────────────────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ ← Fast flow
│ ░░░▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ ← Suspended particles
├─────────────────────────────────────────────────────┤
│ ████                                                │ ← Riffle 1
│     ░░░░░████                                       │ ← Slow zone (gold settles)
│             ████                                    │ ← Riffle 2
│                 ░░░░████                            │
│                         ████                        │ ← Riffle 3
├─────────────────────────────────────────────────────┤
│ ●●●● Gold (settled)                                 │
│      ▒▒▒▒ Gravel (some settled)                    │
│           ░░░░ Mud/dirt (washed out) →             │
└─────────────────────────────────────────────────────┘
                                      → Tailings OUT
```

**How it works:**
1. Material enters with water flow
2. Behind each riffle = low velocity zone
3. Heavy particles (gold) settle in low-velocity zones
4. Light particles (mud) stay suspended, wash out
5. Player adjusts: water flow rate, riffle spacing, sluice angle

**Variables player controls:**
- Water input rate (too fast = gold washes out, too slow = clogs)
- Sluice angle (steeper = faster, but less settling time)
- Riffle design (spacing, height)
- Material input rate (overload = poor separation)

```rust
fn calculate_settling(velocity: f32, particle_density: f32) -> f32 {
    // Stokes settling velocity
    let settling_velocity = (particle_density - WATER_DENSITY) * GRAVITY / DRAG;

    // If settling > flow, particle drops
    // If flow > settling, particle stays suspended
    settling_velocity - velocity.y
}
```

---

## Progression System

### Early Game (Manual)
- **Tools:** Pickaxe, shovel, bucket
- **Transport:** Carry by hand
- **Processing:** Small hand sluice
- **Challenge:** Find good ground, learn sluice mechanics

### Mid Game (Mechanized)
- **Tools:** Small front-end loader
- **Transport:** Wheelbarrow, small conveyor
- **Processing:** Powered trommel, larger sluice
- **Challenge:** Manage pit stability, water supply

### Late Game (Industrial)
- **Tools:** Excavator, haul truck
- **Transport:** Conveyor systems, hoppers
- **Processing:** Wash plant, centrifugal concentrator
- **Challenge:** Large-scale water management, tailings disposal

---

## Risk Mechanics

### Rockfall/Collapse
- **Trigger:** Exceeding angle of repose
- **Effect:** Material falls, can damage equipment, trap player
- **Prevention:** Proper pit design, benching, shoring

### Flooding
- **Trigger:** Hit water table, breach into water body
- **Effect:** Pit fills with water, drowns operation
- **Prevention:** Pumps, proper planning, drainage channels

### Equipment Damage
- **Trigger:** Rockfall, improper use, flooding
- **Effect:** Equipment needs repair, costs money
- **Prevention:** Good engineering practices

### Gold Loss
- **Trigger:** Poor sluice setup
- **Effect:** Gold washes to tailings (lost forever)
- **Prevention:** Test and tune sluice before processing pay dirt

---

## Implementation Phases

### Phase 1: Core Physics (Current Focus)
- [x] Basic falling sand CA
- [x] Material types with density
- [ ] **Angle of repose** (stability/collapse)
- [ ] **Water velocity** (pressure-based flow)
- [ ] **Erosion** (velocity vs cohesion)

### Phase 2: Sluice Mechanics
- [ ] Velocity-based particle settling
- [ ] Riffle placement and effect
- [ ] Adjustable water flow
- [ ] Gold recovery calculation

### Phase 3: Player & World
- [ ] Player sprite with collision
- [ ] Digging mechanics (pickaxe interaction)
- [ ] Inventory (carry material)
- [ ] World generation (layers, water bodies)

### Phase 4: Equipment
- [ ] Rigid body physics (loader, truck)
- [ ] Equipment controls
- [ ] Bucket/scoop interaction with particles

### Phase 5: Progression
- [ ] Economy (sell gold)
- [ ] Equipment upgrades
- [ ] Larger processing equipment

### Phase 6: Polish
- [ ] Sound effects (water, digging, machinery)
- [ ] Particle effects (dust, splashing)
- [ ] UI/HUD
- [ ] Tutorial

---

## Technical Approach

### Keep from Current System
- Chunk-based world (64x64)
- Active cell tracking (only update moving particles)
- Material system with density
- Basic rendering pipeline

### Add
- **Velocity field** for water (sparse - only store for active water)
- **Stability checking** for solids (angle of repose)
- **Rigid body integration** for equipment (separate physics layer)

### Performance Strategy
1. **Sparse velocity storage** - only water cells need velocity
2. **Lazy stability checks** - only check when material added/removed nearby
3. **Dirty rects** - only re-render changed chunks
4. **Sleep detection** - skip stable regions entirely

### Data Structures

```rust
// Material with all properties
#[derive(Clone, Copy)]
pub enum Material {
    Air,
    Water,
    Mud,
    Topsoil,
    Sand,
    Gravel,
    Clay,
    Rock,
    Bedrock,
    Gold,
}

impl Material {
    pub fn density(&self) -> f32 { ... }
    pub fn cohesion(&self) -> f32 { ... }
    pub fn angle_of_repose(&self) -> f32 { ... }
    pub fn is_liquid(&self) -> bool { ... }
    pub fn is_diggable(&self) -> bool { ... }
}

// Velocity stored separately (sparse)
pub struct World {
    chunks: FxHashMap<(i32, i32), Chunk>,
    velocity: FxHashMap<(i32, i32), Vec2>,  // Only for water cells
    // ...
}
```

---

## Success Criteria

### Physics
- [ ] Loose material collapses at correct angle
- [ ] Water flows based on pressure gradient
- [ ] Water velocity erodes banks
- [ ] Heavy particles settle in slow water
- [ ] Light particles wash away in fast water

### Gameplay
- [ ] Player can dig and transport material
- [ ] Sluice separates gold from dirt
- [ ] Poor sluice setup = lost gold
- [ ] Steep pit walls collapse
- [ ] Player can upgrade equipment

### Performance
- [ ] 60 FPS at 512x512 minimum
- [ ] Smooth particle flow (no stuttering)
- [ ] Equipment moves smoothly through particles

---

## References

- [Noita GDC Talk](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design/)
- [Sluice Box Physics](https://www.911metallurgist.com/blog/how-a-sluice-box-works)
- [Angle of Repose](https://en.wikipedia.org/wiki/Angle_of_repose)
- [Stokes Law (settling)](https://en.wikipedia.org/wiki/Stokes%27_law)
- Research doc: `docs/research/particle-physics-approaches.md`

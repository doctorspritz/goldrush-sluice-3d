<!-- TODO: Update this architecture doc to reflect current 3D FLIP + DEM approach -->

# Gold Rush: Physics-Driven Mining Simulation

## Vision

An open-world gold mining game where **everything is physics**. No scripted events, no artificial constraints. Success and failure emerge from the simulation itself.

Player skill = understanding and managing interconnected physical systems.

---

## Core Principle: Emergent Gameplay

```
┌─────────────────────────────────────────────────────────────┐
│                   TRADITIONAL GAME                          │
│                                                             │
│   Player action → Game checks rules → Scripted outcome     │
│   "Tailings full" → Show warning → Block further action    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    THIS GAME                                │
│                                                             │
│   Player action → Physics simulation → Emergent outcome    │
│   Tailings settle → Bed rises → Outflow blocks → Backup   │
│   No warnings. No blocks. Just physics.                    │
└─────────────────────────────────────────────────────────────┘
```

---

## System Architecture

### The World Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCEDURAL TERRAIN                       │
│         (Mountains, valleys, creek beds, gold deposits)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TERRAIN HEIGHTFIELD                      │
│                                                             │
│   base_height[x,z]      - Bedrock, original ground         │
│   sediment_height[x,z]  - Deposited material (grows!)      │
│   material_type[x,z]    - Sand, gravel, clay, permafrost   │
│                                                             │
│   Collapse mechanics: angle of repose, mass transfer       │
│   Plan: dig_test.rs (existing)                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    WATER LAYER                              │
│                                                             │
│   surface_height[x,z]   - Water level                      │
│   flow_x[x,z]           - Velocity field                   │
│   flow_z[x,z]                                              │
│   suspended_sediment[x,z] - Turbidity, settles over time   │
│                                                             │
│   Shallow water equations, mass-conserving                 │
│   Plan: low-fidelity-water-tailings.md                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ACTIVE ZONES                             │
│         (Where full particle simulation runs)               │
│                                                             │
│   FLIP/APIC water       - Vorticity, turbulence            │
│   Drucker-Prager solids - Angle of repose, yield, clog     │
│   Multi-material        - Water, sand, gravel, gold        │
│                                                             │
│   Plans: gpu3d-slurry.md, drucker-prager-sediment-impl.md  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    EQUIPMENT                                │
│         (Physical objects with throughput limits)           │
│                                                             │
│   Excavator   - Digs terrain → spawns particles            │
│   Loader      - Moves pay piles                            │
│   Hopper      - Receives material, feeds downstream        │
│   Trommel     - Separates by size                          │
│   Conveyor    - Transports material (has capacity)         │
│   Sluice      - Gold separation (vorticity-dependent)      │
│                                                             │
│   Equipment doesn't have "stats" - it has geometry         │
│   Throughput = how fast physics moves material through     │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Mining → Processing → Tailings

```
TERRAIN                    PARTICLES                   TERRAIN
(heightfield)              (active zone)               (heightfield)
     │                          │                           ▲
     │ excavate()               │                           │
     │ spawns particles         │                           │ settle()
     ▼                          ▼                           │ deposits sediment
┌─────────┐              ┌─────────────┐              ┌─────────┐
│ Pay Dirt │   loader    │  Wash Plant │   outflow   │ Tailings │
│  Pile    │ ─────────→  │   (active)  │ ──────────→ │  Area    │
└─────────┘              └─────────────┘              └─────────┘
     │                          │
     │ collapse()               │ gold settles in sluice
     ▼                          ▼
  Spreads if              ┌─────────┐
  too steep               │  GOLD   │
                          └─────────┘
```

### Water Cycle

```
SOURCE                     ACTIVE                      SINK
(creek, pond, pump)        (wash plant)               (tailings, ground)
     │                          │                           │
     │ water_surface            │ FLIP/APIC                │ water_surface
     │ heightfield              │ particles                 │ heightfield
     ▼                          ▼                           ▼
┌─────────┐              ┌─────────────┐              ┌─────────┐
│  Water  │   inflow     │  Processing │   outflow   │ Settling │
│  Supply │ ──────────→  │    Zone     │ ──────────→ │   Area   │
└─────────┘              └─────────────┘              └─────────┘
     │                          │                           │
     │ if depleted:             │ if water low:            │ if backs up:
     │ flow → 0                 │ velocity drops           │ outflow blocks
     │                          │ gold blows out           │ floods upstream
     ▼                          ▼                           ▼
  SYSTEM STARVES           GOLD LOST                  SYSTEM FLOODS
```

---

## Zone Transitions

### Heightfield ↔ Particles

```rust
// Terrain → Particles (excavation)
fn excavate(terrain: &mut Terrain, pos: Vec3, radius: f32) -> Vec<Particle> {
    let dug_volume = terrain.remove_material(pos, radius);
    spawn_particles(dug_volume, terrain.material_at(pos))
}

// Particles → Terrain (settling)
fn settle_particles(particles: &mut Vec<Particle>, terrain: &mut Terrain) {
    for p in particles.iter_mut() {
        if p.is_stationary_long_enough() {
            terrain.add_sediment(p.position, p.material, p.volume());
            p.mark_for_removal();
        }
    }
}
```

### Water Heightfield ↔ FLIP Particles

```rust
// Heightfield → Particles (entering active zone)
fn spawn_water_at_boundary(water: &WaterLayer, zone: &mut ActiveZone) {
    let inflow_rate = water.flow_into_zone();
    zone.spawn_water_particles(inflow_rate);
}

// Particles → Heightfield (exiting active zone)
fn drain_to_heightfield(zone: &ActiveZone, water: &mut WaterLayer) {
    let (water_vol, sediment_vol) = zone.outflow_volumes();
    water.add_inflow(zone.outlet_pos, water_vol);
    water.add_sediment(zone.outlet_pos, sediment_vol);
}
```

---

## Failure Modes (All Emergent)

### Water Starvation

```
Source depletes
      ↓
Inflow to plant drops
      ↓
Water velocity in sluice drops
      ↓
Below critical velocity for gold settling
      ↓
Gold particles exit with waste (physics!)
      ↓
GOLD LOST - not a "game over", just lost revenue
```

**Player response**: Manage water sources, build reservoirs, slow processing rate.

### Tailings Backup

```
Sediment settles in tailings area
      ↓
Bed height rises toward outflow level
      ↓
Outflow velocity decreases (less head)
      ↓
Eventually bed reaches water surface
      ↓
Water can't drain, backs up
      ↓
Floods wash plant from downstream
      ↓
SYSTEM FLOODS - gravel gets wet, clogs, jams
```

**Player response**: Build higher berms, excavate tailings, create overflow channel.

### Pay Pile Collapse

```
Loader dumps material on pile
      ↓
Pile exceeds angle of repose
      ↓
Drucker-Prager yield → material flows
      ↓
Collapses toward equipment/access roads
      ↓
BLOCKS ACCESS - physical obstruction
```

**Player response**: Spread piles wider, build retaining walls, manage pile locations.

### Plant Clog

```
Gravel feed rate > processing capacity
      ↓
Material accumulates in hopper/trommel
      ↓
Drucker-Prager: packed material resists flow
      ↓
Yield stress exceeded → chunks break loose
      OR
      Yield stress not exceeded → JAM
      ↓
PLANT STOPS - need to clear blockage
```

**Player response**: Match feed rate to capacity, maintain water flow, clear jams.

### Sluice Clog (The Interesting One)

```
Sediment deposits behind riffles
      ↓
Bed height rises
      ↓
Flow cross-section decreases
      ↓
Water velocity increases (same flow, smaller area)
      OR
      Water backs up (if bed too high)
      ↓
Either: washes away deposits (self-clearing)
Or: completely blocks, water overflows sides
      ↓
DEPENDS ON GEOMETRY - player learns to read it
```

**Player response**: Clean riffles periodically, adjust water flow, proper sluice angle.

---

## Material Types

| Material | Density | Behavior | Gold Content |
|----------|---------|----------|--------------|
| Water | 1.0 | FLIP/APIC, vorticity | - |
| Clay | 1.6 | Cohesive, clogs screens | Low |
| Sand | 1.8 | Settles medium, angle ~32° | Medium |
| Gravel | 2.0 | Settles fast, angle ~35° | Variable |
| Permafrost | 1.4 (thaws) | Frozen solid → thaws to mud | Variable |
| Black Sand | 2.8 | Heavy, concentrates with gold | Indicator |
| Gold | 19.3 | Very heavy, settles in riffles | 100% |

**Key insight**: Gold is ~10x denser than sand. In turbulent water, sand lifts but gold stays. This is the entire basis of gravity separation.

---

## Active Zone Management

Not everything can be full particle simulation. Use LOD:

| Distance | Terrain | Water | Sediment |
|----------|---------|-------|----------|
| **Active** (player working) | Particles | FLIP/APIC | Particles |
| **Near** (visible, interactive) | Heightfield + collapse | Heightfield + flow | Concentration field |
| **Far** (background) | Static heightfield | Static level | None |

Active zones follow player focus:
- Excavator location → terrain particles active
- Wash plant → water particles active
- Tailings outflow → settling active

Zones can overlap. Boundaries handle transitions.

---

## Implementation Roadmap

### Phase 1: Foundation (Current)
- [x] FLIP/APIC water (vorticity works)
- [x] Basic sediment (settling, drag)
- [ ] Drucker-Prager yield (`drucker-prager-sediment-impl.md`)
- [ ] Heightfield terrain collapse (`dig_test.rs` exists)

### Phase 2: World Layer
- [ ] Unified terrain (base + sediment layers)
- [ ] Water heightfield (shallow water equations)
- [ ] Sediment settling to terrain
- [ ] Zone transitions (heightfield ↔ particles)

### Phase 3: Full Loop
- [ ] Water source → plant → tailings flow
- [ ] Excavation → particles → processing
- [ ] Gold separation in sluice
- [ ] Tailings settling feedback

### Phase 4: Equipment
- [ ] Excavator (terrain → particles)
- [ ] Hopper/feeder (buffer, throughput limit)
- [ ] Trommel (size separation)
- [ ] Conveyor (transport, capacity)
- [ ] Sluice (gold recovery)

### Phase 5: World
- [ ] Procedural terrain generation
- [ ] Gold deposit distribution
- [ ] Water sources (creeks, ponds)
- [ ] Seasonal variation

### Phase 6: Polish
- [ ] Multi-material (clay, permafrost)
- [ ] Weather effects
- [ ] Equipment wear
- [ ] Economy (sell gold, buy equipment)

---

## Detailed Plans

| System | Plan File | Status |
|--------|-----------|--------|
| GPU water + sediment | `gpu3d-slurry.md` | On branch |
| Drucker-Prager yield | `drucker-prager-sediment-impl.md` | Ready for Codex |
| Terrain collapse | `dig_test.rs` | Exists |
| Water + tailings | `low-fidelity-water-tailings.md` | Written |
| 3D vorticity | `3d-vorticity-confinement.md` | Merged |

---

## Technical Constraints

### Performance Budget

Target: 60 FPS with
- 500k-1M particles in active zone
- 200x200 heightfield cells (40k)
- Multiple active zones

### GPU Pipeline Order

```
1. P2G scatter (water momentum, sediment count)
2. Sediment fraction
3. Sediment pressure (Drucker-Prager)
4. Boundary conditions
5. Pressure solve (water incompressibility)
6. Porosity drag (sediment slows water)
7. Vorticity confinement
8. G2P (water: FLIP/APIC, sediment: D-P yield)
9. Advection + collision
10. Bed exchange (deposit/entrain)
```

### Memory Budget (1M particles)

| Buffer | Size |
|--------|------|
| Positions | 12 MB |
| Velocities | 12 MB |
| C matrix | 36 MB |
| Material/density | 4 MB |
| Grid (100³) | ~50 MB |
| **Total** | ~120 MB |

---

## Design Principles

### 1. No Invisible Walls
If water can flow there, it flows. If gravel can fall there, it falls. Containment comes from player-built structures, not game boundaries.

### 2. No Scripted Events
"Tailings full" isn't a game event. It's physics: bed rises until it blocks outflow. The player sees it coming and can react (or not).

### 3. Failure is Feedback
Lost gold isn't "game over". It's information: your sluice velocity was wrong. Adjust and continue. The simulation doesn't punish - it teaches.

### 4. Expertise Transfers
Real gold miners should recognize the systems. Real knowledge should help. The game rewards understanding physics, not memorizing game mechanics.

### 5. Emergence Over Design
Don't design "interesting situations". Design accurate physics. Interesting situations emerge when systems interact.

---

## Success Criteria

The game is working when:

1. **A real miner watches and says "yeah, that's how it works"**
2. **Players develop intuition that transfers to other physics scenarios**
3. **Failure modes are predictable from understanding the systems**
4. **No two playthroughs are identical (procedural + emergent)**
5. **Speedrunners optimize physical efficiency, not exploit game logic**

---

## The Dream

A player watches their tailings area slowly fill over hours of play. They notice the bed creeping toward outflow height. They have options:

- Build the berm higher (buys time)
- Dig a new channel (redirects flow)
- Excavate old tailings (creates space, maybe recover fines)
- Do nothing (accept the eventual backup)

None of these are "quest objectives". They're physical realities the player learns to manage. The game never tells them "tailings critical" - they see the water level rising and know what it means.

That's emergent gameplay from physics.

---
title: "Foundational Physics Design Over Quick Workarounds"
type: design-philosophy
category: architecture-decision
component: particle-physics-simulation-engine
severity: high
status: resolved
date: 2025-12-19

problem:
  type: design-philosophy
  core_issue: "Treating physics as quick fixes instead of foundational system"
  symptoms:
    - "Water mountains and unnatural accumulation"
    - "Chunk boundary artifacts"
    - "Magic floating columns from auto-leveling"
    - "Special-case code for each behavior"

root_cause: "Cellular automata with ad-hoc fixes addresses symptoms rather than building consistent physics"

tags:
  - physics-engine
  - particle-systems
  - cellular-automata
  - architecture
  - design-philosophy
  - emergent-behavior

related_files:
  - docs/research/particle-physics-approaches.md
  - plans/goldrush-physics-engine.md
  - plans/fix-water-leveling.md
  - crates/sim/src/world.rs
---

# Foundational Physics Design Over Quick Workarounds

## The Insight

> "We need to think about these things in a more abstract fashion - not looking for quick dirty wins for core gameplay aspects. Need to nail this. No shitty workarounds."

When building physics-driven gameplay, **emergent behavior from consistent rules** beats **special-case code for each mechanic**.

## The Problem Journey

### What We Tried (And Why It Failed)

| Attempt | Approach | Result |
|---------|----------|--------|
| 1 | Basic CA rules | Water formed mountains, no leveling |
| 2 | Column height comparison | Chunk boundaries broke calculation |
| 3 | Global water leveling pass | Fixed leveling, killed FPS |
| 4 | Auto-level static water | Created magic floating columns |
| 5 | Surface flood-fill + gradual leveling | Still had chunk artifacts |

Each "fix" addressed a symptom while creating new edge cases. The code grew more complex with each iteration.

### The Root Cause

We were asking the wrong question:
- **Wrong:** "How do I make water level correctly?"
- **Right:** "What physics rules produce leveling as emergent behavior?"

## The Solution: Hybrid Physics Architecture

### Three-Layer System

```
┌─────────────────────────────────────────────────┐
│  Layer 3: RIGID BODIES (Vehicles/Equipment)     │
│  - Articulated bodies with IK control           │
│  - Separate from grid, overlays particles       │
├─────────────────────────────────────────────────┤
│  Layer 2: VELOCITY FIELD                        │
│  - Every active particle has (vx, vy)           │
│  - Gravity accumulates, drag transfers          │
├─────────────────────────────────────────────────┤
│  Layer 1: STRESS FIELD                          │
│  - Solid cells have stability (0.0 to 1.0)      │
│  - Propagates from anchors with decay           │
└─────────────────────────────────────────────────┘
```

### 1. Structural Integrity via Stress Propagation

**Old approach:** Check angle of repose (geometric)
**New approach:** Stability field propagates from anchors (physical)

```rust
// Stability propagates from bedrock with material-specific decay
fn propagate_stability(world: &mut World, x: i32, y: i32) {
    let below_stability = world.get_stability(x, y + 1);
    let decay = material.stability_decay(); // Rock: 0.99, Soil: 0.90

    let new_stability = below_stability * decay;
    world.set_stability(x, y, new_stability);

    // State change when stability drops
    if new_stability < FAILURE_THRESHOLD {
        convert_to_loose(world, x, y); // Packed → Loose
    }
}
```

**Why better:**
- Handles overhangs (stability decays in all directions)
- Material-specific strength (rock vs dirt)
- Gradual weakening, not binary collapse
- No "check slope angle" special case

### 2. Universal Velocity + Fluid Drag Coupling

**Old approach:** Separate erosion, settling, leveling systems
**New approach:** Everything has velocity, drag couples them

```rust
// The key equation
Drag Force = FluidPenalty × (FluidVelocity - SolidVelocity)
```

**Emergent behaviors (no special-case code):**

| Scenario | What Happens | Why |
|----------|--------------|-----|
| Dirt in fast water | Floats/suspends | Low mass → drag > gravity |
| Gold in fast water | Sinks | High mass → gravity > drag |
| Water seeking level | Flows to equalize | Velocity from pressure gradient |
| Erosion | Material detaches | Velocity overcomes cohesion |
| Sluice separation | Gold settles, dirt washes | Density determines drag/gravity ratio |

**No sluice code needed.** The physics just works.

### 3. Vehicles as Separate Rigid Body Layer

Vehicles exist outside the particle grid:
- Articulated rigid bodies (chassis, boom, stick, bucket)
- IK control (bucket follows cursor)
- Raycasts for ground contact
- Cutting edge removes particles → inventory
- Resistance force makes digging feel heavy

## Material Properties Matrix

Single source of truth for all behavior:

| Material | Density | Stability Decay | Cohesion | Behavior |
|----------|---------|-----------------|----------|----------|
| Water | 1.0 | - | - | Flows, seeks level |
| Mud | 1.3 | - | 0.2 | Viscous, erodes easily |
| Soil | 1.8 | 0.90 | 0.3 | Loose, collapses |
| Sand | 2.0 | 0.85 | 0.1 | Flows when disturbed |
| Gravel | 2.2 | 0.92 | 0.4 | Contains gold |
| Clay | 2.5 | 0.95 | 0.7 | Sticky, hard to wash |
| Rock | 2.8 | 0.99 | 0.9 | Strong, rare collapse |
| Bedrock | 3.0 | 1.00 | 1.0 | Perfect anchor |
| Gold | 5.0 | 0.93 | 0.8 | Heavy, sinks fast |

All gameplay emerges from these properties + universal physics rules.

## The Philosophy

### Old Approach (Special Cases)
```rust
fn update_world() {
    process_falling();
    process_water_leveling();    // Special case
    process_sluice_separation(); // Special case
    process_erosion();           // Special case
    process_rockfall();          // Special case
    process_auto_level();        // Workaround for water
    process_chunk_boundaries();  // Workaround for chunks
}
```

### New Approach (Emergent Physics)
```rust
fn update_world() {
    propagate_stability();  // Stress field
    for cell in active_cells {
        apply_gravity(&mut cell.velocity);
        apply_drag(&mut cell.velocity, fluid_environment);
        move_cell(cell);
    }
}
```

**The same code produces:**
- Water leveling (velocity equalizes pressure)
- Sluice separation (density vs drag)
- Erosion (velocity vs cohesion)
- Rockfall (stability failure)
- No special cases. No workarounds.

## Key Lessons

### 1. Abstract First, Implement Second
Don't start with "how do I make water level?" Start with "what physical principle causes leveling?" Then implement that principle.

### 2. Emergent > Explicit
If you're writing `if (is_sluice) { do_separation(); }`, you're doing it wrong. The separation should emerge from consistent physics applied everywhere.

### 3. Properties > Behaviors
Define material properties (density, cohesion, stability). Let behaviors emerge from physics acting on those properties.

### 4. One System, Many Effects
Universal velocity + drag produces: settling, suspension, erosion, leveling, separation. One system, not five.

### 5. Workarounds Compound
Each workaround creates edge cases that need more workarounds. Foundational design eliminates the need for workarounds.

## Implementation Priority

1. **Velocity field** - Every active cell gets (vx, vy)
2. **Drag coupling** - Fluids transfer momentum to solids
3. **Stress propagation** - Stability from anchors
4. **Material properties** - Density, cohesion, stability decay
5. **Remove workarounds** - Delete auto-leveling hacks

## Related Documentation

- [Particle Physics Approaches](../research/particle-physics-approaches.md) - Comparative analysis of simulation methods
- [Goldrush Physics Engine Plan](../../plans/goldrush-physics-engine.md) - Full game physics design
- [Water Leveling Analysis](../../plans/fix-water-leveling.md) - Root cause analysis of the water problem

## References

- Noita GDC Talk: How they mix CA with fluid dynamics
- Stokes Law: Mathematical basis for settling (density vs drag)
- Stress field propagation: Structural engineering principles applied to particles

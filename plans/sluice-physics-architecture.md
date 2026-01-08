# Sluice Physics Architecture

## Overview

This document captures the architecture discussion for scaling the FLIP simulation from a detailed sluice to world-scale gold mining simulation.

## Grid Resolution for Sluice

For a 1.5m x 0.3m x 0.1m sluice with proper riffle vortex resolution:

| Cell Size | Grid Dims | Total Cells | Can Resolve |
|-----------|-----------|-------------|-------------|
| 16mm (current) | 94x19x6 | 10k | Basic flow only |
| 8mm | 188x38x13 | 93k | Large vortexes |
| **5mm** | 300x60x20 | **360k** | Riffle vortexes |
| 3mm | 500x100x33 | 1.6M | Fine turbulence |

**Recommendation: 5mm cells** - captures riffle physics without excessive cost.

## Full Pipeline Architecture

```
HOPPER → SHAKER DECK → SLUICE → TAILINGS
```

### Hopper
- PBD/DEM for rocks/gravel (500-2k particles)
- Shallow water for spray
- Key mechanic: **BLOCKAGE** (arch formation)

### Shaker Deck
- DEM for rocks/gravel with vibration (2-5k particles)
- Shallow water for flow around obstacles
- Perforated surface - water drains through
- Key mechanic: **JAMMING** (rock stuck in grate)

### Sluice
- Full FLIP for water + sediment (300-500k particles)
- 5mm resolution for vortex capture
- Key mechanics: **CLOGGING** (gravel fills riffles), **BLOWOUT** (too much flow)

### Why DEM over PBD for Rocks

PBD looks "floaty" because:
- Constraint-based, not force-based
- Stiffness tied to iteration count
- Bad friction model

DEM gives:
- Sharp impacts (rocks clack)
- Proper friction (rocks stop, don't drift)
- Realistic stacking
- Mass-correct behavior

At 5-10k particles, DEM is affordable (~1.5ms).

## Compute Budget

```
Target: 16ms/frame @ 60fps

Hopper (DEM + shallow water):     ~1.5ms
Shaker (DEM + shallow water):     ~2ms
Sluice (FLIP):                    ~12ms
Total:                            ~15.5ms
```

## Sediment Physics - SIMPLIFIED

Previous approach (Drucker-Prager + bed heightfield) failed because:
1. Nothing deposits (threshold too high)
2. Everything deposits instantly (threshold too low)
3. Bed doesn't affect flow (boundary coupling broken)
4. Oscillation/instability

**New approach: Friction-only clustering**

```wgsl
// In G2P, for sediment particles
if (density > 1.0) {
    // 1. Normal FLIP/PIC velocity update
    // 2. Add settling velocity
    final_velocity.y -= settling_velocity * dt;

    // 3. If slow, apply friction
    let speed = length(final_velocity);
    if (speed < friction_threshold) {
        let friction = friction_strength * (1.0 - speed / friction_threshold);
        final_velocity *= 1.0 - friction;
    }

    // 4. Vorticity lift (keeps particles suspended in turbulence)
    if (vorticity > vorticity_threshold) {
        final_velocity.y += lift_factor * settling_velocity * dt;
    }
}
```

This creates emergent behavior:
- Sediment settles in slow zones (behind riffles)
- Clustering happens naturally (slow particles slow further)
- Vortices keep particles suspended
- No threshold tuning hell, no heightfield coupling

## World-Scale Architecture

For km-scale world with localized detail:

### Layer 1: Terrain Heightfield (2.5D)
- 1m cells, covers entire world
- Stores: height, material, moisture
- Always active, very cheap

### Layer 2: Surface Water (2D Shallow Water)
- 2-5m cells
- Stores: water height h, velocity (u,v), sediment_load
- Rivers, runoff, drainage
- ~50k cells for km^2

### Layer 3: Active Zones (3D FLIP)
- 5-50cm cells, small regions
- Spawned on demand near player/action
- Full 3D vortex physics
- Expensive but localized

### Layer 4: DEM Solids
- Individual rocks, excavated material
- Only near player/machinery
- Settles back to heightfield when inactive

### Zone Transitions

**Shallow Water -> 3D FLIP (inflow):**
```rust
let flux = cell.h * cell.velocity().dot(boundary.normal) * boundary.width;
let particles_to_spawn = (flux * dt / PARTICLE_VOLUME) as usize;
// Spawn at boundary with shallow water velocity
```

**3D FLIP -> Shallow Water (outflow):**
```rust
// Particles exiting get removed, add mass to shallow water
let total_volume = exiting.iter().map(|p| p.volume).sum();
shallow.add_volume_at(boundary.shallow_pos, total_volume);
```

**Heightfield <-> DEM (digging):**
```rust
// Dig: heightfield -> DEM particles
cell.height -= remove_depth;
dem.spawn(cell.pos, velocity, material);

// Settle: DEM -> heightfield (when stable)
if particle.velocity < threshold && stable_for_n_seconds {
    heightfield.add_height(particle.pos, particle.volume);
    particle.remove();
}
```

## Frozen Zone Visuals

When player leaves an area, freeze it:
- Water surface -> static mesh (with animated shader)
- Rocks -> instanced static meshes
- Sediment -> texture on heightfield
- Store totals for unfreezing later

Rendering frozen zone costs ~0.1ms vs ~12ms for active FLIP.

## Emergent Mechanics (No Fake Game States)

All mechanics emerge from physics:

### Blockage (Hopper)
- Rocks arch naturally from contact forces
- Material piles up behind arch
- Flow rate -> 0 because particles physically can't pass

### Jam (Shaker)
- Rock wedged by vibration + contacts
- Water flow reduced through that hole
- Material accumulates upstream

### Clogging (Sluice)
- Gravel fills recirculation zone
- No room for vortex
- Gold has no shelter, washes out

### Blowout (Sluice)
- Water velocity too high
- Drag > settling force
- Fine particles carried out

UI "warnings" just MEASURE physics:
- "Blockage detected" = flow rate dropped to ~0
- "Riffle filling" = gravel count increasing in zone
- "Gold escaping" = gold particles passing exit plane

# Physics Redesign Summary

**Date:** 2025-12-20

## What Happened

I kept making incremental hacks to "couple" water (Virtual Pipes) with particles (Cellular Automata). Each hack created new problems. Eventually water moved LESS than gold - complete system failure.

## What I Learned

Your core insight was correct from the start:

> "ALL PARTICLES ARE WATER. NON WATER PARTICLES ARE A LITTLE MORE DENSE, OTHERWISE THEY DISPLAY THE SAME FUCKING FLUID PHYSICS"

I finally understood: there shouldn't be TWO physics systems. There should be ONE.

## Research Completed

I studied:
- [Noita GDC Talk](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design/) - They DON'T do unified physics (CA + separate particle effects)
- [Powder Toy Source](https://github.com/The-Powder-Toy/The-Powder-Toy/blob/master/src/simulation/Air.cpp) - Grid pressure/velocity field
- [Virtual Pipes](https://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html) - Shallow water equations
- [Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf) - Truly unified particle physics
- [Sluice Box CFD](https://www.researchgate.net/publication/383446350) - Real gold settling physics

## The Correct Design

Everything is MASS in a PRESSURE-VELOCITY field:

1. **Mass Array**: Each cell has total mass (water + particle)
2. **Pressure**: Calculated from mass above (P = sum(mass_above) × g)
3. **Velocity**: Calculated from pressure gradients
4. **Movement**: All mass moves with velocity, weighted by inverse density
   - Water (density 10) moves at full velocity
   - Gold (density 250) moves at 1/25th velocity
5. **Settling**: When velocity < threshold for a density, material sinks through lighter material

## Files Created

| File | Purpose |
|------|---------|
| `plans/unified-pressure-based-physics.md` | Complete design for new system |
| `docs/compound/failure-incremental-hacks-destroy-systems.md` | Anti-pattern documentation |
| `docs/compound/learning-unified-fluid-particle-physics.md` | Research summary |

## Files That Need Replacement

| File | Why |
|------|-----|
| `crates/sim/src/water.rs` | Hacked mess of Virtual Pipes + particle coupling attempts |
| `crates/sim/src/update.rs` | Separate CA physics for particles |
| `crates/sim/src/fluid.rs` | Separate Navier-Stokes velocity field |

## What Needs To Be Built

A new `crates/sim/src/physics.rs` with:
1. `calculate_mass()` - Distribution of mass in grid
2. `calculate_pressure()` - Pressure from mass above
3. `calculate_velocity()` - Velocity from pressure gradients
4. `advect_mass()` - Move all mass with velocity (density-weighted)
5. `settle_particles()` - Dense sinks through light when velocity low
6. `spread_particles()` - No friction = particles can't form towers

## Testing Criteria

1. Water levels flat within 1-2 seconds
2. Gold sinks through water and settles behind riffles
3. Soil is carried by flow unless it settles in slow zones
4. No 1-pixel walls - particles spread naturally
5. Pressure pushes particles - water doesn't seep through piles
6. Riffles work - gold accumulates, light material washes out

## Next Steps

When you're ready to implement:
1. Read `plans/unified-pressure-based-physics.md` for the complete design
2. Create new `physics.rs` based on that design
3. Replace calls to old systems with new unified physics
4. Test with sluice box scenario

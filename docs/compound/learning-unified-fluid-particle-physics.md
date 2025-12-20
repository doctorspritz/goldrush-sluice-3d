# Learning: Unified Fluid-Particle Physics

**Category:** Physics / Simulation
**Date:** 2025-12-20
**Context:** Goldrush Fluid Miner - Sluice Box Simulation

## The User's Core Request

> "ALL PARTICLES ARE WATER. NON WATER PARTICLES ARE A LITTLE MORE DENSE, OTHERWISE THEY DISPLAY THE SAME FUCKING FLUID PHYSICS"

This is the fundamental insight I failed to understand through many iterations of hacking.

## What I Did Wrong

### The Flawed Mental Model

I thought of the system as:
- **Water**: Fluid that flows (Virtual Pipes)
- **Particles**: Solid things that sit in water (Cellular Automata)
- **Interaction**: Hacks to make them "talk" to each other

### The Correct Mental Model

Everything is MASS in a PRESSURE-VELOCITY FIELD:
- **Water**: Mass with density 10
- **Soil**: Mass with density 30
- **Gold**: Mass with density 250
- **Physics**: ONE system that moves all mass based on pressure gradients

The difference between water and gold is NOT different physics - it's different DENSITY in the same physics.

## Research Summary

### What Noita Does (NOT what user wanted)
- CA for materials
- Separate particle system for "splash" effects
- **Does NOT calculate pressure**
- Masks limitations with visual effects

### What Powder Toy Does (closer to user's need)
- CA for materials
- Separate **air pressure/velocity field**
- Materials interact with pressure based on properties
- See: [Air.cpp](https://github.com/The-Powder-Toy/The-Powder-Toy/blob/master/src/simulation/Air.cpp)

### Real Sluice Box Physics
- Water flows down slope (gravity + pressure)
- Riffles create **low-pressure vortices**
- Gold settles in low-velocity zones behind riffles
- Light material gets carried out by flow
- Optimal angle: ~10 degrees

### Position Based Fluids (academic ideal)
- Everything is particles
- Density constraints keep fluid incompressible
- Truly unified - no separate systems
- Expensive but possible for small scales

## The Correct Architecture

### One Pressure Field
```
pressure[x,y] = sum of all mass above × gravity
```

### One Velocity Field
```
velocity += pressure_gradient × dt
```

### All Mass Moves Together
```
movement = velocity × (reference_density / material_density)
```

Water (density 10) moves at full velocity.
Gold (density 250) moves at 1/25th velocity.

### Settling is Natural
When velocity drops below threshold for a density, that material sinks through lighter material below it.

## Key Equations

From [Virtual Pipes](https://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html):
```
flow_acceleration = gravity × height_difference / pipe_length
```

From [Powder Toy](https://github.com/The-Powder-Toy/The-Powder-Toy):
```
divergence = vx[left] - vx[right] + vy[up] - vy[down]
pressure += divergence × coefficient
velocity += pressure_gradient
```

Extended for unified mass:
```
pressure = cumulative_mass_above × gravity
velocity += (pressure_left - pressure_right, pressure_up - pressure_down + gravity)
displacement = velocity × (10.0 / density) × dt
```

## Files Created

1. `plans/unified-pressure-based-physics.md` - Complete design for the new system
2. `docs/compound/failure-incremental-hacks-destroy-systems.md` - Anti-pattern documentation
3. `docs/compound/learning-unified-fluid-particle-physics.md` - This file

## What Needs to Happen Next

### Delete/Replace
- `crates/sim/src/water.rs` - Current hacked mess
- `crates/sim/src/update.rs` - CA rules for particles (separate physics)
- `crates/sim/src/fluid.rs` - Navier-Stokes (separate velocity field)

### Create New
- `crates/sim/src/physics.rs` - Unified pressure-velocity system

### Modify
- `crates/sim/src/chunk.rs` - Add pressure array, simplify data structures
- `crates/sim/src/world.rs` - Call unified physics instead of separate systems

## Testing Criteria

1. Pour water on slope - it flows down and levels
2. Drop gold in water - it sinks
3. Add soil to flowing water - it gets carried
4. Build sluice with riffles - gold accumulates behind riffles
5. No 1-pixel walls - particles spread naturally
6. Water doesn't seep through piles - pressure pushes or flows over

## References

- [Powder Toy Air.cpp](https://github.com/The-Powder-Toy/The-Powder-Toy/blob/master/src/simulation/Air.cpp)
- [Virtual Pipes Blog](https://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html)
- [Matthias Müller Height Field Paper](https://matthias-research.github.io/pages/publications/hfFluid.pdf)
- [Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf)
- [Sluice Box CFD Research](https://www.researchgate.net/publication/383446350_Optimization_of_Sluice_Box_for_Small_Scale_Mining_Using_Computational_Fluid_Dynamics_CFD)
- [Noita GDC Talk](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design/)

## Tags

#physics #fluid-simulation #unified-systems #learning #falling-sand

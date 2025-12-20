# Failure: Incremental Hacks Destroy Systems

**Category:** Anti-pattern / Failure Mode
**Date:** 2025-12-20
**Context:** Goldrush Fluid Miner - Particle/Water Physics

## The Problem

User asked for unified particle-water physics where "ALL PARTICLES ARE WATER, just denser."

Instead of redesigning the system to be truly unified, I made incremental hacks:
1. Added flow thresholds for particles
2. Added soil→mud transformation
3. Added "turbulence" via velocity field
4. Added "pressure push"
5. Added "diagonal settling"
6. Added "inherent instability"

**Result:** Water now moves LESS than gold. Complete inversion of intended behavior.

## What The User Actually Asked For

> "ALL PARTICLES ARE WATER. NON WATER PARTICLES ARE A LITTLE MORE DENSE, OTHERWISE THEY DISPLAY THE SAME FUCKING FLUID PHYSICS"

> "any particle submerged/in contact with water behaves according to its density"

> "the pressure of the water should 'push' the gold too. it doesn't seep, it attempts to 'push' with velocity. if it can't, it flows over"

> "gold shouldn't be able to form a tower of 1 pixel, because the gold particles aren't that adhered to each other, they have density but not friction"

## What Real Physics Does

1. **Everything is mass in a pressure field**
   - Water has mass, gold has mass, soil has mass
   - Pressure = height × density × gravity
   - Flow goes from high pressure to low pressure
   - ALL mass moves with flow, weighted by density

2. **Density determines settling**
   - Denser materials sink through less dense
   - This is NOT a separate system - it's the SAME pressure physics
   - A gold particle in water experiences buoyancy (pressure difference top vs bottom)

3. **No friction between loose particles**
   - Sand/gold/soil particles don't stick to each other
   - They form angle-of-repose piles, not vertical towers
   - Piles spread under their own weight

4. **Flow carries particles**
   - The SAME velocity field moves water AND particles
   - Heavier particles need more force (F=ma, more m = more F needed)
   - But it's not a threshold - it's continuous

## The Anti-Pattern

**Symptom:** Each fix creates a new problem, requiring another fix.

**Root Cause:** Treating the system as patchable instead of redesigning it.

**The Loop:**
1. "Particles don't move with water" → Add flow threshold hack
2. "Threshold too high" → Lower threshold
3. "Now everything moves too much" → Add density scaling
4. "Gold blocks water" → Add pressure push
5. "Gold forms towers" → Add instability hack
6. "Water moves less than gold" → ???

Each hack interacts with previous hacks in unexpected ways. The system becomes incomprehensible.

## The Solution

**STOP. RESEARCH. REDESIGN.**

1. Study how Noita actually does it (they solved this)
2. Study academic papers on particle-fluid coupling
3. Design ONE unified system from first principles
4. Implement the unified system, replacing the hacked mess

## Key Insight

> "If it doesn't pass the sniff test for what really happens, don't code it."

Before writing ANY physics code, ask:
- Does this match what happens when I pour water on sand?
- Does this match what happens in a gold pan?
- Would a physics textbook agree with this approach?

If no → don't code it. Research more.

## References To Study

- Noita GDC talk (they unified particles and fluids)
- SPH (Smoothed Particle Hydrodynamics) - treats everything as particles
- FLIP/PIC methods - hybrid particle-grid
- Real sluice box physics - how gold actually settles

## Tags

#anti-pattern #physics #incremental-hacks #system-design #failure-mode

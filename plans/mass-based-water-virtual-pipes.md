# Mass-Based Water Using Virtual Pipes Method

## STATUS: DEPRECATED

**This approach was fundamentally flawed.** It created TWO separate physics systems (Virtual Pipes for water, CA for particles) that fought each other. Incremental hacks to couple them made things progressively worse.

**See instead:** `unified-pressure-based-physics.md` - A proper unified system where ALL materials (water, soil, gold) use the SAME pressure-velocity physics.

### What Went Wrong

1. Water had its own physics (Virtual Pipes)
2. Particles had separate physics (CA rules)
3. "Coupling" them required endless hacks
4. Each hack created new problems
5. Eventually water moved LESS than gold (complete inversion)

### The Correct Approach

Everything is MASS in a pressure-velocity field. Water, soil, and gold are the same physics with different densities. See unified plan for details.

---

## ORIGINAL PLAN (for reference only)

## Problem Statement

Current particle-based water forms hills with large volumes because each water "cell" moves independently. We need water that flows as continuous currents, naturally finding a level.

## Proposed Solution: Virtual Pipes Method

Based on research, the **Virtual Pipes** method is ideal for falling sand games:
- Extremely stable (never produces negative water)
- Creates continuous currents
- Fast enough for real-time
- Easy to combine with particle-based materials
- Mass conserving

### How It Works

Instead of moving water particles, we track:
1. **Water amount** per cell (0.0 to N, where 1.0 = full cell)
2. **Flow rates** between cells (volume/time through virtual "pipes")

Each frame:
1. Calculate flow based on water height difference + velocity
2. Scale flows if they would drain a cell below zero
3. Transfer water mass between cells
4. Particles (soil, mud, gold) interact with water level, not water particles

## Data Structure Changes

### Current (particle-based)
```rust
// chunk.rs
pub materials: Box<[Material; CHUNK_AREA]>,  // Water is a material type
pub vel_x: Box<[f32; CHUNK_AREA]>,
pub vel_y: Box<[f32; CHUNK_AREA]>,
```

### New (mass-based water)
```rust
// chunk.rs
pub materials: Box<[Material; CHUNK_AREA]>,  // Solid materials only (Rock, Soil, Gold, Mud)
pub water_mass: Box<[f32; CHUNK_AREA]>,      // Water amount per cell (0.0 = dry, 1.0+ = water)
pub flow_right: Box<[f32; CHUNK_AREA]>,      // Flow to right neighbor (volume/time)
pub flow_down: Box<[f32; CHUNK_AREA]>,       // Flow to bottom neighbor (volume/time)
pub vel_x: Box<[f32; CHUNK_AREA]>,           // Keep for Navier-Stokes vortices
pub vel_y: Box<[f32; CHUNK_AREA]>,
```

## Algorithm: Virtual Pipes

### Step 1: Calculate Flow Acceleration
```rust
const GRAVITY: f32 = 9.8;
const PIPE_AREA: f32 = 1.0;  // Cross-section of virtual pipe
const PIPE_LENGTH: f32 = 1.0;  // Distance between cell centers

fn update_flows(chunk: &mut Chunk, dt: f32) {
    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);

            // Skip if solid material blocks flow
            if chunk.materials[idx].is_solid() {
                chunk.flow_right[idx] = 0.0;
                chunk.flow_down[idx] = 0.0;
                continue;
            }

            let water_here = chunk.water_mass[idx];

            // Flow Right
            if x < CHUNK_SIZE - 1 {
                let idx_right = Chunk::index(x + 1, y);
                if !chunk.materials[idx_right].is_solid() {
                    let water_right = chunk.water_mass[idx_right];
                    let height_diff = water_here - water_right;  // Positive = flow right

                    // Accelerate flow based on pressure difference
                    let acceleration = GRAVITY * height_diff * PIPE_AREA / PIPE_LENGTH;
                    chunk.flow_right[idx] += acceleration * dt;

                    // Apply damping to prevent oscillation
                    chunk.flow_right[idx] *= 0.99;
                }
            }

            // Flow Down (similar, but gravity adds to downward flow)
            if y < CHUNK_SIZE - 1 {
                let idx_down = Chunk::index(x, y + 1);
                if !chunk.materials[idx_down].is_solid() {
                    let water_down = chunk.water_mass[idx_down];
                    let height_diff = water_here - water_down + 1.0;  // +1 for gravity

                    let acceleration = GRAVITY * height_diff * PIPE_AREA / PIPE_LENGTH;
                    chunk.flow_down[idx] += acceleration * dt;
                    chunk.flow_down[idx] *= 0.99;
                }
            }
        }
    }
}
```

### Step 2: Scale Outflows (Critical for Stability)
```rust
fn scale_outflows(chunk: &mut Chunk, dt: f32) {
    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);
            let water = chunk.water_mass[idx];

            if water <= 0.0 {
                // No water to flow
                chunk.flow_right[idx] = chunk.flow_right[idx].max(0.0);  // Can only receive
                chunk.flow_down[idx] = chunk.flow_down[idx].max(0.0);
                continue;
            }

            // Sum all outgoing flows
            let mut total_out = 0.0;

            // Right outflow
            if chunk.flow_right[idx] > 0.0 {
                total_out += chunk.flow_right[idx];
            }
            // Down outflow
            if chunk.flow_down[idx] > 0.0 {
                total_out += chunk.flow_down[idx];
            }
            // Left outflow (from neighbor's flow_right pointing at us)
            if x > 0 {
                let flow_left = -chunk.flow_right[Chunk::index(x - 1, y)];
                if flow_left > 0.0 {
                    total_out += flow_left;
                }
            }
            // Up outflow (from neighbor's flow_down pointing at us)
            if y > 0 {
                let flow_up = -chunk.flow_down[Chunk::index(x, y - 1)];
                if flow_up > 0.0 {
                    total_out += flow_up;
                }
            }

            // Scale factor to prevent draining more than available
            let max_out = water / dt;
            if total_out > max_out {
                let scale = max_out / total_out;

                if chunk.flow_right[idx] > 0.0 {
                    chunk.flow_right[idx] *= scale;
                }
                if chunk.flow_down[idx] > 0.0 {
                    chunk.flow_down[idx] *= scale;
                }
                // Note: neighbor flows scaled in their cells
            }
        }
    }
}
```

### Step 3: Update Water Mass
```rust
fn update_water_mass(chunk: &mut Chunk, dt: f32) {
    // Use scratch buffer to avoid order artifacts
    let mut new_mass = chunk.water_mass.clone();

    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);

            // Outflow right
            if x < CHUNK_SIZE - 1 {
                let flow = chunk.flow_right[idx] * dt;
                new_mass[idx] -= flow;
                new_mass[Chunk::index(x + 1, y)] += flow;
            }

            // Outflow down
            if y < CHUNK_SIZE - 1 {
                let flow = chunk.flow_down[idx] * dt;
                new_mass[idx] -= flow;
                new_mass[Chunk::index(x, y + 1)] += flow;
            }
        }
    }

    // Clamp to valid range (should be unnecessary if scaling is correct)
    for mass in new_mass.iter_mut() {
        *mass = mass.max(0.0);
    }

    chunk.water_mass.copy_from_slice(&new_mass);
}
```

## Particle Interaction with Water

### Particles in Water
```rust
fn update_particle_in_water(world: &mut World, wx: i32, wy: i32) -> bool {
    let material = world.get_material(wx, wy);
    let water_level = world.get_water_mass(wx, wy);

    if water_level < 0.1 {
        // Not enough water to interact
        return false;
    }

    let density = material.density();
    let water_density = 10;  // Water reference density

    // Buoyancy: light particles float, heavy sink
    if density < water_density {
        // Try to move up (float)
        // ...
    } else {
        // Try to move down (sink)
        // Also affected by water flow velocity for suspension
        let (vx, vy) = world.get_water_velocity(wx, wy);
        let speed = vx.hypot(vy);
        let settling_velocity = (density - water_density) as f32 * 0.05;

        if speed > settling_velocity {
            // Suspended - move with flow
            // ...
        } else {
            // Settling - sink
            // ...
        }
    }
}
```

### Water Displaces Air
```rust
// When rendering, water_mass > 0.5 shows as water
// Particles can exist IN water (submerged)
// Water fills around particles automatically
```

## Rendering Changes

```rust
// main.rs - render chunk
for y in 0..CHUNK_SIZE {
    for x in 0..CHUNK_SIZE {
        let material = chunk.get_material(x, y);
        let water = chunk.water_mass[Chunk::index(x, y)];

        if material != Material::Air {
            // Render solid material
            let color = material.color_varied(wx, wy);
            // Tint blue if underwater
            if water > 0.3 {
                // Blend with water color
            }
        } else if water > 0.1 {
            // Render water
            let alpha = (water * 200.0).min(200.0) as u8;
            let color = [30, 100, 200, alpha];
            // Add foam based on flow velocity
        }
    }
}
```

## Implementation Phases

### Phase 1: Add Water Mass Array ✅
- [x] Add `water_mass`, `flow_right`, `flow_down` to Chunk
- [x] Keep `Material::Water` for compatibility (can remove later)
- [x] Update rendering to show water based on mass

### Phase 2: Implement Virtual Pipes ✅
- [x] Add `update_flows()` function
- [x] Add `scale_outflows()` function
- [x] Add `update_water_mass()` function
- [x] Call from world update loop

### Phase 3: Particle-Water Interaction
- [ ] Modify `update_powder()` to check water level for buoyancy/suspension
- [ ] Modify `update_liquid()` (for Mud) to interact with water mass
- [ ] Add water displacement when particles enter water cells

### Phase 4: Cross-Chunk Flow ✅
- [x] Handle flow at chunk boundaries
- [x] Synchronize water mass between chunks

### Phase 5: Navier-Stokes Integration
- [ ] Use NS velocity field to add momentum to flows
- [ ] Create vortices in water by modulating flow rates
- [ ] Coupling: NS → flow rates → water movement

## Files to Modify

| File | Changes |
|------|---------|
| `crates/sim/src/chunk.rs` | Add water_mass, flow_right, flow_down arrays |
| `crates/sim/src/material.rs` | Remove Water variant (or keep for compatibility) |
| `crates/sim/src/water.rs` | NEW: Virtual pipes implementation |
| `crates/sim/src/world.rs` | Integrate water update, particle-water interaction |
| `crates/sim/src/update.rs` | Modify particle updates for water interaction |
| `crates/game/src/main.rs` | Update rendering for mass-based water |

## Acceptance Criteria

- [ ] Water naturally levels to flat surface within 1-2 seconds
- [ ] Water flows as continuous currents (visible in foam/velocity visualization)
- [ ] Particles (soil, mud, gold) interact correctly with water level
- [ ] Mud gets carried by water currents
- [ ] No water "disappears" or "appears" (mass conservation)
- [ ] Performance: 60 FPS at 256x256

## References

- [lisyarus: Simulating Water Over Terrain](https://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html) - Virtual Pipes explanation
- [Matthias Müller: Fast Water Simulation](https://matthias-research.github.io/pages/publications/hfFluid.pdf) - Height field methods
- [W-Shadow: Simple Fluid Simulation](https://w-shadow.com/blog/2009/09/01/simple-fluid-simulation/) - Mass-based CA
- [Karl Sims: Fluid Flow](https://www.karlsims.com/fluid-flow.html) - Stable fluids visualization

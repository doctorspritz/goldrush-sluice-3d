# Unified Pressure-Based Physics System

## The Core Problem

Previous implementation had TWO separate physics systems:
1. **Virtual Pipes** for water (pressure-based, mass-conserving)
2. **Cellular Automata** for particles (rule-based, position swapping)

These systems fought each other. Incremental hacks to couple them made things worse.

## The Solution: One Unified System

**Everything is mass in a pressure-velocity field.**

- Water is mass (density 10)
- Soil is mass (density 30)
- Gold is mass (density 250)
- Air is empty (density 0)

There is ONE velocity field. ALL mass responds to it. Heavier mass responds less.

## Research Foundation

This design is based on:
- [Powder Toy Air.cpp](https://github.com/The-Powder-Toy/The-Powder-Toy/blob/master/src/simulation/Air.cpp) - Grid-based pressure/velocity field
- [Virtual Pipes](https://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html) - Shallow water equations
- [Height Field Fluids](https://matthias-research.github.io/pages/publications/hfFluid.pdf) - Particle-water coupling
- Real sluice box physics - Riffles create vortices where gold settles

## Data Structures

```rust
pub struct Chunk {
    // What material is at each cell (Air, Soil, Gold, Rock, etc.)
    pub materials: Box<[Material; CHUNK_AREA]>,

    // Mass at each cell (water mass + particle mass)
    // Water: mass = water_amount * WATER_DENSITY
    // Particles: mass = PARTICLE_DENSITY (they displace water)
    pub mass: Box<[f32; CHUNK_AREA]>,

    // Pressure field - calculated from mass above
    pub pressure: Box<[f32; CHUNK_AREA]>,

    // Velocity field - all mass moves with this
    pub vel_x: Box<[f32; CHUNK_AREA]>,
    pub vel_y: Box<[f32; CHUNK_AREA]>,

    // Tracking
    pub needs_render: bool,
    pub is_active: bool,
}
```

## Material Properties

```rust
impl Material {
    /// Density determines how the material responds to flow
    /// Higher density = harder to move, sinks through lighter
    pub const fn density(self) -> f32 {
        match self {
            Material::Air => 0.0,
            Material::Water => 10.0,   // Reference density
            Material::Soil => 30.0,    // Sinks in water
            Material::Mud => 40.0,     // Heavier soil
            Material::Gold => 250.0,   // Very heavy, sinks fast
            Material::Rock => f32::INFINITY, // Never moves
        }
    }
}
```

## Algorithm: Unified Physics Update

### Step 1: Calculate Mass Distribution

```rust
fn calculate_mass(chunk: &mut Chunk) {
    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);
            let material = chunk.materials[idx];

            if material == Material::Air {
                // Air cells can have water mass (from flow)
                // mass stays as-is (water that flowed here)
            } else if material.is_solid() {
                chunk.mass[idx] = 0.0; // Solid blocks mass
            } else {
                // Particle displaces water, adds its own mass
                let particle_mass = material.density();
                chunk.mass[idx] = chunk.mass[idx].max(particle_mass);
            }
        }
    }
}
```

### Step 2: Calculate Pressure from Mass Above

Pressure at a point = sum of all mass above it × gravity.

```rust
fn calculate_pressure(chunk: &mut Chunk) {
    const GRAVITY: f32 = 9.8;

    for x in 0..CHUNK_SIZE {
        let mut cumulative_mass = 0.0;

        // Top to bottom - accumulate mass above
        for y in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);

            if chunk.materials[idx].is_solid() {
                // Solid resets pressure (acts as ceiling)
                cumulative_mass = 0.0;
                chunk.pressure[idx] = 0.0;
            } else {
                chunk.pressure[idx] = cumulative_mass * GRAVITY;
                cumulative_mass += chunk.mass[idx];
            }
        }
    }
}
```

### Step 3: Calculate Velocity from Pressure Gradients

Flow goes from high pressure to low pressure.

```rust
fn calculate_velocity(chunk: &mut Chunk, dt: f32) {
    const DAMPING: f32 = 0.99;

    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);

            if chunk.materials[idx].is_solid() {
                chunk.vel_x[idx] = 0.0;
                chunk.vel_y[idx] = 0.0;
                continue;
            }

            // Pressure gradient drives velocity
            let p_left = if x > 0 { chunk.pressure[Chunk::index(x-1, y)] } else { chunk.pressure[idx] };
            let p_right = if x < CHUNK_SIZE-1 { chunk.pressure[Chunk::index(x+1, y)] } else { chunk.pressure[idx] };
            let p_up = if y > 0 { chunk.pressure[Chunk::index(x, y-1)] } else { chunk.pressure[idx] };
            let p_down = if y < CHUNK_SIZE-1 { chunk.pressure[Chunk::index(x, y+1)] } else { chunk.pressure[idx] };

            // Accelerate based on pressure difference
            // Positive pressure_diff means pressure is higher on left, so flow goes right
            let accel_x = (p_left - p_right) * dt;
            let accel_y = (p_up - p_down) * dt + GRAVITY * dt; // +gravity for downward

            chunk.vel_x[idx] = (chunk.vel_x[idx] + accel_x) * DAMPING;
            chunk.vel_y[idx] = (chunk.vel_y[idx] + accel_y) * DAMPING;
        }
    }
}
```

### Step 4: Move Mass with Velocity (Advection)

**Key insight**: ALL mass moves with velocity, but heavier mass moves less.

```rust
fn advect_mass(chunk: &mut Chunk, dt: f32) {
    let mut new_mass = [0.0f32; CHUNK_AREA];
    let mut new_materials = [Material::Air; CHUNK_AREA];

    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);
            let material = chunk.materials[idx];
            let mass = chunk.mass[idx];

            if material.is_solid() || mass < 0.01 {
                new_materials[idx] = material;
                continue;
            }

            let vx = chunk.vel_x[idx];
            let vy = chunk.vel_y[idx];
            let velocity = (vx * vx + vy * vy).sqrt();

            // Movement response is inverse to density
            // Water (density 10) moves easily
            // Gold (density 250) needs 25x more force
            let density = material.density().max(1.0);
            let response = 10.0 / density; // 10.0 = water density as reference

            // Calculate displacement
            let dx = (vx * response * dt).clamp(-1.0, 1.0);
            let dy = (vy * response * dt).clamp(-1.0, 1.0);

            // Determine target cell
            let target_x = (x as f32 + dx).round() as i32;
            let target_y = (y as f32 + dy).round() as i32;

            if target_x >= 0 && target_x < CHUNK_SIZE as i32
               && target_y >= 0 && target_y < CHUNK_SIZE as i32 {
                let target_idx = Chunk::index(target_x as usize, target_y as usize);
                let target_mat = chunk.materials[target_idx];

                if !target_mat.is_solid() {
                    // Can move here - transfer mass
                    new_mass[target_idx] += mass;

                    // Material goes where mass goes (particles move with flow)
                    if material != Material::Air {
                        // Particle moves with flow (if velocity sufficient)
                        // OR sinks if velocity too low (settling)
                        if velocity * response > 0.1 || target_y > y as i32 {
                            new_materials[target_idx] = material;
                        } else {
                            new_materials[idx] = material; // Stay in place
                            new_mass[idx] += mass; // Keep mass here
                            new_mass[target_idx] -= mass; // Undo transfer
                        }
                    }
                } else {
                    // Blocked - mass stays
                    new_mass[idx] += mass;
                    new_materials[idx] = material;
                }
            } else {
                // Out of bounds - keep here
                new_mass[idx] += mass;
                new_materials[idx] = material;
            }
        }
    }

    chunk.mass.copy_from_slice(&new_mass);
    chunk.materials.copy_from_slice(&new_materials);
}
```

### Step 5: Settling (Dense Through Light)

When velocity is too low to carry a particle, it sinks through lighter material.

```rust
fn settle_particles(chunk: &mut Chunk) {
    // Bottom to top, so settled particles don't double-move
    for y in (0..CHUNK_SIZE-1).rev() {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);
            let material = chunk.materials[idx];

            if material == Material::Air || material.is_solid() {
                continue;
            }

            let below_idx = Chunk::index(x, y + 1);
            let below_mat = chunk.materials[below_idx];

            if below_mat.is_solid() {
                continue;
            }

            // Sink if denser than what's below
            if material.density() > below_mat.density() {
                let velocity = (chunk.vel_x[idx].powi(2) + chunk.vel_y[idx].powi(2)).sqrt();
                let settling_threshold = material.density() * 0.01;

                // Only settle if not being carried by flow
                if velocity < settling_threshold {
                    // Swap
                    chunk.materials[idx] = below_mat;
                    chunk.materials[below_idx] = material;

                    // Swap mass
                    let mass_above = chunk.mass[idx];
                    let mass_below = chunk.mass[below_idx];
                    chunk.mass[idx] = mass_below;
                    chunk.mass[below_idx] = mass_above;
                }
            }
        }
    }
}
```

### Step 6: Spreading (No Friction Between Particles)

Particles can't form stable vertical stacks because they have no friction.

```rust
fn spread_particles(chunk: &mut Chunk) {
    for y in 0..CHUNK_SIZE-1 {
        for x in 0..CHUNK_SIZE {
            let idx = Chunk::index(x, y);
            let material = chunk.materials[idx];

            if material == Material::Air || material.is_solid() {
                continue;
            }

            let below_idx = Chunk::index(x, y + 1);
            let below_mat = chunk.materials[below_idx];

            // If sitting on another particle (not solid ground)
            if !below_mat.is_solid() && below_mat != Material::Air {
                // Try to slide diagonally down
                let dirs = if (x + y) % 2 == 0 { [(1, 1), (-1, 1)] } else { [(-1, 1), (1, 1)] };

                for (dx, dy) in dirs {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < CHUNK_SIZE as i32 && ny < CHUNK_SIZE as i32 {
                        let diag_idx = Chunk::index(nx as usize, ny as usize);
                        let diag_mat = chunk.materials[diag_idx];

                        if diag_mat == Material::Air ||
                           (!diag_mat.is_solid() && material.density() > diag_mat.density()) {
                            // Slide
                            chunk.materials[idx] = diag_mat;
                            chunk.materials[diag_idx] = material;

                            let m1 = chunk.mass[idx];
                            let m2 = chunk.mass[diag_idx];
                            chunk.mass[idx] = m2;
                            chunk.mass[diag_idx] = m1;
                            break;
                        }
                    }
                }
            }
        }
    }
}
```

## Complete Update Loop

```rust
pub fn update_physics(chunk: &mut Chunk, dt: f32) {
    // 1. Calculate mass distribution
    calculate_mass(chunk);

    // 2. Calculate pressure from mass above
    calculate_pressure(chunk);

    // 3. Calculate velocity from pressure gradients
    calculate_velocity(chunk, dt);

    // 4. Move all mass with velocity (particles + water)
    advect_mass(chunk, dt);

    // 5. Settle dense materials through light
    settle_particles(chunk);

    // 6. Spread unstable particles
    spread_particles(chunk);

    chunk.needs_render = true;
}
```

## Why This Works for Sluices

### Riffles Create Vortices

When water flows over a riffle (step in terrain):
1. **High pressure builds** upstream of riffle (water piles up)
2. **Low pressure zone** downstream (eddy behind riffle)
3. **Velocity drops** in the low-pressure zone
4. **Heavy particles settle** because velocity < settling_threshold

### Gold Settles, Soil Washes Out

- Gold (density 250) needs velocity > 2.5 to stay suspended
- Soil (density 30) needs velocity > 0.3 to stay suspended
- Behind riffles, velocity drops to ~0.5
- Gold settles, soil gets carried out

### Water Levels Naturally

- Pressure calculated from all mass above
- Water flows from high pressure to low pressure
- Result: water seeks level automatically

## What This Replaces

Delete or heavily modify:
- `water.rs` - Virtual Pipes approach (was water-only)
- `update.rs` - CA rules (particles had separate physics)
- `fluid.rs` - Navier-Stokes (separate velocity field)

Replace with:
- `physics.rs` - Unified pressure-velocity system
- One velocity field for everything
- One pressure field from all mass
- All materials respond to same physics

## Rendering Changes

```rust
fn render_cell(material: Material, mass: f32, vel: (f32, f32)) -> Color {
    let speed = (vel.0.powi(2) + vel.1.powi(2)).sqrt();

    match material {
        Material::Air if mass > 0.1 => {
            // Water (air cell with mass)
            let alpha = (mass * 150.0).min(200.0);
            let foam = (speed * 50.0).min(255.0) as u8;
            Color::rgba(30 + foam/2, 100, 200, alpha as u8)
        }
        Material::Gold => {
            // Gold - tint blue if underwater
            let base = [255, 215, 0, 255];
            if mass > 30.0 { tint_underwater(base) } else { base.into() }
        }
        // ... etc
    }
}
```

## Performance Considerations

1. **Pressure calculation**: O(width × height) - single pass
2. **Velocity update**: O(width × height) - neighbor lookups
3. **Advection**: O(width × height) - simple math
4. **Settling/Spreading**: O(width × height) - local operations

Total: O(n) per frame, same as current system.

For 256×256 at 60fps: ~4M operations/sec, easily achievable.

## Testing Criteria

1. **Water levels flat** within 1-2 seconds of being poured
2. **Gold sinks through water** and settles behind riffles
3. **Soil is carried by flow** unless it settles in slow zones
4. **No 1-pixel walls** - particles spread naturally
5. **Pressure pushes particles** - water doesn't seep through gold piles
6. **Riffles work** - gold accumulates, light material washes out

## References

- [Powder Toy Source](https://github.com/The-Powder-Toy/The-Powder-Toy/blob/master/src/simulation/Air.cpp)
- [Virtual Pipes Blog](https://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html)
- [Matthias Müller Height Field Paper](https://matthias-research.github.io/pages/publications/hfFluid.pdf)
- [Position Based Fluids Paper](https://mmacklin.com/pbf_sig_preprint.pdf)
- [Sluice Box CFD Research](https://www.researchgate.net/publication/383446350_Optimization_of_Sluice_Box_for_Small_Scale_Mining_Using_Computational_Fluid_Dynamics_CFD)

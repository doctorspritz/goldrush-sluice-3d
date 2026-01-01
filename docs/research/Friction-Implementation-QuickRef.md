# Friction Implementation Quick Reference

**TL;DR**: How to add boundary friction to your APIC fluid simulator in 3 implementation options.

---

## Option 1: Simple Tangential Damping (RECOMMENDED)

**Complexity**: ⭐ Easy
**Performance**: ⭐⭐⭐⭐⭐ Excellent (~2-3 FLOPs per boundary particle)
**Visual Quality**: ⭐⭐⭐⭐ Very Good

### Code Implementation

Add to `flip.rs` in the `advect_particles()` method around line 915:

```rust
// After SDF collision detection
if sdf_dist < cell_size * 0.5 {
    let grad = grid.sdf_gradient(particle.position);

    // Push particle out of solid
    let push_dist = cell_size * 0.5 - sdf_dist;
    particle.position += grad * push_dist;

    // ==== FRICTION IMPLEMENTATION ====

    // Decompose velocity into normal and tangential components
    let v_n = particle.velocity.dot(grad);  // Normal component (scalar)
    let v_normal = grad * v_n;              // Normal velocity vector
    let v_tangent = particle.velocity - v_normal;  // Tangential velocity

    // Apply friction to tangential component
    let friction_coeff = 0.3;  // Range: 0.0 (frictionless) to 1.0 (maximum friction)
    let v_tangent_damped = v_tangent * (1.0 - friction_coeff);

    // Cancel inward normal velocity (prevent penetration)
    let v_normal_clamped = if v_n < 0.0 {
        Vec2::ZERO
    } else {
        v_normal
    };

    // Reconstruct velocity
    particle.velocity = v_tangent_damped + v_normal_clamped;
}
```

### Friction Coefficients by Surface Type

```rust
const FRICTION_ICE: f32 = 0.05;        // Nearly frictionless
const FRICTION_SMOOTH_METAL: f32 = 0.2; // Low friction
const FRICTION_ROCK: f32 = 0.3;        // Moderate (default)
const FRICTION_WOOD: f32 = 0.5;        // Higher friction
const FRICTION_RUBBER: f32 = 0.8;      // Very high friction
```

---

## Option 2: Coulomb Friction (Physically Accurate)

**Complexity**: ⭐⭐⭐ Moderate
**Performance**: ⭐⭐⭐⭐ Good (~10-15 FLOPs per boundary particle)
**Visual Quality**: ⭐⭐⭐⭐⭐ Excellent (physics-based)

### Code Implementation

Add this helper function to `flip.rs`:

```rust
/// Apply Coulomb friction at a boundary
///
/// Arguments:
/// - particle_vel: Current particle velocity
/// - boundary_normal: Normal vector pointing away from solid (from SDF gradient)
/// - boundary_vel: Velocity of the boundary (use Vec2::ZERO for static walls)
/// - friction_coeff: Friction coefficient μ (typically 0.1 to 0.8)
///
/// Returns: New particle velocity after friction
fn apply_coulomb_friction(
    particle_vel: Vec2,
    boundary_normal: Vec2,
    boundary_vel: Vec2,
    friction_coeff: f32,
) -> Vec2 {
    // Compute relative velocity (particle relative to boundary)
    let v_rel = particle_vel - boundary_vel;

    // Decompose into normal and tangential components
    let v_n = v_rel.dot(boundary_normal);
    let v_n_vec = boundary_normal * v_n;
    let v_t = v_rel - v_n_vec;
    let v_t_mag = v_t.length();

    // Coulomb friction threshold: f_max = μ * |F_normal|
    // For velocity-based friction: threshold ∝ μ * |v_normal|
    let friction_threshold = friction_coeff * v_n.abs();

    // Apply friction to tangential velocity
    let v_t_new = if v_t_mag > friction_threshold {
        // Kinetic friction: reduce tangential velocity by friction force
        v_t * (1.0 - friction_threshold / v_t_mag)
    } else {
        // Static friction: tangential velocity below threshold, particle "sticks"
        Vec2::ZERO
    };

    // Prevent penetration: cancel inward normal velocity
    let v_n_new = if v_n < 0.0 {
        0.0
    } else {
        v_n
    };

    // Reconstruct absolute velocity
    boundary_vel + boundary_normal * v_n_new + v_t_new
}
```

### Usage in `advect_particles()`:

```rust
if sdf_dist < cell_size * 0.5 {
    let grad = grid.sdf_gradient(particle.position);
    let push_dist = cell_size * 0.5 - sdf_dist;
    particle.position += grad * push_dist;

    // Apply Coulomb friction
    particle.velocity = apply_coulomb_friction(
        particle.velocity,
        grad,
        Vec2::ZERO,  // Static boundary
        0.3,         // Friction coefficient
    );
}
```

---

## Option 3: Material-Specific Friction (Gameplay-Driven)

**Complexity**: ⭐⭐⭐⭐ Complex (requires terrain data)
**Performance**: ⭐⭐⭐ Moderate (+1 texture lookup per boundary particle)
**Visual Quality**: ⭐⭐⭐⭐⭐ Excellent (spatially varying)

### Part 1: Add Friction Map to Grid

In `grid.rs`:

```rust
pub struct Grid {
    // ... existing fields ...

    /// Per-cell friction coefficient (0.0 = frictionless, 1.0 = maximum)
    pub friction_map: Vec<f32>,
}

impl Grid {
    pub fn new(width: usize, height: usize, cell_size: f32) -> Self {
        let cell_count = width * height;
        Self {
            // ... existing fields ...
            friction_map: vec![0.3; cell_count],  // Default friction
        }
    }

    /// Sample friction coefficient at world position (bilinear interpolation)
    pub fn sample_friction(&self, pos: Vec2) -> f32 {
        let x = pos.x / self.cell_size;
        let y = pos.y / self.cell_size;

        let i = x.floor() as usize;
        let j = y.floor() as usize;

        // Clamp to grid bounds
        if i >= self.width - 1 || j >= self.height - 1 {
            return 0.3;  // Default friction at boundaries
        }

        let fx = x - i as f32;
        let fy = y - j as f32;

        // Bilinear interpolation
        let f00 = self.friction_map[self.cell_index(i, j)];
        let f10 = self.friction_map[self.cell_index(i + 1, j)];
        let f01 = self.friction_map[self.cell_index(i, j + 1)];
        let f11 = self.friction_map[self.cell_index(i + 1, j + 1)];

        let f0 = f00 * (1.0 - fx) + f10 * fx;
        let f1 = f01 * (1.0 - fx) + f11 * fx;

        f0 * (1.0 - fy) + f1 * fy
    }

    /// Set friction for a rectangular region
    pub fn set_friction_region(&mut self, x: usize, y: usize, w: usize, h: usize, friction: f32) {
        for j in y..y + h {
            for i in x..x + w {
                if i < self.width && j < self.height {
                    let idx = self.cell_index(i, j);
                    self.friction_map[idx] = friction;
                }
            }
        }
    }
}
```

### Part 2: Add Material Friction Properties

In `particle.rs`:

```rust
impl ParticleMaterial {
    /// Friction coefficient when particle contacts boundary
    /// Higher = more friction (particle slows down more at walls)
    pub fn boundary_friction(&self) -> f32 {
        match self {
            Self::Water => 0.0,       // Frictionless (fluid)
            Self::Mud => 0.4,         // Moderate friction
            Self::Sand => 0.3,        // Lower (rolls easily)
            Self::Magnetite => 0.35,  // Medium
            Self::Gold => 0.5,        // Higher (flaky, catches on roughness)
        }
    }
}
```

### Part 3: Use in Advection

In `flip.rs`:

```rust
fn advect_particles(&mut self, dt: f32) {
    // ... existing code ...

    let grid = &self.grid;

    self.particles.list.par_iter_mut().for_each(|particle| {
        // ... position update ...

        let sdf_dist = grid.sample_sdf(particle.position);

        if sdf_dist < cell_size * 0.5 {
            let grad = grid.sdf_gradient(particle.position);

            // Sample friction at this location
            let terrain_friction = grid.sample_friction(particle.position);

            // Combine terrain friction with material friction
            // Use max (most restrictive) or average based on desired behavior
            let material_friction = particle.material.boundary_friction();
            let effective_friction = terrain_friction.max(material_friction);

            // Apply friction (using Option 1 or Option 2 method)
            let v_n = particle.velocity.dot(grad);
            let v_normal = grad * v_n;
            let v_tangent = particle.velocity - v_normal;
            let v_tangent_damped = v_tangent * (1.0 - effective_friction);
            let v_normal_clamped = if v_n < 0.0 { Vec2::ZERO } else { v_normal };

            particle.velocity = v_tangent_damped + v_normal_clamped;

            // Push out of solid
            let push_dist = cell_size * 0.5 - sdf_dist;
            particle.position += grad * push_dist;
        }
    });
}
```

### Part 4: Configure Sluice Friction

In your sluice setup code:

```rust
// Example: Set high friction at riffles, low friction in chutes
sim.grid.set_friction_region(10, 50, 5, 10, 0.7);  // Riffle area (high friction)
sim.grid.set_friction_region(20, 50, 30, 5, 0.15); // Smooth chute (low friction)
sim.grid.set_friction_region(60, 45, 10, 20, 0.5); // Textured mat (medium)
```

---

## Comparison Table

| Feature | Option 1: Simple | Option 2: Coulomb | Option 3: Material-Specific |
|---------|------------------|-------------------|----------------------------|
| **Lines of code** | ~10 | ~30 | ~80 |
| **FLOPs per particle** | 2-3 | 10-15 | 15-20 |
| **Physics accuracy** | Good | Excellent | Excellent |
| **Visual quality** | Very Good | Excellent | Excellent |
| **Gameplay flexibility** | Low | Medium | High |
| **Implementation time** | 15 min | 1 hour | 3-4 hours |
| **Best for** | Quick prototype | Physics sim | Game mechanics |

---

## Testing Checklist

After implementing friction:

- [ ] **Visual test**: Particles sliding down inclined plane should slow down
- [ ] **No-friction test**: Set `friction_coeff = 0.0`, particles should slide freely
- [ ] **High-friction test**: Set `friction_coeff = 0.9`, particles should stick/stop
- [ ] **Energy conservation**: Total kinetic energy should decrease (friction dissipates energy)
- [ ] **No wall climbing**: Particles shouldn't gain upward velocity at vertical walls
- [ ] **Sluice riffles**: Gold particles should accumulate at high-friction riffles
- [ ] **Performance**: Profile to ensure <1ms additional overhead at 60 FPS

---

## Common Pitfalls

### Pitfall 1: Applying Friction Before Position Correction

❌ **Wrong**:
```rust
// Apply friction first
particle.velocity = apply_friction(particle.velocity, grad, 0.3);

// Then push out of solid
particle.position += grad * push_dist;
```

✅ **Correct**:
```rust
// Push out of solid first
particle.position += grad * push_dist;

// Then apply friction (particle is now at boundary)
particle.velocity = apply_friction(particle.velocity, grad, 0.3);
```

### Pitfall 2: Not Clamping Normal Velocity

❌ **Wrong**:
```rust
// Allows particles to gain velocity INTO solid
particle.velocity = v_tangent_damped + v_normal;
```

✅ **Correct**:
```rust
// Prevent inward normal velocity
let v_normal_clamped = if v_n < 0.0 { Vec2::ZERO } else { v_normal };
particle.velocity = v_tangent_damped + v_normal_clamped;
```

### Pitfall 3: Using Unnormalized Gradient

❌ **Wrong**:
```rust
let grad = grid.sdf_gradient(particle.position);  // May not be unit length
let v_n = particle.velocity.dot(grad);  // ← WRONG if grad not normalized
```

✅ **Correct** (if SDF gradient isn't guaranteed normalized):
```rust
let grad_raw = grid.sdf_gradient(particle.position);
let grad = grad_raw.normalize();  // Ensure unit length
let v_n = particle.velocity.dot(grad);  // ← Now correct
```

Check your `sdf_gradient()` implementation - if it returns normalized gradients, no need to normalize again.

---

## Performance Optimization

If friction becomes a bottleneck (unlikely):

### Optimization 1: Only Apply to Boundary Particles

```rust
// Skip friction for particles far from boundaries
if sdf_dist > cell_size * 2.0 {
    continue;  // Too far from any solid, no friction needed
}
```

### Optimization 2: Precompute Friction Map

For Option 3, avoid sampling friction texture every frame:

```rust
// In particle struct, cache last sampled friction
pub struct Particle {
    // ... existing fields ...
    cached_friction: f32,
}

// Update cached friction only when particle moves to new cell
if particle.moved_to_new_cell() {
    particle.cached_friction = grid.sample_friction(particle.position);
}
```

---

## Recommended Implementation Path

**Week 1**: Implement **Option 1** for immediate results
- Add simple tangential damping to `advect_particles()`
- Test with friction values 0.0, 0.3, 0.7
- Verify particles behave correctly at walls

**Week 2**: Profile and tune
- Measure performance impact (<1ms expected)
- Adjust friction coefficient for desired feel
- Test with 10k+ particles

**Week 3** (Optional): Upgrade to **Option 2** for physics accuracy
- Implement Coulomb friction function
- Compare behavior with Option 1
- Decide if extra complexity is worth it

**Week 4** (Optional): Add **Option 3** for gameplay
- Implement friction map in Grid
- Add material-specific friction
- Configure sluice riffles and chutes for gold panning

---

## Questions?

**Q: My particles are stuck at walls, what's wrong?**
A: Friction coefficient too high. Try 0.2-0.3 instead of 0.7+.

**Q: Particles still slide too much, feels unrealistic.**
A: Increase friction to 0.5-0.7. Also check if SDF gradient is normalized.

**Q: Performance dropped significantly.**
A: Friction should add <5% overhead. Profile to find the real bottleneck (likely elsewhere).

**Q: Should I apply friction to all particles or just sediment?**
A: Apply to ALL particles at boundaries. Water needs friction too for realistic behavior.

**Q: What about particle-particle friction?**
A: That's a separate feature (more complex). Start with boundary friction first.

---

**Last Updated**: 2025-12-21
**Related Docs**: `/docs/research/APIC-MPM-Friction-Research.md` (comprehensive version)

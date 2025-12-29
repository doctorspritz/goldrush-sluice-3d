# Multi-Sediment Types Implementation Plan

## Goal
Add support for different sediment types (quartz sand, black sand/magnetite) with distinct physical properties:
- Different densities → different settling velocities
- Different Shields parameters → different entrainment thresholds
- Mixed beds with material composition tracking

## Research Summary

### Material Properties

| Property | Quartz Sand | Black Sand (Magnetite) |
|----------|-------------|------------------------|
| Density (g/cm³) | 2.65 | 5.0 |
| Typical diameter | 0.3 px | 0.2 px (smaller grains) |
| Shape factor C₂ | 1.0 | 1.1 (angular) |
| Shields θ_cr | 0.045 | 0.06 (harder to move) |
| Color | tan | dark gray/black |

### Key Physics (from [sediment transport research](https://geo.libretexts.org/Bookshelves/Oceanography/Coastal_Dynamics_(Bosboom_and_Stive)/06:_Sediment_transport)):

1. **Settling velocity** - Ferguson-Church already handles density:
   - `w ∝ (ρ_p - ρ_f) × D²` in Stokes regime
   - Heavy minerals settle faster at same size
   - Magnetite grains in nature are often smaller because they settle at same velocity as larger quartz

2. **Critical entrainment** - Shields parameter varies by material:
   - Higher density → higher critical shear stress to move
   - θ_cr for magnetite ~30% higher than quartz ([source](https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/6.6/model-description/critical-thresholds-for-transport-and-erosion))

3. **Hiding/exposure effects** (for mixed beds):
   - Fine particles sheltered by coarse ones
   - Coarse particles more exposed to flow
   - Creates "[armoring](https://www.nature.com/articles/s41467-017-01681-3)" - heavy minerals concentrate at surface

4. **[Placer formation](https://en.wikipedia.org/wiki/Placer_deposit)**:
   - Heavy minerals drop out where velocity decreases
   - Creates natural stratification: heavy at bottom
   - Gold/magnetite concentrate in same locations

---

## Implementation Plan

### Phase 1: Extend ParticleMaterial enum

**File: `crates/sim/src/particle.rs`**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleMaterial {
    Water,
    Sand,       // Quartz sand (existing)
    BlackSand,  // Magnetite/heavy minerals
}

impl ParticleMaterial {
    pub fn density(&self) -> f32 {
        match self {
            Self::Water => 1.0,
            Self::Sand => 2.65,
            Self::BlackSand => 5.0,
        }
    }

    pub fn typical_diameter(&self) -> f32 {
        match self {
            Self::Water => 0.0,
            Self::Sand => 0.3,
            Self::BlackSand => 0.2,  // Smaller (hydraulic equivalence)
        }
    }

    pub fn shields_critical(&self) -> f32 {
        match self {
            Self::Water => 0.0,
            Self::Sand => 0.045,
            Self::BlackSand => 0.06,  // Harder to entrain
        }
    }

    pub fn shape_factor(&self) -> f32 {
        match self {
            Self::Water => 1.0,
            Self::Sand => 1.0,
            Self::BlackSand => 1.1,  // Slightly more angular
        }
    }

    pub fn color(&self) -> [u8; 4] {
        match self {
            Self::Water => [50, 140, 240, 100],
            Self::Sand => [194, 178, 128, 255],
            Self::BlackSand => [40, 40, 45, 255],  // Dark gray
        }
    }
}
```

**Work:**
- Add `BlackSand` variant
- Add all physical properties
- Add `Particle::black_sand()` constructor
- Update `Particles::count_by_material()` to include black sand

---

### Phase 2: Mixed-material deposited cells

**File: `crates/sim/src/grid.rs`**

Current: `deposited: Vec<bool>` - just marks if cell is sediment

New: Store material composition per cell.

```rust
/// Material composition of a deposited cell
#[derive(Clone, Copy, Debug, Default)]
pub struct DepositedCell {
    pub sand_fraction: f32,       // 0.0-1.0
    pub black_sand_fraction: f32, // 0.0-1.0
    // Fractions should sum to 1.0 when cell is deposited
}

impl DepositedCell {
    pub fn is_deposited(&self) -> bool {
        self.sand_fraction + self.black_sand_fraction > 0.0
    }

    /// Weighted average Shields parameter for entrainment
    pub fn effective_shields_critical(&self) -> f32 {
        let sand_shields = 0.045;
        let black_sand_shields = 0.06;
        let total = self.sand_fraction + self.black_sand_fraction;
        if total <= 0.0 { return sand_shields; }

        (self.sand_fraction * sand_shields + self.black_sand_fraction * black_sand_shields) / total
    }

    /// Weighted average density for visualization/physics
    pub fn effective_density(&self) -> f32 {
        let total = self.sand_fraction + self.black_sand_fraction;
        if total <= 0.0 { return 2.65; }

        (self.sand_fraction * 2.65 + self.black_sand_fraction * 5.0) / total
    }

    /// Blend color based on composition
    pub fn color(&self) -> [u8; 4] {
        let sand_color = [194u8, 178, 128, 255];
        let black_color = [40u8, 40, 45, 255];
        let total = self.sand_fraction + self.black_sand_fraction;
        if total <= 0.0 { return sand_color; }

        let t = self.black_sand_fraction / total;
        [
            lerp_u8(sand_color[0], black_color[0], t),
            lerp_u8(sand_color[1], black_color[1], t),
            lerp_u8(sand_color[2], black_color[2], t),
            255,
        ]
    }
}
```

**Grid changes:**
```rust
pub struct Grid {
    // Replace:  deposited: Vec<bool>
    // With:
    pub deposited: Vec<DepositedCell>,

    // Keep deposited_mass for accumulation, but track per-material:
    pub deposited_mass_sand: Vec<f32>,
    pub deposited_mass_black: Vec<f32>,
}
```

**Work:**
- Add `DepositedCell` struct
- Replace `Vec<bool>` with `Vec<DepositedCell>`
- Split `deposited_mass` into per-material accumulators
- Update `set_deposited()` to take material fractions
- Update `is_deposited()` to check struct
- Update `clear_deposited()` to reset struct

---

### Phase 3: Update deposition logic

**File: `crates/sim/src/flip.rs` (Step 8f)**

Current flow:
1. Count particles in each cell
2. If count >= 4, convert to solid

New flow:
1. Count particles by material in each cell
2. Track separate accumulators for sand and black_sand
3. When total mass >= threshold, convert to deposited with appropriate fractions

```rust
// In step_8f_deposition:
for particle in settled_particles {
    let (i, j) = particle_to_cell(particle);
    match particle.material {
        ParticleMaterial::Sand => grid.deposited_mass_sand[idx] += 1.0,
        ParticleMaterial::BlackSand => grid.deposited_mass_black[idx] += 1.0,
        _ => {}
    }
}

// Convert when total mass reaches threshold
let total_mass = grid.deposited_mass_sand[idx] + grid.deposited_mass_black[idx];
if total_mass >= DEPOSITION_THRESHOLD {
    let sand_frac = grid.deposited_mass_sand[idx] / total_mass;
    let black_frac = grid.deposited_mass_black[idx] / total_mass;
    grid.set_deposited_with_composition(i, j, sand_frac, black_frac);
}
```

---

### Phase 4: Update entrainment logic

**File: `crates/sim/src/flip.rs` (Step 8g)**

Current: Uses fixed `CRITICAL_VELOCITY` for all deposited cells

New: Use cell's effective Shields parameter

```rust
// In step_8g_entrainment:
let cell_shields = grid.deposited[idx].effective_shields_critical();

// Higher Shields = higher critical velocity needed
let critical_velocity = BASE_CRITICAL_VELOCITY * (cell_shields / 0.045);

// When eroding, spawn particles proportional to composition
let composition = grid.deposited[idx];
let num_sand = (PARTICLES_PER_CELL as f32 * composition.sand_fraction).round() as usize;
let num_black = (PARTICLES_PER_CELL as f32 * composition.black_sand_fraction).round() as usize;

for _ in 0..num_sand {
    spawn_particle(ParticleMaterial::Sand, ...);
}
for _ in 0..num_black {
    spawn_particle(ParticleMaterial::BlackSand, ...);
}
```

---

### Phase 5: Hiding/exposure effects (optional, for realism)

When multiple sediment types are present, finer particles are protected:

```rust
/// Hiding correction factor (Wu et al. 2000)
/// Reduces effective Shields for small particles hidden by larger ones
fn hiding_exposure_factor(particle_diameter: f32, bed_median_diameter: f32) -> f32 {
    // Smaller particles are hidden: factor > 1 (harder to move)
    // Larger particles are exposed: factor < 1 (easier to move)
    let ratio = particle_diameter / bed_median_diameter;
    ratio.powf(-0.6)  // Empirical exponent from literature
}
```

This creates armoring behavior where:
- Light sand erodes first from mixed beds
- Heavy black sand concentrates at surface
- Eventual equilibrium with black sand "pavement"

---

### Phase 6: Rendering updates

**File: `crates/game/src/render.rs`**

1. Particle rendering already uses `particle.material.color()` - just add black sand color
2. Deposited cell rendering needs to use `DepositedCell::color()` for blended appearance

---

### Phase 7: Input/spawning

Add controls to spawn black sand:
- Separate spawn function or ratio parameter
- Could have "ore vein" areas that spawn mixed material

---

## Testing Strategy

1. **Unit tests** (particle.rs):
   - Verify black sand settles faster than quartz at same size
   - Verify shields_critical is higher for black sand
   - Verify density ordering: black_sand > sand > water

2. **Integration tests**:
   - Mixed material deposition creates correct composition
   - Entrainment respects composition-weighted Shields
   - Particle spawning from entrained cell matches composition

3. **Visual tests**:
   - Black sand visibly settles faster
   - Black sand concentrates at bottom of deposits
   - Harder to wash away deposited black sand

---

## Expected Behavior

1. **Settling**: Black sand drops out of flow sooner (velocity ∝ density)
2. **Stratification**: Natural layering with heavy minerals at bottom
3. **Entrainment resistance**: Black sand beds require stronger flow to erode
4. **Placer formation**: Black sand accumulates where velocity decreases (like behind riffles, in eddies)

---

## Open Questions

1. **Mass balance**: Should 4 black sand particles = 1 cell, or weight by density?
   - Option A: Same count threshold (4 particles = 1 cell regardless of type)
   - Option B: Weight by density (fewer heavy particles needed)
   - Recommend: Option A for simplicity, Option B for realism

2. **Vertical stratification within cells**: Track layers or just fractions?
   - Simple: Just fractions (what we propose)
   - Complex: Track depth order (heavy at bottom, light at top)
   - Recommend: Start simple, add layers if needed

3. **Armoring dynamics**: Implement hiding/exposure or use simple mixing?
   - Simple: Weighted average Shields
   - Complex: Hiding/exposure with surface layer tracking
   - Recommend: Start with weighted average

---

## Implementation Order

1. [ ] Add `BlackSand` to `ParticleMaterial` with all properties
2. [ ] Add `Particle::black_sand()` constructor and spawning methods
3. [ ] Create `DepositedCell` struct with composition tracking
4. [ ] Update `Grid` to use `Vec<DepositedCell>`
5. [ ] Update deposition to track per-material mass
6. [ ] Update entrainment to use composition-weighted Shields
7. [ ] Update entrainment to spawn correct material mix
8. [ ] Add rendering for black sand particles
9. [ ] Add deposited cell color blending
10. [ ] Add spawn controls for mixed material
11. [ ] Write tests
12. [ ] Visual verification

---

## References

- [Critical shear stress and settling velocity](https://geo.libretexts.org/Bookshelves/Oceanography/Coastal_Dynamics_(Bosboom_and_Stive)/06:_Sediment_transport/6.08:_Some_aspects_of_(very)_fine_sediment_transport/6.8.2:_Critical_shear_stress_and_settling_velocity)
- [HEC-RAS Hiding and Exposure Corrections](https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/latest/model-description/hiding-and-exposure-corrections)
- [Placer deposits - Wikipedia](https://en.wikipedia.org/wiki/Placer_deposit)
- [Magnetite properties](https://sandatlas.org/magnetite/)
- [River-bed armoring as granular segregation](https://www.nature.com/articles/s41467-017-01681-3)

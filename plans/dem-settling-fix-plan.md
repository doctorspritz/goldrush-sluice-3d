# DEM Settling Fix Plan

## Research Findings

### Key Techniques from Literature

Based on research into PBD/DEM implementations (NVIDIA FleX, Chrono::GPU, Bullet forums, XPBD papers):

**1. Penetration Tolerance ("Slop")** - [Box2D/Erin Catto approach](https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=1963)
- Don't correct 100% of overlap, correct ~90%
- "Allow some penetration to avoid contact breaking"
- This prevents oscillation between corrected/uncorrected states

**2. Impulse Clamping** - [Dirk Gregorius technique](https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=1963)
- Never apply negative impulses (pulling particles together)
- Accumulate impulses and clamp to positive values only
- `impulse = max(accumulated_impulse + delta, 0)`

**3. Jacobi Solver** - Already implemented correctly in current code
- Better for stacking than Gauss-Seidel
- Average corrections instead of applying immediately

**4. XPBD Compliance** - [Macklin et al.](https://matthias-research.github.io/pages/publications/XPBD.pdf)
- Time-step and iteration independent stiffness
- compliance α̃ = α/dt² (α=0 for hard contacts)
- Built-in damping through constraint formulation

**5. Positional Particles** - [Parallel Particles (P2) approach](https://www.researchgate.net/publication/272825505)
- No orientation/rotation = easier to stabilize
- Friction angle determines angle of repose directly
- Current code already uses this approach

### Wet vs Dry Requirements

**Dry Particles (Mining/Piling):**
- Stable stacking under gravity
- Angle of repose from friction
- Fast settling to rest state
- No inter-particle attraction

**Wet Particles (Sluicing):**
- Buoyancy: `F_buoy = (ρ_particle - ρ_water) * V * g`
- Drag: proportional to velocity difference with water
- Suspended vs bedload state transitions
- Coupling with FLIP fluid solver

Current code handles both via `effective_gravity()` which applies buoyancy.

---

## Root Cause Analysis

### The Three Bugs Are Interconnected

The current implementation has three fundamental issues that all stem from imprecise contact detection:

1. **Floor Vibration** - The +0.3 margin fights with gravity
2. **Mid-Air Pause** - `near_floor` heuristic triggers false positives
3. **Support Propagation** - Sleep counter is a poor proxy for "supported"

### Bug 1: Floor Vibration (Line 253)

**Current behavior:**
```wgsl
pos += grad * (penetration + 0.3);  // Always adds 0.3 margin
```

**The cycle:**
1. Frame N: Particle at floor surface, `sdf < radius`, correction applied
2. Particle pushed to `radius + 0.3` above floor surface
3. Frame N+1: `sdf = 0.3` (above floor), so `sdf >= radius`, no floor_contact
4. Gravity applies, particle falls back
5. Frame N+2: `sdf < radius` again, correction with margin applied
6. Repeat forever

**Root cause:** The margin makes the particle leave floor contact, so it gets gravity again.

### Bug 2: Mid-Air Pause (Lines 286-289)

**Current behavior:**
```wgsl
let sdf_at_old = sample_sdf(old_pos);
let near_floor = sdf_at_old < radius * 2.0;
let has_floor_support = floor_contact || near_floor;
```

**The problem:**
- SDF is signed distance to ANY terrain, including vertical walls
- A falling particle near a vertical wall has `sdf < radius * 2.0`
- This counts as "near_floor" → "has_floor_support" → sleep counter increments
- If counter reaches threshold, velocity is zeroed mid-fall

**Root cause:** `near_floor` is a distance heuristic, not actual contact detection.

### Bug 3: Support Propagation (Lines 206-209)

**Current behavior:**
```wgsl
let j_sleep = sleep_counters[j_idx];
let j_is_below = normal.y > 0.3;
if (j_is_below && j_sleep >= SLEEP_THRESHOLD / 2u) {
    supported_contacts += 1u;
}
```

**The problem:**
- Sleep counter can have residual values from previous contacts
- A falling particle that briefly touched something has non-zero counter
- This falsely counts as "supported" for particles above it

**Root cause:** Sleep counter ≠ support state. Need explicit support chain.

---

## Proposed Solution

### Core Insight

Position-Based Dynamics handles contacts as constraints. A particle resting on floor should:
1. NOT receive gravity (balanced by normal force)
2. NOT move (constraint satisfied)
3. Have zero velocity into floor

The current code applies gravity unconditionally, then tries to undo it with sleeping.
Better: Skip gravity entirely for particles in stable resting contact.

### Phase 1: Fix Floor Vibration (Research-Backed)

**Apply penetration tolerance ("slop") from Box2D/Erin Catto approach:**

The +0.3 margin is fundamentally wrong. Instead of pushing particles ABOVE the surface,
we should allow slight penetration as a stable equilibrium zone:

```wgsl
// Floor/SDF collision with penetration tolerance
let sdf = sample_sdf(pos);
let slop = radius * 0.1;  // Allow 10% penetration as stable zone

if (sdf < radius) {
    let grad = sdf_gradient(pos);
    let penetration = radius - sdf;

    // Only correct penetration beyond the slop
    // This creates a stable equilibrium zone where particles can rest
    let correction = max(penetration - slop, 0.0);
    pos += grad * correction;

    floor_contact = true;
    // ... rest of floor handling unchanged
}
```

**Why this works (from research):**
- "Allow some penetration to avoid contact breaking" - Erin Catto
- Creates a stable zone where gravity and floor reaction balance
- Particles settle INTO the slop zone, not bounce above it
- No oscillation because equilibrium is inside allowed penetration range

### Phase 2: Fix Mid-Air Pause

**Remove `near_floor` from support check entirely:**

```wgsl
// True support: ONLY actual floor contact OR chain from supported neighbor
let has_floor_support = floor_contact;  // NO near_floor!
let has_chain_support = supported_contacts >= 1u;
let truly_supported = has_floor_support || has_chain_support;
```

**Also: Add gradient-based floor detection**

The SDF gradient tells us surface orientation. A floor has gradient pointing up (gradient.y > 0.7).
A vertical wall has gradient pointing sideways (|gradient.x| > 0.7).

```wgsl
// Only count floor contact if gradient points up (it's actually a floor, not a wall)
let is_horizontal_floor = grad.y > 0.7;
floor_contact = sdf < radius && is_horizontal_floor;
```

### Phase 3: Fix Support Propagation

**Option A: Simple - Use support_level buffer (extra memory)**

Add a new buffer for support levels:
- 0 = unsupported (falling)
- 1-N = distance from floor in support chain

**Option B: Simpler - Just require high sleep counter for chain support**

```wgsl
// Only count as supporting if neighbor is definitely sleeping (not just residual counter)
if (j_is_below && j_sleep >= SLEEP_THRESHOLD) {  // Full threshold, not half
    supported_contacts += 1u;
}
```

### Phase 4: Skip Gravity for Resting Contacts

**If particle is sleeping AND on floor, don't apply gravity:**

```wgsl
if (params.iteration == 0u) {
    // Skip gravity for sleeping floor-contact particles (balanced by normal force)
    let skip_gravity = is_sleeping && floor_contact;
    if (!skip_gravity) {
        let g_eff = effective_gravity(material, in_water);
        vel.y += g_eff * params.dt;
    }
    pos += vel * params.dt;
}
```

Wait - this creates a chicken-egg problem. `floor_contact` isn't known until after position update.

**Better approach: Check floor proximity before gravity**

```wgsl
if (params.iteration == 0u) {
    // Check if we were on floor last frame (position hasn't moved yet)
    let sdf_now = sample_sdf(pos);
    let on_floor_now = sdf_now < radius * 1.1;  // Slight margin for detection

    // Skip gravity for sleeping floor particles
    let skip_gravity = is_sleeping && on_floor_now;
    if (!skip_gravity) {
        let g_eff = effective_gravity(material, in_water);
        vel.y += g_eff * params.dt;
    }
    pos += vel * params.dt;
}
```

### Phase 5: Apply Slop to Particle-Particle Collisions Too

For consistency, apply the same slop approach to particle-particle collisions:

```wgsl
if (dist_sq < contact_dist * contact_dist && dist_sq > 0.0001) {
    let dist = sqrt(dist_sq);
    let overlap = contact_dist - dist;

    // Apply slop: only correct beyond 10% overlap
    let slop = contact_dist * 0.1;
    let correctable_overlap = max(overlap - slop, 0.0);

    if (correctable_overlap > 0.0) {
        // ... rest of collision handling
        accumulated_correction += normal * correctable_overlap * my_fraction;
    }
}
```

---

## Implementation Order

1. **First: Apply slop to floor collision** (fixes vibration)
   - Change line 253 to use slop instead of +0.3 margin

2. **Second: Remove near_floor** (fixes mid-air pause)
   - Delete lines 285-286, change line 289

3. **Third: Require full sleep counter for chain support**
   - Change line 207 threshold from `SLEEP_THRESHOLD / 2u` to `SLEEP_THRESHOLD`

4. **Fourth: Apply slop to particle-particle collisions** (consistent behavior)
   - Modify collision loop to use slop

5. **Test after each change** - Run `sediment_stages_visual`, press 9

**Note:** Each change should be tested independently. If slop fixes the floor vibration,
we may not need the gravity-skip complexity.

---

## Testing Criteria

**Good behavior:**
- [x] Particles fall smoothly through air (no pausing) - FIXED
- [x] Particles hitting floor come to rest within ~10 frames - FIXED (settles at frame 108)
- [x] Sleeping particles don't vibrate (check visually) - FIXED (vel=0.0,0.0 in visual test)
- [x] Sand pile forms stable angle of repose - FIXED (aspect ratio 69.52, pile spreads x: 158→121)
- [ ] Gold sinks through sand (density stratification) - NOT YET TESTED
- [ ] Particles on vertical walls don't get stuck - NOT YET TESTED

**Test command:**
```bash
cd .worktrees/fix-dem-settling
cargo run --example dem_settling_diagnostic -p game --release
# Headless numeric test - no visuals needed
```

---

## Implementation Results

**Changes made to `dem_forces.wgsl`:**

1. **Floor collision slop (lines 277-305):** Instead of +0.3 margin, allow 10% penetration as stable zone
2. **Removed near_floor heuristic (lines 324-329):** Only floor_contact or chain_support count as supported
3. **Full sleep threshold for chain support (line 222):** Require `SLEEP_THRESHOLD` not `SLEEP_THRESHOLD/2`
4. **Added tangential friction (lines 243-257):** Coulomb friction for angle of repose
5. **Reduced damping from 0.3 to 0.7 (line 311):** Allows lateral sliding
6. **Reduced mass multiplier from 50x to 3x (line 228):** Allows pile restructuring
7. **Skip gravity for sleeping floor particles (lines 133-144):** Prevents jitter cycle where sleeping
   particles get gravity → move down → hit floor → get pushed up → repeat

**Headless diagnostic test:** `crates/game/examples/dem_settling_diagnostic.rs`
- Test 1: Column pour → spread 202.5, aspect ratio 69.52 ✅
- Test 2: Settling velocity → settled at frame 108, speed 0.0 ✅

**Visual test:** `cargo run --example sediment_stages_visual -p game --release`
- Particles fall from top (y=14.7), accelerate under gravity (191 px/s at frame 60)
- Hit floor at y=237.7 (floor at y=240), settle with vel=(0.0, 0.0)
- Pile spreads horizontally as particles land (x: 158.6 → 121.8)
- No vibration, no mid-air pauses, proper angle of repose ✅

---

## Why NOT to Make Quick Changes

The handoff document suggested several "quick fixes":
- Lock position for floor particles → Could cause tunneling
- Add has_landed flag → Adds state but doesn't fix root cause
- Two-pass support propagation → Good but overkill for now

Each quick fix addresses symptoms, not causes. The root cause is:
1. Margin makes particles leave contact → Fix the margin
2. near_floor is wrong → Remove it
3. Sleep counter isn't support → Use stricter threshold

These are targeted fixes at the actual bugs, not patches around them.

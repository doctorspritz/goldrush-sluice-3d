# DEM Two-State Implementation Progress

**Master Plan:** `/Users/simonheikkila/.claude/plans/dynamic-crunching-cloud.md`

**Last Updated:** 2026-01-03

---

## Level 0: Clean Slate
**Status:** COMPLETE

### Entry Gate
- [x] In `.worktrees/fix-dem-settling` directory
- [x] Uncommitted changes exist (stashed)

### Actions Taken
- Ran `git stash push -m "Broken uncommitted changes - complex wake detection approach"`
- Verified clean git status
- Confirmed build passes
- Confirmed existing tests pass (4/4)

### Exit Gate
- [x] `git status` shows no uncommitted changes
- [x] `cargo build --release` succeeds
- [x] `cargo test -p game --release --test gpu_dem_settling` passes (4/4)

---

## Level 1: Static State Buffer
**Status:** COMPLETE (commit 80e6a46)

### Entry Gate
- [x] Level 0 complete
- [x] No uncommitted changes

### Implementation
Add to `crates/game/src/gpu/dem.rs`:

1. Add `static_state` buffer to `GpuDemSolver` struct:
```rust
static_state: wgpu::Buffer,  // u32 per particle: 0 = dynamic, 1 = static
```

2. Create buffer in `new()`:
```rust
let static_state = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("DEM Static State"),
    size: (max_particles * 4) as u64,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

3. Initialize all particles as DYNAMIC (0)
4. Add to bind group at binding 10 (after sleep_counters at 8, solid_flags at 9)

### Exit Gate
- [x] `cargo build --release` succeeds
- [x] Existing tests still pass (no regression)
- [x] New buffer exists and is bound to shader

### Commit Message
```
feat(dem): add static state buffer for two-state particle model

Particles can now be STATIC (frozen, part of settled pile) or
DYNAMIC (normal physics). This is the foundation for proper
granular settling where settled piles act as solids.
```

---

## Level 2: Static Particle Freeze
**Status:** PENDING

### Entry Gate
- [ ] Level 1 complete
- [ ] Static buffer exists and compiles

### Implementation
In `crates/game/src/gpu/shaders/dem_forces.wgsl`, at TOP of main function:

```wgsl
// Read static state
let is_static = static_states[idx] == 1u;

// STATIC particles are FROZEN - skip ALL physics
if (is_static) {
    // Don't modify position or velocity
    // Just write back unchanged values
    positions[idx] = old_pos;
    velocities[idx] = vel;  // Already zero for static
    return;
}
```

### Exit Gate
- [ ] `cargo build --release` succeeds
- [ ] Existing tests still pass
- [ ] NEW TEST: `test_static_particles_frozen`
  - Create 10 particles, mark 5 as static
  - Run 100 frames
  - Static particles have ZERO position change
  - Dynamic particles fall normally

### Commit Message
```
feat(dem): static particles skip physics entirely

Static particles now return immediately without any position
or velocity updates. This is the core of the two-state model:
static = frozen solid, dynamic = normal physics.
```

---

## Level 3: Dynamic → Static Transition
**Status:** PENDING

### Entry Gate
- [ ] Level 2 complete
- [ ] Static freeze works (test passes)

### Implementation
At END of `dem_forces.wgsl`, before writing back:

```wgsl
// Transition check: slow + supported → become static
let speed_sq = dot(vel, vel);
let STATIC_THRESHOLD_SQ: f32 = 1.0;  // 1 px/s
let STATIC_DELAY: u32 = 30u;  // 30 frames = 0.5s at 60fps

if (!is_static && has_support && speed_sq < STATIC_THRESHOLD_SQ) {
    static_frames[idx] += 1u;
    if (static_frames[idx] >= STATIC_DELAY) {
        static_states[idx] = 1u;  // Become static
        vel = vec2(0.0);  // Zero velocity
    }
} else if (!is_static) {
    static_frames[idx] = 0u;  // Reset counter
}
```

**Note:** Need `static_frames` buffer (u32 per particle) - add in Level 1 if not present.

### Exit Gate
- [ ] `cargo build --release` succeeds
- [ ] Level 2 test still passes (no regression)
- [ ] NEW TEST: `test_settling_becomes_static`
  - Drop particles onto floor
  - After settling, particles have `is_static = 1`
  - Static particles have zero velocity

### Commit Message
```
feat(dem): dynamic particles transition to static when settled

Particles that are slow and supported for 30 frames transition
to static state. This makes settled piles truly frozen.
```

---

## Level 4: Force-Threshold Wake
**Status:** PENDING

### Entry Gate
- [ ] Level 3 complete
- [ ] Settling → static works

### Implementation
In collision detection loop, for static particles:

```wgsl
// Static particles check if force exceeds threshold
if (is_static) {
    let net_force = compute_net_contact_force();
    let weight = mass * params.gravity;
    let wake_threshold = MU_STATIC * weight;

    if (length(net_force) > wake_threshold) {
        static_states[idx] = 0u;  // Wake up
        // Continue with normal physics this frame
    } else {
        // Stay frozen
        positions[idx] = old_pos;
        velocities[idx] = vec2(0.0);
        return;
    }
}
```

### Exit Gate
- [ ] All previous tests pass
- [ ] NEW TEST: `test_gold_on_sand_stays_on_top`
  - Create settled sand pile (all static)
  - Drop gold particle on top
  - Gold sits on surface, doesn't penetrate
  - Sand particles remain static
- [ ] NEW TEST: `test_bulldozer_wakes_pile`
  - Create settled sand pile
  - Push large fast-moving object into pile
  - Sand particles near impact wake up
  - Pile restructures and resettles

### Commit Message
```
feat(dem): static particles wake only when force exceeds threshold

Static particles now resist penetration. Only forces exceeding
μ × weight can wake them. This prevents gold from sinking
through settled sand - the core behavior we need.
```

---

## Level 5: Proper Coulomb Friction
**Status:** PENDING

### Entry Gate
- [ ] Level 4 complete
- [ ] Gold stays on sand

### Implementation
Remove surface roughness hack, implement proper friction:

```wgsl
// DELETE this:
// let roughness = hash_noise(seed) * 0.15;

// ADD proper Coulomb friction:
let tangent = vec2(-normal.y, normal.x);
let v_rel = vel - vel_j;
let v_t = dot(v_rel, tangent);
let v_n = dot(v_rel, normal);

// Static friction: resist motion up to μ × normal force
let normal_force = overlap * CONTACT_STIFFNESS;
let max_friction = MU_STATIC * normal_force;
let friction = min(abs(v_t) * FRICTION_DAMPING, max_friction);
accumulated_vel_correction -= tangent * sign(v_t) * friction;
```

### Exit Gate
- [ ] All previous tests pass
- [ ] NEW TEST: `test_angle_of_repose`
  - Pour sand from height
  - Measure final pile slope
  - Slope should be arctan(μ) ± 5°
  - For μ=0.6, expect 28-34°

### Commit Message
```
feat(dem): implement proper Coulomb friction for angle of repose

Removed random surface roughness hack. Angle of repose now
comes from friction coefficient as physics dictates:
tan(θ) = μ. Sand with μ=0.6 forms ~31° slopes.
```

---

## Level 6: Support-Based Wake Propagation
**Status:** PENDING

### Entry Gate
- [ ] Level 5 complete
- [ ] Angle of repose works

### Implementation
```wgsl
// When a static particle wakes, set a "wake neighbors" flag
if (just_woke) {
    wake_flags[idx] = 1u;
}

// In a second pass (or next frame), check wake_flags of neighbors below
for each neighbor j below me {
    if (wake_flags[j] == 1u && static_states[idx] == 1u) {
        static_states[idx] = 0u;  // I wake too
    }
}
```

### Exit Gate
- [ ] All previous tests pass
- [ ] NEW TEST: `test_support_removal_cascade`
  - Create settled pile on a platform
  - Remove platform (set SDF to far)
  - All particles wake and fall
  - Particles resettle on new floor

### Commit Message
```
feat(dem): implement support-based wake propagation

When a supporting particle wakes, particles above it also wake.
This creates realistic pile collapse when support is removed.
```

---

## DOOM LOOP DETECTION

Track fix attempts here. If any section reaches 3 attempts, STOP and ask user.

### Level 1 Attempts
- Attempt 1: (pending)

### Level 2 Attempts
- Attempt 1: (pending)

### Level 3 Attempts
- Attempt 1: (pending)

### Level 4 Attempts
- Attempt 1: (pending)

### Level 5 Attempts
- Attempt 1: (pending)

### Level 6 Attempts
- Attempt 1: (pending)

---

## Quick Reference: Test Commands

```bash
# All settling tests
cargo test -p game --release --test gpu_dem_settling

# Specific test
cargo test -p game --release --test gpu_dem_settling test_static_particles_frozen

# Visual verification
cargo run --example sediment_stages_visual -p game --release

# Headless diagnostic
cargo run --example dem_settling_diagnostic -p game --release
```

## Quick Reference: Files to Modify

| Level | Files |
|-------|-------|
| 1 | `dem.rs` (add buffer) |
| 2 | `dem_forces.wgsl` (static freeze) |
| 3 | `dem_forces.wgsl` (transition logic), maybe `dem.rs` (static_frames buffer) |
| 4 | `dem_forces.wgsl` (force threshold), `gpu_dem_settling.rs` (new tests) |
| 5 | `dem_forces.wgsl` (Coulomb friction), `gpu_dem_settling.rs` (new tests) |
| 6 | `dem_forces.wgsl` (wake propagation), maybe new shader pass, `gpu_dem_settling.rs` (new tests) |

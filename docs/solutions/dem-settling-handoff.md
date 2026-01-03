# GPU DEM Settling - Handoff for Next Agent

## Branch
`fix/dem-settling` in worktree `.worktrees/fix-dem-settling`

## What Was Done
See `docs/solutions/dem-settling-progress.md` for complete details.

**Working:**
1. Jacobi constraint averaging - fixes horizontal creep
2. Mass-weighted separation (50x for stationary) - gold respects sand pile
3. Reduced collision radius (0.35) - tighter visual packing
4. Sleep counter system with support propagation from floor

## What's Still Broken

### 1. Floor Vibration (Primary Issue)
Particles touching the floor vibrate perpetually instead of settling.

**Root cause analysis:**
- Floor collision pushes particles up with +0.3 margin (line 253 in dem_forces.wgsl)
- Gravity pulls them back down
- Creates micro-bouncing cycle

**Suggested fixes to try:**
- Lock POSITION (not just velocity) for floor-contact particles with high sleep counter
- Add "resting contact" state that skips gravity for floor particles
- Use softer floor collision with no margin, or margin only on initial contact

### 2. Mid-Air Falling Behavior
Particles fall strangely - sometimes stop mid-air briefly.

**Root cause analysis:**
- Sleep counter logic may affect falling particles incorrectly
- The `truly_supported` check includes `near_floor` which is too aggressive
- Counter decay (divide by 2) might still leave some counter from previous contacts

**Suggested fixes to try:**
- Only apply sleep logic AFTER particle has touched something once (add `has_landed` flag)
- Remove `near_floor` from support check, require actual floor_contact
- Separate "falling" state from "settled" state explicitly

### 3. Support Propagation Reliability
Current implementation checks if neighbor below has high sleep counter, but:
- High sleep counter doesn't guarantee "supported" (could be a falling particle with residual counter)
- Propagation is only one level deep per frame

**Suggested fixes to try:**
- Use separate `support_level` buffer (0 = unsupported, floor = max, chain = decays upward)
- Two-pass approach: first pass marks floor contacts, second propagates support
- Only floor-contact particles can reach max sleep counter

## Key Files
- `crates/game/src/gpu/shaders/dem_forces.wgsl` - Main shader, lines 97-320
- `crates/game/src/gpu/dem.rs` - Solver setup, line 869 (radius), line 67 (sleep buffer)
- `crates/game/examples/sediment_stages_visual.rs` - Test app (press 9 for sand_then_gold)

## Testing
```bash
cd .worktrees/fix-dem-settling
cargo run --example sediment_stages_visual -p game --release
# Press 9 for sand_then_gold stage
# Press SPACE to pause, R to reset
```

## Current Constants (dem_forces.wgsl)
```wgsl
const SLEEP_THRESHOLD: u32 = 10u;   // Frames before sleeping
const JITTER_SPEED_SQ: f32 = 4.0;   // ~2 px/s threshold
const WAKE_SPEED_SQ: f32 = 200.0;   // ~14 px/s wake threshold
```

## Physics Context
- Position-Based Dynamics (PBD) approach, not spring-damper
- Particles interact via overlap correction, not forces
- Sleep system is necessary because velocity never truly reaches zero in discrete simulation
- Floor is defined by SDF (signed distance field), sampled at particle position

# Failure: Re-enabling APIC Did Not Fix Velocity Damping

**Date:** 2025-12-27
**Status:** FAILED - Water still flows like honey
**Category:** logic-errors, physics, process-failure

## What Was Tried

Re-enabled APIC affine velocity term in P2G transfer:

### Changes Made (flip.rs)

**Lines 308-316 (U component):**
```rust
// BEFORE:
// DISABLED APIC: affine term may break momentum conservation
// let affine_vel = c_mat * offset;
let _ = c_mat; // suppress warning
self.u_sum[idx] += vel.x * scaled_w;

// AFTER:
let affine_vel = c_mat * offset;
self.u_sum[idx] += (vel.x + affine_vel.x) * scaled_w;
```

**Lines 351-359 (V component):**
```rust
// BEFORE:
// DISABLED APIC: affine term may break momentum conservation
// let affine_vel = c_mat * offset;
let _ = offset; // suppress warning
self.v_sum[idx] += vel.y * scaled_w;

// AFTER:
let affine_vel = c_mat * offset;
self.v_sum[idx] += (vel.y + affine_vel.y) * scaled_w;
```

## Result

**STILL ~2% momentum loss per frame:**

```
MOMENTUM: particles=10115 → P2G_grid=8067 → +gravity=582471 → +pressure=224304 → G2P_particles=9960 → advect=9935
```

- Start: 10115
- End: 9935
- Loss: ~1.8% per frame
- Still compounds to ~65-70% loss per second

## Why It Failed

APIC was NOT the root cause. The hypothesis was wrong.

The momentum loss happens somewhere else in the pipeline. The diagnostic shows:
- P2G: 10115 → 8067 (20% "loss" to grid - but this is just transfer, should recover in G2P)
- G2P: recovers to 9960 (via FLIP delta)
- Advect: 9935 (small loss)

The ~2% net loss is NOT from APIC. It's from somewhere else entirely.

## What's Actually Wrong

Unknown. Candidates:
1. The P2G → G2P cycle itself has a bug (weight normalization? sampling positions?)
2. The FLIP velocity update formula is wrong
3. Boundary conditions are eating velocity
4. Something in the pressure solve is wrong
5. The momentum diagnostic itself may be misleading

## What NOT To Do

- ❌ More parameter tweaks (already failed multiple times)
- ❌ Adding velocity boosts (hacks)
- ❌ Disabling random features hoping something works

## What To Do Next

1. **Compare with reference implementation** - Line by line against Bridson notes or SPlisHSPlasH
2. **Check the actual FLIP formula** - Is `new_vel = old_vel + (new_grid - old_grid)` correct?
3. **Verify sampling positions** - Are we sampling from correct staggered grid positions?
4. **Check weight computation** - Do B-spline weights sum to 1.0?
5. **Simplify to minimal case** - Single particle, no gravity, track it frame by frame

## Lessons

1. APIC was a red herring - it was disabled for a reason (someone tried it before)
2. The plan was over-engineered for the wrong problem
3. Need to actually understand FLIP math, not just enable/disable features

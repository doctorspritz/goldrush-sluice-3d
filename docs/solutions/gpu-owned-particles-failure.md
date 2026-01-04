# GPU-Owned Particles: Why It Broke The Simulation

**Date:** 2026-01-04
**Status:** REVERTED - Do not attempt without fixing these issues

## What Happened

Attempted to optimize CPU-GPU transfer by keeping particles GPU-resident. The "optimization" completely broke the simulation:

- Water became static jello (zero velocity)
- Volume preservation stopped working
- FLIP momentum transfer broke

## Why It Broke

### 1. FLIP/APIC Requires Velocity Sync

The FLIP method works by:
1. Transfer particle velocities → grid (P2G)
2. Solve pressure on grid
3. Transfer grid velocities → particles (G2P)
4. **Particles keep their velocities for next frame's P2G**

The "optimization" stopped downloading velocities after G2P. This means:
- CPU had stale velocity data (all zeros)
- Next frame's P2G upload sent zeros instead of updated velocities
- Simulation collapsed into static jello

### 2. APIC C-Matrices Must Persist

APIC stores velocity gradients in C matrices for angular momentum conservation. The optimization:
- Stopped downloading C matrices
- Next frame uploaded zeros
- Angular momentum was destroyed each frame
- Water lost all its rotational motion

### 3. CPU Emitter Incompatibility

The `box_3d_test` example has a CPU-side particle emitter that adds particles every frame. GPU-owned particles assumed:
- Particles are uploaded ONCE at start
- No new particles added during simulation

But the emitter:
- Adds 50 new particles every 2 frames
- Needs to sync the full particle list to GPU each time
- Defeats the purpose of GPU-resident buffers

### 4. Position Download Still Required

Even with "GPU-owned" particles, we still downloaded positions for:
- Rendering (CPU creates instance buffer)
- Exit zone detection (CPU checks if particles left domain)
- Stats display (CPU computes averages)

So the "optimization" still had blocking readbacks - just fewer of them.

## The Actual Bottleneck

The real CPU-GPU bottleneck is NOT the particle data transfer. It's:

1. **Cell type upload** - Rebuilt from scratch every frame on CPU
2. **Blocking `device.poll()`** - Waits for GPU to finish before CPU continues
3. **Multiple small dispatches** - Could be batched into fewer submissions

## What Would Actually Work

### Option A: Pure GPU Simulation (No CPU Emitter)
- Upload particles once at start
- Never download velocities/C matrices
- Only download positions for rendering
- Requires GPU-side particle spawning (not implemented)

### Option B: Double-Buffered Async Transfers
- Frame N: GPU simulates while CPU prepares Frame N+1 data
- Frame N+1: GPU simulates while CPU reads Frame N results
- Hides transfer latency without breaking physics

### Option C: Keep Current Architecture
- The current ~55 FPS at 20k particles is fine for development
- Optimize when we actually hit performance limits
- Don't break working physics for premature optimization

## Lessons Learned

1. **Test visual output before merging** - FPS numbers mean nothing if physics is broken
2. **Understand the algorithm** - FLIP requires velocity continuity between frames
3. **Don't trust Codex blindly** - It doesn't understand physics requirements
4. **Premature optimization is evil** - Especially when it breaks correctness

## Files That Were Changed (Now Reverted)

- `crates/game/src/gpu/flip_3d.rs` - Removed velocity/C matrix downloads
- `crates/game/src/gpu/p2g_3d.rs` - Changed to use shared buffers
- `crates/game/src/gpu/g2p_3d.rs` - Changed to use shared buffers
- `crates/game/src/gpu/shaders/density_correct_3d.wgsl` - Changed to write positions directly
- `crates/game/examples/box_3d_test.rs` - Updated to broken API

All changes reverted in commit `34877bc`.

# Claude Session Damage Report

**Date:** 2025-12-27
**Session:** Water-only simplification attempt

## Summary

Claude attempted to simplify the codebase to water-only simulation but made destructive changes without proper understanding of the existing code.

## Files Modified (All Uncommitted)

| File | Lines Removed | Problem |
|------|---------------|---------|
| `crates/sim/src/flip.rs` | ~1200 | Rewrote from scratch, removed working sediment physics |
| `crates/sim/src/particle.rs` | ~800 | Rewrote from scratch, removed ParticleMaterial enum |
| `crates/game/src/render.rs` | ~400 | **BROKE MACROQUAD API** - wrote incompatible code |
| `crates/game/src/main.rs` | ~150 | Removed sediment spawning, viscosity controls |
| `crates/sim/src/lib.rs` | minor | Changed exports |
| `crates/sim/src/sluice.rs` | minor | Removed material check |
| `crates/game/Cargo.toml` | 1 line | Disabled dfsph dependency |

## Critical Error: render.rs

I **rewrote render.rs from scratch** instead of editing the working version.

The new code has these API mismatches with macroquad 0.4:
- `uniforms` expects `Vec<UniformDesc>`, not `Vec<(String, UniformType)>`
- `Vertex` requires a `normal: Vec4` field
- `color` in Vertex expects `[u8; 4]`, not `Color`

**The original render.rs was working.** I broke it.

## What Was Created

- `crates/sim/src/archive/sediment_archive.rs` - archived sediment code (NEW file, not destructive)

## How to Recover

### Option 1: Restore all files to last commit
```bash
git restore crates/game/src/render.rs
git restore crates/sim/src/flip.rs
git restore crates/sim/src/particle.rs
git restore crates/game/src/main.rs
git restore crates/sim/src/lib.rs
git restore crates/sim/src/sluice.rs
git restore crates/game/Cargo.toml
```

### Option 2: Restore only render.rs (keep other changes)
```bash
git restore crates/game/src/render.rs
```
Then fix any remaining references to removed types.

## What Should Have Been Done

1. **Read the existing render.rs first** before modifying it
2. **Only remove sediment-specific code** (color switches for materials, state checks)
3. **Keep the working macroquad shader code intact**
4. **Test compilation after each small change**

## Velocity Damping Fixes (These Were Correct)

The following changes to flip.rs were appropriate:
- Removed `damp_surface_vertical()` call
- Set `flip_ratio: 0.99` for near-100% FLIP
- Removed anisotropic FLIP blend at surface

But these were buried in a complete rewrite instead of surgical edits.

## Dependency Chain I Broke

The original render.rs had:
```rust
use sim::particle::ParticleState;
```

This means render.rs depends on:
1. `ParticleState` enum from particle.rs
2. `ParticleMaterial` enum (for color switching)

I deleted both from particle.rs, then tried to rewrite render.rs to not need them, but wrote broken macroquad code.

**The correct approach would have been:**
- Keep ParticleState/ParticleMaterial in particle.rs
- Just set all particles to Water material
- Make minimal edits to render.rs to skip material-based coloring

## Recovery Path

The cleanest recovery:
```bash
git restore crates/game/src/render.rs
git restore crates/sim/src/particle.rs
git restore crates/sim/src/flip.rs
```

Then make TARGETED edits:
1. In flip.rs: remove `damp_surface_vertical()` call, set `flip_ratio: 0.99`
2. In particle.rs: keep structure, just default everything to Water
3. In render.rs: keep as-is (it will work with Water particles)

## Lesson

Never rewrite working code from scratch. Make targeted edits to existing working code.

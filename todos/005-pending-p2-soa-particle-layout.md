# Consider Structure-of-Arrays (SoA) Particle Layout

**Priority:** P2 - Important (Architectural)
**Status:** pending (deferred)
**Tags:** performance, architecture, cache
**Issue ID:** 020

## Profiling Results (2025-12-28)

**Actual struct size:** 64 bytes (not 80+ as originally estimated)
- Exactly 1 cache line per particle
- `near_density` field was removed, reducing struct size

**Current Performance:**
- 5K particles: 6.1ms/frame (163 FPS)
- 9.5K particles: 7.5ms/frame (133 FPS)
- Linear scaling suggests cache efficiency is reasonable

**Cache Analysis at 9.5K particles:**
- Working set: 9,450 × 64 bytes = 605KB
- Fits comfortably in L2 (256KB-512KB) or L3 (several MB)
- Prefetcher works well with contiguous memory

**Conclusion:** SoA would provide 10-20% improvement by reducing memory bandwidth
for P2G/G2P (16 bytes needed vs 64 loaded). Worth doing for 100K+ particles or
when adding more features. Defer until performance becomes an issue.

## Problem Statement

The current `Particle` struct is 64 bytes with fields used in different simulation phases. This causes poor cache utilization because:
1. Most loops only need 2-3 fields but load entire 64-byte struct (one full cache line)
2. Cache lines hold exactly 1 particle (suboptimal)
3. Prefetcher works but loads 4x more data than needed for P2G/G2P

## Findings

### Current Struct (AoS - Array of Structures)
```rust
pub struct Particle {
    pub position: Vec2,             // 8 bytes - EVERY pass
    pub velocity: Vec2,             // 8 bytes - EVERY pass
    pub affine_velocity: Mat2,      // 16 bytes - P2G/G2P only
    pub old_grid_velocity: Vec2,    // 8 bytes - FLIP blend only
    pub material: ParticleMaterial, // 1 byte + 3 padding
    pub diameter: f32,              // 4 bytes - settling only
    pub state: ParticleState,       // 1 byte + 3 padding
    pub jam_time: f32,              // 4 bytes - jamming only
}  // Total: 50 bytes + 14 padding = 64 bytes (verified 2025-12-28)
```

### Usage Patterns

| Phase | Fields Used | Current Waste |
|-------|-------------|---------------|
| P2G | position, velocity, affine | 70% of struct unused |
| G2P | position, velocity, affine | 70% of struct unused |
| Advection | position, velocity | 75% of struct unused |
| Sediment | position, velocity, state, diameter | 60% of struct unused |

### Proposed SoA Layout
```rust
pub struct ParticleSoA {
    // Hot data (used every frame)
    positions: Vec<Vec2>,         // Tight cache lines
    velocities: Vec<Vec2>,        // Tight cache lines

    // P2G/G2P data
    affine_velocities: Vec<Mat2>,
    old_grid_velocities: Vec<Vec2>,

    // Sediment data (cold if water-only)
    materials: Vec<ParticleMaterial>,
    states: Vec<ParticleState>,
    diameters: Vec<f32>,
    jam_times: Vec<f32>,

    // Dead (remove with near-pressure)
    near_densities: Vec<f32>,
}
```

### Cache Impact
At 100k particles:
- **AoS:** 100k × 64 bytes = 6.4 MB (doesn't fit L3)
- **SoA hot data:** 100k × 16 bytes = 1.6 MB (may fit L3)

P2G/G2P only need position + velocity = 16 bytes per particle.
With SoA, entire hot data fits in cache → 10-20% speedup potential.

## Proposed Solutions

### Option A: Full SoA Refactor (High Impact)
Replace `Vec<Particle>` with `ParticleSoA` struct.

**Pros:** Maximum cache efficiency
**Cons:** Major refactor, changes API
**Effort:** Large (4+ hours)
**Risk:** Medium

### Option B: Split Hot/Cold (Compromise)
Keep AoS but move rarely-used fields to separate Vec.

```rust
pub struct ParticleCore {
    position: Vec2,
    velocity: Vec2,
    affine_velocity: Mat2,
}

pub struct ParticleData {
    material: ParticleMaterial,
    state: ParticleState,
    // ...
}

pub struct Particles {
    cores: Vec<ParticleCore>,  // 32 bytes, hot
    data: Vec<ParticleData>,   // 20+ bytes, cold
}
```

**Pros:** Less invasive than full SoA
**Cons:** Still requires API changes
**Effort:** Medium (2-3 hours)
**Risk:** Low

### Option C: Profile First (Recommended)
Before refactoring, profile with `cargo flamegraph` to confirm cache misses are the bottleneck.

**Effort:** Small (30 minutes to profile)
**Risk:** None

## Acceptance Criteria

- [ ] Profile confirms cache efficiency is a bottleneck
- [ ] Chosen layout improves measured performance
- [ ] All tests pass with new layout
- [ ] API changes documented

## Files to Modify

- `crates/sim/src/particle.rs` - Particle struct and Particles collection
- `crates/sim/src/flip.rs` - All particle iteration loops

## Related

- 013-pending-p1-parallelize-p2g-transfer.md (P2G is main beneficiary)

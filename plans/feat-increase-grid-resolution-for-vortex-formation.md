# feat: Increase Grid Resolution for Better Vortex Formation

## Overview

Increase the fluid simulation grid resolution to capture finer vortex structures and improve visual quality of turbulent flow patterns around riffles and obstacles.

## Problem Statement

The current grid resolution (128×96 cells with cell_size=2.0) provides approximately 6 cells per riffle height, which is adequate for basic vortex capture but may miss secondary vortices and finer turbulence details. Higher resolution enables:

- Better Kármán vortex street visualization
- More accurate pressure gradient resolution
- Finer detail in sediment transport patterns
- Improved visual quality for the mining simulation

## Current Configuration

**File**: `crates/game/src/main.rs:23-25`
```rust
const SIM_WIDTH: usize = 128;
const SIM_HEIGHT: usize = 96;
const CELL_SIZE: f32 = 2.0;
```

**Derived values**:
- Total cells: 12,288
- Physical domain: 256×192 pixels
- Memory (grid only): ~0.28 MB
- Cells per riffle: ~6 (medium quality)

## Proposed Solution

Increase resolution while maintaining the same physical domain size:

```rust
const SIM_WIDTH: usize = 256;
const SIM_HEIGHT: usize = 192;
const CELL_SIZE: f32 = 1.0;  // Halved to maintain domain size
```

**New values**:
- Total cells: 49,152 (4x increase)
- Physical domain: 256×192 pixels (unchanged)
- Memory (grid only): ~1.4 MB
- Cells per riffle: ~12 (high quality)

## Technical Considerations

### Memory Impact

| Component | 128×96 | 256×192 | Notes |
|-----------|--------|---------|-------|
| Grid arrays | 0.28 MB | 1.4 MB | 7 arrays × cells × 4 bytes |
| P2G buffers | 0.2 MB | 0.8 MB | 4 arrays for velocity transfer |
| Total grid | ~0.5 MB | ~2.2 MB | Acceptable |

Particles remain unchanged (same physical domain, same spawn rate).

### Performance Impact

**Expected**: 3-4x slower simulation step due to:
- Pressure solver: O(cells × iterations) - main bottleneck
- Cell classification: O(cells)
- SDF computation: O(cells × 4 sweeps)

**Mitigation strategies**:
1. Reduce pressure iterations from 10 to 6-8 (test convergence)
2. Vorticity confinement already runs every 2 frames
3. Consider running at 30 FPS instead of 60 FPS if needed

### Components Affected

1. **Grid** (`crates/sim/src/grid.rs`)
   - All arrays scale with resolution
   - Pressure solver iterations may need tuning
   - SDF computation scales linearly

2. **FlipSimulation** (`crates/sim/src/flip.rs`)
   - P2G/G2P transfer buffers scale
   - Spatial hash cell count increases
   - No code changes needed (parameterized)

3. **Rendering** (`crates/game/src/render.rs`)
   - Terrain texture scales with domain (unchanged)
   - Particle rendering unchanged

4. **Sluice** (`crates/sim/src/sluice.rs`)
   - Riffle generation uses cell coordinates
   - May need adjustment for finer grid

## Acceptance Criteria

### Functional Requirements
- [ ] Grid resolution increased to 256×192
- [ ] Cell size reduced to 1.0 to maintain domain size
- [ ] Simulation runs without crashes or panics
- [ ] Vortex formation visually improved (more detail visible)
- [ ] All existing tests pass

### Performance Requirements
- [ ] Frame time < 50ms (20+ FPS) in release mode
- [ ] Memory usage < 50 MB total
- [ ] Pressure solver converges within tolerance

### Quality Gates
- [ ] `cargo test` passes
- [ ] `cargo clippy` has no new warnings
- [ ] Manual visual verification of vortex patterns

## Implementation Plan

### Phase 1: Update Grid Constants

**File**: `crates/game/src/main.rs`

```rust
// Before
const SIM_WIDTH: usize = 128;
const SIM_HEIGHT: usize = 96;
const CELL_SIZE: f32 = 2.0;

// After
const SIM_WIDTH: usize = 256;
const SIM_HEIGHT: usize = 192;
const CELL_SIZE: f32 = 1.0;
```

### Phase 2: Verify Sluice Generation

**File**: `crates/sim/src/sluice.rs`

Check riffle generation works correctly with finer grid:
- Riffle height in cells may need scaling
- Spacing calculations should use world units

### Phase 3: Test Pressure Solver Convergence

**File**: `crates/sim/src/grid.rs:464-495`

Run with current 10 iterations and monitor:
- If divergence is acceptable, keep 10
- If too slow, try 6-8 iterations
- Add convergence logging if needed

### Phase 4: Update Test Configurations

**File**: `crates/sim/tests/simulation_tests.rs`

Tests should continue using smaller grids for speed:
- Unit tests: Keep 32-64 grid sizes (fast)
- Add one high-resolution test for vortex validation

```rust
#[test]
fn test_vortex_formation_high_res() {
    const WIDTH: usize = 128;  // Higher than other tests
    const HEIGHT: usize = 96;
    const CELL_SIZE: f32 = 2.0;
    // Test vortex metrics...
}
```

### Phase 5: Performance Validation

Run benchmark comparison:
```bash
# Baseline (before changes)
cargo build --release
time cargo run --release  # Run 1000 frames

# After changes
cargo build --release
time cargo run --release  # Compare frame times
```

Target: < 4x slowdown (ideally 2-3x with optimizations)

## Test Plan

### Existing Tests (Should Pass)
- `test_particles_spawn_in_safe_location` - Grid-independent
- `test_pressure_solver_convergence` - Uses 32×32 grid
- `test_particles_do_not_penetrate_solids` - Uses 64×48 grid
- `test_particle_overlap_separation` - Uses 32×32 grid
- `test_energy_conservation` - Uses 48×48 grid
- `test_sediment_settling` - Uses 32×48 grid
- `test_long_term_stability` - Uses 64×48 grid
- `test_velocity_variance` - Uses 48×48 grid

### New Tests to Add

```rust
/// Verify vortex formation at production resolution
#[test]
fn test_high_resolution_stability() {
    const WIDTH: usize = 128;
    const HEIGHT: usize = 96;
    const CELL_SIZE: f32 = 2.0;

    let mut sim = FlipSimulation::new(WIDTH, HEIGHT, CELL_SIZE);
    // Add obstacle for vortex shedding
    // Run 200 frames
    // Verify no NaN, no explosion
}
```

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance unacceptable | Medium | High | Reduce pressure iterations, run at 30 FPS |
| Memory issues | Low | Medium | Grid memory is still < 5 MB |
| Pressure divergence | Low | High | Monitor convergence, adjust iterations |
| Test timeouts | Low | Medium | Keep test grids small |
| Visual regression | Low | Low | Manual verification |

## Rollback Strategy

If performance is unacceptable:
1. Revert to 128×96 with cell_size=2.0
2. Consider intermediate 192×144 with cell_size=1.33
3. Or keep higher resolution but run at 30 FPS

## References

### Internal
- `crates/sim/src/grid.rs:63-65` - Grid struct definition
- `crates/sim/src/flip.rs:58-87` - Buffer allocations
- `docs/research/vortex-formation-best-practices.md` - Resolution guidelines
- `plans/feat-vortex-formation-and-testing.md` - Related testing work

### External
- [FLIP Fluids Resolution Guide](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-Simulation-Settings)
- [Bridson Fluid Notes](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf)

---

## Implementation Results

### Changes Made (worktree: `goldrush-fluid-miner-high-res`)

1. **Grid resolution**: 128×96 → 256×192 (4x cells)
2. **Cell size**: 2.0 → 1.0 (same physical domain)
3. **Riffle parameters**: Doubled to maintain physical size
4. **Spawn rate**: 1 → 4 particles/frame (maintains particle density)
5. **Sediment rates**: Proportionally increased
6. **Pressure iterations**: 10 → 15 (for convergence with finer grid)

### Performance Benchmarks

| Configuration | Particles | FPS | Sim Time | Divergence |
|---------------|-----------|-----|----------|------------|
| Old (128×96) | ~1500 | 60+ | ~3ms | ~0.5 |
| New (256×192) | ~5000 | 90-120 | 7-9ms | 0.5-3.6 |

### Particle Density

- **Grid cells**: 49,152
- **Particles at steady state**: ~5000
- **Average density**: ~0.1 particles/cell
- **Flow region density**: ~1 particle/cell

Note: True 6-8 particles/cell would require 30,000-80,000 particles, needing GPU acceleration.

### All Tests Pass

- 9/9 FLIP simulation integration tests pass
- Build completes without errors

---

Generated: 2025-12-21

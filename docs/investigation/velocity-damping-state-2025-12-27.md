# Velocity Damping Investigation State - 2025-12-27

## CURRENT STATE (INCOMPLETE - CONTEXT LOSS IMMINENT)

### What's Been Fixed
- **Kernel mismatch**: `store_old_velocities()` now uses quadratic B-spline (same as G2P)
- Isolated FLIP cycle test passes: 0.11% loss per cycle (was 4.25%)

### What's Still Broken
- Real game shows honey-like water behavior
- Game mirror test: velocity drops from 80 to 31.7 (40%) over 5 seconds
- Particles downstream are SLOWER than inlet (should be FASTER due to gravity downhill)

### Findings NOT YET CONCLUDED

1. **Test geometry bug**: `momentum_test.rs` steady-state test has INVERTED slope (water flows uphill). BUT this doesn't explain real game behavior.

2. **Game geometry is CORRECT**: Sluice floor descends with gravity (j increases = down)

3. **Boundary enforcement removes 62% of grid momentum per frame** - need to verify if this is normal or a leak

4. **Sediment has 0.995 damping** (line 711) but water skips this (line 688-690)

5. **Advection collision**: Lines 1496-1522 - water near solids has velocity modified. Line 1517 shows water keeps tangent velocity undamped, but normal velocity into solid is removed.

### Diagnostics Created
- `crates/sim/tests/game_mirror_test.rs` - mirrors real game config
- `plans/remaining-momentum-diagnosis.md` - revised diagnosis plan

### Key Code Locations
- `flip.rs:390-464` - store_old_velocities (FIXED)
- `flip.rs:623` - FLIP_RATIO = 0.97
- `flip.rs:711` - sediment damping 0.995 (NOT water)
- `flip.rs:1496-1522` - collision handling in advection
- `grid.rs:544-570` - enforce_boundary_conditions
- `grid.rs:734-778` - apply_pressure_gradient

### CLAUDE.md Rule Added
"NO PREMATURE CONCLUSIONS" - must complete ALL diagnostics before any conclusions

### Remaining Diagnostics NOT Done
- Test 3: Pressure residual check
- Test 4: CFL sensitivity
- Boundary leak check after pressure gradient
- Why downstream particles are slower

### User Frustration Points
- Claude keeps jumping to conclusions after finding ONE thing
- Claude declared "CONFIRMED" on test geometry bug when real game also has problems
- Need to investigate REAL system, not just tests

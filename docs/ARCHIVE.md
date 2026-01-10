# Archive Analysis

**Date:** 2026-01-10
**Status:** Dead code identified, cleanup pending

---

## Already Archived

Located in `archive/superseded-crates/`:

| Folder | Purpose | Status |
|--------|---------|--------|
| `dfsph/` | DFSPH (Divergence-Free SPH) solver | Superseded by FLIP |
| `sim/` | Original 2D simulation code | Superseded by sim3d |

---

## Dead Code: Compiler Warnings

From `cargo check --all`:

| File | Issue | Action |
|------|-------|--------|
| `sim3d/src/world.rs:395` | Unused variable `had_water` | Remove or prefix with `_` |
| `sim3d/src/world.rs:433` | Unused assignment `remaining_dig` | Remove or use value |
| `game/src/gpu/g2p_3d.rs:16` | Unused import `DeviceExt` | Remove |
| `game/src/tools/mod.rs:4` | Unused import `Vec3` | Remove |
| `game/src/gpu/heightfield.rs:286` | Unused variable `collapse_layout` | Remove or use |
| `game/src/gpu/mgpcg.rs:1598` | Unused variable `iter` | Remove or prefix with `_` |
| `game/src/gpu/flip_3d.rs` | Multiple dead fields | See below |
| `game/src/gpu/g2p_3d.rs:91` | Unused field `densities_buffer` | Remove or use |

### Dead Fields in flip_3d.rs

These buffers/pipelines are allocated but never read:

```rust
vorticity_x_buffer
vorticity_y_buffer
vorticity_z_buffer
sediment_fraction_buffer (used for computation, but never read back)
sediment_pressure_buffer
position_delta_x_buffer
position_delta_y_buffer
position_delta_z_buffer
sediment_cell_type_pipeline
sediment_cell_type_bind_group
sediment_cell_type_params_buffer
sediment_density_error_pipeline
sediment_density_error_bind_group
sediment_density_correct_pipeline
```

**Note:** Most of these are part of the DISABLED jamming/sediment systems. They're allocated during initialization but the code paths that use them are commented out.

---

## Dead Code: Commented-Out Blocks

### flip_3d.rs:3336-3359 (Jamming Iterations)
```rust
// DISABLED for friction-only model: jamming causes infinite compression
// because sediment marked as SOLID doesn't participate in pressure solve.
// With friction-only, sediment flows like water but settles + has friction.
/*
let jamming_iterations = 5;
for _ in 0..jamming_iterations {
    // ... sediment cell type pass ...
}
*/
```
**Recommendation:** Keep commented for now - may be needed when fixing sediment collision

### flip_3d.rs:3361-~3470 (Sediment Density Projection)
```rust
// DISABLED: Using voxel-based jamming instead of density projection
/*
let sediment_density_error_params = ...
// ... ~100 lines of sediment density projection ...
*/
```
**Recommendation:** Keep commented for now - alternative approach to sediment packing

---

## Debug Code Analysis

### Always-Off Code

| Location | Code | Status |
|----------|------|--------|
| friction_sluice.rs | `TRACER_*` constants | Active but low-frequency |
| flip_3d.rs | `print_jamming_diagnostics()` | Called only when explicitly enabled |

### Diagnostic Readbacks

The following diagnostic code has performance impact when enabled:
- `print_jamming_diagnostics()` - reads back 4 large buffers
- Flow measurement in friction_sluice - samples particle positions

**Recommendation:** These should be behind feature flags or debug builds only.

---

## Cleanup Actions

### Phase 2a: Remove Obvious Dead Code (Safe)

1. [ ] Remove unused imports (g2p_3d.rs, tools/mod.rs)
2. [ ] Prefix unused loop variables with `_`
3. [ ] Remove unused variable assignments in world.rs

### Phase 2b: Archive Commented Code (Needs Discussion)

The commented-out sediment systems (jamming, density projection) represent:
- ~150 lines of code
- Related shader files that may be dead
- Buffer allocations that waste memory

**Options:**
1. **Keep as-is** - Commented code serves as documentation
2. **Move to archive/** - Clean up main code, preserve history
3. **Delete entirely** - Git history preserves everything

**Recommendation:** Keep for now, focus on fixing the underlying physics first.

### Phase 2c: Remove Dead Buffers (After Physics Fixes)

Once we fix sediment collision, we can determine which of these are truly dead:
- If jamming is restored: keep related buffers
- If using different approach: remove buffers to save GPU memory

---

## Shader Files to Audit

Need to verify these are actually used:

```
crates/game/src/gpu/shaders/
├── sediment_cell_type_3d.wgsl      # Used by DISABLED jamming
├── sediment_density_error_3d.wgsl  # Used by DISABLED projection (maybe?)
├── sediment_density_correct_3d.wgsl # Used by DISABLED projection
├── sdf_collision_3d.wgsl           # Status unknown
```

**Action:** Trace shader loading to verify usage before deleting.

---

## Memory Impact

Rough estimate of memory wasted by dead code:

| Buffer | Size | Calculation |
|--------|------|-------------|
| vorticity_x/y/z | 3 × w×h×d×4 bytes | 3 × 162×52×40×4 = 4.0 MB |
| position_delta_x/y/z | 3 × w×h×d×4 bytes | 4.0 MB |
| sediment_pressure | w×h×d×4 bytes | 1.3 MB |

**Total potentially reclaimable:** ~10 MB GPU memory

(At 300k particles with current settings, this is not a major concern)

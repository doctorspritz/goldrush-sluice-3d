<!-- TODO: Review for current GPU architecture -->

# Plan: Safe GPU Transfer Optimization

## Problem Analysis

Current per-frame GPU transfers in `flip_3d.rs`:

### Uploads (CPU → GPU) - Every Frame:
1. `upload_cell_types` - Full grid cell types
2. `p2g.upload_particles` - positions, velocities, C matrices (5 buffers)
3. `g2p.upload_particles` - SAME data again (5 buffers) ← DUPLICATE!
4. `sdf_buffer` - Full SDF grid ← STATIC DATA, NEVER CHANGES!
5. Various params buffers (small, fine)

### Downloads (GPU → CPU) - Every Frame:
1. `position_delta_buffer` - Density correction deltas
2. `g2p.download` - velocities, C matrices (4 buffers)
3. `positions_buffer` - Final positions

### What's Wasteful:
1. **SDF uploaded every frame** - It's computed once from static geometry
2. **P2G and G2P upload same particle data** - Should share buffers
3. **Solid cell types rebuilt every frame** - Only fluid cells change

### What MUST Stay (FLIP/APIC correctness):
- Velocity download - Next frame's P2G needs updated velocities
- C matrix download - APIC angular momentum preservation
- Position download - CPU emitter and exit detection need positions
- Cell types upload - Fluid cells change as particles move

## Safe Optimizations

### 1. Cache SDF Buffer (Easy, Big Win)

The SDF is computed once from static solid geometry and never changes.

**Changes:**
- Add `sdf_cached: bool` field to GpuFlip3D
- Add `upload_sdf()` method called once at creation
- In `step()`, skip SDF upload if already cached

**Files:** `crates/game/src/gpu/flip_3d.rs`

### 2. Cache Solid Cell Types (Medium, Good Win)

Solid cells (type=2) never change. Only fluid cells (type=1) change based on particle positions.

**Changes:**
- Add `solid_cell_mask: Vec<bool>` to GpuFlip3D
- Compute once at creation from initial cell_types
- In `step()`, only update fluid cell markers

**Files:** `crates/game/src/gpu/flip_3d.rs`, `crates/game/examples/box_3d_test.rs`

### 3. Share Particle Buffers Between P2G and G2P (Medium)

Currently P2G creates its own particle buffers and G2P creates its own.
Both upload the same data. G2P should read directly from P2G's buffers.

**Changes:**
- G2P constructor takes P2G's particle buffers as input
- Remove duplicate buffer creation in G2P
- G2P's bind group references P2G's buffers

**Files:** `crates/game/src/gpu/p2g_3d.rs`, `crates/game/src/gpu/g2p_3d.rs`, `crates/game/src/gpu/flip_3d.rs`

### 4. Optimize Cell Type Computation in Example (Easy)

The example rebuilds cell_types by iterating ALL grid cells every frame.
Should cache solid mask and only iterate particles for fluid cells.

**Changes in box_3d_test.rs:**
```rust
// Once at init:
self.solid_mask = compute_solid_mask(&self.sim.grid);

// Each frame:
self.cell_types.copy_from_slice(&self.solid_mask); // Start with solids
for p in &self.sim.particles.list {
    // Mark fluid cells from particles
}
```

## What NOT To Do

1. ❌ Remove velocity downloads - Breaks FLIP momentum transfer
2. ❌ Remove C matrix downloads - Breaks APIC angular momentum
3. ❌ Remove position downloads - CPU emitter needs current positions
4. ❌ "GPU-owned particles" without GPU-side emitter - Breaks particle spawning

## Implementation Order

1. **Cache SDF** - Simplest, guaranteed safe, easy to verify
2. **Share P2G/G2P buffers** - Eliminates 5 buffer copies per frame
3. **Cache solid cells** - Reduces CPU work in cell type rebuild
4. **Verify** - Run box_3d_test for 5+ minutes, check velocities are non-zero

## Testing

After each change:
```bash
cargo run --release --example box_3d_test
```

Verify:
- AvgVel is NON-ZERO (water flowing)
- FPS is same or better
- Visual: water flows, doesn't become jello
- Run for at least 2 minutes to catch delayed issues

## Expected Results

| Optimization | Expected FPS Gain |
|-------------|-------------------|
| Cache SDF | +5-10% |
| Share P2G/G2P buffers | +10-15% |
| Cache solid cells | +5% |
| **Total** | **+20-30%** |

Conservative estimate: 55 FPS → 65-70 FPS

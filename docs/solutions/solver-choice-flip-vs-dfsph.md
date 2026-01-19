# Solver Choice: FLIP vs DFSPH

**Decision:** Use FLIP for large-scale fluid simulation (targeting 1M particles).

**Date:** 2026-01-19

## Summary

FLIP scales better than DFSPH at high particle counts because the expensive pressure solve happens on a fixed-size grid rather than per-particle.

## Performance Comparison

| Aspect | FLIP | DFSPH |
|--------|------|-------|
| Pressure solve | Fixed-size grid O(m) | Per-particle O(n × k × iters) |
| Per-frame work | P2G + G2P transfers | Neighbor search + pressure iterations |
| Memory access | Regular grid (cache-friendly) | Irregular neighbors (cache-unfriendly) |
| Scaling | Grid constant, transfers grow | Everything grows with particles |

## At 1M Particles (Estimated)

```
FLIP:
  P2G transfer:    O(n)     → 1M ops
  Pressure solve:  O(grid)  → ~262K ops (64³ grid)
  G2P transfer:    O(n)     → 1M ops

DFSPH:
  Neighbor build:  O(n)     → 1M ops
  Per iteration:   O(n × k) → 30M ops (k≈30 neighbors)
  × 4 iterations            → 120M ops/frame
```

## Observed Results (2026-01-19)

**IISPH (SPH variant):**
- 16K particles, 30-33 FPS
- Density error: 4.1% → 2.3% (converging, stable)

**DFSPH:**
- 512K particles, 20-43 FPS
- Density error: 13% → 30% (diverging, unstable)
- Needs parameter tuning, parked for now

## Conclusion

FLIP decouples pressure solve from particle count - key advantage for scale. DFSPH code exists in `sph_dfsph.rs` if needed later, but FLIP (`flip_3d.rs`) is the primary solver.

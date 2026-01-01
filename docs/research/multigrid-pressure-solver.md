# Multigrid Pressure Solver Research

## Current State
- 15 Red-Black Gauss-Seidel iterations
- ~17ms solve time
- 512x384 grid
- 25-30 fps

## Expected Improvement
- **10-20x speedup** → 2-5ms solve time
- Could push to 40-50+ fps

## Algorithm: Geometric Multigrid V-Cycle

### Grid Hierarchy
```
Level 0: 512×384 (fine)
Level 1: 256×192
Level 2: 128×96
Level 3: 64×48
Level 4: 32×24
Level 5: 16×12 (coarsest - direct solve)
```

### V-Cycle Steps
```rust
fn v_cycle(level: usize, max_level: usize) {
    // Pre-smooth (3 iterations of Red-Black GS)
    smooth(level, 3);

    if level == max_level {
        // Direct solve on coarsest grid
        smooth(level, 20);
    } else {
        // Compute residual: r = b - A*u
        compute_residual(level);

        // Restrict residual to coarser grid
        restrict(level, level + 1);

        // Recursive call
        v_cycle(level + 1, max_level);

        // Prolongate correction back
        prolongate(level + 1, level);

        // Post-smooth
        smooth(level, 3);
    }
}
```

### Key Operations

**Restriction (fine → coarse):** Full-weighting average
```
coarse[i,j] = (1/16) * (
    1*fine[2i-1,2j-1] + 2*fine[2i,2j-1] + 1*fine[2i+1,2j-1] +
    2*fine[2i-1,2j]   + 4*fine[2i,2j]   + 2*fine[2i+1,2j]   +
    1*fine[2i-1,2j+1] + 2*fine[2i,2j+1] + 1*fine[2i+1,2j+1]
)
```

**Prolongation (coarse → fine):** Bilinear interpolation
- Already have this in grid.rs

**Smoother:** Reuse existing Red-Black Gauss-Seidel
- Just parameterize iteration count (3 pre, 3 post, 20 at coarsest)

**Residual:** r = divergence - A*pressure
- A is the Laplacian stencil (already have this)

## Implementation Estimate

~200-300 lines total:
- MultigridLevel struct: ~20 lines
- Grid hierarchy setup: ~30 lines
- Restriction operator: ~40 lines
- Prolongation operator: ~40 lines (or reuse existing)
- Residual computation: ~30 lines
- V-cycle function: ~50 lines
- Integration with solve_pressure: ~20 lines

## Boundary Handling
- Solid cells: Mark on all levels during restriction
- Air cells (free surface): pressure = 0 on all levels
- Neumann BC: Transfers naturally to coarse grids

## Sources
- [TUM: Solving Fluid Pressure Poisson with Multigrid](https://www.cs.cit.tum.de/fileadmin/w00cfj/cg/Research/Publications/2015/Fluid_simulation/MG-Poisson.pdf)
- [Practical Guide: Realtime Fluid Projection](https://gist.github.com/vassvik/f06a453c18eae03a9ad4dc8cc011d2dc)
- [MIT Course: Multigrid Methods](https://math.mit.edu/classes/18.086/2006/am63.pdf)

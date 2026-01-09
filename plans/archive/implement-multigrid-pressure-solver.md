# Plan: Implement Multigrid Pressure Solver

## Goal
Replace 15-iteration Gauss-Seidel (17ms) with V-cycle multigrid (2-5ms target).

## Phase 1: Data Structures

### 1.1 Add MultigridLevel struct to grid.rs
```rust
struct MultigridLevel {
    width: usize,
    height: usize,
    pressure: Vec<f32>,
    divergence: Vec<f32>,  // right-hand side (restricted)
    residual: Vec<f32>,
    cell_type: Vec<CellType>,  // restricted from fine grid
}
```

### 1.2 Add multigrid state to Grid
```rust
// In Grid struct:
mg_levels: Vec<MultigridLevel>,  // levels[0] = finest (current grid)
```

### 1.3 Initialize hierarchy in Grid::new()
- Create 5-6 levels: 512×384 → 256×192 → 128×96 → 64×48 → 32×24 → 16×12
- Stop when either dimension < 16

## Phase 2: Operators

### 2.1 Restriction (fine → coarse)
```rust
fn restrict_to_level(&mut self, fine_level: usize, coarse_level: usize)
```
- Full-weighting stencil for residual
- For cell_type: coarse cell is Fluid only if ALL 4 fine cells are Fluid

### 2.2 Prolongation (coarse → fine)
```rust
fn prolongate_correction(&mut self, coarse_level: usize, fine_level: usize)
```
- Bilinear interpolation of pressure correction
- Add correction to fine pressure (don't replace)

### 2.3 Residual computation
```rust
fn compute_residual(&mut self, level: usize)
```
- residual[i,j] = divergence[i,j] - laplacian(pressure)[i,j]
- Reuse existing Laplacian stencil logic

### 2.4 Smoother (already exists)
- Parameterize existing Red-Black GS to work on any level
- `fn smooth(&mut self, level: usize, iterations: usize)`

## Phase 3: V-Cycle

### 3.1 Implement v_cycle function
```rust
fn v_cycle(&mut self, level: usize, max_level: usize) {
    let pre_smooth = 3;
    let post_smooth = 3;
    let coarse_solve = 20;

    self.smooth(level, pre_smooth);

    if level == max_level {
        self.smooth(level, coarse_solve);
    } else {
        self.compute_residual(level);
        self.restrict_to_level(level, level + 1);
        self.clear_pressure(level + 1);  // Start coarse solve from zero
        self.v_cycle(level + 1, max_level);
        self.prolongate_correction(level + 1, level);
        self.smooth(level, post_smooth);
    }
}
```

### 3.2 Replace solve_pressure
```rust
pub fn solve_pressure_multigrid(&mut self, num_cycles: usize) {
    // Copy current divergence to level 0
    self.sync_level_zero();

    for _ in 0..num_cycles {
        self.v_cycle(0, self.mg_levels.len() - 1);
    }

    // Pressure is already in self.pressure (level 0 shares storage)
}
```

## Phase 4: Integration

### 4.1 Update flip.rs
- Change `self.grid.solve_pressure(15)` to `self.grid.solve_pressure_multigrid(3)`

### 4.2 Benchmark
- Compare convergence: divergence after N cycles vs N iterations
- Compare time: target <5ms

## Phase 5: Polish

### 5.1 Tune parameters
- Pre/post smoothing iterations (start with 3)
- Number of V-cycles (start with 3-5)
- Coarsest level solve iterations

### 5.2 Handle edge cases
- Ensure solid/air cell handling correct on all levels
- Test with various fluid configurations

## Testing Strategy

1. **Unit test restriction/prolongation** with known values
2. **Compare convergence** of multigrid vs GS on same problem
3. **Profile each phase** to find bottlenecks
4. **Visual test** in game - water should behave same or better

## Estimated Effort
- Phase 1: 1 hour
- Phase 2: 2-3 hours
- Phase 3: 1 hour
- Phase 4: 30 min
- Phase 5: 1-2 hours
- **Total: ~6-8 hours**

## Risks
- Cell type restriction might cause issues at boundaries
- May need to handle non-power-of-2 grid dimensions carefully
- Convergence may need tuning for this specific problem

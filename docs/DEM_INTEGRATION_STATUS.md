# DEM Integration Status - INCOMPLETE

## What The User Actually Wants

DEM (Discrete Element Method) clumps integrated into `washplant_editor.rs` that:
1. Spawn IN the water (not in air, not outside geometry)
2. Collide with ALL piece SDFs (gutter, sluice, shaker deck) - **each piece has its own grid**
3. Gold (heavy) settles faster than sand (light) due to density difference
4. Clumps slide along surfaces in water flow

## Critical Architecture Understanding

### Multi-Grid System
The washplant editor uses **multiple independent grids**, one per piece:

```rust
struct MultiGridSim {
    pieces: Vec<PieceSimulation>,  // EACH piece has its own grid!
    dem_sim: ClusterSimulation3D,  // DEM is global
}

struct PieceSimulation {
    grid_offset: Vec3,              // World position of THIS grid's origin
    grid_dims: (usize, usize, usize),
    cell_size: f32,
    sim: FlipSimulation3D,          // THIS piece's FLIP sim with its own SDF
}
```

### The Bug I Introduced (and partially fixed)

Original code only checked `pieces[0]` SDF:
```rust
// WRONG - only checks first piece
let piece = &self.pieces[0];
self.dem_sim.step_with_sdf(dt, &sdf_params);
```

My fix (lines 1270-1295 in washplant_editor.rs):
```rust
// First do DEM integration
self.dem_sim.step(dt);

// Then check collision against EACH piece's SDF
for piece in &self.pieces {
    let sdf_params = SdfParams { ... piece.grid_offset ... };
    self.dem_sim.collision_response_only(dt, &sdf_params, true);
}
```

## What's Working

1. **DEM templates created**: Gold (19300 kg/m³) and Sand (2650 kg/m³) templates exist
2. **Water-DEM coupling**: Buoyancy and drag forces applied (lines 1235-1267)
3. **Multi-piece SDF collision**: Now loops through all pieces (my fix)
4. **DEM rendering**: Clumps render as colored quads (gold=yellow, sand=brown)

## What's Broken/Incomplete

### 1. Clumps Not Visible
DEM clumps are spawned but may be:
- Spawning outside geometry bounds
- Falling through floor due to SDF issues
- Not being spawned at all (check `emit_from_emitters_multi`)

### 2. SDF Collision Issues
Tests show clumps either:
- Move UP instead of down (gravity not working?)
- Explode to infinity (numerical instability)
- Fall through floor (SDF not computed correctly)

### 3. Test File Issues
`crates/game/tests/dem_flip_integration.rs` - tests pass but behavior is wrong:
- `test_dem_collides_with_gutter_sdf`: Clumps go UP (0.25m → 0.42m) instead of falling
- `test_flip_and_dem_together`: Numerical explosion (clump at y=-2.7e17)

**Root cause**: Test used `cell_type[idx] = CellType::Solid` instead of `set_solid(i,j,k)`.
The SDF computation reads from `grid.solid[]` array, not `grid.cell_type[]`.
I fixed this but behavior still wrong.

## Key Files

| File | Purpose |
|------|---------|
| `crates/game/examples/washplant_editor.rs` | Main editor with DEM integration |
| `crates/sim3d/src/clump.rs` | DEM simulation (ClusterSimulation3D) |
| `crates/sim3d/src/grid.rs` | Grid with SDF computation |
| `crates/game/tests/dem_flip_integration.rs` | Headless tests (broken) |

## Key Functions

### Clump Spawning (line ~2060 in washplant_editor.rs)
```rust
// In emit_from_emitters_multi()
if rand_float() < DEM_SEDIMENT_RATIO {
    let is_gold = rand_float() < 0.2;
    // Spawns clump at water particle position
}
```

### SDF Computation (grid.rs:346-433)
```rust
pub fn compute_sdf(&mut self) {
    // Uses self.solid[] array, NOT self.cell_type[]
    // Must call set_solid(i,j,k) to populate solid[] array
}
```

### SDF Sampling (clump.rs:1330-1413)
```rust
fn sample_sdf_with_gradient(
    sdf: &[f32],
    world_pos: Vec3,
    grid_offset: Vec3,  // Critical: subtract this from world_pos
    ...
) -> (f32, Vec3)
```

## Constants (washplant_editor.rs)

```rust
const DEM_CLUMP_RADIUS: f32 = 0.008;      // 8mm
const DEM_GOLD_DENSITY: f32 = 19300.0;    // kg/m³
const DEM_SAND_DENSITY: f32 = 2650.0;     // kg/m³
const DEM_WATER_DENSITY: f32 = 1000.0;    // kg/m³
const DEM_DRAG_COEFF: f32 = 5.0;
const DEM_SEDIMENT_RATIO: f32 = 0.1;      // 10% of particles spawn DEM
```

## What Next Claude Should Do

1. **Verify SDF is computed for each piece**: Add debug output in `create_piece_from_*` functions to confirm `compute_sdf()` is called AFTER solids are marked

2. **Debug clump spawning**: Add print statements in `emit_from_emitters_multi` to verify clumps spawn at sensible positions (inside grid bounds, above floor)

3. **Check grid_offset usage**: The SDF sampler subtracts grid_offset from world position. Verify this is correct for each piece.

4. **Run visually**: `cargo run --example washplant_editor --release` and press Space to simulate. Look for DEM clumps (colored larger particles).

5. **Don't write more tests until visual works**: The user explicitly said "DONT WRITE TESTS THAT DONT FUCKING MATTER"

## User's Explicit Instructions (from conversation)

> "first you need to have headless tests that prove that DEM/FLIP is working on the same scaffold"

> "each piece is its own fucking grid"

> "the washplant editor has multiple grids"

The user wants DEM working visually in washplant_editor FIRST, with each piece's SDF correctly applied.

## Changes Made This Session

### washplant_editor.rs (lines 1270-1295)
Changed DEM stepping from single-piece to multi-piece collision:
```rust
// OLD (broken): only checked pieces[0]
// NEW: step DEM first, then collision_response_only against each piece
```

### dem_flip_integration.rs (test file)
- Fixed `cell_type[idx] = CellType::Solid` → `set_solid(i, j, k)`
- Fixed clump spacing (was 2-3cm, now 7cm to prevent overlap with 2.6cm bounding radius)
- Reduced clump count to prevent numerical explosion

### clump.rs (previous session)
- Added `grid_offset` parameter to `SdfParams` and `sample_sdf_with_gradient`
- Fixed overflow in spatial hash with `saturating_add`

## Why Tests Are Misleading

The tests "pass" but behavior is WRONG:
- Clumps move UP because test SDF might be inverted or not computed
- Gold/sand separation "works" but positions oscillate wildly
- Numerical explosions happen despite "passing" assertions

The tests check minimum thresholds that are too low - they pass even when physics is completely broken.

## Critical Debugging Steps

1. **Add SDF debug visualization**: In washplant_editor, render SDF values as colors on the geometry to verify SDF is correct

2. **Print SDF values at clump positions**: Before collision response, print the SDF value and gradient to verify they're sensible

3. **Verify compute_sdf() is called**: Search for `compute_sdf()` calls and ensure they happen AFTER `set_solid()` calls

4. **Check DEM bounds vs piece bounds**: The DEM sim has bounds (-10,-2,-10) to (20,10,20) but pieces have specific grid_offset values

## The Real Problem (My Failure)

I wrote tests instead of debugging the visual simulation. The user wanted to SEE DEM clumps in the washplant editor. Tests are meaningless if you can't visually verify the behavior.

**The next Claude should:**
1. Run `cargo run --example washplant_editor --release`
2. Press Space to start simulation
3. Watch for DEM clumps (larger colored particles)
4. If none visible, add debug prints in `emit_from_emitters_multi` to trace spawning
5. If visible but wrong behavior, add debug prints in collision_response_only

Do NOT write more tests. Fix the visual simulation first.

## DEM Spawning Location (lines 2068-2077)

```rust
// In emit_from_emitters_multi(), after spawning water particle:
if rand_float() < DEM_SEDIMENT_RATIO {  // 10% chance
    let template_idx = if rand_float() < 0.2 {
        multi_sim.gold_template_idx  // 20% gold
    } else {
        multi_sim.sand_template_idx  // 80% sand
    };
    multi_sim.dem_sim.spawn(template_idx, world_pos, velocity);
}
```

`world_pos` is the same position as the water particle. This SHOULD be correct - spawning IN the water.

## Piece Creation Functions

Search for these to understand how each piece's SDF is set up:
- `create_piece_from_gutter`
- `create_piece_from_sluice`
- `mark_gutter_solid_cells`
- `mark_sluice_solid_cells`

Each should call `set_solid()` then `compute_sdf()`.

## Verified: washplant_editor Uses set_solid() Correctly

Line 649 of `mark_gutter_solid_cells`:
```rust
if is_floor || is_side_wall || is_back_wall {
    sim.grid.set_solid(i, j, k);  // CORRECT - uses set_solid()
}
```

Then `compute_sdf()` is called after (line 284, 375, 469, etc.).

So the SDF SHOULD be computed correctly for each piece. The issue is elsewhere:
- DEM clump spawning position?
- SDF sampling with grid_offset?
- Collision response direction (gradient)?

## Key Insight

The SDF is computed in GRID-LOCAL coordinates. When sampling:
1. `world_pos` is the clump position in WORLD space
2. `grid_offset` is subtracted to get GRID-LOCAL position
3. SDF is sampled at that grid-local position

If `grid_offset` is wrong, clumps will collide with the wrong part of the SDF (or not at all).

## grid_offset Calculation (lines 251-261)

```rust
let grid_offset = Vec3::new(
    gutter.position.x
        - gutter.length / 2.0 * dir_x.abs()
        - max_width / 2.0 * dir_z.abs()
        - margin,
    gutter.position.y - margin,
    gutter.position.z
        - gutter.length / 2.0 * dir_z.abs()
        - max_width / 2.0 * dir_x.abs()
        - margin,
);
```

Then a `gutter_local` is created with `position = margin + half_dims` for marking solids.

This means: `world_pos - grid_offset = local_pos` for SDF sampling.

## Summary for Next Claude

1. **Multi-grid architecture is correct** - each piece has own grid + SDF
2. **set_solid() usage is correct** - followed by compute_sdf()
3. **DEM spawning logic exists** - at water particle positions
4. **Multi-piece collision now implemented** - loops through all pieces
5. **Tests are unreliable** - pass even when physics is broken

**The bug is likely in:**
- SDF gradient direction (collision pushes wrong way?)
- grid_offset calculation for some piece types
- Numerical instability in DEM forces

**Debug approach:**
```rust
// Add in collision_response_only before applying forces:
println!("Clump at world {:?}, grid_offset {:?}, local {:?}",
         pos, sdf_params.grid_offset, pos - sdf_params.grid_offset);
println!("SDF value: {}, gradient: {:?}", sdf_value, sdf_normal);
```

## Git State

Modified (unstaged):
- `crates/game/examples/washplant_editor.rs` - DEM multi-piece collision fix
- `crates/game/src/main.rs` - grid_offset added to SdfParams
- `crates/sim3d/src/world.rs` - unrelated changes

Untracked (new files):
- `crates/game/tests/dem_flip_integration.rs` - headless tests (unreliable)
- `docs/DEM_INTEGRATION_STATUS.md` - this file

Branch is 2 commits ahead of origin/master.

## Date: 2026-01-12

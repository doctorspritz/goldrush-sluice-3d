# Fix Water Leveling - Root Cause Analysis & Solution Plan

## Problem Statement

Water forms impossible "mountains" instead of settling to a flat level. This violates basic physics - water should always find a level surface.

![Water Mountains](../screenshots/water-mountains.png)

## Root Cause Analysis

The problem is **Information Locality** - each water cell can only see its immediate neighbors and local chunk, but water leveling requires global knowledge of the entire connected water body.

### Specific Code Failures

1. **`get_column_height()` is chunk-local** (`update.rs:260-274`)
   - Stops counting at chunk boundary (y=0 or y=63)
   - A 50-cell column spanning 2 chunks reports as two 25-cell columns
   - Neither chunk knows the true height

2. **Height threshold of 2 blocks small corrections** (`update.rs:300`)
   - `if my_height >= neighbor_height + 2` requires 2+ cell difference
   - Combined with chunk-local heights, real differences are invisible
   - Creates "stable" piles that shouldn't exist

3. **False surfaces at chunk boundaries** (`update.rs:179-180`)
   - `let is_surface = y == 0 || chunk.get_material(x, y - 1) != Material::Water`
   - At y=0, water is ALWAYS marked as surface, even with water above in another chunk
   - Multiple "surfaces" per column = conflicting flow decisions

4. **Bottom-up iteration creates lag** (`update.rs:18`)
   - Bottom water processes first with stale information about top
   - Creates one-frame information delay
   - Water adjustments "chase" each other

5. **Cross-chunk vertical boundaries ignore column heights** (`world.rs:233-273`)
   - Horizontal boundary processing handles falling, not leveling
   - Only vertical (left/right) boundaries have column-height logic
   - Information can't propagate up/down across chunks

## Why Previous Fixes Failed

| Attempt | Why It Failed |
|---------|---------------|
| Pressure-based | Pressure at individual cells doesn't translate to surface movement |
| Surface-only leveling | Surfaces are misidentified at chunk boundaries |
| Column height comparison | Heights are wrong due to chunk-local counting |
| Multiple boundary iterations | Can't fix fundamentally wrong height calculations |

## The Real Problem

**Cellular automata with local-only information cannot achieve global equilibrium quickly.**

Water at position (0, 0) in chunk A needs to "know" about water at position (63, 63) in chunk B to level correctly. But in pure cellular automata, information travels at most 1 cell per frame. A 128-cell wide water body takes 128+ frames to equilibrate.

## Proposed Solutions

### Option A: Global Water Level Pass (Recommended)

**Concept:** After per-cell updates, run a dedicated water leveling pass that operates on world coordinates, not chunk-local coordinates.

```rust
// In World::update(), after chunk updates:
fn level_water_globally(&mut self) {
    // 1. Find all water surface cells (world coordinates)
    // 2. For each surface cell, get TRUE column height (crossing chunks)
    // 3. Compare with neighbors (also crossing chunks)
    // 4. Move water from taller to shorter columns
}
```

**Pros:**
- Solves the information locality problem directly
- Column heights are accurate across chunks
- Leveling happens in world-space, not chunk-space

**Cons:**
- Additional pass per frame
- Need to track surface cells efficiently

**Implementation:**
1. Add `get_column_height_world(wx, wy)` that crosses chunk boundaries
2. Add `find_water_surface_world(wx, wy)` that crosses chunk boundaries
3. Run leveling pass after main update, before rendering
4. Process surface cells and move water to equalize heights

### Option B: Wave Propagation

**Concept:** Instead of local comparisons, propagate "target water level" information through connected water like a wave.

```rust
struct WaterBody {
    cells: Vec<(i32, i32)>,  // World coordinates
    target_level: i32,       // Computed equilibrium y
}
```

**Pros:**
- Physically accurate
- Handles complex shapes

**Cons:**
- Complex implementation
- Need to track connected components
- Expensive for large water bodies

### Option C: Remove Height Threshold

**Concept:** Change threshold from 2 to 1, allowing any height difference to trigger flow.

```rust
// Change from:
if my_height >= neighbor_height + 2

// To:
if my_height > neighbor_height
```

**Pros:**
- Trivial change
- Faster equilibration

**Cons:**
- May cause oscillation (water bouncing back and forth)
- Doesn't fix chunk boundary issues
- Still slow for large bodies

### Option D: Hybrid - Fix Heights + Lower Threshold

**Concept:** Combine world-coordinate height calculation with threshold of 1.

**Implementation:**
1. Make `get_column_height` work in world coordinates
2. Lower threshold to 1
3. Add anti-oscillation: track "last flow direction" and prefer continuing same direction

## Recommended Approach: Option A

Global Water Level Pass is the cleanest solution because it directly addresses the root cause.

## Implementation Plan

### Phase 1: World-Coordinate Height Functions

Add to `world.rs`:

```rust
/// Get true column height at world coordinates (crosses chunks)
pub fn get_column_height_world(&self, wx: i32, wy: i32) -> u32 {
    let mut height = 0;
    let mut y = wy;

    // Count downward until we hit non-water
    while self.get_material(wx, y) == Material::Water {
        height += 1;
        y += 1;
    }
    height
}

/// Find water surface at world coordinates (crosses chunks)
pub fn find_water_surface_world(&self, wx: i32, wy: i32) -> Option<i32> {
    if self.get_material(wx, wy) != Material::Water {
        return None;
    }

    let mut y = wy;
    while self.get_material(wx, y - 1) == Material::Water {
        y -= 1;
    }
    Some(y)
}
```

### Phase 2: Global Leveling Pass

Add to `world.rs`:

```rust
pub fn update(&mut self) {
    self.frame += 1;

    // ... existing chunk updates ...

    // NEW: Global water leveling
    self.level_water_globally();
}

fn level_water_globally(&mut self) {
    // Collect surface cells that need leveling
    let mut moves: Vec<(i32, i32, i32, i32)> = Vec::new(); // (from_x, from_y, to_x, to_y)

    // Scan visible area for water surfaces
    for (cx, cy) in self.chunks.keys().copied().collect::<Vec<_>>() {
        for lx in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                let wx = cx * CHUNK_SIZE as i32 + lx as i32;
                let wy = cy * CHUNK_SIZE as i32 + ly as i32;

                // Is this a water surface? (water with air above)
                if self.get_material(wx, wy) == Material::Water
                   && self.get_material(wx, wy - 1) != Material::Water {

                    let my_height = self.get_column_height_world(wx, wy);

                    // Check neighbors
                    for dx in [-1, 1] {
                        let nx = wx + dx;
                        let neighbor_surface = self.find_water_surface_world(nx, wy);

                        if let Some(ns) = neighbor_surface {
                            let neighbor_height = self.get_column_height_world(nx, ns);

                            // Flow from taller to shorter
                            if my_height > neighbor_height {
                                // Move surface water to above neighbor's surface
                                if self.get_material(nx, ns - 1) == Material::Air {
                                    moves.push((wx, wy, nx, ns - 1));
                                }
                            }
                        } else if self.get_material(nx, wy) == Material::Air {
                            // Neighbor is air with support - flow there
                            if self.get_material(nx, wy + 1) != Material::Air {
                                moves.push((wx, wy, nx, wy));
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply moves
    for (fx, fy, tx, ty) in moves {
        self.set_material(fx, fy, Material::Air);
        self.set_material(tx, ty, Material::Water);
    }
}
```

### Phase 3: Remove Chunk-Local Leveling

Remove or simplify the complex leveling code in `update.rs` since the global pass handles it:

- Remove `try_surface_flow` water-to-water comparison
- Keep water-to-air flow for immediate spreading
- Remove cross-chunk leveling from `process_vertical_boundary`

### Phase 4: Optimize

1. Only scan chunks that have active water
2. Use dirty flags to skip unchanged regions
3. Limit moves per frame to prevent sudden large shifts

## Implementation Status

### Attempt 1: Global Water Level Pass (Failed - Performance)
- Added world-coordinate height functions crossing chunk boundaries
- Added separate leveling pass after main update
- **Result**: Fixed leveling but tanked FPS due to O(n²) scanning

### Attempt 2: Optimized Global Pass (Failed - Still chunk artifacts)
- Only ran every 4 frames
- Sparse scanning (boundaries + every 8th row)
- Skipped inactive chunks
- **Result**: Better FPS but water still piled at chunk boundaries

### Attempt 3: World-Coordinate Simulation (Failed - Performance)
- Moved ALL simulation to world coordinates
- No chunk boundaries during simulation
- **Result**: Correct behavior but iterating all loaded chunks killed FPS

### Attempt 4: Fixed-Size World (Current - Working)
- Removed infinite world/chunking entirely for now
- Fixed 256x256 simulation area
- Single contiguous update loop
- **Result**: Water levels correctly! But only 21 FPS - needs optimization

## Key Learnings

### Why Chunking Failed for Water

1. **Information Locality Problem**: Cellular automata need global knowledge for water leveling. A cell at chunk boundary can't "see" the true column height spanning multiple chunks.

2. **Chunk boundaries create false surfaces**: At y=0 of any chunk, water was marked as "surface" even if there was water above in the previous chunk.

3. **Height calculations were chunk-local**: A 50-cell water column spanning 2 chunks reported as two 25-cell columns, making leveling impossible.

4. **Boundary processing was bolted-on**: Separate passes for boundaries couldn't fix fundamentally wrong height calculations.

### What Noita Does Differently
- Processes entire visible simulation area in ONE pass (no chunk boundaries during sim)
- Chunks are for loading/saving/rendering, NOT simulation boundaries
- Uses rigid body physics for large water bodies

### The Fundamental Tension
- **Chunking** = great for infinite worlds, terrain, powders
- **Chunking** = terrible for liquids that need global equilibrium

### Performance Reality
- 256x256 = 65,536 cells per frame = 21 FPS
- 1024x768 = 786,432 cells = 1 FPS
- Need ~10x optimization to hit 60 FPS at 256x256

## Future Optimization Ideas

### 1. Sleep Detection for Water Bodies
When water surface reaches equilibrium (all columns same height), mark the water body as "sleeping". Only wake it when:
- New water/material added
- Adjacent cell changes
- Something falls into it

### 2. Skip Static Cells
- Only process cells that moved last frame OR have a neighbor that moved
- Use dirty flags / active cell list
- Most of the world is static rock/air - shouldn't iterate it

### 3. Spatial Partitioning
- Track "active regions" instead of scanning everything
- Only simulate regions with moving particles

### 4. SIMD / Parallelization
- Process multiple cells simultaneously
- Split simulation into horizontal bands for parallel processing
- Rayon for easy parallelism

### 5. Reduce Per-Cell Work
- Cache neighbor lookups
- Avoid redundant `get_material()` calls
- Use unsafe array access where bounds are guaranteed

### 6. Water Body Tracking (Hybrid Approach)
Instead of cellular automata for water leveling:
- Track connected water bodies as entities
- Compute equilibrium level directly: `target_y = total_water_cells / body_width`
- Instantly flatten surface to target level
- Only use CA for falling/splashing, not equilibration

## Acceptance Criteria

- [ ] Water placed anywhere settles to a flat surface within ~2 seconds
- [ ] Water levels correctly across chunk boundaries (no visible steps)
- [ ] No oscillation (water doesn't bounce back and forth)
- [ ] Performance: <1ms for leveling pass with typical water amounts
- [x] Gold still sinks through water correctly (existing tests pass)

## Test Cases

1. **Single chunk pool:** Pour water in a basin within one chunk → should level flat
2. **Multi-chunk pool:** Pour water spanning 3+ chunks → should level flat across all
3. **U-tube test:** Water in U-shaped channel → should equalize levels on both sides
4. **Waterfall:** Water falling from height → should spread at bottom, not pile up
5. **Asymmetric fill:** Add water to one side of existing pool → should redistribute evenly

## References

- Noita GDC Talk: Uses pressure-based system with multi-pass equilibration
- Powder Toy: Uses simple cellular automata with fast equilibration via low threshold
- The fundamental limitation: https://www.redblobgames.com/x/1718-water-levels/

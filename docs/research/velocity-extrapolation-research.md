# Velocity Extrapolation Research

**Date:** 2025-12-28
**Purpose:** Understanding proper velocity extrapolation for FLIP simulation

## Why Velocity Extrapolation is Needed

From [WebGL-PIC-FLIP-Fluid](https://github.com/austinEng/WebGL-PIC-FLIP-Fluid):
> "In this step we extrapolate velocities to push the velocities one grid cell past the fluid cells. **This is important so that particles don't lose energy when they come in contact with air cells.**"

Without velocity extrapolation:
- Particles near fluid boundaries sample from cells with undefined/zero velocities
- The FLIP delta calculation sees phantom velocity changes
- Results in artificial momentum loss at boundaries

## The Fast Marching Method (FMM)

### Origin
Developed by James Sethian (1996) for solving the Eikonal equation. Extended by Adalsteinsson & Sethian (1999) for velocity extension in level set methods.

### Core Algorithm

From [Fast Marching Method GitHub](https://github.com/thinks/fast-marching-method):

```
1. Initialize:
   - Mark fluid cells as KNOWN with their velocities
   - Mark non-fluid neighbors as TRIAL with tentative velocities
   - Add TRIAL cells to priority queue (sorted by distance from fluid)

2. Iteration:
   - Extract cell with smallest distance from queue
   - If not finalized: mark as KNOWN
   - For each unfinalized neighbor:
     - Compute tentative velocity (average of KNOWN neighbors)
     - Add to queue

3. Continue until queue is empty or desired distance reached
```

### Key Properties
- **Monotonic propagation** - values only flow outward from fluid
- **Single-pass** - each cell processed once (O(N log N))
- **Respects distance** - closer cells get values from closer fluid cells

## Simpler Alternatives

### 1. Layered Wavefront (commonly used in games)

From [Houdini FLIP Solver](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html):
- Extrapolate N cells outward from fluid boundary
- Each layer takes average of known neighbors
- "Max Cells to Extrapolate" parameter (typically 1-3)

```rust
for layer in 0..max_layers {
    for each cell in layer {
        if cell is AIR and has FLUID/KNOWN neighbor {
            cell.velocity = average(known_neighbors)
            mark cell as KNOWN
        }
    }
}
```

### 2. Gather Operation (GPU-friendly)

From WebGL implementation:
- Each non-fluid cell checks cardinal neighbors
- Takes velocity from nearest fluid cell
- Can be done in parallel (fragment shader)

## What Our Implementation Needs

### Current State (BROKEN)
```
1. P2G
2. store_old_velocities ← samples undefined air cells
3. forces
4. boundary conditions
5. pressure
6. G2P ← samples undefined air cells
```

### Correct State
```
1. P2G
2. EXTRAPOLATE VELOCITIES ← fill air cells with valid values
3. store_old_velocities ← now samples valid velocities everywhere
4. forces
5. boundary conditions
6. pressure
7. EXTRAPOLATE VELOCITIES ← fill again after pressure changes
8. G2P ← samples valid velocities everywhere
```

## Implementation Choice: Layered Wavefront

For our 2D grid-based simulation, the layered wavefront approach is:
- Simpler to implement than full FMM
- Sufficient for typical FLIP scenarios (1-2 cell extrapolation)
- Easy to understand and debug
- Used by production solvers (Houdini)

We can upgrade to FMM later if needed for accuracy.

## References

- [Adalsteinsson & Sethian - Extension Velocities](https://math.berkeley.edu/~sethian/Papers/sethian.adalsteinsson.extension.pdf.Z)
- [Fast Marching Method - Wikipedia](https://en.wikipedia.org/wiki/Fast_marching_method)
- [WebGL-PIC-FLIP-Fluid](https://github.com/austinEng/WebGL-PIC-FLIP-Fluid)
- [Houdini FLIP Solver Docs](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)
- [Jeff Dicker's FMM Thesis](https://cmps-people.ok.ubc.ca/ylucet/thesis/2006_Honours_Jeff_Dicker.pdf)

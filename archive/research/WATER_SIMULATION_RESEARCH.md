# Mass-Based / Height-Field Water Simulation Research
## Research for Falling Sand Games with Continuous Water Flow

---

## Table of Contents
1. [The Powder Toy's Realistic Pressure Water Mode](#1-the-powder-toys-realistic-pressure-water-mode)
2. [Height-Field Water Simulations](#2-height-field-water-simulations)
3. [Mass Transfer Between Cells](#3-mass-transfer-between-cells)
4. [Combining Mass-Based Water with Particles](#4-combining-mass-based-water-with-particles)
5. [Shallow Water Equations for 2D Games](#5-shallow-water-equations-for-2d-games)
6. [Implementation Examples and Code](#6-implementation-examples-and-code)
7. [Performance Optimization Strategies](#7-performance-optimization-strategies)

---

## 1. The Powder Toy's Realistic Pressure Water Mode

### Overview
The Powder Toy implements realistic water physics through a "water equalization" mode that can be toggled in the simulation options. This mode allows water to behave more realistically (e.g., water levels equalizing in U-shaped tubes) but can cause performance issues with large amounts of water.

### Implementation Details

**Key Feature**: `water_equal_test` flag in the simulation
- Located in `Simulation.h` as `int water_equal_test = 0;`
- Activated with 1/200 probability per frame for liquid particles (Falldown == 2)
- Uses flood-fill algorithm to find and equalize water levels

**Flood Water Algorithm** (from Powder Toy source):
```cpp
bool Simulation::flood_water(int x, int y, int i)
{
    // Uses scanline flood-fill algorithm
    // 1. Creates bitmap to track visited cells (XRES * YRES)
    // 2. Uses coordinate stack to process cells
    // 3. Scans horizontally to find continuous liquid regions
    // 4. Attempts to move water upward to empty spaces above
    // 5. Prioritizes random positions to avoid clustering

    // Key check: looks for empty space above current water level
    if ((y - 1) > originalY && !pmap[y - 1][x])
    {
        // Tries random position first for better distribution
        int randPos = rng.between(x, x2);
        if (!pmap[y - 1][randPos] && eval_move(parts[i].type, randPos, y - 1, nullptr))
            x = randPos;

        move(i, originalX, originalY, float(x), float(y - 1));
        return true;
    }
}
```

**Particle Properties** (from WATR.cpp):
- Advection: 0.6
- AirDrag: 0.01 * CFDS
- AirLoss: 0.98
- Loss: 0.95 (velocity dampening)
- Gravity: 0.1
- Weight: 30
- Falldown: 2 (liquid behavior)

**Air/Pressure Simulation Modes**:
1. **Default**: Both pressure and velocity active
2. **No pressure**: Pressure doesn't affect surroundings
3. **No velocity**: Pressure stays localized
4. **Air off**: All air simulation disabled

**Air Grid Structure** (from Air.h):
```cpp
float ovx[YCELLS][XCELLS];  // Velocity X
float ovy[YCELLS][XCELLS];  // Velocity Y
float opv[YCELLS][XCELLS];  // Pressure
float ohv[YCELLS][XCELLS];  // Ambient Heat
```

### Sources
- [The Powder Toy GitHub](https://github.com/The-Powder-Toy/The-Powder-Toy)
- [Water simulation discussion](https://powdertoy.co.uk/Discussions/Thread/View.html?Thread=2498&PageNum=1#Message=262881)

---

## 2. Height-Field Water Simulations

### Virtual Pipes Method (Recommended for Games)

The virtual pipes method is one of the most efficient approaches for real-time water simulation in games. It operates on a staggered grid storing water heights and flow rates.

**Data Structure**:
For an N×N grid:
- Water height: N×N array
- Horizontal flow (X-direction): (N+1)×N array
- Vertical flow (Y-direction): N×(N+1) array

**Flow represents volume per unit time**, avoiding division-by-zero issues with velocity-based approaches.

### Three-Step Algorithm

#### Step 1: Flow Acceleration
Apply gravitational force based on water surface height differences:

```
For each interior horizontal edge:
  flowX(x,y) = flowX(x,y) × friction_factor
             + [water(x-1,y) + terrain(x-1,y)
                - water(x,y) - terrain(x,y)] × g × dt / dx

For each interior vertical edge:
  flowY(x,y) = flowY(x,y) × friction_factor
             + [water(x,y-1) + terrain(x,y-1)
                - water(x,y) - terrain(x,y)] × g × dt / dy
```

Where `friction_factor = (1 - friction)^dt` for time-independent damping.

#### Step 2: Outflow Scaling
Prevent negative water amounts by capping outgoing flows:

Calculate total outflow leaving each cell from all four adjacent edges. If this exceeds available water:
- Compute `scaling_factor = min(1.0, max_available_outflow / total_outflow)`
- Apply this factor only to edges with outward flow

#### Step 3: Water Column Update
Transport water between cells according to computed flows:

```
For each water cell:
  water(x,y) += [flowX(x,y) + flowY(x,y)
                 - flowX(x+1,y) - flowY(x,y+1)] × dt / (dx × dy)
```

### Key Advantages
- **Stable**: Never produces negative water amounts
- **Fast**: Simple operations on regular grid
- **Mass conserving**: Flow-based approach preserves total water volume
- **GPU-friendly**: Each cell can be updated in parallel

### Extended Virtual Pipes
Research has extended the basic method with:
- Multi-layered heightmaps for complex scenarios
- Flows through fully flooded passages
- Physically-based viscosity models

### Implementation Reference
- [lisyarus blog: Simulating water over terrain](https://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html)
- [webgpu-shallow-water GitHub](https://github.com/lisyarus/webgpu-shallow-water)
- [Extended virtual pipes paper](https://www.researchgate.net/publication/327659939_Extended_virtual_pipes_for_the_stable_and_real-time_simulation_of_small-scale_shallow_water)

### Matthias Müller-Fischer's Height Field Method

**Game Requirements**:
- Cheap computation (small fraction of 15ms frame time)
- Stable with kinematic, fast-moving objects
- Low memory consumption

**Optimization Strategies**:
1. Resolution reduction (trade visual quality for speed)
2. Dimension reduction (3D → 2D)
3. Low-res physics with hi-res appearance (shaders)
4. Simulate only active/visible regions (sleeping)
5. Level of detail (LOD)

**Pipe Method Characteristics**:
- Assumes hydrostatic pressure is direct function of height
- Flow only in vertical and horizontal directions to neighbors
- 100% GPU simulation possible
- Can reduce fluid calculation to half resolution while keeping heightmap at full resolution

### Sources
- [Fast Water Simulation for Games Using Height Fields (GDC)](https://matthias-research.github.io/pages/publications/hfFluid.pdf)
- [GitHub: Height-Field-Water](https://github.com/ucanbizon/Height-Field-Water)

---

## 3. Mass Transfer Between Cells

### Cellular Automata Mass-Based Fluid

This approach from W-Shadow.com provides a simple yet effective mass-based simulation that creates smooth fluid behavior without explicit pressure calculations.

### Core Concept
Water is treated as "slightly compressible liquid" - cells can store excess water proportional to the cell above them, enabling natural equalization.

### Key Parameters

```cpp
MaxMass = 1.0           // Standard full-cell capacity
MaxCompress = 0.02      // Additional storage multiplier per cell depth
MinMass = 0.0001        // Threshold for marking cells as dry
```

### Mass Distribution Formula

The `get_stable_state_b()` function calculates equilibrium distribution:

**Three conditions:**

1. **If total ≤ 1.0**: Lower cell receives 1.0
2. **If 1.0 < total < 2.02**:
   ```
   result = (MaxMass² + total×MaxCompress) / (MaxMass + MaxCompress)
   ```
3. **If total ≥ 2.02**:
   ```
   result = (total + MaxCompress) / 2
   ```

### Flow Algorithm (Per Cell, Per Step)

Process flows in sequence with dual-array approach to avoid order dependencies:

```
1. Downward Flow:
   - Transfer mass to cell below until stable state reached

2. Leftward Flow:
   - Equalize with left neighbor (difference ÷ 4)

3. Rightward Flow:
   - Equalize with right neighbor (difference ÷ 4)

4. Upward Flow:
   - Only compressed water flows up (excess over MaxMass)

Each flow:
- Constrained by remaining mass in source cell
- Applied with 0.5× damping factor for stability
- Written to new_mass array
```

### Implementation Pattern

```cpp
// Dual array approach
float mass[GRID_SIZE];
float new_mass[GRID_SIZE];

void simulate_step() {
    // Calculate all flows for all cells
    for (each cell) {
        calculate_flows_to_neighbors();
        write_to_new_mass_array();
    }

    // Swap arrays
    for (each cell) {
        mass[i] = new_mass[i];
        if (mass[i] < MinMass) {
            convert_to_air();
        }
    }
}
```

### Advantages & Limitations

**Advantages**:
- Simple to implement
- Good visual quality for games
- Fast computation
- Natural-looking flow

**Limitations**:
- Water can form "hills" that spread slowly (unrealistic)
- Not physically accurate
- Issues with momentum conservation

### Source
- [Simple Fluid Simulation With Cellular Automata](https://w-shadow.com/blog/2009/09/01/simple-fluid-simulation/)

---

## 4. Combining Mass-Based Water with Particles

### Hybrid Approaches

Multiple successful approaches exist for combining grid-based fluids with particle-based materials:

### 1. PIC/FLIP Method (Industry Standard)

**Particle-in-Cell (PIC)** and **Fluid-Implicit-Particle (FLIP)** are the de facto standards in industry (used by Houdini, Naiad).

**Core Concept**: Particles represent fluid and carry material properties, while a background grid handles pressure/incompressibility.

**Simulation Loop**:
```
1. Initialize grid and particle positions/velocities
2. Transfer particle velocities to staggered grid (weighted average)
3. [FLIP only] Save copy of grid velocities
4. Calculate and apply external forces on grid
5. Project velocities (enforce incompressibility) on grid
6. Enforce boundary conditions on grid
7. Transfer grid velocities back to particles:
   - PIC: particles = new grid velocity
   - FLIP: particles += (new grid velocity - old grid velocity)
8. Advect particles through velocity field
```

**Particle to Grid Transfer**:
```cpp
// For each grid point, weighted average of nearby particles
// "Nearby" determined by bilinear/trilinear hat function

for (each grid point g) {
    float total_weight = 0;
    vec2 velocity_sum = 0;

    for (each particle p near g) {
        float weight = hat_function(distance(p, g));
        velocity_sum += p.velocity * weight;
        total_weight += weight;
    }

    if (total_weight > 0) {
        g.velocity = velocity_sum / total_weight;
    }
}
```

**Grid to Particle Transfer**:
```cpp
// Simple trilinear interpolation
for (each particle p) {
    vec2 grid_pos = p.position / grid_spacing;
    p.velocity = trilinear_interpolate(grid.velocity, grid_pos);

    // For FLIP: use velocity delta instead
    // p.velocity += (new_grid_velocity - old_grid_velocity)
}
```

**PIC vs FLIP Blending**:
- Pure PIC: Smooth but overly viscous (numerical dissipation)
- Pure FLIP: Preserves detail but can become noisy/chaotic
- **Recommended**: 95% FLIP + 5% PIC blend for stability

### 2. Affine Particle-in-Cell (APIC)

**Improvement over PIC/FLIP**: Removes dissipation while maintaining stability.

**Key Innovation**: Augment each particle with locally affine (rather than constant) velocity description.

**Benefits**:
- Exact conservation of angular momentum
- No dissipation from grid transfers
- More stable than pure FLIP

### 3. Material Point Method (MPM)

**Used for**: Sand, water, mud, and multi-species simulations
**Famous use**: Snow simulation in Disney's Frozen (2013)

**Concept**:
- Particles represent material and carry properties
- Background Eulerian grid for computations
- Material points surrounded by grid for calculating deformation gradients

**Multi-Species Approach**:
- Extend grid degrees of freedom to store info for multiple materials
- Mixture model captures interaction and relative motion
- Unified framework for coupled fluids and solids

**Sand and Water Mixtures**:
- Water: modeled as weakly compressible fluid
- Sand: elastoplastic law with cohesion varying by water saturation
- MPM discretizes governing equations

**Notable Implementations**:
- Drucker-Prager elastoplasticity for sand animation
- Multi-species porous sand and water mixtures

### 4. Sandspiel Approach (Practical Game Implementation)

**Architecture**:
- Fluid simulation on GPU (WebGL Navier-Stokes)
- Particles on CPU (cellular automata)
- Bidirectional data transfer (WebAssembly ↔ WebGL)

**Material Interactions**:
- Dirt absorbs water → becomes mud
- State transitions through `update_*` functions per frame
- Extra cell state in registers (`ra`, `rb`) for properties

**Key Challenge**: Enabling interactions between GPU-based wind and CPU-based particles

### 5. Direct Coupling with Immersed Boundary

**DNS (Direct Numerical Simulation) Approach**:
- Fixed Eulerian grid for fluid equations
- Ghost-cell based immersed boundary method for particles
- Implicit boundary condition incorporation
- Handles fluid-solid coupling efficiently

### Implementation Strategies

**For Falling Sand Games**, consider these hybrid approaches:

#### Option A: Separate Systems with Interaction Layer
```
Grid Layer (water):
  - Height-field or cellular automata
  - Mass-based fluid simulation

Particle Layer (sand, dirt, gold):
  - Individual particle simulation
  - Cellular automata for simple materials

Interaction:
  - Particles displace water (reduce water level)
  - Water exerts force on particles (velocity field)
  - State transitions: dirt + water → mud
```

#### Option B: Unified Particle System with Grid Acceleration
```
All materials as particles:
  - Water particles with fluid properties
  - Solid particles with different properties

Background grid:
  - Bins particles for collision detection
  - Computes pressure/velocity for water particles only
  - Applies forces back to particles
```

#### Option C: MPM-Style Hybrid
```
Materials as particles:
  - Carry position, velocity, mass

Grid for computation:
  - Transfer particle data to grid
  - Solve momentum equations on grid
  - Transfer results back to particles
  - Advect particles
```

### Sources
- [Fluid Simulation For Computer Graphics Tutorial](https://cg.informatik.uni-freiburg.de/intern/seminar/gridFluids_fluid-EulerParticle.pdf)
- [Particle-Based Fluid Simulation (Matthias Müller)](https://matthias-research.github.io/pages/publications/sca03.pdf)
- [FLIP Water Simulator (Matthias Müller)](https://matthias-research.github.io/pages/tenMinutePhysics/18-flip.pdf)
- [Multi-species simulation MPM](https://www.researchgate.net/publication/318612741_Multi-species_simulation_of_porous_sand_and_water_mixtures)
- [Making Sandspiel](https://maxbittker.com/making-sandspiel/)

---

## 5. Shallow Water Equations for 2D Games

### Mathematical Foundation

**Shallow Water Equations (SWE)** are derived from depth-integrating the Navier-Stokes equations when horizontal length scale >> vertical length scale.

**Key Assumption**: Pressure distribution is hydrostatic over flow depth (valid for long, shallow waves).

### What SWE Capture

Beyond simple height fields, SWE describe evolution of a **2D velocity field** normal to water columns, enabling:
- Wave propagation
- Momentum transport
- Realistic flow patterns
- Height and velocity coupling

### SWE System

**Conservation of Mass**:
```
∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
```

**Conservation of Momentum (X)**:
```
∂(hu)/∂t + ∂(hu²)/∂x + ∂(huv)/∂y = -gh ∂(h+b)/∂x + friction_x
```

**Conservation of Momentum (Y)**:
```
∂(hv)/∂t + ∂(huv)/∂x + ∂(hv²)/∂y = -gh ∂(h+b)/∂y + friction_y
```

Where:
- h = water height
- u, v = velocity components
- b = bed/terrain elevation
- g = gravitational acceleration

### Numerical Methods for SWE

#### 1. Finite Volume Method (High Accuracy)

**Approach**:
- Discretize conservation equations using finite volumes
- Use Roe's approximate Riemann solver for fluxes
- Multidimensional limiter for second-order spatial accuracy
- Predictor-corrector time stepping for second-order temporal accuracy

**Advantages**:
- Perfect mass conservation
- Handles wet/dry fronts robustly
- Second-order accuracy possible
- Works on unstructured grids (triangular cells)

**Typical Mass Conservation**: < 2% error, can achieve < 0.003% with tuning

#### 2. SPH-Based Shallow Water

**Instead of grid cells**: Use 2D SPH particles

**Benefits**:
- Height computed from particle density
- Natural mass conservation (particles = mass)
- No grid artifacts
- Easy to handle free surfaces

**Computation**:
```cpp
// For each particle
float density = 0;
for (each neighbor particle j) {
    density += mass[j] * kernel(distance(i, j));
}
height[i] = density / reference_density;
```

#### 3. Predictor-Corrector Scheme

For **height field** with **velocity field** on MAC grid:

**Predictor**:
```
h* = h_old - dt × h × ∇·v
```

**Conservative Advection**:
Use upwinding scheme on MAC grid to advect both height and velocity.

**Corrector**:
Update height and velocity using corrected fluxes.

### Practical Game Implementation

#### Simple Approach for Real-Time Games

**Reduce to height + velocity per cell**:

```cpp
struct WaterCell {
    float height;          // Water column height
    float velocity_x;      // Horizontal velocity
    float velocity_y;      // Vertical velocity
    float terrain;         // Bed elevation (static)
};

void simulate_step(float dt) {
    // 1. Update velocities from pressure gradient
    for (each cell) {
        float surface_x_left = water[x-1].height + water[x-1].terrain;
        float surface_x_right = water[x+1].height + water[x+1].terrain;
        float surface_y_up = water[y-1].height + water[y-1].terrain;
        float surface_y_down = water[y+1].height + water[y+1].terrain;

        velocity_x += -g * (surface_x_right - surface_x_left) / (2*dx) * dt;
        velocity_y += -g * (surface_y_down - surface_y_up) / (2*dy) * dt;

        // Apply friction
        velocity_x *= friction_factor;
        velocity_y *= friction_factor;
    }

    // 2. Update heights from divergence
    for (each cell) {
        float flux_x = (velocity_x - water[x-1].velocity_x) / dx;
        float flux_y = (velocity_y - water[y-1].velocity_y) / dy;

        height += -height * (flux_x + flux_y) * dt;
    }
}
```

### Comparison with Virtual Pipes Method

**Virtual Pipes**:
- Stores **flows** between cells (volume/time)
- Three steps: accelerate, scale, update
- Very stable and simple

**Shallow Water Equations**:
- Stores **velocities** in cells
- Full momentum conservation
- More physically accurate
- Can be less stable (requires smaller timesteps)

**Recommendation**:
- **Virtual Pipes** for simple, stable water simulation
- **Full SWE** for realistic wave propagation and momentum effects

### Falling Sand Game Adaptation

**W-Shadow.com Tutorial**: Simplified approach for "falling sand" style water

**Key Points**:
- Each array element = single pixel
- Much simpler than full SWE
- Faster than incompressible fluids
- Adequate visual quality for sand games

**Typical Implementation**:
```cpp
// Simplified "falling sand" liquid
if (is_liquid(cell)) {
    // Try to fall
    if (empty(below)) move_down();
    // Try to spread
    else if (empty(below_left) || empty(below_right))
        move_diagonal();
    // Try to flow sideways
    else if (empty(left) || empty(right))
        move_horizontal();
}
```

**Enhancement with Mass**:
Combine falling sand rules with mass-based equalization (see Section 3).

### Sources
- [Shallow water equations - Wikipedia](https://en.wikipedia.org/wiki/Shallow_water_equations)
- [W-Shadow: Falling Sand Style Water Simulation](https://w-shadow.com/blog/2009/09/29/falling-sand-style-water-simulation/)
- [Real-time Height-field Simulation of Sand and Water](https://kuiwuchn.github.io/RTWaterAndSand.pdf)
- [Finite Volume SWE Algorithm](https://www.researchgate.net/publication/228624466_Unstructured_Grid_Finite-Volume_Algorithm_for_Shallow-Water_Flow_and_Scalar_Transport_with_Wetting_and_Drying)

---

## 6. Implementation Examples and Code

### Complete Cellular Automata Water Simulator

Based on W-Shadow.com's implementation:

```cpp
class WaterSimulator {
    static constexpr float MaxMass = 1.0f;
    static constexpr float MaxCompress = 0.02f;
    static constexpr float MinMass = 0.0001f;
    static constexpr float MaxSpeed = 1.0f;

    int width, height;
    std::vector<float> mass;
    std::vector<float> new_mass;

    float get_stable_state_b(float total_mass) {
        if (total_mass <= 1.0f) {
            return 1.0f;
        } else if (total_mass < 2.0f * MaxMass + MaxCompress) {
            return (MaxMass * MaxMass + total_mass * MaxCompress) /
                   (MaxMass + MaxCompress);
        } else {
            return (total_mass + MaxCompress) / 2.0f;
        }
    }

    float constrain_flow(float flow, float source_mass) {
        return std::max(0.0f, std::min(flow, source_mass));
    }

    void simulate_compression() {
        // Reset new mass
        std::fill(new_mass.begin(), new_mass.end(), 0.0f);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                float m = mass[idx];

                if (m <= 0) continue;

                float flow = 0;
                float remaining_mass = m;

                // Flow down
                if (y < height - 1) {
                    int below = (y + 1) * width + x;
                    float stable_state = get_stable_state_b(m + mass[below]);
                    flow = constrain_flow(stable_state - mass[below], remaining_mass);
                    flow *= 0.5f; // Dampening

                    new_mass[idx] -= flow;
                    new_mass[below] += flow;
                    remaining_mass -= flow;
                }

                // Flow left
                if (x > 0 && remaining_mass > 0) {
                    int left = y * width + (x - 1);
                    flow = (remaining_mass - mass[left]) / 4.0f;
                    flow = constrain_flow(flow, remaining_mass);
                    flow *= 0.5f;

                    new_mass[idx] -= flow;
                    new_mass[left] += flow;
                    remaining_mass -= flow;
                }

                // Flow right
                if (x < width - 1 && remaining_mass > 0) {
                    int right = y * width + (x + 1);
                    flow = (remaining_mass - mass[right]) / 4.0f;
                    flow = constrain_flow(flow, remaining_mass);
                    flow *= 0.5f;

                    new_mass[idx] -= flow;
                    new_mass[right] += flow;
                    remaining_mass -= flow;
                }

                // Flow up (only compressed water)
                if (y > 0 && remaining_mass > 0) {
                    int above = (y - 1) * width + x;
                    flow = remaining_mass - get_stable_state_b(remaining_mass + mass[above]);
                    flow = constrain_flow(flow, remaining_mass);
                    flow *= 0.5f;

                    new_mass[idx] -= flow;
                    new_mass[above] += flow;
                }
            }
        }

        // Apply new masses and remove near-zero masses
        for (int i = 0; i < mass.size(); i++) {
            mass[i] += new_mass[i];
            if (mass[i] < MinMass) {
                mass[i] = 0;
            }
        }
    }
};
```

### Virtual Pipes Water Simulator

Based on lisyarus blog implementation:

```cpp
class VirtualPipesSimulator {
    int width, height;
    std::vector<float> water;      // N×N
    std::vector<float> terrain;    // N×N (static)
    std::vector<float> flow_x;     // (N+1)×N
    std::vector<float> flow_y;     // N×(N+1)

    float g = 9.81f;               // Gravity
    float friction = 0.1f;         // Friction coefficient
    float dt = 0.016f;             // Time step
    float dx = 1.0f;               // Grid spacing

    void simulate_step() {
        // Step 1: Flow acceleration
        float friction_factor = std::pow(1.0f - friction, dt);

        // Horizontal flows
        for (int y = 0; y < height; y++) {
            for (int x = 1; x < width; x++) {
                int idx = y * (width + 1) + x;
                int left = y * width + (x - 1);
                int right = y * width + x;

                float surface_left = water[left] + terrain[left];
                float surface_right = water[right] + terrain[right];

                flow_x[idx] *= friction_factor;
                flow_x[idx] += (surface_left - surface_right) * g * dt / dx;
            }
        }

        // Vertical flows
        for (int y = 1; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                int above = (y - 1) * width + x;
                int below = y * width + x;

                float surface_above = water[above] + terrain[above];
                float surface_below = water[below] + terrain[below];

                flow_y[idx] *= friction_factor;
                flow_y[idx] += (surface_above - surface_below) * g * dt / dx;
            }
        }

        // Step 2: Outflow scaling
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                float outflow = 0;
                if (flow_x[y * (width + 1) + x] < 0)
                    outflow += -flow_x[y * (width + 1) + x];
                if (flow_x[y * (width + 1) + x + 1] > 0)
                    outflow += flow_x[y * (width + 1) + x + 1];
                if (flow_y[y * width + x] < 0)
                    outflow += -flow_y[y * width + x];
                if (flow_y[(y + 1) * width + x] > 0)
                    outflow += flow_y[(y + 1) * width + x];

                float max_outflow = water[idx] * dx * dx / dt;
                if (outflow > max_outflow && outflow > 0) {
                    float scale = max_outflow / outflow;

                    if (flow_x[y * (width + 1) + x] < 0)
                        flow_x[y * (width + 1) + x] *= scale;
                    if (flow_x[y * (width + 1) + x + 1] > 0)
                        flow_x[y * (width + 1) + x + 1] *= scale;
                    if (flow_y[y * width + x] < 0)
                        flow_y[y * width + x] *= scale;
                    if (flow_y[(y + 1) * width + x] > 0)
                        flow_y[(y + 1) * width + x] *= scale;
                }
            }
        }

        // Step 3: Water update
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;

                float flux = flow_x[y * (width + 1) + x] +
                            flow_y[y * width + x] -
                            flow_x[y * (width + 1) + x + 1] -
                            flow_y[(y + 1) * width + x];

                water[idx] += flux * dt / (dx * dx);
                water[idx] = std::max(0.0f, water[idx]);
            }
        }
    }
};
```

### Jos Stam's Stable Fluids (Grid-Based Incompressible)

Based on Karl Sims tutorial and Jos Stam's paper:

```cpp
class StableFluids {
    int N;  // Grid size
    std::vector<float> vx, vy;      // Velocity field
    std::vector<float> vx0, vy0;    // Previous velocity
    std::vector<float> density;     // Density/dye for visualization
    std::vector<float> density0;

    float dt = 0.1f;
    float diff = 0.0001f;  // Diffusion
    float visc = 0.0001f;  // Viscosity

    void simulate_step() {
        // 1. Add forces
        add_forces();

        // 2. Velocity diffusion
        diffuse(vx0, vx, visc);
        diffuse(vy0, vy, visc);

        // 3. Project (remove divergence)
        project(vx0, vy0);

        // 4. Advect velocity
        advect(vx, vx0, vx0, vy0);
        advect(vy, vy0, vx0, vy0);

        // 5. Project again
        project(vx, vy);

        // 6. Advect density
        diffuse(density0, density, diff);
        advect(density, density0, vx, vy);
    }

    void diffuse(std::vector<float>& x, const std::vector<float>& x0, float diff) {
        float a = dt * diff * N * N;

        // Gauss-Seidel relaxation (20 iterations)
        for (int k = 0; k < 20; k++) {
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    int idx = i * (N + 2) + j;
                    x[idx] = (x0[idx] + a * (
                        x[idx - 1] + x[idx + 1] +
                        x[idx - (N + 2)] + x[idx + (N + 2)]
                    )) / (1 + 4 * a);
                }
            }
        }
    }

    void advect(std::vector<float>& d, const std::vector<float>& d0,
                const std::vector<float>& u, const std::vector<float>& v) {
        float dt0 = dt * N;

        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                int idx = i * (N + 2) + j;

                // Trace particle backwards
                float x = i - dt0 * u[idx];
                float y = j - dt0 * v[idx];

                // Clamp to grid
                x = std::clamp(x, 0.5f, N + 0.5f);
                y = std::clamp(y, 0.5f, N + 0.5f);

                // Bilinear interpolation
                int i0 = (int)x;
                int j0 = (int)y;
                int i1 = i0 + 1;
                int j1 = j0 + 1;

                float s1 = x - i0;
                float s0 = 1 - s1;
                float t1 = y - j0;
                float t0 = 1 - t1;

                d[idx] = s0 * (t0 * d0[i0 * (N + 2) + j0] +
                              t1 * d0[i0 * (N + 2) + j1]) +
                        s1 * (t0 * d0[i1 * (N + 2) + j0] +
                              t1 * d0[i1 * (N + 2) + j1]);
            }
        }
    }

    void project(std::vector<float>& u, std::vector<float>& v) {
        std::vector<float> div(N * N);
        std::vector<float> p(N * N);

        // Compute divergence
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                int idx = i * (N + 2) + j;
                div[idx] = -0.5f * (
                    u[idx + 1] - u[idx - 1] +
                    v[idx + (N + 2)] - v[idx - (N + 2)]
                ) / N;
                p[idx] = 0;
            }
        }

        // Solve for pressure (Gauss-Seidel)
        for (int k = 0; k < 20; k++) {
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {
                    int idx = i * (N + 2) + j;
                    p[idx] = (div[idx] +
                             p[idx - 1] + p[idx + 1] +
                             p[idx - (N + 2)] + p[idx + (N + 2)]) / 4.0f;
                }
            }
        }

        // Subtract pressure gradient
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                int idx = i * (N + 2) + j;
                u[idx] -= 0.5f * N * (p[idx + 1] - p[idx - 1]);
                v[idx] -= 0.5f * N * (p[idx + (N + 2)] - p[idx - (N + 2)]);
            }
        }
    }
};
```

### Sources
- [W-Shadow: Simple Fluid Simulation](https://w-shadow.com/blog/2009/09/01/simple-fluid-simulation/)
- [lisyarus: Simulating water over terrain](https://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html)
- [Karl Sims: Fluid Flow](https://www.karlsims.com/fluid-flow.html)

---

## 7. Performance Optimization Strategies

### Noita's Approach (Production Game)

Noita simulates every pixel using complex cellular automata at scale.

#### Update Order
**Critical**: Update world **bottom-up** for falling sand
- If updating top-down, only bottom pixels fall
- For gases/steam: invert (update top-down, rise first)

#### Chunking System
```
World divided into 64×64 chunks
Each chunk maintains:
  - Dirty rect for pixels needing update
  - Active flag
  - Last update frame counter

Only iterate dirty rect pixels (huge performance gain)
```

#### Multi-threading Strategy
```
Problem: Pixels can cross chunk boundaries
Solution: Per-pixel frame counter

if (pixel.last_update_frame == current_frame) {
    skip; // Already updated this frame
}

Use atomics or per-thread local updates with merge step
```

#### Material-Specific Optimizations
- Static materials (stone): no update rules applied
- Dynamic materials (water, sand): active chunks only
- Fire/burning: temporary state, destroys pixel after timer
- Explosion response: convert static to "collapsing sand" on large explosions

#### Rigid Bodies Integration
```
Each pixel in rigid body:
  - Knows it belongs to rigid body
  - Stores location within body

Rigid bodies use Box2D physics
Can deal collision damage when thrown
```

### General Optimization Techniques

#### 1. Spatial Subdivision
```cpp
// Divide world into active regions
struct ActiveRegion {
    int x0, y0, x1, y1;  // Bounding box
    bool has_moving_water;
    int frames_idle;
};

// Only simulate regions with activity
// Put idle regions to "sleep" after threshold
```

#### 2. Level of Detail (LOD)
```cpp
// Reduce simulation resolution based on distance from camera/player
if (distance_to_camera > far_threshold) {
    simulate_at_quarter_resolution();
} else if (distance_to_camera > mid_threshold) {
    simulate_at_half_resolution();
} else {
    simulate_at_full_resolution();
}
```

#### 3. GPU Acceleration
```glsl
// Compute shader for water simulation
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform sampler2D water_in;
layout(binding = 1) uniform writeonly image2D water_out;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);

    // Perform virtual pipes step
    // All cells update in parallel
    // Write to water_out
}
```

#### 4. Multi-Grid for Pressure Solve
```cpp
// Solve pressure at multiple resolutions
// Most work at low resolution, refine at high resolution

void multi_grid_pressure_solve() {
    // 1. Restrict to coarse grid (downsample)
    restrict_to_coarse();

    // 2. Solve at coarse level
    solve_pressure_coarse();

    // 3. Prolongate to fine grid (upsample)
    prolongate_to_fine();

    // 4. Refine with few iterations at fine level
    refine_pressure_fine();
}
```

#### 5. Sleeping Regions
```cpp
// Track fluid motion
struct Cell {
    float water;
    float velocity;
    bool sleeping;
};

void update() {
    for (each cell) {
        if (cell.sleeping) {
            if (neighbors_changed()) {
                cell.sleeping = false;
            } else {
                continue; // Skip update
            }
        }

        simulate_cell();

        if (cell.velocity < threshold && !changing_for_N_frames) {
            cell.sleeping = true;
        }
    }
}
```

#### 6. Conservative Advection with Reduced Resolution
```cpp
// Physics at low resolution
simulate_water_128x128();

// Render at high resolution with interpolation
render_water_512x512_with_shader_interpolation();
```

### Benchmark Targets

**From research**:
- 100 FPS achievable for 1763 cells with CPU cellular automata (with parallelization)
- GPU approaches can handle 100,000+ cells at 60 FPS
- Virtual pipes method: real-time at game resolutions (tested at 512×512+)
- Noita: Full screen pixel simulation at 60 FPS (with optimizations)

### Sources
- [Noita: GDC Talk on Falling Sand Simulation](https://www.gdcvault.com/play/1025695/Exploring-the-Tech-and-Design/)
- [80.lv: Noita Interview](https://80.lv/articles/noita-a-game-based-on-falling-sand-simulation)

---

## Summary and Recommendations

### For Continuous Water Flow in Falling Sand Games

**Best Approach**: Combine techniques for optimal results

#### Recommended Hybrid System:

1. **Water Simulation**: Virtual Pipes Method
   - Simple, stable, fast
   - Produces continuous flow
   - Mass conserving
   - Easy GPU implementation

2. **Particle Materials**: Cellular Automata
   - Sand, dirt, gold: simple falling rules
   - Efficient for discrete materials

3. **Interaction Layer**:
   - Particles displace water (reduce water height)
   - Water velocity affects particle motion
   - State transitions (dirt + water → mud)

#### Implementation Roadmap:

```
Phase 1: Core Water Simulation
- Implement virtual pipes method
- 2D grid with height + flow arrays
- Basic rendering

Phase 2: Particle System
- Basic falling sand CA
- Multiple material types
- Collision with terrain

Phase 3: Water-Particle Interaction
- Particles reduce water level in cell
- Water flow velocity pushes particles
- Buoyancy for light particles

Phase 4: Advanced Features
- Mud/wet soil mechanics
- Erosion (water + soil → suspended sediment)
- Evaporation/condensation
- GPU acceleration

Phase 5: Optimization
- Chunking system (64×64 or 128×128)
- Sleeping regions
- Multi-threading
- LOD based on camera distance
```

### Key Takeaways

1. **Virtual Pipes** is the sweet spot for games: stable, fast, realistic enough
2. **Cellular Automata** with mass works well for simple water, easier to implement
3. **PIC/FLIP** is industry standard for high-quality fluid but more complex
4. **Shallow Water Equations** provide physics accuracy if needed
5. **Chunking + Dirty Rects** are essential for large worlds
6. **GPU acceleration** enables much larger scales

### Further Reading

**Essential Papers**:
- Matthias Müller-Fischer: "Fast Water Simulation for Games Using Height Fields"
- Jos Stam: "Real-Time Fluid Dynamics for Games"
- Zhu & Bridson: "Animating Sand as a Fluid"

**Open Source Examples**:
- The Powder Toy (C++): Full falling sand game
- lisyarus/webgpu-shallow-water: Virtual pipes reference
- Sandspiel: Modern web-based implementation

**Tutorials**:
- W-Shadow.com: Practical CA fluid simulation
- Karl Sims: Incompressible flow tutorial
- lisyarus blog: Virtual pipes detailed walkthrough

---

## Additional Resources

### GitHub Repositories
- [The-Powder-Toy/The-Powder-Toy](https://github.com/The-Powder-Toy/The-Powder-Toy) - Production falling sand game
- [lisyarus/webgpu-shallow-water](https://github.com/lisyarus/webgpu-shallow-water) - Virtual pipes GPU implementation
- [austinEng/WebGL-PIC-FLIP-Fluid](https://github.com/austinEng/WebGL-PIC-FLIP-Fluid) - PIC/FLIP reference
- [ucanbizon/Height-Field-Water](https://github.com/ucanbizon/Height-Field-Water) - Height field demo

### Academic Papers
- [Real-time Simulation of Large Bodies of Water](https://matthias-research.github.io/pages/publications/hfFluid.pdf)
- [Multi-species simulation of porous sand and water mixtures](https://www.researchgate.net/publication/318612741_Multi-species_simulation_of_porous_sand_and_water_mixtures)
- [Extended virtual pipes](https://www.researchgate.net/publication/327659939_Extended_virtual_pipes_for_the_stable_and_real_time_simulation_of_small-scale_shallow_water)

### Online Tutorials
- [W-Shadow: Simple Fluid Simulation With Cellular Automata](https://w-shadow.com/blog/2009/09/01/simple-fluid-simulation/)
- [W-Shadow: Falling Sand Style Water Simulation](https://w-shadow.com/blog/2009/09/29/falling-sand-style-water-simulation/)
- [Karl Sims: Fluid Flow](https://www.karlsims.com/fluid-flow.html)
- [lisyarus: Simulating water over terrain](https://lisyarus.github.io/blog/posts/simulating-water-over-terrain.html)
- [Making Sandspiel](https://maxbittker.com/making-sandspiel/)

---

*Research compiled December 2025*
*For: Goldrush Fluid Miner - Mass-based water simulation with particle materials*

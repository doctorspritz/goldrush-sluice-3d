# Grid Resolution Best Practices for FLIP/PIC Fluid Simulations

**Research Date:** 2025-12-21
**Focus:** Practical implementation guidance for optimal grid resolution choices in FLIP/PIC simulations

---

## Executive Summary

This document synthesizes current best practices for grid resolution in FLIP/PIC fluid simulations, with emphasis on:
- Particles per cell ratios (8 particles optimal based on Nyquist criterion)
- Grid resolution requirements for vortex formation (2x higher than particle methods)
- Pressure solver convergence optimization (multigrid methods provide 2-3x speedup)
- Performance tradeoffs between resolution and computation time
- Adaptive grid techniques for memory-constrained environments

---

## 1. Recommended Grid Resolutions for Vortex Formation and Capture

### Vortex Formation Requirements

**Key Finding:** Vortices represent potentially very high resolution velocity gradients, which implies that a grid-based solution would require **twice or more the resolution in each direction** compared to particle-based methods.

**Source:** [VorteGrid: Interactive Fluid Simulation](https://github.com/mijagourlay/VorteGrid)

### Specific Resolution Guidelines

For high Reynolds number flows (e.g., flow around cylinders at Re = 22,000):

- **Spanwise resolution ratio:** `dz/D ≈ 0.04` is suggested as sufficient for prediction of:
  - Mean and RMS aerodynamic pressures
  - Mean and RMS velocity fields
  - Small-scale turbulent motions in the wake

  Where `dz` is grid spacing and `D` is the characteristic dimension (cylinder width)

- **Effects of coarser resolution:**
  - Longer vortex formation length behind obstacles
  - Smaller fluctuating velocities around obstacles (except spanwise component)

**Source:** [Spanwise resolution requirements for high-Reynolds-number flows](https://www.sciencedirect.com/science/article/abs/pii/S0045793019302804)

### Practical Resolution Examples

For general FLIP simulations:

- **Low quality (fast):** 64 x 64 x 64 grid with cell size 0.125
- **High quality:** 128 x 128 x 128 grid with cell size 0.0625
- **Very high quality:** Continue doubling resolution while halving cell size

**Source:** [Blender FLIP Fluids Wiki - Domain Simulation Settings](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-Simulation-Settings)

### Alternative: Vortex Particle Methods

For capturing fine vortex details, consider hybrid approaches:

- **Vortex Particle Flow Maps (VPFM):** Combine grid-based velocity reconstruction with particle-based vorticity evolution
- **Advantage:** Tracks vortons (vortex particles) to avoid numerical viscosity that erodes detail in pure grid-based methods
- **Use case:** Sub-grid turbulence and procedural synthesis techniques

**Sources:**
- [Fluid Simulation on Vortex Particle Flow Maps](https://arxiv.org/abs/2505.21946)
- [Procedural Synthesis using Vortex Particle Method](https://www.researchgate.net/publication/220506466_Procedural_Synthesis_using_Vortex_Particle_Method_for_Fluid_Simulation)

---

## 2. Relationship Between Particle Count and Grid Resolution

### Optimal Particles Per Cell

**Industry Standard:** **8 particles per fluid cell** based on the Nyquist sampling criterion

**Source:** [Blender Manual - Fluid Settings](https://docs.blender.org/manual/en/3.4/physics/fluid/type/domain/settings.html)

### Voxel Size to Particle Radius Relationship

**Critical Ratio:** Voxel size should be **3-4 times the particle radius**

- If voxel size is markedly different than particle radius, there may be insufficient particles per voxel
- Ensures proper sampling and prevents under-resolution

**Source:** [X-Particles FLIP/APIC Documentation](https://docs.x-particles.net/html/flipdomain.php)

### Particle Seeding Practices

**Initial seeding:**
- **Standard:** 4 particles seeded randomly in each fluid grid cell at simulation start
- **Higher quality:** 8 particles per cell for better surface reconstruction

**Dynamic particle management:**
- Particles can flow between cells during simulation
- Particles may be deleted if they move outside the narrow band
- Resampling adds new particles respecting maximum particles per cell limit

**Sources:**
- [New PIC/FLIP Simulator](https://blog.yiningkarlli.com/2014/01/flip-simulator.html)
- [Robert Bridson's methods](https://github.com/davrempe/2d-fluid-sim)

### Surface Reconstruction Considerations

**Nyquist criterion application:**
- If 8 particles are seeded per fluid cell, the optimal resolution for surface reconstruction grid = simulation grid resolution
- Prevents aliasing in surface reconstruction

**Source:** [Blender Fluid Settings](https://docs.blender.org/manual/en/3.4/physics/fluid/type/domain/settings.html)

### Advanced Methods: Variable Particle Density

**Power Particle-In-Cell (Power PIC):**
- Improves robustness to varying particle-per-cell ratios
- Maintains volume preservation
- Retains low numerical dissipation
- Better than standard FLIP/APIC for non-uniform particle distributions

**Source:** [The Power Particle-In-Cell Method](https://dl.acm.org/doi/10.1145/3528223.3530066)

---

## 3. Grid Resolution Effects on Pressure Solver Convergence

### Solver Performance Characteristics

**Key Finding:** Pressure projection is the **single most computationally expensive step** in unsteady incompressible fluid simulation.

### Preconditioned Conjugate Gradient (PCG) Solvers

**Traditional approach:**
- FLIP method traditionally uses PCG for solving pressure equations
- Convergence depends on grid resolution and preconditioner quality

**Tolerance settings:**
- Smaller error tolerance → slower convergence but more accurate results
- Error results in divergence left in velocity field → volume changes over time
- Maximum iteration limit should be set significantly higher than needed as fail-safe

**Source:** [Houdini FLIP Solver Documentation](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)

### Multigrid Pressure Solvers

**Performance improvements:**
- **2-3x speedup** compared to standard PCG solvers
- **Resolution-independent convergence rates**
- Particularly effective for large-scale simulations

**Key advantages:**
- Achieves O(N) complexity behavior with Gauss-Seidel line relaxation
- Modest memory footprint
- One order of magnitude residual reduction every 2 iterations (with proper preconditioning)
- Handles irregular domains robustly

**Implementations:**
- Houdini 20 added "Use Multigrid Preconditioner for Pressure Projection" parameter
- Can accommodate grids up to 768² x 1152 voxels with <16GB memory

**Sources:**
- [Data-driven Multi-Grid solver](https://www.sciencedirect.com/science/article/pii/S0045793022002213)
- [Parallel multigrid Poisson solver](https://www.math.ucdavis.edu/~jteran/papers/MST10.pdf)
- [Houdini 20 FLIP Fluids](https://www.sidefx.com/docs/houdini/news/20/fluids.html)

### Adaptive Grid Pressure Solving

**Spatial adaptivity benefits:**
- Employs pressure solver on adaptive grids
- Matches throughput and parallelism of uniform grids
- Significant improvements in both memory and runtime

**Key paper:** "Spatially adaptive FLIP fluid simulations in bifrost" (Michael Bang Nielsen and Robert Bridson, SIGGRAPH 2016)

**Source:** [Spatially adaptive FLIP](https://www.semanticscholar.org/paper/Spatially-adaptive-FLIP-fluid-simulations-in-Nielsen-Bridson/0490f4ba0041a09677c95b65cb6e7c168bf1276f)

### Grid Resolution and Particle Under-resolution

**Critical issue:** Individual particles can become under-resolved on the grid and "disappear" from pressure solve

**Affected by:**
- Particle Radius Scale parameter
- Grid Scale parameter on FLIP Object

**Solution:** Ensure voxel size maintains 3-4x particle radius ratio

**Source:** [Houdini FLIP Solver](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)

### Initial Guess Optimization

**Best practice:** Use pressure from previous timestep as initial guess for solver
- Often provides faster convergence
- Results in faster simulation times overall

**Source:** [FLIP Solver Documentation](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)

---

## 4. Performance Tradeoffs with Higher Resolution

### Memory vs. Resolution

**Linear scaling:**
- Grid resolution increases → memory usage increases cubically (3D simulations)
- Example: Doubling resolution in each dimension = 8x memory usage

### Computation Time vs. Resolution

**Resolution impact on simulation time:**

| Resolution | Relative Speed | Relative Quality |
|-----------|---------------|------------------|
| 64³ | Fastest | Low quality |
| 128³ | Moderate | Good quality |
| 256³ | Slow | High quality |
| 512³+ | Very slow | Very high quality |

### CPU vs. GPU Performance

**GPU advantages:**
- **10-30 milliseconds per frame** on GPU
- **10-30 minutes per frame** on CPU (same parameters)
- Up to **100 FPS** achievable with optimized GPU implementations

**Sources:**
- [Real-Time Simulation and Rendering of 3D Fluids](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids)
- [Interactive Fluid-Particle Simulation](https://research.nvidia.com/sites/default/files/pubs/2010-02_Interactive-Fluid-Particle-Simulation/paper.pdf)

### Higher-Order Schemes vs. Higher Resolution

**Key insight:** On GPU, **higher-order schemes are often better** than simply increasing grid resolution

**Reason:** Math operations are cheap compared to memory bandwidth
- Better visual detail per computation cost
- More efficient use of GPU resources

**Source:** [NVIDIA GPU Gems - Real-Time 3D Fluids](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids)

### Surface Reconstruction Performance

**Rendering optimization:**
- If simulation grid resolution is low compared to screen resolution, little visual benefit in ray casting at high resolution
- Strategy: Draw fluid into smaller off-screen render target, then composite into final image

**Source:** [GPU Gems 3 - Real-Time Fluids](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids)

### Artifact Reduction Techniques

**Surface flickering:**
- More noticeable in highly viscous fluids
- Increasing resolution = smaller voxels = less noticeable flickering
- Note: Higher resolution may affect simulation behavior (fluid may not flow exactly the same)

**Water-specific considerations:**
- Grid resolution artifacts must not be visible
- Use **tricubic interpolation** to filter values
- Higher resolution for tracer images than flow-field grid
- Use smooth bicubic interpolation when reading from tracer images and flow fields

**Sources:**
- [Blender FLIP Fluids Wiki](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-Simulation-Settings)
- [Karl Sims Fluid Flow Tutorial](https://www.karlsims.com/fluid-flow.html)

---

## 5. Adaptive Resolution and Multi-Resolution Grids

### DCGrid: Dynamic Constrained Grid

**State-of-the-art adaptive grid structure** designed for GPU implementation:

**Key features:**
- Hierarchical and adaptive grid structure
- Efficiently varies grid resolution across spatial domain
- Rapid evaluation of local stencils and individual cells
- **Memory-constrained optimization:** Grid adaptation modeled as optimization under maximum memory constraint

**Performance characteristics:**
- Addresses GPU memory limitations
- Effective for smoke flows and complex cloud simulations
- Handles terrain-atmosphere interaction with varying resolution cells
- Manages rapidly changing conditions

**Sources:**
- [DCGrid Paper (ACM)](https://dl.acm.org/doi/10.1145/3522608)
- [DCGrid on Google Research](https://research.google/pubs/dcgrid-an-adaptive-grid-structure-for-memory-constrained-fluid-simulation-on-the-gpu/)

### Adaptive Grid Banding Approach

**Strategy:** Maintain high resolution details at fluid boundaries

**Implementation:**
- Construct band of fine grid cells along fluid boundaries
- Progressively coarsen grid into interior of fluid
- Reduces computational cost while preserving surface detail

**Source:** [Adaptive FLIP solver](http://igorfx.com/hou_adaptive_flip/)

### Phase-Field-FLIP (PF-FLIP)

**For very large-scale multiphase flows:**

**Dual multiresolution scheme:**
- Couples efficient treeless adaptive grid with adaptive particles
- Fast adaptive Poisson solver for high-density-contrast flows
- Spatial adaptivity across all critical simulation components

**Capabilities:**
- Billions of particles supported
- Adaptive 3D resolutions
- High Reynolds numbers and high fluid density contrasts
- Physics-based simulation with unprecedented detail

**Source:** [Adaptive Phase-Field-FLIP](https://www.researchgate.net/publication/394042266_Adaptive_Phase-Field-FLIP_for_Very_Large_Scale_Two-Phase_Fluid_Simulation)

### Multi-Resolution Divergence Removal

**Performance optimization technique:**

**Process:**
1. Average flow field down to half-resolution
2. Process at lower resolution
3. Add differences back to full resolution
4. Recursive application: Most work done at 1/4 or lower resolution

**Result:** Real-time performance on modern GPUs

**Source:** [Karl Sims Fluid Flow](https://www.karlsims.com/fluid-flow.html)

### Adaptive Mesh Refinement (AMR)

**General approach for numerical analysis:**

**Benefits:**
- Complete control of grid resolution
- Less detailed a priori knowledge required compared to static meshes
- Automatic adaptation to sensitive/turbulent regions during simulation
- Applied successfully to two-phase flows, fluid-structure interactions, wave energy converters

**Comparison to alternatives:**
- More flexible than fixed resolution static grids
- More controlled than Lagrangian-based adaptivity (SPH)

**Sources:**
- [Adaptive Mesh Refinement (Wikipedia)](https://en.wikipedia.org/wiki/Adaptive_mesh_refinement)
- [Adaptive grid refinement for hydrodynamic flows](https://hal.science/hal-01145153/document)

### Commercial Implementation: Phoenix FD

**Adaptive Grid feature:**
- Performance optimization that keeps simulator size minimal
- Reduces RAM usage automatically
- Industry-standard commercial implementation

**Source:** [Phoenix FD Liquid Grid](https://docs.chaos.com/display/PHX4MAX/Liquid+Grid)

---

## 6. Industry Standards: Real-Time vs. Offline Simulation Resolutions

### Real-Time Simulation Standards

**Target performance:** 30-100 FPS for interactive applications

**Typical resolutions:**

| Application | Typical Grid Size | Notes |
|------------|------------------|-------|
| Games (real-time) | 32³ - 128³ | Focus on visual plausibility over accuracy |
| Interactive tools | 64³ - 256³ | Balance between quality and interactivity |
| VR/AR | 32³ - 64³ | Frame rate critical for comfort |

**Key techniques for real-time:**
- GPU-based implementation mandatory
- Higher-order schemes over brute-force resolution
- Multi-resolution processing (most work at 1/4 resolution)
- Simplified pressure solvers
- Aggressive culling and LOD

**Sources:**
- [Real-Time Fluids - Optimizing Grid-Based Methods](https://ep.liu.se/ecp/120/015/ecp15120015.pdf)
- [Real-Time Fluid Dynamics for Games (Jos Stam)](http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf)

### Offline/Production Simulation Standards

**Target:** Film-quality visuals, no real-time constraints

**Typical resolutions:**

| Production Level | Grid Size | Examples |
|-----------------|-----------|----------|
| TV/Streaming | 256³ - 512³ | Mid-budget productions |
| Feature film | 512³ - 2048³ | High-budget VFX |
| Hero shots | 2048³+ | Critical sequences, unlimited time |

**Additional quality factors:**
- 8 particles per cell minimum
- Multigrid pressure solvers for convergence
- Adaptive grids for memory efficiency
- Multiple simulation passes for refinement

**Source:** [Blender FLIP Fluids Implementation](https://github.com/rlguy/GridFluidSim3D)

### World Scale Considerations

**Critical for realistic motion:**
- Physical size of domain affects fluid speed and behavior
- Water in glass ripples more quickly than swimming pool
- Must set appropriate scale for realistic speeds

**Best practice:**
- Define domain grid tightly around fluid effect
- Maximizes performance and detail
- Prevents wasted computation on empty space

**Source:** [Domain Simulation Settings](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-Simulation-Settings)

### FLIP vs. APIC Method Selection

**FLIP (Fluid Implicit Particle):**
- High energy, noisy, chaotic simulations
- Better for large-scale simulations
- Desirable for noisy splashes

**APIC (Affine Particle-In-Cell):**
- High vorticity, swirly, stable simulations
- Better for small-scale simulations
- Reduced surface noise
- Better for viscous simulations

**Blending ratio:** 90% FLIP to 10% PIC is common industry practice

**Sources:**
- [Creating Your First FLIP Fluids Simulation](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Creating-Your-First-FLIP-Fluids-Simulation)
- [The affine particle-in-cell method](https://dl.acm.org/doi/10.1145/2766996)

### Grid Convergence Testing

**Professional workflow:**
1. Perform simulation on two or more successively finer grids
2. Refine grid (smaller cells, more cells in domain)
3. Refine time step (reduced)
4. Spatial and temporal discretization errors should asymptotically approach zero
5. Verify solution is grid-independent before production

**Source:** [NASA - Examining Spatial Grid Convergence](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html)

---

## Implementation Recommendations

### For Your Fluid Miner Simulation

Based on the research findings, here are specific recommendations:

#### 1. Grid Resolution Selection

**Current status check:**
- Review your current grid resolution
- Measure performance baseline (FPS, memory usage)

**Vortex formation optimization:**
- Since your simulation involves slurry settling and potential vortex formation:
  - Consider grid resolution at least 2x higher than particle-based equivalents
  - Target spanwise resolution ratio `dz/D ≈ 0.04` for vortex capture
  - For initial testing: Start with 128³, scale up to 256³ if performance allows

#### 2. Particle Configuration

**Particles per cell:**
- Use **8 particles per fluid cell** (Nyquist criterion)
- Maintain voxel size = **3-4x particle radius**
- Monitor particle under-resolution issues in pressure solve

**Current slurry ratio (90% water, 10% solids):**
- This is physically accurate
- Consider if solid particles need different seeding density than water

#### 3. Pressure Solver Optimization

**Immediate improvements:**
- Implement multigrid preconditioner if not already using (2-3x speedup)
- Use previous timestep pressure as initial guess
- Set tolerance appropriately (balance accuracy vs. speed)

**Current APIC implementation:**
- Good choice for smooth vortex formation
- Consider FLIP/APIC blend ratio: 90% FLIP / 10% PIC for stability

#### 4. Performance Optimization

**If hitting performance limits:**
1. Implement adaptive grid with fine resolution at slurry-water interface
2. Use multi-resolution divergence removal
3. Consider GPU implementation for 100x+ speedup
4. Profile to ensure pressure solver is properly optimized

#### 5. Testing and Validation

**Convergence testing:**
- Run same simulation at multiple resolutions (64³, 128³, 256³)
- Compare results to verify grid independence
- Document performance metrics at each resolution

**Vortex formation verification:**
- Visual inspection of vortex structures
- Measure vorticity magnitude
- Compare against expected physical behavior

---

## Key Academic References

### Foundational Papers

1. **Robert Bridson - "Fluid Simulation for Computer Graphics"**
   - Industry standard reference
   - Second edition covers hybrid particle-voxel methods
   - Available: [Routledge](https://www.routledge.com/Fluid-Simulation-for-Computer-Graphics/Bridson/p/book/9781482232837)

2. **SIGGRAPH 2007 Course Notes - Robert Bridson**
   - Comprehensive tutorial on grid-based and particle fluid simulation
   - PDF: [Fluid Simulation Notes](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf)

3. **Nielsen & Bridson - "Spatially adaptive FLIP fluid simulations in bifrost" (SIGGRAPH 2016)**
   - Adaptive multigrid-preconditioned Conjugate Gradient solver
   - Resolution-independent convergence rates
   - [Semantic Scholar](https://www.semanticscholar.org/paper/Spatially-adaptive-FLIP-fluid-simulations-in-Nielsen-Bridson/0490f4ba0041a09677c95b65cb6e7c168bf1276f)

4. **"The affine particle-in-cell method" (SIGGRAPH 2015)**
   - APIC method you're currently using
   - [ACM Digital Library](https://dl.acm.org/doi/10.1145/2766996)

5. **"The Power Particle-In-Cell Method" (SIGGRAPH 2022)**
   - Advanced method robust to varying particle-per-cell ratios
   - [ACM Digital Library](https://dl.acm.org/doi/10.1145/3528223.3530066)

### Pressure Solver Papers

1. **McAdams et al. - "A parallel multigrid Poisson solver" (2010)**
   - O(N) complexity geometric multigrid
   - [PDF](https://www.math.ucdavis.edu/~jteran/papers/MST10.pdf)

2. **"Data-driven Multi-Grid solver for accelerated pressure projection" (2022)**
   - 2-3x speedup demonstrated
   - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0045793022002213)

### Adaptive Grid Papers

1. **"DCGrid: An Adaptive Grid Structure for Memory-Constrained Fluid Simulation on the GPU" (2022)**
   - State-of-the-art memory-constrained adaptive grids
   - [ACM Digital Library](https://dl.acm.org/doi/10.1145/3522608)

2. **"Adaptive Phase-Field-FLIP for Very Large Scale Two-Phase Fluid Simulation"**
   - Billions of particles, adaptive grids
   - [ResearchGate](https://www.researchgate.net/publication/394042266_Adaptive_Phase-Field-FLIP_for_Very_Large_Scale_Two-Phase_Fluid_Simulation)

### Vortex Formation

1. **"Fluid Simulation on Vortex Particle Flow Maps" (2025)**
   - Latest research on vortex particle methods
   - [arXiv](https://arxiv.org/abs/2505.21946)

---

## Practical Implementation Resources

### Open Source Implementations

1. **GridFluidSim3D** (Ryan Guy)
   - Based on Robert Bridson's methods
   - [GitHub](https://github.com/rlguy/GridFluidSim3D)

2. **Blender FLIP Fluids**
   - Production-quality implementation
   - Extensive documentation
   - [GitHub Wiki](https://github.com/rlguy/Blender-FLIP-Fluids/wiki)

3. **GeometricMultigridPressureSolver**
   - Standalone multigrid solver implementation
   - [GitHub](https://github.com/rgoldade/GeometricMultigridPressureSolver)

4. **VorteGrid**
   - Interactive fluid simulation for games
   - Focuses on vortex methods
   - [GitHub](https://github.com/mijagourlay/VorteGrid)

### Commercial Software Documentation

1. **Houdini FLIP Solver**
   - Industry-standard implementation
   - Excellent parameter documentation
   - [SideFX Docs](https://www.sidefx.com/docs/houdini/nodes/dop/flipsolver.html)

2. **Phoenix FD**
   - Commercial implementation with adaptive grids
   - [Chaos Docs](https://docs.chaos.com/display/PHX4MAX/Liquid+Grid)

### Tutorials and Guides

1. **Karl Sims - Fluid Flow Tutorial**
   - Clear explanations of multi-resolution techniques
   - [Website](https://www.karlsims.com/fluid-flow.html)

2. **NVIDIA GPU Gems 3 - Chapter 30**
   - Real-time GPU implementation
   - [NVIDIA Developer](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids)

---

## Conclusion

Grid resolution in FLIP/PIC simulations requires careful balancing of multiple factors:

1. **Vortex capture requires 2x higher resolution** than particle methods
2. **8 particles per cell** is optimal (Nyquist criterion)
3. **Multigrid pressure solvers** provide 2-3x performance improvement
4. **Adaptive grids** offer best memory/quality tradeoff for large-scale simulations
5. **Real-time targets:** 32³-128³; **Offline targets:** 256³-2048³

For your slurry simulation with vortex formation, prioritize:
- Higher resolution at fluid-solid interfaces (adaptive grids)
- Multigrid pressure solver implementation
- Proper particle seeding (8 per cell, 3-4x voxel/particle radius ratio)
- Grid convergence testing to validate resolution choice

The research strongly supports moving to adaptive grid methods if memory becomes a constraint, with DCGrid representing the current state-of-the-art for GPU-based implementations.

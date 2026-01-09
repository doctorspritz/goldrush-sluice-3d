# Vortex Formation Testing in Fluid Simulations: Research Summary

**Research Date**: December 21, 2025
**Purpose**: Identify testing strategies, quantitative metrics, and validation approaches for vortex formation in FLIP/APIC/PIC fluid simulations

---

## Executive Summary

This research surveyed GitHub repositories, open-source fluid simulation projects, and academic implementations to discover how professional fluid simulation projects test and validate vortex formation. Key findings include:

1. **Most projects lack explicit vortex regression tests** - Visual validation dominates over quantitative metrics
2. **Reference data validation** (golden file/snapshot testing) is the most common approach when tests exist
3. **Standard benchmark problems** (Taylor-Green vortex, Karman vortex street, lid-driven cavity) serve as validation cases
4. **Quantitative metrics** (enstrophy, kinetic energy, Strouhal number) are computed but rarely integrated into automated tests
5. **CI/CD integration** for fluid simulations is minimal due to computational expense

---

## 1. Popular Open-Source Fluid Simulators

### 1.1 Blender FLIP Fluids (rlguy/Blender-FLIP-Fluids)
- **Focus**: High-quality FLIP fluid simulation for Blender
- **Vortex Behavior**: APIC demonstrates superior vortex conservation compared to FLIP
  - APIC maintains rolling vortex rings for longer durations
  - Better at conserving rotation direction and energy
  - FLIP exhibits noise and dissipation that breaks up vortex organization
- **Testing Strategy**: Not explicitly documented in public materials
- **Key Insight**: "Choose APIC for high vorticity, swirly, and stable simulations"

**Sources**:
- [Blender-FLIP-Fluids GitHub](https://github.com/rlguy/Blender-FLIP-Fluids)
- [Domain Advanced Settings Wiki](https://github.com/rlguy/Blender-FLIP-Fluids/wiki/Domain-Advanced-Settings)
- [New APIC Solver in FLIP Fluids 1.0.9b](https://flipfluids.com/weekly-development-notes-54-new-apic-solver-in-flip-fluids-1-0-9b/)

### 1.2 SPlisHSPlasH (InteractiveComputerGraphics/SPlisHSPlasH)
- **Type**: SPH-based fluid simulation library
- **Vorticity Methods Implemented**:
  - Micropolar Material Model for Turbulent SPH Fluids (Bender et al., 2017)
  - Position Based Fluids vorticity (Macklin & Müller, 2013)
  - Prescribed Velocity Gradients with Vorticity Diffusion (Peer & Teschner, 2016)
- **Testing Strategy**:
  - Has a `/Tests` directory (structure not publicly detailed)
  - GitHub Actions for multi-platform builds (Linux, Windows, macOS)
- **Key Features**: Current state-of-the-art pressure solvers, viscosity, surface tension, and vorticity methods

**Sources**:
- [SPlisHSPlasH GitHub](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
- [SPlisHSPlasH Website](https://splishsplash.physics-simulation.org/)
- [Features Documentation](https://splishsplash.physics-simulation.org/features/)

### 1.3 libfluid (lukedan/libfluid)
- **Type**: PIC/FLIP/APIC simulation library with path tracer
- **Testing**: Includes a testbed (FLUID_BUILD_TESTBED=ON by default)
- **Requirements**: C++17, GLU, GLFW for visualization
- **Testing Strategy**: Not detailed in public docs

**Source**: [libfluid GitHub](https://github.com/lukedan/libfluid)

### 1.4 Fluid Engine Dev (doyubkim/fluid-engine-dev)
- **Type**: Professional fluid simulation SDK (book companion code)
- **Methods**: Stable fluids smoke, level set liquids, PIC, FLIP, APIC
- **Testing Strategy**:
  - CI/CD via GitHub Actions (multiple platforms)
  - Code coverage tracking via Codecov
  - Unit test framework (structure not detailed in README)
  - Mentioned "unit tests" but specific implementation not visible
- **Key Insight**: Professional-grade project with CI/CD infrastructure

**Sources**:
- [fluid-engine-dev GitHub](https://github.com/doyubkim/fluid-engine-dev)
- [FluidEngine (fork with test info)](https://github.com/ZeusYang/FluidEngine)

### 1.5 WaterSim (SeanBone/WaterSim)
- **Type**: 3D FLIP solver for education
- **Testing Strategy**: **Best documented testing approach found**
  - Unit tests in `3d/tests` directory
  - CMake automatically creates test for each `.cpp` file
  - **Reference data validation**: Uses `validation_data/` folder
    - `validation-config.json`: Initial configuration
    - `ref.nc`: Reference netCDF file with model state at specific timesteps
  - Validates sub-steps of `FLIP::step_FLIP()` function
  - **Dynamic reference generation**: `WRITE_REFERENCE=ON` flag to update golden files
  - Run via `make watersim-tests`

**Key Insight**: This is the clearest example of snapshot/golden file testing for FLIP simulations

**Source**: [WaterSim GitHub](https://github.com/SeanBone/WaterSim)

### 1.6 Taichi-based Simulators

#### LBM_Taichi (hietwll/LBM_Taichi)
- **Type**: Lattice Boltzmann Method in Taichi
- **Vortex Test Case**: Kármán vortex street
  - Reynolds number Re = 200
  - Demonstrates repeating swirling vortex pattern
  - Parameters: 401x101 grid, viscosity 0.005
- **Validation**: Visual (animated GIF), no quantitative metrics documented
- **Key Insight**: "You do NOT need very fine adaptive boundary layers to generate the vortex"

**Source**: [LBM_Taichi GitHub](https://github.com/hietwll/LBM_Taichi)

#### karman_taichi (houkensjtu/karman_taichi)
- **Type**: Incompressible fluid solver in Taichi
- **Critical Implementation Detail**:
  - **Avoids first-order upwind scheme** (has strong numerical viscosity that suppresses vortices)
  - Uses **simple mid-point scheme** to preserve vortex formation
- **Validation Files**:
  - `momentum_valid.py` - momentum equation validation
  - `pressure_valid.py` - pressure field validation
- **Test Setup**:
  - Grid: 320×64 cells
  - Reynolds number: ~150
  - Cylinder approximated as square cells
- **Performance Note**: 5-10 minutes per frame (BiCGSTAB without preconditioners)

**Key Insight**: Numerical scheme choice is critical - wrong discretization can completely suppress vortex formation

**Source**: [karman_taichi GitHub](https://github.com/houkensjtu/karman_taichi)

### 1.7 Rust SPH Implementations

#### OPR (AudranDoublet/opr)
- **Type**: State-of-the-art (2020) SPH in Rust
- **Vorticity Feature**: Turbulent Micropolar SPH Fluids with Foam (Bender et al., 2017)
- **Other Features**: DFSPH pressure solver, high viscosity, surface tension
- **Testing**: No explicit test documentation in README (332 commits suggest thorough development)

**Source**: [OPR GitHub](https://github.com/AudranDoublet/opr)

#### Salva (dimforge/salva)
- **Type**: 2D/3D particle-based fluid simulation for games
- **Integration**: Works with rapier physics engine (two-way coupling)
- **Testing**: Not documented in public materials
- **CI**: Travis CI badge present

**Source**: [Salva GitHub](https://github.com/dimforge/salva)

---

## 2. Standard Benchmark Test Cases

### 2.1 Taylor-Green Vortex

**Description**: Exact solution to incompressible Navier-Stokes for vortex decay under viscous dissipation

**Why It's Useful**:
- Deterministic flow evolution from simple initial conditions
- Analytical solution exists for comparison
- Tests both inviscid and viscous implementations
- 2D and 3D versions available

**Implementations Found**:

#### Lethe (chaos-polymtl.github.io)
- Demonstrates both matrix-based and matrix-free solvers
- **Postprocessing**: Enstrophy and kinetic energy calculation capabilities
- 3D turbulent flow benchmark

**Source**: [Lethe Documentation - Taylor-Green Vortex](https://chaos-polymtl.github.io/lethe/documentation/examples/incompressible-flow/3d-taylor-green-vortex/3d-taylor-green-vortex.html)

#### HiFiLES Solver
- 3D turbulent flow benchmark
- "Particularly useful for code validation because flow evolves deterministically"

**Source**: [HiFiLES Wiki - Taylor-Green Vortex](https://github.com/HiFiLES/HiFiLES-solver/wiki/Taylor-Green-Vortex)

#### IncompressibleNavierStokes.jl
- Julia implementation
- 3D version (computationally heavy)
- Supports multi-threading

**Source**: [IncompressibleNavierStokes.jl - Taylor-Green Vortex 3D](https://agdestein.github.io/IncompressibleNavierStokes.jl/dev/examples/generated/TaylorGreenVortex3D)

#### Oceananigans.jl
- 2D exact solution example
- Vortex decaying under viscous dissipation

**Source**: [Oceananigans.jl - Taylor-Green Vortex](https://clima.github.io/OceananigansDocumentation/v0.15.0/verification/taylor_green_vortex/)

#### Python Implementations
- TGV-v2 (danm7251/TGV-v2): FDM in Python
- tIGAr (david-kamensky): Isogeometric analysis with div-free solutions
- pyranda (LLNL): Fortran-powered finite difference solver

**Sources**:
- [TGV-v2 GitHub](https://github.com/danm7251/TGV-v2-)
- [tIGAr Taylor-Green Demo](https://github.com/david-kamensky/tIGAr/blob/master/demos/taylor-green/taylor-green-2d.py)
- [pyranda Wiki](https://github.com/LLNL/pyranda/wiki/Taylor-Green-Vortex--3D-Navier-Stokes-equations)

### 2.2 Kármán Vortex Street (Cylinder Wake)

**Description**: Repeating pattern of swirling vortices behind a blunt body (cylinder)

**Key Parameters**:
- **Reynolds Number**: Re = UD/ν (where D = diameter, U = flow velocity, ν = kinematic viscosity)
- **Strouhal Number**: St = fD/U (where f = shedding frequency)
  - Typical value: St ≈ 0.2 for cylinders at Re > 1000

**Physical Phenomena**:
- Vortex shedding caused by flow separation
- Regular (periodic) shedding forms Von Kármán street when far from boundaries
- At Re > 500, exhibits counter-rotating streamwise vortex pairs (Mode B)

**Validation Methods**:

#### Experimental Comparison
- Compare against published results (Ghia, Botella, Erturk)
- Velocity profiles through centerline
- Pressure coefficient distributions

#### Computational Methods
- **Large Eddy Simulation (LES)** for turbulent wake (Re = 4×10³)
- **Fast Fourier Transform (FFT)** to determine vortex-shedding frequency
- **Detached-Eddy Simulation (DES)** captures shedding cessation near walls
- **Lagrangian Coherent Structures** analysis for vortex detection

#### Key Metrics
- Strouhal number calculation from lift force spectrum
- Drag and lift coefficients
- Vorticity contours
- Wake formation length

**Common Mistakes**:
- Using first-order upwind scheme (suppresses vortices due to numerical viscosity)
- Grid too coarse (need adequate resolution, but adaptive boundary layers not required)

**Sources**:
- [Tapered Circular Cylinder Simulations (MDPI)](https://www.mdpi.com/2311-5521/9/8/183)
- [Cylinder Near Wall Simulations (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/0142727X96000331)
- [Lagrangian Coherent Structures (AIAA Journal)](https://arc.aiaa.org/doi/10.2514/1.J055051)
- [3D Vorticity Patterns (Experiments in Fluids)](https://link.springer.com/article/10.1007/s00348-009-0629-2)
- [Vortex Shedding Overview (ScienceDirect Topics)](https://www.sciencedirect.com/topics/engineering/vortex-shedding)

### 2.3 Lid-Driven Cavity

**Description**: 2D square cavity with moving top lid - simple geometry, complex vortex structures

**Why It's Standard**:
- "One of the most frequently solved problems in CFD"
- Serves as benchmark for in-house codes
- Simple to execute, complex vortex features at higher Reynolds numbers

**Validation Data Available**:
- Published results: Ghia, Botella, Erturk
- Velocity profiles through centerline (u-velocity at x = 0.5, v-velocity at y = 0.5)
- Vortex core locations and strengths

**Solution Methods Found**:

#### Streamfunction-Vorticity Formulation
- Gauss-Seidel with SOR
- Second-order central difference on uniform staggered grid

#### Projection Method
- Crank-Nicolson/Adams-Bashforth for time integration
- Second-order central difference for spatial discretization

#### Advanced Methods
- GPU-accelerated Chorin projection scheme
- Lattice Boltzmann Method (LBM)
- Physics-Informed Neural Networks (PINNs) with NVIDIA Modulus

**Implementations**:

1. **OpenCMISS** (GFEM and residual-based stabilized)
   - Compares velocity profiles against published results
   - Zero initial condition, lid velocity ramped over 10 seconds

2. **leonardocarv/lid_driven_cavity** (Matlab)
   - Stream-function/vorticity formulation
   - Includes thermal and solutal dispersion

3. **MrJohn6774/lid-driven-cavity-flow** (Fortran)
   - Streamfunction-vorticity Navier-Stokes solver

4. **Sibusiso12302/driven_cavity_flow** (GPU-accelerated)
   - Includes vorticity calculation and visualization functions

5. **Lethe Documentation**
   - Available at [Lid-Driven Cavity Flow Guide](https://chaos-polymtl.github.io/lethe/documentation/examples/incompressible-flow/2d-lid-driven-cavity-flow/lid-driven-cavity-flow.html)

**Reynolds Number Ranges Tested**: Laminar to turbulent regimes (affects vortex formation and structure)

**Sources**:
- [OpenCMISS Example](https://github.com/OpenCMISS-Examples/navierstokes_liddrivencavity)
- [AEROSP 523 Vorticity-Streamfunction Approach](https://mariejvaucher.github.io/aero523-fall24/Examples/Week11/Chapter7.html)
- [lid-driven-cavity GitHub Topic](https://github.com/topics/lid-driven-cavity)
- [Manav-20 OpenFOAM Study](https://github.com/Manav-20/Lid_Driven_Cavity_openFoam)

### 2.4 Spinning Disk / Rotating Fluid

**Description**: Thin film flow over rotating disk - tests centrifugal forces and Coriolis effects

**Physical Phenomena**:
- **Ekman boundary layer** at center drives fluid radially outward
- Inward flow along outer stationary wall
- Competition between Coriolis and viscous forces
- Small-scale turbulent structures

**Applications**:
- Semiconductor wafer etching/processing
- Spinning disk reactors
- Chemical engineering processes

**Validation Methods**:
- Compare against Kármán's theory (analytical solution)
- Experimental measurements of film thickness
- Wave characteristics from Volume-of-Fluid (VoF) simulations
- **Typical accuracy**: 5-10% of analytical predictions

**Flow Regimes**:
- Waveless flow
- 2D waves
- 3D waves
- Fully turbulent (requires LES)

**Key Measurements**:
- Thickness evolution across radius
- Wavy air-liquid interface characterization
- Velocity profile within liquid

**CFD Approaches**:
- Volume-of-Fluid (VoF) method
- Large Eddy Simulation (LES) for turbulence
- Direct numerical simulation (DNS) for thin films

**Sources**:
- [Thin Film Flow over Spinning Disk (Physical Review Fluids)](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.10.024805)
- [ANSYS Laminar Flow Example](https://courses.ansys.com/index.php/courses/laminar-boundary-layer-theory/lessons/simulation-examples-homework-and-quizzes-6/topic/laminar-flow-over-a-spinning-disk-simulation-example/)
- [Fluid Simulation over Rotating Disk (Scientific.Net)](https://www.scientific.net/SSP.346.231)

### 2.5 Vortex Sheet Test (APIC Benchmark)

**Description**: Velocity discontinuity at circular interface induces vorticity, producing intricate flow patterns

**Setup**:
- Velocity inside circle: rotating
- Surrounding fluid: stationary
- Discontinuity induces vorticity at interface

**Method Comparison Results**:
- **FLIP**: Vortex ring becomes disorganized, rotational energy dissipates
- **APIC**: Maintains rolling motion much longer, very good at preserving rotational energy
- **PolyPIC**: Also tested in comparisons

**Source**: [Vortex Sheet Comparison (ResearchGate)](https://www.researchgate.net/figure/Vortex-sheet-We-compare-from-left-to-right-FLIP-APIC-and-PolyPIC-with-2D_fig2_321232128)

---

## 3. Quantitative Metrics for Vortex Testing

### 3.1 Enstrophy

**Definition**: ε = ∫ |ω|² dV (integral of vorticity squared)

**Physical Meaning**:
- Measure of magnitude of vorticity in flow
- Related to kinetic energy dissipation: dE/dt = -ν·enstrophy
- For incompressible flow: directly proportional to energy dissipation rate

**Uses in Testing**:
- Track generation and evolution of turbulence
- Verify conservation properties in inviscid flows
- Validate dissipation rates in viscous flows

**Implementation Examples**:

#### ENSTvisualise (Will-McD/ENSTvisualise)
- R-based tools for SPH turbulence visualization
- `plot_enst()` function with adjustable brightness
- Generates global lists of estimated enstrophy

**Source**: [ENSTvisualise GitHub](https://github.com/Will-McD/ENSTvisualise/)

#### iModel (pedrospeixoto/iModel)
- Matlab finite difference C-grid
- Energy-enstrophy conserving schemes for shallow water
- Planar model implementation

**Source**: [iModel GitHub](https://github.com/pedrospeixoto/iModel)

#### Monaghan's SPH Turbulence Model
- Tracks ratio of enstrophy to kinetic energy over time
- Time variation follows power law (1/t^1.7)
- Converts SPH summations to integrals

**Source**: [A Turbulence Model for SPH (arXiv)](https://arxiv.org/pdf/0911.2523)

#### Conservation Schemes
- Mass, energy, enstrophy, and vorticity conserving discretizations
- MEEVC (Mass, Energy, Enstrophy, Vorticity Conserving) mimetic spectral elements
- Proofs for exact discrete conservation properties

**Calculation Formula** (for incompressible flow):
```
enstrophy = ∫∫∫ (ω_x² + ω_y² + ω_z²) dV
```

**Sources**:
- [A Note on Kinetic Energy, Dissipation and Enstrophy (NASA)](https://ntrs.nasa.gov/api/citations/19980236913/downloads/19980236913.pdf)
- [Expressing TKE as Enstrophy (Physical Review Fluids)](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.10.L022601)

### 3.2 Kinetic Energy

**Definition**: KE = ½∫ρ|v|² dV

**Uses in Testing**:
- Energy conservation verification (inviscid flows)
- Dissipation rate validation (viscous flows)
- Taylor-Green vortex decay rate

**Relationship to Enstrophy**:
```
dKE/dt = -ν·enstrophy  (for incompressible homogeneous flow)
```

**Conservation Properties**:
- Should be exactly conserved in inviscid simulations
- Known decay rate for Taylor-Green vortex
- Can derive exact identities connecting velocity gradients to total KE

**Testing Strategy**: Track KE over time and compare to analytical solutions

### 3.3 Strouhal Number

**Definition**: St = fD/U

Where:
- f = vortex shedding frequency (Hz)
- D = characteristic length (cylinder diameter)
- U = freestream velocity

**Typical Values**:
- Circular cylinder (Re > 1000): St ≈ 0.21
- Rectangular bodies: 0.1 < St < 0.2
- Flow-dependent at lower Reynolds numbers

**Measurement Method**:
1. Record lift force time series: f_y(t)
2. Apply Fast Fourier Transform (FFT)
3. Find dominant frequency peak
4. Calculate: St = f_peak × D / U

**Implementation Notes**:
- Need sufficiently long time series for FFT accuracy
- May require filtering to remove noise
- Compare against published correlations: St = S(Re)

**Example Code** (conceptual):
```python
# Record lift coefficient over time
cl_history = []
for timestep in simulation:
    cl = compute_lift_coefficient()
    cl_history.append(cl)

# FFT to find dominant frequency
frequencies, power = fft(cl_history, dt)
dominant_freq = frequencies[argmax(power)]

# Calculate Strouhal number
strouhal = dominant_freq * diameter / velocity
```

**Sources**:
- [Strouhal Number Wikipedia](https://en.wikipedia.org/wiki/Strouhal_number)
- [Vortex Shedding and Strouhal Number (MIT)](http://web.mit.edu/13.021/demos/lectures/lecture15.pdf)
- [Karman Vortex and Strouhal Number Tutorial](http://www.vibrationdata.com/tutorials2/strouhal.pdf)
- [Engineering Toolbox - Strouhal Number](https://www.engineeringtoolbox.com/strouhal-number-d_582.html)

### 3.4 Circulation

**Definition**: Γ = ∮ v·dl (line integral of velocity around closed path)

**Relationship to Vorticity**: By Stokes' theorem: Γ = ∫∫ ω·dA

**Uses in Testing**:
- Verify vortex strength preservation
- Kelvin circulation theorem validation (inviscid flow)
- Vortex identification and tracking

**Implementation Example**:

#### Combinatorial Vortex Detection (dsnobes)
- MATLAB implementation for PIV data
- Includes `Circulation.m` function
- Multiple vortex detection methods:
  - Maximum Vorticity Method
  - Cross-Sectional Lines Method
  - Winding Angle Method

**Source**: [Combinatorial Vortex Detection GitHub](https://github.com/dsnobes/Combinatorial-Vortex-Detection-Algorithm)

**Calculation Approaches**:
1. Direct line integral around path
2. Area integral of vorticity (Stokes' theorem)
3. Summation over discrete points

### 3.5 Vortex Core Detection Algorithms

#### Lambda-2 (λ₂) Criterion

**Method** (Jeong & Hussain):
1. Decompose velocity gradient: J = S + Ω
   - S = symmetric part (strain-rate tensor)
   - Ω = antisymmetric part (spin tensor)
2. Compute: S² + Ω²
3. Find eigenvalues (all real, as matrix is symmetric)
4. Vortex core where **second eigenvalue λ₂ < 0**

**Advantages**:
- Removes unsteady irrotational straining effects
- Removes viscous pressure minimum elimination
- Threshold can filter small/noise vortices

**Source**: [Lambda2 Method Wikipedia](https://en.wikipedia.org/wiki/Lambda2_method)

#### Q-Criterion

**Method** (Hunt et al., 1988):
- Q = ½(|Ω|² - |S|²)
- Vortex where **Q > 0** (vorticity dominates strain)
- Additional condition: pressure in eddy < ambient pressure

**Interpretation**: Vorticity magnitude > strain rate magnitude in vortex regions

#### Comparison of Methods
- Lambda-2 and Q-criterion are most prominent
- Swirling strength has highest correlation with Rortex
- Q-criterion is second
- Lambda-2 is third in correlation
- All may miss vortices with weak rotational velocities (threshold-dependent)

**Implementation Tools**:
- ParaView: Built-in Lambda-2 calculation
- User-Defined Functions in ANSYS Fluent
- Python libraries (examples in scientific computing)

**Sources**:
- [Detection and Visualization of Vortices](https://www.cavs.msstate.edu/publications/docs/2005/01/3269visHandbook.pdf)
- [Vortex Visualization Comparison (ResearchGate)](https://www.researchgate.net/figure/Vortex-visualization-by-Rortex-Q-criterion-Lambda-2-criterion-and-swirling-strength_fig8_345870293)
- [Lambda2 Criterion Python Implementation](https://www.folkstalk.com/tech/lambda2-criterion-python-with-code-examples/)
- [Vortex Identification Methods (Taylor & Francis)](https://www.tandfonline.com/doi/full/10.1080/19942060.2020.1816496)

### 3.6 Vorticity Calculation

**Definition**: ω = ∇ × v

**3D Components**:
```
ω_x = ∂w/∂y - ∂v/∂z
ω_y = ∂u/∂z - ∂w/∂x
ω_z = ∂v/∂x - ∂u/∂y
```

**2D Simplification**:
```
ω_z = ∂v/∂x - ∂u/∂y
```

**Implementation Examples**:

#### OpenFOAM
```cpp
// Calculate vorticity as curl of velocity
volVectorField vorticity = fvc::curl(U);
```
**Source**: [OpenFOAM vorticity.C](https://github.com/OpenFOAM/OpenFOAM-2.2.x/blob/master/applications/utilities/postProcessing/velocityField/vorticity/vorticity.C)

#### MetPy (Python)
```python
import metpy.calc as mpcalc

# Calculate vertical vorticity
vorticity = mpcalc.vorticity(u, v, dx, dy)
```
**Source**: [MetPy Vorticity Documentation](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.vorticity.html)

#### Finite Difference (General)
```
ω_z(i,j) ≈ (v(i+1,j) - v(i-1,j))/(2Δx) - (u(i,j+1) - u(i,j-1))/(2Δy)
```

**Numerical Scheme Considerations**:
- Central difference: Second-order accurate, no artificial dissipation
- Upwind: First-order, adds numerical viscosity (can suppress vortices!)
- Need adequate grid resolution to capture vorticity gradients

---

## 4. Testing Strategies Found in Real Projects

### 4.1 Reference Data Validation (Golden File Testing)

**Best Example**: WaterSim (SeanBone/WaterSim)

**Approach**:
1. Run simulation with known-good implementation
2. Save state at specific timesteps to reference file
3. For each test run:
   - Execute simulation with same initial conditions
   - Compare state against reference at same timesteps
   - Assert differences below tolerance threshold

**File Structure**:
```
3d/
├── tests/
│   ├── test_pressure.cpp
│   ├── test_advection.cpp
│   └── test_particle_to_grid.cpp
└── validation_data/
    ├── validation-config.json
    └── ref.nc (netCDF reference data)
```

**Advantages**:
- Catches regressions in complex simulations
- Don't need analytical solutions
- Can validate entire simulation pipeline

**Disadvantages**:
- Initial reference may contain bugs
- Brittle (intentional changes require reference update)
- Large file sizes for 3D simulations
- Platform/compiler differences may cause false failures

**Update Mechanism**: `WRITE_REFERENCE=ON` CMake flag

**Similar Approaches in Other Domains**:
- Flutter golden tests (UI snapshots)
- Helm chart golden testing
- API response validation

**Sources**:
- [WaterSim GitHub](https://github.com/SeanBone/WaterSim)
- [Golden File Testing Discussion (JUnit)](https://github.com/junit-team/junit5/discussions/2697)
- [Flutter Golden Tests Guide](https://www.dhiwise.com/post/guide-to-flutter-golden-tests-for-flawless-ui-testing)

### 4.2 Analytical Solution Comparison

**Best Example**: Taylor-Green Vortex (Multiple Implementations)

**Approach**:
1. Set up initial conditions matching analytical solution
2. Run simulation for known time period
3. Compare computed fields to analytical solution:
   - Velocity components: |v_computed - v_analytical| < ε
   - Pressure field
   - Kinetic energy decay rate
   - Enstrophy decay rate

**Advantages**:
- Absolute ground truth validation
- Can measure convergence rates (spatial and temporal)
- Clear pass/fail criteria

**Disadvantages**:
- Limited to simple geometries
- Few analytical solutions exist for vortex-dominated flows
- May not test full feature set

**Example Metrics** (Taylor-Green):
```python
# Analytical kinetic energy decay
KE_analytical = KE_0 * exp(-2 * nu * t)

# Compute relative error
error = abs(KE_computed - KE_analytical) / KE_analytical

assert error < tolerance  # e.g., 1e-3 for 0.1% accuracy
```

### 4.3 Validation Script Approach

**Example**: karman_taichi (houkensjtu/karman_taichi)

**Structure**:
- `momentum_valid.py` - Validates momentum equation discretization
- `pressure_valid.py` - Validates pressure field computation

**Likely Approach** (inferred):
1. Run simulation on simple test case
2. Check conservation properties:
   - Momentum conservation
   - Mass conservation
   - Pressure-velocity coupling
3. Compare against simplified analytical limits

**Advantages**:
- Validates individual components separately
- Easier to debug failures
- Fast execution (simple cases)

**Note**: Actual implementation details not publicly documented

### 4.4 Visual Validation (Most Common)

**Approach**:
- Run simulation
- Generate visualization (images, videos, ParaView files)
- Manually inspect for expected vortex structures
- Qualitative assessment of "looks right"

**Prevalence**: Vast majority of projects rely primarily on this

**Advantages**:
- Easy to implement
- Catches obvious problems
- Good for development/debugging

**Disadvantages**:
- Not automated
- Subjective
- Doesn't catch subtle regressions
- No quantitative metrics
- Can't run in CI/CD

**Common Tools**:
- ParaView for 3D visualization
- matplotlib for 2D plots
- Custom OpenGL renderers
- Blender integration

### 4.5 Energy Conservation Tests

**Approach**:
```rust
#[test]
fn test_energy_conservation_inviscid() {
    let mut sim = create_inviscid_simulation();
    let initial_energy = sim.compute_total_kinetic_energy();

    for _ in 0..1000 {
        sim.step();
    }

    let final_energy = sim.compute_total_kinetic_energy();
    let energy_drift = (final_energy - initial_energy).abs() / initial_energy;

    assert!(energy_drift < 1e-6, "Energy not conserved: drift = {}", energy_drift);
}
```

**Advantages**:
- Tests fundamental physics
- Single scalar value to track
- Should be exactly conserved (inviscid) or decay predictably (viscous)

**Disadvantages**:
- Doesn't directly test vortex formation
- May pass even with incorrect vortex behavior

### 4.6 Benchmark Comparison Tests

**Approach**:
1. Implement standard benchmark (e.g., lid-driven cavity at Re=1000)
2. Extract characteristic values:
   - Primary vortex center location
   - Secondary vortex positions
   - Velocity at specific points
3. Compare to published literature values
4. Assert within tolerance of published results

**Example** (Lid-Driven Cavity):
```python
def test_lid_driven_cavity_re1000():
    sim = run_simulation(Re=1000)

    # Published values from Ghia et al.
    u_centerline_expected = [0, 0.1807, 0.5682, 0.8169, ...]
    u_centerline_computed = sim.get_u_velocity_at_x_centerline()

    max_error = max(abs(u_comp - u_exp)
                    for u_comp, u_exp
                    in zip(u_centerline_computed, u_centerline_expected))

    assert max_error < 0.01, f"Velocity profile error: {max_error}"
```

**Advantages**:
- Well-established validation
- Directly comparable to literature
- Tests actual vortex formation in realistic scenario

**Disadvantages**:
- Computational expense (may be too slow for CI)
- Requires careful grid setup to match published cases
- Sensitive to many parameters

---

## 5. CI/CD Integration Patterns

### 5.1 Current State

**Finding**: Minimal CI/CD for fluid simulations in open-source projects

**Common Setup**:
- Build verification across platforms (Linux, Windows, macOS)
- Basic unit tests (if any)
- Code coverage tracking (rare)
- **No vortex-specific regression tests in CI**

**Examples**:

#### fluid-engine-dev
- GitHub Actions workflows for multi-platform builds
- Codecov integration
- Test framework exists but details not public

#### SPlisHSPlasH
- GitHub Actions for platform builds
- `/Tests` directory exists
- Actual test execution in CI not documented

#### WaterSim
- CTest integration with CMake
- Can run `make watersim-tests`
- No evidence of CI automation (educational project)

### 5.2 Challenges for Fluid Sim CI/CD

1. **Computational Expense**:
   - Realistic simulations take minutes to hours
   - CI typically has 5-10 minute timeouts
   - Full validation suites impractical

2. **Platform Differences**:
   - Floating-point non-determinism across CPUs
   - GPU implementations even less deterministic
   - Golden file comparisons may fail spuriously

3. **Storage Requirements**:
   - 3D reference data files can be gigabytes
   - Git LFS required for large binary files
   - Expensive for open-source projects

4. **Parallelism Complexity**:
   - Thread count affects floating-point summation order
   - Results may differ between local dev and CI runners

### 5.3 Practical CI/CD Strategies

#### Fast Smoke Tests
```yaml
# .github/workflows/test.yml
- name: Fast Smoke Tests
  run: |
    cargo test --lib  # Unit tests only
    cargo test --test smoke_tests  # Quick integration tests
  timeout-minutes: 5
```

**Contents**:
- 2D simulations (much faster than 3D)
- Low resolution (16x16 or 32x32 grids)
- Short time spans (10-100 timesteps)
- Sanity checks rather than full validation

#### Nightly Full Validation
```yaml
# .github/workflows/nightly.yml
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  full-validation:
    timeout-minutes: 120
    steps:
      - name: Run Benchmark Suite
        run: cargo test --release --test benchmarks
```

**Contents**:
- Full resolution benchmark cases
- Reference data comparison
- Performance regression detection
- Generate reports for review

#### Manual Validation Workflow
```yaml
# .github/workflows/manual-validation.yml
on: workflow_dispatch

jobs:
  validate:
    steps:
      - name: Run Full Test Suite
        run: ./scripts/run_validation_suite.sh
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: validation_output/
```

**Use Case**: Before major releases, manual trigger for comprehensive testing

### 5.4 MATLAB/Simulink Approach (Reference)

**CI Support Package for Simulink**:
- Prequalify on desktop to minimize build failures
- Incremental builds on CI system
- Integration with GitLab, Jenkins, GitHub Actions, Azure DevOps
- Example YAML files provided

**Lessons Applicable to Fluid Sims**:
- Desktop validation before CI
- Incremental testing strategy
- CI system integration via standard tools

**Source**: [MATLAB CI/CD Automation](https://www.mathworks.com/products/ci-cd-automation.html)

---

## 6. Recommendations for Your Project

### 6.1 Short-Term: Establish Basic Testing Infrastructure

#### 1. Unit Tests for Vorticity Calculation
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vorticity_solid_body_rotation() {
        // Analytical case: solid body rotation v = ω × r
        // Should give constant vorticity ω everywhere

        let grid = create_test_grid(32, 32);
        let omega = 1.0;  // Angular velocity

        // Set velocity field: u = -ω*y, v = ω*x
        for cell in grid.cells() {
            let (x, y) = cell.position();
            cell.set_velocity(-omega * y, omega * x);
        }

        // Compute vorticity
        grid.compute_vorticity();

        // Check all cells have vorticity ≈ 2ω
        for cell in grid.cells() {
            let vort = cell.vorticity_z();
            assert!((vort - 2.0 * omega).abs() < 1e-6,
                   "Expected vorticity {}, got {}", 2.0 * omega, vort);
        }
    }

    #[test]
    fn test_vorticity_point_vortex() {
        // Analytical case: point vortex
        // v_θ = Γ/(2πr), vorticity = 0 everywhere except origin

        let grid = create_test_grid(64, 64);
        let circulation = 1.0;

        set_point_vortex_velocity(&mut grid, circulation);
        grid.compute_vorticity();

        // Vorticity should be near zero away from origin
        for cell in grid.cells() {
            let distance = cell.distance_from_origin();
            if distance > 0.1 {  // Away from singularity
                let vort = cell.vorticity_z();
                assert!(vort.abs() < 1e-3,
                       "Vorticity should be ~0 at distance {}, got {}",
                       distance, vort);
            }
        }
    }
}
```

#### 2. Enstrophy Calculation and Tracking
```rust
pub struct Simulation {
    // ... existing fields
    enstrophy_history: Vec<f32>,
    kinetic_energy_history: Vec<f32>,
}

impl Simulation {
    pub fn compute_enstrophy(&self) -> f32 {
        let mut enstrophy = 0.0;
        for cell in self.grid.cells() {
            let omega = cell.vorticity();
            enstrophy += omega.magnitude_squared() * cell.volume();
        }
        enstrophy * 0.5  // Convention: 1/2 ∫ ω² dV
    }

    pub fn compute_kinetic_energy(&self) -> f32 {
        let mut ke = 0.0;
        for cell in self.grid.cells() {
            let v = cell.velocity();
            ke += v.magnitude_squared() * cell.volume() * self.density;
        }
        ke * 0.5
    }

    pub fn step(&mut self) {
        // ... existing simulation step

        // Track conservation quantities
        self.enstrophy_history.push(self.compute_enstrophy());
        self.kinetic_energy_history.push(self.compute_kinetic_energy());
    }
}

#[test]
fn test_energy_conservation_inviscid() {
    let mut sim = Simulation::new_inviscid();
    sim.set_initial_conditions_taylor_green();

    let initial_energy = sim.compute_kinetic_energy();

    for _ in 0..100 {
        sim.step();
    }

    let final_energy = sim.compute_kinetic_energy();
    let drift = (final_energy - initial_energy).abs() / initial_energy;

    assert!(drift < 1e-4, "Energy drift too large: {}", drift);
}
```

### 6.2 Medium-Term: Implement Benchmark Tests

#### 1. Taylor-Green Vortex Test
```rust
#[test]
fn test_taylor_green_vortex_2d() {
    let mut sim = Simulation::builder()
        .grid_size(64, 64)
        .domain_size(2.0 * PI, 2.0 * PI)
        .viscosity(0.01)
        .build();

    // Initial conditions: TG vortex
    // u = -cos(x) * sin(y)
    // v =  sin(x) * cos(y)
    for cell in sim.grid.cells_mut() {
        let (x, y) = cell.position();
        let u = -x.cos() * y.sin();
        let v =  x.sin() * y.cos();
        cell.set_velocity(u, v);
    }

    let initial_ke = sim.compute_kinetic_energy();
    let nu = sim.viscosity();
    let t = 1.0;  // Simulation time

    // Run simulation
    while sim.time < t {
        sim.step();
    }

    // Analytical solution: KE decays as exp(-2νt)
    let expected_ke = initial_ke * (-2.0 * nu * t).exp();
    let computed_ke = sim.compute_kinetic_energy();

    let error = (computed_ke - expected_ke).abs() / expected_ke;
    assert!(error < 0.05, "KE error too large: {}%", error * 100.0);
}
```

#### 2. Vortex Sheet Test (APIC Validation)
```rust
#[test]
fn test_vortex_sheet_conservation() {
    let mut sim = Simulation::builder()
        .method(Method::APIC)
        .grid_size(128, 128)
        .particles_per_cell(16)
        .build();

    // Set up: rotating fluid inside circle, stationary outside
    let circle_radius = 0.25;
    let omega = 2.0;

    for particle in sim.particles_mut() {
        let r = particle.position().magnitude();
        if r < circle_radius {
            let theta = particle.position().y.atan2(particle.position().x);
            particle.set_velocity(
                -omega * r * theta.sin(),
                 omega * r * theta.cos()
            );
        }
    }

    // Calculate initial circulation around circle
    let initial_circulation = sim.calculate_circulation_around_circle(circle_radius * 1.5);

    // Run for many steps
    for _ in 0..500 {
        sim.step();
    }

    // Circulation should be conserved (inviscid)
    let final_circulation = sim.calculate_circulation_around_circle(circle_radius * 1.5);
    let drift = (final_circulation - initial_circulation).abs() / initial_circulation.abs();

    assert!(drift < 0.1, "Circulation not conserved: {}% drift", drift * 100.0);

    // Enstrophy should not decay too rapidly (APIC property)
    let initial_enstrophy = sim.enstrophy_history[0];
    let final_enstrophy = *sim.enstrophy_history.last().unwrap();
    let enstrophy_ratio = final_enstrophy / initial_enstrophy;

    assert!(enstrophy_ratio > 0.7,
           "Enstrophy decayed too much: retained only {}%",
           enstrophy_ratio * 100.0);
}
```

### 6.3 Long-Term: Reference Data System

#### 1. Generate Reference Data
```rust
// tests/generate_reference_data.rs
#[test]
#[ignore]  // Only run when --ignored flag passed
fn generate_reference_vortex_sheet() {
    let mut sim = create_standard_vortex_sheet_test();

    let mut reference_data = ReferenceData::new();
    reference_data.save_initial_state(&sim);

    for step in 0..1000 {
        sim.step();

        // Save at specific checkpoints
        if step % 100 == 0 {
            reference_data.save_checkpoint(step, &sim);
        }
    }

    reference_data.write_to_file("tests/reference_data/vortex_sheet.bincode");
}
```

#### 2. Validate Against Reference
```rust
#[test]
fn test_vortex_sheet_regression() {
    let reference = ReferenceData::load("tests/reference_data/vortex_sheet.bincode");
    let mut sim = create_standard_vortex_sheet_test();

    // Verify initial state matches
    assert!(sim.matches_reference_state(&reference.initial_state, 1e-10));

    for step in 0..1000 {
        sim.step();

        if step % 100 == 0 {
            let checkpoint = reference.get_checkpoint(step);

            // Compare particle positions
            assert_particle_positions_match(&sim, &checkpoint, tolerance: 1e-3);

            // Compare velocity field
            assert_velocity_field_matches(&sim, &checkpoint, tolerance: 1e-3);

            // Compare enstrophy
            let enstrophy_diff = (sim.compute_enstrophy() - checkpoint.enstrophy).abs();
            assert!(enstrophy_diff < 1e-4,
                   "Enstrophy mismatch at step {}: diff = {}",
                   step, enstrophy_diff);
        }
    }
}
```

### 6.4 CI/CD Strategy

#### Fast PR Checks
```yaml
# .github/workflows/pr-check.yml
name: PR Checks
on: [pull_request]

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run unit tests
        run: cargo test --lib

      - name: Run fast integration tests
        run: cargo test --test smoke_tests

      - name: Check vorticity calculation
        run: cargo test test_vorticity
```

#### Nightly Validation
```yaml
# .github/workflows/nightly.yml
name: Nightly Validation
on:
  schedule:
    - cron: '0 3 * * *'

jobs:
  full-validation:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run full test suite
        run: cargo test --release

      - name: Run benchmarks
        run: cargo bench

      - name: Check reference data
        run: cargo test --release --test reference_data_tests

      - name: Generate report
        run: ./scripts/generate_validation_report.sh

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: target/validation/
```

### 6.5 Development Workflow Integration

#### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running vorticity tests..."
cargo test test_vorticity --quiet || exit 1

echo "Running smoke tests..."
cargo test --test smoke_tests --quiet || exit 1

echo "All checks passed!"
```

#### Visualization for Development
```rust
// Only compile when running locally, not in CI
#[cfg(all(test, not(ci)))]
mod visual_tests {
    #[test]
    fn visualize_vortex_sheet() {
        let mut sim = create_standard_vortex_sheet_test();
        let mut renderer = Renderer::new("vortex_sheet_test");

        for step in 0..500 {
            sim.step();

            if step % 10 == 0 {
                renderer.render_vorticity(&sim);
                renderer.save_frame(step);
            }
        }

        renderer.create_video("test_output/vortex_sheet.mp4");
        println!("Video saved. Review for visual validation.");
    }
}
```

---

## 7. Key Takeaways and Best Practices

### 7.1 Testing Philosophy

1. **Layered Validation**:
   - Unit tests: Vorticity calculation, individual components
   - Integration tests: Short simulations with analytical solutions
   - Benchmark tests: Standard problems with published results
   - Reference tests: Regression detection for complex scenarios

2. **Quantitative Over Qualitative**:
   - Visual inspection is necessary but not sufficient
   - Track scalar metrics: enstrophy, kinetic energy, circulation
   - Compare to analytical solutions where possible
   - Use published benchmark values

3. **Conservation as Sanity Check**:
   - Energy conservation (inviscid flows)
   - Circulation conservation (inviscid flows)
   - Mass conservation (always)
   - Failure indicates fundamental problem

### 7.2 Common Pitfalls

1. **Numerical Viscosity Suppressing Vortices**:
   - Avoid first-order upwind schemes
   - Use centered differences or higher-order methods
   - Validate discretization on simple rotating flows

2. **Grid Resolution Inadequate**:
   - Vortex cores require sufficient resolution
   - Test with grid refinement studies
   - Compare coarse/medium/fine grids

3. **Time Step Too Large**:
   - CFL condition violations can destroy vortices
   - Adaptive time stepping helps
   - Test with different Δt values

4. **Over-Reliance on Visual Validation**:
   - "Looks good" is not reproducible
   - Subtle regressions go unnoticed
   - Implement quantitative checks

### 7.3 Practical Development Process

1. **Start Simple**:
   - 2D before 3D
   - Low resolution prototypes
   - Analytical test cases first

2. **Incremental Validation**:
   - Add one feature at a time
   - Validate before proceeding
   - Keep working tests passing

3. **Document Expected Behavior**:
   - What should vortex do?
   - What metrics matter?
   - What are acceptable tolerances?

4. **Use Standard Benchmarks**:
   - Taylor-Green vortex for decay
   - Karman vortex street for shedding
   - Lid-driven cavity for recirculation
   - Don't invent new tests if standard ones exist

### 7.4 When to Use Each Approach

| Test Type | Use When | Computational Cost | Maintenance |
|-----------|----------|-------------------|-------------|
| Unit tests | Always | Very low | Low |
| Analytical comparison | Solution exists | Low | Low |
| Reference data | Complex scenarios | Medium-High | Medium |
| Visual inspection | Development/debugging | Low | High (manual) |
| Benchmark comparison | Publication validation | High | Low |

---

## 8. Further Resources

### 8.1 Books and Papers

- **Bridson, "Fluid Simulation for Computer Graphics"** - Foundation for FLIP/PIC methods
- **Bender et al., "Turbulent Micropolar SPH Fluids with Foam" (2017)** - Vorticity in SPH
- **Jiang et al., "The Affine Particle-In-Cell Method" (2015)** - APIC introduction
- **Jeong & Hussain, "On the identification of a vortex" (1995)** - Lambda-2 criterion
- **Monaghan, "A Turbulence Model for SPH" (2011)** - Enstrophy in SPH

### 8.2 Software Tools

- **ParaView**: Visualization with built-in Lambda-2, Q-criterion
- **VisIt**: Large-scale visualization
- **Matplotlib/Mayavi**: Python plotting
- **Blender**: Artistic rendering of fluid simulations

### 8.3 Online Resources

- **SPlisHSPlasH Documentation**: [splishsplash.readthedocs.io](https://splishsplash.readthedocs.io/)
- **Lethe Examples**: [chaos-polymtl.github.io/lethe](https://chaos-polymtl.github.io/lethe)
- **Context7 Documentation**: Use for up-to-date library docs

---

## Appendix A: Example Test Code Structure

```
goldrush-fluid-miner/
├── crates/
│   └── fluid_sim/
│       ├── src/
│       │   ├── lib.rs
│       │   ├── grid.rs
│       │   ├── particles.rs
│       │   ├── vorticity.rs  ← New module
│       │   └── metrics.rs    ← New module (enstrophy, KE, etc.)
│       └── tests/
│           ├── unit/
│           │   ├── test_vorticity_calculation.rs
│           │   ├── test_circulation.rs
│           │   └── test_lambda2.rs
│           ├── integration/
│           │   ├── test_taylor_green.rs
│           │   ├── test_vortex_sheet.rs
│           │   └── test_solid_rotation.rs
│           ├── benchmarks/
│           │   ├── test_lid_cavity.rs
│           │   └── test_karman_street.rs  ← Future work
│           ├── regression/
│           │   └── reference_data_tests.rs
│           └── smoke_tests.rs  ← Fast CI tests
├── tests/
│   └── reference_data/
│       ├── vortex_sheet.bincode
│       ├── taylor_green_2d.bincode
│       └── solid_rotation.bincode
└── scripts/
    ├── generate_reference_data.sh
    └── run_validation_suite.sh
```

---

## Appendix B: Key Equations Summary

### Vorticity
```
ω = ∇ × v
ω_z = ∂v/∂x - ∂u/∂y  (2D)
```

### Enstrophy
```
ε = ∫ |ω|² dV
```

### Kinetic Energy
```
KE = ½∫ρ|v|² dV
```

### Energy-Enstrophy Relation
```
dKE/dt = -ν·ε  (incompressible, homogeneous)
```

### Circulation
```
Γ = ∮ v·dl = ∫∫ ω·dA  (Stokes' theorem)
```

### Strouhal Number
```
St = fD/U
```

### Lambda-2 Criterion
```
S² + Ω² has eigenvalues λ₁ ≥ λ₂ ≥ λ₃
Vortex core where λ₂ < 0
```

### Q-Criterion
```
Q = ½(|Ω|² - |S|²)
Vortex where Q > 0
```

---

## Document Metadata

- **Author**: Research conducted by Claude (Anthropic)
- **Date**: December 21, 2025
- **Repository**: goldrush-fluid-miner
- **Purpose**: Guide implementation of vortex formation testing
- **Status**: Research phase - ready for implementation planning

**Next Steps**:
1. Review this research with team
2. Prioritize which testing strategies to implement first
3. Create implementation plan
4. Set up test infrastructure
5. Implement unit tests for vorticity calculation
6. Add Taylor-Green vortex benchmark
7. Establish CI/CD pipeline with fast smoke tests

---

**End of Research Document**

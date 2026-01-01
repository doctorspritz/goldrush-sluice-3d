# Turbulent Suspension Approaches for Sand in Water

**Problem**: Sand particles drag along the bottom instead of swirling within the water flow.

**Root Cause**: Current code applies constant settling (`SETTLING_FACTOR * GRAVITY * dt`) regardless of flow conditions. Real physics: particles suspend when turbulent diffusion >= settling.

---

## Option A: Rouse-Based Suspension (DEFERRED)

**Source**: [HEC-RAS Rouse-Diffusion Method](https://www.hec.usace.army.mil/confluence/rasdocs/rassed1d/1d-sediment-transport-technical-reference-manual/bed-change/rouse-diffusion-method)

### Theory
The Rouse Number determines vertical concentration profile:
```
P = ws / (κ * u*)

where:
  ws = settling velocity
  κ  = 0.41 (von Karman constant)
  u* = shear velocity (friction velocity)
```

Classification:
- P < 0.8: wash load (fully suspended, uniform profile)
- 0.8 < P < 2.5: suspended load (varies with depth)
- P > 2.5: bedload (concentrated near bed)

### Implementation Sketch
```rust
// In G2P for sand:
let flow_speed = v_grid.length();
let u_star = flow_speed * 0.1;  // Rough approximation: u* ≈ 0.1 * U
let settling_vel = GRAVITY * SETTLING_FACTOR * dt;
let rouse = if u_star > 0.01 {
    settling_vel / (0.41 * u_star)
} else {
    10.0  // High Rouse = bedload when no flow
};

// Reduce settling based on Rouse number
let settling_fraction = match rouse {
    r if r < 0.8 => 0.0,           // Fully suspended
    r if r < 2.5 => (r - 0.8) / 1.7,  // Partially suspended
    _ => 1.0,                       // Full settling (bedload)
};
particle.velocity.y += settling_vel * settling_fraction;
```

### Pros
- Physically grounded
- Simple to implement
- No randomness (deterministic)

### Cons
- Doesn't create "swirling" motion, just reduces settling
- Needs good estimate of shear velocity u*
- Particles would still move horizontally with flow but sink uniformly

---

## Option B: Vorticity-Driven Lift (SELECTED)

**See**: `plans/vorticity-suspension.md`

---

## Option C: Stochastic Turbulent Dispersion (DEFERRED)

**Source**: [Random Walk Particle Models](https://www.sciencedirect.com/science/article/pii/S0301932219306366), [Langevin Models](https://link.springer.com/article/10.1023/A:1015614823809)

### Theory
Add random velocity fluctuations proportional to local turbulence intensity:
```
dV = -V/τ_L * dt + √(2σ²/τ_L) * dW

where:
  τ_L = Lagrangian integral timescale ≈ k/ε
  σ   = turbulent velocity standard deviation
  dW  = Wiener process increment (Gaussian noise)
```

Simplified for games:
```
v += randn() * turbulence_intensity * √dt
```

### Implementation Sketch
```rust
// In G2P for sand:
let turbulence_intensity = vorticity.abs() * TURBULENCE_SCALE;

// Add random velocity in both directions
let mut rng = thread_rng();
let noise_x = rng.gen_range(-1.0..1.0) * turbulence_intensity * dt.sqrt();
let noise_y = rng.gen_range(-1.0..1.0) * turbulence_intensity * dt.sqrt();

particle.velocity.x += noise_x;
particle.velocity.y += noise_y;

// Still apply (reduced) settling
let settling_reduction = (-turbulence_intensity * SUSPENSION_FACTOR).exp();
particle.velocity.y += GRAVITY * SETTLING_FACTOR * dt * settling_reduction;
```

### Pros
- Creates natural "swirling" without deterministic patterns
- Physically motivated (eddy diffusion)
- Works well for visual realism

### Cons
- Adds energy to the system (needs careful tuning)
- Non-deterministic (harder to debug/test)
- Requires random number generation in hot loop

### Refinements for Later
- Use Ornstein-Uhlenbeck process for temporally correlated noise
- Scale τ_L by particle Stokes number for inertial particles
- Consider hindered settling at high concentrations

---

## Option D: Full Drift-Flux Model (FUTURE)

**Source**: [OpenFOAM driftFluxFoam](https://help.sim-flow.com/solvers/drift-flux-foam), [Caltech drift flux](https://authors.library.caltech.edu/25021/1/chap14.pdf)

Solve concentration advection-diffusion equation:
```
∂C/∂t + ∇·(C*v_fluid) = ∇·(K*∇C) - ∂(ws*C)/∂y
```

Too complex for current needs. Would require Eulerian concentration field.

---

## References

1. HEC-RAS Rouse-Diffusion: https://www.hec.usace.army.mil/confluence/rasdocs/rassed1d/1d-sediment-transport-technical-reference-manual/bed-change/rouse-diffusion-method
2. Random Walk Models: https://www.sciencedirect.com/science/article/pii/S0301932219306366
3. Langevin Dispersion: https://link.springer.com/article/10.1023/A:1015614823809
4. Ihmsen Diffuse Particles: https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf
5. Shields Parameter: https://eng.libretexts.org/Bookshelves/Civil_Engineering/Slurry_Transport_(Miedema)/05:_Initiation_of_Motion_and_Sediment_Transport/5.01:_Initiation_of_Motion_of_Particles
6. Zhu-Bridson Sand Fluid: https://www.cs.ubc.ca/~rbridson/docs/zhu-siggraph05-sandfluid.pdf

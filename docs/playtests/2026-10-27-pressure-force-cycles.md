# Pressure-force circulation in sustained water failures

The remaining hydrostatic failures contain a velocity component that more scalar-pressure iterations cannot remove. An opt-in trace now reads actual MAC pressure coefficients and predictors, records complete particle energy/momentum budgets, and computes a closed-loop lower bound. The installed game and the physical model remain unchanged. The new measurement narrows the next change to pressure, mass, gravity and transport consistency.

For the measured update `u = u* - k grad(p)`, define the oriented loop sum `Gamma = h sum(sign * u*/k)`. A scalar pressure has zero circulation around a closed cell-centre loop, so `Gamma = h sum(sign * u/k)` after projection. Consequently `max(abs(u)) >= abs(Gamma)/(h sum(1/k))`, independently of pressure iterations. Two CPU controls verify a zero-circulation uniform-mass gravity field and a gauge-independent example attaining the bound. The runtime uses the actual stored GPU coefficients, rather than reconstructing them from particle masses.

The trace checks 642,384 loops over 54 samples. It excludes unsupported cells, the existing floor-slip band and internal-floor tangent damping, so the stored predictor matches the pressure input. Every measured projected field satisfies the bound, with minimum recorded margin zero. This is a sufficient interior obstruction, not a complete boundary proof.

| Recorded endpoint | Exterior, alpha 0 | Exterior, alpha 1 | SDF, alpha 1 |
|---|---:|---:|---:|
| Physical duration | 2 s | 2 s | 0.1 s |
| Grid / cell size | 24 / 5 mm | 24 / 5 mm | 12 / 10 mm |
| Pressure sweeps | 480 | 480 | 480 |
| Pressure relative residual | 1.436e-5 | 1.449e-5 | 3.606e-6 |
| Unavoidable face speed on strongest predictor loop | 14.933 mm/s | 0.813 mm/s | 0.071 mm/s |
| Actual maximum face speed on that loop | 26.966 mm/s | 2.075 mm/s | 0.426 mm/s |
| Gravity-only rest-speed bound on strongest gravity loop | 1.310 mm/s | 0.0217 mm/s | 0.0344 mm/s |
| Maximum particle speed | 35.535 mm/s | 11.186 mm/s | 8.959 mm/s |

Face bounds and particle maxima are different quantities. The gravity-only calculation sets old motion to zero while retaining that step's face coefficients. It demonstrates an equilibrium obstruction for the measured quadrature; it does not establish that every observed velocity comes directly from gravity, or that a different discretisation cannot fix it. The weights belong to the recorded projection's input state. Future motion can change them.

In all sampled steps, collision changes kinetic energy by zero and gather adds no excess over the measured grid-energy change. This rejects a standalone gather/contact energy-injection explanation for these samples. It does not establish a full discrete energy identity for the complete coupled scheme.

The observation is consistent with [Ding, Shinar and Schroeder's MAC-grid APIC study, Section 2.4.3](https://www.cs.ucr.edu/~craigs/papers/2019-mac-apic/paper.pdf), which reports spurious rest currents when using particle-derived grid masses in pressure projection. Their constant-density projection gives up conservation of the complete scheme. That is a relevant warning, not validation of our phase coupling or a ready replacement for it. Real vortices must remain; removing all circulation would destroy the requested riffle physics.

A C0 B2/B1 compatible-spline midpoint water-marker candidate was implemented and tested, with FLIP velocity/affine evolution retained. It passed 13 of 15 runtime functions, but failed the mixed-volume angular budget at 6.287e-6 and force-free angular budgets up to 8.258e-5. The entire trial is preserved and rejected; the gravity matrix was not run after that conservation failure. Its full-runtime divergence property was not independently qualified. This result concerns that position-only integration change, not every possible compatible advection method. [Schroeder, Roy Chowdhury and Shinar](https://www.cs.ucr.edu/~craigs/papers/2021-div-free/paper.pdf) supply the interpolation construction.

A separate proposed local source change was rejected before implementation. In a two-cell closed box with volumes 6 and 10 and reference 8, dividing excess by the reference gives zero summed source. Dividing each excess by its own volume instead gives `(-2/6 + 2/10)/dt`, which is nonzero. A finite-step correction needs the corresponding consistent transport/compatibility treatment; substituting the denominator alone is insufficient.

The retained code adds explicit single-use G2P velocity readback capped at 16 MiB and access to existing phase buffers. Allocation/copies occur only on request. Phase status must be checked separately; a captured raw buffer does not certify a successful step. The example accepts bounded trace cadence and optional case selectors, refuses output overwrite, and refuses an empty case selection. Default execution still runs all 12 original cases with the original gates.

Fifteen complete case records match the prior committed solver exactly: all 12 default short exterior cases pass, while the two selected sustained exterior cases and one SDF case remain failed. The 15 real-runtime controls, five ordinary pipeline controls and two CPU bound controls pass. A full ordinary replay preserves the complete 31.7 MB canonical saved state, including 116,139 water particles, 4,430 grains, material and contact history; only the three unordered contact-history maps are sorted. Its SHA256 remains `e6934203145f8f31f1ddb61d6fc1ccff5b0ad6a5a32caa25723c0bd7a827256d`. Strict maintained-crate Clippy, formatting and whitespace pass. Engine Clippy has 98 existing warnings and none in changed spans. The unchanged 96-case gravity matrix is not rerun or claimed as passing.

The next physical design must reconcile geometric/represented volume, inertial masses and pressure/gravity work while retaining linear/generalized angular momentum, full-gravity freefall, phase feedback and real vortices. Qualification still requires the existing long equilibrium, mesh/particle/timestep, dynamic energy, SDF/bed and washout gates. Conservative mode remains opt-in. The previous crest-rendering app is unchanged; no fresh native FPS, visual improvement or real-time result is claimed.

The [receipt](2026-10-27-pressure-force-cycles.json), [sample CSV](2026-10-27-pressure-force-cycles.csv), full input-state witnesses, rejected sources and bounded command logs preserve the result. Original dirty files and the user stash remain intact. Heavy checks ran sequentially; the full playable-game objective remains open.

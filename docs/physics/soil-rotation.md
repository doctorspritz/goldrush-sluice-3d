The MountedShellV2 dry-soil experiment integrates free rotation between contact impulses with a coupled exponential midpoint update. This corrects a reproduced 20.589 degree principal-spin lag and reduces the old first-order asymmetric-body drift. Ordinary capped bodies, old bucket models and the water/riffle/terrain solvers retain their existing paths.

For a free interval, world angular momentum `L` is constant. Given principal-frame orientation `R` and principal inertia `I`, angular velocity is `R * I^-1 * R^-1 * L`. The solver applies actual contact impulses first, then uses up to eight fixed-point iterations for the midpoint orientation and advances by a quaternion exponential. The endpoint angular velocity is reconstructed from that same momentum and pose. Intermediate calculations use f64 to avoid accumulating rounded principal-frame transforms for tiny SI inertias; canonical engine state remains f32. Zero spin returns the exact input orientation.

Inner angular stages use the bound `|L| / I_min`. Their angle target is 0.1 rad at the default 240 Hz mechanical rate and scales with the full physical timestep. This scaling matters: a fixed angle cap can reduce the inner stage count when the mechanical timestep is halved, masking refinement by keeping the same effective angular interval. Physical time and angular velocity are never clipped by this refinement. The internal work ceiling is 512 stages; this is a finite numerical budget, not a claim of universal accuracy for arbitrarily fast or degenerate bodies. Contact and CCD evaluations still follow their existing cadence; these angular stages alone do not prove fast-contact collision convergence.

The model is checked against an independent f64 quaternion RK4 reference at constant world momentum. That reference is separately refined from 4,096 to 8,192 stages for ordinary tumbling and from 16,384 to 32,768 for fast tumbling. It does not call the production midpoint update. The reference uses each clod's full geometry-derived inertia rather than the solver's reconstructed principal tensor.

| 0.5-second free-motion control | Mechanical rate | Orientation error | Relative energy error | Relative angular-momentum vector error |
|---|---:|---:|---:|---:|
| Asymmetric, 31.016 rad/s | 240 Hz | 9.213e-5 rad | 6.050e-7 | 2.999e-7 |
| Asymmetric, 31.016 rad/s | 480 Hz | 2.245e-5 rad | 1.178e-6 | 7.897e-7 |
| Asymmetric, 31.016 rad/s | 960 Hz | 5.742e-6 rad | 4.588e-7 | 5.140e-7 |
| Asymmetric, 186.097 rad/s | 240 Hz | 1.424e-3 rad | 6.171e-7 | 2.173e-7 |
| Asymmetric, 186.097 rad/s | 480 Hz | 3.573e-4 rad | 1.601e-7 | 4.642e-7 |
| Asymmetric, 186.097 rad/s | 960 Hz | 1.056e-4 rad | 5.219e-7 | 3.285e-7 |

The 200 rad/s principal-axis control has a 4.998e-6 rad phase error at 240 Hz, versus the old 0.359348 rad, while retaining its checked kinetic energy exactly. Asymmetric energy is not claimed mathematically exact: phase refines approximately quadratically and f32 invariant rounding is bounded separately. Coarse 120 Hz ordinary tumbling is retained as a checked control. Unforced zero-spin clods preserve exact poses through 120 world rebuilds, including a tilted orientation.

Checking the direction of angular momentum matters independently of its norm. [Simo and Wong (1991)](https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620310103) discuss this distinction. Their energy-momentum algorithm and unconditional-stability claim are not attributed to this implementation; the controls above establish the measured scope of this model.

This update does not calibrate aggregate soil strength, cohesive/wet failure, compaction, excavation yield, high-speed contact sweeps or machine hydraulics. Those requirements and complete Stages 2–6 remain open in the [operation plan](../../plans/2026-10-22-realistic-operation.md). The [rotation checkpoint](../playtests/2026-10-23-soil-rotation.md) records contact, persistence, native and matched-bank evidence with the source and binary hashes.

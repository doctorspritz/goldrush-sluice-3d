# Conservative local phase response

The locally implicit phase response now runs through the actual `GpuFlip3D` step and hydraulic worker in explicit validation mode. Launch the native build with `--conservative-phases`, or construct `Hydraulics::new()?.with_conservative_phases()`. Ordinary gameplay still selects the existing phase equations. The full coupling, volume-constraint, packing and sluice-calibration issues remain open. [The foundation evidence](../playtests/2026-10-23-phase-response.md) and [runtime evidence](../playtests/2026-10-23-phase-runtime.md) distinguish local equations, full-step conservation and actual native flow.

The existing solver has three independently reproduced momentum errors: fixed-point PIC transfer bias, pressure acceleration that omits the actual lumped face mass, and grain drag without an equal water reaction. Thirteen zero-gravity cases isolate transfer, pressure, dry grains, wet grains, timestep and coincident clouds. A global momentum subtraction would conceal these local errors; it is not used here.

## Local equations

Quadratic APIC transfer represents each particle on the same staggered MAC faces used by gather. A constituent at a face has volume `Vi*Wi`, relative density `rho_i`, mass `rho_i*Vi*Wi`, and affine velocity `vi + Ci*(xf-xi)`. Water has relative density one. Body predictors use full gravity. Pressure, rather than a reduced gravity shortcut, supplies buoyancy.

For each wetted grain, freeze a nonnegative relaxation rate over the substep and define `a=dt*gamma`, `b=a/(1+a)`, `c=1/(1+a)`. Store `b` and `c` separately so stiff coupling does not lose the small residual through subtraction. The current diameter/fallback rates preserve the existing Ferguson–Church reference law; a nonlinear, porosity-dependent drag closure is not yet established by these tests.

Each face accumulates eight moments:

| Moment | Meaning |
|---|---|
| `Vw` | Water volume |
| `Pw` | Water affine momentum |
| `Vs` | Wetted grain volume |
| `Bv` | Sum of grain volume times `b` |
| `Bm` | Sum of grain mass times `b` |
| `Bp` | Sum of grain mass times `b` times body-predictor velocity |
| `Rv` | Sum of grain volume times `c/rho` |
| `Rp` | Sum of grain volume times `c` times body-predictor velocity |

Let `V=Vw+Vs`, `A=Vw+Bv`, `M=Vw+Bm`, and `K=Vref/V`. `Vref` is the reference represented volume per cell, normally eight in the existing relative particle units. After water gravity/flow/boundary predictors:

```
u0 = (Vw*water_predictor + Bp)/M
q0 = (A*u0 + Rp)/V
kw = K*A/M
k  = K/V * (A*A/M + Rv)
```

The pressure equation is `div(k grad p)=div(q0)`. All reference, packed and cached-stencil solvers use the same face coefficients as the pressure-gradient application. Air has zero pressure; solid neighbours contribute no coefficient. Zero-mobility faces do not introduce a division by zero.

If `G` denotes minus the pressure gradient with timestep absorbed, the face responses are:

```
water_new = u0 + kw*G
grain_new = c*grain_predictor + b*water_new + c*K/rho*G
mixture_new = q0 + k*G
```

These formulas are our algebraic elimination of the coupled backward-Euler equations. An independent double-precision dense solve verifies them; they are not a claim of reproducing another paper's validated implementation. Local drag impulses cancel. Summed pressure impulse is `Vref*G`, permitting cancellation across neighbouring cells. Co-moving phases in freefall remain co-moving. The reference also checks hydrostatic Archimedes settling, subdivision, a stiff limit and the exact backward-Euler energy-loss identity.

Gather samples the same virtual face response for each grain and reconstructs its APIC rows. Water retains its FLIP/PIC blend using its own pre-force water plane. The full phase-force increment enters that blend. Dry grains and faces without water retain the full-gravity grain predictor. This is not yet a validated partially saturated interface model.

## GPU representation and refusal

Portable logical signed-64 sums use two atomic u32 words and an integer carry. A scale of `2^36` resolves thin-support moments more accurately than the initial `2^30` prototype. Readback converts the signed integer magnitude before float conversion. Positive/negative, tiny-value, cancellation and reversed-order controls execute on the real GPU.

Every individual contribution is checked against `2^26 / actual_particle_count` units. Therefore each face moment's absolute sum stays within `2^62` fixed-point units, with a factor-two signed-overflow margin. The bound depends on live count, not reserved capacity. This is intentionally conservative; a rejected value must pause the operation rather than be silently dropped.

The status bitfield reports accumulator input/range failure (1), invalid particle/rate metadata (2), invalid assembled response (4), or non-finite transient particle output (8). IEEE float-exponent inspection detects NaNs and infinities without depending on floating comparison optimisation. The refusal flag stays sticky across GPU-only substeps until a fresh solver is constructed. A 16-byte tail joins the ordinary position/velocity/APIC readback; its flag is inspected before any CPU output is copied. Position repair and SDF advection are guarded. The worker pauses on refusal, keeps material in the authoritative Session and drops transient GPU/DEM state. Reloadable saved-state and paused-tick controls use a valid finite saved affine field that exceeds the accumulator request bound.

Metadata occupies four capacity planes: density, diameter, `b`, `c`. Each water buffer retains the original velocity plane, a status header and an aligned response plane `[u0, kw/k, q0, K/k]`. Pressure mobility has a GPU-ready header plus face coefficients. Clear/publish dispatch ordering permits constant-mobility position repair followed by variable-mobility phase pressure in one submission. Ordinary constructors retain their original two-plane metadata and do not allocate the new phase grid. Pipeline specialisations compile the inactive phase/mobility branches out of their shaders.

`PhaseTransfer3D::storage_bytes` includes moment storage, aligned auxiliary/cell planes, mobility headers and uniforms. Constructor checks precede allocation and cover binding counts, buffer sizes, face/moment indices, particle inputs and metadata. The runtime planner includes this fixed grid, eight extra metadata bytes per reserved particle and both readback tails. This grid budget is not a peak residency or process-RSS guarantee. Mode, tunables, obstacles and pressure state survive storage growth; derived phase fields are reconstructed from the complete authoritative reupload.

## Runtime volume and remaining accuracy gates

The validation path scatters actual represented water and solid volume with the quadratic cell-centred kernel. Raw water home-cell markers and total volume occupy separate aligned planes in the existing U buffer, preserving the sixteen-binding limit. Voxel solid support uses exact one-dimensional kernel integrals `2/3, 1/6, 1/6`, including edges/corners and closed exterior boundaries. Free surfaces do not create density suction. After position repair, volume/markers are recomputed and the actual phase transfer uses the repaired coordinates.

Only occupied water home cells carry pressure unknowns in this mode. The legacy expansion halo can create pressure faces without carrier support, preventing the intended pressure-impulse cancellation. Existing solid cells, riffle SDF and DEM contacts remain. The old hard bed-occupancy mask, one-way porous damping and separate grain repair are not added to the explicit phase forces in validation mode. This does not yet establish a valid dense-bed permeability or rheology law.

The actual full-step probes preserve linear momentum with repair enabled/disabled at 120/240 Hz. They also expose a remaining defect: position-only volume repair changes APIC angular momentum by about `6.57e-6` in one probe, versus about `2e-7` without repair, with no improvement from halving timestep. `sluice-volume-constraint` tracks this failed accuracy gate. A global angular correction is not an acceptable substitute for a consistent constraint.

Before default activation:

1. Correct the volume/pressure constraint and prove angular/energy, lake-at-rest, wall/free-surface and mesh/particle/timestep convergence. Instrument numerical clipping in full operating trials.
2. Establish a voidage/Reynolds-dependent drag closure, partial wetting and packed-bed contact/volume convergence. Do not restore one-way damping or count-based hard bed masks as a substitute.
3. Validate pure water, single grains, porous blocks and real riffles; run actual slope/flow/feed/size washout and cleanup assays with saved-state continuity.
4. Review native motion and measure render FPS separately from simulation speed. Real-time operation and complete game stages 2–6 remain open.

Primary research informing the validation sequence: [Robinson, Luding and Ramaioli](https://arxiv.org/html/1301.0752v2) examine volume-averaged phase coupling and settling/porous cases; [Bai and Stone](https://arxiv.org/abs/1005.4980) address stiff feedback; [Stoyanovskaya and colleagues](https://arxiv.org/abs/1811.06506) examine implicit drag-in-cell integration. Those results do not validate this APIC discretisation, packed soil or sluice recovery.

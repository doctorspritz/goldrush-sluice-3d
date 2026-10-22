# Conservative local phase response

This is a verified GPU **foundation**, not the active playable solver. `GpuFlip3D`, the worker and ordinary saves still use the existing phase equations. The full coupling, packing and sluice-calibration issues remain open. The [23 October evidence](../playtests/2026-10-23-phase-response.md) distinguishes the tested transfer/pressure/gather harness from the runtime integration still required.

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

The status bitfield reports accumulator input/range failure (1), invalid particle/rate metadata (2), or invalid assembled response (4). Failed gather leaves velocities and affine rows intact. This alone is not worker failure handling: joined status readback, worker pause/material custody and guarded advection remain mandatory before runtime activation.

Metadata occupies four capacity planes: density, diameter, `b`, `c`. Each water buffer retains the original velocity plane, a status header and an aligned response plane `[u0, kw/k, q0, K/k]`. Pressure mobility has a GPU-ready header plus face coefficients. Clear/publish dispatch ordering permits constant-mobility position repair followed by variable-mobility phase pressure in one submission. Ordinary constructors retain their original two-plane metadata and do not allocate the new phase grid. Pipeline specialisations compile the inactive phase/mobility branches out of their shaders.

`PhaseTransfer3D::storage_bytes` includes moment storage, aligned auxiliary planes, mobility headers and uniforms. Constructor checks precede allocation and cover binding counts, buffer sizes, face/moment indices, particle inputs and metadata. This grid budget is not a peak residency or process-RSS guarantee. The runtime particle planner does not yet include the phase allocation because the runtime does not construct it.

## Integration still required

1. Integrate this operator into the real `GpuFlip3D` step after particle-position repair, not only this harness. Count actual represented water/solid volume for repair and derive a matching fluid mask; resample counts and momentum after positions move.
2. Share metadata and auxiliary buffers through construction, storage growth and restored pressure state. Join failure status to existing readback without a per-step CPU fence, and guard downstream advection and worker material custody.
3. Replace old grain slip/porosity damping only with the explicit phase forces and validated voidage/drag closure. Do not double-count either force or buoyancy.
4. Validate pure water, single grains, porous blocks, packed beds and partially wet regions; sweep grid, parcel subdivision and timestep. Retain DEM contacts and actual riffle geometry. Compare a complete runtime budget with this harness.
5. Run actual slope/flow/feed/size washout and cleanup assays, live save/restore, native flow review and separate render-FPS/simulation-speed measurements before changing the default. Stages 2–6 of the full operation remain open.

Primary research informing the validation sequence: [Robinson, Luding and Ramaioli](https://arxiv.org/html/1301.0752v2) examine volume-averaged phase coupling and settling/porous cases; [Bai and Stone](https://arxiv.org/abs/1005.4980) address stiff feedback; [Stoyanovskaya and colleagues](https://arxiv.org/abs/1811.06506) examine implicit drag-in-cell integration. Those results do not validate this APIC discretisation, packed soil or sluice recovery.

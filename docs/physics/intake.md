# Conservative pump-to-inlet transfer

The terrain solver stores four solid/water volume concentrations: sediment,
overburden, gravel and paydirt. Suspended gold is a separate mass tag in grams.
The previous pump lowered water depth while keeping concentration fixed, then
credited only water to the sluice. It silently removed `C × withdrawn water`
solids and left the gold tag behind.

The intake now samples a compact 3×3 set of 25 cm terrain cells. Each wet donor
supplies the same fraction of its available water. This avoids preferentially
emptying the first corner of the previous 7×7 stencil. It remains a coarse
finite-volume boundary, not a resolved strainer/suction-pressure calculation.

For each donor, the represented withdrawal is `ΔV = (h_before − h_after) A`.
The requested surface decrement rounds toward retained water, so a request
smaller than one representable surface increment cannot create pump credit.
Each material transfer is `C_k ΔV`; gold transfers by `ΔV / V_before` of the
donor's suspended tag. Ground gold is untouched. The remaining water keeps its
concentration and velocity, so its stored volume and horizontal momentum both
fall with the withdrawal. CPU edits mark the existing GPU upload path dirty.

`Sluice.inlet.ready` owns the withdrawn suspension while its carrier is in
`water_credit_m3`. When water particles actually enter FLIP, the matching
fraction of suspension moves into an inlet grain accumulator. It emits actual
weighted DEM parcels at the water inlet, independently of the hopper feed.
The existing drag, buoyancy, APIC/FLIP, contacts, riffles and outlet handle them.
There is no colour-only sediment injection or delivery into the hopper.

The four physical diameters match the terrain shader's D50 values: 0.1 mm,
1 mm, 10 mm and 2 mm. Gold uses an **assumed 0.1 mm diameter**, not a measured
claim grading. Each emitted parcel is capped at the normal parcel volume, or
one physical grain if the research override is smaller. Admission cycles fairly
through the classes, with at most 32 parcels per fluid update; dilute material batches for at most 0.25
simulation seconds while capacity is available. Fractions smaller than one
physical grain stay explicitly in the accumulator, including trace gold.

The existing three-component `Material` convention maps sediment to sand,
overburden to soil, and gravel plus paydirt to gravel. The four classes remain
separate in the inlet's custody and diagnostic counters. This does not add
porosity, grain-mass conversion, clay cohesion or a new soil calibration.

Full water/solid capacity applies backpressure **before** modifying the river.
Unemitted water and solids remain stored. Source starvation supplies only actual
available water; the HUD shows delivered versus requested L/s. Saves retain the
carrier, pending material, sequence and custody counters. Missing inlet state in
old saves defaults to an empty buffer; their existing water credit is retained.
Negative suspension and inconsistent saved inlet custody are rejected.

The finite particle/buffer capacity is a numerical custody safeguard, **not a
calibrated clogging or pump law**. Sustained dense input can still reach that
limit. Maintaining delivery with physically derived restrictions and a validated
dense-phase representation remains part of the full simulation goal.

This stage does **not** model hose hold-up/priming, pipe pressure or cavitation.
The old specified inlet velocity is retained; pump/nozzle forces are external
boundary conditions, not a closed total-momentum budget. Worker/fluid scheduling
and the grain accumulator introduce numerical admission delay, not a physical
hose residence model. Terrain trace-gold advection still needs an audit against
actual substep material fluxes. Discharged grain tailings are a finite inventory,
not yet a constructed settling pond. These remain requirements of the complete
operation; passing this handoff is insufficient to close `sluice-intake-solids`.

Run `mining-physics/examples/intake_probe.rs` for a controlled initial-value
experiment through the production GPU terrain and FLIP/DEM solvers, including
live save/restore, per-component inventories and actual grain transport. The
native flow benchmark separately checks total bulk solids at the existing
one-part-per-million stage tolerance, water/gold ledgers, rendering FPS and
simulation/wall speed. Neither a short pulse nor a global sum validates local
gold transport, calibrated recovery or long-duration pond operation.

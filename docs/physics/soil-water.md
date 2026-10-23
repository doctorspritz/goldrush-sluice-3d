# Wet soil inventory and material custody

The soil foundation now carries finite water in the current reference pore
space. It makes wet excavation, hauling, redeposition and feeding conservative.
It does not complete the soil continuum, compaction or groundwater-flow stage.

A column's reference bulk volume comes from its four existing strata. True
mineral and gold volume follow the [mass basis](material-mass.md). Their
difference is reference pore volume. This keeps the inherited dry densities and
fixed geometry explicit; porosity cannot yet change under compaction. The new
claim's already ponded columns start saturated, with other columns dry. This is
a stated initial condition, not an infiltration calculation. Version 2/3 saves
start with zero explicit soil water, preserving their recorded inventory.

`SoilWater.pore_m3` owns physical water per column. Excavation samples its column
saturation and carries the corresponding removed void water in
`Material.pore_water_m3`. This is a well-mixed column approximation; it does not
resolve a vertical moisture profile. Material splitting, hand/machine buckets,
falling loads, hopper admission and overflow preserve that water. Dry mass and
grain-solid volume exclude it. Machine hydraulic load uses dry material plus
1000 kg/m³ water; bucket/hopper material colour and roughness vary continuously
with carried saturation. Those colours are presentation approximations.

Deposits return moisture to finite pore capacity. Bed erosion on the current
column model releases water when its supporting void space disappears. This is
a drained response: pore water is not yet advected with a moving failure block.
Expelled water dilutes existing surface suspension and horizontal momentum;
trace-gold mass remains unchanged. A positive `surface_m3` store retains real
water too small for the float surface to represent. It is counted in the water
ledger and is flushed using actual represented depth increments. It is not a
material-error correction or an uncounted water source. The same queue retains GPU depth that cannot be represented as a World surface height and finite outlet impact water. `surface_momentum_m4_s` preserves horizontal momentum belonging to that queued free water; pore water is stationary. Four-class suspension rounding uses a counted mobile precision remainder.

Hopper feeding separates moisture from grain solids into saved
`feed_water_credit_m3`. Shared particle capacity is reserved before taking the
wet load. Moisture enters actual FLIP particles at the paydirt chute, with the
existing paydirt release velocity. River intake retains its separate pump-water
carrier and four-class custody; moisture must not change that mixture's
admission fraction. The original water, contact, riffle and terrain equations,
resolution and timesteps remain. Chute velocity is still a prescribed boundary,
not an energy-conserving free-falling hopper model.

The complete water ledger includes soil pores, unresolved expelled surface water,
carried moisture, flowing channel water and world surface water. Sluice flowing
water accounting starts when moisture is released from the hopper, so retained
hopper moisture is not counted twice. The existing dry/reference-bulk and gold
ledgers remain separate. Moisture was introduced in version 4. Ordinary saves are version 6; unexchanged articulated validation writes version 7; spatial material state writes version 8. Older binaries reject unsupported versions;
current binaries accept versions 2–8 and retain the previous bed-age migration.
A legacy version carrying new moisture is rejected rather than silently losing
water in an old reader.

Dynamic infiltration and seepage, variable porosity and compaction, capillary
retention, effective-stress/pore-pressure strength, wet failure-block transport,
fine physical excavation and constructed pond/pad behavior remain required.
Those must replace these initial-condition and drained-column approximations.
For example, a [Green-Ampt infiltration model](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.5/overview-of-optional-capabilities/modeling-precipitation-and-infiltration/green-ampt)
requires hydraulic conductivity, suction and initial moisture; it also assumes a
simplified front. No such calibrated infiltration or Richards solution is
claimed here. Drying/rewetting of sediment-bearing thin films must be closed
before adding water sinks; deposition now operates at every positive water depth using [counted precision stock](subgrid-bed.md). Under-1-mm rough-bed scour and continuous sloping shoreline convergence remain unresolved. New submerged deposits gain void capacity without automatic saturation: the current model does not draw surface water into that new capacity until a conservative infiltration model exists.

The [machine-contact save contract](machine-contact.md) documents ordinary version 6, optional articulated version 7 and stationary legacy-machine migration. The material and terrain-clock rules above remain in force.

The [spatial material save contract](payload-exchange.md) adds version 8 for payload moments, rigid parcels and boundary receipts. Existing material and terrain-clock migrations remain unchanged.

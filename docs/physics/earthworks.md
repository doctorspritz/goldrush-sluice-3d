# Swept earthwork protocol

The playable excavator removes material intersected by its moving bucket floor and tooth tips. Holding cut on a stationary tool no longer drills a circular patch. Space enables cutting while the operator moves boom/stick/bucket joints; curling can also collect material. The HUD reports soil-force demand and whether ground, bedrock, capacity, force or power limited the stroke. A passive bucket contacts the surface; the tool can retreat from contact. The hand load remains deliberately unbounded at the user's request.

The bucket floor width is 0.82 m. Its heel and tooth offsets match the rendered floor and tooth-tip geometry. A stroke uses a triangulated swept floor, including its side faces. Endpoint-only samples missed the last cut when a tooth left a column, producing about 9% timestep dependence in the first trial. The continuous sweep removes that defect. Material comes from finite 25 cm terrain columns, in the same order as the GPU geology: sediment, overburden, gravel and paydirt. Paydirt's trace gold follows the actual removed fraction. The water depth in each changed column is preserved. Hand digging uses the same local accounting, including gravel, instead of cloning four entire arrays and scanning the whole claim.

The fixed layer arrays still cannot place a new gravel layer above existing overburden. Resolving deposited layer order and physical local soil geometry is a Stage 2 requirement. A fresh machine parks its floor/tooth points at least 5 cm above the bank; old saves retain their pose and can retreat from an existing contact. The bundled flowing demo changes only its idle machine boom from 0.48 to 0.56 radians, preserving all water, grain and terrain state. Save parsing enables correctly rounded floating-point restoration so exact trajectory continuation is not lost by a one-bit material-value change.

Cuts are previewed before motion is accepted. Bedrock, remaining capacity, soil force and available tool work restrict the stroke. No material is removed for rejected travel, including chassis movement rejected by plant obstacles. The floor/tooth contact skin is fixed at 0.1 mm; it is not added each timestep. The carried 0.28 m³ limit applies to the machine bucket. Payload lift work and conservative floor travel contribute to the power budget.

The soil model integrates the level-ground passive pressure approximation:

`Kp = tan(pi/4 + phi/2)^2`

`F = width * (rho*g*depth^2*Kp/2 + 2*cohesion*depth*sqrt(Kp))`

The [SCDOT geotechnical manual](https://www.scdot.org/content/dam/scdot-legacy/business/pdf/geotech/2022-by-chapter/Chapter18-EarthRetainingStructures-12032021.pdf) gives the passive-pressure expression. [Servin et al.'s earthmoving model](https://doi.org/10.1186/s40323-021-00196-3) explains the broader cutting-force formulation, active soil zone and its limitations. This implementation is a simplified vertical-blade resistance envelope. It does not solve the general Reece factors, failure wedges, pore pressure, continuum soil or multibody hydraulics.

Generic material assumptions are explicit: overburden cohesion 12 kPa and friction angle 28°, loose sand 0 kPa / 30°, gravel 1 kPa / 36°. Bulk densities use the domain material's existing mass convention. Soil conditions need measured calibration. The force envelope is 41.3 kN for crowding and 58.9 kN for bucket-dominated movement, using the [Cat 308 CR reference](https://www.cat.com/en_US/products/new/equipment/excavators/mini-excavators/127700.html). These are not this linkage's measured cylinder capacities. The 30 kW tool-power budget is an assumption; fuel, track reaction, tipping and hydraulic leverage are not calibrated here.

Run the domain checks with one compiler job:

```bash
cargo +nightly test -p mining-sim --profile playable --jobs 1 --locked -- --test-threads=1
```

The new cases check stationary tools, passive/rock contact and retreat, force/power/capacity limits, layer/gold/water accounting, deposited gravel, save continuity/defaults, blocked chassis travel, an actual crowd/curl/dump replay, and the same stroke at 30/60/120 Hz. The timestep gate is 0.1% relative volume. That is not a terrain-resolution convergence test.

Native excavation verification uses the original GPU terrain flow and an idle sluice:

```bash
SLUICE_SAVE_PATH=/tmp/earthwork-check/shift.json \
  ./target/playable/mining --validate /tmp/earthwork-check --earthworks-only
```

Use a fresh output directory and the matching built binary. The fixed inspection camera changes presentation only. Replay commands use the actual machine axes, finite loads and falling-load/hopper capture. Live save/restore occurs during the next cut. Screenshots and receipts must finish before a successful exit. `earthworks.json` explicitly limits its claim to excavation and delivery; `render-timing.json` reports active physical time separately from FPS. The soil inventory includes all mobile terrain/suspension, carried/resident material and exported material, with measured bedrock change as a source. Its native reporting tolerance is one ppm of the whole claim; isolated cut tests require 1e-6 m³. Water/gold gates are 50 mL / 1e-6 g.

The 25 cm physical heightfield still limits trench edges and bucket contact, regardless of finer render tessellation. Loads remain lumped volumes: there is no resolved granular failure into the bucket, overhangs, compaction, saturation-dependent strength, entrained bucket water or realistic spill/retention law. Boom/body contact and track dynamics still need work. These remain requirements in `sluice-local-earthworks` and the [full staged plan](../../plans/2026-10-22-realistic-operation.md).

The default small sluice feeds 0.1 L/s. Washing a roughly 0.58 m³ load takes about 97 simulated minutes, so this earthwork check does not establish completion of the full new-shift wash/cleanup loop. Capacity, plant scaling, long-operation pacing, conservative phase exchange and calibration remain open. Preserve the original fluid/riffle fidelity while addressing them.

Terrain collapse uses source-owned eight-neighbour transfers and a separate gather, retaining the existing repose angles and rate. The former red/black pattern raced at shared receivers and diagonal neighbours. Tiles include a one-cell receiver halo and ignore inactive source transfers. The derived transfer buffer costs 128 bytes per cell (2 MiB on the playable claim); no scratch is serialized. The clean river source retains the existing footprint and rate, dilutes all four concentration channels and books the water volume actually represented after surface rounding. The old source created suspended dirt by increasing water depth while keeping concentration fixed.

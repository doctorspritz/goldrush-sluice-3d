# Solver throughput investigation

No experimental implementation is adopted. All 736 initial Rust/WGSL sources are restored byte-for-byte. The restored probe executable also matches its original SHA-256, and the installed signed app remains unchanged. The investigation changes the next optimization decision; it does not establish a speedup or complete the realistic-game objective.

The latest native packet measurement, retained from the preceding stage, spends about 40.0 ms in fluid/readback, 33.9 ms in DEM and 3.5 ms in terrain. Its phase order is preparation, fluid, DEM, finish, terrain. Native rendering/physics were not remeasured here; the prior means remain 55.78 FPS and 0.38659x simulation/wall time. The unchanged app hash is `82f3720c0330cad60ce0770ad003a0a24830dd875b8477e6a4518b51f8e9d15d`.

Fresh headless measurements use the same 117,544-water/4,308-grain save, 12 warm-up domain ticks and 120 measured ticks. These are 60Hz domain-tick units, distinct from native packets and 120Hz fluid/contact substeps. The ordinary first baseline advances at 0.43349x wall time: preparation 1.464 ms, fluid/readback 15.842 ms, DEM 16.670 ms (pairs 14.819 ms, SDF 1.824 ms), finish 1.279 ms and terrain 1.709 ms per tick. Time outside those spans remains part of the total probe wall time.

Batched actual-pass GPU timestamps identify the following largest compute spans per domain tick. They do not insert a fence at every stage. Their complete final state matches the ordinary baseline; their wall timing is diagnostic and is not substituted for normal/native timing.

| Actual GPU pass | Device ms per domain tick |
|---|---:|
|P2G 3D Scatter Pass|2.674|
|Velocity pressure sweeps|2.611|
|G2P 3D Pass|2.356|
|Density pressure sweeps|1.400|
|SDF Collision 3D Pass|0.618|

The sum of all reported GPU compute spans is 11.273 ms; the instrumented fluid/readback wall span is 16.211 ms. Those different boundaries do not isolate a precise transfer/wait cost. Pressure sweeps, particle counts, 1cm grid and original terrain stages are retained.

A three-second CPU sample targets only the verified task-owned baseline probe. Its main-thread stack repeatedly visits the contact candidate loop; other substantial spans sleep while awaiting fluid results. The sample does not establish cache-miss or branch-stall rates. Further attribution needs targeted data, rather than naming memory latency from stack samples alone.

An explicitly instrumented 132-tick run, including 12 warm-up ticks, counts 217,064,138 candidate-loop checks. Of those, 77.4% have one active lane. There are 7,515,009 pair calls; 547,749 leave the source position unchanged and 6,967,260 change it. The diagnostic counters preserve the full baseline state. Their timing is deliberately excluded from performance comparisons.

| Candidate | Baseline pair ms | Candidate pair ms | Headless throughput change | Decision |
|---|---:|---:|---:|---|
|mask interval updates|14.912|14.833|+1.06%|Rejected|
|early separation gate|14.885|14.813|-0.35%|Rejected|
|portable integer lane mask|14.793|14.950|+0.40%|Rejected|
|explicit NEON lane mask|14.973|14.910|+0.13%|Rejected|
|single-lane scalar rejection|14.852|18.066|-9.97%|Rejected|
|two radius lattices|14.857|18.165|-8.81%|Rejected|
|target-expanded memberships|14.950|15.945|-2.76%|Rejected|

Each comparison uses paired same-input fixed-step runs, with before/after/after/before ordering. The first two experiments reuse the initial baseline as their first control; later experiments use fresh controls. Small changes overlap the observed run spread. The slower experiments are clear regressions. A shorter instruction sequence alone did not produce a useful solver improvement. The explicit NEON experiment used bounded local arrays and the standard [load](https://doc.rust-lang.org/core/arch/aarch64/fn.vld1q_u32.html) and [integer reduction](https://doc.rust-lang.org/core/arch/aarch64/fn.vaddvq_u32.html); its extra implementation complexity was removed along with the other trials.

Every recorded comparison ends with the complete canonical state `e6934203145f8f31f1ddb61d6fc1ccff5b0ad6a5a32caa25723c0bd7a827256d`, at 92.60000482946634 physical seconds, 116,139 water particles and 4,430 grains. Only the three unordered contact-history arrays are sorted by their integer tuple keys. No physical array is reordered, no field is removed and no float tolerance is used; signed zero is retained. A verified compressed full state, raw-state hashes, binaries and rejected source snapshots remain in the local artifact directory.

Candidate library commands pass between 138 and 140 checks, including independent ordered-contact reference comparisons, rotational response, cache movement and geometry/budget controls where applicable. A test-only index-type compile error in the first two-layer attempt was corrected before its successful tests and measurements. These candidate checks are not presented as a new native release gate. No full app suite, manual controls test, native capture or five-case GPU integration suite was rerun after rejecting every implementation. The final restoration check rebuilds the original probe to the identical executable hash `736b9341a5598fd3720735b752958b939a77560f2a43fa89df3c140ca7b454c3` and reruns its full-state control successfully.

The next search redesign needs target-motion, memory-locality, branch and visitation evidence. Stable-ID spatial packing or a bounded hierarchy are candidates to evaluate, not accepted solutions. Pure search changes must retain physical state and order; new physical-law work needs its own conservation, work, convergence and calibration gates. Existing mechanical-energy and real-time failures remain open.

Tracked next work: `menace-ny6`; `sluice-terrain-throughput` and `sluice-realtime` remain in progress. [The receipt](2026-10-25-solver-investigation.json) pins the source/input/binary hashes, phase data, complete trial measurements and custody checks. The original dirty checkout and user stash are preserved. Jobs ran sequentially, no new panic-full or WindowServer report appeared, and scoped cleanup left zero owned test browsers or compiler/game jobs. Sixteen shared Playwright MCP and 64 Node processes remain untouched. Installed Beads rejects `bd sync`; the scoped 231-row export preserves 229 unrelated literal rows.

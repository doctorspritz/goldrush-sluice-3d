# Water surface cost, display lag and isolated outlet spray

28 October 2026. Baseline `7be7ff695ebe`. Render-only stage of `menace-7rn`; complete mining operation and real time remain unfinished. [Machine-readable evidence](2026-10-28-surface-latency.json).

## Result

The surface worker now keeps one active reconstruction and the newest pending render snapshot. A newer pending copy replaces an older pending copy, while completed outputs remain bounded and ordered so their particle-pair interpolation endpoints remain correct. Epoch changes reset pairing. No second worker, authoritative state deletion or new water inventory is introduced. Queue wait and coalescing are reported separately from build time.

Covariance moments use aligned SIMD storage while retaining every scalar accumulation order. The same five ordered Jacobi sweeps use constant plane indices. Original scalar/eigen implementations remain independent test controls; no reduced sweep count, pressure schedule, riffle/contact simplification or shader change is included.

Sparse outlet support now approaches the sphere radius of its represented water. Occupancy alone was insufficient: a first trial turned a dilute sheet into visible glitter despite passing volume checks. The accepted version preserves original support once the existing covariance neighbourhood has twelve samples, blending smoothly below that count. Channel support, fully connected outlet radius and endpoint optical volume remain controlled. This is a small isolated-spray correction; the dense outlet still uses the legacy nearest-ellipsoid reconstruction and has visible beading/stepped edges. It does not complete joined outlet rendering.

Native capture now waits for current-epoch presented water, and for its exact clock when frozen. The initial final-candidate sluice frame exposed a race: an empty clock-zero render was reported alongside the restored wet clock-eight state. New negative controls reject empty/unknown surface evidence; every retained final wet frame reports readiness and nonzero presented particles. A compiled pipeline alone is insufficient evidence.

## Measured scope

One finite dug-feed/shaker/sluice snapshot contains 55,608 channel markers, 5,503 outlet markers and 758 physical grain parcels. Sixteen fixed-state CPU rebuilds give these medians:

| Build | Reconstruction | Output scope |
|---|---:|---|
| Reference | 101.710 ms | Original InstanceData |
| SIMD/fixed-plane math | 86.879 ms | Exact same full output fingerprint |
| Final isolated-spray version | 87.840 ms | Intended outlet changes; channel-kernel input hash unchanged |

The math-only variant preserves complete frozen sluice/outlet/river PNG bytes. The final river image remains independently checked; final channel/outlet views include the intentional spray change. CPU timings are reconstruction characterization, not physical simulation speed or a sustained native FPS claim. Field evaluation remains the main CPU reconstruction cost.

Twelve-frame moving outlet captures show median presented lag of **191.5 ms to 120.3 ms of simulation time**. This is one short serial comparison, affected by capture overhead. Queue replacement is separately exercised with gated real reconstruction, ordered completed results, epoch reset and blocked-output shutdown.

| Native case | Wall window | FPS | Physical seconds per wall second | Real time |
|---|---:|---:|---:|---|
| Reference, four markers, conservative phases | 20 s | 59.99 | 0.51666 | FAIL |
| Final render runtime, same four-marker mode | 20 s | 59.99 | 0.51447 | FAIL |
| Default eight markers, legacy phases | 60.01 s | 59.93 | 0.39661 | FAIL |

An intermediate mailbox window measured 0.46056x, illustrating variability; no sustained paired physical speedup is established. The one-minute eight-marker run ends with 116,318 channel plus 10,604 airborne water markers and 2,034 grain parcels. Its late one-second rate falls to around 0.30x. It is a different/default phase mode, not a direct four-marker performance comparison. All native windows retain finite water/gold/bulk/dry-mass budgets and report no worker/render failure. These benchmarks precede the capture-only readiness fix; the final native captures include it. They are not empirical recovery or failure calibration.

## Checks and remaining work

- All **93 app tests pass**, including both explicitly enabled saved-scene profilers, 6,144 bit-exact eigen cases, 2,048 scalar/SIMD moment controls, independent double-precision tensor/orthogonality/ray controls at the unchanged tolerance, outlet volume/continuity, bounded queue ordering/shutdown and capture-readiness negatives. Strict app/test Clippy, five-file formatting and native build pass.
- 278 tracked simulation, physical-GPU and water-shader files match the baseline exactly. This stage does not claim new broad GPU physics qualification from app tests. Native actual-GPU rendering and coupled finite budgets are recorded separately.
- Fresh final frozen sluice/outlet/river and moving outlet captures pass with nonempty current water. Manual inspection rejects the glitter trial; the accepted stream retains continuity, but dense beading, contact silhouette and image quality remain unfinished. Native frames and compact metrics live at `/Users/simonheikkila/.codex/visualizations/2026/10/18/01a14ce3-309f-7782-9a7c-ccd0efaf3227/surface-latency-20261028`.
- Eight markers remain default. The installed app, protected source changes, pinned fixture and stash remain unchanged. New temporary replay saves and redundant captures are deleted after verification. No app bundle or source archive is copied.

Continue in `menace-7rn`: reconstruct a genuinely joined outlet interface with appropriate thin-sheet/spray coverage, reduce remaining field cost, and qualify pump stop/start/surge, shallow drying, off-centre/dense feed, per-size gold washout, burial/clogging and cleanout. Reduced-width lane/head/bed/contact ownership, lateral diversion, calibration and sustained real time remain open in the [hybrid plan](../physics/hybrid-sluice.md). Every unfinished excavation, machinery, pads, mud, wash-plant, settling-pond, tailings, flood/failure and gameplay requirement remains active in the [operation plan](../../plans/2026-10-22-realistic-operation.md).

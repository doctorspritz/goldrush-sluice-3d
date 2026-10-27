# Ordered contact-hierarchy investigation

Three conservative hierarchy variants preserve the complete saved physical state but are slower than the existing fine-axis masks. All experimental source changes are removed. All 410 initial crate Rust/WGSL files and the rebuilt baseline probe match exactly; the signed crest/mud app remains unchanged. No speedup or real-time completion is established.

Each ordinary probe starts from the same accumulated-bed save and runs 12 warm-up/120 measured 60 Hz domain ticks. Physical bodies, 120 Hz contact cadence, original coarse sweep membership, increasing pair order, impulses/friction, water/riffle solver work and terrain stages stay fixed. Scratch IDs are spatially packed without reordering physical bodies. Outward-rounded fat bounds admit cached source queries; only visited targets may move within that source traversal. Invalid values and size/storage limits fall back to the existing search.

The first all-target hierarchy increases pair work from 15.249 to 44.868 ms/domain tick. A three-second/10 ms sample of its verified owned process places 106 of 278 main-thread samples in query traversal. This identifies a prototype cost; it does not measure hardware cache/branch stalls. Restricting the tree to larger targets retains the faster fine grid for small targets. The third variant shares bounded spatial query masks and invalidates them after target refits. The later variants' cache hit/refit costs were not separately instrumented.

| Fresh paired order | Pair ms/domain tick | Seconds for 2 physical seconds |
|---|---:|---:|
| Reference, first | 16.8326 | 5.4466 |
| Larger-target hierarchy | 20.2477 | 5.6385 |
| Shared-query hierarchy | 22.1732 | 5.9000 |
| Reference, last | 15.8163 | 5.1778 |

Against the bracketing reference mean, the larger-target variant increases pair time **24.03%** and wall time 6.14%; the shared-query variant increases them **35.83%** and 11.06%. Both are rejected. These are headless domain-tick measurements, not native render FPS.

Candidate library suites pass 142, 143 and 145 tests respectively. New independent controls cover original-ID linear candidates, moving sources and targets, excluded bodies, touching/subnormal cases, invalid/overflow fallback, population/template edits, query-prefix order, epoch invalidation and cache-budget exhaustion. The 65,536-body resource fixture visits 29 nodes locally and requests about 13.3 MiB of tree vectors. The query-mask cache has a separate 262,144-word budget (2 MiB), plus map metadata; this is not a process-RSS guarantee. Passing physical/search tests does not justify slower adoption.

All complete flow controls, including the restored final run, retain canonical SHA-256 `e6934203145f8f31f1ddb61d6fc1ccff5b0ad6a5a32caa25723c0bd7a827256d`, 116,139 water particles and 4,430 grains at 92.60000482946634 s. Canonicalization sorts only three unordered contact-history arrays by integer keys. No physical array or field is removed and no float tolerance is used. The restored executable remains `31f8259b1466ccd67ba3786c43d808f62ce7dc15de7c5ba3a2510556a6b00899`.

No new native capture, controls replay, render-FPS result, full-workspace gate, Clippy or GPU integration-suite result is claimed. All live production/test sources are back at the verified start state. The [receipt](2026-10-27-contact-hierarchy.json) retains complete comparisons, source/binary/input hashes, rejected prototypes, sample and check scope. Scoped cleanup leaves no task game/compiler/probe or test browser; shared processes are untouched. No new panic-full or WindowServer report appeared. The original dirty checkout, user stash and signed app are preserved.

A generic per-source scalar hierarchy is ruled out for this workload. The existing fine-axis masks remain the computational baseline. Physical pressure/contact convergence and calibration still need work; new numerical methods require their own accuracy, conservation, work and loaded-response gates. Real-time simulation, realistic digging/material exchange, pads, shaker decks, settling pools, environment art and the complete gameplay loop remain in the active objective.

# G2P throughput investigation

Two live grid-to-particle experiments were evaluated and rejected. All 410 initial crate Rust/WGSL files are restored byte for byte. The rebuilt flow-probe executable also matches its baseline SHA-256, and the signed crest/mud app remains unchanged. This stage establishes no performance gain.

The same accumulated-bed save starts with 117,544 water particles and 4,308 grains. Each long probe runs 12 warm-up and 120 measured 60 Hz domain ticks; physical resolution, 120 Hz contacts, pressure work, particles and all models remain fixed. Actual GPU-pass timestamps preserve the batched command buffer. Ordinary runs remain the whole-probe timing reference.

Precomputing repeated one-dimensional B-spline weights keeps the source-level expressions and neighbour sum order, but changes the represented GPU result. Its complete final state has 116,294 water particles and 4,439 grains, versus the reference's 116,139/4,430. The first inspected two-domain-tick window already differs, initially by small velocity/affine values before contact evolution. The first changed dispatch and the compiler instruction cause were not isolated. No claim about improved numerical accuracy or a compiler defect follows. This candidate is removed.

| Timed GPU run | G2P device ms/domain tick | P2G device ms/domain tick | Full-probe wall seconds |
|---|---:|---:|---:|
| Initial 128-thread reference | 1.8294 | 2.1331 | 4.9592 |
| Weight-cache candidate | 1.9989 | 2.2708 | 4.9040 |
| 64-thread scheduling, first | 2.3615 | 2.7367 | 4.9061 |
| 64-thread scheduling, second | 2.3601 | 2.6798 | 4.7990 |
| Fresh 128-thread reference | 2.3757 | 2.6699 | 4.7995 |

The scheduling trial preserves every complete physical state, but the fresh 64/128 controls show no useful gain. Initial device timings drift across other stages too; the table does not isolate a weight-cache slowdown or establish a scheduling speedup. Both candidates are discarded. Hardware clock behaviour and cache/branch counters were not measured.

The restored ordinary control advances 2.000000104 physical seconds in 4.8025 wall seconds, approximately 0.4164× without rendering. Complete canonical state remains `e6934203145f8f31f1ddb61d6fc1ccff5b0ad6a5a32caa25723c0bd7a827256d` at 92.60000482946634 s. Canonicalization sorts only the three unordered contact-history arrays by integer keys; no physical array, field or float tolerance is removed. The restored executable SHA-256 is `31f8259b1466ccd67ba3786c43d808f62ce7dc15de7c5ba3a2510556a6b00899`.

No native capture, new render-FPS result, manual controls replay, full integration/library suite or new Clippy pass is claimed. Production and test sources are unchanged, and the previous native crest/mud verification remains the installed-build evidence. The current game still falls short of physical real time.

The next contact-search candidate should use the existing target-motion evidence: separate spatial packing from stable physical IDs, maintain original coarse-sweep membership and ascending pair order, and update conservative bounds when bodies move. During one source's sequential sweep, its visited targets cannot become later IDs; an unvisited target remains fixed until visited. That dependency may permit bounded cached queries, but requires a proof, independent moving-body oracle tests and measured full-state trials before adoption. No hierarchy prototype is implemented here. New numerical methods need their own accuracy, conservation, convergence and loaded calibration gates.

The [receipt](2026-10-27-g2p-investigation.json) retains all complete-state hashes, timings, source/binary fingerprints, rejected sources and scope limits. The user stash, original dirty checkout, inputs and signed app are preserved. All heavy jobs ran sequentially. Scoped cleanup leaves no game/compiler/test browser; shared Node processes remain untouched. No new panic-full or WindowServer diagnostic appeared. Full real-time, digging/material coupling, water/riffle calibration, pads, shaker decks, settling pools, environment art and gameplay remain part of the active goal.

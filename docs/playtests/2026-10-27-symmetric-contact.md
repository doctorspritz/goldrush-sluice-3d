# Exact symmetric contact rejection — 27 October 2026

The saved 1,040-clod terrain pressure evaluation takes **29.27% less loop time** with identical complete non-timing results: 15.86356673 s to 11.22062146 s averaged over an A/B/B/A sequence. This is a faster exact digging-contact **reference**, not a game FPS improvement or live adoption of the pressure model. The installed crest/refraction rendering build remains unchanged at `bde312449d0e7beb49d1792e776b5384e5da4aef`.

## Change and proof

The original root-tetrahedron filter checks the opposing tetrahedron's faces. An axis-overlapping pair can instead be separated by a face direction from the source. That missed case previously reached expensive rational clipping.

The new filter tries those source-face directions. Floating projections only nominate a candidate direction. For each candidate, the implementation promotes its finite nonzero direction and both actual represented tetrahedra to rational values, then proves that their exact projection intervals are disjoint or touch. Any finite nonzero direction with this proof is valid; correctness does not depend on a rounded normal being the exact face normal or having an exact outward orientation. Invalid or zero directions cannot reject contact. Positive overlap has no tolerance or small-volume cutoff.

The existing opposing-face proof stays in place. Original root tetrahedra alone use the new certificate. Constructed recursive union intersections retain complete rational clipping and the existing precision, depth and work limits. Admission and counters occur before the new proof. Positive integration and accumulation order, pressure coefficients, water/riffle/terrain/sediment solvers and save policies are unchanged. No terrain index ships in this change.

The independent asymmetric fixture proves AABB overlap, failure of the original opposing-face rejection and success of the new source-face rejection. Complete response equality is checked against rational clipping with filtering disabled. The fixture also runs beside two overlapping positive rigid components with nonzero linear/angular motion and an asserted recursive union path. Additional controls preserve positive slivers down to 2^-40 and reject zero, nonfinite, overflowing and misleading projection directions. Existing rotation, thin-positive and overlapping-component controls remain active.

## Actual saved workload

Baseline: `bde312449d0e7beb49d1792e776b5384e5da4aef`. Input: `contact-geometry-20261024/fresh-verified-final/worst-before.bin`, SHA256 `7147a81a9830b2d27fcfca6feef4ef419ef8a09fede2899a7ab6e02af7c8db0b` in the thread artifact root. Immutable baseline and candidate test binaries were copied from Cargo's actual playable-profile artifact. The final full-suite binary adds only the mixed-positive test extension to the measured candidate production code.

| Run order | Baseline / candidate | Terrain loop seconds |
| --- | --- | ---: |
| 1 | Baseline | 15.839599334 |
| 2 | Candidate | 11.210264750 |
| 3 | Candidate | 11.230978166 |
| 4 | Baseline | 15.887534125 |

The full process, including the selected saved-case pose/partition controls and supervisor overhead, averages 18.1280 s before and 13.5466 s after. These are two observations per version on this Mac, not a general throughput guarantee. The earlier exploratory candidate result and sampled profile are excluded from this paired estimate.

All four complete terrain reports have identical compact sorted-key JSON bytes after removing only `cpu_wall_seconds`; array order, finite float representations and signed zero are retained. Canonical SHA256: `71c55beedd8e877e55ffab4de0d65c7ac1b4f694f70e962c65c0f162db38e47f`.

The report covers all 1,040 clod/terrain pairs over the actual 32,258 triangles and 86,766 nonzero tetrahedra: 224 positive and 816 zero overlaps, no failures, unchanged owner encoding and input roundtrip. Potential is 0.0009219550905342688 J and overlap 3.1204546588086887e-6 m³. Force residual is zero; maximum moment and power residuals are 7.160723346098895e-15 Nm and 5.551115123125783e-17 W. Largest-force twelve-pose derivatives and same-material partition controls pass unchanged.

The separate complete saved pressure report also matches every non-timing byte, removing exactly `.rigid_steel_union.cpu_wall_seconds` and `.rigid_steel_union.max_pair_wall_seconds`. Canonical SHA256: `7b4dd5e676b3c4d37626dfba2d5555de279cdda0aa26900878c59711baf0b4a6`. It includes 16,632 admitted clod/clod pairs and the actual 560 clod/steel pairs against 19 parts / 320 rigid tetrahedra, with four positive steel overlaps, zero failures, unchanged recursive work/depth counters, pose derivatives, partition controls and complete owner encoding. The final extended-test binary was used for this comparison.

## Rejected index and profile

An immutable terrain BVH trial first matched the complete terrain report, but loop times were 15.748657625 / 15.907911500 s baseline versus 15.509032166 / 15.405120417 s indexed: about 2.35% lower mean time. The largest square-grid test retained 6,815,408 bytes of index data. That gain did not justify shipping the index for this workload.

Eleven focused controls passed, including 20,480 actual transformed vertex enclosure checks, query order/coefficients and overflow/error precedence. Review found two remaining coverage gaps: the captured-layer positive fixture did not actually overlap, and the largest square-grid resource fixture was smaller than the admitted maximum rectangular grid. No complete CPU/audit/lint/water qualification is claimed for the rejected index. All four prototype source/test files and their tracked patch are preserved under the local `rejected-index/` artifact directory; tracked terrain source was restored byte-for-byte. Issue `menace-b8r` stays open for a measured future need.

A three-second, 10 ms sample of the explicitly owned baseline worker found 186 of 282 worker samples inclusive in `ExactPoly::clip`, 55 in `ExactPoly::moments`, and 261 in `Rule::integrate`. These are sampled inclusive frames, not exact CPU percentages; the sampling run is excluded from timings. That result directed this smaller symmetric proof change at rational clipping rather than terrain scanning.

## Validation and water preservation

Commands ran sequentially with `CARGO_TARGET_DIR` pointing to the shared project target; one owned heavy check, `--jobs 1`, test threads 1. Each check had a bounded owner-specific supervisor and crash-report checks.

```sh
cargo +nightly test -p mining-sim --lib --profile playable --locked --jobs 1 -- --test-threads=1
cargo +nightly test -p mining-sim --features contact-work-audit --lib --profile playable --locked --jobs 1 -- --test-threads=1
cargo +nightly clippy -p mining-sim --features contact-work-audit --all-targets --profile playable --locked --jobs 1 --no-deps -- -D warnings
cargo +nightly fmt -p mining-sim -- --check
cargo +nightly build -p mining-physics --example flow_probe --profile playable --locked --jobs 1
```

Default: **76 passed, 0 failed**. Audit: **118 passed, 0 failed, 24 ignored**. Lint/format/build pass. The two complete saved diagnostics were run explicitly: `saved_bank_finite_erodible_terrain_layer` and `pressure_geometry::saved_bank_finite_pressure_geometry`, with `SLUICE_ENERGY_BOUNDARY` and their respective `SLUICE_TERRAIN_PRESSURE_REPORT` / `SLUICE_PRESSURE_REPORT` output paths. Other ignored diagnostics were not rerun or counted as passes.

The actual ordinary solver continued `storage-final-2026-10-23/grown.json.save.json` for 120 measured ticks after 12 warmup ticks. The endpoint at 92.60000482946634 s contains 116,139 water particles and 4,430 grains. Its complete canonical state bytes equal the preceding qualified ordinary continuation: SHA256 `e6934203145f8f31f1ddb61d6fc1ccff5b0ad6a5a32caa25723c0bd7a827256d`. Normalization sorts only JSON object keys and the three unordered contact-history arrays; every other array and float representation remains intact. Raw serialized bytes are not claimed equal.

Inlet/outlet water accounting error is 0 m³; whole-water residual is -7.703740990905317e-6 m³, dry-mass residual 0.00017585570458322763 kg and gold residual 8.640199666842818e-12 g. The continuation measures about 0.410x physical real time on this run. It is a preservation check, not evidence of real-time physics. No native rendering rerun or app replacement was needed for this reference-only change.

The latest installed executable retains SHA256 `b1c488c1a00ab914d275a21041906b4cc0153fb348b239cfa841f7c7cba8131a`. Original user files and stash remain preserved. No new panic/WindowServer report was observed during checks. No personal browser was used.

## Remaining gates

Pressure forces are still observational and are not applied to Session or the retained-world integrator. The unchanged raw 0.01 J mechanical gate, bedrock ownership, terrain-edit work and lower-reference persistence, live friction/yield/damping/joints/motors/CCD coupling, timestep and clod-size convergence, material calibration and real-time performance remain open. The larger game objective, `menace-ea8` and `menace-6um` stay active. The completed bounded task is `menace-z0y`.

The [machine-readable receipt](2026-10-27-symmetric-contact.json) pins the reports, source manifest, binaries, checks and scope; [paired timing data](2026-10-27-symmetric-contact.csv) preserves individual observations. Local full logs, complete saves, immutable binaries, sampled stack and rejected prototype are in `terrain-index-final-20261027` under the thread artifact root.

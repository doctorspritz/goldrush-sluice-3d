Two guarded division-free friction predicates were tested and rejected for production. Both preserve the original 24 bisections, bracket updates and final Coulomb projection. Only the isolated binary32/std test module contains the candidates; gameplay, contact-history encodings and production kernels are unchanged.

For a sliding contact, write `D=det(K+alpha I)`, `n=adj(K+alpha I)b`, and `R=limit²`. The exact predicate `n.n > R*D²` avoids an inverse. Exact real algebra does not guarantee the same binary32 branch: rounding, determinant cancellation and subnormal flushing can change it. Each prototype therefore uses the old reciprocal comparison whenever its sign cannot be certified.

The first prototype checks ranges every root iteration. The second checks conservative whole-bracket bounds once, then evaluates only the sign margin and uses the original reciprocal on ambiguous iterations. Both keep the original zero-limit warm-start removal and final lane selection. A failed whole-batch qualification uses the original loop. Sticking lanes return their original trial even if their unused bracket differs.

The certificate treats the original computed float determinant/numerators as inputs, preserving their evaluation order. It requires `X=nx²+ny²`, `Q=D²`, `R` and `Y=R*Q` within `[2^-48,2^48]`, positive determinant, and rounded `abs(X-Y) > 128u*(X+Y)` with `u=2^-24`. Correctly rounded division, ordinary IEEE binary32 arithmetic and no fast-math reassociation are assumptions. The conservative rounding bound is below `12u*(X+Y)`, including minor-term FTZ loss; the implemented margin has substantial slack. This contract does not promise other floating modes or platforms.

For the hoisted candidate, `t=k11+k22`, `B=bx²+by²`, and the original `H=norm(b)/limit`. Positive diagonals and `D0 > 64u*t²` bound determinant cancellation. The rounded determinant is monotone over `[0,H]`, so endpoint checks bound `Q` and `Y`. Conservative numerator bounds `(D0/t)²*B/16` and `4*(t+2H)²*B` bound `X`, including cancellation and rounding. All checked bound intermediates lie in the same restricted range. Any failed guard returns to the original loop; borderline cases are never approximated. These arguments support the experiment; they do not replace execution controls or establish a performance benefit.

Normal-only guards are insufficient. With `D=2^60`, `nx=1`, `ny=2^-4`, `R=2^-120`, all shortcut quantities are normal and its sign margin passes, but the original minor squared impulse `2^-128` can flush to zero. The shortcut says outside while the flushed original says exactly on the boundary. An explicit test emulates this counterexample. The tighter range guard rejects it.

The deterministic generator combines rotated positive-definite matrices, mass/impulse scales, zero friction, warm impulses, inverse-cutoff cases, forced qualified sliding and ±1 ULP stick/slide controls. Scalar and width-4 SIMD comparisons require exact output bits. A separate observer checks all original root predicates and range bounds for every qualified relevant lane. This is bounded test coverage, not an exhaustive proof over all float inputs, actual-bank coverage, or physical calibration.

The timing test black-boxes the entire input and returned vector. It alternates mode order over five trials in each of two synthetic datasets and executes at least 1,048,576 SIMD calls per trial. All modes retain the same 24 root steps. Checksums are diagnostic only; exact correctness comes from the separate differential test. Synthetic throughput is not native FPS or complete bank replay throughput.

Eight kernel controls pass; the timing test is separately executed and passes its completion checks. Each candidate agrees bit for bit with production over **32,781 scalar cases and 262,248 SIMD output components** at width 4. This includes exact limits and ±1 ULP stick/slide controls. Of **32,762 overlapping sliding SIMD batches**, 2,602 qualify for the hoisted path; its observer checks 38,382 certified and 24,066 ambiguous original root iterations. These are overlapping synthetic windows, not independent loaded-bank samples. Qualified paths and both fallback kinds are exercised.

| Median of five alternating trials | Existing | Per-iteration guards | Hoisted guards |
|---|---:|---:|---:|
| Broad stress: 1,049,088 SIMD calls | 283.882 ms | 400.308 ms | 297.077 ms |
| Bounded synthetic: 1,048,640 SIMD calls | 284.486 ms | 349.028 ms | 329.608 ms |

Both candidates are slower in both datasets. The per-iteration candidate adds 41.0% / 22.7% time; the hoisted candidate adds 4.6% / 15.9% (broad / bounded). These timings do not establish a gain from avoiding divisions. This is enough to reject these prototypes; it does not characterize real-bank qualification rates or prove every division-free strategy slower. Early short and partial-output benchmarks were superseded by the whole-vector final measurement. An initial test-only missing-trait-import compilation failure was corrected before the passing final controls. No incomplete check is counted as a pass. See the [machine-readable receipt](../playtests/2026-10-24-friction-sign.json).

Reproduce from the checkout, one process at a time:

```bash
cargo +nightly test --manifest-path vendor/rapier3d/Cargo.toml \
  --features enhanced-determinism,serde-serialize --lib contact_constraint_element \
  --profile playable --locked --jobs 1 -- --test-threads=1 --nocapture
cargo +nightly test --manifest-path vendor/rapier3d/Cargo.toml \
  --features enhanced-determinism,serde-serialize --lib certified_sign_candidate_timing \
  --profile playable --locked --jobs 1 -- --ignored --test-threads=1 --nocapture
```

This isolated measurement uses the tracked vendor lockfile, including `wide 1.6.1`. The application lockfile uses `wide 1.7.1`; its dependency graph and actual native/bank throughput are not benchmarked here. Both before and candidates use the same vendor graph. The shared application target directory may be selected with `--target-dir`; this does not switch lockfiles or make the benchmark an application run. The benchmark remains explicitly ignored during routine tests. The candidate functions are gated by test/std/dim3/f32 and cannot be selected by an ordinary game build or save file.

Neither candidate was adopted. No complete saved-bank replay, physical timestep refinement, native frame capture or real-time claim is made for them. Existing retained-soil energy, loaded digging and refinement failures remain open. The next performance investigation should capture the actual sliding batch distribution and compare a materially different algorithm or deterministic execution strategy, with an explicit numerical policy if results change. Preserve Coulomb feasibility/KKT accuracy, physical cadence, material stocks, water/riffles/terrain and existing saved-state semantics.

Clippy completes with `clippy::all` denied for the new module. It reports two pre-existing `let_and_return` warnings in unchanged runtime code; a strict whole-vendor lint pass is not claimed. The new module passes formatting and `git diff --check`. Checks run serially. No gameplay/rendering or complete-operation gate is closed by this evidence.

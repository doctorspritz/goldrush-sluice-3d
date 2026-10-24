# Full-bank energy source and rejected solver trials

The saved zero-actuator failure is now replayed without another 299-step warm-up. The maintained diagnostic restores the complete owner, matches the saved canonical machine/soil and every body's pose and velocity, and independently observes all **1,045 dynamic bodies / 1,040 clods**. This comparison does not claim byte equality of the internal numerical world. A phase-only reproduction reaches the same final body state exactly.

| Energy change on the saved 480 Hz step | Joules |
|---|---:|
| Engine step before final synchronization | 3462.874 |
| Final body velocity synchronization | 121.230 |
| Total independent gain | 3584.104 |
| Machine subtotal | 3375.827 |
| Soil subtotal | 208.277 |

The stick and bucket contribute about 2,148.52 J and 1,324.11 J respectively. The independently calculated total agrees with feedback within 0.0003 J. Final synchronization is not the primary source. These results locate the failure; **coupled passivity still fails**.

Fourteen single-boundary attribution controls retain every clod. Twelve complete and two contact-refresh controls refuse the existing curl limit at 1.451438 rad; the current tolerance is not widened. Selected controls are:

| Diagnostic control | Energy change, J |
|---|---:|
| Current retained solve | 3584.104 |
| No penetration velocity bias | 3220.388 |
| Friction also solved during biased passes | 2784.454 |
| Half warm-start coefficient | 635.790 |
| Zero warm-start coefficient | -56.547 |
| 16 relaxation passes | 1250.047 |
| 64 relaxation passes | 58.369 |

Retained impulses strongly amplify this failure at the current solve convergence. Zeroing history is an attribution control, not an adopted solution. More relaxation reduces the excess, but even 64 passes remain positive and cost about 341 ms on this one saved step versus 40 ms for the baseline. These are diagnostic CPU samples, not native throughput benchmarks. Changing TGS counts or outer subdivisions on an already-warmed boundary does not rescale the incoming history; those controls are **not timestep convergence tests**.

Two backed-up engine prototypes were measured on the same boundary. Refreshing articulated contact gaps after integration gives +3,582.61 J; refreshing current contact Jacobians/inverse masses gives +3,588.48 J. Neither removes the failure. Both are rejected and every vendor source is restored exactly. An unwired prototype attempt is labelled as an unchanged-solver control and excluded from candidate evidence. No runtime force law, history policy, iteration count, physical cadence, particle population or acceptance limit is changed.

Final validation passes the explicitly executed saved-boundary diagnostic, **23 default library checks and 31 featured library checks**, strict featured all-target Clippy and formatting. Three opt-in diagnostics remain ignored in the default featured suite; this turn explicitly executes only the new saved-boundary diagnostic. Full soil/gameplay suites and native rendering are not rerun for this test-only change. All production simulation and rendering sources match the baseline. Scoped cleanup leaves zero test/browser processes; 14 shared Playwright MCP and 56 Node processes remain untouched.

The [receipt](2026-10-24-contact-energy-source.json) records inputs, source hashes, all controls and rejected candidates. Complete body rows and prototype patches remain in `/Users/simonheikkila/.codex/visualizations/2026/10/18/01a14ce3-309f-7782-9a7c-ccd0efaf3227/bank-energy-source-20261024`. The next accuracy work is a converged coupled normal/friction/limit solve across actual machine-to-clod mass ratios, with consistent-history refinement and penetration checks. `sluice-retained-coupling-20261024` and joint-limit work stay open; the retained owner remains experimental and outside Session. Pads, wet/cohesive mud, shaker decks, buildup, settling circuits, real-time performance and the full operation remain required.

Run the diagnostic serially with the two preserved boundary paths and a fresh output file:

```bash
SLUICE_ENERGY_BOUNDARY=/path/to/before.bin \
SLUICE_ENERGY_REFERENCE=/path/to/after.bin \
SLUICE_ENERGY_BREAKDOWN=/tmp/fresh-energy-report.json \
cargo +nightly test -p mining-sim --features contact-history --lib \
  saved_bank_energy_source_breakdown --profile playable --locked --jobs 1 \
  -- --ignored --test-threads=1 --nocapture
```

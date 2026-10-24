# Flowing demo fixture

Regenerated with the restored standalone profile by `mining-physics/examples/make_demo.rs` on Apple M2 Max, 21 October 2026. Three finite terrain cuts are delivered to the hopper; the original FLIP/APIC and CPU DEM advance alongside the repaired terrain solver. The file is a normal complete Session save compressed with zlib. It includes live particles, APIC/DEM contact state, terrain momentum and the complete water ledger, without manufactured recovery.

Rebuild with `cargo +nightly run -p mining-physics --example make_demo --profile playable --jobs 1 -- /tmp/flowing.json`, then zlib-compress that JSON into `flowing.json.zlib`. The application rejects an invalid fixture through the same save validation as gameplay.

The fixture uses `OriginalDemo` with the granular material model: 1 cm cells, 150×52×40 solid-geometry cells plus a 12-cell outlet buffer, and four FLIP/DEM substeps per 1/30-second fluid update. At 40.0167 s it contains 110,881 water particles and 1,599 grain parcels, 0.004 m³ processed and 0.805190 m³ in the hopper. Feed is 0.1 L/s, water 3.75 L/s; water has returned 0.134782 m³ to the site and has a continuous wet route through the graded tailrace to the river. External river input is 4.800000 m³, map discharge 5.560008 m³, and whole-system volume residual is -0.000007281 m³ (about -7.3 mL). Gold ledger error is below 1e-7 g. The lossless compressed file is 9,384,194 bytes (SHA-256 `153d24b16319539cd38db9c342653ae455fc7be0a41633bfb3c163e9d5a8377f`). No procedural surface waves or target recovery are baked into the fixture.

See `docs/playtests/2026-10-20-riffle-dynamics.md` for the physics assumptions and controlled comparisons. The smaller physical feed takes longer to process a full bucket than previous game settings.

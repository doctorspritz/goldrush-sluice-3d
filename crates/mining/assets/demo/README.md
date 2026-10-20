# Flowing demo fixture

Regenerated with the restored standalone profile by `mining-physics/examples/make_demo.rs` on Apple M2 Max, 20 October 2026. Three finite terrain cuts are delivered to the hopper; the original FLIP/APIC, CPU DEM and terrain solvers advance until water and gangue actually leave the sluice. The file is a normal complete Session save compressed with zlib. It includes live particles, APIC/DEM contact state and terrain momentum, without manufactured recovery.

Rebuild with `cargo +nightly run -p mining-physics --example make_demo --profile playable --jobs 1 -- /tmp/flowing.json`, then zlib-compress that JSON into `flowing.json.zlib`. The application rejects an invalid fixture through the same save validation as gameplay.

The fixture uses `OriginalDemo` with the new granular material model: 1 cm cells, 150×52×40 solid-geometry cells plus a 12-cell outlet buffer, and four FLIP/DEM substeps per 1/30-second fluid update. At 12.0167 s it contains 120,278 water particles and 1,025 grain parcels, 0.0012 m³ processed and 0.80799 m³ in the hopper. Feed is 0.1 L/s, water 3.75 L/s; water has returned 0.029965 m³ to the site and gangue/gold have actually reached tailings. Gold ledger error is below 1e-7 g. The lossless compressed file is 9,702,173 bytes (SHA-256 `74a69efe161ebd23e45474b706f9554740d6f0ddf79d4d9093c19c91fcba8645`). No procedural surface waves or target recovery are baked into the fixture.

See `docs/playtests/2026-10-20-riffle-dynamics.md` for the physics assumptions and controlled comparisons. The smaller physical feed takes longer to process a full bucket than previous game settings.

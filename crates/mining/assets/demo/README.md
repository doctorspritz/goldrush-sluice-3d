# Flowing demo fixture

Regenerated with the restored standalone profile by `mining-physics/examples/make_demo.rs` on Apple M2 Max, 20 October 2026. Three finite terrain cuts are delivered to the hopper; the original FLIP/APIC, CPU DEM and terrain solvers advance until water and gangue actually leave the sluice. The file is a normal complete Session save compressed with zlib. It includes live particles, APIC/DEM contact state and terrain momentum, without manufactured recovery.

Rebuild with `cargo +nightly run -p mining-physics --example make_demo --profile playable --jobs 1 -- /tmp/flowing.json`, then zlib-compress that JSON into `flowing.json.zlib`. The application rejects an invalid fixture through the same save validation as gameplay.

The restored fixture uses `OriginalDemo`: 1 cm cells, 150×52×40 solid-geometry cells plus a 12-cell outlet buffer, four FLIP substeps per domain tick. At 28.0167 s it contains 232,395 actual water particles and 27 grain parcels, 0.028 m³ processed and 0.78119 m³ of finite hopper material remaining. The lossless compressed file is 17,060,989 bytes (SHA-256 `934d3c40f7f3922ac16cba493736758f32112562de20e062bc15f6d435b9834b`). No procedural surface waves are baked into the fixture.

The final revision changes only the saved inspection pose to show the whole channel. Fluid, terrain, APIC/DEM contact state and all material/water ledgers are byte-for-byte unchanged from the completed warm-up.

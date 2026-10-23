The contact-history boundary preserves a complete rigid world's numerical state before persistent machine/soil integration is admitted into Session. The explicit `contact-history` Cargo feature exposes `mining_sim::contact_history::ContactHistory`, which owns the world, clones its numerical sets, and supports a bounded binary encoding. The actual machine plus 1,040-clod bank, its raw and solver manifolds, cached impulses and repeated restoration are checked in `articulation/history_tests.rs`.

The world carries bodies, colliders, both joint sets, soft-body state, integration parameters, islands, broad-phase pairs and the narrow-phase contact graph. The derived physics/collision pipelines and CCD lookup workspace are rebuilt. The query dispatcher is stateless code selected explicitly as `Default` or `RigidSoil`; it is reinstalled after decode. Reinstalling it preserves the saved contact-softness and other physical parameters. `world_mut()` supports ordinary body/joint updates; owners must keep the declared dispatcher policy consistent.

| Version 1 field | Encoding |
|---|---|
| Magic | Eight bytes: `SLUICEWH` |
| Schema | Little-endian u16, currently 1 |
| Dispatcher | One byte: 0 default, 1 rigid-soil convex repair |
| World | Bincode 1.3.3, fixed integers and little-endian scalars |
| Payload limit | 64 MiB; oversized payloads are refused |

The schema is tied to the pinned Rapier 0.36.1 state layout and local patches. Cross-version migration requires an explicit schema decision. Unsupported magic/version/policy, truncated data, trailing data and oversized input are rejected. This is an engine-produced numerical boundary; application-level scene, custody and handle-map validation remains necessary before admitting it into Session saves.

JSON cannot encode tuple keys used by the contact maps. The initial unstepped bank boundary is 1,749,508 bytes; after 24 mechanical steps it is 15,620,067 bytes including its header. It preserves float values and cached impulses without reducing geometry, particle counts, physical cadence or solver iterations. That size matters for future save storage and runtime cloning; this checkpoint makes no throughput claim.

Owners must persist their application body/joint handle maps, controller state, material custody and scene revision alongside the numerical world. Before Session adoption, a retained machine model must also handle geometry/payload changes, actual terrain updates, external forces, coordinate wrapping and contact-state validation explicitly. It must retain deterministic continuation through clone and save/load, preserve older saved models, pass independent contact/momentum/energy and loaded timestep-convergence controls, then demonstrate native throughput. Wet moving-soil coupling remains a separate open requirement.

Current Session versions 2–11 retain their existing fields and integration path. No persistent contact world is enabled in gameplay yet. The [recorded 18-second command fixture](../playtests/fixtures/2026-10-24-soil-stroke.commands.json.gz) checks the existing path against complete saved states while the new state boundary is validated independently. The operation's remaining physical and gameplay stages stay open.

Run boundary validation without enabling serialization in ordinary game builds:

```bash
cargo +nightly test -p mining-sim --features contact-history --lib --profile playable --locked --jobs 1 history_tests -- --test-threads=1
```

The explicit [retained mechanical owner](retained-soil.md) now exercises this boundary with the actual controller and full bank, complete canonical custody and application handles. It shares the existing mechanical equations, keeps live workspaces between steps, and checks saved continuation independently of the default Session path. Its measured speed and physical/refinement limitations are reported separately; it is not enabled in gameplay.

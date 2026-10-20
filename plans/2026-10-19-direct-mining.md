# Direct-control mining slice — first playable prototype

Approved scope: first/third-person excavator operation on a finite claim, bucket-edge terrain cutting, carried loads dumped into a hopper, player placement of a sluice, water-driven separation, cleanup and save/load. Base: origin/main at 3af9f067. The previous recovery checkpoint is 01b44a9.

Core requirement: preserve the existing FLIP/APIC sluice, CPU DEM and GPU terrain flow, settling, erosion, sediment transport and collapse. Rendering and controls wrap the original simulation.

- [x] Bevy scene, machine controls and layered excavation with material transfers.
- [x] Placement preview with support, overlap and hopper-connection checks.
- [x] Original solver adapter and matching visible/collision geometry.
- [x] Persist particles, APIC state, DEM contacts, terrain momentum and suspension channels.
- [x] Resume on 20 October with sequential, bounded checks; builds use one compiler job.
- [x] One simulation worker, one active input packet and one pending packet, no render-thread GPU waits or fixed-update catch-up bursts.
- [x] Worker tests for overload, ordered pause/save/load and shutdown.
- [x] Original integration check: 720 FLIP substeps, 120 terrain steps, live restore and balanced ledgers.
- [x] Validate larger live-fluid saves through the same storage path as gameplay.
- [x] Finish full operator replay against the original solvers, including outlet flow and gangue discharge.
- [x] Inspect native screenshots and measure rendering while the plant is running.
- [x] Complete targeted tests, lint and build; publish evidence and commit/push.

Playable code checkpoint: 728822e, pushed to codex/direct-mining. Review: https://github.com/doctorspritz/goldrush-sluice-3d/pull/24 (draft; GitHub CI is separate from local verification).

20 October results: two complete native replays excavated/delivered/processed 0.560 m³, recovered 8.208 g and discharged 0.400 m³ of gangue. Live save/restore and gold/water ledgers passed. The final scheduling run averaged 57.47 FPS during washing at 1440×900, with simulation at 0.385× wall time. The final presentation-only particle splats and corrected side walls were verified from that live save in the standalone app at 49.62 FPS, with 106,660 water particles rendered as 53,330 splats. Details and images: docs/playtests/2026-10-20.md.

The original shader's settling-speed impulse was corrected to exact drag relaxation with mass-correct buoyancy; the original water branch is protected by an explicit GPU regression. An existing settling test now uses a real water-filled column rather than a sparse cloud falling through empty space. The renderer preserves the original collision mesh, corrects its reflected and inward side-face winding and draws actual water particles as GPU-instanced sphere splats. The rendering cap is separate from solver capacity; detached droplets stay three-dimensional and every visible triangle faces out of the collision solid.

Follow-ups are explicitly tracked as sluice-realtime, sluice-calibrate and sluice-machine-dynamics. The prototype is playable; physical calibration, real-time throughput, full vehicle dynamics and the broader game remain ongoing work.

The Mac still reports macOS 26.6.2 and Codex 26.1007.21159. The earlier WindowServer cause remains unproven. No personal-browser automation or safety-configuration changes are part of this work. Only one build/test/game workload is run at once; a new WindowServer/panic report stops further heavy work. Timeouts remain test failures, not crash signals.

Weighted grain parcels remain qualitative; no calibrated recovery claim is made.

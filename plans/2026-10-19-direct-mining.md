# Direct-control mining slice — unfinished checkpoint

Approved scope: first/third-person excavator operation on a finite claim, bucket-edge terrain cutting, carried loads dumped into a hopper, player placement of a sluice, water-driven separation, cleanup and save/load. Base: origin/main at 3af9f067.

Core requirement: preserve the existing FLIP/APIC sluice, CPU DEM and GPU terrain flow, settling, erosion, sediment transport and collapse. The reduced game-layer separation model was removed. New rendering and controls must wrap the original simulation.

Implemented in this checkpoint: Bevy presentation and controls; finite layered excavation and transfers; placement checks; an adapter calling the original GPU solvers; matching visible/collision sluice geometry; particle, APIC, DEM contact and heightfield state persistence. The excavator remains a constrained kinematic machine.

Verification before the incident: a targeted Metal integration test passed 1,440 FLIP substeps and 120 terrain steps, live save/restore and water/gold accounting. Whole-loop replay, current native performance, complete tests/lint and final acceptance remain pending. Earlier screenshots and timing from the removed reduced simulation are not acceptance evidence for this version.

Work stopped after a new WindowServer watchdog incident on 19 October 2026. Do not relaunch GPU workloads on the affected Mac as part of checkpointing. Follow-up work is to complete and verify integration, address scheduling/performance without substituting the original physics, finish the replay and then review the slice as a playable release.

This commit is a recovery checkpoint. The feature remains in progress.

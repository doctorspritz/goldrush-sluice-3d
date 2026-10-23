# Articulated machine validation

`--articulated-machine` enables a floating-base excavator mechanism for physical validation. The ordinary playable machine retains the finite cutting and delivery model in [machine contact](machine-contact.md). The new mode refuses cutting and dumping until soil reactions and payload custody are coupled consistently. Its HUD labels that limitation. This is a prerequisite for realistic digging, not a calibrated hydraulic excavator.

## Bodies, coordinates and contacts

Five dynamic bodies represent the 3,300 kg undercarriage/tracks, 3,000 kg upper house, 1,300 kg boom, 840 kg stick and 250 kg bucket. Unexchanged v7 states distribute carried wet mass over the bucket floor. Explicit [spatial payload transactions](payload-exchange.md) now preserve actual carried COM/inertia after capture or release; normal controls still refuse digging/dumping. The existing component masses, dry friction and geometry remain declared assumptions. Four reduced-coordinate revolute joints connect swing, boom, stick and bucket; the undercarriage is free to translate and rotate. Real joint/body state drives the tool and render geometry. Contacts between adjacent joint bodies are disabled; non-adjacent self contacts, actual terrain, equipment and claim boundaries remain active.

The pinned Rapier 0.36.1 implementation uses the existing deterministic, single-worker mechanical clock, four TGS stages and sixteen internal PGS iterations. Bodies are framed at their centres of mass. Final forward kinematics and body velocities are synchronized from the solved generalized state before calculating momentum and energy. This fixes a reproduced observation error in which public body velocities omitted the last TGS update; generalized velocity is never adjusted to force a conservation result.

Joint limits act dynamically. Validation admits at most 1 mrad of limit error, without clamping the solved pose. The ordinary model retains its previous bounds. Track traction remains the existing friction-limited ideal belt speed controller; gearbox, engine power and wet deformable tracks are unmodeled.

## Actuator approximation and ledger

Each joint uses a force-based implicit spring/damper, with assumed stiffness 20 MN m/rad and damping 100 kN m s/rad. Torque caps use the existing force limits times declared lever arms, not actual cylinder geometry or measured hydraulic curves. Control input moves the elastic reference at the existing joystick rates; it does not prescribe body rotation.

For strain magnitude x, stiffness K and torque cap T, the stored potential is:

`phi = 0.5 K min(x,T/K)^2 + T max(x-T/K,0)`.

Positive reference work is the exact increase of this potential at fixed joint coordinates. A common input fraction bounds the sum of positive work by the declared pump power times the mechanical timestep. Negative reference work is separately reported as return heat. Pressure relief shifts excessive reference strain to T/K and reports the released potential as relief heat. This prevents a stalled command from storing unlimited elastic energy or future motion. A swing coordinate chart change preserves strain across revolutions.

Feedback contains kinetic, gravitational and stored elastic energy, positive pump work, return and relief heat, external wrench work, and an explicit passivity residual. Dissipation from damping and friction is not inferred from that residual or presented as measured heat. External wrench work uses COM displacement and a finite rotation increment; this is a step approximation requiring convergence. Mechanical power may temporarily exceed pump power by releasing stored energy.

`passivity_inputs_accounted` reports coverage only. It is false when a moving belt has a loaded contact because belt work is unmetered. It does not establish complete energy closure, calibrated efficiency or closed engine/fuel accounting. The old fuel proxy remains. Reduced-coordinate motor impulses are unavailable from this pinned engine, represented as absent feedback rather than a fabricated zero tool load.

## Custody and persistence

Carried wet mass and optional spatial moments are checked before every solve. Out-of-band mass edits fail before committing dynamic state. Ordinary cut/dump requests leave bucket, terrain, falling-load, gold and water inventories unchanged and report their unavailable status. The separate explicit spatial-transaction API and bounded payload probe exercise finite material transfer without claiming contact/retention or soil failure. No old excavation transaction is admitted into the new mechanism. Active soil geometry, penetration/separation resistance, granular retention/spill and momentum/energy transfer on pickup or discharge remain required before ordinary activation.

Ordinary shifts still write version 6. Unexchanged articulated shifts write version 7; spatial payload/rigid-parcel state writes version 8. These preserve all ten generalized velocities, four joint coordinates and elastic references, stiffness/damping, constant payload and cumulative work/heat. Active-soil validation writes version 9. The loader accepts versions 2–9, requires v7 or newer for an articulated mechanism and v8 or newer for spatial material state, and requires v9 for active soil. It retains all previous material and terrain-clock migrations. Reset, Demo and explicit legacy load retain an already selected validation mode. A saved v7 state also restores the model without the flag. Contact worlds rebuild at mechanical boundaries; no hidden warm-start state is needed for the checked exact continuation.

## Validation and remaining scope

[The checkpoint evidence](../playtests/2026-10-23-articulated-core.md) separates the CPU mechanics controls, explicitly run GPU pipeline checks, native input/save checks, ordinary cutting replay and rendered motion. COM and momentum checks distinguish world-origin angular momentum from intrinsic angular momentum. Physical timestep refinement is distinct from macro batching and from finer terrain geometry. No passing mechanism test establishes deformable-soil convergence, water real time or measured excavator accuracy. Stages 2–6 of the [operation plan](../../plans/2026-10-22-realistic-operation.md) remain open.

[Rapier joint documentation](https://rapier.rs/docs/user_guides/rust/joints/) describes revolute constraints and finite motor limits; the locally pinned source defines the actual API and defaults. [Servin, Berglund and Nystedt (2021)](https://www.algoryx.se/papers/terrain/) couples active soil aggregates and machine multibody dynamics. That supports the need for an articulated prerequisite, but its soil model, calibration and reported accuracy are not implemented or claimed here.

The opt-in [active-soil experiment](active-soil.md) now couples finite moving clods to the articulated contact solve and writes version 9. It is a drained, frictional case with explicit wet-contact refusal; ordinary articulation still uses its existing contact law. The normal cut/dump transaction, full shell/retention and calibrated soil remain incomplete.

# Spatial payload exchange

The articulated mechanism now has explicit finite material capture/release transactions. They change carried mass, centre of mass and inertia, and solve the corresponding joint velocities. This is the momentum-custody prerequisite for moving soil. It does not yet decide when soil fails, establishes contact or sticks inside the bucket. Ordinary digging retains its previous model; Space/J remain unavailable in mechanism validation. Stages 2–6 remain open.

## State and physical event

`MassMoments` stores mass, centre and the central second moment `C = integral(r r^T dm)`. Rotational inertia is `trace(C) I - C`. Combining distributions uses the parallel-axis theorem. The loader checks relative positive semidefiniteness, symmetry, finite quantities and agreement with actual carried wet mass. Small positive packets are retained even below the old convenience `Material::is_empty` threshold.

The original v7 assumption distributes carried mass over the bucket floor. It remains the fallback for unchanged old states. After an exchange, the bucket uses its steel floor plus the actual carried spatial moments, with its body frame at the combined COM. The tool, joints and rendering retain the engineering wrist frame. Arbitrary off-axis loads therefore contribute their actual COM and inertia. The finite component masses and hydraulic geometry remain assumptions from the [mechanism model](articulated-machine.md).

For fixed coordinates, body Jacobians assemble the ten-coordinate generalized mass matrix. An incoming parcel contributes its momentum at its actual COM, including intrinsic spin:

`M_after v_after = M_before v_before + J_linear^T p + J_angular^T spin`.

Release subtracts the outgoing covector. A double-precision SPD Cholesky solve gives the new generalized velocity, which is registered in the pinned f32 mechanical engine. No diagonal regularization, velocity rescaling, forward kick, artificial chute displacement or extra pump work is used to manufacture a budget. Numerical residuals remain explicit.

Capture is a perfectly inelastic projection onto the articulated velocity space. Its heat is calculated independently as the quadratic velocity mismatch of the old mechanism and incoming parcel about the final velocity. It is not inferred by labelling the observed energy residual as dissipation. Uniform release splits the carried spatial distribution, retaining actual COM velocity and angular momentum; this is not a model of granular sorting or internal slip.

The APIs are transactional. Capacity, finite-source, valid-body and bucket-COM envelope checks precede custody changes. A necessary variance bound also rejects spatial distributions that cannot fit the envelope. These are admission guards, not proof of geometric intersection or retention. Active soil must establish the actual event and couple unilateral contacts and soil failure before calling. The free-system momentum projection currently precedes the next contact solve; complete contact/event convergence remains an activation requirement.

## Free flight and boundary handoff

A `RigidParcel` carries real material, canonical double-precision COM position/velocity, orientation, body moments and world spin. Uniform-gravity COM motion is ballistic. Spin angular momentum remains constant; a midpoint rotation update integrates torque-free motion with bounded angular increments. Its asymmetric rotation error is checked against a finer physical-time reference. Legacy f32 `FallingLoad` positions are checked projections for rendering and terrain queries. Existing non-rigid loads keep their previous trajectory path.

At the existing hopper/terrain handoff, a saved boundary receipt records incoming linear/angular momentum and kinetic energy. The fixed receiving system is the declared external boundary. Received kinetic energy is not called a closed heat budget: finite contact geometry, deposition/repacking potential energy, soil failure, restitution and hopper reactions still need a full coupled model. Handoff uses the existing COM crossing and bulk deposition approximations. It does not simulate a colliding rigid clod or a resolved granular pile.

Carried and airborne spatial state renders as a read-only equivalent ellipsoid whose second moments match the stored distribution. That shape is a schematic aggregate visualization, not a collision surface, packing model or literal micrograin. No extra water particle, pressure iteration change or physical-grid reduction is introduced.

## Persistence and verification

Ordinary saves remain v6 and unexchanged articulated saves remain v7. Version 8 preserves spatial payload moments, the projection ledger, rigid airborne bodies and static-boundary receipts. Current loaders accept versions 2–8, reject incompatible version/state combinations and reject mismatched f32 projections. Exact body and machine continuation, material/gold/pore-water custody, live worker save/restore and actual-GPU integration are checked separately in the [checkpoint evidence](../playtests/2026-10-23-payload-exchange.md).

`--payload-motion` is a bounded native mechanism probe. It requires `--new-shift --articulated-machine --capture-motion`. Its initial finite fixture is withdrawn from site inventory and placed at a declared bucket position as an initial condition, then admitted through spatial capture. Bevy joint controls open the bucket until gravity points away from its floor before two explicit worker transactions release the load. This plane-clearance check does not replace complete bucket/equipment contact or retention. This is a finite-load dynamics/rendering probe, not a scoop/dump operator replay or completed digging physics.

[Servin, Berglund and Nystedt (2021)](https://link.springer.com/article/10.1186/s40323-021-00196-3) couples the inertia/momentum of active soil to equipment through dynamic aggregates. It identifies the inability of a steady resistance law alone to describe accelerating soil correctly. That motivates this prerequisite; the paper's complete soil model, contact calibration and reported accuracy are not implemented or claimed here. Next work remains actual active-soil geometry, friction/cohesion/yield, retention/spill and contact-consistent capture/discharge, followed by physical refinement and native scooping before ordinary activation.

# Saved water-marker sampling

28 October 2026. Explicit experimental full-width mode. This stage changes water quadrature, not the sluice dimensions, 1 cm grid, pressure schedule, riffle geometry, physical mineral diameters or CPU grain-contact geometry. It is a prerequisite for cheaper coupled water, not completion of the reduced-width/lane model or calibrated recovery.

## One water owner and one unit conversion

The reference marker volume remains `HydraulicProfile::particle_m3()` (`dx³/8`, retaining the existing floating-point convention). `Sluice::water_particle_m3()` is that reference times `8/N`, where the saved sampling is eight, four or two markers per cell. Each channel or airborne marker owns this same volume. Pump and feed credits, returned volume and terrain receipt remain actual cubic metres.

The conservative phase transfer still uses the original grid reference. A water marker has transfer weight `8/N`; a mineral parcel has weight `solid_volume/reference_marker_volume`. Increasing water-marker volume must never shrink the mineral's displaced volume or reaction mass. Physical diameters, material inventory and contact templates remain independent of marker sampling.

Intake admission, hopper/screen water room, water-depth diagnostics, solver exits, lip flight, terrain impact and slurry projection now use the saved water volume. The renderer uses it for kernel support and integrated optical density, including its existing capped visual sampling. Visual splats own no additional water.

## Saves and runtime

The setting can change only when the channel and airborne-water lists are empty. Existing pump/feed credits retain their actual volumes. This implementation does not merge live markers or relabel their volume. Reduced sampling requires the original full-width granular profile.

A reduced-sampling shift uses the outer version-15 envelope `{version:15, shift:<complete prior-format shift>}`. Older applications reject it instead of silently applying their fixed marker volume. The current loader refuses a reduced setting hidden in an older envelope, a reference setting falsely wrapped as version 15, unknown sampling values and invalid contained state. Ordinary eight-marker saves omit the added setting and retain their original envelope.

The hydraulic adapter selects conservative phase exchange for the saved reduced setting, including after loading on an ordinary worker. Loading a reference shift again restores the worker's requested phase policy; an explicit conservative-validation constructor remains conservative. Physics scheduling and terrain flow do not change.

A fresh operation can be started from the source build with `cargo +nightly run -p mining --profile playable -- --new-shift --water-markers 4`. A loaded shift retains its saved sampling. The bundled already-flowing demo retains its reference water ownership. The setting also applies when restarting a fresh shift. Eight remains the default; two remains a coarser investigation case.

## Qualification and limits

The [stage report](../playtests/2026-10-28-water-sampling.md) records actual CPU/GPU, complete-operation, rendering and performance evidence. The `water_sampling_probe` example makes a finite dug-feed/screen/sluice case and one requested snapshot for repeatable examination. Delete temporary snapshots after checking them.

Co-located marker-splitting equivalence proves the volume and phase-reaction units. It does not prove that fewer distinct support samples reproduce every evolving water surface. Pump surges/stops, shallow drying, off-centre loading, dense beds, grade-dependent washout, burial/clogging/cleanout and empirical calibration still need operating-envelope comparisons. No changed default, installation or real-time claim follows from fewer markers alone.

Reduced-width vertical water, full-width lane/head coupling, finite bed/contact ownership and lateral diversion remain separate unfinished work in the [hybrid plan](hybrid-sluice.md). All unfinished excavation, pad, mud, wash-plant, settling, tailings, flood/failure and gameplay requirements remain active in the [operation plan](../../plans/2026-10-22-realistic-operation.md).

# Weighted slurry and visible grain populations

The simulation uses volume-weighted DEM parcels rather than one rigid body per physical grain. Under the spherical-grain interpretation, 2 mL of actual solid volume in 100 µm grains represents `2e-6 / (pi/6 * (100e-6)^3)`, approximately 3.82 million grain equivalents. Reference-bulk parcel volume first converts through its constituent mass and mineral density to solid volume; it is not automatically that much solid. See the [material convention](material-mass.md). A recovered gold grain remains an individual physical grain. These counts describe the model and do not measure the site's real particle-size distribution.

## Feed grading and physics

New shifts use the explicit `Alluvial` prototype grading. Soil is a noncohesive 20–200 µm fine fraction, sampled log-uniformly by equal-volume feed parcels; sand remains 0.2–1.2 mm, gravel 1–6 mm, and gold 80–500 µm. Soil and sand now occupy separate parcels, preserving their exact material components. The physical diameter enters the existing settling/drag law, while represented volume and mass enter the existing water/DEM coupling. Grading changes neither the hopper assay nor the water/feed delivery rates.

The grading is a declared assumption, not calibrated geology. It does not include clay cohesion, clay rheology, flocculation, or a measured feed PSD. Saves without `feed_grading` retain `Legacy`, including the former combined soil/sand feed. New saves persist their grading. The refreshed bundled demo is regenerated from finite terrain cuts using the new grading.

## Local suspension optics

`mining_sim::slurry::SlurryField` is a read-only 3 cm reconstruction of actual channel water and moving fine parcels. Water and fine parcels use the same conservative trilinear deposition. Mixing a water histogram with linearly projected fines gave a false concentration change as a homogeneous suspension moved through optical cells; the [crest/projection correction](../playtests/2026-10-22-crest-reconstruction.md) documents the reproduced error. Conservative separable smoothing retains the integrals of water volume, suspended fine volume and geometric grain cross section, including at field edges. Water particles sample this continuous local field. Settled material is excluded from suspended cloudiness.

For a spherical population, total cross section is `N*pi*d²/4 = 3*solid_volume/(2*d)`. The optical approximation uses `Q_ext=2` and extinction `2*cross_section/water_volume` in inverse metres. This accounts for represented size and volume, independently of how many render samples are displayed. RGB scattering remains an artistic approximation; this is not NTU calibration or a multiple-scattering radiative-transfer solution. The dependence of optical response on both concentration and grain size is consistent with [USGS laboratory observations](https://www.usgs.gov/publications/effects-suspended-sediment-concentration-and-grain-size-three-optical-turbidity).

The channel renderer adds two conservative streamwise binomial passes to the optical moments before sampling. This spreads unresolved weighted parcels into longer plumes while retaining water, fine volume and cross-section; it does not advect or diffuse physical sediment. Vertical and cross-stream strata keep their prior filter. The established outlet concentration is sampled before the extra channel reconstruction.

Channel particles carry local extinction in the existing instance buffer. Terrain-water optics retain their existing concentration convention. Airborne outlet water uses the current nearby outlet concentration for presentation; individual airborne sediment mass and optical history are not transported by this change. Its water flight and mass transfer remain authoritative. This outlet approximation should be replaced alongside conserved airborne solids, rather than interpreting brown discharge as an additional material ledger.

## Bounded visible grains

The previous display showed at most 12 proxies per parcel and usually left much of its budget unused. Sampling now distributes the existing 32,768-proxy global budget across the live material, with up to 512 per coarse parcel and eight per subpixel fine parcel. Individual gold receives one proxy. Geometry uses physical grain sizes and does not inflate sand into pebbles. Fine populations are principally visible through suspension optics. Three material batches with exact-sized allocations avoid one entity/draw call per grain and repeated buffer growth; subpixel grain shadows are disabled. Sampling never changes simulated material or particle trajectories.

## What remains unresolved

Particle storage now starts at 4,096 grain slots and grows geometrically while retaining every authoritative parcel, with an explicit allocation preflight and stop on refusal. This is derived numerical storage rather than a physical capacity or clogging law; see [storage and limits](particle-storage.md). A cloud of displayed grains does not add independent collisions, shear response or a resolved packed bed. The previously measured phase-momentum error, inflated coarse contact envelopes, dense-bed penetration and experimental recovery calibration remain open in `sluice-phase-coupling` and `sluice-calibrate`. Increasing the number of render grains cannot fix those equations. The [accuracy protocol](accuracy.md) defines the remaining physical gates.

Read a saved population and reconstruction cost without advancing it:

```bash
cargo +nightly run -p mining-sim --example slurry_probe --profile playable --jobs 1 -- save.json receipt.json
```

Tests cover material-component conservation and grading persistence, legacy save defaults, individual gold versus weighted populations, local plumes, edge-volume conservation, invariance under parcel subdivision, size-dependent optical cross section, settled-water clearing, display budgets and gold visibility. Native captures and measured performance belong in the companion playtest report; they are separate from calibrated slurry dynamics.

Channel optical queries now gather water, suspended solid volume and cross section with the same positive cardinal cubic footprint before division. This removes the first-derivative jumps of trilinear concentration samples at optical-cell planes. The stored projected moments and authoritative particle/material inventories remain unchanged. The established airborne outlet keeps its original trilinear sample before channel reconstruction. This presentation footprint smooths some subcell optical stratification; it is not physical diffusion or a calibrated dense-slurry closure. Homogeneous-mixture controls and nonuniform three-axis gradient checks cover the new sampling.

The screen-space channel reconstruction carries depth and additive water/extinction moments through the same four bilateral passes before division. A slope-aware projective range test suppresses kernel caps without treating an oblique water plane as a depth break. This positive presentation filter keeps a uniform mixture uniform, but is not an optical-volume or physical-transport solver. Its visible envelope can move; authoritative water, sediment and riffle state cannot. Detached outlet columns are excluded from the channel filter. See the [paired-filter native checks](../playtests/2026-10-23-crest-patches.md).

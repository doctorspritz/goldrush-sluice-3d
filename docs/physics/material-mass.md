# Reference bulk inventory and physical grains

`Material` stores conserved reference bulk volumes for soil, sand and gravel,
plus gold in grams. Reference dry densities remain the prototype assumptions:
1800, 1600 and 2000 kg/m³ respectively. Terrain paydirt maps to gravel, with the
same dry density. These values are not measured soil or porosity calibration.

Dry mineral mass is the sum of class reference volumes times their dry densities.
Physical mineral volume is that mass divided by 2650 kg/m³. Add gold mass divided
by 19300 kg/m³ for total grain volume. `Grain` now uses that same mass and true
solid volume for population, packed contact envelopes, DEM registration and FLIP
displacement. Density is total mass / total solid volume. A single quartz tracer
uses the inverse conversion when constructing its reference sand inventory.

Previously 0.02 m³ of soil was 36 kg in the material ledger but became 53 kg in
DEM. Independent class, feed, one-grain and excavation/custody/save controls now
preserve the intended mass. A real FLIP/DEM check compares actual registered
clump-template masses with their material inventory; it does not simply compare
two copies of the same bookkeeping method.

Terrain concentrations and historical `Suspension.solid_m3` fields retain their
reference bulk meaning and serialized values. Channel optics use true grain
volume and grain cross section; environmental optics convert each class before
summing its physical concentration. Saves remain version 3, with existing v2/v3
load handling; derived mass, contacts and subsequent trajectories are corrected.

The native benchmark adds a dry-mass gate alongside bulk, water and gold gates.
A global one-ppm tolerance is relative to the whole site's inventory, so local
handoff controls and actual DEM masses remain necessary. The laboratory assay's
quartz volumes are true mineral volumes; its report schema is 4 and explicitly
records the volume basis and mass-budget start after tracer injection.

This does not implement variable porosity, pore water, soil compaction, swelling,
packed-bed rheology or conservative interphase momentum. Weighted parcel contacts
retain their assumed packing fraction of 0.60. Legacy unweighted individual-grain
compatibility mode retains its older geometric solver treatment and is excluded
from the registered weighted-mass proof. Standard demo/current emitted parcels
use the volume-resolved path. Parent soil and phase-coupling issues remain open.

The mass / grain-volume / bulk-volume distinction follows the
[HEC sediment-property definitions](https://www.hec.usace.army.mil/confluence/rasdocs/d2sd/ras2dsedtr/6.0/model-description/water-and-sediment-properties/sediment-properties).
That reference supports the distinction and nominal quartz density; the inherited
class dry densities and packing fraction above remain this prototype's assumptions.

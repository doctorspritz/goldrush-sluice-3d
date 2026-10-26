# Finite erodible terrain support

`soil_response::pressure::RegisteredTerrainLayer` defines a closed mechanical envelope between two explicitly supplied heightfields. The saved-bank reference uses the actual registered ground triangles above the site's bedrock elevations on the same grid. It does not choose a lower datum, extend a face into an infinite plane, rebuild a hull or add collision thickness.

For each registered projected triangle, the upper and lower surfaces are affine. The envelope is the region between them inside that triangle's finite footprint. Its volume is

```text
V_triangle = projected_area * (thickness_0 + thickness_1 + thickness_2) / 3
```

Actual f32 triangle coefficients are promoted before construction. Matching vertex IDs and projected coordinates are required on both surfaces. A global vertex order chooses the same diagonal on every shared vertical face. Each wedge then has three disjoint tetrahedra. Only duplicate-vertex cells with exactly zero volume are omitted; there is no positive-thickness cutoff. Crossing surfaces, incompatible topology, unsupported holes, unrepresentable positive cells and exhausted bounds return errors.

The complete source domain is bounded at 65,536 triangles and 196,608 tetrahedra. Spatial queries discard only disjoint bounding intervals. Identity rotations use monotone addition of the represented cell bounds; other rotations bound the actual transformed vertices. Nonfinite relative frames and overflowing bounds are rejected. The selected source/target product is checked against the existing 1,048,576 pair-work limit before transformed contact-cell allocation. Geometry and clipping precision limits remain in force.

`Opposing::TerrainLayer` evaluates this finite region as rigid support against the compliant clod's specified pressure field. The terrain partition is disjoint, so cells share one union component and internal seams cannot add a second response. The existing rational rigid-contact integration supplies potential, force, torque and instantaneous power. This does not infer an elastic modulus, yield law or wet-soil parameters for terrain.

`captured_frame` is an explicit frozen observation of supplied coefficients. It consumes the envelope to reuse its bounded storage and never normalizes the captured quaternion. It is not a live geometry rebuild, a material-reference update or permission to discard contact history. Current poses, source cells and global coordinates remain part of the reference definition.

The envelope is mechanical geometry, not another material inventory. Its integrated volume is not substituted for Site's counted bulk, dry mass, gold or pore water. In particular, the registered mesh footprint and the cell-based stock convention are distinct representations. Subgrid stocks are not turned into new collision geometry. Constructing and observing the envelope does not edit either representation.

The lower boundary is supplied explicitly by the current Site environment. The retained-world snapshot does not persist this new envelope. Persisting and validating the lower reference across environment changes remains part of live adoption.

Qualification covers direct projected-volume integrals, sloping lower surfaces, thin and zero columns, shared-face closure before and after frame mapping, finite-box equivalence across seams and domain edges, twelve pose-energy derivatives, geometry-edit response and resource/precision refusal. The actual saved-bank check observes every clod/heightfield pair, selects a real positive-overlap case for pose and material-partition qualification, and preserves the complete owner encoding. Results are pinned in [the terrain qualification report](../playtests/2026-10-25-finite-terrain-layer.md).

Bedrock beneath the lower surface is a separate contact domain. Surface changes correctly change this reference's potential, but the corresponding terrain-edit/material-removal work is not yet coupled. No pressure force is applied to Session or the retained-world step, no strain credit is assigned to old constraints, and saves retain their existing policy. Deep-bedrock ownership, live force/friction/joint/motor/CCD coupling, plasticity, wet contact, physical timestep and clod-size convergence, calibration and real-time performance remain open.

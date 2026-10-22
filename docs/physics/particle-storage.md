# Particle storage and sustained admission

The initial 4,096-grain reservation is derived allocation metadata. It no longer
blocks river withdrawal, inlet parcel emission, hopper feed or loading a valid
larger shift. All grain positions, velocities, APIC matrices, material components,
diameters, rotations, angular velocities and contact history remain authoritative.
No grains are merged, resampled, enlarged or discarded to stay within that count.

`INITIAL_GRAIN_STORAGE` names the initial reservation. `MAX_GRAINS` remains a
historical public alias for older callers; production admission does not use it.
Finite water capacity and the inlet custody buffer still protect represented
material. Those numerical bounds require separate hydraulic/dense-phase work.
The physical grading, water reservation, grid, timestep, pressure sweeps and
contact impulse equations retain their existing configuration.

## Growth contract

`Hydraulics` chooses the next power-of-two grain reservation from the live
population. A restored population receives its required reservation immediately;
a running population grows before its next complete FLIP upload. Allocation
counts are absent from saves, so adapter/storage choices do not reinterpret
physical parcel size or material.

`GpuFlip3D::reallocate_for_upload` requires no prepared or pending particle
readback. It constructs replacement particle storage with the same grid and
paired-transfer setting, copies the existing pressure warm start and gravel
obstacle input in queue order, and preserves public coefficients, boundary
settings, pressure-stencil selection, sorting settings and the optional GPU timer.
The CPU DEM/grid/contact owner remains in place.

After replacement, the caller must upload **all** positions, velocities, APIC
matrices, phase densities, represented volumes, diameters, grid/bed data and SDF
before stepping or sampling the new particle buffers. The new solver marks its
SDF upload invalid so the existing geometry upload runs. This API is not a
standalone resize of partially uploaded buffers.

A real-GPU control compares preallocated and dynamically grown transfers with
nondefault coefficients, obstacles and density projection enabled/disabled.
The cold-rebuild negative control differs when warm pressure matters. Another
control restores and continues the complete physical state above 4,096 grains.
Fixed-step production runs separately compare complete saved states, sorting only
the three unordered contact-map export arrays; particle arrays and floats retain
their original order and precision.

## Resource preflight and failure

The allocation policy plans at 1 KiB per reserved particle, including the existing
profile water reservation, against a 512 MiB planning budget. It also checks the
adapter's storage-binding limit and its largest combined readback allocation:
five 16-byte position/velocity/APIC vectors per reserved particle. Integer
count/size overflow is rejected before construction. The public growth API checks
the same largest-buffer condition independently. The GPU refusal control uses a
device with reduced per-buffer limits where staging is the tighter constraint, without allocating the
oversized request.

This planning estimate is not measured peak RSS or total Metal residency.
Construction can temporarily retain both old and new buffers; grids, renderer,
CPU state and other applications have additional costs. Native measurements in
the companion report apply only to the tested population and workload.

A refusal returns an explicit error. The existing worker publishes the current
material state, pauses the shift and skips the second step of a two-tick packet.
It does not silently lower water delivery at a grain quota. A CPU worker control
injects that refusal and verifies that actual dug material remains unchanged and
serializes/restores correctly. This is a failure-path control, not an allocation
stress test or a transactional rollback of the domain tick.

Save version 5 and its 128,000,000-byte input/output limit remain. Passing the
4,103/4,308-grain controls does not establish that every possible population under
the allocation planning bound fits that save limit.

## Contact scratch storage

Larger populations exposed a projection fallback that gathered and sorted all
candidate pairs at once. Large hash-based sweeps now gather and sort one source
body's candidates, then resolve them in the same lexicographic order. Cell
membership is the snapshot from the start of the sweep, including after an
earlier contact corrects a position. The large-population fallback uses O(N)
candidate scratch rather than retaining a quadratic pair list.

The accelerated mask path can also run above 8,192 bodies. Both the axis masks
and occupied-row matrix are checked before allocation: each active array is at
most 1,048,576 u64 words (8 MiB). Vector capacity may round up; the lookup, body,
contact-history and other solver allocations are separate. An oversized matrix
uses the streaming fallback. The constant small-body all-pairs path remains.

Controls compare the streaming sweep against the original globally sorted pair
order for rotated mixed contacts, compare 8,194 bodies with independent isolated
pair solves, and exercise an oversized occupied-row matrix without allocating
it. These establish ordering and scratch behavior; they do not establish dense
packed-bed convergence or calibrated friction.

## Remaining physical gates

Storage equivalence is a numerical invariant. The weighted parcel contact
representation, two-way momentum exchange, packed-bed porosity, scour/washout,
matched apparatus experiments and long-duration equilibrium remain requirements
of the [accuracy protocol](accuracy.md). Real-time performance and a constructed
recirculating pond remain open. The [native/fixed-step report](../playtests/2026-10-23-grain-storage.md)
records this bounded foundation and its measured limits.

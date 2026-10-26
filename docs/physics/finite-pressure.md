# Finite pressure-field contact

`mining_sim::soil_response::pressure` evaluates instantaneous elastic contact on complete registered closed geometry. It is available as a material-model component and a checked reference. Session and the retained-world step do not select it. It applies no force to the saved bank and supplies no strain-energy credit to the existing solver.

The pressure-field approach uses a scalar material field inside each solid. See [Elandt et al., pressure-field contact](https://arxiv.org/abs/1904.11433) for the model background. This implementation's peak pressure is a specified pressure scale in pascals; it is not calibrated soil stiffness. [Drake's parameter discussion](https://drake.mit.edu/doxygen_cxx/group__hydro__params.html) explains the distinction between hydroelastic and Young's moduli and the role of material depth.

## Geometry and field ownership

`RegisteredSolid::from_hull` retains every registered vertex and ordered face. The declared closed boundary is the complete fan triangulation of those faces. A star partition about the volume centroid supplies finite reference tetrahedra. There is no regenerated hull, collision skin, radius proxy or vertex reduction.

`PressureBody::new` gives each outer triangle zero pressure and the centroid the explicitly supplied positive pressure. Its piecewise-affine field belongs to that reference material. `refined` subdivides the same cells while retaining their original affine fields; these checks establish partition convergence, not convergence under changing physical clod size.

Moving material coordinates use a body CoM frame and validated unit rotations. `captured_frame` separately supports frozen observation of promoted captured coefficients without quaternion normalization. It is not permission to rebuild or reset material references during integration.

## Potential, forces and power

For compliant solids A and B, the implemented potential is

```text
U = integral_(A intersection B) min(p_A, p_B) dV
F_A = integral_(p_A < p_B) gradient(p_A) dV
T_A = integral_(p_A < p_B) (x - CoM_A) cross gradient(p_A) dV
```

B uses its own weak-pressure region. Both wrenches are evaluated independently, with raw pair force/moment closure and instantaneous power residuals reported. Pressure times volume has units of joules. A rigid opponent uses the limiting pressure infinity; its reaction is paired about its actual CoM.

Equal affine pressure over positive volume is nonsmooth. The supplied twists select its one-sided material derivative. Equal pressure rates use the symmetric split. Tie volume is reported explicitly; damping or an artificial regularizer is not introduced. This case still needs qualification in a motion integrator.

## Hollow rigid compounds

`Opposing::RigidCompound` means the union of complete closed parts belonging to one rigid body. Inclusion-exclusion operates between parts, whose internal star tetrahedra are disjoint. Thus steel overlaps are integrated once and a hollow bucket cavity remains open.

Rigid compound clipping, cap ordering and oriented-volume signs use rational arithmetic on the represented coordinates. Positive tetrahedral pressure integrals and moments then use compensated f64 sums. Reported roundoff estimates are estimates, not formal error certificates. No volume epsilon, sign clamp, velocity correction or post-solve energy correction is used.

A bounding interval disjoint from the entire compliant body proves zero overlap and avoids expensive construction. It does not change source geometry, pressure, force targets or material inventory. Compound work is admitted before allocating transformed cells. Other limits bound reference cells, candidate products, clip vertices/faces, rational coordinate size and overlapping-component recursion; exhaustion returns an error rather than reducing geometry.

## Current qualification and remaining integration

The saved-bank check observes all 1,040 clods, all cached clod/clod pairs and every clod/bucket compound pair. It preserves the complete input/owner encoding. Selected actual clod and steel wrenches are also checked independently against pose-energy differences and same-field partitions. Numerical results and source/input hashes are in [the qualification receipt](../playtests/2026-10-25-finite-contact.json).

The separate [finite erodible terrain layer](finite-terrain-layer.md) now observes the registered terrain surface above explicitly supplied bedrock elevations. Its saved-bank qualification covers every clod/heightfield pair, finite edges, shared-cell seams and real-case pose/partition controls. It is a mechanical envelope rather than an additional soil inventory. Bedrock beneath its lower surface and terrain-edit work remain separate domains.

The live solver, joints/motors, friction, damping, yield, CCD, reference persistence and contact-creation/removal work still need a coupled policy. Soil calibration, physical timestep and clod-size convergence, and real-time performance remain open. In particular, the existing saved live-bank baseline still fails its unchanged mechanical-energy gate. This component does not relabel that gain as elastic storage.

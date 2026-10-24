//! Independent registered-geometry witness audit at captured solver poses.
//! Cluster subshape ids and engine contact/distance queries are never inputs.
use super::*;
use rapier3d::parry::shape::Shape;
use serde_json::{json, Value};
use std::collections::HashMap;

const TOLERANCE_M: f64 = 0.00001;
const MAX_LEAVES: usize = 65_536;

/// Exact f32 coefficients promoted before JSON serialization. Do not replace
/// this with the body's stored (possibly previous-substep) world transform.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub(super) struct CapturedPose {
    pub translation: [f64; 3],
    pub rotation_xyzw: [f64; 4],
}
impl CapturedPose {
    fn point(self, p: DVec3) -> DVec3 {
        DQuat::from_array(self.rotation_xyzw) * p + DVec3::from_array(self.translation)
    }
    fn direction(self, p: DVec3) -> DVec3 {
        DQuat::from_array(self.rotation_xyzw) * p
    }
    fn valid(self) -> bool {
        DVec3::from_array(self.translation).is_finite()
            && DQuat::from_array(self.rotation_xyzw).is_finite()
            && (DQuat::from_array(self.rotation_xyzw).length_squared() - 1.).abs() < 0.0001
    }
}
fn promoted(pose: &r::Pose) -> CapturedPose {
    CapturedPose {
        translation: pose.translation.to_array().map(f64::from),
        rotation_xyzw: pose.rotation.to_array().map(f64::from),
    }
}
#[derive(Clone)]
struct Leaf {
    path: Vec<u32>,
    vertices: Vec<DVec3>,
    faces: Vec<[usize; 3]>,
    closed: bool,
    // A registered heightfield represents the ground below its top triangles.
    // Open triangle meshes do not acquire an invented solid side.
    ground_up: Option<DVec3>,
}
fn supporting_faces(vertices: &[DVec3]) -> Result<Vec<[usize; 3]>, String> {
    let mut faces = Vec::new();
    for a in 0..vertices.len() {
        for b in a + 1..vertices.len() {
            for c in b + 1..vertices.len() {
                let n = (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a]);
                if n.length_squared() < 1e-24 {
                    continue;
                }
                let n = n.normalize();
                let distances: Vec<_> = vertices.iter().map(|p| n.dot(*p - vertices[a])).collect();
                let lo = distances.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if hi <= 1e-11 && lo < -1e-11 {
                    faces.push([a, b, c]);
                } else if lo >= -1e-11 && hi > 1e-11 {
                    faces.push([a, c, b]);
                }
            }
        }
    }
    if faces.len() < 4 {
        return Err("degenerate or unproved closed polyhedron".into());
    }
    Ok(faces)
}
fn enumerate(
    shape: &dyn Shape,
    transforms: &[CapturedPose],
    path: &mut Vec<u32>,
    out: &mut Vec<Leaf>,
) -> Result<(), String> {
    if path.len() > 8 {
        return Err("nested geometry exceeds explicit oracle bound".into());
    }
    if let Some(compound) = shape.as_compound() {
        for (index, (pose, child)) in compound.shapes().iter().enumerate() {
            path.push(index as u32);
            let mut nested = transforms.to_vec();
            nested.push(promoted(pose));
            enumerate(child.as_ref(), &nested, path, out)?;
            path.pop();
        }
        return Ok(());
    }
    let transform = |mut p: DVec3| {
        for &pose in transforms.iter().rev() {
            p = pose.point(p);
        }
        p
    };
    let direction = |mut p: DVec3| {
        for &pose in transforms.iter().rev() {
            p = pose.direction(p);
        }
        p
    };
    let mut push = |points: Vec<r::Vector>,
                    closed: bool,
                    ground: bool,
                    child: Option<u32>|
     -> Result<(), String> {
        if out.len() >= MAX_LEAVES {
            return Err("complete registered geometry exceeds oracle leaf bound".into());
        }
        if points.len() > 32 {
            return Err("closed hull exceeds oracle vertex bound; no reduction applied".into());
        }
        let vertices: Vec<_> = points
            .into_iter()
            .map(|p| transform(DVec3::from_array(p.to_array().map(f64::from))))
            .collect();
        if vertices.iter().any(|p| !p.is_finite()) {
            return Err("non-finite registered geometry".into());
        }
        let faces = if closed {
            supporting_faces(&vertices)?
        } else {
            vec![[0, 1, 2]]
        };
        let mut leaf_path = path.clone();
        if let Some(child) = child {
            leaf_path.push(child);
        }
        out.push(Leaf {
            path: leaf_path,
            vertices,
            faces,
            closed,
            ground_up: ground.then(|| direction(DVec3::Y)),
        });
        Ok(())
    };
    if let Some(hull) = shape.as_convex_polyhedron() {
        push(hull.points().to_vec(), true, false, None)
    } else if let Some(cuboid) = shape.as_cuboid() {
        push(
            (0..8)
                .map(|i| {
                    r::Vector::new(
                        if i & 1 == 0 {
                            -cuboid.half_extents.x
                        } else {
                            cuboid.half_extents.x
                        },
                        if i & 2 == 0 {
                            -cuboid.half_extents.y
                        } else {
                            cuboid.half_extents.y
                        },
                        if i & 4 == 0 {
                            -cuboid.half_extents.z
                        } else {
                            cuboid.half_extents.z
                        },
                    )
                })
                .collect(),
            true,
            false,
            None,
        )
    } else if let Some(field) = shape.as_heightfield() {
        for id in 0..field.nrows() * field.ncols() * 2 {
            if let Some(triangle) = field.triangle_at_id(id as u32) {
                push(triangle.vertices().to_vec(), false, true, Some(id as u32))?;
            }
        }
        Ok(())
    } else if let Some(mesh) = shape.as_trimesh() {
        for id in 0..mesh.num_triangles() {
            push(
                mesh.triangle(id as u32).vertices().to_vec(),
                false,
                false,
                Some(id as u32),
            )?;
        }
        Ok(())
    } else if let Some(triangle) = shape.as_triangle() {
        push(triangle.vertices().to_vec(), false, false, None)
    } else {
        Err(format!(
            "unsupported registered shape {:?}; no proxy substituted",
            shape.shape_type()
        ))
    }
}
// Ericson's point/triangle regions in f64, independent of the engine dispatcher.
fn triangle_distance(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0. && d2 <= 0. {
        return ap.length();
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0. && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0. && d1 >= 0. && d3 <= 0. {
        return (p - (a + ab * (d1 / (d1 - d3)))).length();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0. && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0. && d2 >= 0. && d6 <= 0. {
        return (p - (a + ac * (d2 / (d2 - d6)))).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0. && d4 - d3 >= 0. && d5 - d6 >= 0. {
        return (p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))))).length();
    }
    (p - (a + ab * (vb / (va + vb + vc)) + ac * (vc / (va + vb + vc)))).length()
}
fn query(leaf: &Leaf, pose: CapturedPose, point: DVec3) -> (f64, bool) {
    let vertices: Vec<_> = leaf.vertices.iter().map(|&p| pose.point(p)).collect();
    let mut distance = f64::INFINITY;
    let mut maximum_signed = f64::NEG_INFINITY;
    let mut inside_ground = false;
    for &[a, b, c] in &leaf.faces {
        let (a, b, c) = (vertices[a], vertices[b], vertices[c]);
        let cross = (b - a).cross(c - a);
        if cross.length_squared() < 1e-24 {
            continue;
        }
        let normal = cross.normalize();
        distance = distance.min(triangle_distance(point, a, b, c));
        maximum_signed = maximum_signed.max(normal.dot(point - a));
        if let Some(up) = leaf.ground_up {
            let up = pose.direction(up).normalize();
            let normal = if normal.dot(up) >= 0. {
                normal
            } else {
                -normal
            };
            let denominator = normal.dot(up);
            if denominator > 1e-8 {
                let signed = normal.dot(point - a);
                let projected = point - up * (signed / denominator);
                inside_ground |=
                    signed < -TOLERANCE_M && triangle_distance(projected, a, b, c) < 1e-9;
            }
        }
    }
    (
        distance,
        (leaf.closed && maximum_signed < -TOLERANCE_M) || inside_ground,
    )
}

/// Caches only registered geometry, never engine distances or current poses.
/// The caller must use the same immutable scene as the observed solver rows.
pub(super) struct Oracle<'a> {
    world: &'a r::PhysicsWorld,
    registered: HashMap<r::ColliderHandle, Result<Vec<Leaf>, String>>,
}
impl<'a> Oracle<'a> {
    pub fn new(world: &'a r::PhysicsWorld) -> Self {
        Self {
            world,
            registered: HashMap::new(),
        }
    }
    fn side(
        &mut self,
        handle: r::ColliderHandle,
        pose: CapturedPose,
        anchor: DVec3,
        outward: DVec3,
    ) -> Value {
        if !pose.valid()
            || !anchor.is_finite()
            || !outward.is_finite()
            || outward.length_squared() < 1e-20
        {
            return json!({"certified":false,"reason":"missing or invalid captured geometry/normal"});
        }
        let leaves = self.registered.entry(handle).or_insert_with(|| {
            let collider = self
                .world
                .colliders
                .get(handle)
                .ok_or("captured collider generation is not registered")?;
            let mut leaves = Vec::new();
            enumerate(collider.shape(), &[], &mut Vec::new(), &mut leaves)?;
            Ok(leaves)
        });
        let leaves = match leaves {
            Ok(leaves) => leaves,
            Err(error) => return json!({"certified":false,"reason":error}),
        };
        if leaves.is_empty() {
            return json!({"certified":false,"reason":"registered shape has no leaves"});
        }
        let probe = outward.normalize() * (4. * TOLERANCE_M);
        let mut owners = Vec::new();
        let mut nearest = f64::INFINITY;
        let mut inside = false;
        let mut outside_probe_clear = true;
        let mut inward_probe_inside = false;
        let mut closed_sides = true;
        for leaf in leaves.iter() {
            let (distance, interior) = query(leaf, pose, anchor);
            nearest = nearest.min(distance);
            inside |= interior;
            if distance <= TOLERANCE_M {
                owners.push(leaf.path.clone());
            }
            let (outside_distance, outside_inside) = query(leaf, pose, anchor + probe);
            outside_probe_clear &= !outside_inside && outside_distance > TOLERANCE_M;
            inward_probe_inside |= query(leaf, pose, anchor - probe).1;
            closed_sides &= leaf.closed || leaf.ground_up.is_some();
        }
        // A shared internal face is not made exterior merely because it is on
        // two leaf boundaries. Reject ownership ambiguity rather than inventing
        // the child named by a cluster's first contributing manifold.
        let unique = owners.len() == 1;
        let exterior =
            unique && !inside && outside_probe_clear && inward_probe_inside && closed_sides;
        let mut result = json!({"certified":exterior,"registered_leaf_count":leaves.len(),"geometry_complete":true,
            "nearest_registered_leaf_surface_m":nearest,"owner_count":owners.len(),
            "owner_paths":owners.iter().take(16).collect::<Vec<_>>(),"unique_leaf_witness":unique,
            "inside_registered_solid":inside,"outward_probe_clear":outside_probe_clear,
            "inward_probe_inside":inward_probe_inside,"solid_sides_known":closed_sides,
            "ambiguous_or_internal_seam_unproved":!unique || inside || !outside_probe_clear,
            "open_mesh_sidedness_unproved":!closed_sides,"tolerance_m":TOLERANCE_M,
            "exterior_probe_distance_m":probe.length()});
        if leaves.iter().any(|leaf| leaf.ground_up.is_some()) {
            result.as_object_mut().unwrap().insert(
                "heightfield_model".into(),
                json!("registered top triangles with downward solid sides"),
            );
        }
        result
    }
    /// `force_direction1` is the actual solver force on collider one. Its
    /// outward surface normal points the opposite way; collider two reverses it.
    pub fn inspect(
        &mut self,
        handles: [r::ColliderHandle; 2],
        poses: [CapturedPose; 2],
        anchors: [DVec3; 2],
        force_direction1: DVec3,
    ) -> Value {
        let a = self.side(handles[0], poses[0], anchors[0], -force_direction1);
        let b = self.side(handles[1], poses[1], anchors[1], force_direction1);
        json!({"both_exterior_witnesses_certified":a["certified"]==true && b["certified"]==true,
            "sides":[a,b],"witness_normal_gap_m":(anchors[0]-anchors[1]).dot(force_direction1.normalize_or_zero()),
            "cluster_subshape_ids_used_as_ownership":false,"engine_distance_query_used":false,
            "captured_current_collider_poses_used":true,"full_contact_equation_certified":false,
            "geometric_pair_distance_or_penetration_proved":false})
    }
}

fn interval(vertices: &[DVec3], axis: DVec3) -> [f64; 2] {
    vertices
        .iter()
        .map(|p| p.dot(axis))
        .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], d| {
            [lo.min(d), hi.max(d)]
        })
}
fn edge_candidates(leaf: &Leaf) -> Vec<[usize; 2]> {
    let mut edges = std::collections::BTreeSet::new();
    for &[a, b, c] in &leaf.faces {
        for [i, j] in [[a, b], [b, c], [c, a]] {
            edges.insert([i.min(j), i.max(j)]);
        }
    }
    edges.into_iter().collect()
}
fn is_single_polytope(shape: &dyn Shape, depth: usize) -> bool {
    if depth > 8 {
        return false;
    }
    if let Some(compound) = shape.as_compound() {
        return compound.shapes().len() == 1
            && is_single_polytope(compound.shapes()[0].1.as_ref(), depth + 1);
    }
    shape.as_convex_polyhedron().is_some() || shape.as_cuboid().is_some()
}
impl Oracle<'_> {
    fn single_polytope(&self, handle: r::ColliderHandle) -> Result<Leaf, String> {
        let collider = self
            .world
            .colliders
            .get(handle)
            .ok_or("captured collider generation is not registered")?;
        if !is_single_polytope(collider.shape(), 0) {
            return Err("pair SAT admits only one registered closed convex polyhedral leaf per collider; unions, curved shapes and terrain are unproved".into());
        }
        let mut leaves = Vec::new();
        enumerate(collider.shape(), &[], &mut Vec::new(), &mut leaves)?;
        if leaves.len() != 1 || !leaves[0].closed {
            return Err(
                "registered collider did not produce exactly one closed convex leaf".into(),
            );
        }
        Ok(leaves.remove(0))
    }
    /// Independent complete polytope SAT, using registered vertices transformed
    /// by captured current collider poses. Positive best separation proves a
    /// separating axis; a negative value is the least signed overlap over the
    /// complete face/edge-cross set. It is not Euclidean distance when separated.
    /// `force_direction1` points in the actual impulse direction on side one;
    /// its directed support gap is min(A.n)-max(B.n), with no normal reversal.
    pub fn pair_primitive_audit(
        &self,
        handles: [r::ColliderHandle; 2],
        poses: [CapturedPose; 2],
        force_direction1: DVec3,
    ) -> Value {
        if poses.iter().any(|p| !p.valid())
            || !force_direction1.is_finite()
            || force_direction1.length_squared() == 0.
        {
            return json!({"applicable":false,"reason":"invalid captured pose or force direction"});
        }
        let leaves: Vec<_> = match handles
            .into_iter()
            .map(|h| self.single_polytope(h))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(leaves) => leaves,
            Err(error) => {
                return json!({"applicable":false,"reason":error,
                "cluster_subshape_ids_used_as_ownership":false,"engine_distance_query_used":false})
            }
        };
        let vertices: [Vec<DVec3>; 2] = std::array::from_fn(|i| {
            leaves[i]
                .vertices
                .iter()
                .map(|&p| poses[i].point(p))
                .collect()
        });
        let edges: [Vec<[usize; 2]>; 2] = std::array::from_fn(|i| edge_candidates(&leaves[i]));
        let mut best = f64::NEG_INFINITY;
        let mut best_axis = DVec3::ZERO;
        let mut best_source = "";
        let mut best_indices = [0; 4];
        let mut tested_faces = 0;
        let mut tested_crosses = 0;
        let mut parallel_crosses = 0;
        let mut consider = |axis: DVec3, source: &'static str, indices: [usize; 4]| {
            if axis.length_squared() == 0. {
                return false;
            }
            let mut axis = axis.normalize();
            let a = interval(&vertices[0], axis);
            let b = interval(&vertices[1], axis);
            let forward = a[0] - b[1];
            let reverse = b[0] - a[1];
            let gap = if reverse > forward {
                axis = -axis;
                reverse
            } else {
                forward
            };
            if gap > best {
                best = gap;
                best_axis = axis;
                best_source = source;
                best_indices = indices;
            }
            true
        };
        for side in 0..2 {
            for (face, &[a, b, c]) in leaves[side].faces.iter().enumerate() {
                let normal = (vertices[side][b] - vertices[side][a])
                    .cross(vertices[side][c] - vertices[side][a]);
                if consider(normal, "supporting_face", [side, face, 0, 0]) {
                    tested_faces += 1;
                }
            }
        }
        for (ea, &[a, b]) in edges[0].iter().enumerate() {
            for (eb, &[c, d]) in edges[1].iter().enumerate() {
                let cross =
                    (vertices[0][b] - vertices[0][a]).cross(vertices[1][d] - vertices[1][c]);
                if consider(cross, "edge_cross", [ea, eb, 0, 0]) {
                    tested_crosses += 1;
                } else {
                    parallel_crosses += 1;
                }
            }
        }
        assert!(best.is_finite() && tested_faces > 0);
        let raw_normal_length = force_direction1.length();
        let normal = force_direction1 / raw_normal_length;
        let normal_intervals = [
            interval(&vertices[0], normal),
            interval(&vertices[1], normal),
        ];
        let best_intervals = [
            interval(&vertices[0], best_axis),
            interval(&vertices[1], best_axis),
        ];
        let numeric_tolerance_m = 1e-9;
        let evidence: Vec<_> = (0..2).map(|i| {
            let lo=vertices[i].iter().copied().fold(DVec3::splat(f64::INFINITY),DVec3::min);
            let hi=vertices[i].iter().copied().fold(DVec3::splat(f64::NEG_INFINITY),DVec3::max);
            let support=if i==0 {normal_intervals[i][0]} else {normal_intervals[i][1]};
            json!({"collider_handle":handles[i].into_raw_parts(),"registered_leaf_path":leaves[i].path,
                "captured_collider_pose":poses[i],"vertices_collider_local_m":leaves[i].vertices.iter().map(|p|p.to_array()).collect::<Vec<_>>(),
                "vertices_world_m":vertices[i].iter().map(|p|p.to_array()).collect::<Vec<_>>(),
                "supporting_face_vertex_triples":leaves[i].faces,"edge_candidate_vertex_pairs":edges[i],
                "world_bounds_m":[lo.to_array(),hi.to_array()],"constraint_normal_support_vertex_indices":vertices[i].iter().enumerate()
                    .filter_map(|(j,p)|((p.dot(normal)-support).abs()<=numeric_tolerance_m).then_some(j)).collect::<Vec<_>>()})
        }).collect();
        json!({"applicable":true,"complete_polytope_sat_axes_evaluated":true,
            "best_signed_separation_m":best,"best_axis_force_on_side1":best_axis.to_array(),
            "best_axis_support_intervals_m":best_intervals,"best_axis_source":best_source,"best_axis_source_indices":best_indices,
            "separated_beyond_numeric_tolerance":best>numeric_tolerance_m,"overlapping_beyond_numeric_tolerance":best< -numeric_tolerance_m,
            "numeric_classification_tolerance_m":numeric_tolerance_m,"tested_face_axes":tested_faces,
            "tested_edge_cross_axes":tested_crosses,"exactly_parallel_edge_crosses_skipped":parallel_crosses,
            "raw_constraint_force_direction1":force_direction1.to_array(),"raw_constraint_normal_length":raw_normal_length,
            "constraint_normal_unit_force_on_side1":normal.to_array(),"constraint_normal_support_intervals_m":normal_intervals,
            "constraint_normal_signed_interval_gap_m":normal_intervals[0][0]-normal_intervals[1][1],
            "constraint_normal_opposite_interval_gap_m":normal_intervals[1][0]-normal_intervals[0][1],
            "normal_convention":"force on collider one; positive signed gap is min(A dot n)-max(B dot n); best axis oriented to translate collider one away from collider two",
            "sat_value_note":"negative best signed separation is polytope minimum-translation overlap; positive is a SAT directional gap, not Euclidean pair distance",
            "edge_set_note":"all supporting-triangle edges, including coplanar diagonals; this superset includes every real polytope edge and adds only valid separating-axis tests",
            "registered_geometry":evidence,"captured_pose_coefficients_renormalized":false,
            "cluster_subshape_ids_used_as_ownership":false,"engine_distance_query_used":false,
            "different_mtd_alone_proves_invalid_contact":false,"full_contact_equation_certified":false})
    }
}

#[test]
fn primitive_pair_sat_resolves_thin_overlap_separation_and_rigid_rotation() {
    let mut world = r::PhysicsWorld::new();
    let handles = std::array::from_fn(|_| {
        world
            .insert(
                r::RigidBodyBuilder::fixed(),
                r::ColliderBuilder::cuboid(1., 0.1, 0.4),
            )
            .1
    });
    let oracle = Oracle::new(&world);
    let global = DQuat::from_axis_angle(DVec3::new(1., 2., 3.).normalize(), 0.71);
    for (rotation, relative, center_x, expected) in [
        (DQuat::IDENTITY, DQuat::IDENTITY, 1.999, -0.001),
        (global, DQuat::IDENTITY, 1.999, -0.001),
        (
            global,
            DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            1. + f64::from(0.1_f32) - 0.001,
            -0.001,
        ),
        (global, DQuat::IDENTITY, 2.005, 0.005),
    ] {
        let origin = DVec3::new(20., 3., -4.);
        let poses = [
            CapturedPose {
                translation: origin.to_array(),
                rotation_xyzw: rotation.to_array(),
            },
            CapturedPose {
                translation: (origin + rotation * DVec3::new(center_x, 0., 0.)).to_array(),
                rotation_xyzw: (rotation * relative).to_array(),
            },
        ];
        let audit = oracle.pair_primitive_audit(handles, poses, rotation * -DVec3::X);
        assert_eq!(audit["applicable"], true);
        let best = audit["best_signed_separation_m"].as_f64().unwrap();
        let directed = audit["constraint_normal_signed_interval_gap_m"]
            .as_f64()
            .unwrap();
        assert!((best - expected).abs() < 1e-9, "{audit}");
        assert!((directed - expected).abs() < 1e-9, "{audit}");
        assert_eq!(audit["separated_beyond_numeric_tolerance"], expected > 0.);
        assert_eq!(audit["overlapping_beyond_numeric_tolerance"], expected < 0.);
    }
}

#[test]
fn registered_compound_ownership_does_not_guess_cluster_child_or_certify_internal_seam() {
    let mut world = r::PhysicsWorld::new();
    let shape = r::SharedShape::compound(vec![
        (
            r::Pose::translation(-1., 0., 0.),
            r::SharedShape::cuboid(1., 1., 1.),
        ),
        (
            r::Pose::translation(1., 0., 0.),
            r::SharedShape::cuboid(1., 1., 1.),
        ),
    ]);
    let (_, handle) = world.insert(r::RigidBodyBuilder::fixed(), r::ColliderBuilder::new(shape));
    let pose = CapturedPose {
        translation: [20., 3., 22.],
        rotation_xyzw: [0., 0., 0., 1.],
    };
    let mut oracle = Oracle::new(&world);
    let exterior = oracle.side(handle, pose, DVec3::new(22., 3., 22.), DVec3::X);
    assert_eq!(exterior["owner_paths"], json!([[1]]));
    assert_eq!(exterior["certified"], true);
    let seam = oracle.side(handle, pose, DVec3::new(20., 3., 22.), DVec3::X);
    assert_eq!(seam["owner_count"], 2);
    assert_eq!(seam["certified"], false);
    assert_eq!(seam["ambiguous_or_internal_seam_unproved"], true);
    let outside = oracle.side(handle, pose, DVec3::new(22.001, 3., 22.), DVec3::X);
    assert_eq!(outside["owner_count"], 0);
    assert_eq!(outside["certified"], false);
}
#[test]
fn registered_heightfield_surface_ownership_uses_captured_pose_and_actual_triangles() {
    let mut world = r::PhysicsWorld::new();
    let shape = r::SharedShape::heightfield(
        r::Array2::new(2, 2, vec![0.; 4]),
        r::Vector::new(2., 1., 2.),
    );
    let (_, handle) = world.insert(r::RigidBodyBuilder::fixed(), r::ColliderBuilder::new(shape));
    let pose = CapturedPose {
        translation: [20., 3., 22.],
        rotation_xyzw: [0., 0., 0., 1.],
    };
    let mut oracle = Oracle::new(&world);
    let surface = oracle.side(handle, pose, DVec3::new(20.2, 3., 22.5), DVec3::Y);
    assert_eq!(surface["registered_leaf_count"], 2);
    assert_eq!(surface["owner_count"], 1);
    assert_eq!(surface["certified"], true);
    let stale = oracle.side(handle, pose, DVec3::new(0.2, 0., 0.5), DVec3::Y);
    assert_eq!(stale["certified"], false);
    let below = oracle.side(handle, pose, DVec3::new(20.2, 2.999, 22.5), DVec3::Y);
    assert_eq!(below["inside_registered_solid"], true);
    assert_eq!(below["certified"], false);
}

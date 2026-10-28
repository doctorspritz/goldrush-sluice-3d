//! Actual registered closed steel children, including their overlap and hollow
//! cavity. No compound hull, shell simplification or planar proxy is substituted.
use super::*;
use std::time::Instant;
type Clods = HashMap<r::ColliderHandle, (PressureBody, RigidMotion)>;
fn components(
    shape: &dyn rapier3d::parry::shape::Shape,
    depth: usize,
) -> Result<Vec<RegisteredSolid>, String> {
    if depth > 8 {
        return Err("Registered compound nesting bound exceeded".into());
    }
    if let Some(hull) = shape.as_convex_polyhedron() {
        return Ok(vec![RegisteredSolid::from_hull(hull)?]);
    }
    if let Some(compound) = shape.as_compound() {
        let mut solids = Vec::new();
        for (pose, child) in compound.shapes() {
            for solid in components(child.as_ref(), depth + 1)? {
                solids
                    .push(solid.captured_frame(vector(pose.translation), rotation(pose.rotation))?);
                if solids.len() > 1024 {
                    return Err("Registered compound part bound exceeded".into());
                }
            }
        }
        return Ok(solids);
    }
    Err(format!(
        "Unqualified registered compound child {:?}",
        shape.shape_type()
    ))
}
pub(super) struct Steel {
    models: HashMap<r::ColliderHandle, (Vec<RegisteredSolid>, RigidMotion)>,
    pub admitted: usize,
    positive: usize,
    zero: usize,
    failures: usize,
    examples: Vec<serde_json::Value>,
    potential: f64,
    volume: f64,
    force: f64,
    moment: f64,
    power: f64,
    seconds: f64,
    max_pair_seconds: f64,
    visits: usize,
    max_depth: usize,
    energy_roundoff: f64,
    volume_roundoff: f64,
    input_tetrahedra: usize,
    candidate_tetrahedra: usize,
    samples: qualification::Candidates,
}
impl Steel {
    pub fn new(world: &r::PhysicsWorld) -> Result<Self, String> {
        let mut models = HashMap::new();
        for (id, collider) in world.colliders.iter() {
            if collider.shape().as_compound().is_none() {
                continue;
            }
            let mut parts = Vec::new();
            let mut motion = None;
            for solid in components(collider.shape(), 0)? {
                let (solid, m) = captured(world, id, solid)?;
                parts.push(solid);
                motion = Some(m);
            }
            models.insert(id, (parts, motion.ok_or("Empty registered compound")?));
        }
        Ok(Self {
            models,
            admitted: 0,
            positive: 0,
            zero: 0,
            failures: 0,
            examples: Vec::new(),
            potential: 0.,
            volume: 0.,
            force: 0.,
            moment: 0.,
            power: 0.,
            seconds: 0.,
            max_pair_seconds: 0.,
            visits: 0,
            max_depth: 0,
            energy_roundoff: 0.,
            volume_roundoff: 0.,
            input_tetrahedra: 0,
            candidate_tetrahedra: 0,
            samples: qualification::Candidates::new(),
        })
    }
    pub fn observe(&mut self, ids: [r::ColliderHandle; 2], clods: &Clods) -> bool {
        let (a_id, b_id, a, am, parts, bm) = if let (Some((a, am)), Some((parts, bm))) =
            (clods.get(&ids[0]), self.models.get(&ids[1]))
        {
            (ids[0], ids[1], a, *am, parts, *bm)
        } else if let (Some((a, am)), Some((parts, bm))) =
            (clods.get(&ids[1]), self.models.get(&ids[0]))
        {
            (ids[1], ids[0], a, *am, parts, *bm)
        } else {
            return false;
        };
        self.admitted += 1;
        let start = Instant::now();
        let evaluated = pressure::evaluate(a, am, Opposing::RigidCompound(parts), bm);
        let elapsed = start.elapsed().as_secs_f64();
        self.seconds += elapsed;
        self.max_pair_seconds = self.max_pair_seconds.max(elapsed);
        match evaluated {
            Ok(result) => {
                if result.overlap_volume_m3 > 0. {
                    self.positive += 1;
                    qualification::remember(
                        &mut self.samples,
                        "largest_steel_force",
                        result.forces_n[0].length(),
                        [a_id, b_id],
                    );
                } else {
                    self.zero += 1;
                }
                self.potential += result.elastic_energy_j;
                self.volume += result.overlap_volume_m3;
                self.force = self.force.max(result.force_residual_n.length());
                self.moment = self.moment.max(result.moment_residual_nm.length());
                self.power = self.power.max(result.power_residual_w.abs());
                self.visits += result.union_intersection_visits;
                self.input_tetrahedra += result.union_input_tetrahedra;
                self.candidate_tetrahedra += result.union_candidate_tetrahedra;
                self.max_depth = self.max_depth.max(result.union_maximum_depth);
                self.energy_roundoff += result.union_energy_roundoff_estimate_j;
                self.volume_roundoff += result.union_volume_roundoff_estimate_m3;
            }
            Err(error) => {
                self.failures += 1;
                if self.examples.len() < 4 {
                    self.examples.push(serde_json::json!({"colliders":[a_id.into_raw_parts(),b_id.into_raw_parts()],"error":error,"clod":descriptor(a,am),"steel_parts":parts.iter().map(|p|serde_json::json!({"vertices":p.vertices(),"faces":p.faces(),"tets":p.reference_tetrahedra()})).collect::<Vec<_>>(),"steel_com":bm.center_of_mass_m,"steel_linear":bm.linear_velocity_m_s,"steel_angular":bm.angular_velocity_rad_s}));
                }
            }
        }
        if self.admitted.is_multiple_of(100) {
            println!(
                "FINITE_STEEL_PROGRESS pairs={} positive={} failures={} wall_seconds={}",
                self.admitted, self.positive, self.failures, self.seconds
            );
        }
        true
    }
    pub fn report(&self, clods: &Clods) -> serde_json::Value {
        let selected = qualification::selected(&self.samples, &["largest_steel_force"]);
        let qualification: Vec<_> = selected
            .into_iter()
            .map(|(name, ids, _)| {
                let (a, am) = &clods[&ids[0]];
                let (parts, bm) = &self.models[&ids[1]];
                let mut report =
                    qualification::check(name, a, Opposing::RigidCompound(parts), [*am, *bm]);
                report["colliders"] = serde_json::json!(ids.map(|id| id.into_raw_parts()));
                report
            })
            .collect();
        let passed = self.admitted > 0
            && self.failures == 0
            && self.positive + self.zero == self.admitted
            && self.force < 1e-5
            && self.moment < 1e-5
            && self.power < 1e-7
            && qualification.iter().all(|j| j["passed"] == true);
        let parts: usize = self.models.values().map(|(p, _)| p.len()).sum();
        let vertices: usize = self
            .models
            .values()
            .flat_map(|(p, _)| p)
            .map(|s| s.vertices().len())
            .sum();
        let faces: usize = self
            .models
            .values()
            .flat_map(|(p, _)| p)
            .map(|s| s.faces().len())
            .sum();
        let tets: usize = self
            .models
            .values()
            .flat_map(|(p, _)| p)
            .map(|s| s.tetrahedra())
            .sum();
        println!("FINITE_STEEL pairs={} positive={} zero={} failures={} seconds={} max_pair_seconds={} parts={} depth={}",self.admitted,self.positive,self.zero,self.failures,self.seconds,self.max_pair_seconds,parts,self.max_depth);
        serde_json::json!({"passed":passed,"rigid_bodies":self.models.len(),"registered_parts":parts,"vertices":vertices,"faces":faces,"tetrahedra":tets,"admitted_clod_steel_pairs":self.admitted,"positive_overlap_pairs":self.positive,"zero_overlap_pairs":self.zero,"failure_count":self.failures,"failures_first4":self.examples,"potential_j":self.potential,"overlap_m3":self.volume,"max_force_residual_n":self.force,"max_moment_residual_nm":self.moment,"max_power_residual_w":self.power,"cpu_wall_seconds":self.seconds,"max_pair_wall_seconds":self.max_pair_seconds,"subset_intersection_visits":self.visits,"input_rigid_tetrahedra":self.input_tetrahedra,"candidate_rigid_tetrahedra":self.candidate_tetrahedra,"maximum_subset_recursion_depth":self.max_depth,"energy_roundoff_estimate_j":self.energy_roundoff,"volume_roundoff_estimate_m3":self.volume_roundoff,"pose_and_partition_qualification":qualification,"clipping":"exact rational; positive tetrahedral integrals and compensated f64 moments","source_compound_regenerated":false})
    }
}

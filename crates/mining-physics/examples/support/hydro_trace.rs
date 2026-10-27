//! Bounded opt-in work/transport tracing for the actual gravity-column fixture.
use game::gpu::{flip_3d::GpuFlip3D, phase_transfer_3d::PhaseTransfer3D};
use glam::{DVec3, Mat3, Vec3};
use serde_json::{json, Value};
use std::{fs::File, io::Write, path::PathBuf};

pub struct Settings {
    root: Option<PathBuf>,
    every: usize,
    grid: Option<usize>,
    alpha: Option<f32>,
    pressure: Option<u32>,
    hz: Option<usize>,
    per_axis: Option<usize>,
}
impl Settings {
    pub fn from_args(args: &[String]) -> Self {
        let arg = |name: &str| args.iter().find_map(|v| v.strip_prefix(name));
        let root = arg("--trace=").map(PathBuf::from);
        if let Some(root) = &root {
            assert!(!root.exists(), "preserve previous trace");
            std::fs::create_dir_all(root).unwrap();
        }
        let every = arg("--trace-every=")
            .map(|s| s.parse().unwrap())
            .unwrap_or(24);
        assert!((1..=480).contains(&every));
        Self {
            root,
            every,
            grid: arg("--only-grid=").map(|s| s.parse().unwrap()),
            alpha: arg("--only-alpha=").map(|s| s.parse().unwrap()),
            pressure: arg("--only-pressure=").map(|s| s.parse().unwrap()),
            hz: arg("--only-hz=").map(|s| s.parse().unwrap()),
            per_axis: arg("--only-per-axis=").map(|s| s.parse().unwrap()),
        }
    }
    pub fn selected(
        &self,
        n: usize,
        alpha: f32,
        pressure: u32,
        hz: usize,
        per_axis: usize,
    ) -> bool {
        self.grid.is_none_or(|v| v == n)
            && self.alpha.is_none_or(|v| v == alpha)
            && self.pressure.is_none_or(|v| v == pressure)
            && self.hz.is_none_or(|v| v == hz)
            && self.per_axis.is_none_or(|v| v == per_axis)
    }
    pub fn metadata(&self) -> Value {
        json!({"grid":self.grid,"alpha":self.alpha,"pressure_iterations":self.pressure,
            "hz":self.hz,"per_axis":self.per_axis,"trace_directory":self.root,"sample_every_steps":self.every})
    }
    pub fn case(
        &self,
        name: &str,
        h: f32,
        per_axis: usize,
        metadata: impl FnOnce() -> Value,
    ) -> Option<Case> {
        self.root.as_ref().map(|root| {
            let root = root.join(name);
            std::fs::create_dir(&root).unwrap();
            let log = File::create_new(root.join("steps.jsonl")).unwrap();
            std::fs::write(
                root.join("fixture.json"),
                serde_json::to_vec_pretty(&metadata()).unwrap(),
            )
            .unwrap();
            Case {
                root,
                log,
                every: self.every,
                h,
                mass: 1000. * (h as f64 / per_axis as f64).powi(3),
                witness: false,
            }
        })
    }
}
pub struct Input {
    p: Vec<Vec3>,
    v: Vec<Vec3>,
    c: Vec<Mat3>,
}
pub struct Case {
    root: PathBuf,
    log: File,
    every: usize,
    h: f32,
    mass: f64,
    witness: bool,
}
fn budget(p: &[Vec3], v: &[Vec3], c: &[Mat3], mass: f64, h: f32) -> Value {
    let d = (h as f64).powi(2) / 4.;
    let mut kinetic = 0.;
    let mut affine = 0.;
    let mut potential = 0.;
    let mut momentum = DVec3::ZERO;
    let mut angular = DVec3::ZERO;
    for ((p, v), c) in p.iter().zip(v).zip(c) {
        let rows = [c.x_axis, c.y_axis, c.z_axis];
        let spin = DVec3::new(
            (rows[2].y - rows[1].z) as f64,
            (rows[0].z - rows[2].x) as f64,
            (rows[1].x - rows[0].y) as f64,
        ) * d;
        kinetic += 0.5 * mass * v.as_dvec3().length_squared();
        affine += 0.5
            * mass
            * d
            * rows
                .iter()
                .map(|r| r.as_dvec3().length_squared())
                .sum::<f64>();
        potential += mass * 9.81 * p.y as f64;
        momentum += v.as_dvec3() * mass;
        angular += (p.as_dvec3().cross(v.as_dvec3()) + spin) * mass;
    }
    json!({"kinetic_j":kinetic+affine,"point_kinetic_j":kinetic,"affine_kinetic_j":affine,"potential_j":potential,
        "momentum":momentum.to_array(),"angular":angular.to_array(),
        "max_speed_mps":v.iter().map(|v|v.length()).fold(0f32,f32::max),
        "max_affine_norm":c.iter().map(|c|[c.x_axis,c.y_axis,c.z_axis].iter().map(|r|r.length_squared()).sum::<f32>().sqrt()).fold(0f32,f32::max)})
}
fn read_words(d: &wgpu::Device, q: &wgpu::Queue, b: &wgpu::Buffer) -> Vec<u32> {
    let staging = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("phase moment trace"),
        size: b.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut e = d.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(b, 0, &staging, 0, b.size());
    q.submit(Some(e.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    d.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let result = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    result
}
impl Case {
    pub fn input(
        &self,
        step: usize,
        total: usize,
        p: &[Vec3],
        v: &[Vec3],
        c: &[Mat3],
    ) -> Option<Input> {
        (step.is_multiple_of(self.every) || step + 1 == total).then(|| Input {
            p: p.to_vec(),
            v: v.to_vec(),
            c: c.to_vec(),
        })
    }
    pub fn record(
        &mut self,
        step_time: (usize, f32),
        input: Input,
        state: (&[Vec3], &[Vec3], &[Mat3]),
        gather: &[Vec3],
        sim: &GpuFlip3D,
        gpu: (&wgpu::Device, &wgpu::Queue),
    ) {
        let (step, dt) = step_time;
        let (p, v, c) = state;
        let (device, queue) = gpu;
        let before = budget(&input.p, &input.v, &input.c, self.mass, self.h);
        let at_gather = budget(&input.p, gather, c, self.mass, self.h);
        let after = budget(p, v, c, self.mass, self.h);
        let grid = [
            sim.grid_u_buffer(),
            sim.grid_v_buffer(),
            sim.grid_w_buffer(),
        ]
        .map(|b| super::read_grid(device, queue, b));
        let water = sim
            .diagnostic_phase_water()
            .unwrap()
            .map(|b| super::read_grid(device, queue, b));
        let mobility = sim
            .diagnostic_phase_mobility()
            .unwrap()
            .map(|b| super::read_grid(device, queue, b));
        assert!(
            mobility.iter().all(|m| m[0] == 1.),
            "projection mobility was not ready"
        );
        let moments = read_words(device, queue, sim.diagnostic_phase_moments().unwrap());
        let sizes = [
            (sim.width + 1) * sim.height * sim.depth,
            sim.width * (sim.height + 1) * sim.depth,
            sim.width * sim.height * (sim.depth + 1),
        ]
        .map(|n| n as usize);
        let mut base = 0;
        let mut grid_old = 0.;
        let mut grid_predictor = 0.;
        let mut grid_projected = 0.;
        let mut mass_axes = [0.; 3];
        let unit_mass = 1000. * (self.h as f64).powi(3) / 8.;
        let mut faces: [Vec<PressureFace>; 3] = std::array::from_fn(|_| Vec::new());
        for axis in 0..3 {
            let aux = PhaseTransfer3D::aux_offset(sizes[axis] as u64) as usize;
            for face in 0..sizes[axis] {
                let moment = 1 + 8 * (base + face);
                let units = ((u64::from(moments[2 * moment + 1]) << 32)
                    | u64::from(moments[2 * moment])) as i64;
                let volume = units as f64 / 68_719_476_736.;
                let mass = volume * unit_mass;
                mass_axes[axis] += mass;
                faces[axis].push(PressureFace {
                    inverse_mobility: if mobility[axis][face + 1] > 0. {
                        1. / mobility[axis][face + 1] as f64
                    } else {
                        0.
                    },
                    predictor: water[axis][aux + 4 * face + 2] as f64,
                    projected: grid[axis][face] as f64,
                });
                grid_old += 0.5 * mass * (water[axis][face] as f64).powi(2);
                grid_predictor += 0.5 * mass * (water[axis][aux + 4 * face + 2] as f64).powi(2);
                grid_projected += 0.5 * mass * (grid[axis][face] as f64).powi(2);
            }
            base += sizes[axis];
        }
        let n = sim.width as usize;
        assert_eq!(sim.width, sim.height);
        assert_eq!(sim.width, sim.depth);
        let shape = [sim.width, sim.height, sim.depth];
        let markers = PhaseTransfer3D::cell_count_offset(shape) as usize;
        let volumes = PhaseTransfer3D::cell_volume_offset(shape) as usize;
        let mut max_div = 0f64;
        let mut max_density = 0f32;
        let mut excess = 0.;
        let mut occupied = 0;
        let mut total_volume = 0.;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let i = (z * n + y) * n + x;
                    let volume = water[0][volumes + i];
                    total_volume += volume as f64;
                    if water[0][markers + i].to_bits() & 0x7fff_ffff == 0 {
                        continue;
                    }
                    occupied += 1;
                    max_density = max_density.max(volume / 8.);
                    excess += (volume as f64 - 8.).max(0.);
                    let div = (grid[0][(z * n + y) * (n + 1) + x + 1] as f64
                        - grid[0][(z * n + y) * (n + 1) + x] as f64
                        + grid[1][(z * (n + 1) + y + 1) * n + x] as f64
                        - grid[1][(z * (n + 1) + y) * n + x] as f64
                        + grid[2][((z + 1) * n + y) * n + x] as f64
                        - grid[2][(z * n + y) * n + x] as f64)
                        / self.h as f64;
                    max_div = max_div.max(div.abs());
                }
            }
        }
        let cycles = pressure_cycles(
            n,
            self.h,
            dt,
            &faces,
            &water[0][markers..markers + n * n * n],
        );
        let pressure=sim.measure_pressure_residual(device,queue).map(|r|json!({"rms":r.rms,"relative_l2":r.relative_l2,"max_abs":r.max_abs,"fluid_cells":r.fluid_cells}));
        let b = before["kinetic_j"].as_f64().unwrap();
        let g = at_gather["kinetic_j"].as_f64().unwrap();
        let a = after["kinetic_j"].as_f64().unwrap();
        let row = json!({"step":step,"physical_time_s":(step+1) as f64*dt as f64,"before":before,"gather":at_gather,"after":after,
            "grid_old_kinetic_j":grid_old,"grid_predictor_kinetic_j":grid_predictor,"grid_projected_kinetic_j":grid_projected,
            "gather_energy_excess_over_grid_change_j":g-b-(grid_projected-grid_old),"collision_kinetic_change_j":a-g,
            "mechanical_energy_change_j":a-b+after["potential_j"].as_f64().unwrap()-before["potential_j"].as_f64().unwrap(),
            "pressure_cycles":cycles,"mass_per_axis_kg":mass_axes,"occupied_cells":occupied,"max_occupied_grid_divergence_sinv":max_div,
            "max_occupied_relative_density":max_density,"positive_excess_cell_volume_units":excess,"total_cell_volume_units":total_volume,"pressure_residual":pressure});
        writeln!(self.log, "{row}").unwrap();
        self.log.flush().unwrap();
        if !self.witness && row["after"]["max_speed_mps"].as_f64().unwrap() > 0.001 {
            let witness = json!({"step":step,"positions":input.p.iter().map(|p|p.to_array()).collect::<Vec<_>>(),"velocities":input.v.iter().map(|v|v.to_array()).collect::<Vec<_>>(),"affine_rows":input.c.iter().map(|c|c.to_cols_array_2d()).collect::<Vec<_>>(),"metrics":row});
            std::fs::write(
                self.root.join("first-failed-sample-before.json"),
                serde_json::to_vec(&witness).unwrap(),
            )
            .unwrap();
            self.witness = true;
        }
    }
}

// A scalar pressure gradient has zero circulation around every cell-centre
// loop. For u = u* - k grad(p), a nonzero circulation of u*/k therefore gives
// a lower bound on the largest unavoidable face velocity, independent of the
// pressure solver's iteration count. Pure water has k = R/V_face here.
#[derive(Clone, Copy)]
struct PressureFace {
    inverse_mobility: f64,
    predictor: f64,
    projected: f64,
}
fn cycle_metric(faces: [PressureFace; 4], axis: [usize; 2], h: f64, dt: f64) -> Value {
    let signs = [1., 1., -1., -1.];
    let sum = faces.iter().map(|f| f.inverse_mobility).sum::<f64>();
    let circulation = faces
        .iter()
        .zip(signs)
        .map(|(f, s)| s * f.predictor * f.inverse_mobility)
        .sum::<f64>()
        * h;
    let gravity = faces
        .iter()
        .zip(signs)
        .enumerate()
        .map(|(i, (f, s))| {
            if axis[i % 2] == 1 {
                -9.81 * dt * s * f.inverse_mobility
            } else {
                0.
            }
        })
        .sum::<f64>()
        * h;
    let bound = circulation.abs() / (h * sum);
    let observed = faces.iter().map(|f| f.projected.abs()).fold(0f64, f64::max);
    json!({"predictor_circulation_m2ps":circulation,"gravity_only_circulation_m2ps":gravity,
        "unavoidable_face_speed_mps":bound,"gravity_only_rest_speed_bound_mps":gravity.abs()/(h*sum),
        "observed_max_face_speed_mps":observed,"bound_margin_mps":observed-bound,
        "inverse_mobility":faces.map(|f|f.inverse_mobility),"predictor_mps":faces.map(|f|f.predictor),"projected_mps":faces.map(|f|f.projected)})
}
fn pressure_cycles(
    n: usize,
    h: f32,
    dt: f32,
    faces: &[Vec<PressureFace>; 3],
    markers: &[f32],
) -> Value {
    let cell = |p: [usize; 3]| (p[2] * n + p[1]) * n + p[0];
    let solid = |p: [usize; 3]| markers[cell(p)].to_bits() & 0x8000_0000 != 0;
    let occupied = |p: [usize; 3]| markers[cell(p)].to_bits() & 0x7fff_ffff > 0;
    let face = |axis: usize, mut p: [usize; 3]| {
        p[axis] += 1;
        let mut dims = [n; 3];
        dims[axis] += 1;
        faces[axis][(p[2] * dims[1] + p[1]) * dims[0] + p[0]]
    };
    let mut checked = 0;
    let mut worst = Value::Null;
    let mut worst_gravity = Value::Null;
    let mut maximum_bound = -1.;
    let mut maximum_gravity = -1.;
    let mut minimum_margin = f64::INFINITY;
    for z in 1..n - 2 {
        for y in 4..n - 2 {
            for x in 1..n - 2 {
                for axes in [[0usize, 1usize], [0, 2], [1, 2]] {
                    let a = [x, y, z];
                    let mut b = a;
                    b[axes[0]] += 1;
                    let mut c = b;
                    c[axes[1]] += 1;
                    let mut d = a;
                    d[axes[1]] += 1;
                    // All loop cells must be real occupied pressure unknowns. Avoid
                    // the existing floor-slip band and internal-floor tangent damping;
                    // the stored mixture predictor then equals the solve's actual RHS.
                    if [a, b, c, d]
                        .into_iter()
                        .any(|p| !occupied(p) || solid(p) || solid([p[0], p[1] - 1, p[2]]))
                    {
                        continue;
                    }
                    let values = [
                        face(axes[0], a),
                        face(axes[1], b),
                        face(axes[0], d),
                        face(axes[1], a),
                    ];
                    if values.iter().any(|f| f.inverse_mobility <= 0.) {
                        continue;
                    }
                    let mut metric = cycle_metric(values, axes, h as f64, dt as f64);
                    metric["origin_cell"] = json!(a);
                    metric["axes"] = json!(axes);
                    let bound = metric["unavoidable_face_speed_mps"].as_f64().unwrap();
                    let gravity = metric["gravity_only_rest_speed_bound_mps"]
                        .as_f64()
                        .unwrap();
                    minimum_margin =
                        minimum_margin.min(metric["bound_margin_mps"].as_f64().unwrap());
                    checked += 1;
                    if bound > maximum_bound {
                        maximum_bound = bound;
                        worst = metric.clone();
                    }
                    if gravity > maximum_gravity {
                        maximum_gravity = gravity;
                        worst_gravity = metric;
                    }
                }
            }
        }
    }
    // The finite MAC pressure differences cancel algebraically. Float storage
    // may leave a small closure residue; a larger violation means this trace
    // selected a face changed by an unaccounted operator, not physical success.
    assert!(checked>0 && minimum_margin>= -1e-7,"pressure-cycle witness does not bound the actual projected field: count={checked}, margin={minimum_margin}");
    json!({"checked_cycles":checked,"minimum_velocity_bound_margin_mps":minimum_margin,
        "strongest_predictor_cycle":worst,"strongest_gravity_cycle":worst_gravity})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pressure_cycle_bound_is_gauge_independent_and_attainable() {
        let inverse = [1., 2., 1., 1.];
        let predictor = [0., -0.1, 0., -0.1];
        // A pressure field which distributes the unavoidable residual equally.
        // Its four directed differences sum to zero. Reversing one sign or
        // omitting a leg breaks that independent closed-loop construction.
        let pressure = [0., 0.02, -0.14, -0.12];
        let difference = [
            pressure[1] - pressure[0],
            pressure[2] - pressure[1],
            pressure[2] - pressure[3],
            pressure[3] - pressure[0],
        ];
        for gauge in [0., 10., -4.] {
            let faces = std::array::from_fn(|i| PressureFace {
                inverse_mobility: inverse[i],
                predictor: predictor[i],
                projected: predictor[i] - ((difference[i] + gauge) - gauge) / inverse[i],
            });
            let v = cycle_metric(faces, [0, 1], 1., 0.01);
            assert!((v["predictor_circulation_m2ps"].as_f64().unwrap() + 0.1).abs() < 1e-12);
            assert!((v["unavoidable_face_speed_mps"].as_f64().unwrap() - 0.02).abs() < 1e-12);
            assert!((v["observed_max_face_speed_mps"].as_f64().unwrap() - 0.02).abs() < 1e-12);
        }
    }
    #[test]
    fn uniform_mass_gravity_has_no_circulation() {
        let faces = std::array::from_fn(|i| PressureFace {
            inverse_mobility: 1.,
            predictor: if i % 2 == 1 { -0.1 } else { 0. },
            projected: 0.,
        });
        let v = cycle_metric(faces, [0, 1], 0.01, 0.01);
        assert_eq!(v["predictor_circulation_m2ps"], 0.);
        assert_eq!(v["gravity_only_rest_speed_bound_mps"], 0.);
    }
}

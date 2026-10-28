//! Evolving water-only strip benchmark. No game/render/mineral speedup claim.
use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use mining_sim::sluice::Sluice;
use serde_json::{json, Value};
use std::time::Instant;
const DT: f32 = 1. / 240.;
const WARMUP: usize = 64;
const TICKS: usize = 16;
struct Workload {
    depth: usize,
    solver: GpuFlip3D,
    p: Vec<Vec3>,
    v: Vec<Vec3>,
    c: Vec<Mat3>,
    initial: Vec<Vec3>,
    rho: Vec<f32>,
    cells: Vec<u32>,
    sdf: Vec<f32>,
    h: f32,
    lip: f32,
    domain: Vec3,
    resident: bool,
    retired: usize,
}
impl Workload {
    fn reset(&mut self) {
        self.p.clone_from(&self.initial);
        self.v.resize(self.initial.len(), Vec3::ZERO);
        self.c.resize(self.initial.len(), Mat3::ZERO);
        self.rho.resize(self.initial.len(), 1.);
        self.v.fill(Vec3::X * 0.4);
        self.c.fill(Mat3::ZERO);
        self.retired = 0;
    }
    fn tick(&mut self, d: &wgpu::Device, q: &wgpu::Queue) {
        self.solver.step(
            d,
            q,
            &mut self.p,
            &mut self.v,
            &mut self.c,
            &self.rho,
            &self.cells,
            Some(&self.sdf),
            None,
            DT,
            -9.81,
            0.,
            80,
        );
    }
    fn frame(&mut self, d: &wgpu::Device, q: &wgpu::Queue) {
        if self.resident {
            self.solver.step_no_readback(
                d,
                q,
                &mut self.p,
                &mut self.v,
                &mut self.c,
                &self.rho,
                &self.cells,
                Some(&self.sdf),
                None,
                DT,
                -9.81,
                0.,
                80,
            );
            for _ in 1..4 {
                self.solver.step_in_place(
                    d,
                    q,
                    self.p.len() as u32,
                    &self.cells,
                    Some(&self.sdf),
                    None,
                    DT,
                    -9.81,
                    0.,
                    80,
                );
            }
            assert!(self.solver.request_readback(d, q, self.p.len()));
            d.poll(wgpu::Maintain::Wait);
            assert_eq!(
                self.solver
                    .try_readback(d, &mut self.p, &mut self.v, &mut self.c),
                Some(self.rho.len())
            );
        } else {
            for _ in 0..4 {
                self.tick(d, q);
            }
        }
        // Preserve finite pulse ownership when water leaves the outer domain.
        // Both cadences retire at the same four-substep observation boundary.
        for i in (0..self.p.len()).rev() {
            let p = self.p[i];
            if p.x < 0. || p.x >= self.domain.x || p.y < 0. || p.y >= self.domain.y {
                self.p.swap_remove(i);
                self.v.swap_remove(i);
                self.c.swap_remove(i);
                self.rho.swap_remove(i);
                self.retired += 1;
            }
        }
        assert_eq!(self.p.len() + self.retired, self.initial.len());
    }
    fn fingerprint(&self) -> String {
        // Non-cryptographic exact-bit diagnostic; full array equality is tested
        // separately. Never treat this fingerprint as a conservation proof.
        let mut hash = 0xcbf29ce484222325u64;
        for i in 0..self.p.len() {
            for value in self.p[i]
                .to_array()
                .into_iter()
                .chain(self.v[i].to_array())
                .chain(self.c[i].to_cols_array())
            {
                for byte in value.to_bits().to_le_bytes() {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
                }
            }
        }
        format!("{hash:016x}")
    }
    fn sample(&mut self, d: &wgpu::Device, q: &wgpu::Queue) -> Value {
        self.reset();
        for _ in 0..WARMUP / 4 {
            self.frame(d, q);
        }
        let start = Instant::now();
        for _ in 0..TICKS / 4 {
            self.frame(d, q);
        }
        let ms = start.elapsed().as_secs_f64() * 1000. / TICKS as f64;
        assert!(self
            .p
            .iter()
            .all(|p| p.is_finite() && p.z >= 0. && p.z < self.depth as f32 * self.h));
        assert!(self.v.iter().all(|v| v.is_finite()) && self.c.iter().all(|c| c.is_finite()));
        let mean =
            |f: fn(Vec3) -> f64| self.v.iter().map(|v| f(*v)).sum::<f64>() / self.v.len() as f64;
        json!({"depth_cm":self.depth,"particles":self.initial.len(),"live_particles":self.p.len(),"retired_particles":self.retired,"accounted_particles":self.p.len()+self.retired,"state_fnv1a64":self.fingerprint(),"step_ms":ms,"water_only_physical_seconds_per_wall_second":f64::from(DT)*1000./ms,"mean_downstream_ms":mean(|v|f64::from(v.x)),"mean_vertical_ms":mean(|v|f64::from(v.y)),"backflow_fraction":self.v.iter().filter(|v|v.x< -0.05).count() as f64/self.v.len() as f64,"outlet_particles_per_cm":(self.p.iter().filter(|p|p.x>=self.lip).count()+self.retired) as f64/self.depth as f64,"nominal_marker_volume_m3":self.initial.len() as f64*f64::from(self.h).powi(3)/8.,"mean_y_m":self.p.iter().map(|p|f64::from(p.y)).sum::<f64>()/self.p.len() as f64,"max_speed_ms":self.v.iter().map(|v|v.length()).fold(0f32,f32::max),"finite":true})
    }
}
fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) * 0.5
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output = args.get(1).expect("output JSON path");
    let resident = args.iter().any(|arg| arg == "--resident");
    let cycles: usize = args.get(2).map_or(6, |v| v.parse().unwrap());
    assert!((2..=24).contains(&cycles));
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("real GPU");
    let name = adapter.get_info().name;
    let (d, q) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    let sluice = Sluice::default();
    let config = mining_physics::geometry::config(&sluice);
    let nx = config.grid_width + sluice.profile.outlet_buffer();
    let ny = config.grid_height;
    let h = config.cell_size;
    // Use the actual reference centre-section floor/riffles/outlet. A depth-one
    // distance solve extrudes that 2D geometry without inserting lateral walls.
    let mut section = sim3d::Grid3D::new(nx, ny, 1, h);
    for y in 0..ny {
        for x in 0..config.grid_width {
            if config.is_solid(x, y, config.grid_depth / 2) {
                section.set_solid(x, y, 0);
            }
        }
    }
    section.compute_sdf();
    let plane_sdf = section.sdf().to_vec();
    let plane_cells: Vec<u32> = section
        .solid()
        .iter()
        .map(|v| if *v { 2 } else { 0 })
        .collect();
    let mut work = Vec::new();
    for depth in [38, 12, 8] {
        let mut p = Vec::new();
        for z in 0..depth {
            for x in 2..config.grid_width {
                let floor = config.floor_height_at(x);
                for y in floor + 1..=(floor + 4).min(ny - 1) {
                    if plane_cells[y * nx + x] == 2 {
                        continue;
                    }
                    for a in [0.25, 0.75] {
                        for b in [0.25, 0.75] {
                            for c in [0.25, 0.75] {
                                p.push(Vec3::new(x as f32 + a, y as f32 + b, z as f32 + c) * h);
                            }
                        }
                    }
                }
            }
        }
        let mut solver =
            GpuFlip3D::new_periodic_z(&d, nx as u32, ny as u32, depth as u32, h, p.len());
        solver.open_boundaries = 14;
        let count = p.len();
        work.push(Workload {
            depth,
            solver,
            p: p.clone(),
            initial: p,
            v: vec![Vec3::X * 0.4; count],
            c: vec![Mat3::ZERO; count],
            rho: vec![1.; count],
            cells: plane_cells.repeat(depth),
            sdf: plane_sdf.repeat(depth),
            h,
            lip: config.grid_width as f32 * h,
            domain: Vec3::new(nx as f32 * h, ny as f32 * h, depth as f32 * h),
            resident,
            retired: 0,
        });
    }
    // Warm every compiled workload before collecting paired forward/reverse blocks.
    for index in [0, 1, 2, 2, 1, 0] {
        let _ = work[index].sample(&d, &q);
    }
    let mut samples = Vec::new();
    let mut paired = Vec::new();
    for cycle in 0..cycles {
        let mut rows = Vec::new();
        for index in [0, 1, 2, 2, 1, 0] {
            let row = work[index].sample(&d, &q);
            rows.push(row.clone());
            samples.push(json!({"cycle":cycle,"order":samples.len()%6,"sample":row}));
        }
        let reference =
            (rows[0]["step_ms"].as_f64().unwrap() + rows[5]["step_ms"].as_f64().unwrap()) * 0.5;
        paired.push(json!({"cycle":cycle,"12_cm_factor":reference/((rows[1]["step_ms"].as_f64().unwrap()+rows[4]["step_ms"].as_f64().unwrap())*0.5),"8_cm_factor":reference/((rows[2]["step_ms"].as_f64().unwrap()+rows[3]["step_ms"].as_f64().unwrap())*0.5)}));
        println!("cycle {cycle} complete");
    }
    let by_depth:Vec<_>=[38,12,8].into_iter().map(|depth|{let rows:Vec<_>=samples.iter().filter(|v|v["sample"]["depth_cm"]==depth).map(|v|&v["sample"]).collect();let ms:Vec<f64>=rows.iter().map(|v|v["step_ms"].as_f64().unwrap()).collect();json!({"depth_cm":depth,"samples":ms.len(),"median_step_ms":median(&ms),"water_only_physical_ratio":f64::from(DT)*1000./median(&ms),"particles":rows[0]["particles"],"first_state_statistics":rows[0],"last_state_statistics":rows[rows.len()-1]})}).collect();
    let result = json!({"scope":"finite evolving water pulse; no Session, minerals, feed, world or rendering","resident_four_substeps":resident,"readback":"full water/affine state once per four substeps when resident; each substep otherwise","outflow_ownership":"retired to finite count ledger at each four-substep boundary; no marker creation","adapter":name,"grid_xy":[nx,ny],"cell_m":h,"dt":DT,"pressure_sweeps":80,"density_sweeps":40,"warmup_ticks_per_block":WARMUP,"timed_ticks_per_block":TICKS,"cycles":cycles,"width_order_cm":[38,12,8,8,12,38],"geometry":"extruded actual reference centre-section signed distance; no lateral walls","full_width_consequences_qualified":false,"gameplay_speedup_measured":false,"by_depth":by_depth,"paired_cycle_factors":paired,"samples":samples});
    std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    println!("{}", result["by_depth"]);
}

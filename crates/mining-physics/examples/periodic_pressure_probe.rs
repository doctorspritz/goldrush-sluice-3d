//! Projection-only width benchmark. No particle/solid dynamics or gameplay
//! speedup is implied. Supply an output JSON path and optional sample count.
use game::gpu::{
    pressure_3d::GpuPressure3D,
    timing::{GpuStageTimer, FEATURES},
};
use mining_sim::sluice::Sluice;
use std::{collections::BTreeMap, time::Instant};
use wgpu::util::DeviceExt;

struct Workload {
    depth: usize,
    fluid_cells: usize,
    initial: [Vec<f32>; 3],
    buffers: Vec<wgpu::Buffer>,
    solver: GpuPressure3D,
    stages: BTreeMap<String, f64>,
    gpu_samples: Vec<f64>,
    host_samples: Vec<f64>,
}
impl Workload {
    fn project(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        timer: &mut GpuStageTimer,
        record: bool,
    ) {
        let start = Instant::now();
        for (buffer, values) in self.buffers.iter().zip(&self.initial) {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(values));
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        self.solver.clear_pressure(&mut encoder);
        self.solver.encode_profiled(&mut encoder, 80, Some(timer));
        timer.resolve(&mut encoder);
        queue.submit(Some(encoder.finish()));
        timer.begin_map();
        let elapsed = timer.collect(device).unwrap();
        if record {
            self.host_samples
                .push(start.elapsed().as_secs_f64() * 1000.);
            self.gpu_samples
                .push(elapsed.iter().map(|(_, ms)| ms).sum());
            for (label, ms) in elapsed {
                *self.stages.entry(label.into()).or_default() += ms;
            }
        }
    }
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) * 0.5
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let output = args.get(1).expect("output JSON path");
    let samples: usize = args.get(2).map_or(12, |s| s.parse().unwrap());
    assert!((1..=60).contains(&samples));
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("actual GPU");
    assert!(
        adapter.features().contains(FEATURES),
        "actual device timestamps required"
    );
    let name = adapter.get_info().name;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: FEATURES,
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    let s = Sluice::default();
    let config = mining_physics::geometry::config(&s);
    let nx = config.grid_width + s.profile.outlet_buffer();
    let ny = config.grid_height;
    let mut workloads = Vec::new();
    for depth in [38, 12, 8] {
        // Repeat the actual reference channel centre section: floor, riffles,
        // inlet wall and exit buffer. There are no lateral walls in this model.
        let cells: Vec<u32> = (0..nx * ny * depth)
            .map(|i| {
                let x = i % nx;
                let y = (i / nx) % ny;
                if x < config.grid_width && config.is_solid(x, y, config.grid_depth / 2) {
                    2
                } else if y <= config.floor_height_at(x.min(config.grid_width - 1)) + 4 {
                    1
                } else {
                    0
                }
            })
            .collect();
        let sizes = [
            (nx + 1) * ny * depth,
            nx * (ny + 1) * depth,
            nx * ny * (depth + 1),
        ];
        let mut initial: [Vec<f32>; 3] = std::array::from_fn(|axis| vec![0.; sizes[axis]]);
        for z in 0..depth {
            for y in 0..ny {
                for x in 0..=nx {
                    let left = if x > 0 {
                        cells[(z * ny + y) * nx + x - 1]
                    } else {
                        2
                    };
                    let right = if x < nx {
                        cells[(z * ny + y) * nx + x]
                    } else {
                        0
                    };
                    if left != 2 && right != 2 && (left == 1 || right == 1) {
                        initial[0][(z * ny + y) * (nx + 1) + x] =
                            0.4 + 0.02 * (x as f32 * 0.17).sin();
                    }
                }
            }
        }
        let buffers: Vec<_> = initial
            .iter()
            .map(|v| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("periodic width projection"),
                    contents: bytemuck::cast_slice(v),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                })
            })
            .collect();
        let solver = GpuPressure3D::new_periodic_z(
            &device,
            nx as u32,
            ny as u32,
            depth as u32,
            &buffers[0],
            &buffers[1],
            &buffers[2],
            None,
        )
        .unwrap();
        solver.upload_cell_types(&queue, &cells, config.cell_size, 10);
        workloads.push(Workload {
            depth,
            fluid_cells: cells.iter().filter(|&&c| c == 1).count(),
            initial,
            buffers,
            solver,
            stages: BTreeMap::new(),
            gpu_samples: Vec::new(),
            host_samples: Vec::new(),
        });
    }
    // Construct all pipelines first. Alternate widths in forward/reverse order
    // after a common warmup, rather than comparing separately warmed blocks.
    let order = [0usize, 1, 2, 2, 1, 0];
    let warmup_cycles = 8;
    let mut timer = GpuStageTimer::new(&device, &queue);
    for cycle in 0..samples + warmup_cycles {
        for &index in &order {
            workloads[index].project(&device, &queue, &mut timer, cycle >= warmup_cycles);
        }
    }
    let mut results = Vec::new();
    for workload in workloads {
        let residual = workload.solver.measure_residual(&device, &queue).unwrap();
        let count = workload.gpu_samples.len();
        let stages: BTreeMap<_, _> = workload
            .stages
            .iter()
            .map(|(label, ms)| (label, ms / count as f64))
            .collect();
        results.push(serde_json::json!({"shape":[nx,ny,workload.depth],"physical_width_m":workload.depth as f64*config.cell_size as f64,"fluid_cells":workload.fluid_cells,"samples":count,"warmup":warmup_cycles*2,"sweeps":80,"gpu_stage_ms":stages,"gpu_total_ms":workload.gpu_samples.iter().sum::<f64>()/count as f64,"gpu_median_ms":median(&workload.gpu_samples),"gpu_samples_ms":workload.gpu_samples,"host_upload_submit_map_median_ms":median(&workload.host_samples),"relative_pressure_residual":residual.relative_l2,"max_backward_error":residual.max_backward_error}));
        println!("{}", results.last().unwrap());
    }
    let report = serde_json::json!({"adapter":name,"width_order_per_cycle":[38,12,8,8,12,38],"warmup_cycles":warmup_cycles,"measured_cycles":samples,"projection_only":true,"complete_strip_water_solver":false,"gameplay_speedup_measured":false,"geometry":"Repeated reference channel centre section with a four-cell water layer; not an evolved free surface.","timing_scope":"Device passes include divergence,80sweeps and gradient; copies/host waits are excluded from device totals. Host upload/submit/map time reported separately.","results":results});
    std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}

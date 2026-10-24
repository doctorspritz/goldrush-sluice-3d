//! Explicit real-GPU controls for the production river/channel optical helpers.
//! Reference light transport is calculated independently in f64 for known
//! homogeneous columns, across camera angles, depths and mineral cross sections.
use wgpu::util::DeviceExt;

fn probe(inputs: &[[f32; 4]]) -> Vec<[f32; 4]> {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("real GPU");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("river optical controls"),
        size: inputs.len() as u64 * 3 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("independent river columns"),
        contents: bytemuck::cast_slice(inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[],
    });
    let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &empty,
        entries: &[],
    });
    let entries = [false, true].map(|read_only| wgpu::BindGroupLayoutEntry {
        binding: u32::from(read_only),
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input.as_entire_binding(),
            },
        ],
    });
    let source = format!(
        "{}\n{}\n@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {{\n\
         let i=p.x;if i>={}u {{return;}}\n\
         let value=inputs[i];let path=eye_ray_path(value.y,value.z);\n\
         let beta=river_extinction(1.+value.x);\n\
         let transmission=exp(-suspended_extinction(beta)*path);\n\
         let source=suspended_source(beta);\n\
         result[i*3u]=vec4(transmission,path);\n\
         result[i*3u+1u]=vec4(source,beta);\n\
         result[i*3u+2u]=vec4(vec3(0.35,0.27,0.19)*transmission+source*(vec3(1.)-transmission),1.);\n\
         }}",
        include_str!("../../mining/src/water_compose.wgsl"),
        "@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;\n\
         @group(1) @binding(1) var<storage,read> inputs:array<vec4<f32>>;",
        inputs.len(),
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production river optical helpers"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&empty, &layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: output.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &empty_group, &[]);
        pass.set_bind_group(1, &group, &[]);
        pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let values = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec();
    readback.unmap();
    values
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn river_and_channel_columns_match_analytic_angle_and_light_transport() {
    let mut inputs = Vec::new();
    for beta in [
        0., 0.0001, 0.01, 1., 40., 119.9, 120., 120.1, 600., 3000., 30000.,
    ] {
        for path in [0., 0.0001, 0.003, 0.04, 0.3, 0.5] {
            for cosine in [1., 0.8, 0.3, 0.05] {
                inputs.push([beta, path * cosine, cosine, path]);
            }
        }
    }
    let values = probe(&inputs);
    let water_coefficients = [0.8, 0.24, 0.12];
    let mineral_coefficients = [0.5625, 1., 1.5];
    let water_source = [0.035, 0.18, 0.18];
    let mineral_source = [0.22, 0.16, 0.095];
    let background = [0.35, 0.27, 0.19];
    let reduced_fraction = 0.05 + 0.95 * 0.15;
    let mut old_angle_failures = 0;
    let mut old_source_failures = 0;
    for (index, input) in inputs.iter().enumerate() {
        let [beta, view_gap, _, known_path] = input.map(f64::from);
        let actual = &values[index * 3..index * 3 + 3];
        assert!(actual.iter().flatten().all(|v| v.is_finite() && *v >= 0.));
        assert!((f64::from(actual[0][3]) - known_path).abs() < 2e-6);
        assert!((f64::from(actual[1][3]) - beta).abs() < 1e-5 + beta * 2e-7);
        if beta == 0. {
            assert_eq!(
                actual[1][3], 0.,
                "clear-water marker must remain exactly clear"
            );
        }
        for (channel, &water) in water_coefficients.iter().enumerate() {
            let mineral = mineral_coefficients[channel] * beta * reduced_fraction;
            let extinction = water + mineral;
            let expected_transmission = (-extinction * known_path).exp();
            let expected_source =
                (water * water_source[channel] + mineral * mineral_source[channel]) / extinction
                    * 0.8;
            // Independent homogeneous radiative-transfer solution at the far
            // boundary. Holding physical path fixed must be camera-invariant.
            let expected_light = background[channel] * expected_transmission
                + expected_source * (1. - expected_transmission);
            for (actual, expected) in [
                (actual[0][channel], expected_transmission),
                (actual[1][channel], expected_source),
                (actual[2][channel], expected_light),
            ] {
                assert!(
                    (f64::from(actual) - expected).abs() < 1e-5,
                    "case {index}, component {channel}: GPU {actual}, independent {expected}"
                );
            }
            let old_transmission = (-extinction * view_gap).exp();
            old_angle_failures +=
                usize::from((old_transmission - expected_transmission).abs() > 0.01);
            let old_source = water_source[channel] * (1. - (beta / 120.).clamp(0., 1.))
                + [0.22, 0.13, 0.055][channel] * (beta / 120.).clamp(0., 1.);
            old_source_failures += usize::from((old_source - expected_source).abs() > 0.01);
        }
    }
    assert!(
        old_angle_failures > 100,
        "angles must distinguish the old camera-Z path"
    );
    assert!(
        old_source_failures > 100,
        "mixtures must distinguish the old colour ramp"
    );
    println!("river optics: {} independent columns; old camera-Z/source controls fail {} / {} components",
        inputs.len(), old_angle_failures, old_source_failures);
}

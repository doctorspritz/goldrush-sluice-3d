//! Real-GPU checks of the production visible-kernel integral. The independent
//! reference integrates the scalar density numerically in f64, rather than
//! reusing the shader's antiderivative or assuming uniform ray concentration.
use wgpu::util::DeviceExt;

fn device() -> (wgpu::Device, wgpu::Queue) {
    let adapter =
        pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
            .expect("real GPU");
    pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap()
}

fn probe(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    count: usize,
    inputs: &[[f32; 4]],
    body: &str,
) -> Vec<[f32; 4]> {
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("visible water column results"),
        size: count as u64 * 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("visible water column inputs"),
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
    // This is the previous production helper and its final uniform-rescale
    // expression, retained verbatim as an executable GPU negative control.
    // It is intentionally independent of the replacement compositor pipeline.
    let legacy = r#"
fn previous_visible_column_path(column:f32,view_gap:f32,channel:bool)->f32 {
    return min(max(view_gap,0.),select(max(column,0.0002),max(column,0.),channel));
}
fn previous_uniform_rescale(column:vec2<f32>,view_gap:f32)->vec2<f32> {
    let visible=previous_visible_column_path(column.x,view_gap,true);
    return column*(visible/max(column.x,0.000001));
}
"#;
    let source = format!(
        "{}\n{}\n@group(1) @binding(0) var<storage,read_write> result:array<vec4<f32>>;\n\
         @group(1) @binding(1) var<storage,read> inputs:array<vec4<f32>>;\n\
         @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) p:vec3<u32>) {{\n\
         let i=p.x;if i>={}u {{return;}}\n{}\n}}",
        include_str!("../../mining/src/water_depth.wgsl"),
        legacy,
        count,
        body
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production visible water column"),
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
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
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

/// Integrate density directly over the requested eye-ray interval. Keeping
/// zero support in the integrand makes this independent of the GPU's clipping
/// and polynomial primitive; midpoint convergence controls the reference error.
fn quadrature(s: f64, a: f64, closest: f64, front: f64, back: f64, scale: f64) -> f64 {
    if back <= front {
        return 0.;
    }
    let steps = 16_384;
    let step = (back - front) / steps as f64;
    (0..steps)
        .map(|i| {
            let t = front + (i as f64 + 0.5) * step;
            let support = (s - a * (t - closest).powi(2)).max(0.);
            6.5625 * scale * support.powi(3)
        })
        .sum::<f64>()
        * step
}

fn close(actual: f32, expected: f64, context: &str) {
    assert!(actual.is_finite() && actual >= 0., "{context}: {actual}");
    let error = (actual as f64 - expected).abs();
    assert!(
        error <= 0.000_002 + expected.abs() * 0.000_04,
        "{context}: GPU {actual}, independent integral {expected}, error {error}"
    );
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn visible_columns_match_quadrature_through_front_back_and_empty_intervals() {
    let (device, queue) = device();
    let cases: Vec<([f32; 4], [f32; 4])> = [0.15_f32, 0.55, 1.]
        .into_iter()
        .flat_map(|s| {
            [0.004_f32, 0.017].into_iter().flat_map(move |radius| {
                [
                    (1.95, 2.05),
                    (1.999, 2.05),
                    (2.002, 2.008),
                    (1.95, 1.999),
                    (2.05, 2.06),
                    (1.94, 1.95),
                    (2., 2.),
                    (2.01, 1.99),
                ]
                .into_iter()
                .map(move |(front, back)| {
                    ([s, 1. / (radius * radius), 2., front], [back, 0.21, 0., 0.])
                })
            })
        })
        .collect();
    let inputs: Vec<_> = cases.iter().flat_map(|(a, b)| [*a, *b]).collect();
    let values = probe(
        &device,
        &queue,
        cases.len(),
        &inputs,
        r#"
let v=inputs[i*2u];let parameters=inputs[i*2u+1u];
let visible=visible_kernel_column(v.x,v.y,v.z,v.w,parameters.x,parameters.y);
result[i]=vec4(visible,0.,0.,0.);
"#,
    );
    for (index, ((v, p), result)) in cases.iter().zip(values).enumerate() {
        let expected = quadrature(
            v[0] as f64,
            v[1] as f64,
            v[2] as f64,
            v[3] as f64,
            p[0] as f64,
            p[1] as f64,
        );
        close(result[0], expected, &format!("clipped interval {index}"));
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn stratified_suspension_removes_hidden_mud_instead_of_rescaling_its_colour() {
    let (device, queue) = device();
    // The two disjoint kernels form front and rear optical layers. The joined
    // interface is behind the muddy foreground kernel but inside the rear one.
    // Reverse their colours too; a correction must not merely remove dirt.
    let beta = [
        [1000_f32, 0., 0., 0.],
        [0., 1000., 0., 0.],
        [321., 321., 0., 0.],
    ];
    let values = probe(
        &device,
        &queue,
        beta.len(),
        &beta,
        r#"
let front=2.04;let back=2.05;let beta=inputs[i].xy;
let a=1./(0.01*0.01);let b=1./(0.015*0.015);
let before=vec2(visible_kernel_column(1.,a,2.,1.98,back,0.15),
                visible_kernel_column(1.,b,2.035,1.98,back,0.15));
let after=vec2(visible_kernel_column(1.,a,2.,front,back,0.15),
               visible_kernel_column(1.,b,2.035,front,back,0.15));
let correct=vec2(after.x+after.y,dot(after,beta));
let old=previous_uniform_rescale(vec2(before.x+before.y,dot(before,beta)),back-front);
result[i]=vec4(correct,old);
"#,
    );
    let front = 2.04_f32 as f64;
    let back = 2.05_f32 as f64;
    let first = quadrature(
        1.,
        (1_f32 / (0.01 * 0.01)) as f64,
        2.,
        front,
        back,
        0.15_f32 as f64,
    );
    let second = quadrature(
        1.,
        (1_f32 / (0.015 * 0.015)) as f64,
        2.035_f32 as f64,
        front,
        back,
        0.15_f32 as f64,
    );
    assert_eq!(first, 0., "foreground kernel must be entirely hidden");
    for (index, (result, beta)) in values.iter().zip(beta).enumerate() {
        close(result[0], first + second, &format!("layer water {index}"));
        let expected_mineral = first * beta[0] as f64 + second * beta[1] as f64;
        // Extinction is in inverse metres, so compare integrated mineral
        // extinction with the same relative tolerance as the water integral.
        let tolerance = 0.000_01 + expected_mineral.abs() * 0.000_1;
        assert!((result[1] as f64 - expected_mineral).abs() < tolerance);
        let concentration = result[1] / result[0];
        let expected = beta[1];
        assert!((concentration - expected).abs() < 0.01);
        let old_concentration = result[3] / result[2];
        if index < 2 {
            assert!(
                (old_concentration - expected).abs() > 350.,
                "old uniform-rescale control must reproduce the false patch: {old_concentration} vs {expected}"
            );
        } else {
            assert!((old_concentration - expected).abs() < 0.01);
        }
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn sweeping_interface_has_continuous_monotone_visible_columns() {
    let (device, queue) = device();
    const COUNT: usize = 513;
    let inputs: Vec<_> = (0..COUNT)
        .map(|i| [1.975 + i as f32 * (0.05 / (COUNT - 1) as f32), 0., 0., 0.])
        .collect();
    let values = probe(
        &device,
        &queue,
        COUNT,
        &inputs,
        r#"
let front=inputs[i].x;let a=1./(0.017*0.017);
let clear=visible_kernel_column(1.,a,2.,front,2.05,0.21);
let muddy=visible_kernel_column(0.65,a,2.005,front,2.018,0.13);
result[i]=vec4(clear,muddy,(clear+muddy)*321.,0.);
"#,
    );
    for (i, (input, value)) in inputs.iter().zip(&values).enumerate() {
        let front = input[0] as f64;
        let a = (1_f32 / (0.017 * 0.017)) as f64;
        close(
            value[0],
            quadrature(1., a, 2., front, 2.05_f32 as f64, 0.21_f32 as f64),
            &format!("sweep clear {i}"),
        );
        close(
            value[1],
            quadrature(
                0.65_f32 as f64,
                a,
                2.005_f32 as f64,
                front,
                2.018_f32 as f64,
                0.13_f32 as f64,
            ),
            &format!("sweep muddy {i}"),
        );
        let water = value[0] + value[1];
        if water > 1e-8 {
            assert!((value[2] / water - 321.).abs() < 0.001);
        }
        if i > 0 {
            let dt = (input[0] - inputs[i - 1][0]) as f64;
            for (axis, maximum_density) in
                [(0, 6.5625 * 0.21), (1, 6.5625 * 0.13 * 0.65_f64.powi(3))]
            {
                let change = values[i - 1][axis] as f64 - value[axis] as f64;
                assert!(change >= -0.000_002, "visible volume grew at sweep {i}");
                assert!(
                    change <= maximum_density * dt + 0.000_002,
                    "interface discontinuity at {i}: {change}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn projected_visible_columns_preserve_represented_volume_and_clipped_partition() {
    let (device, queue) = device();
    const WIDTH: usize = 256;
    let values = probe(
        &device,
        &queue,
        WIDTH * WIDTH,
        &[[0.; 4]],
        r#"
let q=(vec2(f32(i%256u),f32(i/256u))+0.5)/256.*2.-1.;
let s=1.-dot(q,q);if s<=0. {result[i]=vec4(0.);return;}
let a=1./(0.007*0.007);let center=0.4;let split=center+0.0021;
let all=visible_kernel_column(s,a,center,0.38,0.42,0.21);
let before=visible_kernel_column(s,a,center,0.38,split,0.21);
let after=visible_kernel_column(s,a,center,split,0.42,0.21);
result[i]=vec4(all,before,after,visible_kernel_column(s,a,center,0.42,0.43,0.21));
"#,
    );
    let area = (2. * 0.013 / WIDTH as f64) * (2. * 0.019 / WIDTH as f64);
    let mut integrated = [0.; 3];
    for value in &values {
        assert!(value.iter().all(|v| v.is_finite() && *v >= 0.));
        assert!((value[0] - value[1] - value[2]).abs() < 0.000_001);
        assert_eq!(value[3], 0.);
        for axis in 0..3 {
            integrated[axis] += value[axis] as f64 * area;
        }
    }
    let expected = 4. / 3. * std::f64::consts::PI * 0.013 * 0.007 * 0.019 * 0.21;
    assert!((integrated[0] / expected - 1.).abs() < 0.0001);
    assert!((integrated[0] - integrated[1] - integrated[2]).abs() < expected * 0.0001);
    // Independently integrate the clipped three-dimensional density. The
    // integration domain starts exactly at the visible plane and never uses
    // the shader's projected antiderivative or its midpoint samples.
    let side = 96;
    let slices = 128;
    let split = 0.0021_f64;
    let dz = (0.007 - split) / slices as f64;
    let mut expected_visible = 0.;
    for y in 0..side {
        for x in 0..side {
            let qx = (x as f64 + 0.5) / side as f64 * 2. - 1.;
            let qy = (y as f64 + 0.5) / side as f64 * 2. - 1.;
            for z in 0..slices {
                let qz = (split + (z as f64 + 0.5) * dz) / 0.007;
                expected_visible +=
                    6.5625 * 0.21 * (1. - qx * qx - qy * qy - qz * qz).max(0.).powi(3);
            }
        }
    }
    expected_visible *= (2. * 0.013 / side as f64) * (2. * 0.019 / side as f64) * dz;
    assert!(
        (integrated[2] / expected_visible - 1.).abs() < 0.0005,
        "visible represented volume {} vs independent quadrature {expected_visible}",
        integrated[2]
    );
}
#[test]
#[ignore = "requires real GPU; run explicitly with --ignored"]
fn joined_crest_gradient_matches_density_and_rejects_detached_foreground_drop() {
    let (device, queue) = device();
    let angle = 0.35_f64;
    let normal = [angle.sin(), angle.cos(), 0.];
    let tangent = [angle.cos(), -angle.sin(), 0.];
    // An inclined layer, with a genuinely detached drop in front of it.
    // The layer's level-set gradient must not inherit that drop's cap normal.
    let joined = normal.map(|v| (v * 0.0041) as f32);
    let drop_center: [f32; 3] =
        std::array::from_fn(|a| joined[a] + (normal[a] * 0.006 + tangent[a] * 0.003) as f32);
    let drop_radius = 0.009_f64;
    let front_distance = 0.006 + (drop_radius.powi(2) - 0.003_f64.powi(2)).sqrt();
    let raw_front: [f32; 3] =
        std::array::from_fn(|a| joined[a] + (normal[a] * front_distance) as f32);
    let ray = normal.map(|v| -v as f32);
    let mut inputs = vec![
        [joined[0], joined[1], joined[2], 0.],
        [raw_front[0], raw_front[1], raw_front[2], 0.],
        [ray[0], ray[1], ray[2], 0.],
    ];
    for z in -2..=2 {
        for x in -2..=2 {
            let center: [f32; 3] = std::array::from_fn(|a| {
                (tangent[a] * x as f64 * 0.012 + if a == 2 { z as f64 * 0.012 } else { 0. }) as f32
            });
            inputs.push([center[0], center[1], center[2], 1.]);
            inputs.push([
                (tangent[0] * 0.018) as f32,
                (tangent[1] * 0.018) as f32,
                0.,
                0.15,
            ]);
            inputs.push([
                (normal[0] * 0.008) as f32,
                (normal[1] * 0.008) as f32,
                0.,
                0.,
            ]);
            inputs.push([0., 0., 0.018, 0.]);
        }
    }
    inputs.extend([
        [drop_center[0], drop_center[1], drop_center[2], 0.],
        [0.009, 0., 0., 0.15],
        [0., 0.009, 0., 0.],
        [0., 0., 0.009, 0.],
    ]);
    let values = probe(
        &device,
        &queue,
        4,
        &inputs,
        r#"
var sum=vec3(0.);var density=0.;
for(var kernel=0u;kernel<26u;kernel++) {
    // Invocation 0 omits the detached drop; invocation 1 includes its zero
    // bulk support at the exact same joined point. Invocation 2 reproduces
    // the old raw-front probe and its unweighted gradient on the same scene.
    if i==0u && kernel==25u {continue;}
    let offset=3u+kernel*4u;
    let center=inputs[offset];let axis_x=inputs[offset+1u];
    let axis_y=inputs[offset+2u].xyz;let axis_z=inputs[offset+3u].xyz;
    var point=inputs[0].xyz;var support=center.w;
    if i==2u {
        let inset=min(0.004,0.35*min(length(axis_x.xyz),min(length(axis_y),length(axis_z))));
        point=inputs[1].xyz+inputs[2].xyz*inset;
        support=1.;
    }
    if i==3u {support=1.;}
    sum+=density_gradient(point,center.xyz,axis_x.xyz,axis_y,axis_z,axis_x.w)*support;
    density+=kernel_density(point,center.xyz,axis_x.xyz,axis_y,axis_z,axis_x.w)*support;
}
result[i]=vec4(sum,density);
"#,
    );
    // Independent scalar poly6 density in f64. The input axes are quantized
    // exactly as on the GPU; no production gradient or primitive is reused.
    let scalar_density = |point: [f64; 3]| {
        (0..26)
            .map(|kernel| {
                let offset = 3 + kernel * 4;
                let center = inputs[offset];
                let scale = inputs[offset + 1][3] as f64;
                let mut squared = 0.;
                for axis in &inputs[offset + 1..offset + 4] {
                    let length_squared: f64 = axis[..3].iter().map(|&v| (v as f64).powi(2)).sum();
                    let projected: f64 = (0..3)
                        .map(|a| (point[a] - center[a] as f64) * axis[a] as f64)
                        .sum();
                    squared += (projected / length_squared).powi(2);
                }
                6.5625 * scale * center[3] as f64 * (1. - squared).max(0.).powi(3)
            })
            .sum::<f64>()
    };
    let point = joined.map(|v| v as f64);
    let reference: [f64; 3] = std::array::from_fn(|axis| {
        let (mut lo, mut hi) = (point, point);
        lo[axis] -= 1e-7;
        hi[axis] += 1e-7;
        // density_gradient is the outward (negative scalar-gradient) vector.
        (scalar_density(lo) - scalar_density(hi)) / 2e-7
    });
    let length = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let reference_length = length(reference);
    assert!(
        reference_length > 10.,
        "fixture must resolve a free-surface gradient"
    );
    assert!(
        (scalar_density(point) - 0.5).abs() < 0.03,
        "probe must lie near the connected-water level set"
    );
    for (index, sample) in values[..2].iter().enumerate() {
        assert!(sample.iter().all(|x| x.is_finite()));
        let gradient: [f64; 3] = std::array::from_fn(|a| sample[a] as f64);
        let error = length(std::array::from_fn(|a| gradient[a] - reference[a]));
        assert!(
            error / reference_length < 0.0001,
            "joined gradient {index}: GPU {gradient:?}, independent derivative {reference:?}"
        );
        assert!((sample[3] as f64 - scalar_density(point)).abs() < 0.00001);
    }
    assert_eq!(
        values[0], values[1],
        "a zero-bulk detached drop changed the joined interface"
    );
    // The detached drop's rendering support overlaps the joined point. This
    // additional control proves that bulk weighting is necessary independently
    // of moving the probe away from the old foreground envelope.
    let unweighted: [f64; 3] = std::array::from_fn(|a| values[3][a] as f64);
    let unweighted_length = length(unweighted);
    assert!(unweighted_length > 1.);
    let unweighted_alignment: f64 = (0..3)
        .map(|a| unweighted[a] * reference[a] / (unweighted_length * reference_length))
        .sum();
    assert!(
        unweighted_alignment < 0.995,
        "unweighted common-point control must change the normal: {unweighted_alignment}"
    );
    let old: [f64; 3] = std::array::from_fn(|a| values[2][a] as f64);
    let old_length = length(old);
    assert!(
        old_length > 1.,
        "old front probe must reproduce the drop cap"
    );
    let old_alignment: f64 = (0..3)
        .map(|a| old[a] * reference[a] / (old_length * reference_length))
        .sum();
    assert!(
        old_alignment < 0.9,
        "old unweighted front normal must measurably disagree: cosine {old_alignment}"
    );
}

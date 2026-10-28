//! Explicit real-GPU coverage of the production reflection marcher, texture
//! visibility, colour gather and fallback. The scene is rendered analytically;
//! expected intersections are computed separately in f64, without ray marching.
use glam::{DVec3, Mat4, Vec3};
use wgpu::util::DeviceExt;

const SIZE: u32 = 96;
const NEAR: f32 = 0.1;
const FOV: f32 = 1.;
const FALLBACK: [f32; 3] = [0.02, 0.05, 0.15];
const WALL: [f32; 3] = [0.9, 0.12, 0.05];

const SCENE: &str = r#"
struct Uniforms {
 clip_from_world:mat4x4<f32>, world_from_view:mat4x4<f32>,
 view_from_clip:mat4x4<f32>, view_from_world:mat4x4<f32>,
 viewport:vec4<f32>, sediment:vec4<f32>
};
@group(0) @binding(0) var<uniform> u:Uniforms;
@vertex fn fullscreen(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
 let p=array<vec2<f32>,3>(vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));
 return vec4(p[i],0.,1.);
}
struct ScenePixel { @location(0) colour:vec4<f32>, @builtin(frag_depth) depth:f32 };
@fragment fn analytic_scene(@builtin(position) p:vec4<f32>)->ScenePixel {
 let ndc=vec2(p.x/u.viewport.x*2.-1.,1.-p.y/u.viewport.y*2.);
 let ray=vec3(ndc.x/u.clip_from_world[0][0],ndc.y/u.clip_from_world[1][1],-1.);
 var distance=1000.;var colour=vec3(0.);
 // sediment is test-only scene configuration: wall depth, bed y, wall min/max y.
 let wall=u.sediment.x;
 let on_wall=ray*wall;
 if wall>0. && abs(on_wall.x)<0.95 && on_wall.y>=u.sediment.z && on_wall.y<=u.sediment.w {
   distance=wall;colour=vec3(0.9,0.12,0.05);
 }
 if u.sediment.y<0. && ray.y<0. {
   let bed=u.sediment.y/ray.y;
   if bed>0.05 && bed<10. && bed<distance {distance=bed;colour=vec3(0.75,0.4,0.08);}
 }
 var out:ScenePixel;out.colour=vec4(colour,1.);
 out.depth=select(0.,0.1/distance,distance<1000.);return out;
}
"#;

const PROBE: &str = r#"
struct ReflectionCase { position:vec4<f32>, direction:vec4<f32>, normal:vec4<f32>, fallback:vec4<f32> };
@group(1) @binding(0) var<storage,read> reflection_cases:array<ReflectionCase>;
@group(1) @binding(1) var<storage,read_write> reflection_results:array<vec4<f32>>;
@compute @workgroup_size(64) fn probe_reflection(@builtin(global_invocation_id) id:vec3<u32>) {
 if id.x>=arrayLength(&reflection_cases) {return;}
 let c=reflection_cases[id.x];
 reflection_results[id.x]=channel_reflection(c.position.xyz,c.direction.xyz,c.normal.xyz,c.fallback.xyz);
}
"#;

#[derive(Clone, Copy)]
struct Scene {
    wall_depth: f32,
    bed_y: f32,
    wall_min_y: f32,
    wall_max_y: f32,
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    position: Vec3,
    direction: Vec3,
    normal: Vec3,
    wall_hit: bool,
}

impl Case {
    fn new(name: &'static str, p: [f32; 3], d: [f32; 3], wall_hit: bool) -> Self {
        Self {
            name,
            position: Vec3::from_array(p),
            direction: Vec3::from_array(d).normalize(),
            normal: Vec3::Y,
            wall_hit,
        }
    }

    fn packed(self) -> [f32; 16] {
        let [px, py, pz] = self.position.to_array();
        let [dx, dy, dz] = self.direction.to_array();
        let [nx, ny, nz] = self.normal.to_array();
        [
            px,
            py,
            pz,
            0.,
            dx,
            dy,
            dz,
            0.,
            nx,
            ny,
            nz,
            0.,
            FALLBACK[0],
            FALLBACK[1],
            FALLBACK[2],
            0.,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Material {
    Wall,
    Bed,
}

// Independent geometric oracle. The actual rounded f32 ray is promoted to f64;
// there is no reproduction of the GPU's coarse steps, bisection or texel lookup.
fn intersection(scene: Scene, case: Case) -> Option<(Material, f64, DVec3)> {
    let origin = case.position.as_dvec3() + case.normal.as_dvec3() * 0.001;
    let direction = case.direction.as_dvec3();
    let mut hit = None;
    if scene.wall_depth > 0. && direction.z != 0. {
        let t = (-f64::from(scene.wall_depth) - origin.z) / direction.z;
        let p = origin + direction * t;
        if t > 0.
            && p.x.abs() < 0.95
            && p.y >= f64::from(scene.wall_min_y)
            && p.y <= f64::from(scene.wall_max_y)
        {
            hit = Some((Material::Wall, t, p));
        }
    }
    if scene.bed_y < 0. && direction.y != 0. {
        let t = (f64::from(scene.bed_y) - origin.y) / direction.y;
        let p = origin + direction * t;
        if t > 0. && -p.z > 0.05 && -p.z < 10. && hit.is_none_or(|(_, old, _)| t < old) {
            hit = Some((Material::Bed, t, p));
        }
    }
    hit
}

fn smoothstep(lo: f64, hi: f64, x: f64) -> f64 {
    let t = ((x - lo) / (hi - lo)).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

fn edge_pixels(p: DVec3) -> f64 {
    let f = f64::from((FOV * 0.5).tan().recip());
    let x = (f * p.x / -p.z * 0.5 + 0.5) * f64::from(SIZE);
    let y = (0.5 - f * p.y / -p.z * 0.5) * f64::from(SIZE);
    x.min(y).min(f64::from(SIZE) - x).min(f64::from(SIZE) - y)
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    uniform: wgpu::Buffer,
    scene_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    render_group: wgpu::BindGroup,
    render: wgpu::RenderPipeline,
    textures: wgpu::BindGroup,
    case_layout: wgpu::BindGroupLayout,
    compute: wgpu::ComputePipeline,
}

impl Gpu {
    fn new() -> Self {
        let adapter =
            pollster::block_on(wgpu::Instance::default().request_adapter(&Default::default()))
                .expect("real GPU required for reflection regression");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("reflection fixture uniforms"),
            contents: bytemuck::cast_slice(&[0f32; 72]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let texture = |format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("analytic reflection scene"),
                    size: wgpu::Extent3d {
                        width: SIZE,
                        height: SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let scene_view = texture(wgpu::TextureFormat::Rgba32Float);
        let depth_view = texture(wgpu::TextureFormat::Depth32Float);
        let buffer_type = |ty| wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty,
            count: None,
        };
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[entry(
                0,
                wgpu::ShaderStages::FRAGMENT,
                buffer_type(wgpu::BufferBindingType::Uniform),
            )],
        });
        let render_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &render_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("analytic opaque wall and bed"),
            source: wgpu::ShaderSource::Wgsl(SCENE.into()),
        });
        let render = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&render_layout],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("analytic_scene"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba32Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let texture_type = |sample_type| wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(
                    0,
                    wgpu::ShaderStages::COMPUTE,
                    texture_type(wgpu::TextureSampleType::Float { filterable: false }),
                ),
                entry(
                    2,
                    wgpu::ShaderStages::COMPUTE,
                    texture_type(wgpu::TextureSampleType::Depth),
                ),
                entry(
                    3,
                    wgpu::ShaderStages::COMPUTE,
                    buffer_type(wgpu::BufferBindingType::Uniform),
                ),
            ],
        });
        let textures = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let case_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(
                    0,
                    wgpu::ShaderStages::COMPUTE,
                    buffer_type(wgpu::BufferBindingType::Storage { read_only: true }),
                ),
                entry(
                    1,
                    wgpu::ShaderStages::COMPUTE,
                    buffer_type(wgpu::BufferBindingType::Storage { read_only: false }),
                ),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production water reflection helper"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{PROBE}",
                    include_str!("../../mining/src/water_compose.wgsl")
                )
                .into(),
            ),
        });
        let compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&texture_layout, &case_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("probe_reflection"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            device,
            queue,
            uniform,
            scene_view,
            depth_view,
            render_group,
            render,
            textures,
            case_layout,
            compute,
        }
    }

    fn run(&self, scene: Scene, cases: &[Case]) -> Vec<[f32; 4]> {
        // Infinite reverse-Z perspective: clip.z=near, clip.w=-view.z.
        let f = (FOV * 0.5).tan().recip();
        let projection = Mat4::from_cols_array(&[
            f, 0., 0., 0., 0., f, 0., 0., 0., 0., 0., -1., 0., 0., NEAR, 0.,
        ]);
        let mut uniforms = [0f32; 72];
        for (i, matrix) in [
            projection,
            Mat4::IDENTITY,
            projection.inverse(),
            Mat4::IDENTITY,
        ]
        .iter()
        .enumerate()
        {
            uniforms[i * 16..(i + 1) * 16].copy_from_slice(&matrix.to_cols_array());
        }
        uniforms[64..68].copy_from_slice(&[SIZE as f32, SIZE as f32, 0.045, 1.]);
        uniforms[68..72].copy_from_slice(&[
            scene.wall_depth,
            scene.bed_y,
            scene.wall_min_y,
            scene.wall_max_y,
        ]);
        self.queue
            .write_buffer(&self.uniform, 0, bytemuck::cast_slice(&uniforms));
        let packed: Vec<_> = cases.iter().map(|c| c.packed()).collect();
        let input = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: cases.len() as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.case_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("populate analytic opaque RGB and reverse-Z depth"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.scene_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.render);
            pass.set_bind_group(0, &self.render_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.compute);
            pass.set_bind_group(0, &self.textures, &[]);
            pass.set_bind_group(1, &group, &[]);
            pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let values: Vec<[f32; 4]> =
            bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec();
        readback.unmap();
        values
    }
}

fn check(gpu: &Gpu, scene: Scene, cases: &[Case]) {
    let values = gpu.run(scene, cases);
    for (&case, value) in cases.iter().zip(values) {
        assert!(
            value.iter().all(|v| v.is_finite()),
            "{}: {value:?}",
            case.name
        );
        if !case.wall_hit {
            assert_eq!(
                value,
                [FALLBACK[0], FALLBACK[1], FALLBACK[2], 0.],
                "{}: unexpected reflection; analytic hit {:?}",
                case.name,
                intersection(scene, case)
            );
            continue;
        }
        let (material, t, hit) = intersection(scene, case).expect(case.name);
        assert_eq!(material, Material::Wall, "{}", case.name);
        assert!((0. ..0.8).contains(&t), "{}: {t}", case.name);
        assert!(case.normal.as_dvec3().dot(hit - case.position.as_dvec3()) > 0.0005);
        let expected = smoothstep(0., 16., edge_pixels(hit)) * (1. - smoothstep(0.55, 0.8, t));
        // A bisection endpoint can overshoot by at most 0.05/32 m. These
        // samples stay well inside the coloured wall and its 25 mm support.
        assert!(
            (f64::from(value[3]) - expected).abs() < 0.03,
            "{}: alpha {} versus analytic {expected}; t={t}, edge={}",
            case.name,
            value[3],
            edge_pixels(hit)
        );
        assert!(
            value[3] > 0.15,
            "{}: missing coloured wall {value:?}",
            case.name
        );
        for i in 0..3 {
            let expected_colour = FALLBACK[i] + (WALL[i] - FALLBACK[i]) * value[3];
            assert!(
                (value[i] - expected_colour).abs() < 0.00001,
                "{}: colour {value:?}",
                case.name
            );
        }
    }
}

#[test]
#[ignore = "requires real GPU; run explicitly with --ignored --test-threads=1"]
fn production_reflections_hit_only_observed_above_water_geometry() {
    let gpu = Gpu::new();
    let scene = Scene {
        wall_depth: 1.28,
        bed_y: -0.30,
        wall_min_y: -0.12,
        wall_max_y: 0.85,
    };
    let p = [0., -0.15, -1.];
    let direction = [0.1, 0.65, -0.75];
    let mut tilted = Case::new(
        "inclined tangent plane",
        [-0.1, -0.1, -1.05],
        [0.25, 0.5, -0.83],
        true,
    );
    tilted.normal = Vec3::new(0.05, 1., 0.08).normalize();
    check(
        &gpu,
        scene,
        &[
            Case::new("interior wall", p, direction, true),
            Case::new("oblique wall", [0.1, -0.15, -0.9], [-0.2, 0.6, -0.8], true),
            tilted,
            Case::new("finite range fade", [0., -0.15, -0.8], direction, true),
            Case::new(
                "screen edge fade",
                [0.45, -0.15, -1.],
                [0.4, 0.6, -0.69],
                true,
            ),
            Case::new("below tangent toward bed", p, [0., -0.5, -0.85], false),
            Case::new("reversed reflection", p, [-0.1, -0.65, 0.75], false),
            Case::new("toward camera", p, [0., 0.2, 1.], false),
            Case::new("leaves screen", p, [1., 0.1, -0.05], false),
            Case::new("offscreen origin", [2.5, -0.15, -1.], direction, false),
        ],
    );
    check(
        &gpu,
        Scene {
            wall_depth: 2.2,
            ..scene
        },
        &[Case::new("beyond maximum range", p, direction, false)],
    );
    check(
        &gpu,
        Scene {
            wall_depth: 0.,
            bed_y: 0.,
            ..scene
        },
        &[Case::new("sky has no opaque hit", p, direction, false)],
    );
    // Near-tangent rays can cross quantized depth at a submerged bed/silhouette.
    // Neither the brown floor nor a nearby submerged red wall may be reflected.
    check(
        &gpu,
        Scene {
            wall_depth: 0.,
            bed_y: -0.1505,
            ..scene
        },
        &[Case::new(
            "shallow submerged bed",
            p,
            [0.15, 0.0002, -0.99],
            false,
        )],
    );
    check(
        &gpu,
        Scene {
            wall_min_y: -0.8,
            wall_max_y: -0.151,
            ..scene
        },
        &[Case::new(
            "submerged silhouette within support radius",
            p,
            [0., 0.001, -1.],
            false,
        )],
    );
    check(
        &gpu,
        Scene {
            wall_depth: 0.25,
            bed_y: 0.,
            wall_min_y: -10.,
            wall_max_y: 10.,
        },
        &[
            Case::new(
                "visible near wall control",
                [0., -0.02, -0.1],
                [0., 0.5, -1.],
                true,
            ),
            Case::new(
                "behind-camera origin cannot enter screen",
                [0., 0., 0.02],
                [0., 0.1, -1.],
                false,
            ),
            Case::new(
                "inside near-camera exclusion",
                [0., 0., -0.005],
                [0., 0.1, -1.],
                false,
            ),
        ],
    );
}

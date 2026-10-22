//! Screen-space fluid from authoritative FLIP and saved outflow: nearest particle
//! depth, edge-preserving smoothing, then one shaded surface. Render-only buffers
//! are bounded/reused; neither the particle state nor terrain flow is modified.
use bevy::{
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    ecs::query::QueryItem,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            binding_types::{texture_2d, texture_depth_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        view::{ExtractedView, ViewDepthTexture, ViewTarget},
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc,
};

const MAX_SPLATS: usize = 65_536;
const MAX_RENDER_SPLATS: usize = MAX_SPLATS * 2;
type PresentationKey = (u64, u32, Option<(mining_sim::Vec3, f32, f32)>);
#[derive(Resource, Clone, ExtractResource)]
pub struct WaterSplats {
    revision: u64,
    data: Arc<Vec<InstanceData>>,
    health: Arc<AtomicU8>,
    presentation: crate::water_presentation::Presentation,
    presentation_key: Option<PresentationKey>,
    builds: u64,
    build_ms: f64,
    max_build_ms: f64,
    last_sample_time: f64,
    surface_epoch: u64,
    capture_ms: f64,
    max_capture_ms: f64,
    captures: u64,
    terrain_data: Arc<Vec<InstanceData>>,
    terrain_revision: u64,
    terrain_key: Option<(u64, u64, u64, u64)>,
    fluid_revision: Option<crate::app::FluidRevision>,
    sim_time: f32,
}
impl WaterSplats {
    /// Actual interpolated render kernels, for the opt-in motion evidence only.
    pub fn motion_samples(&self, sluice: &mining_sim::Sluice) -> Vec<serde_json::Value> {
        let Some(p) = sluice.placement else {
            return Vec::new();
        };
        let alpha = self.presentation.alpha;
        let mut points: Vec<_> =
            self.data
                .iter()
                .filter(|v| v.shape[2] == 0.)
                .filter_map(|v| {
                    let c = Vec4::from_array(v.previous_center)
                        .lerp(Vec4::from_array(v.center_radius), alpha);
                    let delta = mining_sim::Vec3::new(c.x, c.y, c.z) - p.inlet;
                    let downstream = delta.dot(p.forward());
                    let height = delta.y + downstream * p.slope_deg.to_radians().tan();
                    (downstream > 0.3 && downstream < 1.35 && height > 0.035 && c.w > 0.0001)
                        .then_some((
                            height,
                            c,
                            downstream,
                            v.previous_shape[3] * (1. - alpha) + v.shape[3] * alpha,
                        ))
                })
                .collect();
        points.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        points.into_iter().take(128).map(|(height,c,downstream,planarity)|serde_json::json!({"world":[c.x,c.y,c.z],"downstream_m":downstream,"height_m":height,"radius_m":c.w,"planarity":planarity})).collect()
    }
    pub fn presentation_report(&self) -> serde_json::Value {
        serde_json::json!({"surface_builds": self.builds,
            "mean_surface_build_ms": self.build_ms / self.builds.max(1) as f64,
            "max_surface_build_ms": self.max_build_ms,
            "mean_capture_ms": self.capture_ms / self.captures.max(1) as f64,
            "max_capture_ms": self.max_capture_ms,
            "interpolation_alpha": self.presentation.alpha,
            "presented_sim_time": self.presentation.time,
            "buffer_underrun_frames": self.presentation.underruns,
            "paired_render_particles": self.presentation.paired})
    }
    pub fn terrain_count(&self) -> usize {
        self.terrain_data.len()
    }
    pub fn terrain_fine_count(&self) -> usize {
        self.terrain_data
            .iter()
            .take_while(|p| p.shape[3] > 1.)
            .count()
    }
}
impl Default for WaterSplats {
    fn default() -> Self {
        Self {
            revision: u64::MAX,
            data: Arc::new(Vec::new()),
            health: Arc::new(AtomicU8::new(0)),
            presentation: Default::default(),
            presentation_key: None,
            builds: 0,
            build_ms: 0.,
            max_build_ms: 0.,
            last_sample_time: -1.,
            surface_epoch: 0,
            capture_ms: 0.,
            max_capture_ms: 0.,
            captures: 0,
            terrain_data: Arc::new(Vec::new()),
            terrain_revision: 0,
            terrain_key: None,
            fluid_revision: None,
            sim_time: 0.,
        }
    }
}
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct InstanceData {
    pub center_radius: [f32; 4],
    pub velocity: [f32; 4],
    pub shape: [f32; 4],
    pub axis_x: [f32; 4],
    pub axis_y: [f32; 4],
    pub axis_z: [f32; 4],
    pub slope_x: [f32; 4],
    pub slope_z: [f32; 4],
    pub flow_z: [f32; 4],
    pub previous_center: [f32; 4],
    pub previous_x: [f32; 4],
    pub previous_y: [f32; 4],
    pub previous_z: [f32; 4],
    pub previous_shape: [f32; 4],
}

pub struct WaterSplatPlugin;
impl Plugin for WaterSplatPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "water_depth.wgsl");
        bevy::asset::embedded_asset!(app, "water_blur.wgsl");
        bevy::asset::embedded_asset!(app, "water_compose.wgsl");
        app.init_resource::<WaterSplats>()
            .insert_non_send_resource(crate::water_surface_worker::SurfaceWorker::default())
            .add_plugins(ExtractResourcePlugin::<WaterSplats>::default());
        app.sub_app_mut(RenderApp)
            .add_systems(RenderStartup, init_pipelines)
            .add_systems(
                Render,
                (prepare_particles, prepare_views).in_set(RenderSystems::PrepareResources),
            )
            .add_render_graph_node::<ViewNodeRunner<FluidNode>>(Core3d, FluidLabel)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::Tonemapping,
                    FluidLabel,
                    Node3d::EndMainPassPostProcessing,
                ),
            );
    }
}
#[derive(Resource)]
struct ParticleBuffer {
    buffer: Buffer,
    count: u32,
    revision: u64,
    terrain_count: u32,
    terrain_fine_count: u32,
    terrain_revision: u64,
}
#[derive(Clone, ShaderType)]
struct FluidUniforms {
    clip_from_world: Mat4,
    world_from_view: Mat4,
    view_from_clip: Mat4,
    view_from_world: Mat4,
    viewport: Vec4,
    sediment: Vec4,
}
#[derive(Component)]
struct FluidTargets {
    size: UVec2,
    river_raw: TextureView,
    river_normals: TextureView,
    river_depth: TextureView,
    raw: TextureView,
    ping: TextureView,
    pong: TextureView,
    depth: TextureView,
    thickness: TextureView,
    normal_sums: TextureView,
    normals: TextureView,
    uniform: UniformBuffer<FluidUniforms>,
}
#[derive(Resource)]
struct FluidPipelines {
    particles: BindGroupLayout,
    thickness_particles: BindGroupLayout,
    blur: BindGroupLayout,
    compose: BindGroupLayout,
    depth_id: CachedRenderPipelineId,
    thickness_id: CachedRenderPipelineId,
    reconstruct_id: CachedRenderPipelineId,
    blur_x: CachedRenderPipelineId,
    blur_y: CachedRenderPipelineId,
    compose_id: CachedRenderPipelineId,
}
fn init_pipelines(
    mut commands: Commands,
    device: Res<RenderDevice>,
    assets: Res<AssetServer>,
    cache: Res<PipelineCache>,
) {
    let storage = BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::VERTEX,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let particles = device.create_bind_group_layout(
        "fluid particles",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::VERTEX_FRAGMENT,
            ((0, storage), (1, uniform_buffer::<FluidUniforms>(false))),
        ),
    );
    let thickness_particles = device.create_bind_group_layout(
        "fluid column inputs",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::VERTEX_FRAGMENT,
            (
                (0, storage),
                (1, uniform_buffer::<FluidUniforms>(false)),
                (
                    2,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
                (3, texture_depth_2d()),
            ),
        ),
    );
    let blur = device.create_bind_group_layout(
        "fluid smoothing",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer::<FluidUniforms>(false),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    let compose = device.create_bind_group_layout(
        "fluid composite",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_depth_2d(),
                uniform_buffer::<FluidUniforms>(false),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
        ),
    );
    let make = |shader: Handle<Shader>,
                layout: &BindGroupLayout,
                vertex: &'static str,
                fragment: &'static str,
                format,
                depth| {
        cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some(format!("water {fragment}").into()),
            layout: vec![layout.clone()],
            vertex: VertexState {
                shader: shader.clone(),
                entry_point: Some(vertex.into()),
                ..default()
            },
            fragment: Some(FragmentState {
                shader,
                entry_point: Some(fragment.into()),
                targets: {
                    let mut targets = vec![Some(ColorTargetState {
                        format,
                        blend: if fragment == "particle_thickness" {
                            Some(BlendState {
                                color: BlendComponent {
                                    src_factor: BlendFactor::One,
                                    dst_factor: BlendFactor::One,
                                    operation: BlendOperation::Add,
                                },
                                alpha: BlendComponent {
                                    src_factor: BlendFactor::One,
                                    dst_factor: BlendFactor::One,
                                    operation: BlendOperation::Add,
                                },
                            })
                        } else {
                            None
                        },
                        write_mask: ColorWrites::ALL,
                    })];
                    if fragment == "particle_depth" || fragment == "particle_thickness" {
                        targets.push(Some(ColorTargetState {
                            format: TextureFormat::Rgba16Float,
                            blend: if fragment == "particle_thickness" {
                                targets[0].as_ref().unwrap().blend
                            } else {
                                None
                            },
                            write_mask: ColorWrites::ALL,
                        }));
                    }
                    targets
                },
                ..default()
            }),
            depth_stencil: depth,
            ..default()
        })
    };
    let depth_shader = assets.load("embedded://mining/water_depth.wgsl");
    let blur_shader = assets.load("embedded://mining/water_blur.wgsl");
    let compose_shader = assets.load("embedded://mining/water_compose.wgsl");
    let depth_id = make(
        depth_shader.clone(),
        &particles,
        "particle_vertex",
        "particle_depth",
        TextureFormat::Rgba32Float,
        Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual,
            stencil: default(),
            bias: default(),
        }),
    );
    let thickness_id = make(
        depth_shader,
        &thickness_particles,
        "particle_vertex",
        "particle_thickness",
        TextureFormat::Rgba16Float,
        None,
    );
    let reconstruct_id = make(
        blur_shader.clone(),
        &blur,
        "fullscreen",
        "reconstruct",
        TextureFormat::R32Float,
        None,
    );
    let blur_x = make(
        blur_shader.clone(),
        &blur,
        "fullscreen",
        "blur_x",
        TextureFormat::R32Float,
        None,
    );
    let blur_y = make(
        blur_shader,
        &blur,
        "fullscreen",
        "blur_y",
        TextureFormat::R32Float,
        None,
    );
    let compose_id = make(
        compose_shader,
        &compose,
        "fullscreen",
        "compose",
        TextureFormat::bevy_default(),
        None,
    );
    commands.insert_resource(FluidPipelines {
        particles,
        thickness_particles,
        blur,
        compose,
        depth_id,
        thickness_id,
        reconstruct_id,
        blur_x,
        blur_y,
        compose_id,
    });
}
fn prepare_particles(
    mut commands: Commands,
    data: Res<WaterSplats>,
    existing: Option<ResMut<ParticleBuffer>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let upload = |buffer: &mut ParticleBuffer| {
        if buffer.revision != data.revision {
            if !data.data.is_empty() {
                queue.write_buffer(&buffer.buffer, 0, bytemuck::cast_slice(&data.data));
            }
            buffer.count = data.data.len() as u32;
            buffer.revision = data.revision;
        }
        if buffer.terrain_revision != data.terrain_revision {
            if !data.terrain_data.is_empty() {
                queue.write_buffer(
                    &buffer.buffer,
                    (MAX_RENDER_SPLATS * size_of::<InstanceData>()) as u64,
                    bytemuck::cast_slice(&data.terrain_data),
                );
            }
            buffer.terrain_count = data.terrain_data.len() as u32;
            buffer.terrain_fine_count = data
                .terrain_data
                .iter()
                .take_while(|p| p.shape[3] > 1.)
                .count() as u32;
            buffer.terrain_revision = data.terrain_revision;
        }
    };
    if let Some(mut buffer) = existing {
        upload(&mut buffer);
        return;
    }
    let mut buffer = ParticleBuffer {
        buffer: device.create_buffer(&BufferDescriptor {
            label: Some("Bounded fluid kernels and continuous terrain water"),
            size: ((MAX_RENDER_SPLATS + crate::terrain_water::MAX_SPLATS)
                * size_of::<InstanceData>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        count: 0,
        revision: u64::MAX,
        terrain_count: 0,
        terrain_fine_count: 0,
        terrain_revision: u64::MAX,
    };
    upload(&mut buffer);
    commands.insert_resource(buffer);
}
fn prepare_views(
    mut commands: Commands,
    water: Res<WaterSplats>,
    mut views: Query<(
        Entity,
        &ExtractedCamera,
        &ExtractedView,
        Option<&mut FluidTargets>,
    )>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    for (entity, camera, view, existing) in &mut views {
        let Some(full) = camera.physical_target_size else {
            continue;
        };
        if full.x == 0 || full.y == 0 {
            continue;
        }
        let size = (full + UVec2::ONE) / 2;
        let world_from_view = view.world_from_view.to_matrix();
        let uniforms = FluidUniforms {
            clip_from_world: view.clip_from_view * world_from_view.inverse(),
            world_from_view,
            view_from_clip: view.clip_from_view.inverse(),
            view_from_world: world_from_view.inverse(),
            sediment: Vec4::new(
                0., // reserved; particle optics carry local slurry extinction
                0., // reserved
                water.sim_time,
                0.,
            ),
            viewport: Vec4::new(
                full.x as f32,
                full.y as f32,
                0.045,
                water.presentation.alpha,
            ),
        };
        if let Some(mut target) = existing {
            if target.size == full {
                target.uniform.set(uniforms);
                target.uniform.write_buffer(&device, &queue);
                continue;
            }
        }
        let texture = |label, format, dimensions: UVec2| {
            device
                .create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: dimensions.x,
                        height: dimensions.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };
        let mut uniform = UniformBuffer::from(uniforms);
        uniform.write_buffer(&device, &queue);
        commands.entity(entity).insert(FluidTargets {
            size: full,
            // River contact uses full-resolution geometry. Particle splats and
            // bilateral smoothing retain their bounded half-resolution targets.
            river_raw: texture("river depth and flow", TextureFormat::Rgba32Float, full),
            river_normals: texture("river normal and film", TextureFormat::Rgba16Float, full),
            river_depth: texture("river depth test", TextureFormat::Depth32Float, full),
            raw: texture("fluid raw depth", TextureFormat::Rgba32Float, size),
            ping: texture("fluid blur ping", TextureFormat::R32Float, size),
            pong: texture("fluid blur pong", TextureFormat::R32Float, size),
            depth: texture("fluid depth test", TextureFormat::Depth32Float, size),
            thickness: texture(
                "fluid optical depth moments",
                TextureFormat::Rgba16Float,
                size,
            ),
            normal_sums: texture(
                "fluid surface normal moments",
                TextureFormat::Rgba16Float,
                size,
            ),
            normals: texture("continuous water normals", TextureFormat::Rgba16Float, size),
            uniform,
        });
    }
}
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct FluidLabel;
#[derive(Default)]
struct FluidNode;
impl ViewNode for FluidNode {
    type ViewQuery = (
        &'static ViewTarget,
        &'static ViewDepthTexture,
        &'static FluidTargets,
    );
    fn run(
        &self,
        _: &mut RenderGraphContext,
        context: &mut RenderContext,
        (view, scene_depth, target): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Some(particles) = world.get_resource::<ParticleBuffer>() else {
            return Ok(());
        };
        if particles.count == 0 && particles.terrain_count == 0 {
            return Ok(());
        }
        let settings = world.resource::<FluidPipelines>();
        let cache = world.resource::<PipelineCache>();
        let health = &world.resource::<WaterSplats>().health;
        for id in [
            settings.depth_id,
            settings.thickness_id,
            settings.reconstruct_id,
            settings.blur_x,
            settings.blur_y,
            settings.compose_id,
        ] {
            if let CachedPipelineState::Err(error) = cache.get_render_pipeline_state(id) {
                if health.swap(2, Ordering::AcqRel) != 2 {
                    eprintln!("WATER RENDER FAILED: {error}");
                }
                return Ok(());
            }
        }
        let (
            Some(depth),
            Some(thickness),
            Some(reconstruct),
            Some(blur_x),
            Some(blur_y),
            Some(compose),
        ) = (
            cache.get_render_pipeline(settings.depth_id),
            cache.get_render_pipeline(settings.thickness_id),
            cache.get_render_pipeline(settings.reconstruct_id),
            cache.get_render_pipeline(settings.blur_x),
            cache.get_render_pipeline(settings.blur_y),
            cache.get_render_pipeline(settings.compose_id),
        )
        else {
            return Ok(());
        };
        let Some(uniform) = target.uniform.binding() else {
            return Ok(());
        };
        let device = context.render_device();
        let particle_group = device.create_bind_group(
            "fluid depth inputs",
            &settings.particles,
            &BindGroupEntries::sequential((particles.buffer.as_entire_binding(), uniform.clone())),
        );
        let thickness_group = device.create_bind_group(
            "fluid column inputs",
            &settings.thickness_particles,
            &BindGroupEntries::sequential((
                particles.buffer.as_entire_binding(),
                uniform.clone(),
                &target.raw,
                scene_depth.view(),
            )),
        );
        let blur_group = |texture: &TextureView| {
            device.create_bind_group(
                "fluid blur inputs",
                &settings.blur,
                &BindGroupEntries::sequential((texture, uniform.clone(), &target.thickness)),
            )
        };
        let raw_group = blur_group(&target.raw);
        let ping_group = blur_group(&target.ping);
        let pong_group = blur_group(&target.pong);
        let post = view.post_process_write();
        let compose_group = device.create_bind_group(
            "fluid compose inputs",
            &settings.compose,
            &BindGroupEntries::sequential((
                post.source,
                &target.pong,
                scene_depth.view(),
                uniform,
                &target.raw,
                &target.thickness,
                &target.normals,
                &target.river_raw,
                &target.river_normals,
                &target.normal_sums,
            )),
        );
        let color = |view| {
            Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Default::default()),
                    store: StoreOp::Store,
                },
            })
        };
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid particle depth"),
                    color_attachments: &[color(&target.raw), color(&target.normals)],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &target.depth,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(0.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            pass.set_pipeline(depth);
            pass.set_bind_group(0, &particle_group, &[]);
            pass.draw(0..6, 0..particles.count);
        }
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("full-resolution river contact"),
                    color_attachments: &[color(&target.river_raw), color(&target.river_normals)],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &target.river_depth,
                        depth_ops: Some(Operations {
                            load: LoadOp::Clear(0.0),
                            store: StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            pass.set_pipeline(depth);
            pass.set_bind_group(0, &particle_group, &[]);
            pass.draw(
                0..(6 * crate::environment::FINE_SUBDIVISIONS.pow(2)) as u32,
                MAX_RENDER_SPLATS as u32..MAX_RENDER_SPLATS as u32 + particles.terrain_fine_count,
            );
            pass.draw(
                0..6,
                MAX_RENDER_SPLATS as u32 + particles.terrain_fine_count
                    ..MAX_RENDER_SPLATS as u32 + particles.terrain_count,
            );
        }
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid volume-normalized optical thickness"),
                    color_attachments: &[color(&target.thickness), color(&target.normal_sums)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            pass.set_pipeline(thickness);
            pass.set_bind_group(0, &thickness_group, &[]);
            pass.draw(0..6, 0..particles.count);
        }
        // Resolve the overlapping density interface before the existing two bilateral
        // iterations. Outlet/river depth is unchanged by this channel pass.
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid overlapping interface"),
                    color_attachments: &[color(&target.pong)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            pass.set_pipeline(reconstruct);
            pass.set_bind_group(0, &raw_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // Two separable bilateral iterations smooth shared fluid normals, without
        // averaging across a depth jump or expanding water into empty pixels.
        for (pipeline, group, output) in [
            (blur_x, &pong_group, &target.ping),
            (blur_y, &ping_group, &target.pong),
            (blur_x, &pong_group, &target.ping),
            (blur_y, &ping_group, &target.pong),
        ] {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid bilateral smoothing"),
                    color_attachments: &[color(output)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid continuous surface"),
                    color_attachments: &[color(post.destination)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            pass.set_pipeline(compose);
            pass.set_bind_group(0, &compose_group, &[]);
            pass.draw(0..3, 0..1);
        }
        health.store(1, Ordering::Release);
        Ok(())
    }
}
pub fn surface_ready(splats: &WaterSplats) -> bool {
    splats.health.load(Ordering::Acquire) == 1
}
pub fn surface_failed(splats: &WaterSplats) -> bool {
    splats.health.load(Ordering::Acquire) == 2
}
/// Build world-space splats from the exact live FLIP state, including detached
/// droplets. Stable hashed identities bound upload/fragment cost at high volumes.
pub fn visible_count(sluice: &mining_sim::Sluice) -> usize {
    let count = sluice.water.len() + sluice.outflow.len();
    count.min(MAX_SPLATS)
}

pub fn sync(
    game: Res<crate::app::Game>,
    time: Res<Time>,
    detail: Res<crate::environment::Detail>,
    mut splats: ResMut<WaterSplats>,
    mut worker: NonSendMut<crate::water_surface_worker::SurfaceWorker>,
) {
    let wall = time.elapsed_secs_f64();
    let s = &game.session.sluice;
    let presentation_key = (
        game.simulation_generation,
        s.cleanups,
        s.placement.map(|p| (p.inlet, p.yaw, p.slope_deg)),
    );
    if splats.presentation_key != Some(presentation_key) {
        splats.presentation = Default::default();
        splats.presentation_key = Some(presentation_key);
        splats.fluid_revision = None;
        splats.last_sample_time = -1.;
        splats.surface_epoch = splats.surface_epoch.wrapping_add(1);
        splats.data = Arc::new(Vec::new());
        splats.revision = splats.revision.wrapping_add(1);
    }
    let key = game.fluid_revision();
    while let Some(surface) = worker.poll() {
        if surface.epoch != splats.surface_epoch {
            continue;
        }
        splats
            .presentation
            .push(surface.sim, surface.wall, surface.data);
        splats.builds += 1;
        splats.build_ms += surface.ms;
        splats.max_build_ms = splats.max_build_ms.max(surface.ms);
    }
    if worker.can_submit()
        && (splats.fluid_revision != Some(key)
            || (s.water.is_empty()
                && s.outflow.is_empty()
                && game.session.sim_time > splats.last_sample_time))
    {
        let started = std::time::Instant::now();
        splats.fluid_revision = Some(key);
        splats.last_sample_time = game.session.sim_time;
        worker.submit(splats.surface_epoch, game.session.sim_time, wall, s);
        let ms = started.elapsed().as_secs_f64() * 1000.;
        splats.captures += 1;
        splats.capture_ms += ms;
        splats.max_capture_ms = splats.max_capture_ms.max(ms);
    }
    if let Some(data) = splats.presentation.advance(wall, game.session.paused) {
        splats.revision = splats.revision.wrapping_add(1);
        splats.data = data;
    }
    splats.sim_time = splats.presentation.time as f32;
    let site = &game.session.site;
    let key = (
        game.simulation_generation,
        site.hydraulic.steps,
        site.revision,
        detail.revision,
    );
    if splats.terrain_key != Some(key) {
        splats.terrain_key = Some(key);
        splats.terrain_revision = splats.terrain_revision.wrapping_add(1);
        splats.terrain_data = Arc::new(crate::terrain_water::instances(site, &detail));
    }
}

pub(crate) fn samples(sluice: &mining_sim::Sluice) -> Vec<(u64, InstanceData)> {
    let Some(p) = sluice.placement else {
        return Vec::new();
    };
    let count = sluice.water.len() + sluice.outflow.len();
    let weight = count as f32 / count.clamp(1, MAX_SPLATS) as f32;
    let radius = sluice.profile.cell() * 0.85 * weight.cbrt();
    let slurry = mining_sim::slurry::SlurryField::new(sluice);
    let outlet_extinction = slurry.outlet_extinction(sluice);
    // Select IDs before evaluating the optical field or transforming positions.
    let mut selected: Vec<_> = sluice
        .water
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id, false, i))
        .chain(
            sluice
                .outflow
                .iter()
                .enumerate()
                .map(|(i, p)| (p.id, true, i)),
        )
        .collect();
    // Bottom-k hashed identities bound cost without reselecting the whole
    // surface when one particle exits and compacts the simulation vector.
    if selected.len() > MAX_SPLATS {
        selected.select_nth_unstable_by_key(MAX_SPLATS, |p| sample_rank(p.0));
        selected.truncate(MAX_SPLATS);
    }
    selected.sort_unstable_by_key(|p| p.0);
    let sampled: Vec<_> = selected
        .into_iter()
        .map(|(id, outlet, index)| {
            if outlet {
                let water = &sluice.outflow[index];
                (
                    id,
                    Vec3::from_array(water.position.to_array()),
                    water.velocity,
                    2.,
                    outlet_extinction,
                )
            } else {
                let water = &sluice.water[index];
                let world = p.to_world(sluice.grid_to_local(water.position));
                (
                    id,
                    Vec3::from_array(world.to_array()),
                    sluice.grid_velocity_to_world(water.velocity),
                    0.,
                    slurry.sample(water.position).1,
                )
            }
        })
        .filter(|(_, p, v, _, _)| p.is_finite() && v.is_finite())
        .collect();
    let points: Vec<_> = sampled.iter().map(|(_, p, _, _, _)| *p).collect();
    let kernels = crate::fluid_surface::kernels(&points, sluice.profile.cell());
    sampled
        .into_iter()
        .zip(kernels)
        .map(|((id, _p, v, kind, extinction), kernel)| {
            let c = kernel.center;
            // Overlapping bulk kernels reconstruct a continuous surface. A
            // detached sample instead uses its represented water volume, so
            // millimetre spray does not inherit a two-centimetre bulk silhouette.
            // The accepted outlet reconstruction retains its existing support.
            let radius = if kind == 0. {
                // A wider bulk reconstruction joins neighbouring crest samples.
                // shape.x below renormalizes the kernel to its represented volume;
                // detached spray and the accepted outlet keep their own support.
                let radius = radius * (1.25 / 0.85);
                let drop = (sluice.profile.particle_m3() * weight as f64 * 3.
                    / (4. * std::f64::consts::PI))
                    .cbrt() as f32;
                drop.min(radius) + (radius - drop.min(radius)) * kernel.bulk
            } else {
                radius
            };
            let axis = |v: Vec3, normal: f32| [v.x, v.y, v.z, normal];
            (
                id,
                InstanceData {
                    center_radius: [c.x, c.y, c.z, radius],
                    velocity: [v.x, v.y, v.z, 0.],
                    shape: [
                        (sluice.profile.particle_m3() * weight as f64
                            / (4. / 3.
                                * std::f64::consts::PI
                                * (radius as f64).powi(3)
                                * kernel.axes[0].length() as f64
                                * kernel.axes[1].length() as f64
                                * kernel.axes[2].length() as f64)) as f32,
                        extinction,
                        kind,
                        if kind == 0. { kernel.planarity } else { 0. },
                    ],
                    axis_x: axis(kernel.axes[0], kernel.normal.x),
                    axis_y: axis(kernel.axes[1], kernel.normal.y),
                    axis_z: axis(kernel.axes[2], kernel.normal.z),
                    slope_x: [0.; 4],
                    slope_z: [0.; 4],
                    flow_z: [0.; 4],
                    ..Default::default()
                },
            )
        })
        .collect()
}

fn sample_rank(mut id: u64) -> u64 {
    id = (id ^ (id >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    id = (id ^ (id >> 27)).wrapping_mul(0x94d049bb133111eb);
    id ^ (id >> 31)
}

#[cfg(test)]
fn instances(sluice: &mining_sim::Sluice) -> Vec<InstanceData> {
    samples(sluice).into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "bounded render profiling; run explicitly with --ignored --nocapture"]
    fn profile_surface_reconstruction_and_correspondence() {
        let mut session = crate::demo::session().unwrap();
        session.sluice.ensure_water_ids();
        let previous = samples(&session.sluice);
        for p in &mut session.sluice.water {
            p.position += p.velocity * (1. / 120.);
        }
        let mut rebuild = Vec::new();
        let mut pairing = Vec::new();
        let mut fingerprint = 0u64;
        for _ in 0..16 {
            let start = std::time::Instant::now();
            let current = samples(std::hint::black_box(&session.sluice));
            rebuild.push(start.elapsed().as_secs_f64() * 1000.);
            let start = std::time::Instant::now();
            let paired = crate::water_presentation::pair_particles(&previous, &current);
            pairing.push(start.elapsed().as_secs_f64() * 1000.);
            // Deterministic byte fingerprint permits an exact-output comparison
            // across implementations; timing has no flaky pass threshold.
            let mut hash = 0xcbf29ce484222325u64;
            for byte in bytemuck::cast_slice::<InstanceData, u8>(&paired) {
                hash = (hash ^ *byte as u64).wrapping_mul(0x100000001b3);
            }
            if fingerprint != 0 {
                assert_eq!(hash, fingerprint);
            }
            fingerprint = hash;
            std::hint::black_box(paired);
        }
        rebuild.sort_by(f64::total_cmp);
        pairing.sort_by(f64::total_cmp);
        let report = serde_json::json!({"water":session.sluice.water.len(),"grains":session.sluice.grains.len(),"samples":previous.len(),"iterations":rebuild.len(),"rebuild_median_ms":rebuild[rebuild.len()/2],"pairing_median_ms":pairing[pairing.len()/2],"output_fnv1a64":format!("{fingerprint:016x}")});
        println!("SURFACE PROFILE: {report}");
        if let Some(path) = std::env::var_os("SLUICE_SURFACE_PROFILE") {
            std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        }
    }
    #[test]
    fn accepted_outlet_uses_no_surface_normal_override() {
        let mut session = mining_sim::Session::new();
        session.sluice.placement = Some(mining_sim::Placement::suggested(&session.site));
        session.sluice.outflow = vec![
            mining_sim::outflow::OutflowParticle {
                id: 0,
                position: mining_sim::Vec3::new(14., 1., 20.),
                velocity: mining_sim::Vec3::Z
            };
            50
        ];
        let before = session.sluice.accounted_water_m3();
        let data = instances(&session.sluice);
        assert!(data.iter().all(|p| p.shape[2] == 2. && p.shape[3] == 0.));
        assert_eq!(before, session.sluice.accounted_water_m3());
    }
    #[test]
    fn detached_water_stays_three_dimensional_and_rendering_is_bounded() {
        use mining_sim::{
            sluice::{Placement, WaterParticle, FLUID_CELL},
            Session, Vec3,
        };
        let mut session = Session::default();
        let sluice = &mut session.sluice;
        let p = Placement::suggested(&session.site);
        sluice.placement = Some(p);
        let y = (sluice.floor_cells() + 1) as f32 * FLUID_CELL;
        let particle = WaterParticle {
            id: 0,
            position: Vec3::new(1.0, y + 0.3, 0.2),
            velocity: Vec3::X,
            affine: Default::default(),
        };
        sluice.water = vec![particle.clone(); MAX_SPLATS + 1];
        sluice.water[2].position.y += 0.4;
        sluice.ensure_water_ids();
        let result = instances(sluice);
        assert!(result.len() <= MAX_SPLATS);
        assert!(result
            .iter()
            .any(|q| (q.center_radius[1] - p.inlet.y - 0.7).abs() < 1e-5));
        assert!((result[0].center_radius[1] - p.inlet.y - 0.3).abs() < 1e-5);
        assert_eq!(sluice.water.len(), MAX_SPLATS + 1);
        sluice.water.clear();
        assert!(instances(sluice).is_empty());
    }
    #[test]
    fn joined_bulk_and_detached_spray_retain_represented_water_volume() {
        use mining_sim::{sluice::WaterParticle, Session, Vec3};
        let mut session = Session::new();
        session.sluice.placement = Some(mining_sim::Placement::suggested(&session.site));
        session.sluice.water = (0..125)
            .map(|i| WaterParticle {
                id: i as u64 + 1,
                position: Vec3::new(
                    0.5 + (i % 5) as f32 * 0.003,
                    0.2 + ((i / 5) % 5) as f32 * 0.003,
                    0.2 + (i / 25) as f32 * 0.003,
                ),
                velocity: Vec3::X,
                affine: Default::default(),
            })
            .collect();
        session.sluice.water.push(WaterParticle {
            id: 126,
            position: Vec3::new(1., 0.5, 0.2),
            velocity: Vec3::Y,
            affine: Default::default(),
        });
        let before = serde_json::to_string(&session.sluice).unwrap();
        let data = samples(&session.sluice);
        let parcel = session.sluice.profile.particle_m3();
        let mut reconstructed = 0.;
        for (_, kernel) in &data {
            let axis_length = |axis: [f32; 4]| {
                (axis[0] as f64 * axis[0] as f64
                    + axis[1] as f64 * axis[1] as f64
                    + axis[2] as f64 * axis[2] as f64)
                    .sqrt()
            };
            let integral = 4. / 3.
                * std::f64::consts::PI
                * (kernel.center_radius[3] as f64).powi(3)
                * axis_length(kernel.axis_x)
                * axis_length(kernel.axis_y)
                * axis_length(kernel.axis_z)
                * kernel.shape[0] as f64;
            assert!((integral / parcel - 1.).abs() < 1e-6);
            reconstructed += integral;
        }
        assert!((reconstructed / (126. * parcel) - 1.).abs() < 1e-6);
        let drop = &data.iter().find(|(id, _)| *id == 126).unwrap().1;
        let physical_radius = (parcel * 3. / (4. * std::f64::consts::PI)).cbrt();
        assert!((drop.center_radius[3] as f64 / physical_radius - 1.).abs() < 1e-6);
        assert!(data[0].1.center_radius[3] > drop.center_radius[3] * 2.);
        assert_eq!(before, serde_json::to_string(&session.sluice).unwrap());
    }

    #[test]
    fn removing_one_particle_keeps_all_other_sampled_identities() {
        use mining_sim::{sluice::WaterParticle, Session, Vec3};
        let mut s = Session::new();
        s.sluice.placement = Some(mining_sim::Placement::suggested(&s.site));
        s.sluice.water = (1..=MAX_SPLATS + 16)
            .map(|id| WaterParticle {
                id: id as u64,
                position: Vec3::new(0.4, 0.2, 0.2),
                velocity: Vec3::ZERO,
                affine: Default::default(),
            })
            .collect();
        let a: std::collections::HashSet<_> = samples(&s.sluice).into_iter().map(|p| p.0).collect();
        let removed = s.sluice.water.remove(1).id;
        let b: std::collections::HashSet<_> = samples(&s.sluice).into_iter().map(|p| p.0).collect();
        assert!(a.difference(&b).all(|id| *id == removed));
        assert!(b.difference(&a).count() <= 1);
    }
}

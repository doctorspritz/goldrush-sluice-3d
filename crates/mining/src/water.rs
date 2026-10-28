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

fn fatal_pipeline_error(error: &bevy::shader::PipelineCacheError) -> bool {
    // Match Bevy's retryable asset states. A queued embedded shader may be
    // extracted one frame after the pipeline; this is not a compile failure.
    !matches!(
        error,
        bevy::shader::PipelineCacheError::ShaderNotLoaded(_)
            | bevy::shader::PipelineCacheError::ShaderImportNotYetAvailable
    )
}

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
    queue_ms: f64,
    max_queue_ms: f64,
    coalesced_requests: u64,
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
    debug_mode: u32,
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
        points.into_iter().take(128).map(|(height,c,downstream,bulk_support)|serde_json::json!({"world":[c.x,c.y,c.z],"downstream_m":downstream,"height_m":height,"radius_m":c.w,"bulk_support":bulk_support})).collect()
    }
    pub fn presentation_report(&self) -> serde_json::Value {
        serde_json::json!({"surface_builds": self.builds,
            "mean_surface_build_ms": self.build_ms / self.builds.max(1) as f64,
            "max_surface_build_ms": self.max_build_ms,
            "mean_surface_queue_ms": self.queue_ms/self.builds.max(1) as f64,
            "max_surface_queue_ms": self.max_queue_ms,
            "coalesced_surface_requests": self.coalesced_requests,
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
            queue_ms: 0.,
            max_queue_ms: 0.,
            coalesced_requests: 0,
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
            debug_mode: 0,
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
    density_near: TextureView,
    density_far: TextureView,
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
    visible_id: CachedRenderPipelineId,
    visible_optical_x: CachedRenderPipelineId,
    visible_optical_y: CachedRenderPipelineId,
    reconstruct_id: CachedRenderPipelineId,
    geometry_x: CachedRenderPipelineId,
    geometry_y: CachedRenderPipelineId,
    optical_x: CachedRenderPipelineId,
    optical_y: CachedRenderPipelineId,
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
                (
                    4,
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
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
                texture_2d(TextureSampleType::Float { filterable: false }),
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
                        blend: if fragment == "particle_thickness"
                            || fragment == "particle_visible_thickness"
                        {
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
                    if fragment == "particle_depth"
                        || fragment == "particle_thickness"
                        || fragment == "particle_visible_thickness"
                    {
                        targets.push(Some(ColorTargetState {
                            format: TextureFormat::Rgba16Float,
                            blend: if fragment == "particle_thickness"
                                || fragment == "particle_visible_thickness"
                            {
                                targets[0].as_ref().unwrap().blend
                            } else {
                                None
                            },
                            write_mask: ColorWrites::ALL,
                        }));
                    }
                    if fragment == "particle_thickness" {
                        // Four RGBA16F targets fit the 32-byte attachment budget.
                        targets.extend([targets[1].clone(), targets[1].clone()]);
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
    let visible_id = make(
        depth_shader.clone(),
        &thickness_particles,
        "particle_vertex",
        "particle_visible_thickness",
        TextureFormat::Rgba16Float,
        None,
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
        TextureFormat::Rgba32Float,
        None,
    );
    let geometry_x = make(
        blur_shader.clone(),
        &blur,
        "fullscreen",
        "geometry_x",
        TextureFormat::Rgba32Float,
        None,
    );
    let geometry_y = make(
        blur_shader.clone(),
        &blur,
        "fullscreen",
        "geometry_y",
        TextureFormat::Rgba32Float,
        None,
    );
    let optical_x = make(
        blur_shader.clone(),
        &blur,
        "fullscreen",
        "optical_x",
        TextureFormat::Rgba32Float,
        None,
    );
    let optical_y = make(
        blur_shader.clone(),
        &blur,
        "fullscreen",
        "optical_y",
        TextureFormat::Rgba32Float,
        None,
    );
    let visible_optical_x = make(
        blur_shader.clone(),
        &blur,
        "fullscreen",
        "visible_optical_x",
        TextureFormat::Rgba32Float,
        None,
    );
    let visible_optical_y = make(
        blur_shader.clone(),
        &blur,
        "fullscreen",
        "visible_optical_y",
        TextureFormat::Rgba32Float,
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
        visible_id,
        visible_optical_x,
        visible_optical_y,
        reconstruct_id,
        geometry_x,
        geometry_y,
        optical_x,
        optical_y,
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
                water.debug_mode as f32,
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
            ping: texture("fluid blur ping", TextureFormat::Rgba32Float, size),
            pong: texture("fluid blur pong", TextureFormat::Rgba32Float, size),
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
            density_near: texture("near interface density", TextureFormat::Rgba16Float, size),
            density_far: texture("far interface density", TextureFormat::Rgba16Float, size),
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
            settings.visible_id,
            settings.visible_optical_x,
            settings.visible_optical_y,
            settings.reconstruct_id,
            settings.geometry_x,
            settings.geometry_y,
            settings.optical_x,
            settings.optical_y,
            settings.compose_id,
        ] {
            if let CachedPipelineState::Err(error) = cache.get_render_pipeline_state(id) {
                if !fatal_pipeline_error(error) {
                    return Ok(());
                }
                if health.swap(2, Ordering::AcqRel) != 2 {
                    eprintln!("WATER RENDER FAILED: {error}");
                }
                return Ok(());
            }
        }
        let (
            Some(depth),
            Some(thickness),
            Some(visible),
            Some(visible_optical_x),
            Some(visible_optical_y),
            Some(reconstruct),
            Some(geometry_x),
            Some(geometry_y),
            Some(optical_x),
            Some(optical_y),
            Some(compose),
        ) = (
            cache.get_render_pipeline(settings.depth_id),
            cache.get_render_pipeline(settings.thickness_id),
            cache.get_render_pipeline(settings.visible_id),
            cache.get_render_pipeline(settings.visible_optical_x),
            cache.get_render_pipeline(settings.visible_optical_y),
            cache.get_render_pipeline(settings.reconstruct_id),
            cache.get_render_pipeline(settings.geometry_x),
            cache.get_render_pipeline(settings.geometry_y),
            cache.get_render_pipeline(settings.optical_x),
            cache.get_render_pipeline(settings.optical_y),
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
                &target.raw,
            )),
        );
        let visible_group = device.create_bind_group(
            "fluid visible column inputs",
            &settings.thickness_particles,
            &BindGroupEntries::sequential((
                particles.buffer.as_entire_binding(),
                uniform.clone(),
                &target.raw,
                scene_depth.view(),
                &target.pong,
            )),
        );
        let blur_group = |texture: &TextureView| {
            device.create_bind_group(
                "fluid blur inputs",
                &settings.blur,
                &BindGroupEntries::sequential((
                    texture,
                    uniform.clone(),
                    &target.thickness,
                    &target.density_near,
                    &target.density_far,
                )),
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
                    color_attachments: &[
                        color(&target.thickness),
                        color(&target.normal_sums),
                        color(&target.density_near),
                        color(&target.density_far),
                    ],
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
        // Establish geometry, then exchange optical moments over that surface.
        // Four crest-preserving shape filters and two optical exchanges
        // share six draws and the existing targets. The outlet keeps exactly
        // its original four filters; the optical-only draws copy it unchanged.
        for (index, pipeline) in
            crate::water_passes::surface_passes(geometry_x, geometry_y, optical_x, optical_y)
                .into_iter()
                .enumerate()
        {
            let (group, output) = if index % 2 == 0 {
                (&pong_group, &target.ping)
            } else {
                (&ping_group, &target.pong)
            };
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
        // The raw column belongs to the particle envelope. Once the joined
        // surface is fixed, reintegrate each kernel from that visible surface,
        // rather than retaining the average colour of hidden front water.
        // Reuse the thickness/ping/pong targets; no allocation grows.
        {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid visible optical interval"),
                    color_attachments: &[color(&target.thickness), color(&target.normal_sums)],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            pass.set_pipeline(visible);
            pass.set_bind_group(0, &visible_group, &[]);
            pass.draw(0..6, 0..particles.count);
        }
        for (pipeline, group, output) in [
            (visible_optical_x, &pong_group, &target.ping),
            (visible_optical_y, &ping_group, &target.pong),
        ] {
            let mut pass = context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("fluid visible optical exchange"),
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
/// A compiled pipeline may have drawn an empty pre-restore snapshot. Evidence
/// capture must also wait for the current epoch's water and, when frozen, its
/// exact clock. This does not delay or alter authoritative simulation.
pub fn capture_surface_ready(
    splats: &WaterSplats,
    sluice: &mining_sim::Sluice,
    sim: f64,
    frozen: bool,
) -> bool {
    surface_ready(splats)
        && (visible_count(sluice) == 0
            || (splats.presentation.paired > 0 && (!frozen || splats.presentation.time == sim)))
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
    capture: Res<crate::motion_capture::MotionCapture>,
    time: Res<Time>,
    detail: Res<crate::environment::Detail>,
    mut splats: ResMut<WaterSplats>,
    mut worker: NonSendMut<crate::water_surface_worker::SurfaceWorker>,
) {
    splats.debug_mode = capture.water_debug;
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
        splats.queue_ms += surface.queue_ms;
        splats.max_queue_ms = splats.max_queue_ms.max(surface.queue_ms);
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
    splats.coalesced_requests = worker.coalesced();
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

// Immutable for one reconstruction snapshot. Compute the original placement
// arithmetic and finite stepped wall geometry once, then reuse their values
// for every render sample. Rebuilding on each call also covers moved/tilted
// equipment without a persistent cache or invalidation protocol.
struct SurfaceGeometry {
    inlet: mining_sim::Vec3,
    forward: mining_sim::Vec3,
    right: mining_sim::Vec3,
    slope: f32,
    floor_offset: f32,
    half_width: f32,
    cell: f32,
    wall_tops: Vec<f32>,
    lateral: [f32; 4],
}
impl SurfaceGeometry {
    fn new(sluice: &mining_sim::Sluice, placement: mining_sim::Placement) -> Self {
        let config = mining_physics::geometry::config(sluice);
        let half_width = sluice.profile.width() * 0.5;
        let right = placement.right();
        let plane = right / half_width;
        Self {
            inlet: placement.inlet,
            forward: placement.forward(),
            right,
            slope: placement.slope_deg.to_radians().tan(),
            floor_offset: (sluice.floor_cells() + 1) as f32 * config.cell_size,
            half_width,
            cell: config.cell_size,
            wall_tops: (0..config.grid_width)
                .map(|column| {
                    (config.floor_height_at(column) + config.riffle_height + config.wall_margin + 1)
                        as f32
                        * config.cell_size
                })
                .collect(),
            lateral: [plane.x, plane.y, plane.z, -plane.dot(placement.inlet)],
        }
    }
    fn world(&self, position: mining_sim::Vec3) -> mining_sim::Vec3 {
        // Preserve the order in grid_to_local followed by Placement::to_world.
        // Algebraically cancelling their slopes would change rounded heights.
        let local = mining_sim::Vec3::new(
            position.x,
            position.y - self.floor_offset + position.x * self.slope,
            position.z - self.half_width - self.cell,
        );
        self.inlet
            + self.forward * local.x
            + mining_sim::Vec3::Y * (local.y - local.x * self.slope)
            + self.right * local.z
    }
    fn velocity(&self, velocity: mining_sim::Vec3) -> mining_sim::Vec3 {
        self.forward * velocity.x + mining_sim::Vec3::Y * velocity.y + self.right * velocity.z
    }
    fn below_wall_top(&self, world: Vec3) -> bool {
        let origin = Vec3::from_array(self.inlet.to_array());
        let x = (world - origin).dot(Vec3::from_array(self.forward.to_array()));
        let column = (x / self.cell)
            .floor()
            .clamp(0., (self.wall_tops.len() - 1) as f32) as usize;
        let height = world.y - origin.y + self.floor_offset;
        height <= self.wall_tops[column]
    }
}

pub(crate) fn samples(sluice: &mining_sim::Sluice) -> Vec<(u64, InstanceData)> {
    let Some(p) = sluice.placement else {
        return Vec::new();
    };
    #[cfg(test)]
    let optical_started = std::time::Instant::now();
    let geometry = SurfaceGeometry::new(sluice, p);
    let count = sluice.water.len() + sluice.outflow.len();
    let weight = count as f32 / count.clamp(1, MAX_SPLATS) as f32;
    let radius =
        sluice.profile.cell() * 0.85 * (weight * sluice.water_sampling().volume_scale()).cbrt();
    let mut slurry = mining_sim::slurry::SlurryField::new(sluice);
    let outlet_extinction = slurry.outlet_extinction(sluice);
    // Preserve the accepted outlet sample above. Airborne feed must not seed
    // a broad suspended-water plume or disappear as if it were already wet.
    let wet = mining_physics::geometry::grain_wetness(sluice);
    if sluice
        .grains
        .iter()
        .zip(&wet)
        .any(|(g, wet)| g.kind == mining_sim::GrainKind::Sand && !g.resting && !wet)
    {
        slurry = mining_sim::slurry::SlurryField::new_with_suspension(sluice, &wet);
    }
    slurry.reconstruct_channel();
    #[cfg(test)]
    crate::fluid_surface::profile::record_stage("optical_field", optical_started.elapsed());
    #[cfg(test)]
    let selection_started = std::time::Instant::now();
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
                let world = geometry.world(water.position);
                (
                    id,
                    Vec3::from_array(world.to_array()),
                    geometry.velocity(water.velocity),
                    0.,
                    slurry.sample_channel(water.position).1,
                )
            }
        })
        .filter(|(_, p, v, _, _)| p.is_finite() && v.is_finite())
        .collect();
    #[cfg(test)]
    crate::fluid_surface::profile::record_stage(
        "selection_and_optical_sampling",
        selection_started.elapsed(),
    );
    let points: Vec<_> = sampled.iter().map(|(_, p, _, _, _)| *p).collect();
    let kernels = crate::fluid_surface::kernels(
        &points,
        sluice.profile.cell(),
        sluice.water_particle_m3() * weight as f64,
        &sampled
            .iter()
            .map(|(_, _, _, kind, _)| *kind == 0.)
            .collect::<Vec<_>>(),
    );
    #[cfg(test)]
    let instances_started = std::time::Instant::now();
    let instances = sampled
        .into_iter()
        .zip(kernels)
        .map(|((id, _p, v, kind, extinction), kernel)| {
            let c = if kind == 0. {
                kernel.channel_center
            } else {
                kernel.center
            };
            let axes = if kind == 0. {
                #[cfg(test)]
                crate::fluid_surface::profile::record_kernel(kernel);
                kernel.channel_axes()
            } else {
                kernel.axes
            };
            // Connected water keeps its reconstruction support. Isolated
            // spray, including the outlet, uses the represented-water radius
            // instead of inheriting the broad bulk envelope. Normalize optical
            // density below by the actual support determinant in either case.
            let support = if kind == 0. {
                // Channel density reconstruction needs the existing support;
                // changing outlet drops must not widen riffle crests.
                radius * (1.0 / 0.85)
            } else {
                radius
            };
            let drop = (sluice.water_particle_m3() * weight as f64 * 3.
                / (4. * std::f64::consts::PI))
                .cbrt() as f32;
            // Occupancy alone cannot distinguish a dilute falling sheet from
            // isolated spray. The existing covariance stencil resolves shape
            // at twelve neighbours. Preserve its broad stream support there;
            // smoothly reduce only the unresolved, nearly isolated droplets.
            let connected = ((kernel.neighbour_count - 1.) / 11.).clamp(0., 1.);
            let support_weight = if kind == 2. {
                kernel
                    .bulk
                    .max(connected * connected * (3. - 2. * connected))
            } else {
                kernel.bulk
            };
            let radius = if kind == 2. && support_weight == 1. {
                support
            } else {
                drop.min(support) + (support - drop.min(support)) * support_weight
            };
            let axis = |v: Vec3, normal: f32| [v.x, v.y, v.z, normal];
            (
                id,
                InstanceData {
                    center_radius: [c.x, c.y, c.z, radius],
                    velocity: [v.x, v.y, v.z, 0.],
                    shape: [
                        (sluice.water_particle_m3() * weight as f64
                            / (4. / 3.
                                * std::f64::consts::PI
                                * (radius as f64).powi(3)
                                * axes[0].length() as f64
                                * axes[1].length() as f64
                                * axes[2].length() as f64)) as f32,
                        extinction,
                        kind,
                        if kind == 0. { kernel.bulk } else { 0. },
                    ],
                    axis_x: axis(axes[0], kernel.normal.x),
                    axis_y: axis(axes[1], kernel.normal.y),
                    axis_z: axis(axes[2], kernel.normal.z),
                    slope_x: [0.; 4],
                    // The lateral slab clips reconstruction at the physical
                    // inner walls. Reuse unused channel fields/stride; outlet
                    // and terrain attributes keep their existing meanings.
                    slope_z: if kind == 0. && geometry.below_wall_top(_p) {
                        geometry.lateral
                    } else {
                        [0.; 4]
                    },
                    flow_z: [0.; 4],
                    ..Default::default()
                },
            )
        })
        .collect();
    #[cfg(test)]
    crate::fluid_surface::profile::record_stage(
        "instance_and_final_axes",
        instances_started.elapsed(),
    );
    instances
}

// Use the same finite, stepped wall height as collision/mesh geometry. A
// splash above the wall must remain free to render outside the lateral slab.
#[cfg(test)]
fn below_wall_top(sluice: &mining_sim::Sluice, world: Vec3) -> bool {
    let Some(p) = sluice.placement else {
        return false;
    };
    let config = mining_physics::geometry::config(sluice);
    let origin = Vec3::from_array(p.inlet.to_array());
    let x = (world - origin).dot(Vec3::from_array(p.forward().to_array()));
    let column = (x / config.cell_size)
        .floor()
        .clamp(0., (config.grid_width - 1) as f32) as usize;
    let top =
        (config.floor_height_at(column) + config.riffle_height + config.wall_margin + 1) as f32;
    let grid_height = world.y - origin.y + (sluice.floor_cells() + 1) as f32 * config.cell_size;
    grid_height <= top * config.cell_size
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
    fn pending_shader_assets_wait_but_real_compilation_errors_fail() {
        use bevy::shader::PipelineCacheError;
        assert!(!fatal_pipeline_error(&PipelineCacheError::ShaderNotLoaded(
            Handle::<Shader>::default().id()
        )));
        assert!(!fatal_pipeline_error(
            &PipelineCacheError::ShaderImportNotYetAvailable
        ));
        assert!(fatal_pipeline_error(
            &PipelineCacheError::CreateShaderModule("invalid shader".into())
        ));
    }

    #[test]
    #[ignore = "bounded render profiling; run explicitly with --ignored --nocapture"]
    fn profile_surface_reconstruction_and_correspondence() {
        let mut session = if let Some(path) = std::env::var_os("SLUICE_SURFACE_LOAD") {
            mining_sim::Session::load_json(&std::fs::read_to_string(path).unwrap()).unwrap()
        } else {
            crate::demo::session().unwrap()
        };
        session.sluice.ensure_water_ids();
        let previous = samples(&session.sluice);
        for p in &mut session.sluice.water {
            p.position += p.velocity * (1. / 120.);
        }
        let mut rebuild = Vec::new();
        let mut pairing = Vec::new();
        let mut fingerprint = 0u64;
        crate::fluid_surface::profile::begin_stages();
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
        let report = serde_json::json!({"water":session.sluice.water.len(),"grains":session.sluice.grains.len(),"samples":previous.len(),"iterations":rebuild.len(),"rebuild_median_ms":rebuild[rebuild.len()/2],"pairing_median_ms":pairing[pairing.len()/2],"output_fnv1a64":format!("{fingerprint:016x}"),"stages":crate::fluid_surface::profile::finish_stages()});
        println!("SURFACE PROFILE: {report}");
        if let Some(path) = std::env::var_os("SLUICE_SURFACE_PROFILE") {
            std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        }
    }
    #[test]
    fn capture_waits_for_restored_water_and_frozen_clock_without_blocking_dry_scenes() {
        let mut s = mining_sim::Sluice::default();
        let mut water = WaterSplats::default();
        assert!(!capture_surface_ready(&water, &s, 8., true));
        water.health.store(1, Ordering::Release);
        assert!(capture_surface_ready(&water, &s, 8., true));
        s.placement = Some(mining_sim::Placement {
            inlet: mining_sim::Vec3::ZERO,
            yaw: 0.,
            slope_deg: 10.,
        });
        s.outflow.push(mining_sim::outflow::OutflowParticle {
            id: 1,
            position: mining_sim::Vec3::ONE,
            velocity: -mining_sim::Vec3::Y,
        });
        // A real empty render result must not qualify a restored wet shift.
        water.presentation.push(0., 0., Arc::new(Vec::new()));
        water.presentation.advance(0., true);
        assert!(!capture_surface_ready(&water, &s, 8., false));
        let particles = samples(&s);
        let data = Arc::new(crate::water_presentation::pair_particles(
            &particles, &particles,
        ));
        water.presentation.push(7., 1., data.clone());
        water.presentation.advance(1., true);
        assert!(capture_surface_ready(&water, &s, 8., false));
        assert!(!capture_surface_ready(&water, &s, 8., true));
        water.presentation.push(8., 2., data);
        water.presentation.advance(2., true);
        assert!(capture_surface_ready(&water, &s, 8., true));
        water.presentation = Default::default();
        assert!(!capture_surface_ready(&water, &s, 8., true));
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
    fn outlet_droplets_use_owned_water_volume_without_inflating_dense_flow() {
        use mining_sim::{outflow::OutflowParticle, sluice::WaterSampling, Vec3};
        for sampling in [
            WaterSampling::Eight,
            WaterSampling::Four,
            WaterSampling::Two,
        ] {
            let mut s = mining_sim::Session::new();
            s.sluice.set_water_sampling(sampling).unwrap();
            s.sluice.placement = Some(mining_sim::Placement::suggested(&s.site));
            s.sluice.outflow = (0..125)
                .map(|i| OutflowParticle {
                    id: i + 1,
                    position: Vec3::new(
                        14. + (i % 5) as f32 * 0.003,
                        1. + ((i / 5) % 5) as f32 * 0.003,
                        20. + (i / 25) as f32 * 0.003,
                    ),
                    velocity: Vec3::new(0.3, -1., 0.),
                })
                .collect();
            s.sluice.outflow.push(OutflowParticle {
                id: 126,
                position: Vec3::new(16., 0.5, 20.),
                velocity: -Vec3::Y,
            });
            let before = serde_json::to_string(&s.sluice).unwrap();
            let data = samples(&s.sluice);
            let volume = s.sluice.water_particle_m3();
            let radius = (volume * 3. / (4. * std::f64::consts::PI)).cbrt();
            let detached = &data.iter().find(|(id, _)| *id == 126).unwrap().1;
            assert!((f64::from(detached.center_radius[3]) / radius - 1.).abs() < 1e-6);
            assert_eq!(detached.axis_x[..3], [1., 0., 0.]);
            assert_eq!(detached.axis_y[..3], [0., 1., 0.]);
            assert_eq!(detached.axis_z[..3], [0., 0., 1.]);
            let bulk_radius = s.sluice.profile.cell() * 0.85 * sampling.volume_scale().cbrt();
            for (id, p) in data {
                if id != 126 {
                    assert_eq!(p.center_radius[3].to_bits(), bulk_radius.to_bits());
                }
                let length = |a: [f32; 4]| Vec3::new(a[0], a[1], a[2]).length() as f64;
                let integral = 4. / 3.
                    * std::f64::consts::PI
                    * f64::from(p.center_radius[3]).powi(3)
                    * length(p.axis_x)
                    * length(p.axis_y)
                    * length(p.axis_z)
                    * f64::from(p.shape[0]);
                assert!((integral / volume - 1.).abs() < 1e-6);
                assert_eq!((p.shape[2], p.shape[3]), (2., 0.));
            }
            assert_eq!(before, serde_json::to_string(&s.sluice).unwrap());
        }
    }

    #[test]
    fn dilute_connected_outlet_retains_stream_support_instead_of_becoming_spray() {
        use mining_sim::{outflow::OutflowParticle, Vec3};
        let mut s = mining_sim::Session::new();
        s.sluice.placement = Some(mining_sim::Placement::suggested(&s.site));
        s.sluice.outflow = (0..125)
            .map(|i| OutflowParticle {
                id: i + 1,
                position: Vec3::new(
                    (i % 5) as f32 * 0.012,
                    ((i / 5) % 5) as f32 * 0.012,
                    (i / 25) as f32 * 0.012,
                ),
                velocity: -Vec3::Y,
            })
            .collect();
        let points: Vec<_> = s
            .sluice
            .outflow
            .iter()
            .map(|p| bevy::prelude::Vec3::from_array(p.position.to_array()))
            .collect();
        let kernels = crate::fluid_surface::kernels(
            &points,
            s.sluice.profile.cell(),
            s.sluice.water_particle_m3(),
            &vec![false; points.len()],
        );
        let data = samples(&s.sluice);
        let mut dilute = 0;
        for (kernel, (_, instance)) in kernels.iter().zip(data) {
            if kernel.neighbour_count >= 12. && kernel.bulk < 0.5 {
                assert_eq!(
                    instance.center_radius[3].to_bits(),
                    (s.sluice.profile.cell() * 0.85).to_bits()
                );
                dilute += 1;
            }
        }
        assert!(dilute > 0, "control must contain dilute connected flow");
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
        let parcel = session.sluice.water_particle_m3();
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
    #[test]
    fn snapshot_geometry_retains_original_transforms_and_stepped_walls() {
        let mut sluice = crate::demo::session().unwrap().sluice;
        for profile in [
            mining_sim::sluice::HydraulicProfile::Legacy,
            mining_sim::sluice::HydraulicProfile::OriginalDemo,
        ] {
            sluice.profile = profile;
            for yaw in [0., 0.37, 1.4, -2.2] {
                for slope_deg in [3., 9.8, 17.] {
                    sluice.placement = Some(mining_sim::Placement {
                        inlet: mining_sim::Vec3::new(14.13, 2.37, -9.21),
                        yaw,
                        slope_deg,
                    });
                    let placement = sluice.placement.unwrap();
                    let geometry = SurfaceGeometry::new(&sluice, placement);
                    let right = placement.right() / (profile.width() * 0.5);
                    assert_eq!(
                        geometry.lateral.map(f32::to_bits),
                        [right.x, right.y, right.z, -right.dot(placement.inlet)].map(f32::to_bits)
                    );
                    for column in 0..geometry.wall_tops.len() {
                        for fraction in [-0.0001, 0.25, 1.0001] {
                            let position = mining_sim::Vec3::new(
                                (column as f32 + fraction) * geometry.cell,
                                geometry.wall_tops[column] + 0.001,
                                profile.width() * 0.7,
                            );
                            assert_eq!(
                                geometry.world(position).to_array().map(f32::to_bits),
                                placement
                                    .to_world(sluice.grid_to_local(position))
                                    .to_array()
                                    .map(f32::to_bits)
                            );
                            let velocity = position * 0.37 - mining_sim::Vec3::ONE;
                            assert_eq!(
                                geometry.velocity(velocity).to_array().map(f32::to_bits),
                                sluice
                                    .grid_velocity_to_world(velocity)
                                    .to_array()
                                    .map(f32::to_bits)
                            );
                            for offset in [-0.0001, 0., 0.0001] {
                                let world = placement.inlet
                                    + placement.forward() * position.x
                                    + mining_sim::Vec3::Y
                                        * (geometry.wall_tops[column] - geometry.floor_offset
                                            + offset);
                                let world = Vec3::from_array(world.to_array());
                                assert_eq!(geometry.below_wall_top(world), below_wall_top(&sluice, world),
                                    "profile={profile:?} yaw={yaw} slope={slope_deg} column={column} fraction={fraction} offset={offset}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reconstruction_clips_below_the_actual_wall_but_retains_above_wall_spray() {
        let mut sluice = crate::demo::session().unwrap().sluice;
        sluice.outflow.clear();
        sluice.water.truncate(1);
        let config = mining_physics::geometry::config(&sluice);
        let x = 0.65;
        let column = (x / config.cell_size) as usize;
        let top = (config.floor_height_at(column) + config.riffle_height + config.wall_margin + 1)
            as f32
            * config.cell_size;
        for (offset, clipped) in [(-0.001, true), (0.001, false)] {
            sluice.water[0].position = mining_sim::Vec3::new(
                x,
                top + offset,
                sluice.profile.width() * 0.5 + config.cell_size,
            );
            let data = samples(&sluice);
            assert_eq!(data[0].1.slope_z != [0.; 4], clipped);
        }
    }

    #[test]
    fn reduced_water_sampling_retains_optical_volume_and_slurry_dilution() {
        use mining_sim::sluice::{WaterParticle, WaterSampling};
        for sampling in [
            WaterSampling::Eight,
            WaterSampling::Four,
            WaterSampling::Two,
        ] {
            let mut s = mining_sim::Session::new();
            s.sluice.placement = Some(mining_sim::Placement::suggested(&s.site));
            s.sluice.set_water_sampling(sampling).unwrap();
            let count = 240 / (8 / sampling.markers_per_cell()) as usize;
            for id in 0..count {
                s.sluice.water.push(WaterParticle {
                    id: id as u64 + 1,
                    position: mining_sim::Vec3::new(
                        0.55 + (id % 10) as f32 * 0.005,
                        0.20 + (id / 30) as f32 * 0.005,
                        0.20 + (id % 3) as f32 * 0.005,
                    ),
                    velocity: mining_sim::Vec3::X * 0.4,
                    affine: Default::default(),
                });
            }
            let expected = s.sluice.stored_water_m3();
            let (projected, _) =
                mining_sim::slurry::SlurryField::new(&s.sluice).projected_volumes();
            assert!((projected - expected).abs() < 1e-14);
            let optical = samples(&s.sluice);
            assert_eq!(optical.len(), count);
            let integral: f64 = optical
                .iter()
                .map(|(_, v)| {
                    let axes = Mat3::from_cols(
                        Vec3::from_slice(&v.axis_x[..3]),
                        Vec3::from_slice(&v.axis_y[..3]),
                        Vec3::from_slice(&v.axis_z[..3]),
                    );
                    f64::from(v.shape[0]) * 4. / 3.
                        * std::f64::consts::PI
                        * f64::from(v.center_radius[3]).powi(3)
                        * f64::from(axes.determinant().abs())
                })
                .sum();
            assert!(
                (integral - expected).abs() < expected * 3e-6,
                "{sampling:?}: {integral} vs {expected}"
            );
        }
    }
}

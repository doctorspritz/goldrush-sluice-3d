//! Screen-space fluid from the authoritative FLIP particles: nearest particle
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
#[derive(Resource, Clone, ExtractResource)]
pub struct WaterSplats {
    revision: u64,
    data: Arc<Vec<InstanceData>>,
    health: Arc<AtomicU8>,
    clock: PresentationClock,
    turbidity: f32,
    terrain_data: Arc<Vec<InstanceData>>,
    terrain_revision: u64,
    terrain_key: Option<(u64, u64, u64)>,
    terrain_clock: PresentationClock,
    fluid_revision: Option<crate::app::FluidRevision>,
}
impl WaterSplats {
    pub fn terrain_count(&self) -> usize {
        self.terrain_data.len()
    }
}
impl Default for WaterSplats {
    fn default() -> Self {
        Self {
            revision: u64::MAX,
            data: Arc::new(Vec::new()),
            health: Arc::new(AtomicU8::new(0)),
            clock: PresentationClock::default(),
            turbidity: 0.0,
            terrain_data: Arc::new(Vec::new()),
            terrain_revision: 0,
            terrain_key: None,
            terrain_clock: PresentationClock::default(),
            fluid_revision: None,
        }
    }
}
/// Render-only advection spans at most one authoritative tick. It follows
/// measured worker cadence, stops on pause/reset, and never writes to Session.
#[derive(Clone, Default)]
struct PresentationClock {
    sim: Option<f64>,
    updated: f64,
    interval: f64,
    tick: f32,
    prediction: f32,
}
impl PresentationClock {
    fn sample(&mut self, sim: f64, wall: f64, paused: bool) {
        self.sample_interval(sim, wall, paused, mining_sim::FIXED_DT * 2.1);
    }
    fn sample_interval(&mut self, sim: f64, wall: f64, paused: bool, max_tick: f32) {
        if self.sim != Some(sim) {
            let delta = self.sim.map(|previous| sim - previous).unwrap_or(0.0);
            if delta > 0.0 && delta <= max_tick as f64 {
                self.interval = (wall - self.updated).max(mining_sim::FIXED_DT as f64);
                self.tick = delta as f32;
            } else {
                self.tick = 0.0;
            }
            self.updated = wall;
            self.sim = Some(sim);
        }
        self.prediction = if paused {
            0.0
        } else {
            self.tick * ((wall - self.updated) / self.interval.max(0.001)).clamp(0.0, 1.0) as f32
        };
    }
}
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct InstanceData {
    pub center_radius: [f32; 4],
    pub velocity: [f32; 4],
    pub shape: [f32; 4],
}

pub struct WaterSplatPlugin;
impl Plugin for WaterSplatPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "water_depth.wgsl");
        bevy::asset::embedded_asset!(app, "water_blur.wgsl");
        bevy::asset::embedded_asset!(app, "water_compose.wgsl");
        app.init_resource::<WaterSplats>()
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
    raw: TextureView,
    ping: TextureView,
    pong: TextureView,
    depth: TextureView,
    uniform: UniformBuffer<FluidUniforms>,
}
#[derive(Resource)]
struct FluidPipelines {
    particles: BindGroupLayout,
    blur: BindGroupLayout,
    compose: BindGroupLayout,
    depth_id: CachedRenderPipelineId,
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
    let blur = device.create_bind_group_layout(
        "fluid smoothing",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer::<FluidUniforms>(false),
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
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
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
        depth_shader,
        &particles,
        "particle_vertex",
        "particle_depth",
        TextureFormat::Rg32Float,
        Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::GreaterEqual,
            stencil: default(),
            bias: default(),
        }),
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
        blur,
        compose,
        depth_id,
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
                    (MAX_SPLATS * size_of::<InstanceData>()) as u64,
                    bytemuck::cast_slice(&data.terrain_data),
                );
            }
            buffer.terrain_count = data.terrain_data.len() as u32;
            buffer.terrain_revision = data.terrain_revision;
        }
    };
    if let Some(mut buffer) = existing {
        upload(&mut buffer);
        return;
    }
    let mut buffer = ParticleBuffer {
        buffer: device.create_buffer(&BufferDescriptor {
            label: Some("Bounded sluice and terrain render splats"),
            size: ((MAX_SPLATS + crate::terrain_water::MAX_SPLATS) * size_of::<InstanceData>())
                as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        count: 0,
        revision: u64::MAX,
        terrain_count: 0,
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
            sediment: Vec4::new(water.turbidity, water.terrain_clock.prediction, 0.0, 0.0),
            viewport: Vec4::new(full.x as f32, full.y as f32, 0.045, water.clock.prediction),
        };
        if let Some(mut target) = existing {
            if target.size == size {
                target.uniform.set(uniforms);
                target.uniform.write_buffer(&device, &queue);
                continue;
            }
        }
        let texture = |label, format| {
            device
                .create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: size.x,
                        height: size.y,
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
            size,
            raw: texture("fluid raw depth", TextureFormat::Rg32Float),
            ping: texture("fluid blur ping", TextureFormat::R32Float),
            pong: texture("fluid blur pong", TextureFormat::R32Float),
            depth: texture("fluid depth test", TextureFormat::Depth32Float),
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
        let (Some(depth), Some(blur_x), Some(blur_y), Some(compose)) = (
            cache.get_render_pipeline(settings.depth_id),
            cache.get_render_pipeline(settings.blur_x),
            cache.get_render_pipeline(settings.blur_y),
            cache.get_render_pipeline(settings.compose_id),
        ) else {
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
        let blur_group = |texture: &TextureView| {
            device.create_bind_group(
                "fluid blur inputs",
                &settings.blur,
                &BindGroupEntries::sequential((texture, uniform.clone())),
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
                    color_attachments: &[color(&target.raw)],
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
            pass.draw(
                0..6,
                MAX_SPLATS as u32..MAX_SPLATS as u32 + particles.terrain_count,
            );
        }
        // Two separable bilateral iterations smooth shared fluid normals, without
        // averaging across a depth jump or expanding water into empty pixels.
        for (pipeline, group, output) in [
            (blur_x, &raw_group, &target.ping),
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
/// droplets. A deterministic stride bounds upload/fragment cost at high volumes.
pub fn visible_count(sluice: &mining_sim::Sluice) -> usize {
    let stride = sluice.water.len().div_ceil(MAX_SPLATS).max(1);
    sluice.water.len().div_ceil(stride)
}

pub fn sync(game: Res<crate::app::Game>, time: Res<Time>, mut splats: ResMut<WaterSplats>) {
    splats.clock.sample(
        game.session.sim_time - game.session.sluice.fluid_clock as f64,
        time.elapsed_secs_f64(),
        game.session.paused,
    );
    let key = game.fluid_revision();
    if splats.fluid_revision != Some(key) {
        splats.fluid_revision = Some(key);
        splats.revision = game.snapshot_id;
        splats.data = Arc::new(instances(&game.session.sluice));
        let sluice = &game.session.sluice;
        let fines: f64 = sluice
            .grains
            .iter()
            .filter(|g| g.kind == mining_sim::GrainKind::Sand && !g.resting)
            .map(|g| g.solid_volume())
            .sum();
        splats.turbidity = (fines
            / (sluice.water.len() as f64 * sluice.profile.particle_m3()).max(1e-9))
        .clamp(0.0, 1.0) as f32;
    }
    let site = &game.session.site;
    splats.terrain_clock.sample_interval(
        site.hydraulic.steps as f64 * 3.0 * mining_sim::FIXED_DT as f64,
        time.elapsed_secs_f64(),
        game.session.paused,
        0.051,
    );
    let key = (
        game.simulation_generation,
        site.hydraulic.steps,
        site.revision,
    );
    if splats.terrain_key != Some(key) {
        splats.terrain_key = Some(key);
        splats.terrain_revision = splats.terrain_revision.wrapping_add(1);
        splats.terrain_data =
            Arc::new(crate::terrain_water::instances(site, game.session.sim_time));
    }
}

fn instances(sluice: &mining_sim::Sluice) -> Vec<InstanceData> {
    let Some(p) = sluice.placement else {
        return Vec::new();
    };
    let stride = sluice.water.len().div_ceil(MAX_SPLATS).max(1);
    let radius = sluice.profile.cell() * 0.6 * (stride as f32).cbrt();
    sluice
        .water
        .iter()
        .step_by(stride)
        .filter_map(|water| {
            if !water.position.is_finite() || !water.velocity.is_finite() {
                return None;
            }
            let world = p.to_world(sluice.grid_to_local(water.position));
            let v = p.forward() * water.velocity.x
                + mining_sim::Vec3::Y * water.velocity.y
                + p.right() * water.velocity.z;
            Some(InstanceData {
                center_radius: [world.x, world.y, world.z, radius],
                velocity: [v.x, v.y, v.z, 0.0],
                shape: [0.; 4],
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn presentation_uses_the_fluid_cadence_between_domain_ticks() {
        let mut clock = PresentationClock::default();
        let dt = mining_sim::FIXED_DT as f64 * 2.0;
        clock.sample(10.0, 0.0, false);
        clock.sample(10.0 + dt, 0.04, false);
        clock.sample(10.0 + dt, 0.06, false);
        assert!((clock.prediction - dt as f32 * 0.5).abs() < 1e-6);
        clock.sample(10.0 + dt, 10.0, false);
        assert!((clock.prediction - dt as f32).abs() < 1e-6);
    }
    #[test]
    fn presentation_advects_only_one_tick_and_stops_on_pause_or_restore() {
        let mut clock = PresentationClock::default();
        let dt = mining_sim::FIXED_DT as f64;
        clock.sample(20.0, 0.0, false);
        clock.sample(20.0 + dt, 0.12, false);
        clock.sample(20.0 + dt, 0.18, false);
        assert!((clock.prediction - mining_sim::FIXED_DT * 0.5).abs() < 1e-6);
        clock.sample(20.0 + dt, 100.0, false);
        assert_eq!(clock.prediction, mining_sim::FIXED_DT);
        clock.sample(20.0 + dt, 100.0, true);
        assert_eq!(clock.prediction, 0.0);
        clock.sample(0.0, 100.0, false);
        assert_eq!(clock.prediction, 0.0);
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
            position: Vec3::new(1.0, y + 0.3, 0.2),
            velocity: Vec3::X,
            affine: Default::default(),
        };
        sluice.water = vec![particle.clone(); MAX_SPLATS + 1];
        sluice.water[2].position.y += 0.4;
        let result = instances(sluice);
        assert!(result.len() <= MAX_SPLATS);
        assert!((result[1].center_radius[1] - result[0].center_radius[1] - 0.4).abs() < 1e-5);
        assert!((result[0].center_radius[1] - p.inlet.y - 0.3).abs() < 1e-5);
        assert_eq!(sluice.water.len(), MAX_SPLATS + 1);
        sluice.water.clear();
        assert!(instances(sluice).is_empty());
    }
}

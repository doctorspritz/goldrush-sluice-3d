//! Instanced sphere splats from live FLIP particles, following Bevy 0.17's
//! custom_shader_instancing render-command pattern. No per-particle entities.
use bevy::pbr::SetMeshViewBindingArrayBindGroup;
use bevy::{
    core_pipeline::core_3d::Transparent3d,
    ecs::{
        query::QueryItem,
        system::{lifetimeless::*, SystemParamItem},
    },
    mesh::{MeshVertexBufferLayoutRef, VertexBufferLayout},
    pbr::{
        MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup, SetMeshViewBindGroup,
    },
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::RenderDevice,
        sync_world::MainEntity,
        view::ExtractedView,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};

use std::sync::Arc;

const SHADER_PATH: &str = "embedded://mining/water_splats.wgsl";
// Rendering LOD only: every particle remains in the authoritative solver/save.
const MAX_SPLATS: usize = 65_536;

#[derive(Component, Clone)]
pub struct WaterSplats {
    revision: u64,
    data: Arc<Vec<InstanceData>>,
}

impl Default for WaterSplats {
    fn default() -> Self {
        Self {
            revision: 0,
            data: Arc::new(Vec::new()),
        }
    }
}

impl ExtractComponent for WaterSplats {
    type QueryData = &'static WaterSplats;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self> {
        Some(item.clone())
    }
}

pub struct WaterSplatPlugin;

impl Plugin for WaterSplatPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "water_splats.wgsl");
        app.add_plugins(ExtractComponentPlugin::<WaterSplats>::default());
        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
            .add_systems(RenderStartup, init_custom_pipeline)
            .add_systems(
                Render,
                (
                    queue_custom.in_set(RenderSystems::QueueMeshes),
                    prepare_instance_buffers.in_set(RenderSystems::PrepareResources),
                ),
            );
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct InstanceData {
    center_radius: [f32; 4],
    velocity: [f32; 4],
}

#[allow(
    clippy::too_many_arguments,
    reason = "Bevy render queue resources are independent system parameters"
)]
fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CustomPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    material_meshes: Query<(Entity, &MainEntity), With<WaterSplats>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let draw_custom = transparent_3d_draw_functions.read().id::<DrawCustom>();

    for (view, msaa) in &views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&view.retained_view_entity)
        else {
            continue;
        };

        let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());

        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        let rangefinder = view.rangefinder3d();
        for (entity, main_entity) in &material_meshes {
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*main_entity)
            else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };
            let key =
                view_key | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology());
            let pipeline = pipelines
                .specialize(&pipeline_cache, &custom_pipeline, key, &mesh.layout)
                .unwrap();
            transparent_phase.add(Transparent3d {
                entity: (entity, *main_entity),
                pipeline,
                draw_function: draw_custom,
                distance: rangefinder.distance_translation(&mesh_instance.translation),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
}

#[derive(Component)]
struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
    capacity: usize,
    revision: u64,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    mut query: Query<(Entity, &WaterSplats, Option<&mut InstanceBuffer>)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<bevy::render::renderer::RenderQueue>,
) {
    for (entity, data, existing) in &mut query {
        if let Some(mut buffer) = existing {
            if buffer.revision == data.revision {
                continue;
            }
            if buffer.capacity >= data.data.len() {
                if !data.data.is_empty() {
                    render_queue.write_buffer(&buffer.buffer, 0, bytemuck::cast_slice(&data.data));
                }
                buffer.length = data.data.len();
                buffer.revision = data.revision;
                continue;
            }
        }
        let capacity = data.data.len().max(1).next_power_of_two();
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("FLIP water splats"),
            size: (capacity * size_of::<InstanceData>()) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !data.data.is_empty() {
            render_queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data.data));
        }
        commands.entity(entity).insert(InstanceBuffer {
            buffer,
            length: data.data.len(),
            capacity,
            revision: data.revision,
        });
    }
}

#[derive(Resource)]
struct CustomPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

fn init_custom_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
) {
    commands.insert_resource(CustomPipeline {
        shader: asset_server.load(SHADER_PATH),
        mesh_pipeline: mesh_pipeline.clone(),
    });
}

impl SpecializedMeshPipeline for CustomPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        descriptor.vertex.shader = self.shader.clone();
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3, // shader locations 0-2 are taken up by Position, Normal and UV attributes
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size(),
                    shader_location: 4,
                },
            ],
        });
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        // Sphere impostors write their real curved depth. This makes intersecting
        // splats merge without per-particle sorting or alpha overdraw through walls.
        descriptor.primitive.cull_mode = None;
        if let Some(depth) = &mut descriptor.depth_stencil {
            depth.depth_write_enabled = true;
        }
        for target in descriptor
            .fragment
            .as_mut()
            .unwrap()
            .targets
            .iter_mut()
            .flatten()
        {
            target.blend = None;
        }
        Ok(descriptor)
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    DrawMeshInstanced,
);

struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = Read<InstanceBuffer>;

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        instance_buffer: Option<&'w InstanceBuffer>,
        (meshes, render_mesh_instances, mesh_allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        // A borrow check workaround.
        let mesh_allocator = mesh_allocator.into_inner();

        let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(item.main_entity())
        else {
            return RenderCommandResult::Skip;
        };
        let Some(gpu_mesh) = meshes.into_inner().get(mesh_instance.mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };
        let Some(instance_buffer) = instance_buffer else {
            return RenderCommandResult::Skip;
        };
        let Some(vertex_buffer_slice) =
            mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id)
        else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_buffer_slice) =
                    mesh_allocator.mesh_index_slice(&mesh_instance.mesh_asset_id)
                else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), 0, *index_format);
                pass.draw_indexed(
                    index_buffer_slice.range.start..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    0..instance_buffer.length as u32,
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(vertex_buffer_slice.range, 0..instance_buffer.length as u32);
            }
        }
        RenderCommandResult::Success
    }
}

/// Build world-space splats from the exact live FLIP state, including detached
/// droplets. A deterministic stride bounds upload/fragment cost at high volumes.
pub fn visible_count(sluice: &mining_sim::Sluice) -> usize {
    let stride = sluice.water.len().div_ceil(MAX_SPLATS).max(1);
    sluice.water.len().div_ceil(stride)
}

pub fn sync(game: Res<crate::app::Game>, mut query: Query<&mut WaterSplats>) {
    let Ok(mut splats) = query.single_mut() else {
        return;
    };
    if splats.revision == game.snapshot_id {
        return;
    }
    splats.revision = game.snapshot_id;
    splats.data = Arc::new(instances(&game.session.sluice));
}

fn instances(sluice: &mining_sim::Sluice) -> Vec<InstanceData> {
    let Some(p) = sluice.placement else {
        return Vec::new();
    };
    let stride = sluice.water.len().div_ceil(MAX_SPLATS).max(1);
    let radius = 0.012 * (stride as f32).cbrt();
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
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
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

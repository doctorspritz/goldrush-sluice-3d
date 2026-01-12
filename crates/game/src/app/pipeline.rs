use wgpu::*;

pub enum PipelinePreset {
    OpaqueMesh,
    OpaqueInstanced,
    Transparent,
    Lines,
}

impl PipelinePreset {
    fn topology(&self) -> PrimitiveTopology {
        match self {
            PipelinePreset::Lines => PrimitiveTopology::LineList,
            _ => PrimitiveTopology::TriangleList,
        }
    }

    fn cull_mode(&self) -> Option<Face> {
        match self {
            PipelinePreset::OpaqueMesh | PipelinePreset::OpaqueInstanced => Some(Face::Back),
            _ => None,
        }
    }

    fn write_mask(&self) -> ColorWrites {
        ColorWrites::ALL
    }

    fn blend(&self) -> Option<BlendState> {
        match self {
            PipelinePreset::Transparent => Some(BlendState::ALPHA_BLENDING),
            _ => None,
        }
    }

    fn depth_write_enabled(&self) -> bool {
        match self {
            PipelinePreset::Transparent => false,
            _ => true,
        }
    }
}

impl crate::app::context::GpuContext {
    pub fn create_pipeline(
        &self,
        preset: PipelinePreset,
        shader_source: &str,
        vertex_layouts: &[VertexBufferLayout],
        additional_bind_group_layouts: &[&BindGroupLayout],
    ) -> RenderPipeline {
        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let mut bind_group_layouts = vec![&self.view_bind_group_layout];
        bind_group_layouts.extend(additional_bind_group_layouts);

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("pipeline_layout"),
                bind_group_layouts: &bind_group_layouts,
                push_constant_ranges: &[],
            });

        let target = Some(ColorTargetState {
            format: self.config.format,
            blend: preset.blend(),
            write_mask: preset.write_mask(),
        });

        let depth_stencil = Some(DepthStencilState {
            format: self.depth_format(),
            depth_compare: CompareFunction::Less,
            depth_write_enabled: preset.depth_write_enabled(),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        });

        self.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: vertex_layouts,
            },
            primitive: PrimitiveState {
                topology: preset.topology(),
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: preset.cull_mode(),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[target],
            }),
            multiview: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_preset_topology() {
        assert_eq!(PipelinePreset::OpaqueMesh.topology(), PrimitiveTopology::TriangleList);
        assert_eq!(PipelinePreset::OpaqueInstanced.topology(), PrimitiveTopology::TriangleList);
        assert_eq!(PipelinePreset::Transparent.topology(), PrimitiveTopology::TriangleList);
        assert_eq!(PipelinePreset::Lines.topology(), PrimitiveTopology::LineList);
    }

    #[test]
    fn test_pipeline_preset_cull() {
        assert_eq!(PipelinePreset::OpaqueMesh.cull_mode(), Some(Face::Back));
        assert_eq!(PipelinePreset::OpaqueInstanced.cull_mode(), Some(Face::Back));
        assert_eq!(PipelinePreset::Transparent.cull_mode(), None);
        assert_eq!(PipelinePreset::Lines.cull_mode(), None);
    }

    #[test]
    fn test_pipeline_preset_depth_write() {
        assert!(PipelinePreset::OpaqueMesh.depth_write_enabled());
        assert!(PipelinePreset::OpaqueInstanced.depth_write_enabled());
        assert!(!PipelinePreset::Transparent.depth_write_enabled());
        assert!(PipelinePreset::Lines.depth_write_enabled());
    }

    #[test]
    fn test_pipeline_preset_blend() {
        assert!(PipelinePreset::OpaqueMesh.blend().is_none());
        assert!(PipelinePreset::OpaqueInstanced.blend().is_none());
        assert!(PipelinePreset::Transparent.blend().is_some());
        assert!(PipelinePreset::Lines.blend().is_none());
    }
}

//! View uniform structures for the unified app framework.
//!
//! Contains `ViewUniforms` which holds camera and projection data
//! that is uploaded to the GPU each frame.

use bytemuck::{Pod, Zeroable};

/// View uniforms containing camera and projection data.
///
/// This struct is uploaded to the GPU each frame and is bound at
/// group(0), binding(0) for use in shaders.
///
/// WGSL usage:
/// ```wgsl
/// struct ViewUniforms {
///     view_proj: mat4x4<f32>,
///     camera_pos: vec3<f32>,
///     _pad: f32,
/// }
/// @group(0) @binding(0) var<uniform> view: ViewUniforms;
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ViewUniforms {
    /// Combined view-projection matrix (projection * view)
    pub view_proj: [[f32; 4]; 4],
    /// Camera position in world space
    pub camera_pos: [f32; 3],
    /// Padding to align to 16 bytes
    pub _pad: f32,
}

impl Default for ViewUniforms {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_pos: [0.0, 0.0, 0.0],
            _pad: 0.0,
        }
    }
}

impl ViewUniforms {
    /// Creates new view uniforms with the given view-projection matrix and camera position.
    pub fn new(view_proj: [[f32; 4]; 4], camera_pos: [f32; 3]) -> Self {
        Self {
            view_proj,
            camera_pos,
            _pad: 0.0,
        }
    }

    /// Creates view uniforms from glam types.
    ///
    /// # Arguments
    /// * `view_proj` - Combined view-projection matrix
    /// * `camera_pos` - Camera position in world space
    pub fn from_glam(view_proj: glam::Mat4, camera_pos: glam::Vec3) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _pad: 0.0,
        }
    }

    // TODO: Add `from_camera(camera: &FlyCamera, aspect: f32)` method
    // once FlyCamera is implemented in Package B.
    // This method will compute the view-projection matrix from the camera's
    // position, orientation, and projection parameters.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_uniforms_size() {
        // ViewUniforms should be exactly 80 bytes:
        // - view_proj: 64 bytes (4x4 f32)
        // - camera_pos: 12 bytes (3 f32)
        // - _pad: 4 bytes (1 f32)
        assert_eq!(std::mem::size_of::<ViewUniforms>(), 80);
    }

    #[test]
    fn test_view_uniforms_alignment() {
        // Should be 4-byte aligned (f32 alignment)
        assert_eq!(std::mem::align_of::<ViewUniforms>(), 4);
    }

    #[test]
    fn test_view_uniforms_default() {
        let uniforms = ViewUniforms::default();
        // Default should be identity matrix
        assert_eq!(uniforms.view_proj[0][0], 1.0);
        assert_eq!(uniforms.view_proj[1][1], 1.0);
        assert_eq!(uniforms.view_proj[2][2], 1.0);
        assert_eq!(uniforms.view_proj[3][3], 1.0);
        // Camera at origin
        assert_eq!(uniforms.camera_pos, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_view_uniforms_from_glam() {
        let mat = glam::Mat4::IDENTITY;
        let pos = glam::Vec3::new(1.0, 2.0, 3.0);
        let uniforms = ViewUniforms::from_glam(mat, pos);

        assert_eq!(uniforms.camera_pos, [1.0, 2.0, 3.0]);
        assert_eq!(uniforms.view_proj[0][0], 1.0);
        assert_eq!(uniforms.view_proj[3][3], 1.0);
    }
}

use bytemuck::{Pod, Zeroable};

use super::camera::FlyCamera;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ViewUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _pad: f32,
}

impl ViewUniforms {
    pub fn from_camera(camera: &FlyCamera, aspect: f32) -> Self {
        let view_proj = camera.view_projection(aspect);
        let view_proj_array = view_proj.to_cols_array_2d();

        Self {
            view_proj: view_proj_array,
            camera_pos: camera.position.to_array(),
            _pad: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view_uniforms_from_camera() {
        let camera = FlyCamera::new();
        let uniforms = ViewUniforms::from_camera(&camera, 16.0 / 9.0);

        assert_eq!(uniforms.camera_pos[0], camera.position.x);
        assert_eq!(uniforms.camera_pos[1], camera.position.y);
        assert_eq!(uniforms.camera_pos[2], camera.position.z);
    }
}

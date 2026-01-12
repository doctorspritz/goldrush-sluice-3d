use glam::{Mat4, Vec3};

pub struct FlyCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
    pub sensitivity: f32,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Clone, Copy, Default)]
pub struct InputState {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            speed: 5.0,
            sensitivity: 0.003,
            fov: 1.047, // 60 degrees in radians
            near: 0.01,
            far: 100.0,
        }
    }
}

impl FlyCamera {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_position(mut self, pos: Vec3) -> Self {
        self.position = pos;
        self
    }

    pub fn with_target(mut self, target: Vec3) -> Self {
        let dir = (target - self.position).normalize();

        // Extract yaw and pitch from direction
        // yaw is rotation around Y axis, pitch is rotation around X axis
        self.yaw = dir.z.atan2(dir.x);
        self.pitch = (-dir.y).asin();

        self
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    pub fn right(&self) -> Vec3 {
        Vec3::new(
            (self.yaw + std::f32::consts::PI / 2.0).cos(),
            0.0,
            (self.yaw + std::f32::consts::PI / 2.0).sin(),
        )
    }

    pub fn up(&self) -> Vec3 {
        Vec3::Y
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), self.up())
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect, self.near, self.far)
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    pub fn on_mouse_move(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw += delta_x * self.sensitivity;
        self.pitch += delta_y * self.sensitivity;

        // Clamp pitch to ±89 degrees
        const PITCH_LIMIT: f32 = 1.5533; // ~89 degrees
        self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn on_scroll(&mut self, delta: f32) {
        self.position += self.forward() * delta * self.speed * 0.1;
    }

    pub fn update(&mut self, input: &InputState, dt: f32) {
        let mut movement = Vec3::ZERO;

        if input.forward {
            movement += self.forward();
        }
        if input.back {
            movement -= self.forward();
        }
        if input.right {
            movement += self.right();
        }
        if input.left {
            movement -= self.right();
        }
        if input.up {
            movement += self.up();
        }
        if input.down {
            movement -= self.up();
        }

        if movement.length_squared() > 0.0 {
            self.position += movement.normalize() * self.speed * dt;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_forward_direction() {
        let mut cam = FlyCamera::new();
        cam.yaw = 0.0;
        cam.pitch = 0.0;
        let fwd = cam.forward();
        assert!((fwd.x - 1.0).abs() < 0.001, "Forward X should be ~1.0");
        assert!(fwd.y.abs() < 0.001, "Forward Y should be ~0.0");
        assert!(fwd.z.abs() < 0.001, "Forward Z should be ~0.0");
    }

    #[test]
    fn test_camera_pitch_clamp() {
        let mut cam = FlyCamera::new();
        cam.on_mouse_move(0.0, 10000.0);
        assert!(cam.pitch <= 1.554, "Pitch should be clamped to ~89°");
        assert!(cam.pitch >= -1.554, "Pitch should be clamped to ~-89°");
    }

    #[test]
    fn test_camera_wasd_movement() {
        let mut cam = FlyCamera::new();
        cam.position = Vec3::ZERO;
        cam.yaw = 0.0;
        cam.speed = 10.0;

        let input = InputState {
            forward: true,
            ..Default::default()
        };
        cam.update(&input, 1.0);

        assert!(cam.position.x > 9.0, "Should move ~10 units forward");
    }

    #[test]
    fn test_camera_right_strafe() {
        let mut cam = FlyCamera::new();
        cam.position = Vec3::ZERO;
        cam.yaw = 0.0;
        cam.speed = 10.0;

        let input = InputState {
            right: true,
            ..Default::default()
        };
        cam.update(&input, 1.0);

        assert!(cam.position.z > 9.0, "Should move right along +Z");
    }

    #[test]
    fn test_camera_up_movement() {
        let mut cam = FlyCamera::new();
        cam.position = Vec3::ZERO;
        cam.speed = 10.0;

        let input = InputState {
            up: true,
            ..Default::default()
        };
        cam.update(&input, 1.0);

        assert!(cam.position.y > 9.0, "Should move up");
    }
}

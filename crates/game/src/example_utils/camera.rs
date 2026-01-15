use glam::{Mat4, Vec3};

pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

impl Camera {
    pub fn new(yaw: f32, pitch: f32, distance: f32, target: Vec3) -> Self {
        Self {
            yaw,
            pitch,
            distance,
            target,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.target
            + Vec3::new(
                self.distance * self.yaw.cos() * self.pitch.cos(),
                self.distance * self.pitch.sin(),
                self.distance * self.yaw.sin() * self.pitch.cos(),
            )
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.target, Vec3::Y)
    }

    pub fn proj_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 100.0)
    }

    pub fn view_proj_matrix(&self, aspect: f32) -> Mat4 {
        self.proj_matrix(aspect) * self.view_matrix()
    }

    pub fn handle_mouse_move(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(-1.4, 1.4);
    }

    pub fn handle_zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta * 0.1).clamp(0.5, 20.0);
    }
}

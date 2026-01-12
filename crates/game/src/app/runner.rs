use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::Window,
};

use super::{camera::FlyCamera, camera::InputState, context::GpuContext, uniforms::ViewUniforms};

pub trait App: 'static {
    fn init(ctx: &GpuContext) -> Self;
    fn update(&mut self, ctx: &GpuContext, dt: f32);
    fn render(
        &mut self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    );

    fn on_key(&mut self, _key: winit::keyboard::KeyCode, _pressed: bool) {}
    fn on_resize(&mut self, _ctx: &GpuContext) {}
    fn camera(&self) -> &FlyCamera;
    fn camera_mut(&mut self) -> &mut FlyCamera;
    fn title() -> &'static str {
        "App"
    }
}

pub fn run<A: App>() -> ! {
    let event_loop = EventLoop::new().unwrap();
    let mut runner = AppRunner::<A>::new();
    let _ = event_loop.run_app(&mut runner);
    std::process::exit(0)
}

struct AppRunner<A: App> {
    window: Option<Arc<Window>>,
    ctx: Option<GpuContext>,
    app: Option<A>,
    input_state: InputState,
    last_time: Option<std::time::Instant>,
}

impl<A: App> AppRunner<A> {
    fn new() -> Self {
        Self {
            window: None,
            ctx: None,
            app: None,
            input_state: InputState::default(),
            last_time: None,
        }
    }
}

impl<A: App> ApplicationHandler for AppRunner<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title(A::title())
                            .with_inner_size(winit::dpi::LogicalSize::new(1920, 1080)),
                    )
                    .unwrap(),
            );
            self.window = Some(window.clone());

            let ctx = pollster::block_on(GpuContext::new(window));
            self.app = Some(A::init(&ctx));
            self.ctx = Some(ctx);
            self.last_time = Some(std::time::Instant::now());
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let (Some(ctx), Some(app)) = (&mut self.ctx, &mut self.app) {
            match event {
                WindowEvent::Resized(size) => {
                    ctx.resize(size.width, size.height);
                    app.on_resize(ctx);
                }
                WindowEvent::KeyboardInput {
                    event,
                    is_synthetic: false,
                    ..
                } => {
                    use winit::event::ElementState;
                    let pressed = event.state == ElementState::Pressed;
                    use winit::keyboard::KeyCode;

                    match event.logical_key.clone() {
                        winit::keyboard::Key::Named(key) => {
                            let code = match key {
                                winit::keyboard::NamedKey::ArrowUp => Some(KeyCode::ArrowUp),
                                winit::keyboard::NamedKey::ArrowDown => Some(KeyCode::ArrowDown),
                                winit::keyboard::NamedKey::ArrowLeft => Some(KeyCode::ArrowLeft),
                                winit::keyboard::NamedKey::ArrowRight => Some(KeyCode::ArrowRight),
                                winit::keyboard::NamedKey::Space => Some(KeyCode::Space),
                                winit::keyboard::NamedKey::Shift => Some(KeyCode::ShiftLeft),
                                _ => None,
                            };
                            if let Some(code) = code {
                                app.on_key(code, pressed);
                            }
                        }
                        winit::keyboard::Key::Character(ch) => {
                            let code = match ch.as_str() {
                                "w" | "W" => Some(KeyCode::KeyW),
                                "a" | "A" => Some(KeyCode::KeyA),
                                "s" | "S" => Some(KeyCode::KeyS),
                                "d" | "D" => Some(KeyCode::KeyD),
                                _ => None,
                            };
                            if let Some(code) = code {
                                app.on_key(code, pressed);
                                match code {
                                    KeyCode::KeyW => self.input_state.forward = pressed,
                                    KeyCode::KeyA => self.input_state.left = pressed,
                                    KeyCode::KeyS => self.input_state.back = pressed,
                                    KeyCode::KeyD => self.input_state.right = pressed,
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }

                    if let winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) =
                        event.logical_key.clone()
                    {
                        self.input_state.up = pressed;
                    }
                    if let winit::keyboard::Key::Named(winit::keyboard::NamedKey::Shift) =
                        event.logical_key.clone()
                    {
                        self.input_state.down = pressed;
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    use winit::event::MouseScrollDelta;
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                    };
                    app.camera_mut().on_scroll(scroll);
                }
                WindowEvent::CloseRequested => {
                    std::process::exit(0);
                }
                _ => {}
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(app) = &mut self.app {
            if let DeviceEvent::MouseMotion { delta } = event {
                app.camera_mut()
                    .on_mouse_move(delta.0 as f32, delta.1 as f32);
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let (Some(ctx), Some(app), Some(last_time)) = (&self.ctx, &mut self.app, self.last_time)
        {
            let now = std::time::Instant::now();
            let dt = (now - last_time).as_secs_f32().min(0.1);
            self.last_time = Some(now);

            // Update camera from input
            app.camera_mut().update(&self.input_state, dt);

            // Update app logic
            app.update(ctx, dt);

            // Update view uniforms
            let aspect = ctx.config.width as f32 / ctx.config.height as f32;
            let uniforms = ViewUniforms::from_camera(app.camera(), aspect);
            ctx.update_view_uniforms(&uniforms);

            // Render
            let surface_texture = ctx.surface.get_current_texture().unwrap();
            let view = surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            app.render(ctx, &mut encoder, &view);

            ctx.queue.submit(std::iter::once(encoder.finish()));
            surface_texture.present();

            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestApp {
        camera: FlyCamera,
    }

    impl App for TestApp {
        fn init(_ctx: &GpuContext) -> Self {
            Self {
                camera: FlyCamera::new(),
            }
        }

        fn update(&mut self, _ctx: &GpuContext, _dt: f32) {}

        fn render(
            &mut self,
            _ctx: &GpuContext,
            _encoder: &mut wgpu::CommandEncoder,
            _view: &wgpu::TextureView,
        ) {
        }

        fn camera(&self) -> &FlyCamera {
            &self.camera
        }

        fn camera_mut(&mut self) -> &mut FlyCamera {
            &mut self.camera
        }

        fn title() -> &'static str {
            "Test"
        }
    }

    #[test]
    fn test_input_state_default() {
        let input = InputState::default();
        assert!(!input.forward);
        assert!(!input.back);
        assert!(!input.left);
        assert!(!input.right);
        assert!(!input.up);
        assert!(!input.down);
    }
}

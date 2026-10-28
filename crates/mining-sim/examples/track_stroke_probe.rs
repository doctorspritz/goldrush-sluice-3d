//! Bounded joystick strokes on declared flat layered soil; not calibration.
use mining_sim::{Excavator, MachineInput, Site};
fn main() {
    for (boom, stick, curl) in [
        (-0.22, 0., 0.),
        (-0.22, -0.25, 0.2),
        (-0.22, -0.4, 0.4),
        (-0.1, -0.25, 0.2),
        (-0.1, -0.4, 0.4),
        (0., -0.25, 0.2),
        (-0.3, -0.5, 0.25),
    ] {
        let mut s = Site::new();
        for i in 0..s.gold_g.len() {
            s.world.bedrock_elevation[i] = 0.05;
            s.world.overburden_thickness[i] = 0.1;
            s.world.paydirt_thickness[i] = 0.2;
            s.world.gravel_thickness[i] = 0.;
            s.world.terrain_sediment[i] = 0.;
            s.world.water_surface[i] = 0.45;
            s.gold_g[i] = 0.125;
        }
        let mut m = Excavator::new(&s);
        for _ in 0..240 {
            m.step(
                MachineInput {
                    cut: true,
                    boom,
                    stick,
                    curl,
                    ..Default::default()
                },
                &mut s,
                1. / 60.,
            );
        }
        println!("stroke={boom}/{stick}/{curl} volume={:.8} lip_y={:.6} heel_y={:.6} tilt={:.4} power={:.3} force={:.3}",m.bucket.volume(),m.arm().lip.y,m.arm().heel.y,m.tracks.feedback.tilt_deg,m.digging.tool_power_w,m.digging.soil_force_n);
    }
}

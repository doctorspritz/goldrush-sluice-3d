//! Main-thread bridge. Input and presentation continue while the one simulation
//! worker owns the authoritative shift and waits for the original GPU solvers.
use crate::{app::Game, validation::Validation};
use bevy::prelude::*;
use mining_physics::worker::{Command, Work, Worker};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

pub struct Runtime {
    worker: Worker,
    next_tick: Instant,
    failed: bool,
    sequence: u64,
    submitted: VecDeque<Prediction>,
}
struct Prediction {
    sequence: u64,
    look: [f32; 2],
    pause: Option<bool>,
}
fn acknowledge(
    submitted: &mut VecDeque<Prediction>,
    sequence: u64,
    mut look: [f32; 2],
) -> ([f32; 2], Option<bool>) {
    while submitted.front().is_some_and(|p| p.sequence <= sequence) {
        submitted.pop_front();
    }
    let mut paused = None;
    for prediction in submitted {
        look[0] += prediction.look[0];
        look[1] += prediction.look[1];
        if prediction.pause.is_some() {
            paused = prediction.pause;
        }
    }
    (look, paused)
}

impl Runtime {
    pub fn new(session: mining_sim::Session) -> Result<Self, String> {
        Ok(Self {
            worker: if std::env::args().any(|a| a == "--conservative-phases") {
                Worker::with_session_conservative(session)?
            } else {
                Worker::with_session(session)?
            },
            next_tick: Instant::now(),
            failed: false,
            sequence: 1,
            submitted: VecDeque::new(),
        })
    }
}

pub fn receive(
    mut game: ResMut<Game>,
    mut runtime: NonSendMut<Runtime>,
    mut validation: ResMut<Validation>,
) {
    if runtime.failed {
        return;
    }
    for _ in 0..2 {
        match runtime.worker.poll() {
            Ok(Some(frame)) => {
                validation.record_step(frame.step_ms, frame.phases_ms);
                let (look, paused) =
                    acknowledge(&mut runtime.submitted, frame.sequence, game.look_delta);
                game.session = *frame.session;
                game.simulation_generation = frame.generation;
                // Include mouse input in the queued packet as well as newer input;
                // an older snapshot must never rewind the user's current camera.
                game.session.player.yaw += look[0];
                game.session.player.pitch =
                    (game.session.player.pitch + look[1]).clamp(-1.30, 1.10);
                if let Some(paused) = paused {
                    game.session.paused = paused;
                }
                if let Some(paused) = game.pending.iter().rev().find_map(|c| {
                    if let Command::Pause(p) = c {
                        Some(*p)
                    } else {
                        None
                    }
                }) {
                    game.session.paused = paused;
                }
                game.snapshot_id += 1;
                game.step_ms = frame.step_ms;
                if let Some(replay) = frame.replay {
                    validation.replay = replay;
                }
                if validation.restore_requested && game.session.message == "Shift restored" {
                    validation.saved = true;
                }
                if validation.automated() {
                    if let Some(error) = frame.failure {
                        validation.failure = Some(error);
                    }
                    if game.session.message.starts_with("Save failed:")
                        || game.session.message.starts_with("Load failed:")
                    {
                        validation.failure = Some(game.session.message.clone());
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                game.session.message = e.clone();
                game.session.paused = true;
                runtime.failed = true;
                if validation.automated() {
                    validation.failure = Some(e);
                }
            }
        }
    }
}

pub fn dispatch(
    mut game: ResMut<Game>,
    mut runtime: NonSendMut<Runtime>,
    validation: Res<Validation>,
) {
    if game.session.paused && game.pending.is_empty() && game.look_delta == [0.0; 2] {
        runtime.next_tick = Instant::now();
        return;
    }
    let now = Instant::now();
    // At most active + pending. No simulation executes on the render thread.
    for _ in 0..2 {
        if runtime.failed || !runtime.worker.can_submit() || !tick_due(&mut runtime.next_tick, now)
        {
            return;
        }
        // Share publication when both fixed steps are already due. Each step
        // still runs the complete solver; commands and mouse deltas apply once.
        let ticks = if tick_due(&mut runtime.next_tick, now) {
            2
        } else {
            1
        };
        let work = Work {
            ticks,
            sequence: runtime.sequence,
            controls: game.take_controls(),
            commands: std::mem::take(&mut game.pending),
            look_delta: std::mem::take(&mut game.look_delta),
            advance: !validation.active() || validation.ready(),
            replay: validation.active()
                && !(validation.earthworks_only && validation.replay.phase >= 15),
        };
        let prediction = Prediction {
            sequence: work.sequence,
            look: work.look_delta,
            pause: work.commands.iter().rev().find_map(|c| {
                if let Command::Pause(p) = c {
                    Some(*p)
                } else {
                    None
                }
            }),
        };
        if let Err(e) = runtime.worker.submit(work) {
            game.session.message = e;
            game.session.paused = true;
            runtime.failed = true;
        } else {
            runtime.submitted.push_back(prediction);
            runtime.sequence += 1;
        }
    }
}
fn tick_due(next: &mut Instant, now: Instant) -> bool {
    if now < *next {
        return false;
    }
    let dt = Duration::from_secs_f32(mining_sim::FIXED_DT);
    // Bound debt after a stall/resume; never accumulate a long catch-up run.
    if now.duration_since(*next) > dt * 2 {
        *next = now - dt;
    }
    *next += dt;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_tick_scheduling_is_independent_of_render_rate_and_debt_is_bounded() {
        let start = Instant::now();
        let mut next = start;
        let mut ticks = 0;
        for frame in 0..=59 {
            let now = start + Duration::from_secs_f64(frame as f64 / 59.0);
            for _ in 0..2 {
                if tick_due(&mut next, now) {
                    ticks += 1;
                } else {
                    break;
                }
            }
        }
        assert!(
            (60..=61).contains(&ticks),
            "59FPS should still schedule 60 physics ticks: {ticks}"
        );
        let now = start + Duration::from_secs(10);
        assert!(tick_due(&mut next, now));
        assert!(tick_due(&mut next, now));
        assert!(!tick_due(&mut next, now));
    }
    #[test]
    fn delayed_snapshot_preserves_unacknowledged_look_and_pause_once() {
        let mut submitted = VecDeque::from([
            Prediction {
                sequence: 1,
                look: [-0.1, 0.2],
                pause: Some(false),
            },
            Prediction {
                sequence: 2,
                look: [0.2, 0.1],
                pause: Some(true),
            },
        ]);
        let (look, paused) = acknowledge(&mut submitted, 1, [0.3, 0.4]);
        assert!((look[0] - 0.5).abs() < 1e-6);
        assert!((look[1] - 0.5).abs() < 1e-6);
        assert_eq!(paused, Some(true));
        assert_eq!(submitted.len(), 1);
        let (look, paused) = acknowledge(&mut submitted, 2, [0.1, -0.2]);
        assert_eq!(look, [0.1, -0.2]);
        assert_eq!(paused, None);
        assert!(submitted.is_empty());
    }
}

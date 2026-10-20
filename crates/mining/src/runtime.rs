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
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            worker: Worker::new()?,
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
    match runtime.worker.poll() {
        Ok(Some(frame)) => {
            let (look, paused) =
                acknowledge(&mut runtime.submitted, frame.sequence, game.look_delta);
            game.session = *frame.session;
            // Include mouse input in the queued packet as well as newer input;
            // an older snapshot must never rewind the user's current camera.
            game.session.player.yaw += look[0];
            game.session.player.pitch = (game.session.player.pitch + look[1]).clamp(-1.30, 1.10);
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
        Ok(None) => {}
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

pub fn dispatch(
    mut game: ResMut<Game>,
    mut runtime: NonSendMut<Runtime>,
    validation: Res<Validation>,
) {
    if runtime.failed || !runtime.worker.can_submit() || Instant::now() < runtime.next_tick {
        return;
    }
    let work = Work {
        sequence: runtime.sequence,
        controls: game.controls,
        commands: std::mem::take(&mut game.pending),
        look_delta: std::mem::take(&mut game.look_delta),
        advance: !validation.active() || validation.ready(),
        replay: validation.active(),
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
    // No accumulated debt: a slow step schedules one subsequent tick, never a
    // burst of fixed updates on the render thread after a long frame or resume.
    runtime.next_tick =
        (runtime.next_tick + Duration::from_secs_f32(mining_sim::FIXED_DT)).max(Instant::now());
}

#[cfg(test)]
mod tests {
    use super::*;
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

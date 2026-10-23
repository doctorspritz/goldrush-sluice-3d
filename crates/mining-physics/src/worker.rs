//! A single simulation owner with at most two input packets (active + pending). Rendering never waits
//! for GPU work, and a slow step cannot produce an unbounded catch-up queue.
use crate::Hydraulics;
use mining_sim::{replay::OperatorReplay, session::Controls, storage, Action, Session};
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, SyncSender, TryRecvError},
    thread::{self, JoinHandle},
    time::Instant,
};

pub enum Command {
    Action(Action),
    Pause(bool),
    Water(f32),
    Feed(f32),
    Save(PathBuf),
    Load(PathBuf),
    Reset(Box<Session>),
}

#[derive(Default)]
pub struct Work {
    pub sequence: u64,
    pub controls: Controls,
    pub commands: Vec<Command>,
    pub look_delta: [f32; 2],
    pub advance: bool,
    /// At most two unchanged fixed steps, sharing one snapshot publication.
    pub ticks: u8,
    pub replay: bool,
}

pub struct Snapshot {
    pub generation: u64,
    pub sequence: u64,
    pub session: Box<Session>,
    pub replay: Option<OperatorReplay>,
    pub step_ms: f64,
    pub phases_ms: [f64; 5],
    pub failure: Option<String>,
}

pub struct Worker {
    requests: Option<SyncSender<Work>>,
    replies: Receiver<Result<Snapshot, String>>,
    thread: Option<JoinHandle<()>>,
    outstanding: usize,
    ready: bool,
}

fn configure_worker_thread() {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn pthread_set_qos_class_self_np(class: u32, relative_priority: i32) -> i32;
        }
        // Normal macOS user-initiated QoS, scoped to this one worker. This is
        // not a real-time scheduler policy and changes no system configuration.
        let status = unsafe { pthread_set_qos_class_self_np(0x19, 0) };
        if status != 0 {
            eprintln!("Simulation worker QoS could not be set: {status}");
        }
    }
}
impl Worker {
    pub fn new() -> Result<Self, String> {
        Self::with_session(Session::new())
    }
    pub fn with_session(session: Session) -> Result<Self, String> {
        Self::spawn(Hydraulics::new, session)
    }
    pub fn with_session_conservative(session: Session) -> Result<Self, String> {
        Self::spawn(
            || Hydraulics::new().map(Hydraulics::with_conservative_phases),
            session,
        )
    }
    fn spawn<T: Solver + 'static>(
        create: impl FnOnce() -> Result<T, String> + Send + 'static,
        mut session: Session,
    ) -> Result<Self, String> {
        let (tx, requests) = mpsc::sync_channel::<Work>(2);
        let (replies, rx) = mpsc::sync_channel(2);
        let handle = thread::Builder::new()
            .name("sluice-simulation".into())
            .spawn(move || {
                configure_worker_thread();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut solver = create()?;
                    session.sluice.ensure_water_ids();
                    let mut replay = None;
                    let mut generation = 0;
                    if replies
                        .send(Ok(snapshot(0, &session, &replay, 0.0, None)))
                        .is_err()
                    {
                        return Ok(());
                    }
                    while let Ok(work) = requests.recv() {
                        let started = Instant::now();
                        if apply(&mut session, &mut solver, work.commands) {
                            generation += 1;
                        }
                        session.sluice.ensure_water_ids();
                        session.player.yaw += work.look_delta[0];
                        session.player.pitch =
                            (session.player.pitch + work.look_delta[1]).clamp(-1.30, 1.10);
                        let mut failure = None;
                        let mut phases = [0.; 5];
                        for _ in 0..work.ticks.clamp(1, 2) {
                            if work.advance && !session.paused {
                                let controls = if work.replay {
                                    replay
                                        .get_or_insert_with(OperatorReplay::default)
                                        .controls(&mut session)
                                } else {
                                    work.controls
                                };
                                session.step(controls);
                                if let Err(e) = solver.step(&mut session) {
                                    session.message = format!(
                                        "Simulation stopped: {e}. Save your shift with F5."
                                    );
                                    session.paused = true;
                                    failure = Some(e);
                                }
                                for (sum, p) in phases.iter_mut().zip(solver.timings()) {
                                    *sum += p;
                                }
                                if let Some(e) = replay.as_ref().and_then(|r| r.error.clone()) {
                                    failure = Some(e);
                                }
                            }
                        }
                        let mut frame = snapshot(
                            work.sequence,
                            &session,
                            &replay,
                            started.elapsed().as_secs_f64() * 1000.0,
                            failure,
                        );
                        frame.generation = generation;
                        frame.phases_ms = phases;
                        frame.step_ms = started.elapsed().as_secs_f64() * 1000.0;
                        if replies.send(Ok(frame)).is_err() {
                            break;
                        }
                    }
                    Ok::<(), String>(())
                }));
                let error = match result {
                    Ok(Ok(())) => return,
                    Ok(Err(e)) => e,
                    Err(_) => "The simulation worker stopped unexpectedly".into(),
                };
                let _ = replies.send(Err(error));
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            requests: Some(tx),
            replies: rx,
            thread: Some(handle),
            outstanding: 0,
            ready: false,
        })
    }
    pub fn can_submit(&self) -> bool {
        self.ready && self.outstanding < 2
    }
    pub fn submit(&mut self, work: Work) -> Result<(), String> {
        if !self.can_submit() {
            return Err("Simulation input buffer is full".into());
        }
        self.requests
            .as_ref()
            .ok_or("Simulation is closed")?
            .try_send(work)
            .map_err(|e| e.to_string())?;
        self.outstanding += 1;
        Ok(())
    }
    pub fn poll(&mut self) -> Result<Option<Snapshot>, String> {
        match self.replies.try_recv() {
            Ok(result) => {
                if self.ready {
                    self.outstanding = self.outstanding.saturating_sub(1);
                }
                self.ready = true;
                result.map(Some)
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err("Simulation worker disconnected".into()),
        }
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.requests.take();
        // Drain both bounded replies while shutting down so publishing a final
        // snapshot cannot block the worker after its input has been closed.
        if let Some(handle) = self.thread.take() {
            while !handle.is_finished() {
                let _ = self
                    .replies
                    .recv_timeout(std::time::Duration::from_millis(10));
            }
            let _ = handle.join();
        }
    }
}
trait Solver {
    fn timings(&self) -> [f64; 5] {
        [0.; 5]
    }
    fn step(&mut self, session: &mut Session) -> Result<(), String>;
    fn restore(&mut self);
}
impl Solver for Hydraulics {
    fn timings(&self) -> [f64; 5] {
        let t = self.timings;
        [
            t.prepare_ms,
            t.fluid_ms,
            t.dem_ms,
            t.finish_ms,
            t.terrain_ms,
        ]
    }
    fn step(&mut self, session: &mut Session) -> Result<(), String> {
        Hydraulics::step(self, session)
    }
    fn restore(&mut self) {
        Hydraulics::restore(self)
    }
}
fn snapshot(
    sequence: u64,
    session: &Session,
    replay: &Option<OperatorReplay>,
    step_ms: f64,
    failure: Option<String>,
) -> Snapshot {
    Snapshot {
        generation: 0,
        sequence,
        session: Box::new(session.clone()),
        replay: replay.clone(),
        step_ms,
        phases_ms: [0.; 5],
        failure,
    }
}
fn apply(session: &mut Session, solver: &mut impl Solver, commands: Vec<Command>) -> bool {
    let mut restored_state = false;
    for command in commands {
        match command {
            Command::Action(action) => {
                let _ = session.on_action(action);
            }
            Command::Reset(mut replacement) => {
                replacement.site.revision = session.site.revision + 1;
                *session = *replacement;
                solver.restore();
                restored_state = true;
            }
            Command::Pause(paused) => session.paused = paused,
            Command::Water(delta) if session.near_plant() => {
                session.sluice.flow_m3s = (session.sluice.flow_m3s + delta).clamp(0.001, 0.120)
            }
            Command::Feed(delta) if session.near_plant() => {
                session.sluice.feed_m3s = (session.sluice.feed_m3s + delta).clamp(0.00001, 0.02)
            }
            Command::Water(_) | Command::Feed(_) => {}
            Command::Save(path) => {
                session.message = match storage::save(session, &path) {
                    Ok(()) => "Shift saved. F9 restores this shift.".into(),
                    Err(e) => format!("Save failed: {e}"),
                };
            }
            Command::Load(path) => match storage::load(&path) {
                Ok(mut restored) => {
                    restored.site.revision = session.site.revision + 1;
                    *session = restored;
                    solver.restore();
                    restored_state = true;
                }
                Err(e) => session.message = format!("Load failed: {e}"),
            },
        }
    }
    restored_state
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    struct SlowSolver;
    impl Solver for SlowSolver {
        fn step(&mut self, _: &mut Session) -> Result<(), String> {
            thread::sleep(Duration::from_millis(30));
            Ok(())
        }
        fn restore(&mut self) {}
    }
    fn receive(worker: &mut Worker) -> Snapshot {
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(frame) = worker.poll().unwrap() {
                return frame;
            }
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn two_tick_packets_preserve_fixed_steps_and_apply_actions_once() {
        let session = Session::new();
        let mut worker = Worker::spawn(|| Ok(SlowSolver), session).unwrap();
        receive(&mut worker);
        worker
            .submit(Work {
                advance: true,
                ticks: 2,
                look_delta: [0.3, 0.],
                commands: vec![Command::Action(Action::EnterExit)],
                ..Default::default()
            })
            .unwrap();
        let frame = receive(&mut worker);
        assert!((frame.session.sim_time - 2. * mining_sim::FIXED_DT as f64).abs() < 1e-9);
        assert_eq!(
            frame.session.player.mode,
            mining_sim::ControlMode::Excavator
        );
        // The dynamic chassis can settle/rotate during both ticks. Look is
        // relative to the resulting cab, and must still be applied once.
        let cab_yaw = frame.session.machine.heading + frame.session.machine.swing;
        assert!((frame.session.player.yaw - cab_yaw - 0.3).abs() < 1e-6);
    }
    #[test]
    fn slow_steps_are_bounded_and_pause_save_load_are_ordered() {
        let mut worker = Worker::spawn(|| Ok(SlowSolver), Session::new()).unwrap();
        receive(&mut worker);
        let path = std::env::temp_dir().join(format!("sluice-worker-{}.json", std::process::id()));
        worker
            .submit(Work {
                advance: true,
                sequence: 1,
                commands: vec![Command::Action(Action::EnterExit)],
                ..Default::default()
            })
            .unwrap();
        worker
            .submit(Work {
                sequence: 2,
                look_delta: [0.3, 0.0],
                advance: true,
                ..Default::default()
            })
            .unwrap();
        assert!(
            worker.submit(Work::default()).is_err(),
            "overload must not accumulate requests"
        );
        let frame = receive(&mut worker);
        assert_eq!(
            frame.session.player.mode,
            mining_sim::ControlMode::Excavator
        );
        assert!((frame.session.sim_time - mining_sim::FIXED_DT as f64).abs() < 1e-9);
        let previous_yaw = frame.session.player.yaw;
        let frame = receive(&mut worker);
        assert_eq!(frame.sequence, 2);
        assert!((frame.session.player.yaw - previous_yaw - 0.3).abs() < 1e-6);
        assert!((frame.session.sim_time - 2.0 * mining_sim::FIXED_DT as f64).abs() < 1e-9);
        worker
            .submit(Work {
                advance: true,
                commands: vec![Command::Pause(true), Command::Save(path.clone())],
                ..Default::default()
            })
            .unwrap();
        let paused = receive(&mut worker);
        assert_eq!(paused.session.sim_time, frame.session.sim_time);
        worker
            .submit(Work {
                commands: vec![Command::Load(path.clone())],
                ..Default::default()
            })
            .unwrap();
        let restored = receive(&mut worker);
        assert_eq!(restored.session.sim_time, frame.session.sim_time);
        assert_eq!(paused.generation, 0);
        assert_eq!(
            restored.generation, 1,
            "paused restore must invalidate render caches"
        );
        assert_eq!(
            restored.session.machine.arm().lip,
            frame.session.machine.arm().lip
        );
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn a_storage_refusal_pauses_once_and_keeps_the_shift_saveable() {
        struct RefusedStorage;
        impl Solver for RefusedStorage {
            fn step(&mut self, _: &mut Session) -> Result<(), String> {
                Err("Particle storage request exceeds the planning/adapter budget".into())
            }
            fn restore(&mut self) {}
        }
        let mut session = Session::new();
        session.player.hand =
            session
                .site
                .dig(mining_sim::Vec3::new(21., 0., 18.), 0.3, 0.1, 0.005);
        assert!(session.player.hand.volume() > 0.);
        let hand = serde_json::to_string(&session.player.hand).unwrap();
        let initial_gold = session.accounted_gold();
        let mut worker = Worker::spawn(|| Ok(RefusedStorage), session).unwrap();
        receive(&mut worker);
        worker
            .submit(Work {
                advance: true,
                ticks: 2,
                ..Default::default()
            })
            .unwrap();
        let stopped = receive(&mut worker);
        assert!(stopped.session.paused && stopped.failure.as_deref().unwrap().contains("storage"));
        assert!((stopped.session.sim_time - mining_sim::FIXED_DT as f64).abs() < 1e-9);
        assert_eq!(
            serde_json::to_string(&stopped.session.player.hand).unwrap(),
            hand
        );
        assert!((stopped.session.accounted_gold() - initial_gold).abs() < 1e-9);
        let saved = stopped.session.save_json().unwrap();
        let restored = Session::load_json(&saved).unwrap();
        assert_eq!(serde_json::to_string(&restored.player.hand).unwrap(), hand);
        worker
            .submit(Work {
                advance: true,
                ticks: 2,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            receive(&mut worker).session.sim_time,
            stopped.session.sim_time
        );
    }
}

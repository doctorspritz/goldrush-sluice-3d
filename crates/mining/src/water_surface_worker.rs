//! One bounded render-only worker. At most one reconstruction is active and
//! one pending snapshot is retained; a newer snapshot replaces that pending
//! render copy. Authoritative water, grains and material are never modified.
use crate::{water::InstanceData, water_presentation::pair_particles};
use mining_sim::Sluice;
use std::{
    sync::{mpsc, Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

struct Work {
    epoch: u64,
    sim: f64,
    wall: f64,
    sluice: Box<Sluice>,
    enqueued: Instant,
}
#[derive(Default)]
struct Pending {
    latest: Option<Work>,
    closed: bool,
}
#[derive(Default)]
struct Mailbox {
    state: Mutex<Pending>,
    available: Condvar,
}
impl Mailbox {
    fn replace(&self, work: Work) -> Option<Work> {
        let mut state = self.state.lock().expect("Surface mailbox");
        assert!(!state.closed, "Surface worker is closed");
        let replaced = state.latest.replace(work);
        self.available.notify_one();
        replaced
    }
    fn take(&self) -> Option<Work> {
        let mut state = self.state.lock().expect("Surface mailbox");
        loop {
            if let Some(work) = state.latest.take() {
                return Some(work);
            }
            if state.closed {
                return None;
            }
            state = self.available.wait(state).expect("Surface mailbox");
        }
    }
    fn close(&self) {
        let mut state = self.state.lock().expect("Surface mailbox");
        state.closed = true;
        self.available.notify_one();
    }
}

pub(crate) struct Surface {
    pub epoch: u64,
    pub sim: f64,
    pub wall: f64,
    pub data: Arc<Vec<InstanceData>>,
    pub ms: f64,
    pub queue_ms: f64,
}
pub(crate) struct SurfaceWorker {
    requests: Option<Arc<Mailbox>>,
    replies: mpsc::Receiver<Surface>,
    thread: Option<JoinHandle<()>>,
    coalesced: u64,
}
impl Default for SurfaceWorker {
    fn default() -> Self {
        Self::spawn(crate::water::samples)
    }
}
impl SurfaceWorker {
    fn spawn(mut sample: impl FnMut(&Sluice) -> Vec<(u64, InstanceData)> + Send + 'static) -> Self {
        let requests = Arc::new(Mailbox::default());
        let mailbox = Arc::clone(&requests);
        // Keep completed outputs bounded and ordered. Interpolation's previous
        // kernels must refer to the previously delivered reconstruction.
        let (replies, rx) = mpsc::sync_channel(2);
        let thread = thread::Builder::new()
            .name("sluice-water-surface".into())
            .spawn(move || {
                let mut previous: Option<(u64, Vec<(u64, InstanceData)>)> = None;
                while let Some(work) = mailbox.take() {
                    let started = Instant::now();
                    let queue_ms = started.duration_since(work.enqueued).as_secs_f64() * 1000.;
                    let current = sample(&work.sluice);
                    let prior = previous
                        .as_ref()
                        .filter(|(epoch, _)| *epoch == work.epoch)
                        .map(|(_, data)| data)
                        .unwrap_or(&current);
                    let data = Arc::new(pair_particles(prior, &current));
                    previous = Some((work.epoch, current));
                    if replies
                        .send(Surface {
                            epoch: work.epoch,
                            sim: work.sim,
                            wall: work.wall,
                            data,
                            ms: started.elapsed().as_secs_f64() * 1000.,
                            queue_ms,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .expect("Water surface worker must start");
        Self {
            requests: Some(requests),
            replies: rx,
            thread: Some(thread),
            coalesced: 0,
        }
    }
    pub fn can_submit(&self) -> bool {
        self.requests.is_some()
    }
    pub fn coalesced(&self) -> u64 {
        self.coalesced
    }
    pub fn submit(&mut self, epoch: u64, sim: f64, wall: f64, sluice: &Sluice) {
        let Some(mailbox) = &self.requests else {
            return;
        };
        let work = Work {
            epoch,
            sim,
            wall,
            sluice: Box::new(sluice.clone()),
            enqueued: Instant::now(),
        };
        // The returned render copy drops outside the mailbox lock. A slow
        // reconstruction never creates an unbounded queue or a second worker.
        self.coalesced += u64::from(mailbox.replace(work).is_some());
    }
    pub fn poll(&mut self) -> Option<Surface> {
        match self.replies.try_recv() {
            Ok(surface) => Some(surface),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => panic!("Water surface worker disconnected"),
        }
    }
}

impl Drop for SurfaceWorker {
    fn drop(&mut self) {
        if let Some(mailbox) = self.requests.take() {
            mailbox.close();
        }
        if let Some(handle) = self.thread.take() {
            while !handle.is_finished() {
                let _ = self.replies.recv_timeout(Duration::from_millis(10));
            }
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    fn tiny_sluice(x: f32) -> Sluice {
        let mut s = Sluice::default();
        s.placement = Some(mining_sim::Placement {
            inlet: mining_sim::Vec3::ZERO,
            yaw: 0.,
            slope_deg: 10.,
        });
        s.water.push(mining_sim::sluice::WaterParticle {
            id: 17,
            position: mining_sim::Vec3::new(x, 0.2, 0.2),
            velocity: mining_sim::Vec3::X,
            affine: Default::default(),
        });
        s
    }
    #[test]
    fn pending_work_keeps_newest_snapshot_without_reordering_completed_outputs() {
        let (entered, started) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let count = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&count);
        let mut worker = SurfaceWorker::spawn(move |s| {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                entered.send(()).unwrap();
                gate.recv().unwrap();
            }
            crate::water::samples(s)
        });
        let s = tiny_sluice(0.5);
        worker.submit(1, 1., 1., &s);
        started.recv_timeout(Duration::from_secs(2)).unwrap();
        for sim in [2., 3., 4.] {
            worker.submit(1, sim, sim, &s);
        }
        assert_eq!(worker.coalesced(), 2);
        assert!(worker.can_submit());
        release.send(()).unwrap();
        let first = worker.replies.recv_timeout(Duration::from_secs(2)).unwrap();
        let last = worker.replies.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!((first.epoch, first.sim, first.wall), (1, 1., 1.));
        assert_eq!((last.epoch, last.sim, last.wall), (1, 4., 4.));
        assert_eq!(count.load(Ordering::SeqCst), 2);
        assert!(worker.poll().is_none());
    }
    #[test]
    fn replacement_across_epochs_discards_old_pending_shapes_and_resets_pairing() {
        let (entered, started) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let mut first = true;
        let mut worker = SurfaceWorker::spawn(move |s| {
            if first {
                first = false;
                entered.send(()).unwrap();
                gate.recv().unwrap();
            }
            crate::water::samples(s)
        });
        worker.submit(1, 1., 1., &tiny_sluice(0.5));
        started.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.submit(2, 2., 2., &tiny_sluice(0.6));
        worker.submit(3, 3., 3., &tiny_sluice(0.8));
        release.send(()).unwrap();
        assert_eq!(
            worker
                .replies
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .epoch,
            1
        );
        let latest = worker.replies.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(latest.epoch, 3);
        assert_eq!(latest.data.len(), 1);
        assert_eq!(latest.data[0].previous_center, latest.data[0].center_radius);
        assert_eq!(worker.coalesced(), 1);
    }
    #[test]
    fn shutdown_drains_bounded_unread_outputs_and_the_last_pending_snapshot() {
        let (entered, started) = mpsc::channel();
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let mut worker = SurfaceWorker::spawn(move |s| {
            count.fetch_add(1, Ordering::SeqCst);
            entered.send(()).unwrap();
            crate::water::samples(s)
        });
        for sim in [1., 2., 3.] {
            worker.submit(1, sim, sim, &Sluice::default());
            started.recv_timeout(Duration::from_secs(2)).unwrap();
        }
        // Two outputs are unread; the third producer is finishing or blocked.
        worker.submit(1, 4., 4., &Sluice::default());
        drop(worker);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
    }
}

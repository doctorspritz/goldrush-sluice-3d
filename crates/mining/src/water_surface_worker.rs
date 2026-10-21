//! One bounded render-only worker. Expensive covariance/optics reconstruction
//! and particle correspondence never execute on the drawing thread.
use crate::{water::InstanceData, water_presentation::pair_particles};
use mining_sim::Sluice;
use std::{
    sync::{mpsc, Arc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

struct Work {
    epoch: u64,
    sim: f64,
    wall: f64,
    sluice: Box<Sluice>,
}
pub(crate) struct Surface {
    pub epoch: u64,
    pub sim: f64,
    pub wall: f64,
    pub data: Arc<Vec<InstanceData>>,
    pub ms: f64,
}
pub(crate) struct SurfaceWorker {
    requests: Option<mpsc::SyncSender<Work>>,
    replies: mpsc::Receiver<Surface>,
    thread: Option<JoinHandle<()>>,
    outstanding: usize,
}
impl Default for SurfaceWorker {
    fn default() -> Self {
        let (tx, requests) = mpsc::sync_channel::<Work>(2);
        let (replies, rx) = mpsc::sync_channel(2);
        let thread = thread::Builder::new()
            .name("sluice-water-surface".into())
            .spawn(move || {
                let mut previous: Option<(u64, Vec<(u64, InstanceData)>)> = None;
                while let Ok(work) = requests.recv() {
                    let started = Instant::now();
                    let current = crate::water::samples(&work.sluice);
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
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .expect("Water surface worker must start");
        Self {
            requests: Some(tx),
            replies: rx,
            thread: Some(thread),
            outstanding: 0,
        }
    }
}
impl SurfaceWorker {
    pub fn can_submit(&self) -> bool {
        self.outstanding < 2
    }
    pub fn submit(&mut self, epoch: u64, sim: f64, wall: f64, sluice: &Sluice) {
        if !self.can_submit() {
            return;
        }
        let work = Work {
            epoch,
            sim,
            wall,
            sluice: Box::new(sluice.clone()),
        };
        assert!(
            self.requests.as_ref().unwrap().try_send(work).is_ok(),
            "Bounded surface queue"
        );
        self.outstanding += 1;
    }
    pub fn poll(&mut self) -> Option<Surface> {
        match self.replies.try_recv() {
            Ok(surface) => {
                self.outstanding -= 1;
                Some(surface)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => panic!("Water surface worker disconnected"),
        }
    }
}

impl Drop for SurfaceWorker {
    fn drop(&mut self) {
        self.requests.take();
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
    #[test]
    fn pending_surface_work_is_bounded_and_shutdown_drains_unread_replies() {
        let mut worker = SurfaceWorker::default();
        let sluice = Sluice::default();
        worker.submit(1, 0., 0., &sluice);
        worker.submit(2, 0., 0., &sluice);
        assert!(!worker.can_submit());
        // Drop with both replies unread: the producer must finish normally.
        drop(worker);
    }
}

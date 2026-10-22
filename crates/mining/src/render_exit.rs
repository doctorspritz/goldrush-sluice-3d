//! Drain the last pipelined frame before Winit clears the main world. Bevy
//! 0.17.3's channel Drop waits without servicing MainThreadExecutor; the render
//! schedule may still need that executor. See the retained sluice-render-exit
//! native stack. Keep normal pipelined rendering and all resource destructors.
use bevy::{
    ecs::schedule::MainThreadExecutor,
    prelude::*,
    render::pipelined_rendering::{RenderAppChannels, RenderExtractApp},
    tasks::ComputeTaskPool,
};
use std::future::Future;

pub struct RenderExitPlugin;
impl Plugin for RenderExitPlugin {
    fn build(&self, app: &mut App) {
        let Some(render) = app.get_sub_app_mut(RenderExtractApp) else {
            return;
        };
        let Some(mut extract) = render.take_extract() else {
            return;
        };
        let mut finished = false;
        render.set_extract(move |main, render_world| {
            if finished {
                return;
            }
            extract(main, render_world);
            if main
                .get_resource::<Messages<AppExit>>()
                .is_some_and(|messages| !messages.is_empty())
            {
                finish_frame(main);
                finished = true;
            }
        });
    }
}

fn finish_with_executor<T: Send + 'static>(
    executor: &MainThreadExecutor,
    future: impl Future<Output = T> + Send,
) -> T {
    ComputeTaskPool::get()
        .scope_with_executor(true, Some(&*executor.0), |scope| scope.spawn(future))
        .pop()
        .expect("one final render result")
}

fn finish_frame(main: &mut World) {
    let Some(mut channels) = main.remove_resource::<RenderAppChannels>() else {
        return;
    };
    let executor = main.resource::<MainThreadExecutor>().clone();
    let render = finish_with_executor(&executor, channels.recv());
    if render.is_none() {
        main.write_message(AppExit::error());
    }
    // Close the render request channel, then destroy the returned non-send
    // world on its owner thread. Its final commands and GPU work were serviced.
    drop(channels);
    drop(render);
    println!("RENDER SHUTDOWN: final frame drained on the main thread");
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::app::SubApp;
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Arc,
        },
        thread,
        time::Duration,
    };

    #[test]
    fn final_render_wait_services_work_on_the_executor_owner() {
        ComputeTaskPool::get_or_init(|| bevy::tasks::TaskPoolBuilder::new().num_threads(1).build());
        let (tx, rx) = mpsc::sync_channel(1);
        let handle = thread::spawn(move || {
            let executor = MainThreadExecutor::new();
            let owner = thread::current().id();
            let task = executor
                .0
                .spawn(async move { thread::current().id() == owner });
            let result = finish_with_executor(&executor, task);
            let _ = tx.send(result);
        });
        assert!(rx
            .recv_timeout(Duration::from_secs(3))
            .expect("final frame must not deadlock"));
        handle.join().unwrap();
    }

    #[test]
    fn normal_extraction_runs_until_exit_and_is_not_repeated_after_drain() {
        let count = Arc::new(AtomicUsize::new(0));
        let observed = count.clone();
        let mut app = App::new();
        app.add_message::<AppExit>();
        let mut render = SubApp::new();
        render.set_extract(move |_, _| {
            observed.fetch_add(1, Ordering::SeqCst);
        });
        app.insert_sub_app(RenderExtractApp, render);
        app.add_plugins(RenderExitPlugin);
        let mut render = app.remove_sub_app(RenderExtractApp).unwrap();
        render.extract(app.world_mut());
        assert_eq!(count.load(Ordering::SeqCst), 1);
        app.world_mut().write_message(AppExit::Success);
        render.extract(app.world_mut());
        assert_eq!(count.load(Ordering::SeqCst), 2);
        render.extract(app.world_mut());
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }
}

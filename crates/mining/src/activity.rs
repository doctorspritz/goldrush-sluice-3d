//! A running user-requested simulation should not enter App Nap when its
//! window is covered. Pause releases the activity; idle system sleep is allowed.
use crate::app::Game;
use bevy::prelude::*;
#[cfg(target_os = "macos")]
use objc2::{
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
};
#[cfg(target_os = "macos")]
use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};
#[derive(Default)]
pub struct SimulationActivity {
    #[cfg(target_os = "macos")]
    token: Option<Retained<ProtocolObject<dyn NSObjectProtocol>>>,
}
#[cfg(target_os = "macos")]
impl SimulationActivity {
    fn set_running(&mut self, running: bool) {
        if running && self.token.is_none() {
            self.token = Some(
                NSProcessInfo::processInfo().beginActivityWithOptions_reason(
                    NSActivityOptions::UserInitiatedAllowingIdleSystemSleep,
                    &NSString::from_str("Running the mining simulation"),
                ),
            );
        } else if !running {
            self.end();
        }
    }
    fn end(&mut self) {
        if let Some(token) = self.token.take() {
            // SAFETY: This token came from beginActivityWithOptions_reason for
            // this process and is ended exactly once, before its final release.
            unsafe {
                NSProcessInfo::processInfo().endActivity(&token);
            }
        }
    }
}
#[cfg(target_os = "macos")]
impl Drop for SimulationActivity {
    fn drop(&mut self) {
        self.end();
    }
}
#[cfg(target_os = "macos")]
pub fn sync(game: Res<Game>, mut activity: NonSendMut<SimulationActivity>) {
    activity.set_running(!game.session.paused);
}
#[cfg(not(target_os = "macos"))]
pub fn sync(_: Res<Game>, _: NonSendMut<SimulationActivity>) {}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    #[test]
    fn active_simulation_reuses_and_releases_its_activity() {
        let mut activity = SimulationActivity::default();
        activity.set_running(true);
        let token = activity.token.as_ref().unwrap().clone();
        activity.set_running(true);
        assert!(std::ptr::eq::<ProtocolObject<dyn NSObjectProtocol>>(
            token.as_ref(),
            activity.token.as_ref().unwrap().as_ref()
        ));
        activity.set_running(false);
        assert!(activity.token.is_none());
        activity.set_running(false);
        activity.set_running(true);
        assert!(activity.token.is_some());
        activity.set_running(false);
    }
}

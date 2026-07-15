//! Internal child-process isolation and termination helpers.

use std::process::{Child, Command};

#[cfg(all(test, unix))]
static SUBPROCESS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Serializes unit tests that create and terminate isolated process groups.
#[cfg(all(test, unix))]
pub(crate) fn lock_subprocess_tests() -> std::sync::MutexGuard<'static, ()> {
    SUBPROCESS_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Starts a child in its own process group on Unix so inherited pipe handles
/// cannot outlive a timeout indefinitely.
pub(crate) fn isolate_process_tree(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        command.process_group(0);
    }
}

/// Terminates the child and every process in its isolated group, then reaps it.
pub(crate) fn terminate_process_tree(child: &mut Child) {
    terminate_process_group(child.id());
    let _ = child.kill();
    let _ = child.wait();
}

/// Terminates descendants that retained an inherited pipe after the direct
/// child exited. This is a no-op when the group is already empty.
pub(crate) fn terminate_process_group(process_group_id: u32) {
    #[cfg(unix)]
    if let Ok(raw_pid) = i32::try_from(process_group_id) {
        if let Some(process_group) = rustix::process::Pid::from_raw(raw_pid) {
            let _ =
                rustix::process::kill_process_group(process_group, rustix::process::Signal::KILL);
        }
    }
}

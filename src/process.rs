//! Internal child-process isolation and termination helpers.

use std::io;
use std::process::{Child, Command};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::Duration;

const MAX_OUTPUT_DRAIN_WAIT: Duration = Duration::from_secs(1);

#[cfg(all(test, unix))]
static SUBPROCESS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Serializes unit tests that create and terminate isolated process groups.
#[cfg(all(test, unix))]
pub(crate) fn lock_subprocess_tests() -> std::sync::MutexGuard<'static, ()> {
    SUBPROCESS_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Starts a child in its own process group on Unix.
pub(crate) fn isolate_process_tree(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        command.process_group(0);
    }
}

/// Terminates the isolated Unix process group and the direct child, then reaps it.
///
/// Platforms without a safe standard-library process-tree primitive still
/// terminate the direct child. Deadline-aware readers prevent any surviving
/// descendant that retained a pipe from blocking the caller indefinitely.
pub(crate) fn terminate_process_tree(child: &mut Child) {
    terminate_process_group(child.id());
    let _ = child.kill();
    let _ = child.wait();
}

/// Terminates Unix descendants that retained an inherited pipe after the
/// direct child exited. This is a no-op on other platforms or when the group
/// is already empty.
pub(crate) fn terminate_process_group(process_group_id: u32) {
    #[cfg(unix)]
    if let Ok(raw_pid) = i32::try_from(process_group_id) {
        if let Some(process_group) = rustix::process::Pid::from_raw(raw_pid) {
            let _ =
                rustix::process::kill_process_group(process_group, rustix::process::Signal::KILL);
        }
    }
}

/// Result of waiting for a background output reader.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ReaderWaitError {
    /// The reader did not finish before the caller's deadline.
    TimedOut,
    /// The reader thread stopped without returning a result.
    Panicked,
}

/// A background reader whose result can be collected without an unbounded join.
pub(crate) struct ReaderThread<T> {
    receiver: Receiver<T>,
}

impl<T> ReaderThread<T> {
    /// Waits at most `timeout` for the reader result.
    pub(crate) fn wait(self, timeout: Duration) -> Result<T, ReaderWaitError> {
        match self.receiver.recv_timeout(timeout) {
            Ok(result) => Ok(result),
            Err(RecvTimeoutError::Timeout) => Err(ReaderWaitError::TimedOut),
            Err(RecvTimeoutError::Disconnected) => Err(ReaderWaitError::Panicked),
        }
    }
}

/// Runs an output reader on a named thread and returns a deadline-aware handle.
///
/// Dropping the handle detaches a reader that is still blocked on an inherited
/// pipe. This is preferable to allowing an untrusted subprocess descendant to
/// keep the main process blocked after the direct child has exited or been
/// terminated.
pub(crate) fn spawn_reader_thread<T, F>(
    name: impl Into<String>,
    read: F,
) -> io::Result<ReaderThread<T>>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::Builder::new().name(name.into()).spawn(move || {
        let result = read();
        let _ = sender.send(result);
    })?;
    Ok(ReaderThread { receiver })
}

/// Caps the time spent draining pipes after the direct child has exited.
pub(crate) fn output_drain_timeout(remaining_command_time: Duration) -> Duration {
    remaining_command_time.min(MAX_OUTPUT_DRAIN_WAIT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_reader_returns_completed_result() {
        let reader =
            spawn_reader_thread("torch-check-reader-test", || 42).expect("reader thread starts");
        assert_eq!(reader.wait(Duration::from_secs(1)), Ok(42));
    }

    #[test]
    fn output_reader_wait_is_bounded() {
        let reader = spawn_reader_thread("torch-check-reader-timeout-test", || {
            thread::sleep(Duration::from_millis(100));
            42
        })
        .expect("reader thread starts");
        assert_eq!(
            reader.wait(Duration::from_millis(1)),
            Err(ReaderWaitError::TimedOut)
        );
    }

    #[test]
    fn output_drain_wait_is_capped() {
        assert_eq!(
            output_drain_timeout(Duration::from_secs(60)),
            MAX_OUTPUT_DRAIN_WAIT
        );
        assert_eq!(
            output_drain_timeout(Duration::from_millis(20)),
            Duration::from_millis(20)
        );
    }
}

//! A persistent worker pool: `run` hands `tasks` indices to the workers and
//! the calling thread, and returns once every task has finished, so the
//! task closure may borrow the caller's data.

use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long a worker that finished a job, and a caller waiting for the
/// last tasks, poll before sleeping. A decode step runs a pooled Kernel
/// every few hundred microseconds: workers polling between them start the
/// next one at once, where a sleeping worker's wake-up (tens to hundreds of
/// microseconds) left the caller running most of its tasks alone.
const SPIN: Duration = Duration::from_micros(500);

/// Polls `ready` until it holds or [`SPIN`] has passed; whether it holds.
fn spin_until(ready: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    let mut polls = 0_u32;
    while !ready() {
        std::hint::spin_loop();
        polls = polls.wrapping_add(1);
        if polls.is_multiple_of(256) && start.elapsed() >= SPIN {
            return ready();
        }
    }
    true
}

type Task = dyn Fn(usize) + Sync;

struct Job {
    /// Lifetime-extended: only called while `run` blocks on `pending`.
    task: *const Task,
    tasks: usize,
    next: AtomicUsize,
    pending: AtomicUsize,
    finished: Mutex<bool>,
    done: Condvar,
}

// SAFETY: `task` points to a `Sync` closure, called concurrently only
// while the `run` call that owns it is blocked waiting for completion.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    fn work(&self) {
        loop {
            let index = self.next.fetch_add(1, Ordering::AcqRel);
            if index >= self.tasks {
                return;
            }
            // SAFETY: `index < tasks`, so `run` has not returned yet (it
            // waits for every claimed task to finish).
            unsafe { (*self.task)(index) };
            if self.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
                *self.finished.lock().unwrap() = true;
                self.done.notify_all();
            }
        }
    }
}

#[derive(Default)]
struct State {
    job: Option<Arc<Job>>,
    generation: u64,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    /// `State::generation`, readable without the lock, for polling workers.
    posted: AtomicU64,
    stopping: AtomicBool,
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// One `run` at a time.
    running: Mutex<()>,
}

impl Pool {
    /// A pool running tasks on `threads` threads in all (the caller and
    /// `threads - 1` workers).
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            posted: AtomicU64::new(0),
            stopping: AtomicBool::new(false),
        });
        let workers = (1..threads.max(1))
            .map(|_| {
                let shared = shared.clone();
                std::thread::spawn(move || {
                    let mut seen = 0;
                    loop {
                        spin_until(|| {
                            shared.posted.load(Ordering::Acquire) != seen
                                || shared.stopping.load(Ordering::Acquire)
                        });
                        let job = {
                            let mut state = shared.state.lock().unwrap();
                            while !state.shutdown && state.generation == seen {
                                state = shared.wake.wait(state).unwrap();
                            }
                            if state.shutdown {
                                return;
                            }
                            seen = state.generation;
                            state.job.clone()
                        };
                        if let Some(job) = job {
                            job.work();
                        }
                    }
                })
            })
            .collect();
        Self {
            shared,
            workers,
            running: Mutex::new(()),
        }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Calls `task(i)` for every `i < tasks` across the pool and returns
    /// when all have finished.
    pub fn run<'a>(&self, tasks: usize, task: &'a (dyn Fn(usize) + Sync + 'a)) {
        if tasks == 0 {
            return;
        }
        if tasks == 1 || self.workers.is_empty() {
            (0..tasks).for_each(task);
            return;
        }
        let _running = self.running.lock().unwrap();
        // SAFETY: the reference only outlives this call inside `Job`, and
        // `Job::work` stops calling it once every index is claimed; this
        // call returns only after every claimed task has finished.
        let task: &'static Task =
            unsafe { std::mem::transmute::<&'a (dyn Fn(usize) + Sync + 'a), &'static Task>(task) };
        let job = Arc::new(Job {
            task,
            tasks,
            next: AtomicUsize::new(0),
            pending: AtomicUsize::new(tasks),
            finished: Mutex::new(false),
            done: Condvar::new(),
        });
        {
            let mut state = self.shared.state.lock().unwrap();
            state.job = Some(job.clone());
            state.generation += 1;
            self.shared
                .posted
                .store(state.generation, Ordering::Release);
        }
        self.shared.wake.notify_all();
        job.work();
        if !spin_until(|| job.pending.load(Ordering::Acquire) == 0) {
            let mut finished = job.finished.lock().unwrap();
            while !*finished {
                finished = job.done.wait(finished).unwrap();
            }
        }
        self.shared.state.lock().unwrap().job = None;
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().shutdown = true;
        self.shared.stopping.store(true, Ordering::Release);
        self.shared.wake.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

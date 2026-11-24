//! The physical cores this process may run on, for the decode pool
//! ([`super::DECODE_THREADS_ENV`] `=cores`): a decode step streams weights
//! from memory, and on hosts with two hardware threads per core one thread
//! per core can decode faster than every logical CPU (Intel Xeon 6, run
//! 45406355532), as llama.cpp's default thread count assumes.

#[cfg(any(target_os = "linux", test))]
use std::collections::BTreeSet;

/// One thread per physical core this process may run on, never more than
/// the logical CPUs it may use (affinity and CPU quota); every logical CPU
/// where the platform does not say which ones share a core.
pub(super) fn one_per_core() -> usize {
    let logical = std::thread::available_parallelism().map_or(1, |count| count.get());
    physical_cores().map_or(logical, |cores| cores.clamp(1, logical))
}

/// The physical cores among the CPUs this process may run on: the distinct
/// sets of hardware threads sharing a core, as Linux reports them.
#[cfg(target_os = "linux")]
fn physical_cores() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let allowed = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?;
    let mut cores = BTreeSet::new();
    for cpu in cpu_list(allowed.trim())? {
        let siblings = std::fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ))
        .ok()?;
        cores.insert(cpu_list(siblings.trim())?);
    }
    (!cores.is_empty()).then_some(cores.len())
}

#[cfg(not(target_os = "linux"))]
fn physical_cores() -> Option<usize> {
    None
}

/// The CPUs of a Linux CPU list (`0-3,8,10-11`), ascending; `None` when
/// it does not parse.
#[cfg(any(target_os = "linux", test))]
pub(super) fn cpu_list(list: &str) -> Option<BTreeSet<usize>> {
    let mut cpus = BTreeSet::new();
    for part in list.split(',').filter(|part| !part.is_empty()) {
        let (first, last): (usize, usize) = match part.split_once('-') {
            Some((first, last)) => (first.parse().ok()?, last.parse().ok()?),
            None => {
                let cpu = part.parse().ok()?;
                (cpu, cpu)
            }
        };
        if first > last {
            return None;
        }
        cpus.extend(first..=last);
    }
    Some(cpus)
}

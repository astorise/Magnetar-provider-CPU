//! Step capture and replay on the optimized CPU Provider
//! (`capture-cpu-steps`): a capture records the step's Kernel invocations
//! as the Runtime dispatches them; a replay runs them again directly, every
//! advancing attribute increased by the replay offset, without the
//! Runtime's per-node dispatch (plan resolution, admission, observations).
//!
//! Every resource lives in host storage under its identity, so a replay
//! reads and writes the same resources a per-node run would, whatever
//! storage they have now: a replay is never stale. A host write during the
//! capture (data a replay would not repeat) fails the capture.

use super::*;
use magnetar_provider_api::{StepCaptureId, StepReplayError};

#[derive(Default)]
pub(super) struct Recording {
    pub(super) advancing: Vec<String>,
    /// The invocations the step dispatched, in order.
    pub(super) invocations: Vec<KernelInvocation>,
    /// Why the capture cannot be replayed, if something made it so.
    pub(super) failure: Option<String>,
}

#[derive(Default)]
pub(super) struct CaptureState {
    pub(super) recording: Option<Recording>,
    pub(super) captures: BTreeMap<StepCaptureId, Arc<Recording>>,
    pub(super) next: u64,
}

impl CaptureState {
    /// Fails the capture in progress, if any, because of `what`.
    pub(super) fn poison(&mut self, what: &str) {
        if let Some(recording) = &mut self.recording {
            recording
                .failure
                .get_or_insert_with(|| format!("{what} during the capture"));
        }
    }
}

/// `invocation` with every integer attribute named in `advancing`
/// increased by `offset`.
pub(super) fn advanced(
    invocation: &KernelInvocation,
    advancing: &[String],
    offset: u64,
) -> Result<KernelInvocation, StepReplayError> {
    let mut invocation = invocation.clone();
    if offset == 0 {
        return Ok(invocation);
    }
    let offset = i64::try_from(offset).map_err(|_| StepReplayError::Invalidated {
        reason: format!("replay offset {offset} does not fit an attribute"),
    })?;
    for name in advancing {
        if let Some(OperatorAttributeValue::Integer(value)) = invocation.attributes.get_mut(name) {
            *value = value
                .checked_add(offset)
                .ok_or_else(|| StepReplayError::Invalidated {
                    reason: format!("attribute {name} overflows at offset {offset}"),
                })?;
        }
    }
    Ok(invocation)
}

/// A replay failure, as the Provider reports it.
pub(super) fn replay_failed(reason: String) -> StepReplayError {
    StepReplayError::Failed(ProviderExecutionError::new(
        ProviderExecutionErrorCode::ExecutionFailed,
        ProviderExecutionPhase::Complete,
        ProviderBinding::new(NATIVE_CPU_PROVIDER_NAME),
        None,
        reason,
    ))
}

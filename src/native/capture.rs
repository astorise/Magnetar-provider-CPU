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
    /// How a replay runs the invocations, worked out at the first replay.
    pub(super) schedule: std::sync::OnceLock<Vec<Scheduled>>,
}

/// One step of a replay: recorded invocations by index, run in this
/// order; several when they are dense products of one input run in one
/// pool run (`NativeState::schedule`).
pub(super) struct Scheduled {
    pub(super) invocations: Vec<usize>,
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
/// increased by `offset`: borrowed when it has none to increase (a step's
/// matmuls), so a replay copies only the invocations it changes.
pub(super) fn advanced<'a>(
    invocation: &'a KernelInvocation,
    advancing: &[String],
    offset: u64,
) -> Result<std::borrow::Cow<'a, KernelInvocation>, StepReplayError> {
    let changes = offset != 0
        && advancing.iter().any(|name| {
            matches!(
                invocation.attributes.get(name),
                Some(OperatorAttributeValue::Integer(_))
            )
        });
    if !changes {
        return Ok(std::borrow::Cow::Borrowed(invocation));
    }
    let mut invocation = invocation.clone();
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
    Ok(std::borrow::Cow::Owned(invocation))
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

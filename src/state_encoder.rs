// State delta encoder: serializes ActorState deltas into compact binary frames
// for over-the-wire transmission. Uses bincode for fast zero-copy encoding.
// The encoder maintains a pre-allocated reusable buffer to avoid heap churn
// on the hot broadcast path.

use crate::types::{now_ms, ActorId, ActorState, StateDelta};
use anyhow::Result;

pub struct StateEncoder {
    buf: Vec<u8>,
}

impl StateEncoder {
    /// Create an encoder with a pre-allocated buffer of given initial capacity.
    pub fn new(initial_cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(initial_cap),
        }
    }

    /// Encode a state delta into the internal buffer and return a reference.
    /// The returned slice is valid until the next call to encode().
    pub fn encode(&mut self, delta: &StateDelta) -> Result<&[u8]> {
        self.buf.clear();
        bincode::serialize_into(&mut self.buf, delta)?;
        Ok(&self.buf)
    }

    /// Build a StateDelta from the given actor states and removed IDs.
    pub fn build_delta(tick: u64, updates: Vec<ActorState>, removed: Vec<ActorId>) -> StateDelta {
        StateDelta {
            tick,
            timestamp_ms: now_ms(),
            updates,
            removed,
        }
    }

    /// Decode a binary frame back into a StateDelta (used by client / tests).
    pub fn decode(bytes: &[u8]) -> Result<StateDelta> {
        bincode::deserialize(bytes).map_err(Into::into)
    }
}

/// Compute delta: only actors whose state changed since the last snapshot.
/// `previous` and `current` are both sorted by actor ID.
/// Returns (changed actors, removed actor IDs).
pub fn diff_states(
    previous: &[ActorState],
    current: &[ActorState],
) -> (Vec<ActorState>, Vec<ActorId>) {
    use std::collections::HashMap;

    let prev_map: HashMap<ActorId, &ActorState> = previous.iter().map(|s| (s.id, s)).collect();
    let curr_map: HashMap<ActorId, &ActorState> = current.iter().map(|s| (s.id, s)).collect();

    let mut changed = Vec::new();
    for (&id, &curr) in &curr_map {
        match prev_map.get(&id) {
            None => changed.push(curr.clone()),
            Some(&prev) => {
                // Include if position or utterance changed.
                if (prev.position.x - curr.position.x).abs() > f32::EPSILON
                    || (prev.position.y - curr.position.y).abs() > f32::EPSILON
                    || prev.last_utterance != curr.last_utterance
                    || prev.tick != curr.tick
                {
                    changed.push(curr.clone());
                }
            }
        }
    }

    let removed: Vec<ActorId> = prev_map
        .keys()
        .filter(|id| !curr_map.contains_key(id))
        .copied()
        .collect();

    (changed, removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActorId, GridCell, Position};

    fn make_state(id: u64, x: f32, y: f32, tick: u64) -> ActorState {
        ActorState {
            id: ActorId(id),
            name: format!("A{id}"),
            position: Position::new(x, y),
            cell: GridCell(0, 0),
            tick,
            last_utterance: None,
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let mut enc = StateEncoder::new(4096);
        let delta =
            StateEncoder::build_delta(42, vec![make_state(1, 1.0, 2.0, 42)], vec![ActorId(99)]);
        let bytes = enc.encode(&delta).unwrap().to_vec();
        let back = StateEncoder::decode(&bytes).unwrap();
        assert_eq!(back.tick, 42);
        assert_eq!(back.updates.len(), 1);
        assert_eq!(back.removed, vec![ActorId(99)]);
        assert_eq!(back.updates[0].id, ActorId(1));
    }

    #[test]
    fn diff_detects_moved_actor() {
        let prev = vec![make_state(1, 0.0, 0.0, 1)];
        let curr = vec![make_state(1, 10.0, 0.0, 2)];
        let (changed, removed) = diff_states(&prev, &curr);
        assert_eq!(changed.len(), 1);
        assert!(removed.is_empty());
    }

    #[test]
    fn diff_detects_removed_actor() {
        let prev = vec![make_state(1, 0.0, 0.0, 1), make_state(2, 5.0, 5.0, 1)];
        let curr = vec![make_state(1, 0.0, 0.0, 2)];
        let (changed, removed) = diff_states(&prev, &curr);
        assert!(removed.contains(&ActorId(2)));
        // Actor 1 changed (tick bumped)
        assert_eq!(changed.len(), 1);
    }

    #[test]
    fn diff_no_change_is_empty() {
        let states = vec![make_state(1, 0.0, 0.0, 5)];
        let (changed, removed) = diff_states(&states, &states);
        assert!(changed.is_empty());
        assert!(removed.is_empty());
    }

    #[test]
    fn buffer_reuse_across_ticks() {
        let mut enc = StateEncoder::new(256);
        for tick in 0..100u64 {
            let delta =
                StateEncoder::build_delta(tick, vec![make_state(1, 0.0, 0.0, tick)], vec![]);
            let _bytes = enc.encode(&delta).unwrap();
        }
        // No panic = buffer reuse works correctly
    }
}

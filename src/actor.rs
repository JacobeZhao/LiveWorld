// Per-actor state and lifecycle.
// An Actor lives in its own thread / async task and owns its message queue consumer.
// The hot path (process_message) must not block.

use crate::spsc_queue::{spsc_queue, SpscConsumer, SpscProducer};
use crate::types::{ActorId, ActorMessage, ActorSpec, ActorState, GridCell, Position};

const QUEUE_SIZE: usize = 1024; // power of 2

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActorLifecycle {
    Spawning,
    Active,
    ShuttingDown,
    Dead,
}

/// All mutable state owned by a single Actor.
pub struct Actor {
    pub spec: ActorSpec,
    pub state: ActorState,
    pub lifecycle: ActorLifecycle,
    /// Inbox consumer — polled by the actor's task.
    inbox: SpscConsumer<ActorMessage, QUEUE_SIZE>,
}

/// Outbox handle handed to external code that wants to send messages to this Actor.
pub struct ActorHandle {
    pub id: ActorId,
    pub(crate) sender: SpscProducer<ActorMessage, QUEUE_SIZE>,
}

impl Clone for ActorHandle {
    fn clone(&self) -> Self {
        ActorHandle {
            id: self.id,
            sender: self.sender.clone(),
        }
    }
}

impl ActorHandle {
    /// Non-blocking send. Returns false if the inbox is full.
    #[inline]
    pub fn send(&self, msg: ActorMessage) -> bool {
        self.sender.push(msg)
    }
}

impl Actor {
    /// Create a new Actor from a spec. Returns the actor and its external handle.
    pub fn spawn(spec: ActorSpec) -> (Actor, ActorHandle) {
        let (tx, rx) = spsc_queue::<ActorMessage, QUEUE_SIZE>();
        let id = spec.id;
        let initial_state = ActorState {
            id,
            name: spec.name.clone(),
            position: spec.position,
            cell: spec.position.to_grid_cell(10.0), // default cell_size; overwritten by runtime
            tick: 0,
            last_utterance: None,
        };
        let actor = Actor {
            spec,
            state: initial_state,
            lifecycle: ActorLifecycle::Spawning,
            inbox: rx,
        };
        let handle = ActorHandle { id, sender: tx };
        (actor, handle)
    }

    /// Activate the actor (called once after spatial registration).
    #[inline]
    pub fn activate(&mut self, cell: GridCell) {
        self.state.cell = cell;
        self.lifecycle = ActorLifecycle::Active;
    }

    /// Process all pending messages in the inbox. Called every tick.
    /// Returns a list of side-effects for the world engine to apply.
    pub fn drain_inbox(&mut self) -> Vec<ActorEffect> {
        let mut effects = Vec::new();
        while let Some(msg) = self.inbox.pop() {
            match msg {
                ActorMessage::Move { to } => {
                    effects.push(ActorEffect::Move {
                        id: self.spec.id,
                        to,
                    });
                }
                ActorMessage::Speak { text } => {
                    self.state.last_utterance = Some(text.clone());
                    effects.push(ActorEffect::Speak {
                        id: self.spec.id,
                        text,
                    });
                }
                ActorMessage::Interact { target, action } => {
                    effects.push(ActorEffect::Interact {
                        source: self.spec.id,
                        target,
                        action,
                    });
                }
                ActorMessage::Shutdown => {
                    self.lifecycle = ActorLifecycle::ShuttingDown;
                    break;
                }
            }
        }
        effects
    }

    /// Apply a Move effect to this actor's state (position & cell updated by runtime).
    #[inline]
    pub fn apply_move(&mut self, to: Position, new_cell: GridCell) {
        self.state.position = to;
        self.state.cell = new_cell;
    }

    /// Advance the actor's tick counter.
    #[inline]
    pub fn tick(&mut self, tick: u64) {
        self.state.tick = tick;
    }

    /// Produce an immutable snapshot of this actor's state for broadcasting.
    #[inline]
    pub fn snapshot(&self) -> ActorState {
        self.state.clone()
    }

    #[inline]
    pub fn id(&self) -> ActorId {
        self.spec.id
    }

    #[inline]
    pub fn is_alive(&self) -> bool {
        matches!(
            self.lifecycle,
            ActorLifecycle::Active | ActorLifecycle::Spawning
        )
    }
}

/// Side-effects emitted by drain_inbox, consumed by the world engine.
#[derive(Debug, Clone)]
pub enum ActorEffect {
    Move {
        id: ActorId,
        to: Position,
    },
    Speak {
        id: ActorId,
        text: String,
    },
    Interact {
        source: ActorId,
        target: ActorId,
        action: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActorId, LlmModel, Position};

    fn make_spec(id: u64) -> ActorSpec {
        ActorSpec {
            id: ActorId(id),
            name: format!("Agent{id}"),
            personality: "curious".to_string(),
            backstory: "A wanderer".to_string(),
            model: LlmModel::Mock,
            position: Position::new(5.0, 5.0),
        }
    }

    #[test]
    fn spawn_activates_correctly() {
        let (mut actor, _handle) = Actor::spawn(make_spec(1));
        assert_eq!(actor.lifecycle, ActorLifecycle::Spawning);
        actor.activate(GridCell(0, 0));
        assert_eq!(actor.lifecycle, ActorLifecycle::Active);
        assert!(actor.is_alive());
    }

    #[test]
    fn move_message_produces_effect() {
        let (mut actor, handle) = Actor::spawn(make_spec(2));
        actor.activate(GridCell(0, 0));
        handle.send(ActorMessage::Move {
            to: Position::new(50.0, 50.0),
        });
        let effects = actor.drain_inbox();
        assert_eq!(effects.len(), 1);
        matches!(&effects[0], ActorEffect::Move { .. });
    }

    #[test]
    fn shutdown_message_stops_processing() {
        let (mut actor, handle) = Actor::spawn(make_spec(3));
        actor.activate(GridCell(0, 0));
        handle.send(ActorMessage::Shutdown);
        handle.send(ActorMessage::Move {
            to: Position::new(1.0, 1.0),
        }); // after shutdown
        actor.drain_inbox();
        assert_eq!(actor.lifecycle, ActorLifecycle::ShuttingDown);
    }

    #[test]
    fn speak_updates_utterance() {
        let (mut actor, handle) = Actor::spawn(make_spec(4));
        actor.activate(GridCell(0, 0));
        handle.send(ActorMessage::Speak {
            text: "Hello world".to_string(),
        });
        actor.drain_inbox();
        assert_eq!(actor.state.last_utterance.as_deref(), Some("Hello world"));
    }

    #[test]
    fn snapshot_reflects_current_state() {
        let (mut actor, handle) = Actor::spawn(make_spec(5));
        actor.activate(GridCell(0, 0));
        handle.send(ActorMessage::Move {
            to: Position::new(20.0, 20.0),
        });
        let effects = actor.drain_inbox();
        if let ActorEffect::Move { to, .. } = &effects[0] {
            actor.apply_move(*to, GridCell(2, 2));
        }
        let snap = actor.snapshot();
        assert_eq!(snap.position, Position::new(20.0, 20.0));
        assert_eq!(snap.cell, GridCell(2, 2));
    }
}

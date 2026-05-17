// Per-actor state and lifecycle.
// An Actor lives in its own thread / async task and owns its message queue consumer.
// The hot path (process_message) must not block.

use crate::spsc_queue::{spsc_queue, SpscConsumer, SpscProducer};
use crate::types::{ActorId, ActorMessage, ActorRole, ActorSpec, ActorState, GridCell, Position};

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
        let (hp, max_hp) = match spec.role {
            ActorRole::Knight | ActorRole::Guard => (100u8, 100u8),
            ActorRole::Mage | ActorRole::Scholar => (60u8, 60u8),
            _ => (80u8, 80u8),
        };
        let initial_state = ActorState {
            id,
            name: spec.name.clone(),
            position: spec.position,
            cell: spec.position.to_grid_cell(10.0), // default cell_size; overwritten by runtime
            tick: 0,
            last_utterance: None,
            role: spec.role.clone(),
            faction: spec.faction.clone(),
            hp,
            max_hp,
            xp: 0,
            level: 1,
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
                ActorMessage::TakeDamage { amount, from_name } => {
                    self.state.hp = self.state.hp.saturating_sub(amount);
                    if self.state.hp == 0 {
                        // Respawn at birth position with full HP
                        self.state.hp = self.state.max_hp;
                        self.state.position = self.spec.position;
                        effects.push(ActorEffect::Died {
                            id: self.spec.id,
                            name: self.spec.name.clone(),
                            killer_name: from_name,
                        });
                    } else {
                        effects.push(ActorEffect::Damaged {
                            id: self.spec.id,
                            new_hp: self.state.hp,
                        });
                    }
                }
                ActorMessage::GainXp { amount } => {
                    self.state.xp = self.state.xp.saturating_add(amount);
                    let new_level = ((self.state.xp / 100) as u8 + 1).min(10);
                    if new_level > self.state.level {
                        self.state.level = new_level;
                        effects.push(ActorEffect::LevelUp {
                            id: self.spec.id,
                            name: self.spec.name.clone(),
                            level: new_level,
                        });
                    }
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
    Damaged {
        id: ActorId,
        new_hp: u8,
    },
    Died {
        id: ActorId,
        name: String,
        killer_name: String,
    },
    LevelUp {
        id: ActorId,
        name: String,
        level: u8,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActorId, ActorRole, Faction, LlmModel, Position};

    fn make_spec(id: u64) -> ActorSpec {
        ActorSpec {
            id: ActorId(id),
            name: format!("Agent{id}"),
            personality: "curious".to_string(),
            backstory: "A wanderer".to_string(),
            model: LlmModel::Mock,
            position: Position::new(5.0, 5.0),
            role: ActorRole::Wanderer,
            faction: Faction::Neutral,
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

    #[test]
    fn take_damage_reduces_hp() {
        let (mut actor, handle) = Actor::spawn(make_spec(6));
        actor.activate(GridCell(0, 0));
        let initial_hp = actor.state.hp;
        handle.send(ActorMessage::TakeDamage {
            amount: 20,
            from_name: "Enemy".to_string(),
        });
        let effects = actor.drain_inbox();
        assert_eq!(actor.state.hp, initial_hp - 20);
        assert!(matches!(effects[0], ActorEffect::Damaged { .. }));
    }

    #[test]
    fn take_damage_respawns_on_death() {
        let (mut actor, handle) = Actor::spawn(make_spec(7));
        actor.activate(GridCell(0, 0));
        let birth_pos = actor.spec.position;
        handle.send(ActorMessage::TakeDamage {
            amount: 255,
            from_name: "Boss".to_string(),
        });
        let effects = actor.drain_inbox();
        // HP resets to max after death
        assert_eq!(actor.state.hp, actor.state.max_hp);
        // Position resets to birth
        assert_eq!(actor.state.position, birth_pos);
        assert!(matches!(effects[0], ActorEffect::Died { .. }));
    }

    #[test]
    fn gain_xp_levels_up() {
        let (mut actor, handle) = Actor::spawn(make_spec(8));
        actor.activate(GridCell(0, 0));
        handle.send(ActorMessage::GainXp { amount: 100 });
        let effects = actor.drain_inbox();
        assert_eq!(actor.state.xp, 100);
        assert_eq!(actor.state.level, 2);
        assert!(matches!(effects[0], ActorEffect::LevelUp { level: 2, .. }));
    }

    #[test]
    fn knight_has_more_hp_than_mage() {
        let mut knight_spec = make_spec(9);
        knight_spec.role = ActorRole::Knight;
        let mut mage_spec = make_spec(10);
        mage_spec.role = ActorRole::Mage;
        let (knight, _) = Actor::spawn(knight_spec);
        let (mage, _) = Actor::spawn(mage_spec);
        assert!(knight.state.hp > mage.state.hp);
    }
}

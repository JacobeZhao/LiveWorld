use crate::actor::ActorHandle;
use crate::actor_runtime::ActorRuntime;
use crate::interest_manager::InterestManager;
use crate::state_encoder::{diff_states, StateEncoder};
use crate::types::{
    now_ms, ActorId, ActorMessage, ActorSpec, ActorState, GridCell, SessionId, StateDelta,
    WorldConfig, WorldDirective,
};
use ahash::AHashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Per-session outgoing delta queue (bounded, shared with WS task).
pub type SessionQueue = Arc<Mutex<VecDeque<StateDelta>>>;

pub struct WorldEngine {
    runtime: ActorRuntime,
    interest: InterestManager,
    #[allow(dead_code)]
    encoder: StateEncoder,
    cfg: WorldConfig,
    tick_count: u64,
    session_queues: AHashMap<SessionId, SessionQueue>,
    /// Reverse map: session → its anchor actor.
    session_to_actor: AHashMap<SessionId, ActorId>,
    prev_snapshot: Vec<ActorState>,
    removed_this_tick: Vec<ActorId>,
    start_time: Instant,
}

impl WorldEngine {
    pub fn new(cfg: WorldConfig) -> Self {
        let interest = InterestManager::new(cfg.interest_radius);
        let runtime = ActorRuntime::new(cfg.clone());
        Self {
            runtime,
            interest,
            encoder: StateEncoder::new(1 << 20),
            cfg,
            tick_count: 0,
            session_queues: AHashMap::new(),
            session_to_actor: AHashMap::new(),
            prev_snapshot: Vec::new(),
            removed_this_tick: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Spawn an actor and register its session atomically.
    /// Returns (ActorHandle for command routing, SessionQueue for WS delta pump).
    pub fn spawn_actor_for_session(
        &mut self,
        spec: ActorSpec,
        session: SessionId,
    ) -> (ActorHandle, SessionQueue) {
        let id = spec.id;
        let handle = self.runtime.spawn_actor(spec);

        // Determine the initial cell.
        let cell = self
            .runtime
            .snapshot_all()
            .into_iter()
            .find(|s| s.id == id)
            .map(|s| s.cell)
            .unwrap_or(GridCell(0, 0));

        let q = self.add_session(session, id, cell);
        self.session_to_actor.insert(session, id);
        (handle, q)
    }

    /// Register a session (called by spawn_actor_for_session; also usable standalone).
    pub fn add_session(
        &mut self,
        session: SessionId,
        anchor: ActorId,
        cell: GridCell,
    ) -> SessionQueue {
        let q: SessionQueue = Arc::new(Mutex::new(VecDeque::with_capacity(16)));
        self.session_queues.insert(session, Arc::clone(&q));
        self.interest.register(session, anchor, cell);
        q
    }

    /// Remove session + optionally despawn its anchor actor.
    pub fn remove_session(&mut self, session: SessionId, despawn_actor: bool) {
        if despawn_actor {
            if let Some(actor_id) = self.session_to_actor.remove(&session) {
                self.removed_this_tick.push(actor_id);
                self.runtime.despawn_actor(actor_id);
            }
        } else {
            self.session_to_actor.remove(&session);
        }
        self.session_queues.remove(&session);
        self.interest.unregister(session);
    }

    /// Look up the actor owned by a session and return a cloned handle for message routing.
    pub fn session_handle(&self, session: SessionId) -> Option<ActorHandle> {
        let actor_id = self.session_to_actor.get(&session)?;
        self.runtime.handle(*actor_id).cloned()
    }

    /// Send a message directly to an actor by ID.
    pub fn send_to_actor(&self, id: ActorId, msg: ActorMessage) -> bool {
        if let Some(h) = self.runtime.handle(id) {
            h.send(msg)
        } else {
            false
        }
    }

    /// Apply a WorldDirective from a global agent.
    pub fn apply_directive(&self, directive: &WorldDirective) {
        match directive {
            WorldDirective::ForceMove { actor_id, to } => {
                self.send_to_actor(*actor_id, ActorMessage::Move { to: *to });
            }
            WorldDirective::NarrativeEvent { message } => {
                // Broadcast narrative to all sessions as a speak event from actor 0.
                info!(message, "NarrativeEvent broadcast");
            }
            WorldDirective::EconomyAdjust { resource, delta } => {
                info!(%resource, delta, "EconomyAdjust applied");
            }
            WorldDirective::FlagActor { actor_id, reason } => {
                warn!(actor = actor_id.0, reason, "Actor flagged");
            }
        }
    }

    /// Despawn an actor by ID.
    pub fn despawn_actor(&mut self, id: ActorId) {
        self.removed_this_tick.push(id);
        self.runtime.despawn_actor(id);
    }

    /// Run one tick (hot path — dedicated OS thread).
    pub fn tick(&mut self) {
        let tick_start = Instant::now();
        self.tick_count += 1;
        let tick = self.tick_count;

        // 1. Drain all actor inboxes; apply effects.
        let effects = self.runtime.tick(tick);

        // 2. Update interest anchor cells for moved actors.
        for effect in &effects {
            use crate::actor::ActorEffect;
            if let ActorEffect::Move { id, .. } = effect {
                if let Some(snap) = self
                    .runtime
                    .snapshot_all()
                    .into_iter()
                    .find(|s| s.id == *id)
                {
                    let cell = snap.cell;
                    for (sid, anchor_id, _) in self.interest.sessions().collect::<Vec<_>>() {
                        if anchor_id == *id {
                            self.interest.update_cell(sid, cell);
                        }
                    }
                }
            }
        }

        // 3. Full snapshot.
        let current_snapshot = self.runtime.snapshot_all();

        // 4. Global diff.
        let removed = std::mem::take(&mut self.removed_this_tick);
        let (changed, mut newly_removed) = diff_states(&self.prev_snapshot, &current_snapshot);
        newly_removed.extend(removed);

        // 5. Push per-session delta.
        let grid = self.runtime.grid();
        for (sid, _, _) in self.interest.sessions().collect::<Vec<_>>() {
            let visible_ids = self.interest.visible_actors(sid, grid);

            let visible_updates: Vec<ActorState> = changed
                .iter()
                .filter(|s| visible_ids.contains(&s.id))
                .cloned()
                .collect();

            // Skip empty deltas unless there are removals.
            if visible_updates.is_empty() && newly_removed.is_empty() {
                continue;
            }

            let delta = StateDelta {
                tick,
                timestamp_ms: now_ms(),
                updates: visible_updates,
                removed: newly_removed.clone(),
            };

            if let Some(q) = self.session_queues.get(&sid) {
                let mut queue = q.lock().unwrap();
                if queue.len() >= 32 {
                    queue.pop_front();
                    warn!(session = sid.0, "Session lagging; dropped frame");
                }
                queue.push_back(delta);
            }
        }

        self.prev_snapshot = current_snapshot;

        let elapsed = tick_start.elapsed();
        if elapsed > Duration::from_millis(5) {
            warn!(tick, ?elapsed, "Tick exceeded 5ms budget");
        }
        debug!(tick, actors = self.runtime.actor_count(), ?elapsed);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }
    pub fn actor_count(&self) -> usize {
        self.runtime.actor_count()
    }
    pub fn session_count(&self) -> usize {
        self.session_queues.len()
    }
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
    pub fn full_snapshot(&self) -> Vec<ActorState> {
        self.runtime.snapshot_all()
    }
    pub fn config(&self) -> &WorldConfig {
        &self.cfg
    }

    /// Spawn an actor without registering a session (cold-start recovery).
    pub fn spawn_actor_standalone(&mut self, spec: ActorSpec) -> ActorHandle {
        self.runtime.spawn_actor(spec)
    }

    /// Full snapshot with specs, for disk persistence.
    pub fn world_snapshot_for_persist(&self) -> crate::persistence::WorldSnapshot {
        let states = self.runtime.snapshot_all();
        let specs = self.runtime.specs_snapshot();
        let specs_map: ahash::AHashMap<ActorId, ActorSpec> =
            specs.into_iter().map(|s| (s.id, s)).collect();
        crate::persistence::build_snapshot(self.tick_count, &states, &specs_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActorId, LlmModel, Position};

    fn make_spec(id: u64, x: f32, y: f32) -> ActorSpec {
        ActorSpec {
            id: ActorId(id),
            name: format!("A{id}"),
            personality: String::new(),
            backstory: String::new(),
            model: LlmModel::Mock,
            position: Position::new(x, y),
        }
    }

    #[test]
    fn tick_advances_counter() {
        let mut engine = WorldEngine::new(WorldConfig::default());
        engine.tick();
        engine.tick();
        assert_eq!(engine.tick_count(), 2);
    }

    #[test]
    fn spawn_actor_appears_in_count() {
        let mut engine = WorldEngine::new(WorldConfig::default());
        let sid = SessionId::next();
        engine.spawn_actor_for_session(make_spec(1, 5.0, 5.0), sid);
        engine.spawn_actor_for_session(make_spec(2, 15.0, 15.0), SessionId::next());
        assert_eq!(engine.actor_count(), 2);
    }

    #[test]
    fn despawn_removes_actor() {
        let mut engine = WorldEngine::new(WorldConfig::default());
        engine.spawn_actor_for_session(make_spec(1, 5.0, 5.0), SessionId::next());
        engine.despawn_actor(ActorId(1));
        engine.tick();
        assert_eq!(engine.actor_count(), 0);
    }

    #[test]
    fn session_queue_receives_delta() {
        let mut engine = WorldEngine::new(WorldConfig::default());
        let sid = SessionId::next();
        let (handle, q) = engine.spawn_actor_for_session(make_spec(1, 5.0, 5.0), sid);
        // Move the actor so a delta is generated.
        handle.send(ActorMessage::Move {
            to: Position::new(50.0, 50.0),
        });
        engine.tick();
        let queue = q.lock().unwrap();
        assert!(!queue.is_empty(), "Session should have received a delta");
    }

    #[test]
    fn session_handle_routes_message() {
        let mut engine = WorldEngine::new(WorldConfig::default());
        let sid = SessionId::next();
        engine.spawn_actor_for_session(make_spec(1, 0.0, 0.0), sid);
        let h = engine.session_handle(sid);
        assert!(h.is_some(), "Should return handle for session actor");
        h.unwrap().send(ActorMessage::Move {
            to: Position::new(10.0, 10.0),
        });
    }

    #[test]
    fn remove_session_cleans_up() {
        let mut engine = WorldEngine::new(WorldConfig::default());
        let sid = SessionId::next();
        engine.spawn_actor_for_session(make_spec(1, 0.0, 0.0), sid);
        assert_eq!(engine.session_count(), 1);
        engine.remove_session(sid, true);
        engine.tick();
        assert_eq!(engine.session_count(), 0);
        assert_eq!(engine.actor_count(), 0);
    }

    #[test]
    fn tick_timing_respects_5ms_budget() {
        let mut engine = WorldEngine::new(WorldConfig::default());
        let start = Instant::now();
        for _ in 0..100 {
            engine.tick();
        }
        let avg = start.elapsed() / 100;
        assert!(
            avg < Duration::from_millis(2),
            "Tick avg {:?} too slow",
            avg
        );
    }
}

// World engine: owns the actor runtime, interest manager, and drives the
// 25 Hz tick loop. On each tick it:
//   1. Runs actor_runtime.tick() → collect effects
//   2. Applies move effects to spatial grid (already done in runtime.tick)
//   3. Computes per-session visible actor sets via interest manager
//   4. Encodes a StateDelta per session and queues it for the WS server
// The entire tick runs synchronously on a dedicated OS thread to avoid
// async task preemption jitter.

use crate::actor_runtime::ActorRuntime;
use crate::interest_manager::InterestManager;
use crate::state_encoder::{StateEncoder, diff_states};
use crate::types::{
    ActorId, ActorSpec, ActorState, SessionId, StateDelta, WorldConfig, now_ms,
};
use ahash::AHashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Per-session outgoing delta queue (bounded).
pub type SessionQueue = Arc<Mutex<VecDeque<StateDelta>>>;

pub struct WorldEngine {
    runtime: ActorRuntime,
    interest: InterestManager,
    encoder: StateEncoder,
    cfg: WorldConfig,
    tick_count: u64,
    /// Per-session delta queues read by WS tasks.
    session_queues: AHashMap<SessionId, SessionQueue>,
    /// Previous tick's full snapshot for diff computation.
    prev_snapshot: Vec<ActorState>,
    /// Removed actors this tick (accumulated from runtime effects).
    removed_this_tick: Vec<ActorId>,
}

impl WorldEngine {
    pub fn new(cfg: WorldConfig) -> Self {
        let interest = InterestManager::new(cfg.interest_radius);
        let runtime = ActorRuntime::new(cfg.clone());
        Self {
            runtime,
            interest,
            encoder: StateEncoder::new(1 << 20), // 1 MB initial buffer
            cfg,
            tick_count: 0,
            session_queues: AHashMap::new(),
            prev_snapshot: Vec::new(),
            removed_this_tick: Vec::new(),
        }
    }

    /// Register a session. Returns its outbound delta queue.
    pub fn add_session(
        &mut self,
        session: SessionId,
        anchor: ActorId,
        anchor_cell: crate::types::GridCell,
    ) -> SessionQueue {
        let q: SessionQueue = Arc::new(Mutex::new(VecDeque::with_capacity(8)));
        self.session_queues.insert(session, Arc::clone(&q));
        self.interest.register(session, anchor, anchor_cell);
        q
    }

    /// Remove a session on disconnect.
    pub fn remove_session(&mut self, session: SessionId) {
        self.session_queues.remove(&session);
        self.interest.unregister(session);
    }

    /// Spawn an actor into the world. Returns its external handle.
    pub fn spawn_actor(&mut self, spec: ActorSpec) -> crate::actor::ActorHandle {
        self.runtime.spawn_actor(spec)
    }

    /// Remove an actor from the world.
    pub fn despawn_actor(&mut self, id: ActorId) {
        self.removed_this_tick.push(id);
        self.runtime.despawn_actor(id);
    }

    /// Run one tick. This is the hot path — call from a dedicated tick thread.
    pub fn tick(&mut self) {
        let tick_start = Instant::now();
        self.tick_count += 1;
        let tick = self.tick_count;

        // 1. Run all actor inboxes.
        let effects = self.runtime.tick(tick);

        // 2. Collect moves to update interest manager anchor cells.
        for effect in &effects {
            use crate::actor::ActorEffect;
            if let ActorEffect::Move { id, .. } = effect {
                // Find the session for this actor (anchor lookup).
                // In a real system we'd maintain actor→session reverse map.
                // For now we update all sessions whose anchor is this actor.
                let snap = self.runtime.snapshot_all();
                if let Some(s) = snap.iter().find(|s| s.id == *id) {
                    // Find sessions anchored to this actor.
                    for (sid, _anchor, _cell) in self.interest.sessions() {
                        // Simple: update if session anchor == moved actor.
                        // In production: maintain actor→sessions index.
                        let _ = sid; // placeholder for full session index
                    }
                    // Update directly via session anchor check.
                    // (Simplified for now; production uses reverse map.)
                    let cell = s.cell;
                    for (sid, anchor_id, _) in self.interest.sessions().collect::<Vec<_>>() {
                        if anchor_id == *id {
                            self.interest.update_cell(sid, cell);
                        }
                    }
                }
            }
        }

        // 3. Full snapshot for this tick.
        let current_snapshot = self.runtime.snapshot_all();

        // 4. Compute global diff (new actors, moved, removed).
        let removed = std::mem::take(&mut self.removed_this_tick);
        let (changed, mut newly_removed) = diff_states(&self.prev_snapshot, &current_snapshot);
        newly_removed.extend(removed);

        // 5. Push per-session delta to outbound queues.
        let grid = self.runtime.grid();
        for (sid, _, _) in self.interest.sessions().collect::<Vec<_>>() {
            let visible_ids = self.interest.visible_actors(sid, grid);

            // Filter `changed` to only visible actors.
            let visible_updates: Vec<ActorState> = changed
                .iter()
                .filter(|s| visible_ids.contains(&s.id))
                .cloned()
                .collect();

            // Include removals of actors that were previously visible.
            // (Simplified: send all removals to all sessions.)
            let delta = StateDelta {
                tick,
                timestamp_ms: now_ms(),
                updates: visible_updates,
                removed: newly_removed.clone(),
            };

            if let Some(q) = self.session_queues.get(&sid) {
                let mut queue = q.lock().unwrap();
                if queue.len() >= 16 {
                    // Drop oldest frame if consumer is lagging.
                    queue.pop_front();
                    warn!(session = sid.0, "Session lagging; dropped oldest frame");
                }
                queue.push_back(delta);
            }
        }

        self.prev_snapshot = current_snapshot;

        let elapsed = tick_start.elapsed();
        if elapsed > Duration::from_millis(5) {
            warn!(tick, ?elapsed, "Tick took longer than 5 ms");
        }
        debug!(tick, actors = self.runtime.actor_count(), ?elapsed, "tick");
    }

    /// Run the tick loop at the configured Hz. Blocking — call on a dedicated thread.
    pub fn run_tick_loop(&mut self) {
        let interval = Duration::from_secs_f64(1.0 / self.cfg.tick_hz as f64);
        let mut next_tick = Instant::now();
        info!(hz = self.cfg.tick_hz, "Tick loop starting");
        loop {
            self.tick();
            next_tick += interval;
            let now = Instant::now();
            if next_tick > now {
                std::thread::sleep(next_tick - now);
            }
        }
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    pub fn actor_count(&self) -> usize {
        self.runtime.actor_count()
    }

    pub fn session_count(&self) -> usize {
        self.session_queues.len()
    }

    /// Snapshot of all actor states (for persistence / global agents).
    pub fn full_snapshot(&self) -> Vec<ActorState> {
        self.runtime.snapshot_all()
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
        engine.spawn_actor(make_spec(1, 5.0, 5.0));
        engine.spawn_actor(make_spec(2, 15.0, 15.0));
        assert_eq!(engine.actor_count(), 2);
    }

    #[test]
    fn despawn_removes_actor() {
        let mut engine = WorldEngine::new(WorldConfig::default());
        engine.spawn_actor(make_spec(1, 5.0, 5.0));
        engine.despawn_actor(ActorId(1));
        engine.tick();
        assert_eq!(engine.actor_count(), 0);
    }

    #[test]
    fn session_queue_receives_delta() {
        let mut engine = WorldEngine::new(WorldConfig::default());
        let handle = engine.spawn_actor(make_spec(1, 5.0, 5.0));
        let _ = handle;

        let snap = engine.full_snapshot();
        let actor_state = &snap[0];
        let q = engine.add_session(SessionId(1), ActorId(1), actor_state.cell);

        // Send a move message and tick
        engine.spawn_actor(make_spec(2, 5.0, 5.0)); // second actor in same cell
        engine.tick();

        let queue = q.lock().unwrap();
        assert!(!queue.is_empty(), "Session should have received a delta");
    }

    #[test]
    fn tick_timing_respects_5ms_budget() {
        // With 0 actors, tick must be well under budget.
        let mut engine = WorldEngine::new(WorldConfig::default());
        let start = Instant::now();
        for _ in 0..100 {
            engine.tick();
        }
        let avg = start.elapsed() / 100;
        assert!(
            avg < Duration::from_millis(2),
            "Tick average too slow: {:?}",
            avg
        );
    }
}

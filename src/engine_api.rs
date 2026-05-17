/// `EngineApi` — uniform interface for both single-node `WorldEngine` and
/// multi-shard `ShardedEngine`.  The WebSocket server, metrics endpoint, and
/// global agents all talk through this trait; only `main.rs` decides which
/// concrete implementation to wire in.
use crate::actor::ActorHandle;
use crate::types::{ActorSpec, ActorState, SessionId, WorldDirective};
use crate::world_engine::{SessionQueue, WorldEngine};

pub trait EngineApi: Send {
    fn spawn_actor_for_session(
        &mut self,
        spec: ActorSpec,
        session: SessionId,
    ) -> (ActorHandle, SessionQueue);

    /// Spawn an actor without a client session (used for cold-start recovery).
    fn spawn_actor_standalone(&mut self, spec: ActorSpec) -> ActorHandle;

    fn remove_session(&mut self, session: SessionId, despawn_actor: bool);

    fn apply_directive(&self, directive: &WorldDirective);

    fn tick(&mut self);
    fn tick_count(&self) -> u64;
    fn actor_count(&self) -> usize;
    fn session_count(&self) -> usize;
    fn uptime_secs(&self) -> u64;
    fn full_snapshot(&self) -> Vec<ActorState>;

    /// Full snapshot including specs, for disk persistence.
    fn world_snapshot_for_persist(&self) -> crate::persistence::WorldSnapshot;
}

// ── WorldEngine → EngineApi ───────────────────────────────────────────────────

impl EngineApi for WorldEngine {
    fn spawn_actor_for_session(
        &mut self,
        spec: ActorSpec,
        session: SessionId,
    ) -> (ActorHandle, SessionQueue) {
        WorldEngine::spawn_actor_for_session(self, spec, session)
    }

    fn remove_session(&mut self, session: SessionId, despawn_actor: bool) {
        WorldEngine::remove_session(self, session, despawn_actor)
    }

    fn apply_directive(&self, directive: &WorldDirective) {
        WorldEngine::apply_directive(self, directive)
    }

    fn tick(&mut self) {
        WorldEngine::tick(self)
    }

    fn tick_count(&self) -> u64 {
        WorldEngine::tick_count(self)
    }

    fn actor_count(&self) -> usize {
        WorldEngine::actor_count(self)
    }

    fn session_count(&self) -> usize {
        WorldEngine::session_count(self)
    }

    fn uptime_secs(&self) -> u64 {
        WorldEngine::uptime_secs(self)
    }

    fn full_snapshot(&self) -> Vec<ActorState> {
        WorldEngine::full_snapshot(self)
    }

    fn spawn_actor_standalone(&mut self, spec: ActorSpec) -> ActorHandle {
        WorldEngine::spawn_actor_standalone(self, spec)
    }

    fn world_snapshot_for_persist(&self) -> crate::persistence::WorldSnapshot {
        WorldEngine::world_snapshot_for_persist(self)
    }
}

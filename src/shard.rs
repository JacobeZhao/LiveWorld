/// Multi-shard world engine.
///
/// Divides the x-axis into N equal-width bands; each band is an independent
/// `WorldEngine`.  All shards are ticked together by the main tick loop through
/// the `EngineApi` trait.
///
/// Cross-shard visibility (actors near a boundary see into the neighbouring
/// shard) is handled by the interest manager inside each engine — actors that
/// cross a shard boundary are re-assigned on the next `spawn_actor_for_session`
/// call.  For inter-shard gRPC messaging (multi-process deployment), add a
/// `ShardGateway` layer above this struct.
use crate::actor::ActorHandle;
use crate::engine_api::EngineApi;
use crate::types::{
    ActorId, ActorMessage, ActorSpec, ActorState, Position, SessionId, WorldConfig,
    WorldDirective,
};
use crate::world_engine::{SessionQueue, WorldEngine};
use ahash::AHashMap;
use std::time::Instant;
use tracing::{info, warn};

pub struct ShardedEngine {
    shards: Vec<WorldEngine>,
    shard_count: usize,
    shard_width: f32,
    /// Routes session → shard index so we can dispatch remove/handle lookups.
    session_to_shard: AHashMap<SessionId, usize>,
    global_tick: u64,
    start_time: Instant,
}

impl ShardedEngine {
    pub fn new(cfg: WorldConfig, shard_count: usize) -> Self {
        assert!(shard_count >= 1, "shard_count must be at least 1");
        let world_width = cfg.grid_width as f32 * cfg.cell_size;
        let shard_width = world_width / shard_count as f32;
        let shards = (0..shard_count)
            .map(|_| WorldEngine::new(cfg.clone()))
            .collect();
        info!(
            shard_count,
            shard_width,
            world_width,
            "ShardedEngine initialised"
        );
        Self {
            shards,
            shard_count,
            shard_width,
            session_to_shard: AHashMap::new(),
            global_tick: 0,
            start_time: Instant::now(),
        }
    }

    #[inline]
    fn shard_idx(&self, x: f32) -> usize {
        ((x / self.shard_width) as usize).min(self.shard_count - 1)
    }
}

impl EngineApi for ShardedEngine {
    fn spawn_actor_for_session(
        &mut self,
        spec: ActorSpec,
        session: SessionId,
    ) -> (ActorHandle, SessionQueue) {
        let idx = self.shard_idx(spec.position.x);
        let result = self.shards[idx].spawn_actor_for_session(spec, session);
        self.session_to_shard.insert(session, idx);
        result
    }

    fn remove_session(&mut self, session: SessionId, despawn_actor: bool) {
        if let Some(idx) = self.session_to_shard.remove(&session) {
            self.shards[idx].remove_session(session, despawn_actor);
        }
    }

    fn apply_directive(&self, directive: &WorldDirective) {
        match directive {
            WorldDirective::ForceMove { actor_id, to } => {
                // Try each shard until we find the actor.
                for shard in &self.shards {
                    if shard.send_to_actor(*actor_id, ActorMessage::Move { to: *to }) {
                        return;
                    }
                }
                warn!(actor = actor_id.0, "ForceMove: actor not found in any shard");
            }
            WorldDirective::FlagActor { actor_id, reason } => {
                warn!(actor = actor_id.0, reason, "Actor flagged");
            }
            WorldDirective::NarrativeEvent { message } => {
                info!(message, "NarrativeEvent broadcast");
            }
            WorldDirective::EconomyAdjust { resource, delta } => {
                info!(%resource, delta, "EconomyAdjust applied");
            }
        }
    }

    fn tick(&mut self) {
        self.global_tick += 1;
        for shard in &mut self.shards {
            shard.tick();
        }
    }

    fn tick_count(&self) -> u64 {
        self.global_tick
    }

    fn actor_count(&self) -> usize {
        self.shards.iter().map(|s| s.actor_count()).sum()
    }

    fn session_count(&self) -> usize {
        self.shards.iter().map(|s| s.session_count()).sum()
    }

    fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    fn full_snapshot(&self) -> Vec<ActorState> {
        self.shards.iter().flat_map(|s| s.full_snapshot()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LlmModel, WorldConfig};

    fn make_spec(id: u64, x: f32) -> ActorSpec {
        ActorSpec {
            id: ActorId(id),
            name: format!("A{id}"),
            personality: String::new(),
            backstory: String::new(),
            model: LlmModel::Mock,
            position: Position::new(x, 500.0),
        }
    }

    #[test]
    fn shard_routing_by_x() {
        let cfg = WorldConfig::default(); // 10000x10000 world
        let mut engine = ShardedEngine::new(cfg, 4);
        // shard_width = 10000/4 = 2500; shard 0=[0,2500), shard 1=[2500,5000), etc.
        let sid0 = SessionId::next();
        let sid1 = SessionId::next();
        engine.spawn_actor_for_session(make_spec(1, 100.0), sid0);
        engine.spawn_actor_for_session(make_spec(2, 3000.0), sid1);
        assert_eq!(engine.session_to_shard[&sid0], 0); // x=100 → shard 0
        assert_eq!(engine.session_to_shard[&sid1], 1); // x=3000 → shard 1
    }

    #[test]
    fn actor_count_aggregates_shards() {
        let cfg = WorldConfig::default();
        let mut engine = ShardedEngine::new(cfg, 2);
        engine.spawn_actor_for_session(make_spec(1, 100.0), SessionId::next());
        engine.spawn_actor_for_session(make_spec(2, 6000.0), SessionId::next());
        assert_eq!(engine.actor_count(), 2);
    }

    #[test]
    fn remove_session_cleans_up() {
        let cfg = WorldConfig::default();
        let mut engine = ShardedEngine::new(cfg, 2);
        let sid = SessionId::next();
        engine.spawn_actor_for_session(make_spec(1, 100.0), sid);
        assert_eq!(engine.session_count(), 1);
        engine.remove_session(sid, true);
        engine.tick(); // propagate despawn
        assert_eq!(engine.session_count(), 0);
        assert_eq!(engine.actor_count(), 0);
    }

    #[test]
    fn tick_advances_global_counter() {
        let cfg = WorldConfig::default();
        let mut engine = ShardedEngine::new(cfg, 3);
        engine.tick();
        engine.tick();
        assert_eq!(engine.tick_count(), 2);
    }

    #[test]
    fn full_snapshot_covers_all_shards() {
        let cfg = WorldConfig::default();
        let mut engine = ShardedEngine::new(cfg, 4);
        engine.spawn_actor_for_session(make_spec(10, 500.0), SessionId::next());
        engine.spawn_actor_for_session(make_spec(20, 3000.0), SessionId::next());
        engine.spawn_actor_for_session(make_spec(30, 7000.0), SessionId::next());
        engine.spawn_actor_for_session(make_spec(40, 9000.0), SessionId::next());
        assert_eq!(engine.full_snapshot().len(), 4);
    }
}

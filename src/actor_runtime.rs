// Actor runtime: owns all live actors, routes messages, and runs the per-tick
// drain cycle. Single-threaded during tick processing; actor decisions are
// async and run on a separate Tokio task pool.

use crate::actor::{Actor, ActorEffect, ActorHandle};
use crate::spatial_grid::SpatialGrid;
use crate::types::{ActorId, ActorSpec, WorldConfig};
use ahash::AHashMap;
use std::time::Instant;

pub struct ActorRuntime {
    actors: AHashMap<ActorId, Actor>,
    handles: AHashMap<ActorId, ActorHandle>,
    grid: SpatialGrid,
    cfg: WorldConfig,
    stats: RuntimeStats,
}

#[derive(Default, Debug, Clone)]
pub struct RuntimeStats {
    pub total_actors: usize,
    pub total_effects_last_tick: usize,
    pub tick_count: u64,
}

impl ActorRuntime {
    pub fn new(cfg: WorldConfig) -> Self {
        let grid = SpatialGrid::new(&cfg);
        Self {
            actors: AHashMap::with_capacity(cfg.max_actors),
            handles: AHashMap::with_capacity(cfg.max_actors),
            grid,
            cfg,
            stats: RuntimeStats::default(),
        }
    }

    /// Register a new actor. Returns its external handle.
    pub fn spawn_actor(&mut self, spec: ActorSpec) -> ActorHandle {
        let id = spec.id;
        let (mut actor, handle) = Actor::spawn(spec);

        // Register in spatial grid.
        let cell = self.grid.insert(id, actor.state.position);
        actor.activate(cell);

        // Store cloned handle for routing.
        let routing_handle = handle.clone();
        self.actors.insert(id, actor);
        self.handles.insert(id, routing_handle);
        self.stats.total_actors = self.actors.len();
        handle
    }

    /// Remove an actor from the runtime and the spatial grid.
    pub fn despawn_actor(&mut self, id: ActorId) -> bool {
        if let Some(actor) = self.actors.remove(&id) {
            self.grid.remove(id, actor.state.cell);
            self.handles.remove(&id);
            self.stats.total_actors = self.actors.len();
            true
        } else {
            false
        }
    }

    /// Get an external handle to send messages to an actor.
    #[inline]
    pub fn handle(&self, id: ActorId) -> Option<&ActorHandle> {
        self.handles.get(&id)
    }

    /// Run one tick: drain all actor inboxes and apply effects.
    /// Returns list of effects for the world engine to process (broadcast, etc.).
    pub fn tick(&mut self, tick_num: u64) -> Vec<ActorEffect> {
        let _t = Instant::now();
        let mut all_effects = Vec::new();
        let mut to_despawn = Vec::new();

        for actor in self.actors.values_mut() {
            actor.tick(tick_num);
            let effects = actor.drain_inbox();
            for effect in &effects {
                if let ActorEffect::Move { id, to } = effect {
                    let new_cell = self.grid.move_actor(*id, actor.state.cell, *to);
                    actor.apply_move(*to, new_cell);
                }
            }
            all_effects.extend(effects);

            if !actor.is_alive() {
                to_despawn.push(actor.id());
            }
        }

        for id in to_despawn {
            self.despawn_actor(id);
        }

        self.stats.total_effects_last_tick = all_effects.len();
        self.stats.tick_count = tick_num;
        all_effects
    }

    /// Snapshot all actor states for broadcasting.
    pub fn snapshot_all(&self) -> Vec<crate::types::ActorState> {
        self.actors.values().map(|a| a.snapshot()).collect()
    }

    /// Snapshot actor states for a given set of actor IDs.
    pub fn snapshot_subset(&self, ids: &[ActorId]) -> Vec<crate::types::ActorState> {
        ids.iter()
            .filter_map(|id| self.actors.get(id).map(|a| a.snapshot()))
            .collect()
    }

    pub fn actor_count(&self) -> usize {
        self.actors.len()
    }

    pub fn grid(&self) -> &SpatialGrid {
        &self.grid
    }

    pub fn stats(&self) -> &RuntimeStats {
        &self.stats
    }

    /// Snapshot specs of all live actors (for persistence).
    pub fn specs_snapshot(&self) -> Vec<crate::types::ActorSpec> {
        self.actors.values().map(|a| a.spec.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActorId, ActorMessage, LlmModel, Position};

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
    fn spawn_and_count() {
        let mut rt = ActorRuntime::new(WorldConfig::default());
        for i in 0..10 {
            rt.spawn_actor(make_spec(i, i as f32 * 5.0, 0.0));
        }
        assert_eq!(rt.actor_count(), 10);
    }

    #[test]
    fn despawn_removes_actor() {
        let mut rt = ActorRuntime::new(WorldConfig::default());
        rt.spawn_actor(make_spec(1, 0.0, 0.0));
        assert!(rt.despawn_actor(ActorId(1)));
        assert_eq!(rt.actor_count(), 0);
        assert!(!rt.despawn_actor(ActorId(1))); // idempotent
    }

    #[test]
    fn tick_applies_move_effect() {
        let mut rt = ActorRuntime::new(WorldConfig::default());
        let handle = rt.spawn_actor(make_spec(42, 0.0, 0.0));
        handle.send(ActorMessage::Move {
            to: Position::new(50.0, 50.0),
        });
        let effects = rt.tick(1);
        assert!(!effects.is_empty(), "Expected at least one Move effect");
        // After tick, actor state should reflect new position
        let snap = rt.snapshot_all();
        let actor = snap.iter().find(|a| a.id == ActorId(42)).unwrap();
        assert_eq!(actor.position, Position::new(50.0, 50.0));
    }

    #[test]
    fn shutdown_despawns_actor_next_tick() {
        let mut rt = ActorRuntime::new(WorldConfig::default());
        let handle = rt.spawn_actor(make_spec(99, 0.0, 0.0));
        handle.send(ActorMessage::Shutdown);
        rt.tick(1); // processes shutdown
        rt.tick(2); // despawn happens during tick that finds !is_alive
        assert_eq!(rt.actor_count(), 0);
    }

    #[test]
    fn grid_consistency_after_moves() {
        let mut rt = ActorRuntime::new(WorldConfig::default());
        let h1 = rt.spawn_actor(make_spec(1, 5.0, 5.0));
        let h2 = rt.spawn_actor(make_spec(2, 5.0, 5.0));
        h1.send(ActorMessage::Move {
            to: Position::new(500.0, 500.0),
        });
        h2.send(ActorMessage::Move {
            to: Position::new(500.0, 500.0),
        });
        rt.tick(1);

        // Both actors should now be in the same far cell
        let snap = rt.snapshot_all();
        for s in &snap {
            assert_eq!(s.cell, crate::types::GridCell(50, 50));
        }
        // Grid invariant: each actor in its claimed cell
        for s in &snap {
            assert!(
                rt.grid().contains(s.id, s.cell),
                "Grid invariant violated for {:?}",
                s.id
            );
        }
    }
}

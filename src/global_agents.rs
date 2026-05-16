// Global agent framework: Director, Economy, AntiCheat.
// Each runs as an independent Tokio task, reads world snapshots asynchronously,
// generates directives, and injects events into the world via a command channel.
// They never run on the tick thread.

use crate::llm_adapter::{LlmAdapter, LlmRequest, LlmResponse};
use crate::semantic_cache::SemanticCache;
use crate::types::{ActorId, ActorState, LlmModel};
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio::time;
use tracing::{info, warn};

// ── World directive (global agent → world engine) ─────────────────────────────

#[derive(Debug, Clone)]
pub enum WorldDirective {
    /// Forcibly move an actor (e.g. anti-cheat teleport correction).
    ForceMove { actor_id: ActorId, to: crate::types::Position },
    /// Broadcast a narrative event to all players.
    NarrativeEvent { message: String },
    /// Adjust economy: inject resources into the world.
    EconomyAdjust { resource: String, delta: i64 },
    /// Flag actor for review (anti-cheat).
    FlagActor { actor_id: ActorId, reason: String },
}

/// World snapshot shared between the tick thread and global agents.
pub type SharedSnapshot = Arc<Mutex<Vec<ActorState>>>;

// ── Global Agent trait ────────────────────────────────────────────────────────

pub trait GlobalAgent: Send + Sync {
    fn name(&self) -> &'static str;
}

// ── Director Agent ────────────────────────────────────────────────────────────

/// Drives narrative events and world storylines.
pub struct DirectorAgent {
    llm: Arc<AsyncMutex<SemanticCache>>,
    snapshot: SharedSnapshot,
    directive_tx: mpsc::Sender<WorldDirective>,
    interval: Duration,
    decision_count: Arc<std::sync::atomic::AtomicU64>,
}

impl DirectorAgent {
    pub fn new(
        llm: Arc<AsyncMutex<SemanticCache>>,
        snapshot: SharedSnapshot,
        directive_tx: mpsc::Sender<WorldDirective>,
        interval: Duration,
    ) -> Self {
        Self {
            llm,
            snapshot,
            directive_tx,
            interval,
            decision_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    pub fn decision_count(&self) -> u64 {
        self.decision_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub async fn run(self) {
        let mut ticker = time::interval(self.interval);
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        info!("DirectorAgent started");

        loop {
            ticker.tick().await;

            let actor_count = {
                let snap = self.snapshot.lock().unwrap();
                snap.len()
            };

            let req = LlmRequest {
                model: LlmModel::Mock,
                system_prompt: "You are the world director. Generate a brief narrative event \
                    that makes the world more interesting. Output: EVENT <description>"
                    .to_string(),
                user_prompt: format!(
                    "There are {actor_count} active characters in the world. \
                     Create a world event."
                ),
                max_tokens: 80,
            };

            let resp = {
                let mut cache = self.llm.lock().await;
                cache.complete(req).await
            };

            match resp {
                Ok(r) => {
                    self.decision_count
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if let Some(event) = r.text.strip_prefix("EVENT ") {
                        info!(event = event.trim(), "Director narrative event");
                        let _ = self
                            .directive_tx
                            .send(WorldDirective::NarrativeEvent {
                                message: event.trim().to_string(),
                            })
                            .await;
                    }
                }
                Err(e) => warn!(err = %e, "DirectorAgent LLM error"),
            }
        }
    }
}

// ── Economy Agent ─────────────────────────────────────────────────────────────

/// Monitors world economy and injects resource adjustments.
pub struct EconomyAgent {
    snapshot: SharedSnapshot,
    directive_tx: mpsc::Sender<WorldDirective>,
    interval: Duration,
    decision_count: Arc<std::sync::atomic::AtomicU64>,
}

impl EconomyAgent {
    pub fn new(
        snapshot: SharedSnapshot,
        directive_tx: mpsc::Sender<WorldDirective>,
        interval: Duration,
    ) -> Self {
        Self {
            snapshot,
            directive_tx,
            interval,
            decision_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    pub fn decision_count(&self) -> u64 {
        self.decision_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub async fn run(self) {
        let mut ticker = time::interval(self.interval);
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        info!("EconomyAgent started");

        loop {
            ticker.tick().await;

            let actor_count = {
                let snap = self.snapshot.lock().unwrap();
                snap.len()
            };

            self.decision_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Simple heuristic: if many actors, inject more resources.
            let delta = if actor_count > 1000 { 100i64 } else { 10 };
            let _ = self
                .directive_tx
                .send(WorldDirective::EconomyAdjust {
                    resource: "gold".to_string(),
                    delta,
                })
                .await;

            info!(actors = actor_count, delta, "EconomyAgent adjustment");
        }
    }
}

// ── AntiCheat Agent ───────────────────────────────────────────────────────────

/// Detects impossible movements (teleporting faster than max speed).
pub struct AntiCheatAgent {
    snapshot: SharedSnapshot,
    directive_tx: mpsc::Sender<WorldDirective>,
    interval: Duration,
    max_speed_per_tick: f32,
    decision_count: Arc<std::sync::atomic::AtomicU64>,
    last_positions: Arc<Mutex<ahash::AHashMap<ActorId, crate::types::Position>>>,
}

impl AntiCheatAgent {
    pub fn new(
        snapshot: SharedSnapshot,
        directive_tx: mpsc::Sender<WorldDirective>,
        interval: Duration,
        max_speed_per_tick: f32,
    ) -> Self {
        Self {
            snapshot,
            directive_tx,
            interval,
            max_speed_per_tick,
            decision_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            last_positions: Arc::new(Mutex::new(ahash::AHashMap::new())),
        }
    }

    pub fn decision_count(&self) -> u64 {
        self.decision_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub async fn run(self) {
        let mut ticker = time::interval(self.interval);
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
        info!("AntiCheatAgent started");

        loop {
            ticker.tick().await;
            self.decision_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Collect state and violations with the lock held (no await inside).
            let violations: Vec<(ActorId, f32)> = {
                let current: Vec<ActorState> = self.snapshot.lock().unwrap().clone();
                let mut last = self.last_positions.lock().unwrap();
                let mut v = Vec::new();
                for state in &current {
                    if let Some(&prev_pos) = last.get(&state.id) {
                        let dist = ((state.position.x - prev_pos.x).powi(2)
                            + (state.position.y - prev_pos.y).powi(2))
                        .sqrt();
                        if dist > self.max_speed_per_tick {
                            warn!(actor = state.id.0, dist, max = self.max_speed_per_tick,
                                "Speed violation detected");
                            v.push((state.id, dist));
                        }
                    }
                    last.insert(state.id, state.position);
                }
                v
            }; // MutexGuard dropped here, before any await

            // Send directives outside the lock.
            for (actor_id, dist) in violations {
                let _ = self
                    .directive_tx
                    .send(WorldDirective::FlagActor {
                        actor_id,
                        reason: format!("Speed violation: moved {dist:.1} units in one interval"),
                    })
                    .await;
            }
        }
    }
}

// ── Directive processor ───────────────────────────────────────────────────────

/// Receives WorldDirectives and applies them to the world engine.
/// Runs as a Tokio task, reading from the channel.
pub async fn process_directives(mut rx: mpsc::Receiver<WorldDirective>) {
    while let Some(directive) = rx.recv().await {
        match &directive {
            WorldDirective::NarrativeEvent { message } => {
                info!(message, "WorldDirective: NarrativeEvent");
            }
            WorldDirective::EconomyAdjust { resource, delta } => {
                info!(%resource, delta, "WorldDirective: EconomyAdjust");
            }
            WorldDirective::FlagActor { actor_id, reason } => {
                warn!(actor = actor_id.0, reason, "WorldDirective: FlagActor");
            }
            WorldDirective::ForceMove { actor_id, to } => {
                info!(actor = actor_id.0, ?to, "WorldDirective: ForceMove");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_adapter::MockLlm;
    use crate::types::{ActorId, GridCell, Position};
    use std::time::Duration;

    fn make_snapshot_shared() -> SharedSnapshot {
        Arc::new(Mutex::new(vec![]))
    }

    fn make_cache() -> Arc<AsyncMutex<SemanticCache>> {
        let llm = Arc::new(MockLlm::new().with_response("EVENT A great storm is coming"));
        Arc::new(AsyncMutex::new(SemanticCache::new(10, llm)))
    }

    #[tokio::test]
    async fn economy_agent_increments_decisions() {
        let snap = make_snapshot_shared();
        let (tx, mut rx) = mpsc::channel(32);
        let agent = EconomyAgent::new(Arc::clone(&snap), tx, Duration::from_millis(10));
        let count = Arc::clone(&agent.decision_count);

        let task = tokio::spawn(agent.run());

        // Let it run for ~50ms → at least 3 decisions
        tokio::time::sleep(Duration::from_millis(55)).await;
        task.abort();

        assert!(
            count.load(std::sync::atomic::Ordering::Relaxed) >= 3,
            "Expected ≥3 decisions, got {}",
            count.load(std::sync::atomic::Ordering::Relaxed)
        );

        // Drain directives
        while rx.try_recv().is_ok() {}
    }

    #[tokio::test]
    async fn anticheat_flags_teleporter() {
        let snap = make_snapshot_shared();
        let (tx, mut rx) = mpsc::channel(32);

        // Pre-populate with an actor
        {
            let mut s = snap.lock().unwrap();
            s.push(ActorState {
                id: ActorId(1),
                name: "Cheater".to_string(),
                position: Position::new(0.0, 0.0),
                cell: GridCell(0, 0),
                tick: 0,
                last_utterance: None,
            });
        }

        let agent =
            AntiCheatAgent::new(Arc::clone(&snap), tx, Duration::from_millis(10), 50.0);

        // Run one tick cycle to record positions
        let task = tokio::spawn(agent.run());
        tokio::time::sleep(Duration::from_millis(15)).await;

        // Teleport the actor far away
        {
            let mut s = snap.lock().unwrap();
            s[0].position = Position::new(9999.0, 9999.0);
        }

        tokio::time::sleep(Duration::from_millis(20)).await;
        task.abort();

        // Should have received a FlagActor directive
        let directive = rx.try_recv().ok();
        assert!(
            matches!(directive, Some(WorldDirective::FlagActor { .. })),
            "Expected FlagActor directive, got {:?}",
            directive
        );
    }

    #[tokio::test]
    async fn directive_processor_runs_without_panic() {
        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(process_directives(rx));

        tx.send(WorldDirective::NarrativeEvent { message: "Test".to_string() })
            .await
            .unwrap();
        tx.send(WorldDirective::EconomyAdjust {
            resource: "wood".to_string(),
            delta: 50,
        })
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(10)).await;
        // No panic = pass
    }
}

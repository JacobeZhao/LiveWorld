use liveworld::agent_decision::{AgentDecisionLoop, DecisionConfig};
use liveworld::circuit_breaker::CircuitBreaker;
use liveworld::engine_api::EngineApi;
use liveworld::global_agents::{
    process_directives, AntiCheatAgent, DirectorAgent, EconomyAgent, SharedSnapshot, WorldDirective,
};
use liveworld::llm_adapter::{create_adapter, MockLlm};
use liveworld::metrics::run_http_server;
use liveworld::persistence::{self, SnapshotStore};
use liveworld::redis_sync::run_redis_sync;
use liveworld::semantic_cache::SemanticCache;
use liveworld::shard::ShardedEngine;
use liveworld::types::WorldConfig;
use liveworld::world_engine::WorldEngine;
use liveworld::ws_server::SharedEngine;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── Logging ────────────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("LiveWorld engine starting…");

    // ── Configuration ──────────────────────────────────────────────────────────
    let cfg = WorldConfig::default();

    // ── Engine: single-node or sharded (set SHARD_COUNT=N to enable) ───────────
    let shard_count: usize = std::env::var("SHARD_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let shared_engine: SharedEngine = if shard_count > 1 {
        info!(shard_count, "Starting in sharded mode");
        Arc::new(Mutex::new(
            Box::new(ShardedEngine::new(cfg.clone(), shard_count)) as Box<dyn EngineApi + Send>,
        ))
    } else {
        info!("Starting in single-node mode");
        Arc::new(Mutex::new(
            Box::new(WorldEngine::new(cfg.clone())) as Box<dyn EngineApi + Send>
        ))
    };

    // ── JWT secret warning ─────────────────────────────────────────────────────
    if std::env::var("JWT_SECRET").is_err() {
        tracing::warn!("JWT_SECRET not set — auth disabled (dev mode)");
    }

    // ── Snapshot store ─────────────────────────────────────────────────────────
    let mut store = SnapshotStore::new("data/snapshots", 5)?;

    // ── Shared world snapshot (for global agents) ──────────────────────────────
    let world_snapshot: SharedSnapshot = Arc::new(Mutex::new(ahash::AHashMap::new()));

    // ── Cold-start recovery: restore actors + decision loops from disk ─────────
    {
        match store.read_latest() {
            Ok(Some(snap)) => {
                let specs = persistence::restore_actors(&snap);
                info!(
                    count = specs.len(),
                    "Cold start: restoring world from snapshot"
                );
                let cb = Arc::new(CircuitBreaker::new(5, Duration::from_secs(30)));
                for spec in specs {
                    let handle = {
                        let mut eng = shared_engine.lock().unwrap();
                        eng.spawn_actor_standalone(spec.clone())
                    };
                    let adapter = create_adapter(&spec.model);
                    let actor_cache = Arc::new(AsyncMutex::new(SemanticCache::new(256, adapter)));
                    let dl = AgentDecisionLoop::new(
                        spec,
                        handle,
                        actor_cache,
                        DecisionConfig::default(),
                        Arc::clone(&cb),
                    );
                    tokio::spawn(dl.run(Arc::clone(&world_snapshot)));
                }
            }
            Ok(None) => info!("No snapshot found — starting fresh"),
            Err(e) => tracing::warn!(err = %e, "Failed to read snapshot; starting fresh"),
        }
    }

    // ── LLM cache for global agents ────────────────────────────────────────────
    let mock_llm = Arc::new(MockLlm::new());
    let cache = Arc::new(AsyncMutex::new(SemanticCache::new(256, mock_llm)));

    // ── Directive channel → engine ─────────────────────────────────────────────
    let (dir_tx, dir_rx) = mpsc::channel::<WorldDirective>(1024);
    {
        let engine = Arc::clone(&shared_engine);
        tokio::spawn(process_directives(dir_rx, engine));
    }

    // ── Global agents ──────────────────────────────────────────────────────────
    {
        let snap = Arc::clone(&world_snapshot);
        let tx = dir_tx.clone();
        let c = Arc::clone(&cache);
        tokio::spawn(async move {
            DirectorAgent::new(c, snap, tx, Duration::from_secs(10))
                .run()
                .await;
        });
    }
    {
        let snap = Arc::clone(&world_snapshot);
        let tx = dir_tx.clone();
        tokio::spawn(async move {
            EconomyAgent::new(snap, tx, Duration::from_secs(5))
                .run()
                .await;
        });
    }
    {
        let snap = Arc::clone(&world_snapshot);
        let tx = dir_tx;
        tokio::spawn(async move {
            AntiCheatAgent::new(snap, tx, Duration::from_millis(500), 200.0)
                .run()
                .await;
        });
    }

    // ── Periodic snapshot: persist world state to disk ────────────────────────
    // `store` is moved here; it is only written from this one task.
    // The tick loop owns SharedSnapshot updates; this task only writes to disk.
    {
        let engine = Arc::clone(&shared_engine);
        let interval_secs = cfg.snapshot_interval_secs;
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                let (tick, world_snap) = {
                    let eng = engine.lock().unwrap();
                    (eng.tick_count(), eng.world_snapshot_for_persist())
                };
                if let Err(e) = store.write(&world_snap) {
                    tracing::warn!(tick, err = %e, "Failed to write snapshot");
                } else {
                    info!(tick, actors = world_snap.actors.len(), "Snapshot persisted");
                }
            }
        });
    }

    // ── Tick loop (dedicated OS thread) ────────────────────────────────────────
    // Merges local actor states into SharedSnapshot while preserving remote actors
    // that were injected by Redis cross-pod sync.
    {
        let engine = Arc::clone(&shared_engine);
        let snap = Arc::clone(&world_snapshot);
        let tick_hz = cfg.tick_hz;
        std::thread::spawn(move || {
            let tick_interval = Duration::from_secs_f64(1.0 / tick_hz as f64);
            let mut next_tick = std::time::Instant::now();
            // Track which IDs were local last cycle so we can evict departed actors.
            let mut prev_local_ids: ahash::AHashSet<liveworld::types::ActorId> =
                ahash::AHashSet::new();
            info!(hz = tick_hz, "Tick loop started on dedicated thread");
            loop {
                let states_opt = {
                    let mut eng = engine.lock().unwrap();
                    eng.tick();
                    let count = eng.tick_count();
                    if count.is_multiple_of(25) {
                        Some(eng.full_snapshot())
                    } else {
                        None
                    }
                };
                if let Some(states) = states_opt {
                    let local_ids: ahash::AHashSet<liveworld::types::ActorId> =
                        states.iter().map(|s| s.id).collect();
                    if let Ok(mut shared) = snap.lock() {
                        // Remove actors that were local last tick but are gone now.
                        // Remote actors (not in prev_local_ids) are preserved.
                        shared
                            .retain(|id, _| !prev_local_ids.contains(id) || local_ids.contains(id));
                        for state in states {
                            shared.insert(state.id, state);
                        }
                    }
                    prev_local_ids = local_ids;
                }
                next_tick += tick_interval;
                let now = std::time::Instant::now();
                if next_tick > now {
                    std::thread::sleep(next_tick - now);
                }
            }
        });
    }

    // ── HTTP server: metrics + frontend + /auth/token + /health (port 8081) ────
    {
        let engine = Arc::clone(&shared_engine);
        tokio::spawn(async move {
            if let Err(e) = run_http_server(engine, 8081).await {
                tracing::error!("HTTP server error: {e}");
            }
        });
    }

    // ── Redis cross-pod sync (no-op when REDIS_URL unset) ─────────────────────
    {
        let snap = Arc::clone(&world_snapshot);
        tokio::spawn(async move {
            if let Err(e) = run_redis_sync(snap).await {
                tracing::warn!(err = %e, "Redis sync exited");
            }
        });
    }

    // ── Graceful shutdown ──────────────────────────────────────────────────────
    let shutdown = tokio::spawn(async {
        tokio::signal::ctrl_c()
            .await
            .expect("Ctrl-C listener failed");
        info!("Shutdown signal received — exiting.");
    });

    // ── WebSocket server (blocks until shutdown) ───────────────────────────────
    info!("WebSocket server on port {}", cfg.ws_port);
    tokio::select! {
        res = liveworld::ws_server::run_ws_server(shared_engine, cfg, world_snapshot) => {
            if let Err(e) = res { tracing::error!("WS server error: {e}"); }
        }
        _ = shutdown => {}
    }

    Ok(())
}

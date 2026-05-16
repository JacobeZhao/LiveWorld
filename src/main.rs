use liveworld::global_agents::{
    AntiCheatAgent, DirectorAgent, EconomyAgent, SharedSnapshot, WorldDirective, process_directives,
};
use liveworld::llm_adapter::MockLlm;
use liveworld::persistence::SnapshotStore;
use liveworld::semantic_cache::SemanticCache;
use liveworld::types::WorldConfig;
use liveworld::world_engine::WorldEngine;
use liveworld::ws_server::SharedEngine;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ── Logging setup ──────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    info!("LiveWorld engine starting...");

    // ── Configuration ──────────────────────────────────────────────────────
    let cfg = WorldConfig::default();

    // ── World engine ───────────────────────────────────────────────────────
    let engine = WorldEngine::new(cfg.clone());
    let shared_engine: SharedEngine = Arc::new(Mutex::new(engine));

    // ── Snapshot store ─────────────────────────────────────────────────────
    let _store = SnapshotStore::new("data/snapshots", 5)?;

    // ── Shared world snapshot (for global agents) ──────────────────────────
    let world_snapshot: SharedSnapshot = Arc::new(Mutex::new(Vec::new()));

    // ── LLM cache for global agents ────────────────────────────────────────
    let mock_llm = Arc::new(MockLlm::new());
    let cache = Arc::new(AsyncMutex::new(SemanticCache::new(256, mock_llm)));

    // ── Directive channel ──────────────────────────────────────────────────
    let (dir_tx, dir_rx) = mpsc::channel::<WorldDirective>(1024);
    tokio::spawn(process_directives(dir_rx));

    // ── Global agents ──────────────────────────────────────────────────────
    {
        let snap = Arc::clone(&world_snapshot);
        let tx = dir_tx.clone();
        let c = Arc::clone(&cache);
        tokio::spawn(async move {
            DirectorAgent::new(c, snap, tx, Duration::from_secs(10)).run().await;
        });
    }
    {
        let snap = Arc::clone(&world_snapshot);
        let tx = dir_tx.clone();
        tokio::spawn(async move {
            EconomyAgent::new(snap, tx, Duration::from_secs(5)).run().await;
        });
    }
    {
        let snap = Arc::clone(&world_snapshot);
        let tx = dir_tx.clone();
        tokio::spawn(async move {
            AntiCheatAgent::new(snap, tx, Duration::from_millis(500), 200.0).run().await;
        });
    }

    // ── Periodic snapshot task ─────────────────────────────────────────────
    {
        let engine = Arc::clone(&shared_engine);
        let snap = Arc::clone(&world_snapshot);
        let interval_secs = cfg.snapshot_interval_secs;
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                let (tick, states) = {
                    let eng = engine.lock().unwrap();
                    (eng.tick_count(), eng.full_snapshot())
                };
                {
                    let mut shared = snap.lock().unwrap();
                    *shared = states;
                }
                info!(tick, "Periodic snapshot");
            }
        });
    }

    // ── Tick loop (dedicated OS thread, not on Tokio pool) ─────────────────
    {
        let engine = Arc::clone(&shared_engine);
        let snap = Arc::clone(&world_snapshot);
        std::thread::spawn(move || {
            let tick_interval = Duration::from_secs_f64(1.0 / cfg.tick_hz as f64);
            let mut next_tick = std::time::Instant::now();
            info!(hz = cfg.tick_hz, "Tick loop started on dedicated thread");
            loop {
                let states = {
                    let mut eng = engine.lock().unwrap();
                    eng.tick();
                    let count = eng.tick_count();
                    if count % 25 == 0 {
                        Some(eng.full_snapshot())
                    } else {
                        None
                    }
                };
                if let Some(s) = states {
                    if let Ok(mut shared) = snap.lock() {
                        *shared = s;
                    }
                }
                next_tick += tick_interval;
                let now = std::time::Instant::now();
                if next_tick > now {
                    std::thread::sleep(next_tick - now);
                }
            }
        });
    }

    // ── WebSocket server (runs on Tokio runtime) ───────────────────────────
    info!("Starting WebSocket server on port {}", cfg.ws_port);
    liveworld::ws_server::run_ws_server(shared_engine, cfg).await?;

    Ok(())
}

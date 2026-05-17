// WebSocket server: accepts connections, authenticates, routes client commands
// to the world engine, and streams state deltas back to each client at 25 Hz.

use crate::actor::ActorHandle;
use crate::agent_decision::{AgentDecisionLoop, DecisionConfig};
use crate::circuit_breaker::CircuitBreaker;
use crate::engine_api::EngineApi;
use crate::global_agents::SharedSnapshot;
use crate::llm_adapter::create_adapter;
use crate::metrics;
use crate::semantic_cache::SemanticCache;
use crate::types::{
    ActorId, ActorMessage, ActorSpec, ClientCommand, LlmModel, Position, ServerMessage, SessionId,
    WorldConfig,
};
use crate::world_engine::SessionQueue;
use ahash::AHashMap;
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{interval, Duration};
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

/// Thread-safe, type-erased handle to any engine implementation.
/// Use `Box<dyn EngineApi + Send>` so callers are decoupled from the
/// concrete type (WorldEngine or ShardedEngine).
pub type SharedEngine = Arc<Mutex<Box<dyn EngineApi + Send>>>;

// ── Per-IP connection limiter ─────────────────────────────────────────────────

struct ConnectionLimiter {
    counts: AHashMap<IpAddr, u32>,
    max_per_ip: u32,
}

impl ConnectionLimiter {
    fn new(max_per_ip: u32) -> Self {
        Self {
            counts: AHashMap::new(),
            max_per_ip,
        }
    }

    fn try_acquire(&mut self, ip: IpAddr) -> bool {
        let count = self.counts.entry(ip).or_insert(0);
        if *count >= self.max_per_ip {
            false
        } else {
            *count += 1;
            true
        }
    }

    fn release(&mut self, ip: IpAddr) {
        if let Some(count) = self.counts.get_mut(&ip) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.counts.remove(&ip);
            }
        }
    }
}

// ── Rate limiter (token bucket, per-connection) ───────────────────────────────

struct RateLimiter {
    count: u32,
    window_start: Instant,
    limit: u32,
}

impl RateLimiter {
    fn new(limit: u32) -> Self {
        Self {
            count: 0,
            window_start: Instant::now(),
            limit,
        }
    }

    fn allow(&mut self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.window_start) >= Duration::from_secs(1) {
            self.count = 0;
            self.window_start = now;
        }
        if self.count >= self.limit {
            false
        } else {
            self.count += 1;
            true
        }
    }
}

// ── Engine commands (WS handler → command processor task) ────────────────────

pub enum EngineCommand {
    CreateActor {
        session: SessionId,
        name: String,
        personality: String,
        backstory: String,
        model: LlmModel,
        position: Position,
        /// Oneshot reply carries (ActorId, queue) back to the connection handler.
        reply: oneshot::Sender<Result<(ActorId, SessionQueue), String>>,
    },
    MoveActor {
        session: SessionId,
        to: Position,
    },
    ChatActor {
        session: SessionId,
        text: String,
    },
    DestroyActor {
        session: SessionId,
        actor_id: ActorId,
    },
    SessionDisconnect {
        session: SessionId,
    },
}

// ── Public entry point ────────────────────────────────────────────────────────

pub async fn run_ws_server(
    engine: SharedEngine,
    cfg: WorldConfig,
    world_snapshot: SharedSnapshot,
) -> Result<()> {
    let addr = format!("0.0.0.0:{}", cfg.ws_port);
    let listener = TcpListener::bind(&addr).await?;
    info!("WebSocket server listening on {}", addr);

    let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>(4096);
    tokio::spawn(command_processor(
        cmd_rx,
        Arc::clone(&engine),
        world_snapshot,
    ));

    let limiter = Arc::new(Mutex::new(ConnectionLimiter::new(10))); // max 10 connections per IP

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                let ip = addr.ip();
                {
                    let mut lim = limiter.lock().unwrap();
                    if !lim.try_acquire(ip) {
                        warn!(%addr, "Per-IP connection limit reached; rejecting");
                        continue;
                    }
                }
                let session = SessionId::next();
                info!(session = session.0, %addr, "New connection");
                let lim = Arc::clone(&limiter);
                let tx = cmd_tx.clone();
                let eng = Arc::clone(&engine);
                tokio::spawn(async move {
                    handle_connection(stream, addr, session, tx, eng).await;
                    lim.lock().unwrap().release(ip);
                });
            }
            Err(e) => error!("Accept error: {}", e),
        }
    }
}

// ── Command processor ─────────────────────────────────────────────────────────

/// Runs as a single task; owns all session-local state so that
/// MoveActor / ChatActor bypass the engine lock entirely (SPSC push is lock-free).
async fn command_processor(
    mut cmd_rx: mpsc::Receiver<EngineCommand>,
    engine: SharedEngine,
    world_snapshot: SharedSnapshot,
) {
    let mut session_handles: AHashMap<SessionId, ActorHandle> = AHashMap::new();
    // Per-session AgentDecisionLoop join handles — aborted on disconnect/destroy.
    let mut decision_handles: AHashMap<SessionId, tokio::task::JoinHandle<()>> = AHashMap::new();
    // Shared LLM cache per model type (lazy-initialised).
    let mut llm_caches: AHashMap<String, Arc<tokio::sync::Mutex<SemanticCache>>> = AHashMap::new();
    // One circuit breaker shared across all agents on this pod.
    let circuit_breaker = Arc::new(CircuitBreaker::new(5, Duration::from_secs(30)));

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            EngineCommand::CreateActor {
                session,
                name,
                personality,
                backstory,
                model,
                position,
                reply,
            } => {
                if name.len() > 64 {
                    let _ = reply.send(Err("Name too long (max 64 chars)".into()));
                    continue;
                }
                if personality.len() > 256 {
                    let _ = reply.send(Err("Personality too long (max 256 chars)".into()));
                    continue;
                }
                if backstory.len() > 512 {
                    let _ = reply.send(Err("Backstory too long (max 512 chars)".into()));
                    continue;
                }
                let position = Position::new(
                    position.x.clamp(0.0, 9_999.0),
                    position.y.clamp(0.0, 9_999.0),
                );
                let id = ActorId::next();
                let model_key = model.to_string();
                let model_for_cache = model.clone();
                let spec = ActorSpec {
                    id,
                    name,
                    personality,
                    backstory,
                    model,
                    position,
                };
                let spec_for_dl = spec.clone();
                let (handle, queue) = {
                    let mut eng = engine.lock().unwrap();
                    eng.spawn_actor_for_session(spec, session)
                };
                let handle_for_dl = handle.clone();
                session_handles.insert(session, handle);
                info!(session = session.0, actor = id.0, "Actor spawned");

                // Get or create a per-model LLM cache, then spawn the decision loop.
                let llm_cache = llm_caches
                    .entry(model_key)
                    .or_insert_with(|| {
                        let adapter = create_adapter(&model_for_cache);
                        Arc::new(tokio::sync::Mutex::new(SemanticCache::new(256, adapter)))
                    })
                    .clone();
                let dl = AgentDecisionLoop::new(
                    spec_for_dl,
                    handle_for_dl,
                    llm_cache,
                    DecisionConfig::default(),
                    Arc::clone(&circuit_breaker),
                );
                let snap = Arc::clone(&world_snapshot);
                decision_handles.insert(session, tokio::spawn(dl.run(snap)));

                let _ = reply.send(Ok((id, queue)));
            }

            EngineCommand::MoveActor { session, to } => {
                let to = Position::new(to.x.clamp(0.0, 9_999.0), to.y.clamp(0.0, 9_999.0));
                if let Some(handle) = session_handles.get(&session) {
                    handle.send(ActorMessage::Move { to });
                }
            }

            EngineCommand::ChatActor { session, text } => {
                if text.len() > 500 {
                    continue;
                }
                if let Some(handle) = session_handles.get(&session) {
                    handle.send(ActorMessage::Speak { text });
                }
            }

            EngineCommand::DestroyActor {
                session,
                actor_id: _,
            } => {
                session_handles.remove(&session);
                if let Some(jh) = decision_handles.remove(&session) {
                    jh.abort();
                }
                engine.lock().unwrap().remove_session(session, true);
            }

            EngineCommand::SessionDisconnect { session } => {
                session_handles.remove(&session);
                if let Some(jh) = decision_handles.remove(&session) {
                    jh.abort();
                }
                engine.lock().unwrap().remove_session(session, true);
                info!(session = session.0, "Session disconnected");
            }
        }
    }
}

// ── Per-connection handler ────────────────────────────────────────────────────

// ── Plain-HTTP fallback on the WS port ───────────────────────────────────────

/// Called when a TCP connection on the WS port does NOT contain an `Upgrade: websocket`
/// header — i.e. a browser hitting the address directly, a health check, etc.
async fn serve_http(mut stream: TcpStream, engine: SharedEngine) {
    let mut buf = [0u8; 4096];
    let n = match stream.read(&mut buf).await {
        Ok(n) if n > 0 => n,
        _ => return,
    };
    let req = std::str::from_utf8(&buf[..n]).unwrap_or("");
    let first_line = req.lines().next().unwrap_or("");
    let method = first_line.split_whitespace().next().unwrap_or("GET");
    let path = first_line.split_whitespace().nth(1).unwrap_or("/");
    let body = req
        .split("\r\n\r\n")
        .nth(1)
        .or_else(|| req.split("\n\n").nth(1))
        .unwrap_or("")
        .trim();

    let (status, ct, body) = metrics::handle_request(method, path, body, &engine).await;
    let response = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {ct}\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes()).await;
}

async fn handle_connection(
    stream: TcpStream,
    addr: SocketAddr,
    session: SessionId,
    cmd_tx: mpsc::Sender<EngineCommand>,
    engine: SharedEngine,
) {
    // Peek to decide: WebSocket upgrade or plain HTTP.
    let mut peek = [0u8; 512];
    let n = stream.peek(&mut peek).await.unwrap_or(0);
    let is_ws = std::str::from_utf8(&peek[..n])
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("upgrade: websocket");

    if !is_ws {
        serve_http(stream, engine).await;
        return;
    }

    let ws_stream = match tokio_tungstenite::accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            warn!(%addr, "WebSocket handshake failed: {}", e);
            return;
        }
    };

    let (mut ws_sink, mut ws_source) = ws_stream.split();

    // Internal channel: all outbound messages go through here so the inbound
    // loop and pump task don't need to share the sink.
    let (out_tx, mut out_rx) = mpsc::channel::<Vec<u8>>(128);

    metrics::inc_ws_connections();
    let out_task = tokio::spawn(async move {
        while let Some(bytes) = out_rx.recv().await {
            if ws_sink.send(Message::Binary(bytes)).await.is_err() {
                break;
            }
        }
    });

    // Auth: if LIVEWORLD_TOKEN is set, first message must be {"token":"..."}.
    if std::env::var("LIVEWORLD_TOKEN").is_ok() {
        match ws_source.next().await {
            Some(Ok(Message::Text(text))) => {
                let provided = serde_json::from_str::<serde_json::Value>(&text)
                    .ok()
                    .and_then(|v| v["token"].as_str().map(str::to_owned));
                if !crate::auth::validate_token(provided.as_deref()) {
                    send_error(&out_tx, 401, "Unauthorized").await;
                    out_task.abort();
                    return;
                }
            }
            _ => {
                out_task.abort();
                return;
            }
        }
    }

    let mut rate = RateLimiter::new(20); // 20 commands / second
    let mut pump_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut has_actor = false;

    while let Some(msg) = ws_source.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if !rate.allow() {
                    send_error(&out_tx, 429, "Rate limit exceeded").await;
                    continue;
                }

                match serde_json::from_str::<ClientCommand>(&text) {
                    Ok(cmd) => {
                        handle_client_command(
                            cmd,
                            session,
                            &cmd_tx,
                            &out_tx,
                            &mut pump_task,
                            &mut has_actor,
                        )
                        .await;
                    }
                    Err(e) => {
                        send_error(&out_tx, 400, &e.to_string()).await;
                    }
                }
            }
            Ok(Message::Binary(bytes)) => {
                // Accept binary JSON for clients that prefer it.
                if !rate.allow() {
                    send_error(&out_tx, 429, "Rate limit exceeded").await;
                    continue;
                }
                if let Ok(text) = std::str::from_utf8(&bytes) {
                    match serde_json::from_str::<ClientCommand>(text) {
                        Ok(cmd) => {
                            handle_client_command(
                                cmd,
                                session,
                                &cmd_tx,
                                &out_tx,
                                &mut pump_task,
                                &mut has_actor,
                            )
                            .await;
                        }
                        Err(e) => send_error(&out_tx, 400, &e.to_string()).await,
                    }
                }
            }
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {} // auto-handled by tungstenite
            _ => {}
        }
    }

    // Cleanup on disconnect.
    if let Some(p) = pump_task {
        p.abort();
    }
    out_task.abort();
    metrics::dec_ws_connections();
    let _ = cmd_tx
        .send(EngineCommand::SessionDisconnect { session })
        .await;
    info!(session = session.0, %addr, "Connection closed");
}

async fn handle_client_command(
    cmd: ClientCommand,
    session: SessionId,
    cmd_tx: &mpsc::Sender<EngineCommand>,
    out_tx: &mpsc::Sender<Vec<u8>>,
    pump_task: &mut Option<tokio::task::JoinHandle<()>>,
    has_actor: &mut bool,
) {
    match cmd {
        ClientCommand::CreateActor {
            name,
            personality,
            backstory,
            model,
            position,
        } => {
            if *has_actor {
                send_error(out_tx, 409, "Session already has an actor").await;
                return;
            }
            let (reply_tx, reply_rx) = oneshot::channel();
            if cmd_tx
                .send(EngineCommand::CreateActor {
                    session,
                    name,
                    personality,
                    backstory,
                    model,
                    position,
                    reply: reply_tx,
                })
                .await
                .is_err()
            {
                return;
            }
            match reply_rx.await {
                Ok(Ok((actor_id, queue))) => {
                    *has_actor = true;
                    // Spawn delta pump: drains the session queue every tick.
                    let pump_out = out_tx.clone();
                    *pump_task = Some(tokio::spawn(async move {
                        let mut ticker = interval(Duration::from_millis(40)); // 25 Hz
                        loop {
                            ticker.tick().await;
                            let deltas: Vec<_> = {
                                let mut q = queue.lock().unwrap();
                                q.drain(..).collect()
                            };
                            for delta in deltas {
                                let bytes = serde_json::to_vec(&ServerMessage::WorldDelta(delta))
                                    .unwrap_or_default();
                                if pump_out.send(bytes).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }));
                    let bytes = serde_json::to_vec(&ServerMessage::ActorCreated { actor_id })
                        .unwrap_or_default();
                    let _ = out_tx.send(bytes).await;
                }
                Ok(Err(e)) => send_error(out_tx, 400, &e).await,
                Err(_) => {} // sender dropped (engine shutting down)
            }
        }

        ClientCommand::MoveActor { to, .. } => {
            let _ = cmd_tx.send(EngineCommand::MoveActor { session, to }).await;
        }

        ClientCommand::ChatActor { text, .. } => {
            let _ = cmd_tx
                .send(EngineCommand::ChatActor { session, text })
                .await;
        }

        ClientCommand::DestroyActor { actor_id } => {
            if let Some(p) = pump_task.take() {
                p.abort();
            }
            *has_actor = false;
            let _ = cmd_tx
                .send(EngineCommand::DestroyActor { session, actor_id })
                .await;
        }
    }
}

async fn send_error(out_tx: &mpsc::Sender<Vec<u8>>, code: u32, message: &str) {
    let bytes = serde_json::to_vec(&ServerMessage::Error {
        code,
        message: message.to_owned(),
    })
    .unwrap_or_default();
    let _ = out_tx.send(bytes).await;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_unique() {
        let a = SessionId::next();
        let b = SessionId::next();
        assert_ne!(a, b);
    }

    #[test]
    fn client_command_deserializes() {
        let json = r#"{
            "type": "CreateActor",
            "name": "Wanderer",
            "personality": "curious",
            "backstory": "A traveler",
            "model": "Mock",
            "position": {"x": 100.0, "y": 200.0}
        }"#;
        let cmd: ClientCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, ClientCommand::CreateActor { .. }));
    }

    #[test]
    fn server_message_serializes() {
        let msg = ServerMessage::ActorCreated {
            actor_id: ActorId(42),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("ActorCreated"));
        assert!(json.contains("42"));
    }

    #[test]
    fn move_command_deserializes() {
        let json = r#"{"type":"MoveActor","actor_id":1,"to":{"x":50.0,"y":75.0}}"#;
        let cmd: ClientCommand = serde_json::from_str(json).unwrap();
        assert!(matches!(cmd, ClientCommand::MoveActor { .. }));
    }

    #[tokio::test]
    async fn command_channel_round_trip() {
        let (tx, mut rx) = mpsc::channel::<EngineCommand>(8);
        tx.send(EngineCommand::SessionDisconnect {
            session: SessionId(999),
        })
        .await
        .unwrap();
        let cmd = rx.recv().await.unwrap();
        assert!(matches!(cmd, EngineCommand::SessionDisconnect { .. }));
    }

    #[test]
    fn rate_limiter_blocks_after_limit() {
        let mut rl = RateLimiter::new(3);
        assert!(rl.allow());
        assert!(rl.allow());
        assert!(rl.allow());
        assert!(!rl.allow()); // 4th denied
    }
}

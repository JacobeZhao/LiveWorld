// WebSocket server: accepts connections, authenticates, routes client commands
// to the world engine, and streams state deltas back to each client at 25 Hz.

use crate::actor::ActorHandle;
use crate::types::{
    ActorId, ActorMessage, ActorSpec, ClientCommand, LlmModel, Position, ServerMessage,
    SessionId, WorldConfig,
};
use crate::world_engine::{SessionQueue, WorldEngine};
use ahash::AHashMap;
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{Duration, interval};
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

/// Thread-safe handle to the world engine.
pub type SharedEngine = Arc<Mutex<WorldEngine>>;

// ── Rate limiter (token bucket, per-connection) ───────────────────────────────

struct RateLimiter {
    count: u32,
    window_start: Instant,
    limit: u32,
}

impl RateLimiter {
    fn new(limit: u32) -> Self {
        Self { count: 0, window_start: Instant::now(), limit }
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

pub async fn run_ws_server(engine: SharedEngine, cfg: WorldConfig) -> Result<()> {
    let addr = format!("0.0.0.0:{}", cfg.ws_port);
    let listener = TcpListener::bind(&addr).await?;
    info!("WebSocket server listening on {}", addr);

    let (cmd_tx, cmd_rx) = mpsc::channel::<EngineCommand>(4096);
    // Single command-processor task serialises all engine mutations.
    tokio::spawn(command_processor(cmd_rx, Arc::clone(&engine)));

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                let session = SessionId::next();
                info!(session = session.0, %addr, "New connection");
                tokio::spawn(handle_connection(stream, addr, session, cmd_tx.clone()));
            }
            Err(e) => error!("Accept error: {}", e),
        }
    }
}

// ── Command processor ─────────────────────────────────────────────────────────

/// Runs as a single task; owns a local `session_handles` map so that
/// MoveActor / ChatActor bypass the engine lock entirely (SPSC push is lock-free).
async fn command_processor(
    mut cmd_rx: mpsc::Receiver<EngineCommand>,
    engine: SharedEngine,
) {
    let mut session_handles: AHashMap<SessionId, ActorHandle> = AHashMap::new();

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
                let position = Position::new(
                    position.x.clamp(0.0, 9_999.0),
                    position.y.clamp(0.0, 9_999.0),
                );
                let id = ActorId::next();
                let spec =
                    ActorSpec { id, name, personality, backstory, model, position };
                let (handle, queue) = {
                    let mut eng = engine.lock().unwrap();
                    eng.spawn_actor_for_session(spec, session)
                };
                session_handles.insert(session, handle);
                info!(session = session.0, actor = id.0, "Actor spawned");
                let _ = reply.send(Ok((id, queue)));
            }

            EngineCommand::MoveActor { session, to } => {
                let to = Position::new(
                    to.x.clamp(0.0, 9_999.0),
                    to.y.clamp(0.0, 9_999.0),
                );
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

            EngineCommand::DestroyActor { session, actor_id: _ } => {
                session_handles.remove(&session);
                let mut eng = engine.lock().unwrap();
                eng.remove_session(session, true);
            }

            EngineCommand::SessionDisconnect { session } => {
                session_handles.remove(&session);
                let mut eng = engine.lock().unwrap();
                eng.remove_session(session, true);
                info!(session = session.0, "Session disconnected");
            }
        }
    }
}

// ── Per-connection handler ────────────────────────────────────────────────────

async fn handle_connection(
    stream: TcpStream,
    addr: SocketAddr,
    session: SessionId,
    cmd_tx: mpsc::Sender<EngineCommand>,
) {
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

    let out_task = tokio::spawn(async move {
        while let Some(bytes) = out_rx.recv().await {
            if ws_sink.send(Message::Binary(bytes.into())).await.is_err() {
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
    let _ = cmd_tx.send(EngineCommand::SessionDisconnect { session }).await;
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
            let _ = cmd_tx.send(EngineCommand::ChatActor { session, text }).await;
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
    use crate::types::WorldConfig;

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
        let msg = ServerMessage::ActorCreated { actor_id: ActorId(42) };
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
        tx.send(EngineCommand::SessionDisconnect { session: SessionId(999) })
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

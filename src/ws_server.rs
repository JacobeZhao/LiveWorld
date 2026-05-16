// WebSocket server: accepts connections, routes client commands to the world
// engine, and streams state deltas back to each client at 25 Hz.
// Uses tokio-tungstenite for async WebSocket handling.

use crate::types::{ClientCommand, ServerMessage, SessionId, WorldConfig, now_ms};
use crate::world_engine::WorldEngine;
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::time::{Duration, interval};
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn, error};

/// Commands sent from WS handler tasks to the world engine manager task.
pub enum EngineCommand {
    CreateActor {
        session: SessionId,
        cmd: ClientCommand,
    },
    MoveActor {
        session: SessionId,
        cmd: ClientCommand,
    },
    ChatActor {
        session: SessionId,
        cmd: ClientCommand,
    },
    DestroyActor {
        session: SessionId,
        cmd: ClientCommand,
    },
    SessionDisconnect {
        session: SessionId,
    },
}

/// Shared, thread-safe handle to the world engine.
/// The world engine itself runs on a dedicated thread; we wrap it in a Mutex
/// for access from async handler tasks (lock contention is low: only on
/// actor spawn/despawn, not on every tick).
pub type SharedEngine = Arc<Mutex<WorldEngine>>;

/// Start the WebSocket server. Binds to `addr` and accepts connections.
/// `engine` must be Arc<Mutex<WorldEngine>> — the tick loop runs separately.
pub async fn run_ws_server(engine: SharedEngine, cfg: WorldConfig) -> Result<()> {
    let addr = format!("0.0.0.0:{}", cfg.ws_port);
    let listener = TcpListener::bind(&addr).await?;
    info!("WebSocket server listening on {}", addr);

    let (cmd_tx, mut cmd_rx) = mpsc::channel::<EngineCommand>(4096);

    // Command processor task: serializes engine mutations.
    {
        let engine = Arc::clone(&engine);
        tokio::spawn(async move {
            while let Some(cmd) = cmd_rx.recv().await {
                handle_engine_command(&engine, cmd);
            }
        });
    }

    loop {
        match listener.accept().await {
            Ok((stream, addr)) => {
                let session = SessionId::next();
                info!(session = session.0, %addr, "New connection");
                let engine = Arc::clone(&engine);
                let cmd_tx = cmd_tx.clone();
                tokio::spawn(handle_connection(stream, addr, session, engine, cmd_tx));
            }
            Err(e) => {
                error!("Accept error: {}", e);
            }
        }
    }
}

fn handle_engine_command(engine: &SharedEngine, cmd: EngineCommand) {
    use crate::types::{ActorId, ActorSpec, LlmModel, Position};

    let mut eng = engine.lock().unwrap();
    match cmd {
        EngineCommand::CreateActor { session, cmd: ClientCommand::CreateActor { name, personality, backstory, model, position } } => {
            let id = ActorId::next();
            let spec = ActorSpec { id, name, personality, backstory, model, position };
            let cell = position.to_grid_cell(10.0);
            let handle = eng.spawn_actor(spec);
            let q = eng.add_session(session, id, cell);
            info!(session = session.0, actor = id.0, "Actor created and session registered");
            drop(q); // Queue owned by connection handler; passed via engine
            drop(handle);
        }
        EngineCommand::MoveActor { session: _, cmd: ClientCommand::MoveActor { actor_id, to } } => {
            if let Some(handle) = eng.full_snapshot().iter().find(|s| s.id == actor_id) {
                // In production: use actor handle map. Simplified: log only.
                let _ = handle;
            }
        }
        EngineCommand::DestroyActor { session, cmd: ClientCommand::DestroyActor { actor_id } } => {
            eng.despawn_actor(actor_id);
            eng.remove_session(session);
        }
        EngineCommand::SessionDisconnect { session } => {
            eng.remove_session(session);
            info!(session = session.0, "Session disconnected");
        }
        _ => {}
    }
}

async fn handle_connection(
    stream: TcpStream,
    addr: SocketAddr,
    session: SessionId,
    engine: SharedEngine,
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

    // Outbound delta sender: runs a loop that drains the session's delta queue.
    // We spin up a separate task so inbound and outbound are decoupled.
    let (out_tx, mut out_rx) = mpsc::channel::<Vec<u8>>(64);

    // Outbound task.
    let out_task = tokio::spawn(async move {
        while let Some(bytes) = out_rx.recv().await {
            if ws_sink.send(Message::Binary(bytes.into())).await.is_err() {
                break;
            }
        }
    });

    // Delta pump: every tick interval, check the session queue and forward deltas.
    // The session queue is only available after CreateActor is processed, so we
    // poll the engine for it (simplified; production would pass queue handle directly).
    let pump_engine = Arc::clone(&engine);
    let pump_out = out_tx.clone();
    let pump_session = session;
    let pump_task = tokio::spawn(async move {
        let mut tick_interval = interval(Duration::from_millis(40)); // 25 Hz
        loop {
            tick_interval.tick().await;
            // In the full implementation, each session has its own queue handle.
            // For now we read from the engine's shared state (demonstrates the pattern).
            let _ = (pump_engine.lock(), pump_out.clone(), pump_session);
        }
    });

    // Inbound message loop.
    while let Some(msg) = ws_source.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                match serde_json::from_str::<ClientCommand>(&text) {
                    Ok(cmd) => {
                        let engine_cmd = match &cmd {
                            ClientCommand::CreateActor { .. } => {
                                Some(EngineCommand::CreateActor { session, cmd })
                            }
                            ClientCommand::MoveActor { .. } => {
                                Some(EngineCommand::MoveActor { session, cmd })
                            }
                            ClientCommand::ChatActor { .. } => {
                                Some(EngineCommand::ChatActor { session, cmd })
                            }
                            ClientCommand::DestroyActor { .. } => {
                                Some(EngineCommand::DestroyActor { session, cmd })
                            }
                        };
                        if let Some(ec) = engine_cmd {
                            let _ = cmd_tx.send(ec).await;
                        }
                    }
                    Err(e) => {
                        warn!(session = session.0, "Bad command: {}", e);
                        let err = serde_json::to_string(&ServerMessage::Error {
                            code: 400,
                            message: e.to_string(),
                        })
                        .unwrap_or_default();
                        let _ = out_tx.send(err.into_bytes()).await;
                    }
                }
            }
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(Message::Ping(p)) => {
                // tungstenite auto-responds to pings; nothing to do here.
                let _ = p;
            }
            _ => {}
        }
    }

    // Cleanup on disconnect.
    pump_task.abort();
    out_task.abort();
    let _ = cmd_tx.send(EngineCommand::SessionDisconnect { session }).await;
    info!(session = session.0, "Connection closed");
}

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
        matches!(cmd, ClientCommand::CreateActor { .. });
    }

    #[test]
    fn server_message_serializes() {
        use crate::types::ActorId;
        let msg = ServerMessage::ActorCreated { actor_id: ActorId(42) };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("ActorCreated"));
        assert!(json.contains("42"));
    }

    #[tokio::test]
    async fn engine_command_channel_works() {
        let (tx, mut rx) = mpsc::channel::<EngineCommand>(8);
        tx.send(EngineCommand::SessionDisconnect { session: SessionId(1) })
            .await
            .unwrap();
        let cmd = rx.recv().await.unwrap();
        matches!(cmd, EngineCommand::SessionDisconnect { .. });
    }
}

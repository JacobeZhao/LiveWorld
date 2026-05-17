use futures_util::{SinkExt, StreamExt};
/// WebSocket integration tests.
///
/// These tests start a real server on a random port, connect via WebSocket,
/// and verify the full request/response cycle end-to-end.
///
/// Run with: cargo test --test ws_integration
use liveworld::engine_api::EngineApi;
use liveworld::global_agents::SharedSnapshot;
use liveworld::types::{ClientCommand, LlmModel, Position, ServerMessage, WorldConfig};
use liveworld::world_engine::WorldEngine;
use liveworld::ws_server::SharedEngine;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::Message};

async fn start_test_server() -> (u16, tokio::task::JoinHandle<()>) {
    // Bind on port 0 so the OS picks a free port.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let mut cfg = WorldConfig::default();
    cfg.ws_port = port;

    let engine: SharedEngine = Arc::new(Mutex::new(
        Box::new(WorldEngine::new(cfg.clone())) as Box<dyn EngineApi + Send>
    ));
    let snapshot: SharedSnapshot = Arc::new(Mutex::new(ahash::AHashMap::new()));

    let handle = tokio::spawn(async move {
        let _ = liveworld::ws_server::run_ws_server(engine, cfg, snapshot).await;
    });

    // Give the server a moment to bind.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (port, handle)
}

#[tokio::test]
async fn test_create_actor_returns_id() {
    let (port, server) = start_test_server().await;

    let url = format!("ws://127.0.0.1:{port}");
    let (mut ws, _) = connect_async(&url).await.expect("WS connect failed");

    let cmd = ClientCommand::CreateActor {
        name: "TestBot".to_string(),
        personality: "friendly".to_string(),
        backstory: "A test robot".to_string(),
        model: LlmModel::Mock,
        position: Position::new(100.0, 100.0),
        role: None,
        faction: None,
    };
    let json = serde_json::to_string(&cmd).unwrap();
    ws.send(Message::Text(json.into())).await.unwrap();

    // Expect ActorCreated response.
    if let Some(Ok(Message::Binary(bytes))) = ws.next().await {
        let msg: ServerMessage = serde_json::from_slice(&bytes).unwrap();
        assert!(matches!(msg, ServerMessage::ActorCreated { .. }));
    } else {
        panic!("Expected Binary ActorCreated message");
    }

    ws.close(None).await.ok();
    server.abort();
}

#[tokio::test]
async fn test_move_actor() {
    let (port, server) = start_test_server().await;

    let url = format!("ws://127.0.0.1:{port}");
    let (mut ws, _) = connect_async(&url).await.expect("WS connect failed");

    // Create actor first.
    let create = ClientCommand::CreateActor {
        name: "Mover".to_string(),
        personality: "wanderer".to_string(),
        backstory: "Likes to walk".to_string(),
        model: LlmModel::Mock,
        position: Position::new(0.0, 0.0),
        role: None,
        faction: None,
    };
    ws.send(Message::Text(
        serde_json::to_string(&create).unwrap().into(),
    ))
    .await
    .unwrap();

    let actor_id = match ws.next().await {
        Some(Ok(Message::Binary(b))) => {
            match serde_json::from_slice::<ServerMessage>(&b).unwrap() {
                ServerMessage::ActorCreated { actor_id } => actor_id,
                other => panic!("Unexpected: {:?}", other),
            }
        }
        other => panic!("Expected Binary: {:?}", other),
    };

    // Send move command.
    let mv = ClientCommand::MoveActor {
        actor_id,
        to: Position::new(500.0, 500.0),
    };
    ws.send(Message::Text(serde_json::to_string(&mv).unwrap().into()))
        .await
        .unwrap();

    // No error response expected — success is silent.
    ws.close(None).await.ok();
    server.abort();
}

#[tokio::test]
async fn test_duplicate_actor_rejected() {
    let (port, server) = start_test_server().await;

    let url = format!("ws://127.0.0.1:{port}");
    let (mut ws, _) = connect_async(&url).await.expect("WS connect failed");

    let create = || ClientCommand::CreateActor {
        name: "Bot".to_string(),
        personality: "bold".to_string(),
        backstory: "Lives twice".to_string(),
        model: LlmModel::Mock,
        position: Position::new(0.0, 0.0),
        role: None,
        faction: None,
    };

    // First creation succeeds.
    ws.send(Message::Text(
        serde_json::to_string(&create()).unwrap().into(),
    ))
    .await
    .unwrap();
    let _ = ws.next().await; // consume ActorCreated

    // Second creation on the same session should fail with 409.
    ws.send(Message::Text(
        serde_json::to_string(&create()).unwrap().into(),
    ))
    .await
    .unwrap();
    if let Some(Ok(Message::Binary(b))) = ws.next().await {
        match serde_json::from_slice::<ServerMessage>(&b).unwrap() {
            ServerMessage::Error { code, .. } => assert_eq!(code, 409),
            other => panic!("Expected Error 409, got: {:?}", other),
        }
    }

    ws.close(None).await.ok();
    server.abort();
}

#[tokio::test]
async fn test_name_too_long_rejected() {
    let (port, server) = start_test_server().await;

    let url = format!("ws://127.0.0.1:{port}");
    let (mut ws, _) = connect_async(&url).await.expect("WS connect failed");

    let create = ClientCommand::CreateActor {
        name: "x".repeat(65),
        personality: "test".to_string(),
        backstory: "test".to_string(),
        model: LlmModel::Mock,
        position: Position::new(0.0, 0.0),
        role: None,
        faction: None,
    };
    ws.send(Message::Text(
        serde_json::to_string(&create).unwrap().into(),
    ))
    .await
    .unwrap();

    if let Some(Ok(Message::Binary(b))) = ws.next().await {
        match serde_json::from_slice::<ServerMessage>(&b).unwrap() {
            ServerMessage::Error { code, .. } => assert_eq!(code, 400),
            other => panic!("Expected Error 400, got: {:?}", other),
        }
    }

    ws.close(None).await.ok();
    server.abort();
}

#[tokio::test]
async fn test_rate_limit_enforced() {
    let (port, server) = start_test_server().await;

    let url = format!("ws://127.0.0.1:{port}");
    let (mut ws, _) = connect_async(&url).await.expect("WS connect failed");

    // Send 25 move commands rapidly (limit is 20/s). The first create is needed for session.
    let create = ClientCommand::CreateActor {
        name: "RateBot".to_string(),
        personality: "fast".to_string(),
        backstory: "test".to_string(),
        model: LlmModel::Mock,
        position: Position::new(0.0, 0.0),
        role: None,
        faction: None,
    };
    ws.send(Message::Text(
        serde_json::to_string(&create).unwrap().into(),
    ))
    .await
    .unwrap();
    let _ = ws.next().await; // consume ActorCreated

    // Flood with move commands — some should be rate-limited.
    for i in 0..25u32 {
        let mv = ClientCommand::MoveActor {
            actor_id: liveworld::types::ActorId(1),
            to: Position::new(i as f32, 0.0),
        };
        ws.send(Message::Text(serde_json::to_string(&mv).unwrap().into()))
            .await
            .unwrap();
    }

    // Read responses — expect at least one 429.
    let mut got_429 = false;
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(200);
    loop {
        match tokio::time::timeout_at(deadline, ws.next()).await {
            Ok(Some(Ok(Message::Binary(b)))) => {
                if let Ok(ServerMessage::Error { code: 429, .. }) =
                    serde_json::from_slice::<ServerMessage>(&b)
                {
                    got_429 = true;
                    break;
                }
            }
            _ => break,
        }
    }
    assert!(got_429, "Expected a 429 rate-limit response");

    ws.close(None).await.ok();
    server.abort();
}

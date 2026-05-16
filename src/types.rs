use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

// ── ID types ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ActorId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GridCell(pub i32, pub i32);

static NEXT_ACTOR_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

impl ActorId {
    #[inline]
    pub fn next() -> Self {
        ActorId(NEXT_ACTOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl SessionId {
    #[inline]
    pub fn next() -> Self {
        SessionId(NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed))
    }
}

// ── World position ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

impl Position {
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn to_grid_cell(self, cell_size: f32) -> GridCell {
        GridCell(
            (self.x / cell_size).floor() as i32,
            (self.y / cell_size).floor() as i32,
        )
    }

    #[inline]
    pub fn distance_cells(self, other: Position, cell_size: f32) -> i32 {
        let a = self.to_grid_cell(cell_size);
        let b = other.to_grid_cell(cell_size);
        (a.0 - b.0).abs().max((a.1 - b.1).abs())
    }
}

// ── LLM model enum ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LlmModel {
    Gpt4o,
    ClaudeSonnet,
    ClaudeOpus,
    Ollama(String),
    Mock,
}

impl std::fmt::Display for LlmModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmModel::Gpt4o => write!(f, "gpt-4o"),
            LlmModel::ClaudeSonnet => write!(f, "claude-sonnet-4-6"),
            LlmModel::ClaudeOpus => write!(f, "claude-opus-4-7"),
            LlmModel::Ollama(m) => write!(f, "ollama/{m}"),
            LlmModel::Mock => write!(f, "mock"),
        }
    }
}

// ── Actor state ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorSpec {
    pub id: ActorId,
    pub name: String,
    pub personality: String,
    pub backstory: String,
    pub model: LlmModel,
    pub position: Position,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorState {
    pub id: ActorId,
    pub name: String,
    pub position: Position,
    pub cell: GridCell,
    pub tick: u64,
    pub last_utterance: Option<String>,
}

// ── World events (Actor → Actor messages) ────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActorMessage {
    Move { to: Position },
    Speak { text: String },
    Interact { target: ActorId, action: String },
    Shutdown,
}

// ── State delta frame (server → client, per-tick) ────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDelta {
    pub tick: u64,
    pub timestamp_ms: u64,
    pub updates: Vec<ActorState>,
    pub removed: Vec<ActorId>,
}

// ── World config ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    pub grid_width: i32,
    pub grid_height: i32,
    pub cell_size: f32,
    pub interest_radius: i32,
    pub tick_hz: u32,
    pub snapshot_interval_secs: u64,
    pub max_actors: usize,
    pub ws_port: u16,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            grid_width: 1000,
            grid_height: 1000,
            cell_size: 10.0,
            interest_radius: 5,
            tick_hz: 25,
            snapshot_interval_secs: 60,
            max_actors: 100_000,
            ws_port: 8080,
        }
    }
}

// ── Client commands (client → server) ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientCommand {
    CreateActor {
        name: String,
        personality: String,
        backstory: String,
        model: LlmModel,
        position: Position,
    },
    MoveActor {
        actor_id: ActorId,
        to: Position,
    },
    ChatActor {
        actor_id: ActorId,
        text: String,
    },
    DestroyActor {
        actor_id: ActorId,
    },
}

// ── Server responses ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    ActorCreated { actor_id: ActorId },
    WorldDelta(StateDelta),
    Error { code: u32, message: String },
}

// ── Utility: current timestamp in ms ─────────────────────────────────────────

#[inline]
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actor_id_monotonic() {
        let a = ActorId::next();
        let b = ActorId::next();
        assert!(b.0 > a.0);
    }

    #[test]
    fn position_to_grid_cell() {
        let p = Position::new(25.5, 35.9);
        let cell = p.to_grid_cell(10.0);
        assert_eq!(cell, GridCell(2, 3));
    }

    #[test]
    fn position_grid_edge() {
        let p = Position::new(0.0, 0.0);
        assert_eq!(p.to_grid_cell(10.0), GridCell(0, 0));
    }

    #[test]
    fn distance_cells() {
        let a = Position::new(5.0, 5.0);
        let b = Position::new(55.0, 5.0);
        assert_eq!(a.distance_cells(b, 10.0), 5);
    }

    #[test]
    fn actor_state_serde_roundtrip() {
        let state = ActorState {
            id: ActorId(42),
            name: "TestBot".to_string(),
            position: Position::new(1.0, 2.0),
            cell: GridCell(0, 0),
            tick: 100,
            last_utterance: Some("Hello".to_string()),
        };
        let json = serde_json::to_string(&state).unwrap();
        let back: ActorState = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, state.id);
        assert_eq!(back.name, state.name);
        assert_eq!(back.tick, state.tick);
    }
}

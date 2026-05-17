// Persistence: periodic full-world snapshot to disk using bincode.
// Format: snapshot_<tick>.bin with a "latest" symlink/pointer file.
// Recovery: load latest snapshot and re-inject all actors into the runtime.

use crate::types::{ActorSpec, ActorState, LlmModel, Position};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn};

/// Everything needed to restore a world from cold start.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub tick: u64,
    pub timestamp_ms: u64,
    pub actors: Vec<ActorSnapshot>,
}

/// Snapshot of a single actor (spec + state).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorSnapshot {
    pub spec: PersistedSpec,
    pub state: ActorState,
}

/// The subset of ActorSpec that needs to be persisted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedSpec {
    pub id: crate::types::ActorId,
    pub name: String,
    pub personality: String,
    pub backstory: String,
    pub model_key: String,
}

impl PersistedSpec {
    pub fn from_spec(spec: &ActorSpec) -> Self {
        Self {
            id: spec.id,
            name: spec.name.clone(),
            personality: spec.personality.clone(),
            backstory: spec.backstory.clone(),
            model_key: spec.model.to_string(),
        }
    }

    pub fn into_spec(self, position: Position) -> ActorSpec {
        let model = match self.model_key.as_str() {
            "gpt-4o" => LlmModel::Gpt4o,
            "claude-sonnet-4-6" => LlmModel::ClaudeSonnet,
            "claude-opus-4-7" => LlmModel::ClaudeOpus,
            "mock" => LlmModel::Mock,
            other if other.starts_with("ollama/") => LlmModel::Ollama(other[7..].to_string()),
            _ => LlmModel::Mock,
        };
        ActorSpec {
            id: self.id,
            name: self.name,
            personality: self.personality,
            backstory: self.backstory,
            model,
            position,
        }
    }
}

pub struct SnapshotStore {
    dir: PathBuf,
    max_snapshots: usize,
}

impl SnapshotStore {
    pub fn new(dir: impl Into<PathBuf>, max_snapshots: usize) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create snapshot dir: {}", dir.display()))?;
        Ok(Self { dir, max_snapshots })
    }

    /// Write a snapshot to disk. Returns the file path.
    pub fn write(&mut self, snapshot: &WorldSnapshot) -> Result<PathBuf> {
        let filename = format!("snapshot_{:020}.bin", snapshot.tick);
        let path = self.dir.join(&filename);

        let bytes = bincode::serialize(snapshot).context("Failed to serialize snapshot")?;
        fs::write(&path, &bytes)
            .with_context(|| format!("Failed to write snapshot to {}", path.display()))?;

        // Write pointer to latest snapshot.
        let latest_path = self.dir.join("latest.txt");
        fs::write(&latest_path, filename).context("Failed to update latest pointer")?;

        info!(tick = snapshot.tick, actors = snapshot.actors.len(), path = %path.display(), "Snapshot written");

        self.prune_old_snapshots()?;
        Ok(path)
    }

    /// Read the latest snapshot from disk.
    pub fn read_latest(&self) -> Result<Option<WorldSnapshot>> {
        let latest_path = self.dir.join("latest.txt");
        if !latest_path.exists() {
            return Ok(None);
        }

        let filename = fs::read_to_string(&latest_path).context("Failed to read latest pointer")?;
        let path = self.dir.join(filename.trim());

        if !path.exists() {
            warn!(path = %path.display(), "Latest pointer references missing file");
            return Ok(None);
        }

        let bytes = fs::read(&path)
            .with_context(|| format!("Failed to read snapshot from {}", path.display()))?;
        let snapshot: WorldSnapshot =
            bincode::deserialize(&bytes).context("Failed to deserialize snapshot")?;

        info!(
            tick = snapshot.tick,
            actors = snapshot.actors.len(),
            "Snapshot loaded"
        );
        Ok(Some(snapshot))
    }

    /// Delete old snapshots keeping only the most recent `max_snapshots`.
    fn prune_old_snapshots(&self) -> Result<()> {
        let mut entries: Vec<PathBuf> = fs::read_dir(&self.dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("snapshot_") && n.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .collect();

        entries.sort();

        if entries.len() > self.max_snapshots {
            let to_remove = entries.len() - self.max_snapshots;
            for path in entries.iter().take(to_remove) {
                fs::remove_file(path)?;
            }
        }
        Ok(())
    }
}

/// Build a WorldSnapshot from current world state.
pub fn build_snapshot(
    tick: u64,
    actor_states: &[ActorState],
    actor_specs: &ahash::AHashMap<crate::types::ActorId, ActorSpec>,
) -> WorldSnapshot {
    let actors = actor_states
        .iter()
        .filter_map(|state| {
            actor_specs.get(&state.id).map(|spec| ActorSnapshot {
                spec: PersistedSpec::from_spec(spec),
                state: state.clone(),
            })
        })
        .collect();

    WorldSnapshot {
        tick,
        timestamp_ms: crate::types::now_ms(),
        actors,
    }
}

/// Restore from snapshot: returns a list of ActorSpecs ready to be spawned.
pub fn restore_actors(snapshot: &WorldSnapshot) -> Vec<ActorSpec> {
    let start = Instant::now();
    let specs: Vec<ActorSpec> = snapshot
        .actors
        .iter()
        .map(|a| a.spec.clone().into_spec(a.state.position))
        .collect();
    info!(
        count = specs.len(),
        elapsed_ms = start.elapsed().as_millis(),
        "Actors restored from snapshot"
    );
    specs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActorId, GridCell, LlmModel, Position};
    use tempfile::tempdir;

    fn make_state(id: u64) -> ActorState {
        ActorState {
            id: ActorId(id),
            name: format!("A{id}"),
            position: Position::new(id as f32 * 10.0, 0.0),
            cell: GridCell(id as i32, 0),
            tick: 100,
            last_utterance: None,
        }
    }

    fn make_snapshot(tick: u64, n: usize) -> WorldSnapshot {
        WorldSnapshot {
            tick,
            timestamp_ms: 0,
            actors: (1..=n as u64)
                .map(|i| ActorSnapshot {
                    spec: PersistedSpec {
                        id: ActorId(i),
                        name: format!("A{i}"),
                        personality: "curious".to_string(),
                        backstory: "wanderer".to_string(),
                        model_key: "mock".to_string(),
                    },
                    state: make_state(i),
                })
                .collect(),
        }
    }

    #[test]
    fn write_and_read_roundtrip() {
        let dir = tempdir().unwrap();
        let mut store = SnapshotStore::new(dir.path(), 5).unwrap();

        let snap = make_snapshot(42, 100);
        store.write(&snap).unwrap();

        let loaded = store.read_latest().unwrap().unwrap();
        assert_eq!(loaded.tick, 42);
        assert_eq!(loaded.actors.len(), 100);
        assert_eq!(loaded.actors[0].spec.id, ActorId(1));
    }

    #[test]
    fn restore_produces_correct_specs() {
        let snap = make_snapshot(1, 50);
        let specs = restore_actors(&snap);
        assert_eq!(specs.len(), 50);
        for spec in &specs {
            assert_eq!(spec.model, LlmModel::Mock);
        }
    }

    #[test]
    fn prune_keeps_max_snapshots() {
        let dir = tempdir().unwrap();
        let mut store = SnapshotStore::new(dir.path(), 3).unwrap();
        for tick in 0..10u64 {
            store.write(&make_snapshot(tick, 1)).unwrap();
        }
        let count = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name().to_string_lossy().starts_with("snapshot_")
                    && e.file_name().to_string_lossy().ends_with(".bin")
            })
            .count();
        assert_eq!(count, 3, "Expected exactly 3 snapshots after pruning");
    }

    #[test]
    fn latest_returns_none_when_empty() {
        let dir = tempdir().unwrap();
        let store = SnapshotStore::new(dir.path(), 5).unwrap();
        assert!(store.read_latest().unwrap().is_none());
    }

    #[test]
    fn recovery_time_under_2s_for_10k_actors() {
        let dir = tempdir().unwrap();
        let mut store = SnapshotStore::new(dir.path(), 5).unwrap();
        let snap = make_snapshot(1, 10_000);

        let write_start = Instant::now();
        store.write(&snap).unwrap();
        let write_ms = write_start.elapsed().as_millis();

        let read_start = Instant::now();
        let loaded = store.read_latest().unwrap().unwrap();
        let _specs = restore_actors(&loaded);
        let read_ms = read_start.elapsed().as_millis();

        assert!(
            read_ms < 2000,
            "Recovery took {}ms, must be < 2000ms",
            read_ms
        );
        println!("Write: {}ms, Recovery: {}ms", write_ms, read_ms);
    }

    #[test]
    fn persisted_spec_model_roundtrip() {
        for model in [
            LlmModel::Mock,
            LlmModel::Gpt4o,
            LlmModel::ClaudeSonnet,
            LlmModel::Ollama("llama3".to_string()),
        ] {
            let spec = ActorSpec {
                id: ActorId(1),
                name: "x".to_string(),
                personality: "y".to_string(),
                backstory: "z".to_string(),
                model: model.clone(),
                position: Position::new(0.0, 0.0),
            };
            let persisted = PersistedSpec::from_spec(&spec);
            let restored = persisted.into_spec(Position::new(0.0, 0.0));
            assert_eq!(restored.model, model);
        }
    }
}

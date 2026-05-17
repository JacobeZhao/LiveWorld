/// Cross-pod world state synchronisation via Redis pub/sub.
///
/// When `REDIS_URL` is set:
///   - A publisher task sends only the *delta* (new/updated + removed actors)
///     to `liveworld:world_state` once per second, tagged with the pod's `POD_NAME`.
///   - A subscriber task merges remote deltas into the local `SharedSnapshot`,
///     removing tombstoned actors and adding new ones (local actors are authoritative).
///   - Both publisher and subscriber reconnect with exponential back-off on error.
///
/// When `REDIS_URL` is NOT set → single-pod mode; this module is a no-op.
use crate::global_agents::SharedSnapshot;
use crate::types::{ActorId, ActorState};
use anyhow::Result;
use futures_util::StreamExt;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{error, info, warn};

const CHANNEL: &str = "liveworld:world_state";
const PUBLISH_INTERVAL_MS: u64 = 1_000;
const RECONNECT_BASE_MS: u64 = 500;
const RECONNECT_MAX_MS: u64 = 30_000;

#[derive(Serialize, Deserialize)]
struct SyncPayload {
    pod_id: String,
    /// Actors that were added or updated since the last publish.
    updates: Vec<ActorState>,
    /// Actor IDs that were removed since the last publish.
    removed: Vec<ActorId>,
}

pub async fn run_redis_sync(local_snapshot: SharedSnapshot) -> Result<()> {
    let url = match std::env::var("REDIS_URL") {
        Ok(u) if !u.is_empty() => u,
        _ => {
            info!("REDIS_URL not set — cross-pod sync disabled (single-pod mode)");
            return Ok(());
        }
    };

    let pod_id =
        std::env::var("POD_NAME").unwrap_or_else(|_| format!("pod-{}", std::process::id()));

    info!(%url, %pod_id, "Redis cross-pod sync enabled");

    let client = redis::Client::open(url.as_str())?;

    // Publisher: compute delta vs previous snapshot, publish only changes.
    {
        let pub_client = client.clone();
        let pub_snap = std::sync::Arc::clone(&local_snapshot);
        let my_pod_id = pod_id.clone();
        tokio::spawn(async move {
            publisher_loop(pub_client, pub_snap, my_pod_id).await;
        });
    }

    // Subscriber: reconnect loop.
    subscriber_loop(client, local_snapshot, pod_id).await;

    Ok(())
}

async fn publisher_loop(client: redis::Client, snap: SharedSnapshot, pod_id: String) {
    let mut ticker = tokio::time::interval(Duration::from_millis(PUBLISH_INTERVAL_MS));
    let mut prev: ahash::AHashMap<ActorId, ActorState> = ahash::AHashMap::new();
    let mut backoff_ms = RECONNECT_BASE_MS;

    loop {
        ticker.tick().await;

        let conn_result = client.get_async_connection().await;
        let mut conn = match conn_result {
            Ok(c) => {
                backoff_ms = RECONNECT_BASE_MS;
                c
            }
            Err(e) => {
                warn!(err = %e, backoff_ms, "Redis publish connect failed; retrying");
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(RECONNECT_MAX_MS);
                continue;
            }
        };

        let current: ahash::AHashMap<ActorId, ActorState> = snap.lock().unwrap().clone();

        // Compute delta.
        let updates: Vec<ActorState> = current
            .values()
            .filter(|s| prev.get(&s.id) != Some(*s))
            .cloned()
            .collect();
        let removed: Vec<ActorId> = prev
            .keys()
            .filter(|id| !current.contains_key(id))
            .copied()
            .collect();

        prev = current;

        if updates.is_empty() && removed.is_empty() {
            continue;
        }

        let payload = SyncPayload {
            pod_id: pod_id.clone(),
            updates,
            removed,
        };
        match serde_json::to_string(&payload) {
            Ok(json) => {
                let res: std::result::Result<i64, _> = conn.publish(CHANNEL, &json).await;
                if let Err(e) = res {
                    error!(err = %e, "Redis publish failed");
                }
            }
            Err(e) => error!(err = %e, "Failed to serialise delta for Redis"),
        }
    }
}

async fn subscriber_loop(client: redis::Client, local_snapshot: SharedSnapshot, pod_id: String) {
    let mut backoff_ms = RECONNECT_BASE_MS;
    loop {
        match client.get_async_connection().await {
            Ok(conn) => {
                backoff_ms = RECONNECT_BASE_MS;
                info!("Redis subscriber connected");
                if let Err(e) = run_subscriber(conn, &local_snapshot, &pod_id).await {
                    warn!(err = %e, "Redis subscriber error; reconnecting");
                }
            }
            Err(e) => {
                warn!(err = %e, backoff_ms, "Redis subscriber connect failed; retrying");
            }
        }
        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        backoff_ms = (backoff_ms * 2).min(RECONNECT_MAX_MS);
    }
}

async fn run_subscriber(
    conn: redis::aio::Connection,
    local_snapshot: &SharedSnapshot,
    pod_id: &str,
) -> Result<()> {
    let mut pubsub = conn.into_pubsub();
    pubsub.subscribe(CHANNEL).await?;
    let mut stream = pubsub.on_message();
    while let Some(msg) = stream.next().await {
        let raw: String = match msg.get_payload() {
            Ok(p) => p,
            Err(e) => {
                warn!(err = %e, "Failed to decode Redis message");
                continue;
            }
        };
        match serde_json::from_str::<SyncPayload>(&raw) {
            Ok(payload) if payload.pod_id != pod_id => {
                apply_remote_delta(local_snapshot, payload.updates, payload.removed);
            }
            Ok(_) => {} // own message; skip
            Err(e) => warn!(err = %e, "Failed to deserialise remote delta"),
        }
    }
    Ok(())
}

/// Merge a remote delta into the local snapshot.
/// Local actors (those present in the local engine) are never overwritten.
/// Remote actors that appear in `removed` are evicted from the snapshot.
fn apply_remote_delta(
    local_snapshot: &SharedSnapshot,
    updates: Vec<ActorState>,
    removed: Vec<ActorId>,
) {
    let mut snap = local_snapshot.lock().unwrap();
    for id in removed {
        snap.remove(&id);
    }
    for state in updates {
        snap.entry(state.id).or_insert(state);
    }
}

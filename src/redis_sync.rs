/// Cross-pod world state synchronisation via Redis pub/sub.
///
/// When `REDIS_URL` is set:
///   - A publisher task sends the local actor snapshot to `liveworld:world_state`
///     every second, tagged with the pod's `POD_NAME`.
///   - A subscriber task merges snapshots from *other* pods into the local
///     `SharedSnapshot` so clients on any pod can see the full world.
///
/// When `REDIS_URL` is NOT set → single-pod mode; this module is a no-op.
use crate::global_agents::SharedSnapshot;
use crate::types::ActorState;
use anyhow::Result;
use futures_util::StreamExt;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

const CHANNEL: &str = "liveworld:world_state";
const PUBLISH_INTERVAL_MS: u64 = 1_000;

#[derive(Serialize, Deserialize)]
struct SyncPayload {
    pod_id: String,
    states: Vec<ActorState>,
}

pub async fn run_redis_sync(local_snapshot: SharedSnapshot) -> Result<()> {
    let url = match std::env::var("REDIS_URL") {
        Ok(u) if !u.is_empty() => u,
        _ => {
            info!("REDIS_URL not set — cross-pod sync disabled (single-pod mode)");
            return Ok(());
        }
    };

    let pod_id = std::env::var("POD_NAME").unwrap_or_else(|_| {
        format!("pod-{}", std::process::id())
    });

    info!(%url, %pod_id, "Redis cross-pod sync enabled");

    let client = redis::Client::open(url.as_str())?;

    // Publisher: push local snapshot every second.
    {
        let pub_client = client.clone();
        let pub_snap = Arc::clone(&local_snapshot);
        let my_pod_id = pod_id.clone();
        tokio::spawn(async move {
            let mut ticker =
                tokio::time::interval(Duration::from_millis(PUBLISH_INTERVAL_MS));
            loop {
                ticker.tick().await;
                let mut conn = match pub_client.get_async_connection().await {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(err = %e, "Redis publish connect failed; retrying");
                        continue;
                    }
                };
                let states: Vec<ActorState> = pub_snap.lock().unwrap().clone();
                let payload = SyncPayload { pod_id: my_pod_id.clone(), states };
                match serde_json::to_string(&payload) {
                    Ok(json) => {
                        let _: std::result::Result<i64, _> =
                            conn.publish(CHANNEL, &json).await;
                    }
                    Err(e) => error!(err = %e, "Failed to serialise snapshot for Redis"),
                }
            }
        });
    }

    // Subscriber: merge snapshots from other pods.
    let mut pubsub = client.get_async_connection().await?.into_pubsub();
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
                merge_remote_states(&local_snapshot, payload.states);
            }
            Ok(_) => {} // own message; skip
            Err(e) => warn!(err = %e, "Failed to deserialise remote snapshot"),
        }
    }

    Ok(())
}

/// Add remote actors that are not present locally. Local actors are authoritative
/// for their own positions and are never overwritten by remote data.
fn merge_remote_states(local_snapshot: &SharedSnapshot, remote: Vec<ActorState>) {
    let mut snap = local_snapshot.lock().unwrap();
    let local_ids: ahash::AHashSet<_> = snap.iter().map(|s| s.id).collect();
    for state in remote {
        if !local_ids.contains(&state.id) {
            snap.push(state);
        }
    }
}

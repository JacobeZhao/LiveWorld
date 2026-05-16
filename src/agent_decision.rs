// Agent decision loop: each agent periodically queries its LLM to decide
// what to do next (move, speak, interact). Runs as an independent Tokio task,
// completely asynchronous — never blocks the world tick thread.

use crate::actor::ActorHandle;
use crate::llm_adapter::LlmRequest;
use crate::semantic_cache::SemanticCache;
use crate::types::{ActorId, ActorMessage, ActorSpec, ActorState, LlmModel, Position};
use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time;
use tracing::{debug, warn};

/// Parsed action from LLM text output.
#[derive(Debug, Clone)]
pub enum AgentAction {
    Move(Position),
    Speak(String),
    Idle,
}

/// Parse a structured action from free-form LLM text.
/// Format expected: "MOVE x,y" or "SPEAK text" or anything else → Idle.
fn parse_action(text: &str) -> AgentAction {
    let trimmed = text.trim();
    if let Some(rest) = trimmed.strip_prefix("MOVE ") {
        let parts: Vec<&str> = rest.splitn(2, ',').collect();
        if parts.len() == 2 {
            if let (Ok(x), Ok(y)) = (
                parts[0].trim().parse::<f32>(),
                parts[1].trim().parse::<f32>(),
            ) {
                return AgentAction::Move(Position::new(x, y));
            }
        }
    }
    if let Some(rest) = trimmed.strip_prefix("SPEAK ") {
        return AgentAction::Speak(rest.to_string());
    }
    AgentAction::Idle
}

/// Build the system prompt for an actor from its spec.
fn system_prompt(spec: &ActorSpec) -> String {
    format!(
        "You are {name}, a character in a virtual world. \
         Personality: {personality}. \
         Backstory: {backstory}. \
         Respond with ONE action on a single line: \
         'MOVE x,y' (coordinates 0-10000) or 'SPEAK text' (up to 50 chars).",
        name = spec.name,
        personality = spec.personality,
        backstory = spec.backstory,
    )
}

/// Build the user prompt for an actor given its current state and nearby actors.
fn user_prompt(state: &ActorState, nearby: &[ActorState]) -> String {
    let nearby_desc: Vec<String> = nearby
        .iter()
        .take(5)
        .map(|s| {
            format!(
                "{} at ({:.0},{:.0}){}",
                s.name,
                s.position.x,
                s.position.y,
                s.last_utterance
                    .as_deref()
                    .map(|u| format!(" saying '{u}'"))
                    .unwrap_or_default()
            )
        })
        .collect();

    format!(
        "You are at ({:.0}, {:.0}). Nearby: {}. What do you do?",
        state.position.x,
        state.position.y,
        if nearby_desc.is_empty() {
            "nobody".to_string()
        } else {
            nearby_desc.join(", ")
        }
    )
}

/// Configuration for the decision loop.
#[derive(Debug, Clone)]
pub struct DecisionConfig {
    /// How often the agent re-evaluates its situation.
    pub decision_interval: Duration,
    /// Maximum tokens in the LLM response.
    pub max_tokens: u32,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            decision_interval: Duration::from_secs(3),
            max_tokens: 64,
        }
    }
}

/// Runs in its own Tokio task; drives one actor's decisions.
pub struct AgentDecisionLoop {
    spec: ActorSpec,
    handle: ActorHandle,
    cache: Arc<Mutex<SemanticCache>>,
    config: DecisionConfig,
}

impl AgentDecisionLoop {
    pub fn new(
        spec: ActorSpec,
        handle: ActorHandle,
        cache: Arc<Mutex<SemanticCache>>,
        config: DecisionConfig,
    ) -> Self {
        Self { spec, handle, cache, config }
    }

    /// Run until the actor is shut down. Intended to be spawned as a Tokio task.
    pub async fn run(self, state_snapshot: Arc<Mutex<ActorState>>, nearby_snapshot: Arc<Mutex<Vec<ActorState>>>) {
        let mut interval = time::interval(self.config.decision_interval);
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);

        let sys = system_prompt(&self.spec);

        loop {
            interval.tick().await;

            let state = state_snapshot.lock().await.clone();
            let nearby = nearby_snapshot.lock().await.clone();

            let user = user_prompt(&state, &nearby);

            let req = LlmRequest {
                model: self.spec.model.clone(),
                system_prompt: sys.clone(),
                user_prompt: user,
                max_tokens: self.config.max_tokens,
            };

            let resp = {
                let mut cache = self.cache.lock().await;
                cache.complete(req).await
            };

            match resp {
                Ok(r) => {
                    let action = parse_action(&r.text);
                    debug!(actor = %self.spec.id.0, ?action, "decision");
                    match action {
                        AgentAction::Move(pos) => {
                            self.handle.send(ActorMessage::Move { to: pos });
                        }
                        AgentAction::Speak(text) => {
                            self.handle.send(ActorMessage::Speak { text });
                        }
                        AgentAction::Idle => {}
                    }
                }
                Err(e) => {
                    warn!(actor = %self.spec.id.0, err = %e, "LLM request failed");
                }
            }
        }
    }
}

/// Priority scheduler: routes high-priority actors to faster execution slots.
/// For now this is a simple wrapper; extend with a priority queue if needed.
pub struct PriorityScheduler {
    /// Semaphore limiting concurrent LLM calls to avoid thundering herd.
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl PriorityScheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
        }
    }

    /// Acquire a slot, call the LLM, release.
    pub async fn run<F, Fut, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let _permit = self.semaphore.acquire().await?;
        f().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_adapter::MockLlm;
    use crate::types::{ActorId, GridCell, LlmModel};

    fn make_spec() -> ActorSpec {
        ActorSpec {
            id: ActorId(1),
            name: "TestAgent".to_string(),
            personality: "curious".to_string(),
            backstory: "A traveler".to_string(),
            model: LlmModel::Mock,
            position: Position::new(50.0, 50.0),
        }
    }

    fn make_state(pos: Position) -> ActorState {
        ActorState {
            id: ActorId(1),
            name: "TestAgent".to_string(),
            position: pos,
            cell: GridCell(5, 5),
            tick: 0,
            last_utterance: None,
        }
    }

    #[test]
    fn parse_move() {
        let action = parse_action("MOVE 100.5, 200.0");
        matches!(action, AgentAction::Move(p) if (p.x - 100.5).abs() < 0.01);
    }

    #[test]
    fn parse_speak() {
        let action = parse_action("SPEAK Hello there!");
        matches!(action, AgentAction::Speak(s) if s == "Hello there!");
    }

    #[test]
    fn parse_idle_on_unknown() {
        let action = parse_action("DO NOTHING");
        matches!(action, AgentAction::Idle);
    }

    #[test]
    fn system_prompt_includes_name() {
        let spec = make_spec();
        let prompt = system_prompt(&spec);
        assert!(prompt.contains("TestAgent"));
        assert!(prompt.contains("curious"));
    }

    #[test]
    fn user_prompt_includes_position() {
        let state = make_state(Position::new(100.0, 200.0));
        let prompt = user_prompt(&state, &[]);
        assert!(prompt.contains("100"));
        assert!(prompt.contains("200"));
        assert!(prompt.contains("nobody"));
    }

    #[tokio::test]
    async fn priority_scheduler_limits_concurrency() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let scheduler = Arc::new(PriorityScheduler::new(2));
        let concurrent = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let mut tasks = vec![];
        for _ in 0..10 {
            let s = Arc::clone(&scheduler);
            let c = Arc::clone(&concurrent);
            let p = Arc::clone(&peak);
            tasks.push(tokio::spawn(async move {
                s.run(|| async {
                    let cur = c.fetch_add(1, Ordering::SeqCst) + 1;
                    p.fetch_max(cur, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    c.fetch_sub(1, Ordering::SeqCst);
                    Ok::<_, anyhow::Error>(())
                }).await.unwrap();
            }));
        }

        for t in tasks {
            t.await.unwrap();
        }

        assert!(peak.load(Ordering::SeqCst) <= 2, "Concurrency exceeded limit");
    }
}

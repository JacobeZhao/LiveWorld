// Agent decision loop: each agent periodically queries its LLM to decide
// what to do next (move, speak, attack). Runs as an independent Tokio task,
// completely asynchronous — never blocks the world tick thread.

use crate::actor::ActorHandle;
use crate::circuit_breaker::CircuitBreaker;
use crate::llm_adapter::LlmRequest;
use crate::metrics;
use crate::semantic_cache::SemanticCache;
use crate::types::{ActorMessage, ActorRole, ActorSpec, ActorState, Faction, Position};
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
    Attack(String), // target actor name
    Idle,
}

/// Parse a structured action from free-form LLM text.
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
    if let Some(rest) = trimmed.strip_prefix("ATTACK ") {
        return AgentAction::Attack(rest.trim().to_string());
    }
    AgentAction::Idle
}

/// Build the system prompt for an actor from its spec (role + faction aware).
fn system_prompt(spec: &ActorSpec) -> String {
    let role_hint = match spec.role {
        ActorRole::Merchant => "seek others to trade with, share market news",
        ActorRole::Scholar => "observe and share wisdom or historical lore",
        ActorRole::Knight => "patrol, protect allies, challenge enemies when you meet them",
        ActorRole::Mage => "move mysteriously, speak of arcane matters",
        ActorRole::Bard => "entertain everyone nearby, tell stories and sing",
        ActorRole::Guard => "stay near your post, watch for enemies",
        ActorRole::Wanderer => "roam freely and explore the world",
    };
    let faction_hint = match spec.faction {
        Faction::Empire => "You serve the Empire. Alliance members are your enemies.",
        Faction::Alliance => "You fight for the Alliance. Empire soldiers are your enemies.",
        Faction::WanderersGuild => "You owe loyalty to no one. Trade with all, trust few.",
        Faction::MagesCircle => "Knowledge is power. Avoid combat; study and advise.",
        Faction::Neutral => "You are independent and neutral.",
    };
    format!(
        "You are {name}, a character in a fantasy world at war. \
         Role: {role_hint}. {faction_hint} \
         Personality: {personality}. Backstory: {backstory}. \
         Respond with ONE action on a single line: \
         'MOVE x,y' (coordinates 0-10000) | 'SPEAK text (max 60 chars)' | \
         'ATTACK name' (only use against hostile faction members).",
        name = spec.name,
        role_hint = role_hint,
        faction_hint = faction_hint,
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
            let rel = if state.faction.is_hostile_to(&s.faction) {
                "ENEMY"
            } else {
                "ally"
            };
            format!(
                "{} [{rel} HP:{}/{}] at ({:.0},{:.0}){}",
                s.name,
                s.hp,
                s.max_hp,
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
        "You are at ({:.0},{:.0}) HP:{}/{} XP:{} Lv:{}. Nearby: {}. What do you do?",
        state.position.x,
        state.position.y,
        state.hp,
        state.max_hp,
        state.xp,
        state.level,
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
    /// Shared circuit breaker — one per LLM backend, not per agent.
    circuit_breaker: Arc<CircuitBreaker>,
}

impl AgentDecisionLoop {
    pub fn new(
        spec: ActorSpec,
        handle: ActorHandle,
        cache: Arc<Mutex<SemanticCache>>,
        config: DecisionConfig,
        circuit_breaker: Arc<CircuitBreaker>,
    ) -> Self {
        Self {
            spec,
            handle,
            cache,
            config,
            circuit_breaker,
        }
    }

    /// Convenience constructor with a dedicated (non-shared) circuit breaker.
    pub fn new_with_default_breaker(
        spec: ActorSpec,
        handle: ActorHandle,
        cache: Arc<Mutex<SemanticCache>>,
        config: DecisionConfig,
    ) -> Self {
        let cb = Arc::new(CircuitBreaker::new(5, Duration::from_secs(30)));
        Self::new(spec, handle, cache, config, cb)
    }

    /// Run until the actor is shut down or absent from snapshot for too long.
    pub async fn run(self, world_snapshot: crate::global_agents::SharedSnapshot) {
        let mut interval = time::interval(self.config.decision_interval);
        interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);

        let sys = system_prompt(&self.spec);
        let my_id = self.spec.id;
        const MAX_MISSES: u32 = 10;
        let mut consecutive_misses: u32 = 0;

        loop {
            interval.tick().await;

            // Extract own state and nearby actors without holding the lock across await.
            let (state_opt, nearby) = {
                let snap = world_snapshot.lock().unwrap();
                let my_state = snap.get(&my_id).cloned();
                let nearby = if let Some(ref st) = my_state {
                    snap.values()
                        .filter(|s| s.id != my_id)
                        .filter(|s| {
                            let dx = s.position.x - st.position.x;
                            let dy = s.position.y - st.position.y;
                            (dx * dx + dy * dy).sqrt() < 500.0
                        })
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                } else {
                    vec![]
                };
                (my_state, nearby)
            };

            let state = match state_opt {
                Some(s) => {
                    consecutive_misses = 0;
                    s
                }
                None => {
                    consecutive_misses += 1;
                    if consecutive_misses >= MAX_MISSES {
                        debug!(actor = %my_id.0, "Actor absent from snapshot for {MAX_MISSES} cycles — stopping decision loop");
                        return;
                    }
                    continue;
                }
            };

            let user = user_prompt(&state, &nearby);

            let req = LlmRequest {
                model: self.spec.model.clone(),
                system_prompt: sys.clone(),
                user_prompt: user,
                max_tokens: self.config.max_tokens,
            };

            // Circuit breaker: skip LLM call when backend is unhealthy.
            if self.circuit_breaker.is_open() {
                debug!(
                    actor = %self.spec.id.0,
                    state = self.circuit_breaker.state_name(),
                    "Circuit open — skipping LLM call"
                );
                continue;
            }

            metrics::inc_llm_calls();
            let resp = {
                let mut cache = self.cache.lock().await;
                cache.complete(req).await
            };

            match resp {
                Ok(r) => {
                    self.circuit_breaker.record_success();
                    let action = parse_action(&r.text);
                    debug!(actor = %self.spec.id.0, ?action, "decision");
                    match action {
                        AgentAction::Move(pos) => {
                            self.handle.send(ActorMessage::Move { to: pos });
                        }
                        AgentAction::Speak(text) => {
                            self.handle.send(ActorMessage::Speak { text });
                        }
                        AgentAction::Attack(target_name) => {
                            // Look up the target actor by name in the snapshot.
                            let target_id = {
                                let snap = world_snapshot.lock().unwrap();
                                snap.values()
                                    .find(|s| {
                                        s.name.eq_ignore_ascii_case(&target_name) && s.id != my_id
                                    })
                                    .map(|s| s.id)
                            };
                            if let Some(tid) = target_id {
                                // Route attack via Interact effect → world engine resolves combat.
                                self.handle.send(ActorMessage::Interact {
                                    target: tid,
                                    action: "ATTACK".to_string(),
                                });
                            }
                        }
                        AgentAction::Idle => {}
                    }
                }
                Err(e) => {
                    metrics::inc_llm_errors();
                    self.circuit_breaker.record_failure();
                    warn!(actor = %self.spec.id.0, err = %e, "LLM request failed");
                }
            }
        }
    }
}

/// Priority scheduler: routes high-priority actors to faster execution slots.
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
    use crate::types::{ActorId, ActorRole, Faction, GridCell, LlmModel};

    fn make_spec() -> ActorSpec {
        ActorSpec {
            id: ActorId(1),
            name: "TestAgent".to_string(),
            personality: "curious".to_string(),
            backstory: "A traveler".to_string(),
            model: LlmModel::Mock,
            position: Position::new(50.0, 50.0),
            role: ActorRole::Knight,
            faction: Faction::Empire,
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
            role: ActorRole::Knight,
            faction: Faction::Empire,
            hp: 100,
            max_hp: 100,
            xp: 0,
            level: 1,
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
    fn parse_attack() {
        let action = parse_action("ATTACK Elena");
        matches!(action, AgentAction::Attack(n) if n == "Elena");
    }

    #[test]
    fn parse_idle_on_unknown() {
        let action = parse_action("DO NOTHING");
        matches!(action, AgentAction::Idle);
    }

    #[test]
    fn system_prompt_includes_name_and_faction() {
        let spec = make_spec();
        let prompt = system_prompt(&spec);
        assert!(prompt.contains("TestAgent"));
        assert!(prompt.contains("curious"));
        assert!(prompt.contains("Empire"));
        assert!(prompt.contains("Alliance"));
    }

    #[test]
    fn user_prompt_includes_position_and_hp() {
        let state = make_state(Position::new(100.0, 200.0));
        let prompt = user_prompt(&state, &[]);
        assert!(prompt.contains("100"));
        assert!(prompt.contains("200"));
        assert!(prompt.contains("nobody"));
        assert!(prompt.contains("HP:"));
    }

    #[test]
    fn user_prompt_labels_enemy() {
        let state = make_state(Position::new(0.0, 0.0));
        let enemy = ActorState {
            id: ActorId(2),
            name: "Enemy".to_string(),
            position: Position::new(10.0, 10.0),
            cell: GridCell(1, 1),
            tick: 0,
            last_utterance: None,
            role: ActorRole::Knight,
            faction: Faction::Alliance, // hostile to Empire
            hp: 80,
            max_hp: 100,
            xp: 0,
            level: 1,
        };
        let prompt = user_prompt(&state, &[enemy]);
        assert!(prompt.contains("ENEMY"));
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
                })
                .await
                .unwrap();
            }));
        }

        for t in tasks {
            t.await.unwrap();
        }

        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "Concurrency exceeded limit"
        );
    }
}

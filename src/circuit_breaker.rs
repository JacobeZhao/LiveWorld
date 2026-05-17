/// Half-open circuit breaker for LLM API calls.
///
/// States:
///   Closed   → normal operation; opens after `failure_threshold` consecutive errors
///   Open     → blocks all calls; transitions to HalfOpen after `reset_timeout`
///   HalfOpen → allows one probe call; success → Closed, failure → Open again
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::{info, warn};

#[derive(Debug)]
enum State {
    Closed { consecutive_failures: u32 },
    Open { opened_at: Instant },
    HalfOpen,
}

pub struct CircuitBreaker {
    state: Mutex<State>,
    failure_threshold: u32,
    reset_timeout: Duration,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            state: Mutex::new(State::Closed {
                consecutive_failures: 0,
            }),
            failure_threshold,
            reset_timeout,
        }
    }

    /// Returns true if the call should be rejected (circuit is open).
    /// Also transitions Open → HalfOpen when the reset timeout expires.
    pub fn is_open(&self) -> bool {
        let mut st = self.state.lock().unwrap();
        match *st {
            State::Open { opened_at } => {
                if opened_at.elapsed() >= self.reset_timeout {
                    *st = State::HalfOpen;
                    info!("Circuit breaker → HalfOpen (probe allowed)");
                    false
                } else {
                    true
                }
            }
            _ => false,
        }
    }

    pub fn record_success(&self) {
        let mut st = self.state.lock().unwrap();
        if !matches!(
            *st,
            State::Closed {
                consecutive_failures: 0
            }
        ) {
            info!("Circuit breaker → Closed");
        }
        *st = State::Closed {
            consecutive_failures: 0,
        };
    }

    pub fn record_failure(&self) {
        let mut st = self.state.lock().unwrap();
        match *st {
            State::Closed {
                consecutive_failures,
            } => {
                let n = consecutive_failures + 1;
                if n >= self.failure_threshold {
                    *st = State::Open {
                        opened_at: Instant::now(),
                    };
                    warn!(threshold = self.failure_threshold, "Circuit breaker → Open");
                } else {
                    *st = State::Closed {
                        consecutive_failures: n,
                    };
                }
            }
            State::HalfOpen => {
                *st = State::Open {
                    opened_at: Instant::now(),
                };
                warn!("Circuit breaker → Open (probe failed)");
            }
            State::Open { .. } => {}
        }
    }

    pub fn state_name(&self) -> &'static str {
        match *self.state.lock().unwrap() {
            State::Closed { .. } => "closed",
            State::Open { .. } => "open",
            State::HalfOpen => "half-open",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_after_threshold() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60));
        assert!(!cb.is_open());
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.is_open());
        cb.record_failure(); // 3rd failure → Open
        assert!(cb.is_open());
    }

    #[test]
    fn success_resets_to_closed() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure(); // Open
        assert!(cb.is_open());
        // Simulate reset timeout by directly writing HalfOpen
        *cb.state.lock().unwrap() = State::HalfOpen;
        assert!(!cb.is_open());
        cb.record_success();
        assert!(!cb.is_open());
        assert_eq!(cb.state_name(), "closed");
    }

    #[test]
    fn half_open_probe_failure_reopens() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(60));
        cb.record_failure(); // Open
        *cb.state.lock().unwrap() = State::HalfOpen; // simulate timeout
        cb.record_failure(); // probe fails → re-Open
        assert!(cb.is_open());
    }
}

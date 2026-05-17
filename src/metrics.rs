/// HTTP server (port 8081) serving:
///   GET  /          → static/index.html (frontend)
///   GET  /metrics   → Prometheus text format
///   POST /auth/token → issue a JWT (body: {"user_id":"…"})
///   anything else   → 404
use crate::ws_server::SharedEngine;
use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tracing::{error, info};

// ── Global counters (incremented by other modules) ────────────────────────────

pub static LLM_CALLS_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static LLM_ERRORS_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static WS_CONNECTIONS_ACTIVE: AtomicU64 = AtomicU64::new(0);

#[inline]
pub fn inc_llm_calls() {
    LLM_CALLS_TOTAL.fetch_add(1, Ordering::Relaxed);
}
#[inline]
pub fn inc_llm_errors() {
    LLM_ERRORS_TOTAL.fetch_add(1, Ordering::Relaxed);
}
#[inline]
pub fn inc_ws_connections() {
    WS_CONNECTIONS_ACTIVE.fetch_add(1, Ordering::Relaxed);
}
#[inline]
pub fn dec_ws_connections() {
    WS_CONNECTIONS_ACTIVE.fetch_sub(1, Ordering::Relaxed);
}

// ── HTTP server ───────────────────────────────────────────────────────────────

pub async fn run_http_server(engine: SharedEngine, port: u16) -> Result<()> {
    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    info!("HTTP server on port {port}  (metrics + frontend + /auth/token)");

    loop {
        match listener.accept().await {
            Ok((mut stream, _)) => {
                let engine = engine.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = match stream.read(&mut buf).await {
                        Ok(n) if n > 0 => n,
                        _ => return,
                    };
                    let req = std::str::from_utf8(&buf[..n]).unwrap_or("");
                    let first_line = req.lines().next().unwrap_or("");
                    let method = first_line.split_whitespace().next().unwrap_or("GET");
                    let path = first_line.split_whitespace().nth(1).unwrap_or("/");

                    // Body (for POST requests): text after the blank line
                    let body_str = req
                        .split("\r\n\r\n")
                        .nth(1)
                        .or_else(|| req.split("\n\n").nth(1))
                        .unwrap_or("")
                        .trim();

                    let (status, content_type, body) =
                        handle_request(method, path, body_str, &engine).await;

                    let response = format!(
                        "HTTP/1.1 {status}\r\n\
                         Content-Type: {content_type}\r\n\
                         Content-Length: {}\r\n\
                         Access-Control-Allow-Origin: *\r\n\
                         Connection: close\r\n\
                         \r\n\
                         {body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
            Err(e) => error!("HTTP accept error: {e}"),
        }
    }
}

pub async fn handle_request(
    method: &str,
    path: &str,
    body: &str,
    engine: &SharedEngine,
) -> (&'static str, &'static str, String) {
    match (method, path) {
        // ── Prometheus metrics ──────────────────────────────────────────────
        ("GET", "/metrics") => {
            let (actors, sessions, ticks, uptime) = {
                let eng = engine.lock().unwrap();
                (
                    eng.actor_count(),
                    eng.session_count(),
                    eng.tick_count(),
                    eng.uptime_secs(),
                )
            };
            let llm_calls = LLM_CALLS_TOTAL.load(Ordering::Relaxed);
            let llm_errors = LLM_ERRORS_TOTAL.load(Ordering::Relaxed);
            let ws_active = WS_CONNECTIONS_ACTIVE.load(Ordering::Relaxed);

            let body = format!(
                "# HELP liveworld_actors_total Active actor count\n\
                 # TYPE liveworld_actors_total gauge\n\
                 liveworld_actors_total {actors}\n\
                 # HELP liveworld_sessions_total Active session count\n\
                 # TYPE liveworld_sessions_total gauge\n\
                 liveworld_sessions_total {sessions}\n\
                 # HELP liveworld_ticks_total World tick counter\n\
                 # TYPE liveworld_ticks_total counter\n\
                 liveworld_ticks_total {ticks}\n\
                 # HELP liveworld_uptime_seconds Server uptime\n\
                 # TYPE liveworld_uptime_seconds gauge\n\
                 liveworld_uptime_seconds {uptime}\n\
                 # HELP liveworld_llm_calls_total Cumulative LLM calls\n\
                 # TYPE liveworld_llm_calls_total counter\n\
                 liveworld_llm_calls_total {llm_calls}\n\
                 # HELP liveworld_llm_errors_total Cumulative LLM errors\n\
                 # TYPE liveworld_llm_errors_total counter\n\
                 liveworld_llm_errors_total {llm_errors}\n\
                 # HELP liveworld_ws_connections_active Live WebSocket connections\n\
                 # TYPE liveworld_ws_connections_active gauge\n\
                 liveworld_ws_connections_active {ws_active}\n"
            );
            ("200 OK", "text/plain; version=0.0.4; charset=utf-8", body)
        }

        // ── JWT token issuance ──────────────────────────────────────────────
        ("POST", "/auth/token") => {
            let user_id = serde_json::from_str::<serde_json::Value>(body)
                .ok()
                .and_then(|v| v["user_id"].as_str().map(str::to_owned));

            match user_id {
                None => (
                    "400 Bad Request",
                    "application/json",
                    r#"{"error":"Missing user_id"}"#.to_string(),
                ),
                Some(uid) => match crate::auth::issue_token(&uid) {
                    Some(token) => (
                        "200 OK",
                        "application/json",
                        format!(r#"{{"token":"{token}"}}"#),
                    ),
                    None => (
                        "200 OK",
                        "application/json",
                        r#"{"message":"Auth disabled — no token required"}"#.to_string(),
                    ),
                },
            }
        }

        // ── JWT revocation ──────────────────────────────────────────────────
        ("POST", "/auth/revoke") => {
            let jti = serde_json::from_str::<serde_json::Value>(body)
                .ok()
                .and_then(|v| v["jti"].as_str().map(str::to_owned));
            match jti {
                None => (
                    "400 Bad Request",
                    "application/json",
                    r#"{"error":"Missing jti"}"#.to_string(),
                ),
                Some(j) => {
                    crate::auth::revoke_token(&j);
                    (
                        "200 OK",
                        "application/json",
                        r#"{"status":"revoked"}"#.to_string(),
                    )
                }
            }
        }

        // ── Health check (for k8s liveness/readiness probes) ───────────────
        ("GET", "/health") => {
            let (actors, sessions, uptime) = {
                let eng = engine.lock().unwrap();
                (eng.actor_count(), eng.session_count(), eng.uptime_secs())
            };
            let body = format!(
                r#"{{"status":"ok","actors":{actors},"sessions":{sessions},"uptime_secs":{uptime}}}"#
            );
            ("200 OK", "application/json", body)
        }

        // ── Frontend HTML ────────────────────────────────────────────────────
        ("GET", "/" | "/index.html") => {
            let html = tokio::fs::read_to_string("static/index.html")
                .await
                .unwrap_or_else(|_| {
                    "<h1>LiveWorld</h1><p>static/index.html not found.</p>".to_string()
                });
            ("200 OK", "text/html; charset=utf-8", html)
        }

        _ => ("404 Not Found", "text/plain", "Not Found".to_string()),
    }
}

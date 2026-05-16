use crate::ws_server::SharedEngine;
use anyhow::Result;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tracing::{error, info};

/// Minimal HTTP server on `port` that serves:
///   GET /metrics  → JSON metrics snapshot
///   GET /         → static index.html (from ./static/index.html)
///   anything else → 404
pub async fn run_http_server(engine: SharedEngine, port: u16) -> Result<()> {
    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    info!("HTTP server listening on port {port} (metrics + frontend)");

    loop {
        match listener.accept().await {
            Ok((mut stream, _)) => {
                let engine = engine.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let n = match stream.read(&mut buf).await {
                        Ok(n) => n,
                        Err(_) => return,
                    };
                    let request = std::str::from_utf8(&buf[..n]).unwrap_or("");
                    let path = request
                        .lines()
                        .next()
                        .and_then(|l| l.split_whitespace().nth(1))
                        .unwrap_or("/");

                    let (status, content_type, body) = match path {
                        "/metrics" => {
                            let (actors, sessions, ticks, uptime) = {
                                let eng = engine.lock().unwrap();
                                (
                                    eng.actor_count(),
                                    eng.session_count(),
                                    eng.tick_count(),
                                    eng.uptime_secs(),
                                )
                            };
                            let body = format!(
                                r#"{{"actors":{actors},"sessions":{sessions},"ticks":{ticks},"uptime_secs":{uptime}}}"#
                            );
                            ("200 OK", "application/json", body)
                        }
                        "/" | "/index.html" => {
                            let html = tokio::fs::read_to_string("static/index.html")
                                .await
                                .unwrap_or_else(|_| {
                                    "<h1>LiveWorld</h1><p>static/index.html not found</p>"
                                        .to_string()
                                });
                            ("200 OK", "text/html; charset=utf-8", html)
                        }
                        _ => ("404 Not Found", "text/plain", "Not Found".to_string()),
                    };

                    let response = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
            Err(e) => error!("HTTP accept error: {e}"),
        }
    }
}

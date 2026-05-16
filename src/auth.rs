/// Token-based auth helper.
///
/// If `LIVEWORLD_TOKEN` env var is set, every WebSocket connection must
/// supply a matching token in its first message: `{"token":"<value>"}`.
/// If the env var is absent, auth is skipped (dev/local mode).
pub fn validate_token(provided: Option<&str>) -> bool {
    match std::env::var("LIVEWORLD_TOKEN") {
        Err(_) => true, // no token configured → dev mode, allow all
        Ok(expected) => provided.map(|t| t == expected.as_str()).unwrap_or(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_env_var_allows_all() {
        std::env::remove_var("LIVEWORLD_TOKEN");
        assert!(validate_token(None));
        assert!(validate_token(Some("anything")));
    }
}

/// JWT-based authentication.
///
/// If `JWT_SECRET` env var is set:
///   - WebSocket clients must send `{"token":"<JWT>"}` as their first message.
///   - HTTP `POST /auth/token` with `{"user_id":"<id>"}` issues a fresh token.
///   - Tokens are HS256-signed, expire after `JWT_TTL_SECS` (default 24 h).
///
/// If `JWT_SECRET` is NOT set → dev/open mode, all connections accepted.
use crate::jwt;

const DEFAULT_TTL_SECS: u64 = 86_400; // 24 h

/// Validate a JWT string.  Returns `true` if valid or if auth is disabled.
pub fn validate_token(token: Option<&str>) -> bool {
    validate_with_secret(token, jwt_secret().as_deref())
}

/// Issue a signed JWT for `user_id`.  Returns `None` when auth is disabled.
pub fn issue_token(user_id: &str) -> Option<String> {
    let secret = jwt_secret()?;
    let ttl = std::env::var("JWT_TTL_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TTL_SECS);
    jwt::issue(user_id, secret.as_bytes(), ttl).ok()
}

// ── Internal helpers (testable without env var mutations) ─────────────────────

pub(crate) fn validate_with_secret(token: Option<&str>, secret: Option<&str>) -> bool {
    match secret {
        None => true,
        Some(s) => token
            .map(|t| jwt::validate(t, s.as_bytes()).is_ok())
            .unwrap_or(false),
    }
}

fn jwt_secret() -> Option<String> {
    std::env::var("JWT_SECRET").ok().filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &str = "test-secret-key-at-least-32-bytes!";

    #[test]
    fn no_secret_allows_all() {
        assert!(validate_with_secret(None, None));
        assert!(validate_with_secret(Some("anything"), None));
    }

    #[test]
    fn valid_token_accepted() {
        let token = crate::jwt::issue("user1", SECRET.as_bytes(), 3600).unwrap();
        assert!(validate_with_secret(Some(&token), Some(SECRET)));
    }

    #[test]
    fn missing_token_rejected_when_secret_set() {
        assert!(!validate_with_secret(None, Some(SECRET)));
    }

    #[test]
    fn bad_token_rejected() {
        assert!(!validate_with_secret(Some("not.a.jwt"), Some(SECRET)));
    }
}

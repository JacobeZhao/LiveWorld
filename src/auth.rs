/// JWT-based authentication.
///
/// If `JWT_SECRET` env var is set:
///   - WebSocket clients must send `{"token":"<JWT>"}` as their first message.
///   - HTTP `POST /auth/token` with `{"user_id":"<id>"}` issues a fresh token.
///   - HTTP `POST /auth/revoke` with `{"jti":"<id>"}` adds the jti to the revocation set.
///   - Tokens are HS256-signed, expire after `JWT_TTL_SECS` (default 24 h).
///
/// If `JWT_SECRET` is NOT set → dev/open mode, all connections accepted.
use crate::jwt;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

const DEFAULT_TTL_SECS: u64 = 86_400; // 24 h

// ── In-process token revocation set ──────────────────────────────────────────

static REVOKED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

fn revoked() -> &'static Mutex<HashSet<String>> {
    REVOKED.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Add a JWT ID to the revocation set. Revoked tokens are rejected even if
/// the signature and expiry are valid.
pub fn revoke_token(jti: &str) {
    revoked().lock().unwrap().insert(jti.to_string());
}

// ── Public API ────────────────────────────────────────────────────────────────

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
            .and_then(|t| jwt::validate(t, s.as_bytes()).ok())
            .map(|claims| !revoked().lock().unwrap().contains(&claims.jti))
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

    #[test]
    fn revoked_token_rejected() {
        let token = crate::jwt::issue("revokeuser", SECRET.as_bytes(), 3600).unwrap();
        let claims = crate::jwt::validate(&token, SECRET.as_bytes()).unwrap();
        revoke_token(&claims.jti);
        assert!(!validate_with_secret(Some(&token), Some(SECRET)));
    }
}

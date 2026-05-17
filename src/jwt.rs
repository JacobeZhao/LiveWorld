/// Minimal HS256 JWT implementation (pure Rust, no ring dependency).
/// Uses hmac + sha2 + base64 from the RustCrypto family.
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const HEADER: &str = r#"{"alg":"HS256","typ":"JWT"}"#;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID or session identifier).
    pub sub: String,
    /// Issued-at (Unix seconds).
    pub iat: u64,
    /// Expiry (Unix seconds).
    pub exp: u64,
    /// JWT ID — unique per token, used for revocation.
    pub jti: String,
}

/// Issue a signed HS256 JWT valid for `ttl_secs` seconds.
pub fn issue(sub: &str, secret: &[u8], ttl_secs: u64) -> anyhow::Result<String> {
    let now = now_secs();
    let jti = format!("{sub}-{now}");
    let claims = Claims { sub: sub.to_owned(), iat: now, exp: now + ttl_secs, jti };
    let header_b64 = URL_SAFE_NO_PAD.encode(HEADER);
    let payload_b64 = URL_SAFE_NO_PAD.encode(serde_json::to_string(&claims)?);
    let signing_input = format!("{header_b64}.{payload_b64}");
    let sig = sign(&signing_input, secret)?;
    Ok(format!("{signing_input}.{sig}"))
}

/// Validate a JWT and return its claims.  Fails on bad signature or expiry.
pub fn validate(token: &str, secret: &[u8]) -> anyhow::Result<Claims> {
    let parts: Vec<&str> = token.splitn(3, '.').collect();
    anyhow::ensure!(parts.len() == 3, "Malformed JWT");

    let signing_input = format!("{}.{}", parts[0], parts[1]);
    let expected = sign(&signing_input, secret)?;
    anyhow::ensure!(
        constant_time_eq(parts[2].as_bytes(), expected.as_bytes()),
        "Invalid JWT signature"
    );

    let payload = URL_SAFE_NO_PAD.decode(parts[1])?;
    let claims: Claims = serde_json::from_slice(&payload)?;
    anyhow::ensure!(claims.exp >= now_secs(), "JWT expired");
    Ok(claims)
}

fn sign(data: &str, secret: &[u8]) -> anyhow::Result<String> {
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|e| anyhow::anyhow!("HMAC key error: {e}"))?;
    mac.update(data.as_bytes());
    Ok(URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes()))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Constant-time byte comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &[u8] = b"super-secret-test-key-32-bytes!!";

    #[test]
    fn issue_and_validate_roundtrip() {
        let token = issue("user42", SECRET, 3600).unwrap();
        let claims = validate(&token, SECRET).unwrap();
        assert_eq!(claims.sub, "user42");
    }

    #[test]
    fn wrong_secret_rejected() {
        let token = issue("u", SECRET, 3600).unwrap();
        assert!(validate(&token, b"wrong-secret").is_err());
    }

    #[test]
    fn expired_token_rejected() {
        let token = issue("u", SECRET, 0).unwrap(); // expires immediately (ttl=0)
        // Give it 1 second to expire
        std::thread::sleep(std::time::Duration::from_secs(1));
        assert!(validate(&token, SECRET).is_err());
    }

    #[test]
    fn tampered_payload_rejected() {
        let mut token = issue("u", SECRET, 3600).unwrap();
        // flip one char in the payload section
        let idx = token.find('.').unwrap() + 5;
        unsafe {
            let b = token.as_bytes_mut();
            b[idx] ^= 0x01;
        }
        assert!(validate(&token, SECRET).is_err());
    }
}

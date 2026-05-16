# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM rust:1.83-slim-bookworm AS builder

# Install OpenSSL headers (required by reqwest/native-tls on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache dependencies first (layer caching)
COPY Cargo.toml Cargo.lock ./
# Dummy source to compile deps without the real src
RUN mkdir src && echo 'fn main(){}' > src/main.rs && \
    echo 'pub fn placeholder(){}' > src/lib.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Build actual source
COPY src ./src
COPY benches ./benches
COPY static ./static
RUN cargo build --release --bin liveworld

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/liveworld /app/liveworld
COPY --from=builder /build/static /app/static

# Writable data directory for snapshots
RUN mkdir -p /app/data/snapshots

EXPOSE 8080
EXPOSE 8081

ENV RUST_LOG=info

ENTRYPOINT ["/app/liveworld"]

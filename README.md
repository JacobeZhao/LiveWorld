---
title: LiveWorld
emoji: ⚔
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8080
pinned: false
---

# LiveWorld

**百万人在线的实时多智能体虚拟世界引擎**

高并发、低延迟的实时平台：每个用户可创建自主 AI Agent，世界以 25 Hz 频率同步所有角色状态。

## 性能指标（Windows 11, x86_64, 非隔离核心）

| 指标 | 测量值 | 目标 |
|------|--------|------|
| Actor 间消息 P99 | **100 ns** | ≤ 200 ns ✓ |
| Actor 间消息 P99.9 | **100 ns** | ≤ 500 ns ✓ |
| 世界广播 P99 (1000 会话) | **1.46 ms** | ≤ 5 ms ✓ |
| 语义缓存命中延迟 | **< 1 ms** | ≤ 1 ms ✓ |
| 快照恢复 (10 000 Actor) | **< 500 ms** | ≤ 2 s ✓ |
| 10M 消息吞吐 | **~925 M/s** | — |

## 架构

```
WS Gateway (Tokio + tungstenite)
    │  JWT 认证 / per-IP 连接限制 / 速率限制 (20 cmd/s)
    │
World Engine (25 Hz tick loop, dedicated thread)
    ├── Actor Runtime (lock-free SPSC, 无锁消息队列)
    ├── Spatial Grid (O(1) 空间查询)
    ├── Interest Manager (兴趣过滤，视野半径)
    └── State Encoder (增量帧编码)
    │
Intelligence Layer (async Tokio tasks, per-actor)
    ├── AgentDecisionLoop (每 actor 独立决策循环，3s 间隔)
    ├── LLM Adapter (OpenAI / Anthropic / Ollama / Mock)
    ├── Semantic Cache (LRU + hash，零重复调用，per-model)
    ├── Priority Scheduler (并发限流)
    └── Circuit Breaker (半开路断路器，5 次失败触发)
    │
Global Agents (异步，不阻塞 tick)
    ├── DirectorAgent (叙事事件，10s 间隔)
    ├── EconomyAgent (经济平衡，5s 间隔)
    └── AntiCheatAgent (速度异常检测，500ms 间隔)
    │
Persistence (周期快照写入磁盘 + 冷启动恢复)
    │
Cross-pod Sync (Redis pub/sub，可选)
```

## 快速开始

### 环境要求

- Rust 1.75+（GNU 工具链）
- MinGW（Windows 必须）

```powershell
# Windows: 安装 MinGW（一次性操作）
scoop install mingw

# 将 MinGW 加入 PATH（每次会话或写入 Profile）
$env:PATH = "$env:USERPROFILE\scoop\apps\mingw\current\bin;" + $env:PATH
```

### 构建与测试

```powershell
# 运行全部单元测试（88 个）
cargo test

# 运行性能基准测试
cargo bench --bench actor_ipc
cargo bench --bench broadcast

# 启动服务器
cargo run --release
```

### 配置

```powershell
# 复制示例配置
cp .env.example .env
# 编辑 .env，填写 JWT_SECRET、LLM API Key 等
```

### 配置 JWT 认证

```bash
# 启用 JWT（留空则为开放模式）
export JWT_SECRET=your-secret-at-least-32-chars

# 获取 token
curl -X POST http://localhost:8081/auth/token -d '{"user_id":"alice"}'
# → {"token":"eyJ..."}

# 连接 WebSocket（认证启用时，首条消息必须是 token）
# {"token":"eyJ..."}
```

### 配置 LLM

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# 本地 Ollama
export OLLAMA_URL=http://localhost:11434

# 运行真实 LLM 集成测试（默认 #[ignore]）
cargo test -- --ignored
```

### 水平分片（单 pod 内）

```bash
# 将 x 轴切分为 4 个 WorldEngine 实例
export SHARD_COUNT=4
cargo run --release
```

### 跨 pod 状态同步（多节点）

```bash
# 启动 Redis
docker run -d -p 6379:6379 redis:7

# 每个 pod 配置
export REDIS_URL=redis://localhost:6379
cargo run --release
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `types.rs` | 核心数据类型 (ActorId, Position, StateDelta, WorldDirective, ...) |
| `spsc_queue.rs` | 无锁 SPSC 环形队列，Actor 间通信热路径 |
| `spatial_grid.rs` | 世界格网，O(1) 空间插入/移动/查询 |
| `actor.rs` | 单 Actor 结构与状态机 |
| `actor_runtime.rs` | Actor 池，Tick 内批量消息处理 |
| `interest_manager.rs` | 按视野半径过滤可见 Actor |
| `state_encoder.rs` | 增量帧 bincode 编码 |
| `llm_adapter.rs` | 统一 LLM 接口（Mock/OpenAI/Anthropic/Ollama）+ 工厂函数 |
| `semantic_cache.rs` | LRU 语义缓存，重复 Prompt 直接返回 |
| `circuit_breaker.rs` | 半开路断路器，LLM 后端故障保护 |
| `agent_decision.rs` | 每 Actor 异步决策循环 + 优先级调度器 |
| `engine_api.rs` | `EngineApi` trait，多态引擎分发 |
| `world_engine.rs` | 25 Hz Tick 主循环，兴趣过滤，Session 管理 |
| `shard.rs` | `ShardedEngine`：按 x 轴分片到 N 个 WorldEngine |
| `persistence.rs` | 周期快照写入 + 冷启动恢复 |
| `global_agents.rs` | Director / Economy / AntiCheat 全局 Agent |
| `jwt.rs` | HS256 JWT 签发与验证（纯 Rust） |
| `auth.rs` | JWT 认证中间件 |
| `metrics.rs` | Prometheus 指标 + `/health` + `/auth/token` HTTP 服务 |
| `redis_sync.rs` | Redis pub/sub 跨 pod 状态同步 |
| `ws_server.rs` | WebSocket 接入，客户端命令路由，per-IP 连接限制 |

## HTTP 端点（端口 8081）

| 路径 | 说明 |
|------|------|
| `GET /health` | k8s liveness/readiness 探针 → `{"status":"ok"}` |
| `GET /metrics` | Prometheus 文本格式（7 项指标） |
| `POST /auth/token` | 签发 JWT，body: `{"user_id":"alice"}` |
| `GET /` | 前端 HTML5 Canvas 客户端 |

## Kubernetes 部署

```bash
# 应用所有资源（StatefulSet + Services + HPA + PDB + Ingress）
kubectl apply -f k8s/

# 查看 pod 状态
kubectl -n liveworld get pods

# 配置密钥（修改 k8s/secret.yaml 后 apply）
kubectl -n liveworld apply -f k8s/secret.yaml
```

StatefulSet 保证每个 pod 有独立的 PVC（`ReadWriteOnce`），无需共享存储。配合 Redis 实现跨 pod actor 可见性。

## CI

GitHub Actions（`.github/workflows/ci.yml`）：

- `cargo fmt --check`
- `cargo clippy -- -D warnings`
- `cargo test`
- Docker 镜像构建

## 许可证

MIT

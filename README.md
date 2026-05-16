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
    │
World Engine (25 Hz tick loop, dedicated thread)
    ├── Actor Runtime (lock-free SPSC, 无锁消息队列)
    ├── Spatial Grid (O(1) 空间查询)
    ├── Interest Manager (兴趣过滤)
    └── State Encoder (增量帧编码)
    │
Intelligence Layer (async Tokio tasks)
    ├── LLM Adapter (OpenAI / Anthropic / Ollama / Mock)
    ├── Semantic Cache (LRU + hash, 零重复调用)
    └── Priority Scheduler (并发限流)
    │
Global Agents (异步，不阻塞 tick)
    ├── DirectorAgent (叙事事件)
    ├── EconomyAgent (经济平衡)
    └── AntiCheatAgent (速度异常检测)
    │
Persistence (60s 快照，< 2s 恢复)
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
# 运行全部单元测试（68 个）
cargo test

# 运行性能基准测试
cargo bench --bench actor_ipc
cargo bench --bench broadcast

# 启动服务器
cargo run --release
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

## 模块说明

| 模块 | 职责 |
|------|------|
| `types.rs` | 核心数据类型 (ActorId, Position, StateDelta, ...) |
| `spsc_queue.rs` | 无锁 SPSC 环形队列，Actor 间通信热路径 |
| `spatial_grid.rs` | 世界格网，O(1) 空间插入/移动/查询 |
| `actor.rs` | 单 Actor 结构与状态机 |
| `interest_manager.rs` | 按视野半径过滤可见 Actor |
| `actor_runtime.rs` | Actor 池，Tick 内批量消息处理 |
| `state_encoder.rs` | 增量帧 bincode 编码 |
| `llm_adapter.rs` | 统一 LLM 接口（Mock/OpenAI/Anthropic/Ollama）|
| `semantic_cache.rs` | LRU 语义缓存，重复 Prompt 直接返回 |
| `agent_decision.rs` | 每 Agent 异步决策循环 + 优先级调度器 |
| `world_engine.rs` | 25 Hz Tick 主循环，兴趣过滤，Session 管理 |
| `persistence.rs` | 周期快照写入 + 冷启动恢复 |
| `global_agents.rs` | Director / Economy / AntiCheat 全局 Agent |
| `ws_server.rs` | WebSocket 接入，客户端命令路由 |

## 扩展到百万用户

当前单节点目标：10 000+ 并发 Actor。

扩展到 1M+ 用户的路径：
1. **地理分片**：按世界坐标 Hash 将 Actor 分配到不同节点
2. **节点间协议**：gRPC 跨节点消息路由（接口桩已存在）
3. **接入层**：Nginx/HAProxy 或自研网关做 WS 负载均衡
4. **状态同步**：Redis Cluster 做跨节点 Actor 位置索引

## 许可证

MIT

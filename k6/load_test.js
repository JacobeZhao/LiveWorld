/**
 * k6 WebSocket load test for LiveWorld.
 *
 * Usage:
 *   k6 run k6/load_test.js
 *   k6 run --vus 100 --duration 60s k6/load_test.js
 *   WS_URL=ws://prod.example.com:8080 k6 run k6/load_test.js
 *
 * Targets:
 *   - P99 world broadcast ≤ 5 ms at 1000 sessions
 *   - Zero error rate under sustained 500 VU load
 */

import ws from "k6/ws";
import { check, sleep } from "k6";
import { Counter, Trend, Rate } from "k6/metrics";

const wsUrl = __ENV.WS_URL || "ws://127.0.0.1:8080";

// Custom metrics
const createLatency = new Trend("actor_create_latency_ms", true);
const deltaLatency  = new Trend("world_delta_latency_ms", true);
const errorRate     = new Rate("ws_error_rate");
const msgCount      = new Counter("ws_messages_total");

export const options = {
  stages: [
    { duration: "10s", target: 100  },  // ramp up
    { duration: "30s", target: 500  },  // sustain
    { duration: "20s", target: 1000 },  // peak load
    { duration: "10s", target: 0    },  // ramp down
  ],
  thresholds: {
    actor_create_latency_ms: ["p(99)<500"],   // actor creation P99 < 500 ms
    world_delta_latency_ms:  ["p(99)<10"],    // delta delivery P99 < 10 ms
    ws_error_rate:           ["rate<0.01"],   // error rate < 1%
    ws_messages_total:       ["count>0"],
  },
};

export default function () {
  const startMs = Date.now();

  const res = ws.connect(wsUrl, {}, function (socket) {
    let actorCreated = false;
    let createSentAt = 0;

    socket.on("open", function () {
      // Create actor immediately on connect.
      createSentAt = Date.now();
      socket.send(
        JSON.stringify({
          type: "CreateActor",
          name: `Agent-${__VU}-${__ITER}`,
          personality: "curious explorer",
          backstory: "A traveler from a distant land",
          model: "Mock",
          position: {
            x: Math.random() * 9000,
            y: Math.random() * 9000,
          },
        })
      );
    });

    socket.on("message", function (data) {
      msgCount.add(1);
      let msg;
      try {
        msg = JSON.parse(data);
      } catch {
        errorRate.add(1);
        return;
      }

      if (msg.type === "ActorCreated") {
        createLatency.add(Date.now() - createSentAt);
        actorCreated = true;

        // Begin moving every 2 seconds.
        socket.setInterval(function () {
          if (!actorCreated) return;
          socket.send(
            JSON.stringify({
              type: "MoveActor",
              actor_id: msg.actor_id,
              to: {
                x: Math.random() * 9000,
                y: Math.random() * 9000,
              },
            })
          );
        }, 2000);
      } else if (msg.type === "WorldDelta") {
        const nowMs = Date.now();
        if (msg.timestamp_ms) {
          deltaLatency.add(nowMs - msg.timestamp_ms);
        }
      } else if (msg.type === "Error") {
        // 429 under load is expected; everything else is an error.
        if (msg.code !== 429) {
          errorRate.add(1);
        }
      }
    });

    socket.on("error", function (e) {
      errorRate.add(1);
    });

    // Hold connection for 30 seconds.
    sleep(30);
    socket.close();
  });

  check(res, { "WS connected": (r) => r && r.status === 101 });
}

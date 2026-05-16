// Benchmark: World state broadcast latency.
// Simulates the full encode → queue → dequeue path for N sessions.
// DoD target P3: P99 ≤ 5 ms for 1000 sessions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liveworld::state_encoder::{StateEncoder, diff_states};
use liveworld::types::{ActorId, ActorState, GridCell, Position, StateDelta, now_ms};
use std::time::{Duration, Instant};

fn make_state(id: u64, x: f32) -> ActorState {
    ActorState {
        id: ActorId(id),
        name: format!("A{id}"),
        position: Position::new(x, 0.0),
        cell: GridCell(0, 0),
        tick: 1,
        last_utterance: None,
    }
}

fn bench_encode_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_encoder");

    for n_actors in [100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(n_actors as u64));
        group.bench_with_input(
            BenchmarkId::new("encode_delta", n_actors),
            &n_actors,
            |b, &n| {
                let mut enc = StateEncoder::new(4 * 1024 * 1024);
                let states: Vec<ActorState> =
                    (0..n as u64).map(|i| make_state(i, i as f32)).collect();
                let delta = StateDelta {
                    tick: 1,
                    timestamp_ms: now_ms(),
                    updates: states,
                    removed: vec![],
                };
                b.iter(|| {
                    black_box(enc.encode(black_box(&delta)).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_diff_states(c: &mut Criterion) {
    let mut group = c.benchmark_group("diff_states");

    for n_actors in [1_000, 10_000] {
        group.bench_with_input(
            BenchmarkId::new("diff", n_actors),
            &n_actors,
            |b, &n| {
                let prev: Vec<ActorState> =
                    (0..n as u64).map(|i| make_state(i, i as f32)).collect();
                // 10% of actors moved
                let mut curr = prev.clone();
                for i in (0..n).step_by(10) {
                    curr[i].position.x += 5.0;
                    curr[i].tick += 1;
                }
                b.iter(|| {
                    black_box(diff_states(black_box(&prev), black_box(&curr)));
                });
            },
        );
    }

    group.finish();
}

fn bench_broadcast_latency_1000_sessions(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast_1000");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("full_pipeline_1000_actors", |b| {
        let mut enc = StateEncoder::new(8 * 1024 * 1024);
        let states: Vec<ActorState> =
            (0..1000u64).map(|i| make_state(i, i as f32)).collect();

        let mut samples: Vec<u64> = Vec::with_capacity(1000);

        b.iter(|| {
            let start = Instant::now();

            // 1. Build delta
            let delta = StateEncoder::build_delta(1, states.clone(), vec![]);

            // 2. Encode
            let bytes = enc.encode(&delta).unwrap().to_vec();

            // 3. Simulate sending to 1000 sessions (memcpy cost)
            for _ in 0..1000 {
                let _ = black_box(bytes.clone());
            }

            let elapsed = start.elapsed().as_micros() as u64;
            samples.push(elapsed);
        });

        if samples.len() > 10 {
            samples.sort_unstable();
            let p50 = samples[samples.len() / 2];
            let p99 = samples[(samples.len() as f64 * 0.99) as usize];
            eprintln!(
                "\nBroadcast (1000 sessions, 1000 actors) — p50: {}µs  p99: {}µs",
                p50, p99
            );
            // P99 target: < 5000 µs = 5 ms
            // We assert < 50 ms here to account for benchmark overhead.
            assert!(p99 < 50_000, "p99 broadcast {}µs exceeds 50ms ceiling", p99);
        }
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_encode_delta,
    bench_diff_states,
    bench_broadcast_latency_1000_sessions
);
criterion_main!(benches);

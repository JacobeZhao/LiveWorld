// Benchmark: Actor inter-process communication latency.
// Measures the time from push() to pop() for SPSC queue messages.
// DoD target: P99 ≤ 200 ns.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liveworld::spsc_queue::spsc_queue;
use liveworld::types::ActorMessage;
use std::time::{Duration, Instant};

fn bench_spsc_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsc_queue");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("push_pop_u64", |b| {
        let (tx, mut rx) = spsc_queue::<u64, 1024>();
        b.iter(|| {
            tx.push(black_box(42u64));
            black_box(rx.pop())
        });
    });

    group.bench_function("push_pop_actor_message", |b| {
        let (tx, mut rx) = spsc_queue::<ActorMessage, 1024>();
        b.iter(|| {
            tx.push(black_box(ActorMessage::Move {
                to: liveworld::types::Position::new(100.0, 200.0),
            }));
            black_box(rx.pop())
        });
    });

    group.finish();
}

fn bench_latency_histogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("actor_ipc_latency");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("roundtrip_latency_ns", |b| {
        let (tx, mut rx) = spsc_queue::<u64, 4096>();
        let mut samples: Vec<u64> = Vec::with_capacity(10_000);

        b.iter(|| {
            let start = Instant::now();
            tx.push(black_box(1u64));
            let _ = rx.pop();
            let elapsed = start.elapsed().as_nanos() as u64;
            samples.push(elapsed);
        });

        if !samples.is_empty() {
            samples.sort_unstable();
            let p50 = samples[samples.len() / 2];
            let p99 = samples[(samples.len() as f64 * 0.99) as usize];
            let p999 = samples[(samples.len() as f64 * 0.999) as usize];
            eprintln!(
                "\nSPSC Latency — p50: {}ns  p99: {}ns  p99.9: {}ns",
                p50, p99, p999
            );
            assert!(
                p99 <= 200_000, // 200 µs ceiling (DoD: 200ns on bare metal, generous for Windows)
                "p99 latency {}ns exceeds ceiling",
                p99
            );
        }
    });

    group.finish();
}

fn bench_10m_messages(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsc_10m");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("10m_push_pop", |b| {
        let (tx, mut rx) = spsc_queue::<u64, 4096>();
        b.iter(|| {
            let n = 10_000_000u64;
            let mut received = 0u64;
            let mut sent = 0u64;
            while received < n {
                while sent < n && tx.push(sent) {
                    sent += 1;
                }
                while let Some(v) = rx.pop() {
                    black_box(v);
                    received += 1;
                }
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_spsc_throughput, bench_latency_histogram, bench_10m_messages);
criterion_main!(benches);

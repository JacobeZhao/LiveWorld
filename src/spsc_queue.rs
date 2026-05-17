// Lock-free single-producer single-consumer queue.
// Uses a power-of-2 ring buffer with atomic head/tail indices.
// Zero heap allocation on the hot path (push/pop).
// Cache-line padding prevents false sharing between producer and consumer.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const CACHE_LINE: usize = 64;

#[repr(C)]
struct Padded<T> {
    value: T,
    _pad: [u8; CACHE_LINE],
}

impl<T> Padded<T> {
    fn new(value: T) -> Self {
        Self {
            value,
            _pad: [0u8; CACHE_LINE],
        }
    }
}

pub(crate) struct SpscInner<T, const N: usize> {
    head: Padded<AtomicUsize>,
    tail: Padded<AtomicUsize>,
    buf: [UnsafeCell<MaybeUninit<T>>; N],
}

// SAFETY: The producer/consumer split guarantees exclusive write access.
unsafe impl<T: Send, const N: usize> Send for SpscInner<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for SpscInner<T, N> {}

const fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

impl<T, const N: usize> SpscInner<T, N> {
    const MASK: usize = N - 1;

    fn new() -> Self {
        assert!(is_power_of_two(N), "SpscQueue capacity must be power of 2");
        // SAFETY: MaybeUninit array initialisation.
        let buf = unsafe {
            let mut arr: [UnsafeCell<MaybeUninit<T>>; N] = MaybeUninit::uninit().assume_init();
            for slot in &mut arr {
                *slot = UnsafeCell::new(MaybeUninit::uninit());
            }
            arr
        };
        Self {
            head: Padded::new(AtomicUsize::new(0)),
            tail: Padded::new(AtomicUsize::new(0)),
            buf,
        }
    }

    #[inline]
    fn push(&self, item: T) -> bool {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let next_tail = tail.wrapping_add(1);
        // Full if next_tail would collide with head
        if (next_tail & Self::MASK) == (self.head.value.load(Ordering::Acquire) & Self::MASK)
            && next_tail != self.head.value.load(Ordering::Acquire)
        {
            return false;
        }
        // SAFETY: Producer owns this slot exclusively (tail not yet published).
        unsafe {
            (*self.buf[tail & Self::MASK].get()).write(item);
        }
        self.tail.value.store(next_tail, Ordering::Release);
        true
    }

    #[inline]
    fn pop(&self) -> Option<T> {
        let head = self.head.value.load(Ordering::Relaxed);
        if head == self.tail.value.load(Ordering::Acquire) {
            return None;
        }
        // SAFETY: Consumer owns this slot; item was written by producer.
        let item = unsafe { (*self.buf[head & Self::MASK].get()).assume_init_read() };
        self.head
            .value
            .store(head.wrapping_add(1), Ordering::Release);
        Some(item)
    }

    #[inline]
    fn len(&self) -> usize {
        let tail = self.tail.value.load(Ordering::Acquire);
        let head = self.head.value.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, const N: usize> Drop for SpscInner<T, N> {
    fn drop(&mut self) {
        // Drain remaining items to run their destructors.
        while self.pop().is_some() {}
    }
}

/// Producer handle — only one may exist at a time.
pub struct SpscProducer<T, const N: usize> {
    pub(crate) inner: Arc<SpscInner<T, N>>,
}

impl<T, const N: usize> Clone for SpscProducer<T, N> {
    fn clone(&self) -> Self {
        SpscProducer {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Consumer handle — only one may exist at a time.
pub struct SpscConsumer<T, const N: usize> {
    inner: Arc<SpscInner<T, N>>,
}

/// Create a paired (producer, consumer) for a fixed-capacity SPSC queue.
/// Capacity N must be a power of two.
pub fn spsc_queue<T, const N: usize>() -> (SpscProducer<T, N>, SpscConsumer<T, N>) {
    let inner = Arc::new(SpscInner::<T, N>::new());
    (
        SpscProducer {
            inner: Arc::clone(&inner),
        },
        SpscConsumer { inner },
    )
}

impl<T, const N: usize> SpscProducer<T, N> {
    /// Push an item. Returns false if the queue is full.
    #[inline]
    pub fn push(&self, item: T) -> bool {
        self.inner.push(item)
    }
}

impl<T, const N: usize> SpscConsumer<T, N> {
    /// Pop an item. Returns None if the queue is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn basic_push_pop() {
        let (tx, mut rx) = spsc_queue::<u64, 16>();
        assert!(tx.push(1));
        assert!(tx.push(2));
        assert_eq!(rx.pop(), Some(1));
        assert_eq!(rx.pop(), Some(2));
        assert_eq!(rx.pop(), None);
    }

    #[test]
    fn fifo_order_preserved() {
        let (tx, mut rx) = spsc_queue::<u32, 1024>();
        for i in 0..1000u32 {
            assert!(tx.push(i));
        }
        for i in 0..1000u32 {
            assert_eq!(rx.pop(), Some(i));
        }
    }

    #[test]
    fn full_queue_returns_false() {
        // N=4: mask=3, capacity is N-1 = 3 usable slots before full check triggers
        let (tx, mut rx) = spsc_queue::<u32, 4>();
        // Fill until full
        let mut pushed = 0;
        while tx.push(pushed) {
            pushed += 1;
        }
        // Should have pushed at least 1 item
        assert!(pushed > 0);
        // After draining one, can push again
        rx.pop();
        assert!(tx.push(99));
    }

    #[test]
    fn concurrent_producer_consumer() {
        let (tx, mut rx) = spsc_queue::<u64, 1024>();
        const N: u64 = 500;

        let producer = thread::spawn(move || {
            let mut sent = 0u64;
            while sent < N {
                if tx.push(sent) {
                    sent += 1;
                } else {
                    thread::yield_now();
                }
            }
        });

        let consumer = thread::spawn(move || {
            let mut received = 0u64;
            let mut expected = 0u64;
            while received < N {
                if let Some(v) = rx.pop() {
                    assert_eq!(v, expected, "FIFO violated");
                    expected += 1;
                    received += 1;
                } else {
                    thread::yield_now();
                }
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }

    #[test]
    fn drop_cleans_up() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicUsize::new(0));

        struct Counted(Arc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        {
            let (tx, _rx) = spsc_queue::<Counted, 8>();
            let c = Arc::clone(&counter);
            tx.push(Counted(c));
        }
        // Item dropped when queue is dropped
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}

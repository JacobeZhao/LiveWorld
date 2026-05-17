// Semantic cache for LLM responses.
// Key = hash(system_prompt || user_prompt || model).
// Eviction policy: LRU with a fixed capacity.
// Metrics: hit/miss counters observable via stats().

use crate::llm_adapter::{LlmAdapter, LlmRequest, LlmResponse};
use ahash::AHashMap;
use anyhow::Result;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

fn hash_request(req: &LlmRequest) -> u64 {
    let mut hasher = ahash::AHasher::default();
    req.model.to_string().hash(&mut hasher);
    req.system_prompt.hash(&mut hasher);
    req.user_prompt.hash(&mut hasher);
    hasher.finish()
}

pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

pub struct SemanticCache {
    capacity: usize,
    entries: AHashMap<u64, LlmResponse>,
    lru_order: VecDeque<u64>,
    hits: AtomicU64,
    misses: AtomicU64,
    inner: Arc<dyn LlmAdapter>,
}

// AtomicU64 is Send+Sync; AHashMap+VecDeque are Send when T is.
// We use SemanticCache only from async tasks, not across threads simultaneously,
// but wrap in Arc<Mutex> at the call site for shared access.
unsafe impl Send for SemanticCache {}
unsafe impl Sync for SemanticCache {}

impl SemanticCache {
    pub fn new(capacity: usize, inner: Arc<dyn LlmAdapter>) -> Self {
        assert!(capacity > 0, "cache capacity must be > 0");
        Self {
            capacity,
            entries: AHashMap::with_capacity(capacity),
            lru_order: VecDeque::with_capacity(capacity),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inner,
        }
    }

    /// Complete a request, using the cache if possible.
    pub async fn complete(&mut self, req: LlmRequest) -> Result<LlmResponse> {
        let key = hash_request(&req);

        if self.entries.contains_key(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            self.touch(key);
            // SAFETY: key exists because contains_key just confirmed it.
            let cached = self.entries.get(&key).unwrap().clone();
            return Ok(cached);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let resp = self.inner.complete(req).await?;
        self.insert(key, resp.clone());
        Ok(resp)
    }

    fn touch(&mut self, key: u64) {
        if let Some(pos) = self.lru_order.iter().position(|&k| k == key) {
            self.lru_order.remove(pos);
            self.lru_order.push_front(key);
        }
    }

    fn insert(&mut self, key: u64, resp: LlmResponse) {
        if self.entries.len() >= self.capacity {
            if let Some(evicted) = self.lru_order.pop_back() {
                self.entries.remove(&evicted);
            }
        }
        self.entries.insert(key, resp);
        self.lru_order.push_front(key);
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed) as f64;
        let m = self.misses.load(Ordering::Relaxed) as f64;
        if h + m == 0.0 {
            0.0
        } else {
            h / (h + m)
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_adapter::MockLlm;
    use crate::types::LlmModel;

    fn req(prompt: &str) -> LlmRequest {
        LlmRequest {
            model: LlmModel::Mock,
            system_prompt: "sys".to_string(),
            user_prompt: prompt.to_string(),
            max_tokens: 32,
        }
    }

    #[tokio::test]
    async fn cache_hit_on_repeat() {
        let mock = Arc::new(MockLlm::new());
        let mut cache = SemanticCache::new(10, mock);

        // First call = miss
        cache.complete(req("hello")).await.unwrap();
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Repeat 99 times = 99 hits
        for _ in 0..99 {
            cache.complete(req("hello")).await.unwrap();
        }
        assert_eq!(cache.stats().hits, 99);
    }

    #[tokio::test]
    async fn different_prompts_are_different_keys() {
        let mock = Arc::new(MockLlm::new());
        let mut cache = SemanticCache::new(10, mock);
        cache.complete(req("hello")).await.unwrap();
        cache.complete(req("world")).await.unwrap();
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.len(), 2);
    }

    #[tokio::test]
    async fn lru_eviction_respects_capacity() {
        let mock = Arc::new(MockLlm::new());
        let mut cache = SemanticCache::new(3, mock);
        cache.complete(req("a")).await.unwrap();
        cache.complete(req("b")).await.unwrap();
        cache.complete(req("c")).await.unwrap();
        assert_eq!(cache.len(), 3);
        // Insert 4th: evicts LRU ("a")
        cache.complete(req("d")).await.unwrap();
        assert_eq!(cache.len(), 3);
        // "a" should now be a miss
        cache.complete(req("a")).await.unwrap();
        assert_eq!(cache.stats().misses, 5); // a,b,c,d,a = 5
    }

    #[tokio::test]
    async fn hit_rate_correct() {
        let mock = Arc::new(MockLlm::new());
        let mut cache = SemanticCache::new(10, mock);
        cache.complete(req("x")).await.unwrap();
        for _ in 0..3 {
            cache.complete(req("x")).await.unwrap();
        }
        assert!((cache.hit_rate() - 0.75).abs() < 0.01);
    }

    #[tokio::test]
    async fn hit_latency_is_fast() {
        use crate::llm_adapter::MockLlm;
        use std::time::Duration;
        use std::time::Instant;
        // Mock with 100ms delay — cache should bypass it
        let slow_mock = Arc::new(MockLlm::new().with_delay(Duration::from_millis(100)));
        let mut cache = SemanticCache::new(10, slow_mock);
        cache.complete(req("slow")).await.unwrap(); // miss: 100ms
        let start = Instant::now();
        cache.complete(req("slow")).await.unwrap(); // hit: < 1ms
        assert!(
            start.elapsed() < Duration::from_millis(5),
            "Cache hit was too slow: {:?}",
            start.elapsed()
        );
    }
}

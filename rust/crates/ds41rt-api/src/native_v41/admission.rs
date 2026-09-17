//! Bounded waiting before handing a request to the CUDA worker.
use std::{sync::{Arc, atomic::{AtomicU64, Ordering}}, time::{Duration, Instant}};
use tokio::sync::{mpsc, Semaphore};

#[derive(Clone)]
pub(super) struct Admission {
    waiters: Arc<Semaphore>,
    wait: Duration,
    waits: Arc<AtomicU64>,
    wait_ms: Arc<AtomicU64>,
    rejects: Arc<AtomicU64>,
}
#[derive(Debug, PartialEq)]
pub(super) enum Rejected { Overloaded, Closed }
impl Admission {
    pub fn new(depth: usize, wait: Duration) -> Self {
        Self { waiters: Arc::new(Semaphore::new(depth)), wait,
            waits: Arc::default(), wait_ms: Arc::default(), rejects: Arc::default() }
    }
    pub fn metrics(&self) -> serde_json::Value {
        serde_json::json!({"http_queue_waits": self.waits.load(Ordering::Relaxed),
            "http_queue_wait_ms_sum": self.wait_ms.load(Ordering::Relaxed),
            "http_queue_rejects": self.rejects.load(Ordering::Relaxed)})
    }
    pub async fn reserve<T>(&self, queue: mpsc::Sender<T>) -> Result<mpsc::OwnedPermit<T>, Rejected> {
        match queue.clone().try_reserve_owned() {
            Ok(permit) => return Ok(permit),
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(Rejected::Closed),
            Err(mpsc::error::TrySendError::Full(_)) => {}
        }
        // Without this bound, timeout(send) retains arbitrarily many prompts.
        let _waiter = self.waiters.try_acquire().map_err(|_| {
            self.rejects.fetch_add(1, Ordering::Relaxed); Rejected::Overloaded
        })?;
        self.waits.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let result = tokio::time::timeout(self.wait, queue.reserve_owned()).await;
        self.wait_ms.fetch_add(started.elapsed().as_millis().min(u64::MAX as u128) as u64, Ordering::Relaxed);
        match result {
            Ok(Ok(permit)) => Ok(permit),
            Ok(Err(_)) => Err(Rejected::Closed),
            Err(_) => { self.rejects.fetch_add(1, Ordering::Relaxed); Err(Rejected::Overloaded) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn full_queue_waits_for_worker_and_timeout_rejects() {
        let (tx, mut rx) = mpsc::channel(1); tx.send(1).await.unwrap();
        let admission = Admission::new(1, Duration::from_millis(20));
        assert!(matches!(admission.reserve(tx.clone()).await, Err(Rejected::Overloaded)));
        let pending = tokio::spawn({let a=admission.clone(); let tx=tx.clone(); async move {a.reserve(tx).await}});
        tokio::task::yield_now().await;
        assert!(!pending.is_finished());
        assert_eq!(rx.recv().await, Some(1));
        pending.await.unwrap().unwrap().send(2);
        assert_eq!(rx.recv().await, Some(2));
    }
    #[tokio::test]
    async fn waiters_are_bounded_and_cancel_releases_slot() {
        let (tx, _rx) = mpsc::channel(1); tx.send(1).await.unwrap();
        let admission = Admission::new(1, Duration::from_secs(60));
        let pending = tokio::spawn({let a=admission.clone(); let tx=tx.clone(); async move {a.reserve(tx).await}});
        tokio::task::yield_now().await;
        assert!(matches!(admission.reserve(tx.clone()).await, Err(Rejected::Overloaded)));
        pending.abort(); let _ = pending.await;
        assert_eq!(admission.waiters.available_permits(), 1);
    }
    #[tokio::test]
    async fn shutdown_wakes_waiter_as_unavailable() {
        let (tx, rx) = mpsc::channel(1); tx.send(1).await.unwrap();
        let admission = Admission::new(1, Duration::from_secs(60));
        let pending = tokio::spawn(async move {admission.reserve(tx).await});
        tokio::task::yield_now().await; drop(rx);
        assert!(matches!(pending.await.unwrap(), Err(Rejected::Closed)));
    }
}

//! Independent distributed lanes linked only by the same prompt's cache dependencies.
use super::*;
use crate::v41_backbone_cache::CacheLease;
use crate::v41_requests::RequestTokens;
use ds41rt_transport::ExpertV2SourceKind;
use std::cell::RefCell;
use tokio::sync::Notify;

pub(super) struct EncoderFlow<'s, 'q, 'a> {
    pub predecessor: Option<&'s [Notify; 20]>,
    pub successor: Option<&'s [Notify; 20]>,
    pub previous_commit: Option<&'s Notify>,
    pub suffix: &'s RefCell<&'q mut DeviceOwner<'a, EncoderSuffix<'a>>>,
    pub keep_running: &'s dyn Fn() -> bool,
}
struct StreamGuard<'r, 'q, 'a> {
    requests: &'r RefCell<&'q mut Requests<'a>>,
    lease: CacheLease,
    complete: bool,
}
impl Drop for StreamGuard<'_, '_, '_> {
    fn drop(&mut self) {
        if self.complete {
            return;
        }
        let mut requests = self.requests.borrow_mut();
        if requests.cache().request_id(self.lease).is_ok() {
            if let Err(error) = requests.release(self.lease) {
                tracing::error!(%error, "revoking cancelled distributed encoder stream");
            }
        }
    }
}
impl<'w, 'a> DistributedTargetPass<'w, 'a> {
    /// Each lane reserves a new chunk after committing its preceding chunk.
    /// Layer publication, suffix capture and history commit preserve this prompt's
    /// causal order. No other request or decode lane participates in these waits.
    /// # Safety
    /// Passes and transports have independent mutable storage and the same model,
    /// placement and capacity. Poll both on one CUDA-owning executor thread.
    pub async unsafe fn execute_encoder_stream(
        &mut self,
        other: &mut Self,
        requests: &mut Requests<'a>,
        lease: CacheLease,
        chunks: &[&[u32]],
        transports: [&mut DeviceOwner<'a, NativeTp4Wave<'a>>; 2],
        suffix: &mut DeviceOwner<'a, EncoderSuffix<'a>>,
        keep_running: &dyn Fn() -> bool,
    ) -> Result<()> {
        ensure!(
            !chunks.is_empty() && chunks.iter().all(|chunk| !chunk.is_empty()),
            "empty encoder stream"
        );
        ensure!(
            self.state == State::Idle && other.state == State::Idle,
            "distributed encoder lane still in use"
        );
        ensure!(
            self.map == other.map,
            "distributed encoder lane placement differs"
        );
        let requests = RefCell::new(requests);
        let mut guard = StreamGuard {
            requests: &requests,
            lease,
            complete: false,
        };
        let suffix = RefCell::new(suffix);
        let published: Vec<[Notify; 20]> = (0..chunks.len())
            .map(|_| std::array::from_fn(|_| Notify::new()))
            .collect();
        let reserved: Vec<Notify> = (0..chunks.len()).map(|_| Notify::new()).collect();
        let committed: Vec<Notify> = (0..chunks.len()).map(|_| Notify::new()).collect();
        let [first, second] = transports;
        tokio::try_join!(
            biased;
            unsafe { self.encoder_stream_lane(0, &requests, lease, chunks, first, &suffix,
                &published, &reserved, &committed, keep_running) },
            unsafe { other.encoder_stream_lane(1, &requests, lease, chunks, second, &suffix,
                &published, &reserved, &committed, keep_running) },
        )?;
        guard.complete = true;
        Ok(())
    }
    async unsafe fn encoder_stream_lane(
        &mut self,
        parity: usize,
        requests: &RefCell<&mut Requests<'a>>,
        lease: CacheLease,
        chunks: &[&[u32]],
        transport: &mut DeviceOwner<'a, NativeTp4Wave<'a>>,
        suffix: &RefCell<&mut DeviceOwner<'a, EncoderSuffix<'a>>>,
        published: &[[Notify; 20]],
        reserved: &[Notify],
        committed: &[Notify],
        keep_running: &dyn Fn() -> bool,
    ) -> Result<()> {
        for index in (parity..chunks.len()).step_by(2) {
            if index != 0 {
                reserved[index - 1].notified().await;
            }
            ensure!(keep_running(), "client disconnected");
            let chunk = chunks[index];
            let mut batch = requests.borrow_mut().reserve_encoder(&[RequestTokens {
                lease,
                tokens: chunk,
                image_mask: None,
                kind: ExpertV2SourceKind::Prefill,
            }])?;
            let mut guard = ReservedPassGuard {
                pass: self,
                requests,
                batch: &mut batch,
                complete: false,
            };
            reserved[index].notify_one();
            let started = std::time::Instant::now();
            let flow = EncoderFlow {
                predecessor: index.checked_sub(1).map(|i| &published[i]),
                successor: (index + 1 < chunks.len()).then_some(&published[index]),
                previous_commit: index.checked_sub(1).map(|i| &committed[i]),
                suffix,
                keep_running,
            };
            unsafe {
                guard
                    .pass
                    .execute_inner(
                        requests,
                        guard.batch,
                        transport,
                        0,
                        &[],
                        None,
                        None,
                        false,
                        Some(flow),
                    )
                    .await?;
            }
            ensure!(keep_running(), "client disconnected");
            guard.pass.enqueue_cache_commit(
                &requests.borrow(),
                guard.batch,
                &[chunk.len() as u32],
            )?;
            while !guard.pass.poll_cache_commit()? {
                tokio::task::yield_now().await;
            }
            guard.pass.commit(
                &mut requests.borrow_mut(),
                guard.batch,
                &[chunk.len() as u32],
            )?;
            guard.complete = true;
            committed[index].notify_one();
            crate::v41_native_serve::console::totals::prefill(chunk.len());
            crate::v41_native_serve::console::Prefill::done(crate::v41_native_serve::console::PrefillKind::Chunk,
                parity, index, chunks.len(), chunk.len(), started);
        }
        Ok(())
    }
}

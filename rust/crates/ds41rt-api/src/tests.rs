use super::*;
use axum::body::{to_bytes, Body};
use axum::http::{Method, Request};
use futures::StreamExt;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tower::ServiceExt;

use crate::request::{
    real_ds4_full_prompt_text, real_ds4_full_request_prompt_text, request_max_tokens,
    request_sampling_params, request_uses_greedy_sampling, validate_request,
};

fn test_state(backend: ApiBackend, transport: ApiTransport) -> ApiState {
    ApiState {
        config: ApiConfig {
            backend,
            transport,
            model_id: DEFAULT_MODEL_ID.to_owned(),
            expert_targets: Vec::new(),
            real_slice: None,
            real_full: None,
            real_full_executor: None,
        },
        service_instance_id: "00000000-0000-4000-8000-000000000001".to_owned(),
        next_request_id: AtomicU64::new(1),
        tool_continuations: Mutex::new(crate::continuation::ToolContinuationCache::default()),
    }
}

#[derive(Debug)]
struct CapturingRealFullExecutor {
    requests: Arc<Mutex<Vec<RealFullRequest>>>,
    info: RealFullInfo,
}

impl RealFullRequestExecutor for CapturingRealFullExecutor {
    fn execute_real_full_request(&self, request: RealFullRequest) -> Result<RealFullInfo, String> {
        self.requests.lock().unwrap().push(request);
        Ok(self.info.clone())
    }
}

#[derive(Debug)]
struct ThreadRecordingRealFullExecutor {
    calls: Arc<Mutex<Vec<(u64, thread::ThreadId)>>>,
    finishes: Arc<Mutex<Vec<(String, thread::ThreadId)>>>,
    info: RealFullInfo,
}

#[derive(Debug)]
struct BatchRecordingRealFullExecutor {
    batches: Arc<Mutex<Vec<Vec<String>>>>,
    info: RealFullInfo,
}

#[derive(Debug)]
struct PersistentBatchRecordingRealFullExecutor {
    batches: Arc<Mutex<Vec<Vec<(String, usize, usize)>>>>,
    finishes: Arc<Mutex<Vec<String>>>,
    info: RealFullInfo,
}

#[derive(Debug)]
struct RetryableAdmissionRealFullExecutor {
    batches: Arc<Mutex<Vec<Vec<(String, usize)>>>>,
    finishes: Arc<Mutex<Vec<String>>>,
    rejected_once: AtomicBool,
    info: RealFullInfo,
}

#[derive(Debug)]
struct ConcurrencyLimitedRealFullExecutor {
    first_cycle_started: mpsc::SyncSender<()>,
    release_first_cycle: Mutex<mpsc::Receiver<()>>,
    held_once: AtomicBool,
    info: RealFullInfo,
}

#[derive(Debug)]
struct HeadOfLineAdmissionRealFullExecutor {
    batches: Arc<Mutex<Vec<(String, usize)>>>,
    finishes: Arc<Mutex<Vec<String>>>,
    first_large_attempted: mpsc::SyncSender<()>,
    release_first_large_attempt: Mutex<mpsc::Receiver<()>>,
    held_first_large_attempt: AtomicBool,
    small_admitted: AtomicBool,
    info: RealFullInfo,
}

#[derive(Debug)]
struct PendingCancellationRealFullExecutor {
    batches: Arc<Mutex<Vec<(String, usize)>>>,
    finishes: Arc<Mutex<Vec<String>>>,
    first_cycle_started: mpsc::SyncSender<()>,
    release_first_cycle: Mutex<mpsc::Receiver<()>>,
    held_once: AtomicBool,
    info: RealFullInfo,
}

#[derive(Debug)]
struct ActiveOwnerAdmissionRealFullExecutor {
    batches: Arc<Mutex<Vec<(String, usize)>>>,
    finishes: Arc<Mutex<Vec<String>>>,
    first_cycle_started: mpsc::SyncSender<()>,
    release_first_cycle: Mutex<mpsc::Receiver<()>>,
    held_once: AtomicBool,
    active_finished: AtomicBool,
    info: RealFullInfo,
}

#[derive(Debug)]
struct BoundaryJoinRealFullExecutor {
    batches: Arc<Mutex<Vec<Vec<(String, usize)>>>>,
    finishes: Arc<Mutex<Vec<String>>>,
    first_cycle_started: mpsc::SyncSender<()>,
    release_first_cycle: Mutex<mpsc::Receiver<()>>,
    held_once: AtomicBool,
    info: RealFullInfo,
}

#[derive(Debug)]
struct TerminalFinishBarrierRealFullExecutor {
    finish_started: mpsc::SyncSender<()>,
    release_finish: Mutex<mpsc::Receiver<()>>,
    info: RealFullInfo,
}

#[derive(Debug)]
struct BufferedResponseCancellationRealFullExecutor {
    first_cycle_started: mpsc::SyncSender<()>,
    release_first_cycle: Mutex<mpsc::Receiver<()>>,
    held_once: AtomicBool,
    finishes: Arc<Mutex<Vec<String>>>,
    info: RealFullInfo,
}

impl RealFullRequestExecutor for BatchRecordingRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Ok(self.info.clone())
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        self.batches.lock().unwrap().push(
            requests
                .iter()
                .map(|request| request.sequence_id.clone())
                .collect(),
        );
        requests
            .into_iter()
            .map(|_| Ok(RealFullDecodeCycle::single_token(self.info.clone())))
            .collect()
    }

    fn real_full_decode_cycle_batch_coalesce_timeout(
        &self,
        _request: &RealFullRequest,
    ) -> Option<Duration> {
        Some(Duration::from_millis(100))
    }

    fn real_full_decode_cycle_batch_max_size(&self, _request: &RealFullRequest) -> usize {
        4
    }
}

impl RealFullRequestExecutor for PersistentBatchRecordingRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Err("persistent batch test requires cycle execution".to_owned())
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        self.batches.lock().unwrap().push(
            requests
                .iter()
                .map(|request| {
                    (
                        request.sequence_id.clone(),
                        request.generated_token_ids.len(),
                        request.decode_budget,
                    )
                })
                .collect(),
        );
        if requests.len() == 1 && requests[0].generated_token_ids.is_empty() {
            thread::sleep(Duration::from_millis(10));
        }
        requests
            .into_iter()
            .map(|request| {
                let (token_id, text) = if request.generated_token_ids.is_empty() {
                    (101, "a")
                } else {
                    (102, "stop")
                };
                let mut info = self.info.clone();
                info.status = "ready".to_owned();
                info.blocker.clear();
                info.failed_requirements.clear();
                info.scheduler_terminal_lm_head_sampled_token_id = Some(token_id);
                info.scheduler_terminal_lm_head_sampled_text = Some(text.to_owned());
                Ok(RealFullDecodeCycle {
                    info,
                    generated_tokens: vec![RealFullGeneratedToken {
                        token_id,
                        text: Some(text.to_owned()),
                    }],
                })
            })
            .collect()
    }

    fn real_full_decode_cycle_batch_max_size(&self, _request: &RealFullRequest) -> usize {
        4
    }

    fn real_full_decode_cycle_batch_coalesce_timeout(
        &self,
        _request: &RealFullRequest,
    ) -> Option<Duration> {
        Some(Duration::from_millis(10))
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        self.finishes.lock().unwrap().push(sequence_id.to_owned());
        Ok(())
    }
}

impl RealFullRequestExecutor for BufferedResponseCancellationRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Err("buffered-response cancellation test requires cycle execution".to_owned())
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        if !self.held_once.swap(true, Ordering::SeqCst) {
            self.first_cycle_started
                .send(())
                .expect("notifying buffered-response cycle start");
            self.release_first_cycle
                .lock()
                .unwrap()
                .recv()
                .expect("releasing buffered-response cycle");
        }
        requests
            .into_iter()
            .map(|request| {
                let token_id = 100 + request.generated_token_ids.len();
                let mut info = self.info.clone();
                info.status = "ready".to_owned();
                info.blocker.clear();
                info.failed_requirements.clear();
                info.scheduler_terminal_lm_head_sampled_token_id = Some(token_id);
                info.scheduler_terminal_lm_head_sampled_text = Some("a".to_owned());
                Ok(RealFullDecodeCycle {
                    info,
                    generated_tokens: vec![RealFullGeneratedToken {
                        token_id,
                        text: Some("a".to_owned()),
                    }],
                })
            })
            .collect()
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        self.finishes.lock().unwrap().push(sequence_id.to_owned());
        Ok(())
    }
}

impl RealFullRequestExecutor for RetryableAdmissionRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Err("retryable admission test requires cycle execution".to_owned())
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        self.batches.lock().unwrap().push(
            requests
                .iter()
                .map(|request| {
                    (
                        request.sequence_id.clone(),
                        request.generated_token_ids.len(),
                    )
                })
                .collect(),
        );
        requests
            .into_iter()
            .map(|request| {
                if request.sequence_id == "sequence-b"
                    && request.generated_token_ids.is_empty()
                    && !self.rejected_once.swap(true, Ordering::SeqCst)
                {
                    return Err("test transient KV admission exhaustion".to_owned());
                }
                let token_id = if request.generated_token_ids.is_empty() {
                    1
                } else {
                    2
                };
                let mut info = self.info.clone();
                info.status = "ready".to_owned();
                info.blocker.clear();
                info.failed_requirements.clear();
                info.scheduler_terminal_lm_head_sampled_token_id = Some(token_id);
                Ok(RealFullDecodeCycle {
                    info,
                    generated_tokens: vec![RealFullGeneratedToken {
                        token_id,
                        text: Some(token_id.to_string()),
                    }],
                })
            })
            .collect()
    }

    fn real_full_decode_cycle_batch_coalesce_timeout(
        &self,
        _request: &RealFullRequest,
    ) -> Option<Duration> {
        Some(Duration::from_millis(100))
    }

    fn real_full_decode_cycle_batch_max_size(&self, _request: &RealFullRequest) -> usize {
        2
    }

    fn real_full_retryable_admission_error(&self, _request: &RealFullRequest, error: &str) -> bool {
        error == "test transient KV admission exhaustion"
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        self.finishes.lock().unwrap().push(sequence_id.to_owned());
        Ok(())
    }
}

impl RealFullRequestExecutor for ConcurrencyLimitedRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Err("concurrency-limit test requires cycle execution".to_owned())
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        if !self.held_once.swap(true, Ordering::SeqCst) {
            self.first_cycle_started
                .send(())
                .expect("notifying first test cycle");
            self.release_first_cycle
                .lock()
                .unwrap()
                .recv()
                .expect("releasing first test cycle");
        }
        requests
            .into_iter()
            .map(|request| {
                let token_id = if request.generated_token_ids.is_empty() {
                    1
                } else {
                    2
                };
                let mut info = self.info.clone();
                info.status = "ready".to_owned();
                info.blocker.clear();
                info.failed_requirements.clear();
                info.scheduler_terminal_lm_head_sampled_token_id = Some(token_id);
                Ok(RealFullDecodeCycle {
                    info,
                    generated_tokens: vec![RealFullGeneratedToken {
                        token_id,
                        text: Some(token_id.to_string()),
                    }],
                })
            })
            .collect()
    }

    fn real_full_decode_cycle_batch_max_size(&self, _request: &RealFullRequest) -> usize {
        1
    }

    fn real_full_max_concurrent_sequences(&self) -> usize {
        1
    }
}

impl RealFullRequestExecutor for HeadOfLineAdmissionRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Err("head-of-line admission test requires cycle execution".to_owned())
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        assert_eq!(requests.len(), 1);
        let request = requests.into_iter().next().unwrap();
        self.batches.lock().unwrap().push((
            request.sequence_id.clone(),
            request.generated_token_ids.len(),
        ));
        if request.sequence_id == "large-sequence"
            && request.generated_token_ids.is_empty()
            && !self.small_admitted.load(Ordering::SeqCst)
        {
            if !self.held_first_large_attempt.swap(true, Ordering::SeqCst) {
                self.first_large_attempted
                    .send(())
                    .expect("notifying first blocked large admission");
                self.release_first_large_attempt
                    .lock()
                    .unwrap()
                    .recv()
                    .expect("releasing first blocked large admission");
            }
            return vec![Err("test capacity admission exhaustion".to_owned())];
        }
        if request.sequence_id == "small-sequence" && request.generated_token_ids.is_empty() {
            self.small_admitted.store(true, Ordering::SeqCst);
        }

        let token_id = if request.generated_token_ids.is_empty() {
            1
        } else {
            2
        };
        let mut info = self.info.clone();
        info.status = "ready".to_owned();
        info.blocker.clear();
        info.failed_requirements.clear();
        info.scheduler_terminal_lm_head_sampled_token_id = Some(token_id);
        vec![Ok(RealFullDecodeCycle {
            info,
            generated_tokens: vec![RealFullGeneratedToken {
                token_id,
                text: Some(token_id.to_string()),
            }],
        })]
    }

    fn real_full_decode_cycle_batch_max_size(&self, _request: &RealFullRequest) -> usize {
        1
    }

    fn real_full_max_concurrent_sequences(&self) -> usize {
        2
    }

    fn real_full_retryable_admission_error(&self, _request: &RealFullRequest, error: &str) -> bool {
        error == "test capacity admission exhaustion"
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        self.finishes.lock().unwrap().push(sequence_id.to_owned());
        Ok(())
    }
}

impl RealFullRequestExecutor for PendingCancellationRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Err("pending cancellation test requires cycle execution".to_owned())
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        assert_eq!(requests.len(), 1);
        let request = requests.into_iter().next().unwrap();
        self.batches.lock().unwrap().push((
            request.sequence_id.clone(),
            request.generated_token_ids.len(),
        ));
        if !self.held_once.swap(true, Ordering::SeqCst) {
            self.first_cycle_started
                .send(())
                .expect("notifying held active cycle");
            self.release_first_cycle
                .lock()
                .unwrap()
                .recv()
                .expect("releasing held active cycle");
        }
        let token_id = if request.generated_token_ids.is_empty() {
            1
        } else {
            2
        };
        let mut info = self.info.clone();
        info.status = "ready".to_owned();
        info.blocker.clear();
        info.failed_requirements.clear();
        info.scheduler_terminal_lm_head_sampled_token_id = Some(token_id);
        vec![Ok(RealFullDecodeCycle {
            info,
            generated_tokens: vec![RealFullGeneratedToken {
                token_id,
                text: Some(token_id.to_string()),
            }],
        })]
    }

    fn real_full_decode_cycle_batch_max_size(&self, _request: &RealFullRequest) -> usize {
        1
    }

    fn real_full_max_concurrent_sequences(&self) -> usize {
        2
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        self.finishes.lock().unwrap().push(sequence_id.to_owned());
        Ok(())
    }
}

impl RealFullRequestExecutor for ActiveOwnerAdmissionRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Err("active-owner admission test requires cycle execution".to_owned())
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        assert_eq!(requests.len(), 1);
        let request = requests.into_iter().next().unwrap();
        self.batches.lock().unwrap().push((
            request.sequence_id.clone(),
            request.generated_token_ids.len(),
        ));
        if request.sequence_id == "blocked-sequence"
            && request.generated_token_ids.is_empty()
            && !self.active_finished.load(Ordering::SeqCst)
        {
            return vec![Err("test active-owner capacity exhaustion".to_owned())];
        }
        if request.sequence_id == "active-sequence"
            && request.generated_token_ids.is_empty()
            && !self.held_once.swap(true, Ordering::SeqCst)
        {
            self.first_cycle_started
                .send(())
                .expect("notifying held active-owner cycle");
            self.release_first_cycle
                .lock()
                .unwrap()
                .recv()
                .expect("releasing held active-owner cycle");
        }
        let token_id = if request.generated_token_ids.is_empty() {
            1
        } else {
            2
        };
        let mut info = self.info.clone();
        info.status = "ready".to_owned();
        info.blocker.clear();
        info.failed_requirements.clear();
        info.scheduler_terminal_lm_head_sampled_token_id = Some(token_id);
        vec![Ok(RealFullDecodeCycle {
            info,
            generated_tokens: vec![RealFullGeneratedToken {
                token_id,
                text: Some(token_id.to_string()),
            }],
        })]
    }

    fn real_full_decode_cycle_batch_max_size(&self, _request: &RealFullRequest) -> usize {
        2
    }

    fn real_full_max_concurrent_sequences(&self) -> usize {
        2
    }

    fn real_full_retryable_admission_error(&self, _request: &RealFullRequest, error: &str) -> bool {
        error == "test active-owner capacity exhaustion"
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        self.finishes.lock().unwrap().push(sequence_id.to_owned());
        if sequence_id == "active-sequence" {
            self.active_finished.store(true, Ordering::SeqCst);
        }
        Ok(())
    }
}

impl RealFullRequestExecutor for BoundaryJoinRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Err("boundary-join test requires cycle execution".to_owned())
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        self.batches.lock().unwrap().push(
            requests
                .iter()
                .map(|request| {
                    (
                        request.sequence_id.clone(),
                        request.generated_token_ids.len(),
                    )
                })
                .collect(),
        );
        if !self.held_once.swap(true, Ordering::SeqCst) {
            self.first_cycle_started
                .send(())
                .expect("notifying boundary-join first cycle");
            self.release_first_cycle
                .lock()
                .unwrap()
                .recv()
                .expect("releasing boundary-join first cycle");
        }
        requests
            .into_iter()
            .map(|request| {
                let token_id = request.generated_token_ids.len() + 1;
                let mut info = self.info.clone();
                info.status = "ready".to_owned();
                info.blocker.clear();
                info.failed_requirements.clear();
                info.scheduler_terminal_lm_head_sampled_token_id = Some(token_id);
                Ok(RealFullDecodeCycle {
                    info,
                    generated_tokens: vec![RealFullGeneratedToken {
                        token_id,
                        text: Some(token_id.to_string()),
                    }],
                })
            })
            .collect()
    }

    fn real_full_decode_cycle_batch_coalesce_timeout(
        &self,
        _request: &RealFullRequest,
    ) -> Option<Duration> {
        Some(Duration::from_millis(10))
    }

    fn real_full_decode_cycle_batch_max_size(&self, _request: &RealFullRequest) -> usize {
        4
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        self.finishes.lock().unwrap().push(sequence_id.to_owned());
        Ok(())
    }
}

impl RealFullRequestExecutor for TerminalFinishBarrierRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Err("terminal-finish ordering test requires cycle execution".to_owned())
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        requests
            .into_iter()
            .map(|_| {
                let mut info = self.info.clone();
                info.status = "ready".to_owned();
                info.blocker.clear();
                info.failed_requirements.clear();
                info.scheduler_terminal_lm_head_sampled_token_id = Some(1);
                Ok(RealFullDecodeCycle {
                    info,
                    generated_tokens: vec![RealFullGeneratedToken {
                        token_id: 1,
                        text: Some("done".to_owned()),
                    }],
                })
            })
            .collect()
    }

    fn real_full_decode_cycle_batch_max_size(&self, _request: &RealFullRequest) -> usize {
        1
    }

    fn finish_real_full_sequence(&self, _sequence_id: &str) -> Result<(), String> {
        self.finish_started
            .send(())
            .expect("notifying terminal cleanup start");
        self.release_finish
            .lock()
            .unwrap()
            .recv()
            .expect("releasing terminal cleanup");
        Ok(())
    }
}

impl RealFullRequestExecutor for ThreadRecordingRealFullExecutor {
    fn execute_real_full_request(&self, request: RealFullRequest) -> Result<RealFullInfo, String> {
        self.calls
            .lock()
            .unwrap()
            .push((request.request_index, thread::current().id()));
        Ok(self.info.clone())
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        self.finishes
            .lock()
            .unwrap()
            .push((sequence_id.to_owned(), thread::current().id()));
        Ok(())
    }
}

#[test]
fn api_transport_parses_verbs_host() {
    let transport = ApiTransport::parse("verbs-host").expect("verbs-host parses");

    assert_eq!(transport, ApiTransport::VerbsHost);
    assert_eq!(transport.label(), "verbs-host");
}

#[derive(Debug)]
struct StepSamplingRealFullExecutor {
    requests: Arc<Mutex<Vec<RealFullRequest>>>,
    base: RealFullInfo,
    tokens: Vec<(usize, String)>,
}

impl RealFullRequestExecutor for StepSamplingRealFullExecutor {
    fn execute_real_full_request(&self, request: RealFullRequest) -> Result<RealFullInfo, String> {
        let token = self.tokens.get(request.decode_step_index).ok_or_else(|| {
            format!(
                "missing token for decode step {}",
                request.decode_step_index
            )
        })?;
        let mut info = self.base.clone();
        info.request_prefill_tokens = if request.generated_token_ids.is_empty() {
            request.prompt_tokens
        } else {
            0
        };
        info.request_prefill_chunks = usize::from(info.request_prefill_tokens > 0);
        info.request_decode_budget = request.max_tokens;
        info.scheduler_terminal_lm_head_top_token_id = Some(token.0);
        info.scheduler_terminal_lm_head_sampled_token_id = Some(token.0);
        info.scheduler_terminal_lm_head_sampled_text = Some(token.1.clone());
        self.requests.lock().unwrap().push(request);
        Ok(info)
    }
}

#[derive(Debug)]
struct FinishingStepSamplingRealFullExecutor {
    inner: StepSamplingRealFullExecutor,
    finishes: Arc<Mutex<Vec<String>>>,
}

impl RealFullRequestExecutor for FinishingStepSamplingRealFullExecutor {
    fn execute_real_full_request(&self, request: RealFullRequest) -> Result<RealFullInfo, String> {
        self.inner.execute_real_full_request(request)
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        self.finishes.lock().unwrap().push(sequence_id.to_owned());
        Ok(())
    }
}

#[derive(Debug)]
struct FailingFinishingRealFullExecutor {
    sequences: Arc<Mutex<Vec<String>>>,
    finishes: Arc<Mutex<Vec<String>>>,
}

impl RealFullRequestExecutor for FailingFinishingRealFullExecutor {
    fn execute_real_full_request(&self, request: RealFullRequest) -> Result<RealFullInfo, String> {
        self.sequences
            .lock()
            .unwrap()
            .push(request.sequence_id.clone());
        Err("intentional real-full execution failure".to_owned())
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        self.finishes.lock().unwrap().push(sequence_id.to_owned());
        Ok(())
    }
}

#[derive(Debug)]
struct CycleSamplingRealFullExecutor {
    requests: Arc<Mutex<Vec<RealFullRequest>>>,
    base: RealFullInfo,
    tokens: Vec<(usize, String)>,
}

impl RealFullRequestExecutor for CycleSamplingRealFullExecutor {
    fn execute_real_full_request(&self, _request: RealFullRequest) -> Result<RealFullInfo, String> {
        Err("cycle executor must be called through execute_real_full_decode_cycle".to_owned())
    }

    fn execute_real_full_decode_cycle(
        &self,
        request: RealFullRequest,
    ) -> Result<RealFullDecodeCycle, String> {
        let mut info = self.base.clone();
        info.request_prefill_tokens = if request.generated_token_ids.is_empty() {
            request.prompt_tokens
        } else {
            0
        };
        info.request_prefill_chunks = usize::from(info.request_prefill_tokens > 0);
        let last = self
            .tokens
            .last()
            .ok_or_else(|| "cycle executor requires at least one token".to_owned())?;
        info.scheduler_terminal_lm_head_top_token_id = Some(last.0);
        info.scheduler_terminal_lm_head_sampled_token_id = Some(last.0);
        info.scheduler_terminal_lm_head_sampled_text = Some(last.1.clone());
        self.requests.lock().unwrap().push(request);
        Ok(RealFullDecodeCycle {
            info,
            generated_tokens: self
                .tokens
                .iter()
                .map(|(token_id, text)| RealFullGeneratedToken {
                    token_id: *token_id,
                    text: Some(text.clone()),
                })
                .collect(),
        })
    }
}

#[derive(Debug)]
struct BlockingStepSamplingRealFullExecutor {
    requests: Arc<Mutex<Vec<RealFullRequest>>>,
    base: RealFullInfo,
    token: (usize, String),
    entered: mpsc::Sender<()>,
    release: Mutex<mpsc::Receiver<()>>,
}

impl RealFullRequestExecutor for BlockingStepSamplingRealFullExecutor {
    fn execute_real_full_request(&self, request: RealFullRequest) -> Result<RealFullInfo, String> {
        let _ = self.entered.send(());
        self.release
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .map_err(|err| format!("test executor release wait failed: {err}"))?;
        let mut info = self.base.clone();
        info.request_prefill_tokens = if request.generated_token_ids.is_empty() {
            request.prompt_tokens
        } else {
            0
        };
        info.request_prefill_chunks = usize::from(info.request_prefill_tokens > 0);
        info.request_decode_budget = request.max_tokens;
        info.scheduler_terminal_lm_head_top_token_id = Some(self.token.0);
        info.scheduler_terminal_lm_head_sampled_token_id = Some(self.token.0);
        info.scheduler_terminal_lm_head_sampled_text = Some(self.token.1.clone());
        self.requests.lock().unwrap().push(request);
        Ok(info)
    }
}

fn base_request(content: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "ds41rt-tiny".to_owned(),
        messages: vec![ChatMessage {
            role: "user".to_owned(),
            content: Some(Value::String(content.to_owned())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
            ..ChatMessage::default()
        }],
        stream: false,
        stream_options: None,
        max_tokens: Some(16),
        max_completion_tokens: None,
        min_tokens: None,
        ignore_eos: None,
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: None,
        seed: None,
        stop: None,
        response_format: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        tool_decoding_assistance: None,
        thinking: None,
        enable_thinking: None,
        reasoning_effort: None,
    }
}

#[test]
fn real_ds4_full_sampling_defaults_to_greedy_and_requires_explicit_opt_in() {
    let mut request = base_request("hi");
    request.temperature = None;
    assert!(request_uses_greedy_sampling(&request));

    request.temperature = Some(0.0);
    assert!(request_uses_greedy_sampling(&request));

    request.temperature = Some(0.7);
    assert!(!request_uses_greedy_sampling(&request));
}

#[test]
fn non_greedy_sampling_resolves_request_options_and_seeded_uniforms() {
    let mut request = base_request("hi");
    request.temperature = Some(1.2);
    request.top_p = Some(0.8);
    request.top_k = Some(32);
    request.seed = Some(42);

    let sampling = request_sampling_params(&request);
    assert!(!sampling.is_greedy());
    assert_eq!(sampling.temperature(), 1.2);
    assert_eq!(sampling.top_p(), 0.8);
    assert_eq!(sampling.top_k(), 32);
    assert_eq!(sampling.seed(), 42);
    assert_eq!(sampling.random_uniform(0), sampling.random_uniform(0));
    assert_ne!(sampling.random_uniform(0), sampling.random_uniform(1));
    assert!((0.0..1.0).contains(&sampling.random_uniform(0)));
}

#[test]
fn request_accepts_openai_max_completion_tokens() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_completion_tokens": 10_000
    }))
    .unwrap();

    assert_eq!(request_max_tokens(&request), 10_000);
    assert!(validate_request(&request).is_ok());
}

#[test]
fn request_defaults_to_a_useful_chat_completion_budget() {
    let mut request = base_request("hello");
    request.max_tokens = None;

    assert_eq!(request_max_tokens(&request), 16 * 1024);
    assert!(validate_request(&request).is_ok());
}

#[test]
fn request_accepts_exact_generation_extensions_and_rejects_an_impossible_floor() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 128,
        "min_tokens": 128,
        "ignore_eos": true
    }))
    .unwrap();

    assert_eq!(request.min_tokens, Some(128));
    assert_eq!(request.ignore_eos, Some(true));
    validate_request(&request).unwrap();

    let mut impossible = request;
    impossible.min_tokens = Some(129);
    let error = validate_request(&impossible).unwrap_err();
    assert_eq!(error.param.as_deref(), Some("min_tokens"));
}

#[test]
fn request_validates_openai_json_schema_strict_subset() {
    let valid: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "test",
        "messages": [{"role": "user", "content": "answer"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_1",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "details": {
                            "type": "object",
                            "properties": {"score": {"type": "number"}},
                            "required": ["score"],
                            "additionalProperties": false
                        }
                    },
                    "required": ["answer", "details"],
                    "additionalProperties": false
                }
            }
        }
    }))
    .unwrap();
    validate_request(&valid).unwrap();
    let prompt = real_ds4_full_request_prompt_text(&valid);
    assert!(prompt.contains("## Response Format:"));
    assert!(prompt.contains("\"additionalProperties\": false"));

    let mut missing_required = valid.clone();
    let Some(ResponseFormat::JsonSchema { json_schema }) =
        missing_required.response_format.as_mut()
    else {
        panic!("test request has JSON Schema response format")
    };
    json_schema.schema["required"] = json!(["answer"]);
    let error = validate_request(&missing_required).unwrap_err();
    assert!(error
        .message
        .contains("required must contain every property"));
}

#[test]
fn request_accepts_combined_response_and_tool_contracts() {
    let mut request = base_request("answer or call");
    request.response_format = Some(ResponseFormat::JsonObject);
    request.tools = Some(vec![lookup_tool()]);
    validate_request(&request).unwrap();
    let constraint = crate::constrained::request_constraint(&request)
        .unwrap()
        .expect("combined response/tool request should be constrained");
    let RealFullConstraintGrammar::StructuralTag {
        structural_tag_json,
    } = &constraint.grammar
    else {
        panic!("combined response/tool request should use a structural grammar")
    };
    let grammar: serde_json::Value = serde_json::from_str(structural_tag_json).unwrap();
    assert_eq!(grammar["format"]["type"], "or");
    assert_eq!(grammar["format"]["elements"][0]["type"], "json_schema");
    assert_eq!(grammar["format"]["elements"][1]["type"], "tag");

    request.tool_choice = Some(ToolChoice::Mode("none".to_owned()));
    validate_request(&request).unwrap();
    assert!(matches!(
        crate::constrained::request_constraint(&request)
            .unwrap()
            .unwrap()
            .grammar,
        RealFullConstraintGrammar::Json
    ));
}

#[test]
fn strict_zero_argument_tool_accepts_omitted_parameters() {
    let mut request = base_request("ping");
    request.tools = Some(vec![ChatTool {
        tool_type: "function".to_owned(),
        function: ChatFunction {
            name: "ping".to_owned(),
            description: None,
            parameters: None,
            strict: Some(true),
        },
    }]);
    request.tool_choice = Some(ToolChoice::Mode("required".to_owned()));
    validate_request(&request).unwrap();
    assert!(crate::constrained::request_constraint(&request)
        .unwrap()
        .is_some());
}

#[test]
fn historical_zero_argument_tool_calls_accept_omitted_null_or_object_arguments() {
    for arguments in [None, Some(Value::Null), Some(json!({"query": "bird"}))] {
        let mut function = json!({"name": "lookup"});
        if let Some(arguments) = arguments {
            function["arguments"] = arguments;
        }
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test",
            "messages": [{
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": function
                }]
            }]
        }))
        .unwrap();

        let arguments = &request.messages[0].tool_calls.as_ref().unwrap()[0]
            .function
            .arguments;
        assert!(matches!(
            serde_json::from_str::<Value>(arguments),
            Ok(Value::Object(_))
        ));
        assert!(validate_request(&request).is_ok());
    }
}

fn lookup_tool() -> ChatTool {
    ChatTool {
        tool_type: "function".to_owned(),
        function: ChatFunction {
            name: "lookup".to_owned(),
            description: Some("Look up a value.".to_owned()),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            })),
            strict: None,
        },
    }
}

fn blocked_real_full_info() -> RealFullInfo {
    RealFullInfo {
        status: "blocked".to_owned(),
        model_id: DEFAULT_MODEL_ID.to_owned(),
        model_facts: ds41rt_core::ModelFacts::default(),
        snapshot_path: None,
        catalog_hash: "cataloghash".to_owned(),
        quantization_recipe: "test-quantization-recipe".to_owned(),
        tensor_count: 234_689,
        startup_diagnostic_mode: "preflight-report".to_owned(),
        coordinator_resident_preload_status: "loaded".to_owned(),
        coordinator_resident_preload_selected_tensors: 1_024,
        coordinator_resident_preload_selected_bytes: 38_000_000_000,
        coordinator_resident_preload_loaded_bytes: 38_000_000_000,
        layer_count: 78,
        dense_layer_count: 3,
        sparse_layer_count: 75,
        kv_layout: "glm52-compressed-bf16".to_owned(),
        kv_bytes_per_token: 95_232,
        request_prefill_tokens: 4,
        request_prefill_chunks: 1,
        request_kv_snapshot_restore_ms: 0.0,
        request_decode_budget: 8,
        request_mtp_verify_rows: 4,
        request_mtp_accepted_rows: 4,
        request_dspark_joint_batch_width: 0,
        request_dspark_joint_cohort_width: 0,
        request_dspark_joint_cohort_count: 0,
        request_dspark_2x2_wavefront: false,
        request_coordinator_graph_slots: 0,
        request_coordinator_graph_captured_graphs: 0,
        request_coordinator_graph_captures: 0,
        request_coordinator_graph_launches: 0,
        request_candidate_layerwaves: 780,
        request_deferred_layerwaves: 0,
        scheduler_iterations: 234,
        selected_layerwaves: 312,
        sparse_expert_batches: 225,
        request_expert_batch_rows: 1_200,
        request_expert_batch_routes: 9_600,
        request_expert_prefill_rows: 300,
        request_expert_decode_rows: 600,
        request_expert_mtp_verify_rows: 300,
        request_expert_prefill_routes: 2_400,
        request_expert_decode_routes: 4_800,
        request_expert_mtp_verify_routes: 2_400,
        kv_read_blocks: 390,
        committed_kv_writes: 234,
        tentative_kv_writes: 624,
        request_committed_mtp_writes: 156,
        request_discarded_mtp_writes: 156,
        request_backed_kv_writes: 858,
        request_backed_kv_bytes: 1_333_248,
        request_kv_reservation_bytes: 1_523_712,
        request_byte_backed_scheduler_trace: true,
        scheduler_numeric_progression_passed: true,
        scheduler_numeric_progression_source_rows: 1_033,
        scheduler_numeric_progression_hidden_dim: 4,
        scheduler_numeric_progression_visible_checksum: 482_274.0,
        scheduler_numeric_progression_rejected_mtp_checksum: 2_808.0,
        request_numeric_progression_selected_prefill_rows: 312,
        request_numeric_progression_selected_decode_rows: 624,
        request_numeric_progression_selected_mtp_rows: 312,
        request_numeric_progression_attention_value_updates: 4_992,
        request_numeric_progression_mlp_value_updates: 4_992,
        scheduler_full_context_device_attention_complete: false,
        scheduler_terminal_lm_head_sample_status: "blocked".to_owned(),
        scheduler_terminal_lm_head_sample_passed: false,
        scheduler_terminal_lm_head_uses_final_decode_device_hidden: true,
        scheduler_terminal_lm_head_covers_full_vocabulary: false,
        scheduler_terminal_lm_head_logits_evaluated: 1_024,
        scheduler_terminal_lm_head_vocab_size: 151_552,
        scheduler_terminal_lm_head_top_token_id: Some(21),
        scheduler_terminal_lm_head_sampled_token_id: Some(42),
        scheduler_terminal_lm_head_sampled_text: None,
        scheduler_terminal_lm_head_sample_top_k: Some(8),
        scheduler_terminal_lm_head_sample_top_p: Some(0.95),
        scheduler_terminal_lm_head_argmax_backend: Some(
            "cpu-reference-lm-head-argmax-bf16".to_owned(),
        ),
        scheduler_terminal_lm_head_sampler_backend: Some(
            "cpu-reference-lm-head-sample-topk-topp-bf16".to_owned(),
        ),
        scheduler_terminal_lm_head_blocker: Some(
            "scheduler terminal lm_head sample must use full-vocabulary preloaded-resident CUDA argmax plus non-greedy top-k/top-p sampler"
                .to_owned(),
        ),
        protocol: "ExpertProtocolV2".to_owned(),
        decode_wire_request_bytes_per_touched_host: 12_444,
        decode_wire_response_bytes_per_touched_host: 12_384,
        prefill_wire_request_bytes_per_touched_host: 6_322_272,
        prefill_wire_response_bytes_per_touched_host: 6_291_552,
        mtp_wire_request_bytes_per_touched_host: 98_880,
        mtp_wire_response_bytes_per_touched_host: 98_400,
        decode_full_sparse_roundtrip_wire_bytes: 7_448_400,
        prefill_full_sparse_roundtrip_wire_bytes: 3_784_147_200,
        mtp_full_sparse_roundtrip_wire_bytes: 59_184_000,
        scheduler_sparse_tcp_dispatch_status: "not-configured".to_owned(),
        scheduler_sparse_tcp_dispatch_targets: 0,
        scheduler_sparse_tcp_dispatch_sparse_layers: 0,
        scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer: 0,
        scheduler_sparse_tcp_dispatch_batches: 0,
        scheduler_sparse_tcp_dispatch_host_batches: 0,
        scheduler_sparse_tcp_dispatch_global_rows: 0,
        scheduler_sparse_tcp_dispatch_host_rows: 0,
        scheduler_sparse_tcp_dispatch_routes: 0,
        scheduler_sparse_tcp_dispatch_request_wire_bytes: 0,
        scheduler_sparse_tcp_dispatch_response_wire_bytes: 0,
        scheduler_sparse_tcp_dispatch_output_values: 0,
        scheduler_sparse_tcp_dispatch_output_finite_values: 0,
        scheduler_sparse_tcp_dispatch_output_nonzero_values: 0,
        scheduler_sparse_tcp_dispatch_output_checksum: 0.0,
        scheduler_sparse_tcp_dispatch_passed: false,
        scheduler_sparse_tcp_dispatch_expected_real_executor_id: 98_765,
        scheduler_sparse_tcp_dispatch_response_executor_ids_observed: 3,
        scheduler_sparse_tcp_dispatch_real_executor_responses: 2,
        scheduler_sparse_tcp_dispatch_non_real_executor_responses: 1,
        scheduler_sparse_tcp_dispatch_all_responses_real_checkpoint_experts: false,
        scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4: false,
        scheduler_sparse_tcp_dispatch_consumed_by_residual: false,
        sampling_default_lm_head_chunk_passed: true,
        sampling_default_lm_head_chunk_rows_scored: 1_024,
        sampling_default_lm_head_chunk_lm_head_bytes_read: 12_582_912,
        sampling_default_lm_head_chunk_top_token_id: Some(21),
        sampling_default_lm_head_chunk_top_logit: Some(0.320_822_24),
        sampling_default_lm_head_chunk_uses_real_dense_prefix: true,
        sampling_default_lm_head_chunk_residual_source_dense_layers: 3,
        sampling_default_lm_head_chunk_residual_source_dense_weight_bytes_read: 884_736,
        sampling_default_lm_head_chunk_residual_after_checksum: Some(-0.132_872_547_954_320_9),
        blocker: "real-ds4-full is not runnable yet".to_owned(),
        failed_requirements: vec![
            "full_residual_stream_execution".to_owned(),
            "full_vocab_sampling".to_owned(),
        ],
    }
}

#[test]
fn thread_pinned_real_full_executor_reuses_one_worker_thread() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let executor = ThreadPinnedRealFullRequestExecutor::spawn(
        "ds41rt-api-thread-pinned-real-full-test",
        ThreadRecordingRealFullExecutor {
            calls: Arc::clone(&calls),
            finishes: Arc::clone(&finishes),
            info: blocked_real_full_info(),
        },
    )
    .expect("spawning thread-pinned real-full executor");
    let main_thread = thread::current().id();

    for request_index in 1..=3 {
        executor
            .execute_real_full_request(RealFullRequest::new(request_index, "prompt", 1, 1))
            .expect("executing request on pinned worker");
    }
    executor
        .finish_real_full_sequence("finished-sequence")
        .expect("finishing sequence on pinned worker");

    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0].0, 1);
    assert_eq!(calls[1].0, 2);
    assert_eq!(calls[2].0, 3);
    assert_ne!(calls[0].1, main_thread);
    assert_eq!(calls[1].1, calls[0].1);
    assert_eq!(calls[2].1, calls[0].1);
    let finishes = finishes.lock().unwrap();
    assert_eq!(
        finishes.as_slice(),
        &[("finished-sequence".to_owned(), calls[0].1)]
    );
}

#[test]
fn thread_pinned_real_full_executor_rejects_partial_cpu_assignment() {
    let result = ThreadPinnedRealFullRequestExecutor::spawn_pool_with_cpu_affinity(
        "ds41rt-api-thread-pinned-real-full-affinity-test",
        BatchRecordingRealFullExecutor {
            batches: Arc::new(Mutex::new(Vec::new())),
            info: blocked_real_full_info(),
        },
        2,
        &[0],
    );
    let error = match result {
        Ok(_) => panic!("partial CPU assignment must fail before spawning workers"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert!(error
        .to_string()
        .contains("1 CPU assignments for 2 workers"));
}

#[test]
fn thread_pinned_real_full_executor_surfaces_affinity_startup_failure() {
    let result = ThreadPinnedRealFullRequestExecutor::spawn_pool_with_cpu_affinity(
        "ds41rt-api-thread-pinned-real-full-invalid-cpu-test",
        BatchRecordingRealFullExecutor {
            batches: Arc::new(Mutex::new(Vec::new())),
            info: blocked_real_full_info(),
        },
        1,
        &[usize::MAX],
    );
    let error = match result {
        Ok(_) => panic!("invalid CPU assignment must fail worker startup"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    assert!(error.to_string().contains("CPU index"));
}

#[test]
fn thread_pinned_real_full_executor_coalesces_two_decode_cycles() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let executor = Arc::new(
        ThreadPinnedRealFullRequestExecutor::spawn(
            "ds41rt-api-thread-pinned-real-full-batch-test",
            BatchRecordingRealFullExecutor {
                batches: Arc::clone(&batches),
                info: blocked_real_full_info(),
            },
        )
        .expect("spawning batched thread-pinned real-full executor"),
    );
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let handles = ["sequence-b", "sequence-a"].map(|sequence_id| {
        let executor = Arc::clone(&executor);
        let barrier = Arc::clone(&barrier);
        let sequence_id = sequence_id.to_owned();
        thread::spawn(move || {
            barrier.wait();
            executor
                .execute_real_full_decode_cycle_on_worker(
                    0,
                    RealFullRequest::new_decode_step_for_sequence(
                        1,
                        sequence_id,
                        "prompt",
                        1,
                        1,
                        Vec::new(),
                        0,
                        1,
                    ),
                )
                .expect("executing coalesced decode cycle")
        })
    });
    for handle in handles {
        handle.join().expect("joining coalesced request");
    }

    assert_eq!(
        batches.lock().unwrap().as_slice(),
        &[vec!["sequence-a".to_owned(), "sequence-b".to_owned()]]
    );
}

#[test]
fn thread_pinned_real_full_executor_coalesces_four_decode_cycles() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let executor = Arc::new(
        ThreadPinnedRealFullRequestExecutor::spawn(
            "ds41rt-api-thread-pinned-real-full-wide-batch-test",
            BatchRecordingRealFullExecutor {
                batches: Arc::clone(&batches),
                info: blocked_real_full_info(),
            },
        )
        .expect("spawning wide-batched thread-pinned real-full executor"),
    );
    let barrier = Arc::new(std::sync::Barrier::new(4));
    let handles = ["sequence-d", "sequence-b", "sequence-c", "sequence-a"].map(|sequence_id| {
        let executor = Arc::clone(&executor);
        let barrier = Arc::clone(&barrier);
        let sequence_id = sequence_id.to_owned();
        thread::spawn(move || {
            barrier.wait();
            executor
                .execute_real_full_decode_cycle_on_worker(
                    0,
                    RealFullRequest::new_decode_step_for_sequence(
                        1,
                        sequence_id,
                        "prompt",
                        1,
                        1,
                        Vec::new(),
                        0,
                        1,
                    ),
                )
                .expect("executing wide-coalesced decode cycle")
        })
    });
    for handle in handles {
        handle.join().expect("joining wide-coalesced request");
    }

    assert_eq!(
        batches.lock().unwrap().as_slice(),
        &[vec![
            "sequence-a".to_owned(),
            "sequence-b".to_owned(),
            "sequence-c".to_owned(),
            "sequence-d".to_owned(),
        ]]
    );
}

#[test]
fn thread_pinned_real_full_releases_terminal_sequence_before_delivering_final_event() {
    let (finish_started, finish_started_receiver) = mpsc::sync_channel(1);
    let (release_finish, release_finish_receiver) = mpsc::sync_channel(1);
    let executor = ThreadPinnedRealFullRequestExecutor::spawn(
        "ds41rt-api-thread-pinned-real-full-terminal-finish-order-test",
        TerminalFinishBarrierRealFullExecutor {
            finish_started,
            release_finish: Mutex::new(release_finish_receiver),
            info: blocked_real_full_info(),
        },
    )
    .expect("spawning terminal-finish ordering executor");
    let mut receiver = executor
        .start_real_full_sequence_on_worker(
            0,
            RealFullSequenceRequest {
                request: RealFullRequest::new_decode_step_for_sequence(
                    100,
                    "terminal-sequence",
                    "prompt",
                    1,
                    1,
                    Vec::new(),
                    0,
                    1,
                ),
                max_output_tokens: 1,
                min_output_tokens: 0,
                ignore_eos: false,
                stop_token_ids: Vec::new(),
                stop_texts: Vec::new(),
            },
        )
        .expect("starting terminal-finish ordering sequence");

    finish_started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("terminal sequence reached cleanup");
    assert!(
        matches!(
            receiver.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ),
        "the terminal event must not make the HTTP turn observable before scheduler cleanup"
    );
    release_finish
        .send(())
        .expect("releasing terminal sequence cleanup");
    let event = receiver
        .blocking_recv()
        .expect("terminal event after cleanup")
        .expect("terminal cycle succeeds");
    assert_eq!(event.cycle.generated_tokens[0].token_id, 1);
    assert!(receiver.blocking_recv().is_none());
}

#[test]
fn thread_pinned_real_full_executor_keeps_persistent_c4_sequences_in_one_recurrent_wave() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let executor = Arc::new(
        ThreadPinnedRealFullRequestExecutor::spawn(
            "ds41rt-api-thread-pinned-real-full-persistent-c4-test",
            PersistentBatchRecordingRealFullExecutor {
                batches: Arc::clone(&batches),
                finishes: Arc::clone(&finishes),
                info: blocked_real_full_info(),
            },
        )
        .expect("spawning persistent C4 real-full executor"),
    );
    let barrier = Arc::new(std::sync::Barrier::new(4));
    let handles = ["sequence-d", "sequence-b", "sequence-c", "sequence-a"].map(|sequence_id| {
        let executor = Arc::clone(&executor);
        let barrier = Arc::clone(&barrier);
        let sequence_id = sequence_id.to_owned();
        thread::spawn(move || {
            barrier.wait();
            executor
                .start_real_full_sequence_on_worker(
                    0,
                    RealFullSequenceRequest {
                        request: RealFullRequest::new_decode_step_for_sequence(
                            100,
                            sequence_id,
                            "prompt",
                            1,
                            1,
                            Vec::new(),
                            0,
                            2,
                        ),
                        max_output_tokens: 2,
                        min_output_tokens: 0,
                        ignore_eos: false,
                        stop_token_ids: vec![2],
                        stop_texts: Vec::new(),
                    },
                )
                .expect("starting persistent real-full sequence")
        })
    });
    let mut receivers = handles.map(|handle| handle.join().expect("joining sequence admission"));
    for receiver in &mut receivers {
        let mut tokens = Vec::new();
        while let Some(event) = receiver.blocking_recv() {
            let event = event.expect("persistent real-full sequence cycle");
            tokens.extend(
                event
                    .cycle
                    .generated_tokens
                    .into_iter()
                    .map(|token| token.token_id),
            );
        }
        assert_eq!(tokens, vec![101, 102]);
    }

    let batches = batches.lock().unwrap();
    let initial = batches
        .iter()
        .filter(|batch| {
            batch
                .iter()
                .any(|(_, generated, _decode_budget)| *generated == 0)
        })
        .collect::<Vec<_>>();
    assert_eq!(initial.len(), 1);
    assert_eq!(initial[0].len(), 4);
    let recurrent = batches
        .iter()
        .filter(|batch| {
            batch
                .iter()
                .any(|(_, generated, _decode_budget)| *generated > 0)
        })
        .collect::<Vec<_>>();
    assert_eq!(recurrent.len(), 1);
    assert_eq!(recurrent[0].len(), 4);
    assert!(recurrent[0]
        .iter()
        .all(|(_, generated, decode_budget)| *generated == 1 && *decode_budget == 2));

    let mut finishes = finishes.lock().unwrap().clone();
    finishes.sort();
    assert_eq!(
        finishes,
        vec![
            "sequence-a".to_owned(),
            "sequence-b".to_owned(),
            "sequence-c".to_owned(),
            "sequence-d".to_owned(),
        ]
    );
}

#[test]
fn thread_pinned_real_full_persistent_wave_drains_through_every_width_without_padding() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let executor = Arc::new(
        ThreadPinnedRealFullRequestExecutor::spawn(
            "ds41rt-api-thread-pinned-real-full-width-drain-test",
            PersistentBatchRecordingRealFullExecutor {
                batches: Arc::clone(&batches),
                finishes: Arc::clone(&finishes),
                info: blocked_real_full_info(),
            },
        )
        .expect("spawning persistent width-drain executor"),
    );
    let barrier = Arc::new(std::sync::Barrier::new(4));
    let handles = [
        ("sequence-a", 1usize),
        ("sequence-b", 2),
        ("sequence-c", 3),
        ("sequence-d", 4),
    ]
    .map(|(sequence_id, max_output_tokens)| {
        let executor = Arc::clone(&executor);
        let barrier = Arc::clone(&barrier);
        let sequence_id = sequence_id.to_owned();
        thread::spawn(move || {
            barrier.wait();
            let receiver = executor
                .start_real_full_sequence_on_worker(
                    0,
                    RealFullSequenceRequest {
                        request: RealFullRequest::new_decode_step_for_sequence(
                            100,
                            sequence_id,
                            "prompt",
                            1,
                            1,
                            Vec::new(),
                            0,
                            max_output_tokens,
                        ),
                        max_output_tokens,
                        min_output_tokens: 0,
                        ignore_eos: false,
                        stop_token_ids: Vec::new(),
                        stop_texts: Vec::new(),
                    },
                )
                .expect("starting width-drain sequence");
            (receiver, max_output_tokens)
        })
    });
    let mut receivers = handles.map(|handle| handle.join().expect("joining width-drain admission"));
    for (receiver, max_output_tokens) in &mut receivers {
        let mut tokens = Vec::new();
        while let Some(event) = receiver.blocking_recv() {
            tokens.extend(
                event
                    .expect("width-drain sequence succeeds")
                    .cycle
                    .generated_tokens
                    .into_iter()
                    .map(|token| token.token_id),
            );
        }
        assert_eq!(tokens.len(), *max_output_tokens);
    }

    let batches = batches.lock().unwrap();
    assert_eq!(
        batches.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![4, 3, 2, 1]
    );
    for batch in batches.iter() {
        assert!(batch
            .iter()
            .all(|(sequence_id, _, _)| sequence_id.starts_with("sequence-")));
    }
    let mut finishes = finishes.lock().unwrap().clone();
    finishes.sort();
    assert_eq!(
        finishes,
        vec![
            "sequence-a".to_owned(),
            "sequence-b".to_owned(),
            "sequence-c".to_owned(),
            "sequence-d".to_owned(),
        ]
    );
}

#[test]
fn thread_pinned_real_full_admits_new_members_at_a_complete_cycle_boundary() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let (first_cycle_started, first_cycle_started_receiver) = mpsc::sync_channel(1);
    let (release_first_cycle, release_first_cycle_receiver) = mpsc::sync_channel(1);
    let executor = ThreadPinnedRealFullRequestExecutor::spawn(
        "ds41rt-api-thread-pinned-real-full-boundary-join-test",
        BoundaryJoinRealFullExecutor {
            batches: Arc::clone(&batches),
            finishes: Arc::clone(&finishes),
            first_cycle_started,
            release_first_cycle: Mutex::new(release_first_cycle_receiver),
            held_once: AtomicBool::new(false),
            info: blocked_real_full_info(),
        },
    )
    .expect("spawning boundary-join executor");
    let request = |sequence_id: &str| RealFullSequenceRequest {
        request: RealFullRequest::new_decode_step_for_sequence(
            100,
            sequence_id,
            "prompt",
            1,
            1,
            Vec::new(),
            0,
            2,
        ),
        max_output_tokens: 2,
        min_output_tokens: 0,
        ignore_eos: false,
        stop_token_ids: Vec::new(),
        stop_texts: Vec::new(),
    };

    let mut receiver_a = executor
        .start_real_full_sequence_on_worker(0, request("sequence-a"))
        .expect("starting first boundary-join sequence");
    first_cycle_started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("first boundary-join cycle started");
    let mut receiver_b = executor
        .start_real_full_sequence_on_worker(0, request("sequence-b"))
        .expect("queueing second boundary-join sequence");
    let mut receiver_c = executor
        .start_real_full_sequence_on_worker(0, request("sequence-c"))
        .expect("queueing third boundary-join sequence");
    release_first_cycle
        .send(())
        .expect("releasing first boundary-join cycle");

    for receiver in [&mut receiver_a, &mut receiver_b, &mut receiver_c] {
        let mut tokens = Vec::new();
        while let Some(event) = receiver.blocking_recv() {
            tokens.extend(
                event
                    .expect("boundary-join sequence succeeds")
                    .cycle
                    .generated_tokens
                    .into_iter()
                    .map(|token| token.token_id),
            );
        }
        assert_eq!(tokens, vec![1, 2]);
    }

    let batches = batches.lock().unwrap();
    assert_eq!(
        batches.as_slice(),
        &[
            vec![("sequence-a".to_owned(), 0)],
            vec![("sequence-b".to_owned(), 0), ("sequence-c".to_owned(), 0),],
            vec![
                ("sequence-a".to_owned(), 1),
                ("sequence-b".to_owned(), 1),
                ("sequence-c".to_owned(), 1),
            ],
        ]
    );
    let mut finishes = finishes.lock().unwrap().clone();
    finishes.sort();
    assert_eq!(
        finishes,
        vec![
            "sequence-a".to_owned(),
            "sequence-b".to_owned(),
            "sequence-c".to_owned(),
        ]
    );
}

#[test]
fn thread_pinned_real_full_executor_retries_transient_initial_admission() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let executor = Arc::new(
        ThreadPinnedRealFullRequestExecutor::spawn(
            "ds41rt-api-thread-pinned-real-full-admission-retry-test",
            RetryableAdmissionRealFullExecutor {
                batches: Arc::clone(&batches),
                finishes: Arc::clone(&finishes),
                rejected_once: AtomicBool::new(false),
                info: blocked_real_full_info(),
            },
        )
        .expect("spawning admission-retry real-full executor"),
    );
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let handles = ["sequence-a", "sequence-b"].map(|sequence_id| {
        let executor = Arc::clone(&executor);
        let barrier = Arc::clone(&barrier);
        thread::spawn(move || {
            barrier.wait();
            executor
                .start_real_full_sequence_on_worker(
                    0,
                    RealFullSequenceRequest {
                        request: RealFullRequest::new_decode_step_for_sequence(
                            100,
                            sequence_id,
                            "prompt",
                            1,
                            1,
                            Vec::new(),
                            0,
                            2,
                        ),
                        max_output_tokens: 2,
                        min_output_tokens: 0,
                        ignore_eos: false,
                        stop_token_ids: vec![2],
                        stop_texts: Vec::new(),
                    },
                )
                .expect("starting admission-retry sequence")
        })
    });
    let mut receivers = handles.map(|handle| handle.join().expect("joining sequence admission"));
    for receiver in &mut receivers {
        let mut tokens = Vec::new();
        while let Some(event) = receiver.blocking_recv() {
            let event = event.expect("transient admission must be retried internally");
            tokens.extend(
                event
                    .cycle
                    .generated_tokens
                    .into_iter()
                    .map(|token| token.token_id),
            );
        }
        assert_eq!(tokens, vec![1, 2]);
    }

    let batches = batches.lock().unwrap();
    assert_eq!(
        batches.first().map(Vec::as_slice),
        Some([("sequence-a".to_owned(), 0), ("sequence-b".to_owned(), 0),].as_slice())
    );
    assert!(batches
        .iter()
        .any(|batch| batch == &[("sequence-b".to_owned(), 0)]));
    let mut finishes = finishes.lock().unwrap().clone();
    finishes.sort();
    assert_eq!(
        finishes,
        vec!["sequence-a".to_owned(), "sequence-b".to_owned()]
    );
}

#[test]
fn thread_pinned_real_full_sequence_honors_minimum_output_before_stop_tokens() {
    let executor = ThreadPinnedRealFullRequestExecutor::spawn(
        "ds41rt-api-thread-pinned-real-full-min-output-test",
        RetryableAdmissionRealFullExecutor {
            batches: Arc::new(Mutex::new(Vec::new())),
            finishes: Arc::new(Mutex::new(Vec::new())),
            rejected_once: AtomicBool::new(true),
            info: blocked_real_full_info(),
        },
    )
    .expect("spawning minimum-output real-full executor");
    let mut receiver = executor
        .start_real_full_sequence_on_worker(
            0,
            RealFullSequenceRequest {
                request: RealFullRequest::new_decode_step_for_sequence(
                    100,
                    "sequence-a",
                    "prompt",
                    1,
                    1,
                    Vec::new(),
                    0,
                    3,
                ),
                max_output_tokens: 3,
                min_output_tokens: 3,
                ignore_eos: false,
                stop_token_ids: vec![2],
                stop_texts: Vec::new(),
            },
        )
        .expect("starting minimum-output sequence");

    let mut tokens = Vec::new();
    while let Some(event) = receiver.blocking_recv() {
        tokens.extend(
            event
                .expect("minimum-output sequence cycle")
                .cycle
                .generated_tokens
                .into_iter()
                .map(|token| token.token_id),
        );
    }
    assert_eq!(tokens, vec![1, 2, 2]);
}

#[test]
fn thread_pinned_real_full_queue_caps_total_active_and_pending_sequences() {
    let (first_cycle_started, first_cycle_started_receiver) = mpsc::sync_channel(1);
    let (release_first_cycle, release_first_cycle_receiver) = mpsc::sync_channel(1);
    let executor = ThreadPinnedRealFullRequestExecutor::spawn(
        "ds41rt-api-thread-pinned-real-full-concurrency-limit-test",
        ConcurrencyLimitedRealFullExecutor {
            first_cycle_started,
            release_first_cycle: Mutex::new(release_first_cycle_receiver),
            held_once: AtomicBool::new(false),
            info: blocked_real_full_info(),
        },
    )
    .expect("spawning concurrency-limited real-full executor");
    let request = |sequence_id: &str| RealFullSequenceRequest {
        request: RealFullRequest::new_decode_step_for_sequence(
            100,
            sequence_id,
            "prompt",
            1,
            1,
            Vec::new(),
            0,
            2,
        ),
        max_output_tokens: 2,
        min_output_tokens: 0,
        ignore_eos: false,
        stop_token_ids: vec![2],
        stop_texts: Vec::new(),
    };
    let first_receiver = executor
        .start_real_full_sequence_on_worker(0, request("active-sequence"))
        .expect("starting active sequence");
    first_cycle_started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("first sequence reached execution");
    let mut overflow_receiver = executor
        .start_real_full_sequence_on_worker(0, request("overflow-sequence"))
        .expect("submitting overflow sequence");
    release_first_cycle
        .send(())
        .expect("releasing active sequence execution");

    let error = overflow_receiver
        .blocking_recv()
        .expect("overflow error event")
        .expect_err("sequence above the configured total must be rejected");
    assert!(error.contains("active=1 queued=0 max=1"));
    drop(first_receiver);
}

#[test]
fn thread_pinned_real_full_capacity_retry_does_not_block_a_fitting_request() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let (first_large_attempted, first_large_attempted_receiver) = mpsc::sync_channel(1);
    let (release_first_large_attempt, release_first_large_attempt_receiver) = mpsc::sync_channel(1);
    let executor = ThreadPinnedRealFullRequestExecutor::spawn(
        "ds41rt-api-thread-pinned-real-full-head-of-line-test",
        HeadOfLineAdmissionRealFullExecutor {
            batches: Arc::clone(&batches),
            finishes: Arc::clone(&finishes),
            first_large_attempted,
            release_first_large_attempt: Mutex::new(release_first_large_attempt_receiver),
            held_first_large_attempt: AtomicBool::new(false),
            small_admitted: AtomicBool::new(false),
            info: blocked_real_full_info(),
        },
    )
    .expect("spawning head-of-line admission executor");
    let request = |sequence_id: &str| RealFullSequenceRequest {
        request: RealFullRequest::new_decode_step_for_sequence(
            100,
            sequence_id,
            "prompt",
            1,
            1,
            Vec::new(),
            0,
            2,
        ),
        max_output_tokens: 2,
        min_output_tokens: 0,
        ignore_eos: false,
        stop_token_ids: vec![2],
        stop_texts: Vec::new(),
    };

    let mut large_receiver = executor
        .start_real_full_sequence_on_worker(0, request("large-sequence"))
        .expect("starting capacity-blocked large sequence");
    first_large_attempted_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("large sequence reached initial admission");
    let mut small_receiver = executor
        .start_real_full_sequence_on_worker(0, request("small-sequence"))
        .expect("starting fitting small sequence");
    release_first_large_attempt
        .send(())
        .expect("releasing blocked large admission");

    let mut small_tokens = Vec::new();
    while let Some(event) = small_receiver.blocking_recv() {
        small_tokens.extend(
            event
                .expect("small sequence must bypass the blocked large sequence")
                .cycle
                .generated_tokens
                .into_iter()
                .map(|token| token.token_id),
        );
    }
    assert_eq!(small_tokens, vec![1, 2]);
    let mut large_tokens = Vec::new();
    while let Some(event) = large_receiver.blocking_recv() {
        large_tokens.extend(
            event
                .expect("large sequence must retry after capacity becomes available")
                .cycle
                .generated_tokens
                .into_iter()
                .map(|token| token.token_id),
        );
    }
    assert_eq!(large_tokens, vec![1, 2]);

    let batches = batches.lock().unwrap();
    let small_initial = batches
        .iter()
        .position(|batch| batch == &("small-sequence".to_owned(), 0))
        .expect("small sequence was admitted");
    let large_success = batches
        .iter()
        .rposition(|batch| batch == &("large-sequence".to_owned(), 0))
        .expect("large sequence was retried");
    assert!(
        small_initial < large_success,
        "the fitting small request must run before the blocked large retry succeeds"
    );
    assert!(
        batches[..small_initial]
            .iter()
            .filter(|batch| batch == &&("large-sequence".to_owned(), 0))
            .count()
            >= 2,
        "the blocked request must rotate through the pending tail"
    );
    let mut finishes = finishes.lock().unwrap().clone();
    finishes.sort();
    assert_eq!(
        finishes,
        vec!["large-sequence".to_owned(), "small-sequence".to_owned()]
    );
}

#[test]
fn thread_pinned_real_full_prunes_a_cancelled_pending_request() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let (first_cycle_started, first_cycle_started_receiver) = mpsc::sync_channel(1);
    let (release_first_cycle, release_first_cycle_receiver) = mpsc::sync_channel(1);
    let executor = ThreadPinnedRealFullRequestExecutor::spawn(
        "ds41rt-api-thread-pinned-real-full-pending-cancel-test",
        PendingCancellationRealFullExecutor {
            batches: Arc::clone(&batches),
            finishes: Arc::clone(&finishes),
            first_cycle_started,
            release_first_cycle: Mutex::new(release_first_cycle_receiver),
            held_once: AtomicBool::new(false),
            info: blocked_real_full_info(),
        },
    )
    .expect("spawning pending-cancellation executor");
    let request = |sequence_id: &str| RealFullSequenceRequest {
        request: RealFullRequest::new_decode_step_for_sequence(
            100,
            sequence_id,
            "prompt",
            1,
            1,
            Vec::new(),
            0,
            2,
        ),
        max_output_tokens: 2,
        min_output_tokens: 0,
        ignore_eos: false,
        stop_token_ids: vec![2],
        stop_texts: Vec::new(),
    };

    let mut active_receiver = executor
        .start_real_full_sequence_on_worker(0, request("active-sequence"))
        .expect("starting active sequence");
    first_cycle_started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("active sequence reached execution");
    let cancelled_receiver = executor
        .start_real_full_sequence_on_worker(0, request("cancelled-sequence"))
        .expect("queueing sequence that will be cancelled");
    drop(cancelled_receiver);
    release_first_cycle
        .send(())
        .expect("releasing active sequence execution");
    while let Some(event) = active_receiver.blocking_recv() {
        event.expect("active sequence succeeds");
    }

    let mut probe_receiver = executor
        .start_real_full_sequence_on_worker(0, request("probe-sequence"))
        .expect("starting post-cancellation probe");
    while let Some(event) = probe_receiver.blocking_recv() {
        event.expect("probe sequence succeeds after pruning cancellation");
    }

    let batches = batches.lock().unwrap();
    assert!(
        batches
            .iter()
            .all(|(sequence_id, _)| sequence_id != "cancelled-sequence"),
        "a closed pending receiver must be pruned before execution"
    );
    let mut finishes = finishes.lock().unwrap().clone();
    finishes.sort();
    assert_eq!(
        finishes,
        vec!["active-sequence".to_owned(), "probe-sequence".to_owned()]
    );
}

#[test]
fn thread_pinned_real_full_blocked_admission_cannot_starve_the_active_owner() {
    let batches = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let (first_cycle_started, first_cycle_started_receiver) = mpsc::sync_channel(1);
    let (release_first_cycle, release_first_cycle_receiver) = mpsc::sync_channel(1);
    let executor = ThreadPinnedRealFullRequestExecutor::spawn(
        "ds41rt-api-thread-pinned-real-full-active-owner-test",
        ActiveOwnerAdmissionRealFullExecutor {
            batches: Arc::clone(&batches),
            finishes: Arc::clone(&finishes),
            first_cycle_started,
            release_first_cycle: Mutex::new(release_first_cycle_receiver),
            held_once: AtomicBool::new(false),
            active_finished: AtomicBool::new(false),
            info: blocked_real_full_info(),
        },
    )
    .expect("spawning active-owner admission executor");
    let request = |sequence_id: &str| RealFullSequenceRequest {
        request: RealFullRequest::new_decode_step_for_sequence(
            100,
            sequence_id,
            "prompt",
            1,
            1,
            Vec::new(),
            0,
            2,
        ),
        max_output_tokens: 2,
        min_output_tokens: 0,
        ignore_eos: false,
        stop_token_ids: vec![2],
        stop_texts: Vec::new(),
    };

    let active_receiver = executor
        .start_real_full_sequence_on_worker(0, request("active-sequence"))
        .expect("starting active capacity owner");
    first_cycle_started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("active owner reached execution");
    let blocked_receiver = executor
        .start_real_full_sequence_on_worker(0, request("blocked-sequence"))
        .expect("queueing capacity-blocked sequence");
    release_first_cycle
        .send(())
        .expect("releasing active owner execution");

    let drain = |mut receiver: tokio::sync::mpsc::UnboundedReceiver<
        Result<RealFullSequenceCycle, String>,
    >| {
        let (tokens_sender, tokens_receiver) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let mut tokens = Vec::new();
            while let Some(event) = receiver.blocking_recv() {
                tokens.extend(
                    event
                        .expect("active-owner test sequence succeeds")
                        .cycle
                        .generated_tokens
                        .into_iter()
                        .map(|token| token.token_id),
                );
            }
            let _ = tokens_sender.send(tokens);
        });
        tokens_receiver
    };
    let active_tokens = drain(active_receiver)
        .recv_timeout(Duration::from_secs(1))
        .expect("blocked admission must not starve the active owner");
    let blocked_tokens = drain(blocked_receiver)
        .recv_timeout(Duration::from_secs(1))
        .expect("blocked request must run after the active owner releases capacity");
    assert_eq!(active_tokens, vec![1, 2]);
    assert_eq!(blocked_tokens, vec![1, 2]);

    let batches = batches.lock().unwrap();
    let active_recurrent = batches
        .iter()
        .position(|batch| batch == &("active-sequence".to_owned(), 1))
        .expect("active owner reached its recurrent cycle");
    let blocked_success = batches
        .iter()
        .rposition(|batch| batch == &("blocked-sequence".to_owned(), 0))
        .expect("blocked request was retried");
    assert!(active_recurrent < blocked_success);
    let mut finishes = finishes.lock().unwrap().clone();
    finishes.sort();
    assert_eq!(
        finishes,
        vec!["active-sequence".to_owned(), "blocked-sequence".to_owned()]
    );
}

fn blocked_runtime_sample_real_full_info(token_id: usize, text: &str) -> RealFullInfo {
    let mut full = blocked_real_full_info();
    full.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    full.request_decode_budget = 1;
    full.request_mtp_verify_rows = 1;
    full.request_mtp_accepted_rows = 1;
    full.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    full.scheduler_terminal_lm_head_sample_passed = true;
    full.scheduler_terminal_lm_head_uses_final_decode_device_hidden = true;
    full.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    full.scheduler_terminal_lm_head_logits_evaluated = full.scheduler_terminal_lm_head_vocab_size;
    full.scheduler_terminal_lm_head_top_token_id = Some(token_id);
    full.scheduler_terminal_lm_head_sampled_token_id = Some(token_id);
    full.scheduler_terminal_lm_head_sampled_text = Some(text.to_owned());
    full.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    full.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    full.scheduler_terminal_lm_head_blocker = None;
    full
}

fn ready_runtime_sample_real_full_info(token_id: usize, text: &str) -> RealFullInfo {
    let mut full = blocked_runtime_sample_real_full_info(token_id, text);
    full.status = "ready".to_owned();
    full.scheduler_full_context_device_attention_complete = true;
    full.blocker.clear();
    full.failed_requirements.clear();
    full
}

fn mock_tool_call_config(requests: Arc<Mutex<Vec<RealFullRequest>>>) -> ApiConfig {
    let output = "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"query\" string=\"true\">42</｜DSML｜parameter>\n<｜DSML｜parameter name=\"limit\" string=\"false\">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    let mut config = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc).config;
    config.real_full = Some(blocked_real_full_info());
    config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests,
        base: ready_runtime_sample_real_full_info(42, output),
        tokens: vec![(42, output.to_owned())],
    }));
    config
}

fn write_single_token_tokenizer(snapshot_path: &Path, prompt: &str) {
    let escaped_prompt = serde_json::to_string(prompt).expect("serializing tokenizer prompt");
    fs::write(
        snapshot_path.join("tokenizer.json"),
        format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{"[UNK]":0,{escaped_prompt}:7}},"unk_token":"[UNK]"}}}}"#
        ),
    )
    .expect("writing tokenizer fixture");
}

fn write_split_utf8_tokenizer(snapshot_path: &Path) {
    fs::write(
        snapshot_path.join("tokenizer.json"),
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"model":{"type":"WordLevel","vocab":{"[UNK]":0,"<unused>":1,"ðŁ":2,"¦":3,"ľ":4},"unk_token":"[UNK]"}}"#,
    )
    .expect("writing split-UTF-8 tokenizer fixture");
}

#[test]
fn real_full_prompt_text_uses_deepseek_v4_role_tokens() {
    let messages = vec![
        ChatMessage {
            role: "system".to_owned(),
            content: Some(Value::String("Be terse.".to_owned())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
            ..ChatMessage::default()
        },
        ChatMessage {
            role: "user".to_owned(),
            content: Some(Value::String("hi".to_owned())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
            ..ChatMessage::default()
        },
    ];

    let prompt = real_ds4_full_prompt_text(&messages);

    assert_eq!(
        prompt,
        "<｜begin▁of▁sentence｜>Be terse.<｜User｜>hi<｜Assistant｜></think>"
    );
}

#[test]
fn real_ds4_full_prompt_enables_and_round_trips_reasoning() {
    let mut request = base_request("Work it out.");
    request.thinking = Some(ThinkingConfig {
        thinking_type: "enabled".to_owned(),
        clear_thinking: Some(false),
    });
    assert!(real_ds4_full_request_prompt_text(&request).ends_with("<｜Assistant｜><think>"));

    request.messages.push(ChatMessage {
        role: "assistant".to_owned(),
        content: Some(Value::String("The answer.".to_owned())),
        name: None,
        tool_call_id: None,
        tool_calls: None,
        reasoning_content: Some("Private work.".to_owned()),
        ..ChatMessage::default()
    });
    request.messages.push(ChatMessage {
        role: "user".to_owned(),
        content: Some(Value::String("Continue.".to_owned())),
        name: None,
        tool_call_id: None,
        tool_calls: None,
        reasoning_content: None,
        ..ChatMessage::default()
    });
    let prompt = real_ds4_full_request_prompt_text(&request);
    assert!(!prompt.contains("Private work."));
    assert!(prompt
        .contains("<｜Assistant｜></think>The answer.<｜end▁of▁sentence｜><｜User｜>Continue."));
    assert!(prompt.ends_with("<｜Assistant｜><think>"));
}

#[test]
fn real_ds4_full_prompt_renders_tools_calls_and_results_in_native_format() {
    let mut request = base_request("Find the weather.");
    request.tools = Some(vec![lookup_tool()]);
    request.tool_choice = Some(ToolChoice::Mode("required".to_owned()));
    request.messages.push(ChatMessage {
        role: "assistant".to_owned(),
        content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Some(vec![ToolCall {
            id: "call_weather".to_owned(),
            tool_type: "function".to_owned(),
            function: ToolCallFunction {
                name: "lookup".to_owned(),
                arguments: r#"{"query":"Taipei","limit":2}"#.to_owned(),
            },
        }]),
        reasoning_content: None,
        ..ChatMessage::default()
    });
    request.messages.push(ChatMessage {
        role: "tool".to_owned(),
        content: Some(Value::String("sunny".to_owned())),
        name: None,
        tool_call_id: Some("call_weather".to_owned()),
        tool_calls: None,
        reasoning_content: None,
        ..ChatMessage::default()
    });

    validate_request(&request).unwrap();
    let prompt = real_ds4_full_request_prompt_text(&request);

    assert!(prompt.starts_with("<｜begin▁of▁sentence｜>## Tools\n"));
    assert!(prompt.contains(
        r#"{"name": "lookup", "description": "Look up a value.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}"#
    ));
    assert!(prompt.contains("You MUST invoke at least one available tool."));
    assert!(prompt.contains(
        "<｜Assistant｜></think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"query\" string=\"true\">Taipei</｜DSML｜parameter>\n<｜DSML｜parameter name=\"limit\" string=\"false\">2</｜DSML｜parameter>"
    ));
    assert!(prompt.contains("<｜User｜><tool_result>sunny</tool_result><｜Assistant｜></think>"));
}

#[test]
fn real_ds4_full_prompt_honors_none_and_named_tool_choices() {
    let mut request = base_request("Find a value.");
    let mut other = lookup_tool();
    other.function.name = "other".to_owned();
    request.tools = Some(vec![lookup_tool(), other]);
    request.tool_choice = Some(ToolChoice::Mode("none".to_owned()));

    let none_prompt = real_ds4_full_request_prompt_text(&request);
    assert!(!none_prompt.contains("# Tools"));
    assert!(!none_prompt.contains("\"name\": \"lookup\""));

    request.tool_choice = Some(ToolChoice::Specific {
        tool_type: "function".to_owned(),
        function: ToolChoiceFunction {
            name: "lookup".to_owned(),
        },
    });
    validate_request(&request).unwrap();
    let named_prompt = real_ds4_full_request_prompt_text(&request);
    assert!(named_prompt.contains("\"name\": \"lookup\""));
    assert!(!named_prompt.contains("\"name\": \"other\""));
    assert!(named_prompt.contains("You MUST invoke the tool named \"lookup\"."));
}

async fn request_json(method: Method, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let app = router();
    request_json_from_router(app, method, uri, body).await
}

async fn request_json_with_config(
    config: ApiConfig,
    method: Method,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let app = router_with_config(config);
    request_json_from_router(app, method, uri, body).await
}

async fn request_json_from_router(
    app: axum::Router,
    method: Method,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut builder = Request::builder().method(method).uri(uri);
    if body.is_some() {
        builder = builder.header("content-type", "application/json");
    }
    let request = builder
        .body(match body {
            Some(value) => Body::from(value.to_string()),
            None => Body::empty(),
        })
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let value = serde_json::from_slice(&bytes).unwrap();
    (status, value)
}

async fn request_text(method: Method, uri: &str, body: Value) -> (StatusCode, String) {
    let app = router();
    request_text_from_router(app, method, uri, body).await
}

async fn request_text_with_config(
    config: ApiConfig,
    method: Method,
    uri: &str,
    body: Value,
) -> (StatusCode, String) {
    let app = router_with_config(config);
    request_text_from_router(app, method, uri, body).await
}

async fn request_text_from_router(
    app: axum::Router,
    method: Method,
    uri: &str,
    body: Value,
) -> (StatusCode, String) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    (status, text)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disconnecting_non_stream_real_full_request_cancels_persistent_sequence() {
    let (first_cycle_started, first_cycle_started_receiver) = mpsc::sync_channel(1);
    let (release_first_cycle, release_first_cycle_receiver) = mpsc::sync_channel(1);
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let executor = ThreadPinnedRealFullRequestExecutor::spawn(
        "ds41rt-api-buffered-response-cancellation-test",
        BufferedResponseCancellationRealFullExecutor {
            first_cycle_started,
            release_first_cycle: Mutex::new(release_first_cycle_receiver),
            held_once: AtomicBool::new(false),
            finishes: Arc::clone(&finishes),
            info: blocked_runtime_sample_real_full_info(100, "a"),
        },
    )
    .expect("spawning buffered-response cancellation executor");
    let mut config = ApiConfig::default();
    config.backend = ApiBackend::RealDs4Full;
    config.real_full = Some(blocked_real_full_info());
    config.real_full_executor = Some(Arc::new(executor));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router_with_config(config))
            .await
            .unwrap();
    });
    let body = json!({
        "model": format!("{}-full", DEFAULT_MODEL_ID),
        "messages": [{"role": "user", "content": "Write a long answer."}],
        "max_tokens": 4096,
        "temperature": 0
    })
    .to_string();
    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    client
        .write_all(
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    first_cycle_started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("generation starts before the buffered response exists");
    let mut response_probe = [0_u8; 1];
    assert!(
        tokio::time::timeout(Duration::from_millis(25), client.read(&mut response_probe))
            .await
            .is_err(),
        "buffered non-streaming generation must not send headers or body bytes before completion"
    );

    drop(client);
    release_first_cycle
        .send(())
        .expect("releasing in-flight decode cycle after disconnect");

    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if !finishes.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("disconnected request releases its persistent execution lane");
    assert_eq!(finishes.lock().unwrap().len(), 1);
    server.abort();
}

mod routes;

#[tokio::test]
async fn non_streaming_tiny_completion_is_deterministic() {
    let state = test_state(ApiBackend::Tiny, ApiTransport::Inproc);
    let output = build_completion(&state, base_request("Say hello in five words."))
        .await
        .unwrap();
    assert_eq!(
        output.content.as_deref(),
        Some("hello from ds41rt tiny backend")
    );
    assert_eq!(output.finish_reason, "stop");
    assert_eq!(output.metrics.backend_mode, "tiny");
    assert_eq!(output.metrics.transport_backend, "inproc");
    assert!(output.metrics.prefill_chunk_count > 0);
    assert!(output.metrics.layerwave_prefill_rows > 0);
    assert_eq!(output.metrics.layerwave_decode_rows, 1);
    assert_eq!(output.metrics.prompt_tokens, output.usage.prompt_tokens);
    assert!(output.metrics.real_full.is_none());
}

#[tokio::test]
async fn stop_string_truncates_output() {
    let state = test_state(ApiBackend::Tiny, ApiTransport::Inproc);
    let mut request = base_request("Say hello.");
    request.stop = Some(StopSpec::One("tiny".to_owned()));
    let output = build_completion(&state, request).await.unwrap();
    assert_eq!(output.content.as_deref(), Some("hello from ds41rt "));
}

#[tokio::test]
async fn required_tool_choice_parses_mock_model_output() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config = mock_tool_call_config(Arc::clone(&requests));

    let mut request = base_request("Use the lookup tool.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    request.tools = Some(vec![lookup_tool()]);
    request.tool_choice = Some(ToolChoice::Mode("required".to_owned()));
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content, None);
    let tool_calls = output.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "lookup");
    assert_eq!(
        serde_json::from_str::<Value>(&tool_calls[0].function.arguments).unwrap(),
        json!({"query": "42", "limit": 3})
    );
    assert_eq!(output.finish_reason, "tool_calls");
    assert_eq!(output.usage.completion_tokens, 1);
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(captured[0].prompt.contains("# Tools"));
    assert!(captured[0].prompt.contains("\"name\": \"lookup\""));
}

#[tokio::test]
async fn non_streaming_tool_followup_reuses_exact_continuation_token_ids() {
    let snapshot = tempfile::tempdir().expect("creating continuation tokenizer snapshot");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    let mut full = blocked_real_full_info();
    full.snapshot_path = Some(snapshot.path().display().to_string());
    state.config.real_full = Some(full);
    state.config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base: ready_runtime_sample_real_full_info(42, "done"),
        tokens: vec![(42, "done".to_owned())],
    }));

    let mut request = base_request("Find the weather.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    request.tools = Some(vec![lookup_tool()]);
    request.messages.push(ChatMessage {
        role: "assistant".to_owned(),
        content: None,
        name: None,
        tool_call_id: None,
        tool_calls: Some(vec![ToolCall {
            id: "call_weather".to_owned(),
            tool_type: "function".to_owned(),
            function: ToolCallFunction {
                name: "lookup".to_owned(),
                arguments: r#"{"query":"Taipei","limit":2}"#.to_owned(),
            },
        }]),
        reasoning_content: None,
        ..ChatMessage::default()
    });
    request.messages.push(ChatMessage {
        role: "tool".to_owned(),
        content: Some(Value::String("sunny".to_owned())),
        name: None,
        tool_call_id: Some("call_weather".to_owned()),
        tool_calls: None,
        reasoning_content: None,
        ..ChatMessage::default()
    });

    let prompt = real_ds4_full_request_prompt_text(&request);
    let suffix_offset = prompt
        .find("<｜User｜><tool_result>")
        .expect("tool result suffix in canonical prompt");
    let (prefix, suffix) = prompt.split_at(suffix_offset);
    write_single_token_tokenizer(snapshot.path(), suffix);
    state.tool_continuations.lock().unwrap().insert(
        vec!["call_weather".to_owned()],
        prefix.to_owned(),
        Arc::new(vec![11, 12]),
    );

    build_completion(&state, request)
        .await
        .expect("non-streaming continuation completion");

    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].prompt_tokens, 3);
    assert_eq!(
        captured[0]
            .prompt_token_ids
            .as_ref()
            .map(|token_ids| token_ids.as_slice()),
        Some([11, 12, 7].as_slice())
    );
}

#[tokio::test]
async fn real_ds4_full_decode_preserves_fragmented_tool_markers() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    state.config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base: ready_runtime_sample_real_full_info(42, "\n\n<｜DSML｜tool_calls>\n"),
        tokens: vec![
            (42, "\n\n<｜DSML｜tool_calls>\n".to_owned()),
            (43, "<｜DSML｜invoke name=\"lookup\">\n".to_owned()),
            (
                44,
                "<｜DSML｜parameter name=\"query\" string=\"true\">Taipei</｜DSML｜parameter>"
                    .to_owned(),
            ),
            (45, "\n</｜DSML｜invoke>".to_owned()),
            (46, "\n</｜DSML｜tool_calls>".to_owned()),
        ],
    }));

    let mut request = base_request("Use lookup.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(5);
    request.tools = Some(vec![lookup_tool()]);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.finish_reason, "tool_calls");
    assert_eq!(output.content, None);
    let tool_calls = output.tool_calls.unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "lookup");
    assert_eq!(
        serde_json::from_str::<Value>(&tool_calls[0].function.arguments).unwrap(),
        json!({"query": "Taipei"})
    );
    assert_eq!(requests.lock().unwrap().len(), 5);
}

#[tokio::test]
async fn real_ds4_full_route_serializes_mock_tool_call() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let (status, body) = request_json_with_config(
        mock_tool_call_config(Arc::clone(&requests)),
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": format!("{}-full", DEFAULT_MODEL_ID),
            "messages": [{"role": "user", "content": "Use lookup."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"}
                        }
                    }
                }
            }],
            "tool_choice": "required",
            "max_tokens": 1
        })),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(body["choices"][0]["message"]["content"].is_null());
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    let call = &body["choices"][0]["message"]["tool_calls"][0];
    assert!(call["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "lookup");
    assert_eq!(
        serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(),
        json!({"query": "42", "limit": 3})
    );
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn real_ds4_full_route_streams_mock_tool_call_delta() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let (status, text) = request_text_with_config(
        mock_tool_call_config(Arc::clone(&requests)),
        Method::POST,
        "/v1/chat/completions",
        json!({
            "model": format!("{}-full", DEFAULT_MODEL_ID),
            "stream": true,
            "messages": [{"role": "user", "content": "Use lookup."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"}
                        }
                    }
                }
            }],
            "tool_choice": "required",
            "max_tokens": 1
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(text.contains("\"role\":\"assistant\""));
    assert!(text.contains("\"tool_calls\":[{\"index\":0"));
    assert!(text.contains("\"id\":\"call_"));
    assert!(text.contains("\"name\":\"lookup\""));
    let arguments = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| serde_json::from_str::<Value>(payload).unwrap())
        .filter_map(|frame| {
            frame["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .map(str::to_owned)
        })
        .collect::<String>();
    assert_eq!(
        serde_json::from_str::<Value>(&arguments).unwrap(),
        json!({"query": "42", "limit": 3})
    );
    assert!(text.contains("\"finish_reason\":\"tool_calls\""));
    assert!(!text.contains("<｜DSML｜tool_calls>"));
    assert!(text.contains("[DONE]"));
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn real_ds4_full_route_streams_fragmented_tool_call_as_decode_steps_arrive() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut base = ready_runtime_sample_real_full_info(42, "unused");
    base.request_decode_budget = 1;
    let mut config = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc).config;
    config.real_full = Some(blocked_real_full_info());
    config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base,
        tokens: vec![
            (10, "Checking. ".to_owned()),
            (11, "\n\n<｜DSML｜tool_".to_owned()),
            (12, "calls>\n<｜DSML｜invoke name=\"look".to_owned()),
            (
                13,
                "up\">\n<｜DSML｜parameter name=\"query\" string=\"true\">".to_owned(),
            ),
            (14, "Tai".to_owned()),
            (15, "pei</｜DSML｜parameter>".to_owned()),
            (16, "\n</｜DSML｜invoke>".to_owned()),
            (17, "\n</｜DSML｜tool_calls>".to_owned()),
        ],
    }));

    let (status, text) = request_text_with_config(
        config,
        Method::POST,
        "/v1/chat/completions",
        json!({
            "model": format!("{}-full", DEFAULT_MODEL_ID),
            "stream": true,
            "messages": [{"role": "user", "content": "Use lookup."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}}
                    }
                }
            }],
            "tool_choice": "required",
            "max_tokens": 8
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let mut content = String::new();
    let mut name = String::new();
    let mut arguments = String::new();
    let mut tool_frame_count = 0;
    for payload in text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|payload| *payload != "[DONE]")
    {
        let frame: Value = serde_json::from_str(payload).unwrap();
        let delta = &frame["choices"][0]["delta"];
        if let Some(chunk) = delta["content"].as_str() {
            content.push_str(chunk);
        }
        if let Some(tool_calls) = delta["tool_calls"].as_array() {
            tool_frame_count += 1;
            let function = &tool_calls[0]["function"];
            if let Some(chunk) = function["name"].as_str() {
                name.push_str(chunk);
            }
            if let Some(chunk) = function["arguments"].as_str() {
                arguments.push_str(chunk);
            }
        }
    }

    assert_eq!(content, "Checking. ");
    assert_eq!(name, "lookup");
    assert_eq!(
        serde_json::from_str::<Value>(&arguments).unwrap(),
        json!({"query": "Taipei"})
    );
    assert_eq!(tool_frame_count, 1);
    assert!(text.contains("\"finish_reason\":\"tool_calls\""));
    assert!(!text.contains("<｜DSML｜tool_calls>"));
    assert!(!text.contains("<｜DSML｜parameter"));
    assert!(text.contains("[DONE]"));
    assert_eq!(requests.lock().unwrap().len(), 8);
}

#[tokio::test]
async fn synthetic_ds4_layer_inproc_sums_expert_partials() {
    let state = test_state(ApiBackend::SyntheticDs4Layer, ApiTransport::Inproc);
    let mut request = base_request("Run synthetic layer.");
    request.model = "ds41rt-synthetic-ds4-layer".to_owned();
    let output = build_completion(&state, request).await.unwrap();
    let content = output.content.unwrap();
    assert!(content.contains("synthetic ds4 layer ok"));
    assert!(content.contains("hidden=4096"));
    assert!(content.contains("top_k=6"));
    assert!(content.contains("prefill_chunks=1"));
    assert!(content.contains("prefill_rows=4"));
    assert!(content.contains("decode_rows=1"));
    assert!(content.contains("decode_partials=1"));
    assert_eq!(output.metrics.backend_mode, "synthetic-ds4-layer");
    assert_eq!(output.metrics.transport_backend, "inproc");
    assert_eq!(output.metrics.prefill_chunk_count, 1);
    assert_eq!(output.metrics.layerwave_prefill_rows, 4);
    assert_eq!(output.metrics.layerwave_decode_rows, 1);
    assert!(output.metrics.prefill_tokens_per_sec.is_some());
}

#[tokio::test]
async fn synthetic_ds4_layer_tcp_uses_protocol_v2_expert_service() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let _ = ds41rt_transport::serve_synthetic_protocol_v2_tcp_listener(listener).await;
    });
    let mut state = test_state(ApiBackend::SyntheticDs4Layer, ApiTransport::Tcp);
    state.config.expert_targets = vec![addr.to_string()];
    let mut request = base_request("Run synthetic layer over ProtocolV2 TCP.");
    request.model = "ds41rt-synthetic-ds4-layer".to_owned();

    let output = build_completion(&state, request).await.unwrap();
    server.abort();

    let content = output.content.unwrap();
    assert!(content.contains("synthetic ds4 layer ok"));
    assert!(content.contains("hidden=4096"));
    assert!(content.contains("prefill_rows="));
    assert!(content.contains("decode_rows=1"));
    assert!(content.contains("decode_partials=1"));
    assert_eq!(output.metrics.backend_mode, "synthetic-ds4-layer");
    assert_eq!(output.metrics.transport_backend, "tcp");
    assert_eq!(output.metrics.prefill_chunk_count, 1);
    assert!(output.metrics.layerwave_prefill_rows > 0);
    assert_eq!(output.metrics.layerwave_decode_rows, 1);
    assert!(output.metrics.prefill_tokens_per_sec.is_some());
}

#[tokio::test]
async fn real_ds4_full_backend_reports_blocked_preflight_status() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(8);
    let output = build_completion(&state, request).await.unwrap();
    let content = output.content.unwrap();
    assert!(content.contains("real ds4 full status=blocked"));
    assert!(content.contains(
        "startup_diagnostic_mode=preflight-report tensors=234689 coordinator_resident_preload_status=loaded"
    ));
    assert!(content.contains(
        "coordinator_resident_preload_selected_tensors=1024 coordinator_resident_preload_selected_bytes=38000000000 coordinator_resident_preload_loaded_bytes=38000000000"
    ));
    assert!(content.contains("layers=78 dense_layers=3 sparse_layers=75"));
    assert!(content.contains("kv_layout=glm52-compressed-bf16 kv_bytes_per_token=95232"));
    assert!(content.contains("scheduler_iterations=234 selected_layerwaves=312"));
    assert!(content.contains(
        "scheduler_numeric_progression_passed=true scheduler_numeric_progression_source_rows=1033"
    ));
    assert!(content.contains(
        "scheduler_numeric_progression_hidden_dim=4 scheduler_numeric_progression_visible_checksum=482274"
    ));
    assert!(content.contains("scheduler_numeric_progression_rejected_mtp_checksum=2808"));
    assert!(content.contains("scheduler_full_context_device_attention_complete=false"));
    assert!(content.contains(
        "scheduler_terminal_lm_head_sample_status=blocked scheduler_terminal_lm_head_sample_passed=false"
    ));
    assert!(content.contains(
        "scheduler_terminal_lm_head_uses_final_decode_device_hidden=true scheduler_terminal_lm_head_covers_full_vocabulary=false"
    ));
    assert!(content.contains(
        "scheduler_terminal_lm_head_logits_evaluated=1024 scheduler_terminal_lm_head_vocab_size=151552"
    ));
    assert!(content.contains(
        "scheduler_terminal_lm_head_top_token_id=Some(21) scheduler_terminal_lm_head_sampled_token_id=Some(42)"
    ));
    assert!(content.contains(
        "scheduler_terminal_lm_head_sample_top_k=Some(8) scheduler_terminal_lm_head_sample_top_p=Some(0.95)"
    ));
    assert!(content.contains(
        "scheduler_terminal_lm_head_argmax_backend=Some(\"cpu-reference-lm-head-argmax-bf16\")"
    ));
    assert!(content.contains(
        "scheduler_terminal_lm_head_sampler_backend=Some(\"cpu-reference-lm-head-sample-topk-topp-bf16\")"
    ));
    assert!(content.contains(
        "scheduler_terminal_lm_head_blocker=Some(\"scheduler terminal lm_head sample must use full-vocabulary preloaded-resident CUDA argmax plus non-greedy top-k/top-p sampler\")"
    ));
    assert!(content
        .contains("protocol=ExpertProtocolV2 decode_wire_request_bytes_per_touched_host=12444"));
    assert!(content.contains(
        "decode_wire_response_bytes_per_touched_host=12384 prefill_wire_request_bytes_per_touched_host=6322272"
    ));
    assert!(content.contains(
        "prefill_wire_response_bytes_per_touched_host=6291552 mtp_wire_request_bytes_per_touched_host=98880"
    ));
    assert!(content.contains(
        "mtp_wire_response_bytes_per_touched_host=98400 decode_full_sparse_roundtrip_wire_bytes=7448400"
    ));
    assert!(content.contains(
        "prefill_full_sparse_roundtrip_wire_bytes=3784147200 mtp_full_sparse_roundtrip_wire_bytes=59184000"
    ));
    assert!(content.contains(
        "scheduler_sparse_tcp_dispatch_status=not-configured scheduler_sparse_tcp_dispatch_targets=0"
    ));
    assert!(content.contains(
        "scheduler_sparse_tcp_dispatch_batches=0 scheduler_sparse_tcp_dispatch_host_batches=0"
    ));
    assert!(content.contains(
        "scheduler_sparse_tcp_dispatch_output_values=0 scheduler_sparse_tcp_dispatch_output_finite_values=0"
    ));
    assert!(content.contains(
        "scheduler_sparse_tcp_dispatch_passed=false scheduler_sparse_tcp_dispatch_expected_real_executor_id=98765"
    ));
    assert!(content.contains(
        "scheduler_sparse_tcp_dispatch_response_executor_ids_observed=3 scheduler_sparse_tcp_dispatch_real_executor_responses=2"
    ));
    assert!(content.contains(
        "scheduler_sparse_tcp_dispatch_non_real_executor_responses=1 scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4=false"
    ));
    assert!(content.contains("scheduler_sparse_tcp_dispatch_consumed_by_residual=false"));
    assert!(content.contains(
        "sampling_default_lm_head_chunk_passed=true sampling_default_lm_head_chunk_rows_scored=1024"
    ));
    assert!(content.contains(
        "sampling_default_lm_head_chunk_lm_head_bytes_read=12582912 sampling_default_lm_head_chunk_top_token_id=Some(21)"
    ));
    assert!(content.contains("sampling_default_lm_head_chunk_top_logit=Some(0.32082224)"));
    assert!(content.contains(
        "sampling_default_lm_head_chunk_uses_real_dense_prefix=true sampling_default_lm_head_chunk_residual_source_dense_layers=3"
    ));
    assert!(content
        .contains("sampling_default_lm_head_chunk_residual_source_dense_weight_bytes_read=884736"));
    assert!(content.contains(
        "sampling_default_lm_head_chunk_residual_after_checksum=Some(-0.1328725479543209)"
    ));
    assert!(content.contains("request_scheduler_summary_source=api-trace"));
    assert!(content.contains("request_prefill_tokens=3 request_prefill_chunks=1"));
    assert!(content
        .contains("request_decode_budget=8 request_mtp_verify_rows=4 request_mtp_accepted_rows=2"));
    assert!(content.contains("request_candidate_layerwaves=430"));
    assert!(content.contains("request_layerwaves=430 request_deferred_layerwaves=0"));
    assert!(content.contains("request_admitted_iterations=129 request_sparse_batches=43"));
    assert!(content.contains("request_expert_batch_rows=645 request_expert_batch_routes=3870"));
    assert!(content.contains("request_expert_source_modes=[prefill_chunk,decode_step,mtp_verify]"));
    assert!(content.contains(
        "request_expert_prefill_rows=129 request_expert_decode_rows=344 request_expert_mtp_verify_rows=172"
    ));
    assert!(content.contains(
        "request_expert_prefill_routes=774 request_expert_decode_routes=2064 request_expert_mtp_verify_routes=1032"
    ));
    assert!(content.contains(
        "request_expert_source_modes_covered=true request_expert_route_entries_match_source_rows=true"
    ));
    assert!(content.contains("request_kv_reads=1935 request_committed_kv_writes=387"));
    assert!(content.contains("request_tentative_kv_writes=172 request_committed_mtp_writes=86"));
    assert!(content.contains("request_discarded_mtp_writes=86 request_backed_kv_writes=473"));
    assert!(content.contains("request_backed_kv_bytes=642304 request_kv_reservation_bytes=741120"));
    assert!(content.contains("request_byte_backed_scheduler_trace=true"));
    assert!(content.contains(
        "request_numeric_progression_passed=true request_numeric_progression_source_rows=15"
    ));
    assert!(content.contains(
        "request_numeric_progression_hidden_dim=4 request_numeric_progression_selected_prefill_rows=129"
    ));
    assert!(content.contains(
        "request_numeric_progression_selected_decode_rows=344 request_numeric_progression_selected_mtp_rows=172"
    ));
    assert!(content.contains(
        "request_numeric_progression_attention_value_updates=2580 request_numeric_progression_mlp_value_updates=2580"
    ));
    assert!(content.contains(
        "request_numeric_progression_visible_checksum=2580 request_numeric_progression_rejected_mtp_checksum=774"
    ));
    assert!(content.contains("failed=[full_residual_stream_execution,full_vocab_sampling]"));
    assert_eq!(output.metrics.backend_mode, "real-ds4-full");
    assert_eq!(output.metrics.transport_backend, "inproc");
    assert_eq!(output.metrics.prefill_chunk_count, 1);
    assert_eq!(output.metrics.layerwave_prefill_rows, 3);
    assert_eq!(output.metrics.layerwave_decode_rows, 8);
    assert!(output.metrics.prefill_tokens_per_sec.is_none());
    let diagnostics = output.metrics.real_full.as_ref().unwrap();
    assert_eq!(diagnostics.status, "blocked");
    assert!(!diagnostics.request_scheduler_summary_runtime_reported);
    assert_eq!(
        diagnostics.scheduler_sparse_tcp_dispatch_status,
        "not-configured"
    );
    assert_eq!(
        diagnostics.scheduler_sparse_tcp_dispatch_expected_real_executor_id,
        98_765
    );
    assert_eq!(
        diagnostics.scheduler_sparse_tcp_dispatch_real_executor_responses,
        2
    );
    assert!(!diagnostics.scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4);
    assert!(!diagnostics.scheduler_sparse_tcp_dispatch_all_responses_real_checkpoint_experts);
    assert!(!diagnostics.scheduler_sparse_tcp_dispatch_consumed_by_residual);
    assert_eq!(diagnostics.request_expert_batch_rows, 645);
    assert_eq!(diagnostics.request_expert_batch_routes, 3_870);
}

#[tokio::test]
async fn real_ds4_full_backend_uses_snapshot_tokenizer_for_prompt_count() {
    let snapshot = tempfile::tempdir().expect("creating tokenizer snapshot");
    write_single_token_tokenizer(
        snapshot.path(),
        "<｜begin▁of▁sentence｜><｜User｜>Use real full.<｜Assistant｜></think>",
    );
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    let mut full = blocked_real_full_info();
    full.snapshot_path = Some(snapshot.path().display().to_string());
    state.config.real_full = Some(full);

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    let output = build_completion(&state, request).await.unwrap();
    let content = output.content.as_deref().unwrap();

    assert_eq!(output.usage.prompt_tokens, 1);
    assert_eq!(output.metrics.prompt_tokens, 1);
    assert!(content.contains("request_prefill_tokens=1 request_prefill_chunks=1"));
}

#[tokio::test]
async fn real_ds4_full_response_serializes_structured_diagnostics() {
    let mut config = ApiConfig {
        backend: ApiBackend::RealDs4Full,
        transport: ApiTransport::Tcp,
        ..ApiConfig::default()
    };
    config.real_full = Some(blocked_real_full_info());
    let (status, body) = request_json_with_config(
        config,
        Method::POST,
        "/v1/chat/completions",
        Some(json!({
            "model": format!("{}-full", DEFAULT_MODEL_ID),
            "messages": [{"role": "user", "content": "Use real full."}],
            "max_tokens": 8
        })),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["metrics"]["backend_mode"], "real-ds4-full");
    assert_eq!(body["metrics"]["transport_backend"], "tcp");
    assert_eq!(body["metrics"]["real_full"]["status"], "blocked");
    assert_eq!(body["metrics"]["real_full"]["model_id"], DEFAULT_MODEL_ID);
    let service_instance_id = body["metrics"]["real_full"]["service_instance_id"]
        .as_str()
        .expect("real-full diagnostics must carry a service instance UUID");
    let parsed_service_instance_id = Uuid::parse_str(service_instance_id).unwrap();
    assert_eq!(
        parsed_service_instance_id.get_version(),
        Some(uuid::Version::Random)
    );
    assert_eq!(parsed_service_instance_id.to_string(), service_instance_id);
    assert_eq!(body["metrics"]["real_full"]["catalog_hash"], "cataloghash");
    assert_eq!(body["metrics"]["real_full"]["layer_count"], 78);
    assert_eq!(body["metrics"]["real_full"]["dense_layer_count"], 3);
    assert_eq!(body["metrics"]["real_full"]["sparse_layer_count"], 75);
    assert_eq!(body["metrics"]["real_full"]["protocol"], "ExpertProtocolV2");
    assert_eq!(
        body["metrics"]["real_full"]["blocker"],
        "real-ds4-full is not runnable yet"
    );
    assert_eq!(
        body["metrics"]["real_full"]["failed_requirements"],
        json!(["full_residual_stream_execution", "full_vocab_sampling"])
    );
    assert_eq!(
        body["metrics"]["real_full"]["scheduler_sparse_tcp_dispatch_status"],
        "not-configured"
    );
    assert_eq!(
        body["metrics"]["real_full"]["scheduler_terminal_lm_head_logits_evaluated"],
        1024
    );
    assert_eq!(
        body["metrics"]["real_full"]["scheduler_terminal_lm_head_vocab_size"],
        151552
    );
    assert_eq!(
        body["metrics"]["real_full"]["scheduler_terminal_lm_head_top_token_id"],
        21
    );
    assert_eq!(
        body["metrics"]["real_full"]["scheduler_terminal_lm_head_sampled_token_id"],
        42
    );
    assert_eq!(
        body["metrics"]["real_full"]["scheduler_terminal_lm_head_sample_top_k"],
        8
    );
    assert_eq!(
        body["metrics"]["real_full"]["scheduler_terminal_lm_head_sample_top_p"],
        0.95
    );
    assert_eq!(
        body["metrics"]["real_full"]["quantization_recipe"],
        "test-quantization-recipe"
    );
    assert_eq!(
        body["metrics"]["real_full"]
            ["scheduler_sparse_tcp_dispatch_all_responses_real_checkpoint_experts"],
        false
    );
    assert_eq!(
        body["metrics"]["real_full"]["scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4"],
        false
    );
    assert_eq!(
        body["metrics"]["real_full"]["scheduler_sparse_tcp_dispatch_consumed_by_residual"],
        false
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_numeric_progression_passed"],
        true
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_scheduler_summary_runtime_reported"],
        false
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_coordinator_graph_slots"],
        0
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_coordinator_graph_launches"],
        0
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_candidate_layerwaves"],
        430
    );
    assert_eq!(body["metrics"]["real_full"]["request_layerwaves"], 430);
    assert_eq!(
        body["metrics"]["real_full"]["request_deferred_layerwaves"],
        0
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_expert_prefill_rows"],
        129
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_expert_decode_rows"],
        344
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_expert_mtp_verify_rows"],
        172
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_expert_prefill_routes"],
        774
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_expert_decode_routes"],
        2064
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_expert_mtp_verify_routes"],
        1032
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_expert_source_modes_covered"],
        true
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_expert_route_entries_match_source_rows"],
        true
    );
    assert_eq!(body["metrics"]["real_full"]["request_kv_reads"], 1935);
    assert_eq!(
        body["metrics"]["real_full"]["request_committed_kv_writes"],
        387
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_tentative_kv_writes"],
        172
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_committed_mtp_writes"],
        86
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_discarded_mtp_writes"],
        86
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_backed_kv_writes"],
        473
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_backed_kv_bytes"],
        642_304
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_kv_reservation_bytes"],
        741_120
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_byte_backed_scheduler_trace"],
        true
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_numeric_progression_selected_prefill_rows"],
        129
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_numeric_progression_selected_decode_rows"],
        344
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_numeric_progression_selected_mtp_rows"],
        172
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_numeric_progression_attention_value_updates"],
        2580
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_numeric_progression_mlp_value_updates"],
        2580
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_numeric_progression_visible_checksum"],
        2580.0
    );
    assert_eq!(
        body["metrics"]["real_full"]["request_numeric_progression_rejected_mtp_checksum"],
        774.0
    );
}

#[tokio::test]
async fn real_ds4_full_backend_reports_chunked_request_prefill_for_long_prompt() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());

    let long_prompt = vec!["chunk"; 512].join(" ");
    let mut request = base_request(&long_prompt);
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(2);
    let output = build_completion(&state, request).await.unwrap();
    let content = output.content.as_deref().unwrap();

    assert_eq!(output.usage.prompt_tokens, 512);
    assert_eq!(output.metrics.prompt_tokens, 512);
    assert_eq!(output.metrics.prefill_chunk_count, 1);
    assert_eq!(output.metrics.layerwave_prefill_rows, 512);
    assert_eq!(output.metrics.layerwave_decode_rows, 2);
    assert!(content.contains("request_prefill_tokens=512 request_prefill_chunks=1"));
    assert!(content
        .contains("request_decode_budget=2 request_mtp_verify_rows=2 request_mtp_accepted_rows=2"));
    assert!(content.contains("request_candidate_layerwaves=172"));
    assert!(content.contains("request_layerwaves=172 request_deferred_layerwaves=0"));
    assert!(content.contains("request_admitted_iterations=129 request_sparse_batches=43"));
    assert!(content.contains("request_expert_batch_rows=22188 request_expert_batch_routes=133128"));
    assert!(content.contains(
        "request_expert_prefill_rows=22016 request_expert_decode_rows=86 request_expert_mtp_verify_rows=86"
    ));
    assert!(content.contains(
        "request_expert_prefill_routes=132096 request_expert_decode_routes=516 request_expert_mtp_verify_routes=516"
    ));
    assert!(content.contains(
        "request_numeric_progression_passed=true request_numeric_progression_source_rows=516"
    ));
    assert!(content.contains(
        "request_numeric_progression_hidden_dim=4 request_numeric_progression_selected_prefill_rows=22016"
    ));
    assert!(content.contains(
        "request_numeric_progression_selected_decode_rows=86 request_numeric_progression_selected_mtp_rows=86"
    ));
    let diagnostics = output.metrics.real_full.as_ref().unwrap();
    assert_eq!(diagnostics.request_prefill_tokens, 512);
    assert_eq!(diagnostics.request_prefill_chunks, 1);
    assert_eq!(diagnostics.request_candidate_layerwaves, 172);
    assert_eq!(diagnostics.request_layerwaves, 172);
    assert_eq!(diagnostics.request_admitted_iterations, 129);
    assert_eq!(diagnostics.request_sparse_batches, 43);
    assert_eq!(diagnostics.request_expert_prefill_rows, 22_016);
    assert_eq!(diagnostics.request_expert_decode_rows, 86);
    assert_eq!(diagnostics.request_expert_mtp_verify_rows, 86);
    assert_eq!(diagnostics.request_expert_prefill_routes, 132_096);
    assert_eq!(diagnostics.request_expert_decode_routes, 516);
    assert_eq!(diagnostics.request_expert_mtp_verify_routes, 516);
    assert!(diagnostics.request_expert_source_modes_covered);
    assert!(diagnostics.request_expert_route_entries_match_source_rows);
    assert!(diagnostics.request_byte_backed_scheduler_trace);
    assert_eq!(
        diagnostics.request_numeric_progression_selected_prefill_rows,
        22_016
    );
    assert_eq!(
        diagnostics.request_numeric_progression_selected_decode_rows,
        86
    );
    assert_eq!(
        diagnostics.request_numeric_progression_selected_mtp_rows,
        86
    );
}

#[tokio::test]
async fn real_ds4_full_backend_keeps_diagnostics_when_overall_preflight_is_blocked() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    let mut full = blocked_real_full_info();
    full.scheduler_terminal_lm_head_sample_status = "passed".to_owned();
    full.scheduler_terminal_lm_head_sample_passed = true;
    full.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    full.scheduler_terminal_lm_head_logits_evaluated = full.scheduler_terminal_lm_head_vocab_size;
    full.scheduler_terminal_lm_head_blocker = None;
    state.config.real_full = Some(full);

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(8);
    let output = build_completion(&state, request).await.unwrap();
    let content = output.content.unwrap();

    assert!(content.contains("real ds4 full status=blocked"));
    assert!(content.contains("failed=[full_residual_stream_execution,full_vocab_sampling]"));
    assert!(!content.contains("ds41rt-token:42"));
    let diagnostics = output.metrics.real_full.as_ref().unwrap();
    assert_eq!(diagnostics.status, "blocked");
    assert_eq!(
        diagnostics.failed_requirements,
        vec![
            "full_residual_stream_execution".to_owned(),
            "full_vocab_sampling".to_owned()
        ]
    );
    assert_eq!(
        diagnostics.blocker.as_deref(),
        Some("real-ds4-full is not runnable yet")
    );
    assert_eq!(
        diagnostics.scheduler_terminal_lm_head_sample_status,
        "passed"
    );
    assert!(!diagnostics.scheduler_sparse_tcp_dispatch_passed);
    assert_eq!(diagnostics.request_prefill_chunks, 1);
    assert!(diagnostics.request_numeric_progression_passed);
}

#[tokio::test]
async fn real_ds4_full_backend_returns_terminal_sample_token_when_gate_passes() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    let mut full = blocked_real_full_info();
    full.status = "ready".to_owned();
    full.scheduler_full_context_device_attention_complete = true;
    full.scheduler_terminal_lm_head_sample_status = "passed".to_owned();
    full.scheduler_terminal_lm_head_sample_passed = true;
    full.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    full.scheduler_terminal_lm_head_logits_evaluated = full.scheduler_terminal_lm_head_vocab_size;
    full.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    full.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    full.scheduler_terminal_lm_head_blocker = None;
    full.blocker.clear();
    full.failed_requirements.clear();
    state.config.real_full = Some(full);

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("ds41rt-token:42"));
    assert_eq!(output.finish_reason, "length");
    assert_eq!(output.usage.completion_tokens, 1);
    assert_eq!(output.metrics.backend_mode, "real-ds4-full");
    assert_eq!(output.metrics.transport_backend, "inproc");
    assert_eq!(output.metrics.prefill_chunk_count, 1);
    assert_eq!(output.metrics.layerwave_prefill_rows, 3);
    assert_eq!(output.metrics.layerwave_decode_rows, 1);
    assert!(output.metrics.real_full.is_none());
}

#[tokio::test]
async fn real_ds4_full_single_token_failure_finishes_sequence() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    let sequences = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    state.config.real_full_executor = Some(Arc::new(FailingFinishingRealFullExecutor {
        sequences: Arc::clone(&sequences),
        finishes: Arc::clone(&finishes),
    }));

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    let error = build_completion(&state, request)
        .await
        .expect_err("single-token execution should fail");

    assert!(error
        .message
        .contains("intentional real-full execution failure"));
    assert_eq!(
        finishes.lock().unwrap().as_slice(),
        sequences.lock().unwrap().as_slice()
    );
}

#[tokio::test]
async fn real_ds4_full_single_token_uses_persistent_scheduling_when_available() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    let batches = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let mut info = blocked_runtime_sample_real_full_info(42, "a");
    info.scheduler_full_context_device_attention_complete = true;
    let executor = ThreadPinnedRealFullRequestExecutor::spawn(
        "ds41rt-api-single-token-persistent-test",
        PersistentBatchRecordingRealFullExecutor {
            batches: Arc::clone(&batches),
            finishes: Arc::clone(&finishes),
            info,
        },
    )
    .expect("spawning persistent single-token executor");
    state.config.real_full_executor = Some(Arc::new(executor));

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("a"));
    assert_eq!(output.finish_reason, "length");
    assert_eq!(output.usage.completion_tokens, 1);
    let batches = batches.lock().unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].len(), 1);
    assert_eq!(batches[0][0].1, 0);
    assert_eq!(batches[0][0].2, 1);
    assert_eq!(finishes.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn real_ds4_full_backend_reports_multi_token_blocker_without_request_executor() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    let mut full = blocked_real_full_info();
    full.status = "ready".to_owned();
    full.scheduler_full_context_device_attention_complete = true;
    full.scheduler_terminal_lm_head_sample_status = "passed".to_owned();
    full.scheduler_terminal_lm_head_sample_passed = true;
    full.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    full.scheduler_terminal_lm_head_logits_evaluated = full.scheduler_terminal_lm_head_vocab_size;
    full.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    full.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    full.scheduler_terminal_lm_head_blocker = None;
    full.blocker.clear();
    full.failed_requirements.clear();
    state.config.real_full = Some(full);

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(2);
    let output = build_completion(&state, request).await.unwrap();
    let content = output.content.unwrap();

    assert!(content.contains("real ds4 full status=blocked"));
    assert!(content.contains("request_decode_budget=2"));
    assert!(content.contains("multi-token decode requires a live request executor"));
    assert!(content.contains("failed=[multi_token_decode_loop]"));
    assert!(!content.contains("ds41rt-token:42"));
    let diagnostics = output.metrics.real_full.as_ref().unwrap();
    assert_eq!(diagnostics.status, "blocked");
    assert_eq!(
        diagnostics.blocker.as_deref(),
        Some(
            "real-full multi-token decode requires a live request executor: request_decode_budget=2; preflight terminal lm_head sample covers one final decode row"
        )
    );
    assert_eq!(
        diagnostics.failed_requirements,
        vec!["multi_token_decode_loop".to_owned()]
    );
    assert!(diagnostics.scheduler_terminal_lm_head_sample_passed);
    assert_eq!(diagnostics.request_decode_budget, 2);
}

#[tokio::test]
async fn real_ds4_full_backend_runs_executor_decode_loop_for_multi_token_requests() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut base = blocked_real_full_info();
    base.status = "ready".to_owned();
    base.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    base.request_decode_budget = 1;
    base.request_mtp_verify_rows = 1;
    base.request_mtp_accepted_rows = 1;
    base.request_dspark_joint_batch_width = 4;
    base.request_dspark_joint_cohort_width = 2;
    base.request_dspark_joint_cohort_count = 2;
    base.request_dspark_2x2_wavefront = true;
    base.scheduler_full_context_device_attention_complete = true;
    base.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    base.scheduler_terminal_lm_head_sample_passed = true;
    base.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    base.scheduler_terminal_lm_head_logits_evaluated = base.scheduler_terminal_lm_head_vocab_size;
    base.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    base.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    base.scheduler_terminal_lm_head_blocker = None;
    base.blocker.clear();
    base.failed_requirements.clear();
    state.config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base,
        tokens: vec![
            (42, "A".to_owned()),
            (43, "B".to_owned()),
            (44, "C".to_owned()),
        ],
    }));

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(3);
    request.temperature = None;
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("ABC"));
    assert_eq!(output.finish_reason, "length");
    assert_eq!(output.usage.completion_tokens, 3);
    assert_eq!(output.metrics.layerwave_decode_rows, 3);
    assert_eq!(output.metrics.prefill_chunk_count, 1);
    assert_eq!(output.metrics.layerwave_prefill_rows, 3);
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 3);
    assert_eq!(captured[1].sequence_id, captured[0].sequence_id);
    assert_eq!(captured[2].sequence_id, captured[0].sequence_id);
    assert_eq!(captured[0].request_index, 1);
    assert_eq!(captured[0].max_tokens, 1);
    assert!(captured[0].generated_token_ids.is_empty());
    assert_eq!(captured[0].decode_step_index, 0);
    assert_eq!(captured[0].decode_budget, 3);
    assert!(captured[0].greedy_sampling);
    assert_eq!(captured[1].request_index, 2);
    assert_eq!(captured[1].max_tokens, 1);
    assert_eq!(captured[1].generated_token_ids, vec![42]);
    assert_eq!(captured[1].decode_step_index, 1);
    assert_eq!(captured[1].decode_budget, 3);
    assert!(captured[1].greedy_sampling);
    assert_eq!(captured[2].request_index, 3);
    assert_eq!(captured[2].max_tokens, 1);
    assert_eq!(captured[2].generated_token_ids, vec![42, 43]);
    assert_eq!(captured[2].decode_step_index, 2);
    assert_eq!(captured[2].decode_budget, 3);
    assert!(captured[2].greedy_sampling);
    let diagnostics = output.metrics.real_full.as_ref().unwrap();
    assert!(diagnostics.request_scheduler_summary_runtime_reported);
    assert_eq!(diagnostics.status, "ready");
    assert_eq!(diagnostics.request_decode_budget, 1);
    assert_eq!(
        diagnostics.scheduler_terminal_lm_head_sample_status,
        "sampled"
    );
    assert!(diagnostics.scheduler_terminal_lm_head_sample_passed);
    assert_eq!(
        diagnostics.scheduler_terminal_lm_head_sampled_token_id,
        Some(44)
    );
    assert_eq!(diagnostics.mtp_verify_cycles, 3);
    assert_eq!(diagnostics.mtp_draft_tokens, 3);
    assert_eq!(diagnostics.mtp_accepted_draft_tokens, 3);
    assert_eq!(diagnostics.mtp_emitted_tokens_from_verify, 3);
    assert_eq!(diagnostics.mtp_full_match_cycles, 3);
    assert_eq!(diagnostics.mtp_accepted_draft_lengths, vec![1, 1, 1]);
    assert_eq!(diagnostics.mtp_verify_cycle_ms.len(), 3);
    assert_eq!(diagnostics.dspark_joint_cycles, 3);
    assert_eq!(diagnostics.dspark_max_joint_batch_width, 4);
    assert_eq!(diagnostics.dspark_max_joint_cohort_width, 2);
    assert_eq!(diagnostics.dspark_max_joint_cohort_count, 2);
    assert_eq!(diagnostics.dspark_2x2_wavefront_cycles, 3);
}

#[tokio::test]
async fn real_ds4_full_backend_consumes_verified_tokens_from_one_decode_cycle() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut base = blocked_real_full_info();
    base.status = "ready".to_owned();
    base.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    base.request_decode_budget = 1;
    base.request_mtp_verify_rows = 2;
    base.request_mtp_accepted_rows = 2;
    base.scheduler_full_context_device_attention_complete = true;
    base.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    base.scheduler_terminal_lm_head_sample_passed = true;
    base.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    base.scheduler_terminal_lm_head_logits_evaluated = base.scheduler_terminal_lm_head_vocab_size;
    base.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    base.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    base.scheduler_terminal_lm_head_blocker = None;
    base.blocker.clear();
    base.failed_requirements.clear();
    state.config.real_full_executor = Some(Arc::new(CycleSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base,
        tokens: vec![
            (42, "A".to_owned()),
            (43, "B".to_owned()),
            (44, "C".to_owned()),
        ],
    }));

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(3);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("ABC"));
    assert_eq!(output.usage.completion_tokens, 3);
    assert_eq!(output.metrics.layerwave_decode_rows, 1);
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(captured[0].generated_token_ids.is_empty());
    assert_eq!(captured[0].decode_step_index, 0);
    let diagnostics = output.metrics.real_full.as_ref().unwrap();
    assert_eq!(diagnostics.mtp_verify_cycles, 1);
    assert_eq!(diagnostics.mtp_draft_tokens, 2);
    assert_eq!(diagnostics.mtp_accepted_draft_tokens, 2);
    assert_eq!(diagnostics.mtp_emitted_tokens_from_verify, 3);
    assert_eq!(diagnostics.mtp_full_match_cycles, 1);
    assert_eq!(diagnostics.mtp_accepted_draft_lengths, vec![2]);
    assert_eq!(diagnostics.mtp_verify_cycle_ms.len(), 1);
}

#[tokio::test]
async fn real_ds4_full_backend_stream_decodes_split_utf8_tokens() {
    let snapshot = tempfile::tempdir().expect("creating tokenizer snapshot");
    write_split_utf8_tokenizer(snapshot.path());
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    let mut preflight = blocked_real_full_info();
    preflight.snapshot_path = Some(snapshot.path().display().to_string());
    state.config.real_full = Some(preflight);

    let mut base = blocked_real_full_info();
    base.status = "ready".to_owned();
    base.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    base.request_decode_budget = 1;
    base.scheduler_full_context_device_attention_complete = true;
    base.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    base.scheduler_terminal_lm_head_sample_passed = true;
    base.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    base.scheduler_terminal_lm_head_logits_evaluated = base.scheduler_terminal_lm_head_vocab_size;
    base.scheduler_terminal_lm_head_blocker = None;
    base.blocker.clear();
    base.failed_requirements.clear();
    state.config.real_full_executor = Some(Arc::new(CycleSamplingRealFullExecutor {
        requests: Arc::new(Mutex::new(Vec::new())),
        base,
        tokens: vec![
            (2, "�".to_owned()),
            (3, "�".to_owned()),
            (4, "�".to_owned()),
        ],
    }));

    let mut request = base_request("Show the parrot.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(3);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("🦜"));
    assert_eq!(output.usage.completion_tokens, 3);
    assert!(!output.content.as_deref().unwrap().contains('�'));
}

#[tokio::test]
async fn real_ds4_full_backend_decode_loop_stops_at_ds4_chat_marker() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let mut base = blocked_real_full_info();
    base.status = "ready".to_owned();
    base.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    base.request_decode_budget = 1;
    base.scheduler_full_context_device_attention_complete = true;
    base.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    base.scheduler_terminal_lm_head_sample_passed = true;
    base.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    base.scheduler_terminal_lm_head_logits_evaluated = base.scheduler_terminal_lm_head_vocab_size;
    base.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    base.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    base.scheduler_terminal_lm_head_blocker = None;
    base.blocker.clear();
    base.failed_requirements.clear();
    state.config.real_full_executor = Some(Arc::new(FinishingStepSamplingRealFullExecutor {
        inner: StepSamplingRealFullExecutor {
            requests: Arc::clone(&requests),
            base,
            tokens: vec![
                (42, "A".to_owned()),
                (43, "B".to_owned()),
                (128_803, "<｜User｜>".to_owned()),
                (44, "C".to_owned()),
            ],
        },
        finishes: Arc::clone(&finishes),
    }));

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(4);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("AB"));
    assert_eq!(output.finish_reason, "stop");
    assert_eq!(output.usage.completion_tokens, 2);
    assert_eq!(output.metrics.layerwave_decode_rows, 3);
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 3);
    assert_eq!(captured[2].generated_token_ids, vec![42, 43]);
    assert_eq!(captured[2].decode_step_index, 2);
    assert_eq!(
        finishes.lock().unwrap().as_slice(),
        &[captured[0].sequence_id.clone()]
    );
}

#[tokio::test]
async fn real_ds4_full_backend_separates_reasoning_prefix_before_think_close() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut base = blocked_real_full_info();
    base.status = "ready".to_owned();
    base.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    base.request_decode_budget = 1;
    base.scheduler_full_context_device_attention_complete = true;
    base.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    base.scheduler_terminal_lm_head_sample_passed = true;
    base.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    base.scheduler_terminal_lm_head_logits_evaluated = base.scheduler_terminal_lm_head_vocab_size;
    base.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    base.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    base.scheduler_terminal_lm_head_blocker = None;
    base.blocker.clear();
    base.failed_requirements.clear();
    state.config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base,
        tokens: vec![
            (42, "0.00005000".to_owned()),
            (128_822, "</think>".to_owned()),
            (43, "Hi".to_owned()),
        ],
    }));

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(3);
    request.enable_thinking = Some(true);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("Hi"));
    assert_eq!(output.reasoning_content.as_deref(), Some("0.00005000"));
    assert!(!output.content.as_deref().unwrap().contains("</think>"));
    assert!(!output.content.as_deref().unwrap().contains("0.00005000"));
    assert_eq!(output.usage.completion_tokens, 3);
    assert_eq!(output.metrics.layerwave_decode_rows, 3);
    assert_eq!(requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn real_ds4_full_backend_strips_inline_think_close_before_visible_text() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    let mut full = blocked_runtime_sample_real_full_info(42, "0.00005000</think>Hi");
    full.status = "ready".to_owned();
    full.scheduler_full_context_device_attention_complete = true;
    full.blocker.clear();
    full.failed_requirements.clear();
    state.config.real_full = Some(full);

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("Hi"));
    assert_eq!(output.usage.completion_tokens, 1);
}

#[tokio::test]
async fn real_ds4_full_backend_decode_loop_stops_when_runtime_diagnostics_blocked() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    let requests = Arc::new(Mutex::new(Vec::new()));
    state.config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base: blocked_runtime_sample_real_full_info(41, "unused"),
        tokens: vec![
            (42, "A".to_owned()),
            (43, "B".to_owned()),
            (44, "C".to_owned()),
        ],
    }));

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(3);
    let output = build_completion(&state, request).await.unwrap();
    let content = output.content.as_deref().unwrap();

    assert!(content.contains("real ds4 full status=blocked"));
    assert!(content.contains("scheduler_full_context_device_attention_complete=false"));
    assert!(content.contains("failed=[full_residual_stream_execution,full_vocab_sampling]"));
    assert!(!content.contains("A"));
    assert!(!content.contains("B"));
    assert!(!content.contains("C"));
    assert_eq!(output.metrics.layerwave_decode_rows, 1);
    let diagnostics = output.metrics.real_full.as_ref().unwrap();
    assert!(diagnostics.request_scheduler_summary_runtime_reported);
    assert_eq!(diagnostics.status, "blocked");
    assert_eq!(
        diagnostics.startup_diagnostic_mode,
        "request-scheduler-execution"
    );
    assert_eq!(
        diagnostics.blocker.as_deref(),
        Some("real-ds4-full is not runnable yet")
    );
    assert_eq!(
        diagnostics.failed_requirements,
        vec![
            "full_residual_stream_execution".to_owned(),
            "full_vocab_sampling".to_owned()
        ]
    );
    assert!(!diagnostics.scheduler_full_context_device_attention_complete);
    assert!(diagnostics.scheduler_terminal_lm_head_sample_passed);
    assert!(diagnostics.scheduler_terminal_lm_head_covers_full_vocabulary);

    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].decode_budget, 3);
    assert!(captured[0].generated_token_ids.is_empty());
}

#[tokio::test]
async fn real_ds4_full_backend_prefers_decoded_terminal_sample_text() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    let mut full = blocked_real_full_info();
    full.status = "ready".to_owned();
    full.scheduler_full_context_device_attention_complete = true;
    full.scheduler_terminal_lm_head_sample_status = "passed".to_owned();
    full.scheduler_terminal_lm_head_sample_passed = true;
    full.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    full.scheduler_terminal_lm_head_logits_evaluated = full.scheduler_terminal_lm_head_vocab_size;
    full.scheduler_terminal_lm_head_sampled_text = Some("decoded sample phrase".to_owned());
    full.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    full.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    full.scheduler_terminal_lm_head_blocker = None;
    full.blocker.clear();
    full.failed_requirements.clear();
    state.config.real_full = Some(full);

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("decoded sample phrase"));
    assert_eq!(output.finish_reason, "length");
    assert_eq!(output.usage.completion_tokens, 1);
}

#[tokio::test]
async fn real_ds4_full_runtime_output_uses_sampled_token_over_diagnostic_argmax() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    let mut full = blocked_real_full_info();
    full.status = "ready".to_owned();
    full.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    full.request_decode_budget = 1;
    full.scheduler_full_context_device_attention_complete = true;
    full.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    full.scheduler_terminal_lm_head_sample_passed = true;
    full.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    full.scheduler_terminal_lm_head_logits_evaluated = full.scheduler_terminal_lm_head_vocab_size;
    full.scheduler_terminal_lm_head_top_token_id = Some(21);
    full.scheduler_terminal_lm_head_sampled_token_id = Some(42);
    full.scheduler_terminal_lm_head_sampled_text = Some("sampled-token".to_owned());
    full.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    full.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    full.scheduler_terminal_lm_head_blocker = None;
    full.blocker.clear();
    full.failed_requirements.clear();
    state.config.real_full = Some(full);

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("sampled-token"));
    assert!(output.metrics.real_full.is_none());
}

#[tokio::test]
async fn real_ds4_full_streaming_preserves_terminal_sample_chunk() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    let mut full = blocked_real_full_info();
    full.status = "ready".to_owned();
    full.scheduler_full_context_device_attention_complete = true;
    full.scheduler_terminal_lm_head_sample_status = "passed".to_owned();
    full.scheduler_terminal_lm_head_sample_passed = true;
    full.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    full.scheduler_terminal_lm_head_logits_evaluated = full.scheduler_terminal_lm_head_vocab_size;
    full.scheduler_terminal_lm_head_sampled_text = Some("decoded sample phrase".to_owned());
    full.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    full.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    full.scheduler_terminal_lm_head_blocker = None;
    full.blocker.clear();
    full.failed_requirements.clear();
    state.config.real_full = Some(full);

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    let output = build_completion(&state, request).await.unwrap();
    let response = chat_stream_response(output, true);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();

    assert_eq!(text.matches("\"content\":").count(), 1);
    assert!(text.contains("\"content\":\"decoded sample phrase\""));
    assert!(!text.contains("\"content\":\"decoded\""));
    assert!(text.contains("\"choices\":[],\"usage\":{"));
    assert!(text.contains("\"completion_tokens\":1"));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn real_ds4_full_route_streams_executor_decode_steps_as_sse_chunks() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut base = blocked_real_full_info();
    base.status = "ready".to_owned();
    base.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    base.request_decode_budget = 1;
    base.request_mtp_verify_rows = 1;
    base.request_mtp_accepted_rows = 1;
    base.request_dspark_joint_batch_width = 4;
    base.request_dspark_joint_cohort_width = 2;
    base.request_dspark_joint_cohort_count = 2;
    base.request_dspark_2x2_wavefront = true;
    base.scheduler_full_context_device_attention_complete = true;
    base.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    base.scheduler_terminal_lm_head_sample_passed = true;
    base.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    base.scheduler_terminal_lm_head_logits_evaluated = base.scheduler_terminal_lm_head_vocab_size;
    base.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    base.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    base.scheduler_terminal_lm_head_blocker = None;
    base.blocker.clear();
    base.failed_requirements.clear();

    let mut config = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc).config;
    config.real_full = Some(base.clone());
    config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base,
        tokens: vec![
            (42, "A".to_owned()),
            (43, "B".to_owned()),
            (44, "C".to_owned()),
        ],
    }));

    let (status, text) = request_text_with_config(
        config,
        Method::POST,
        "/v1/chat/completions",
        json!({
            "model": format!("{}-full", DEFAULT_MODEL_ID),
            "stream": true,
            "stream_options": {"include_usage": true},
            "messages": [{"role": "user", "content": "Use real full."}],
            "max_tokens": 3
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert!(text.contains("\"role\":\"assistant\""));
    assert_eq!(text.matches("\"content\":").count(), 3);
    let a = text.find("\"content\":\"A\"").unwrap();
    let b = text.find("\"content\":\"B\"").unwrap();
    let c = text.find("\"content\":\"C\"").unwrap();
    assert!(a < b && b < c);
    assert!(text.contains("\"finish_reason\":\"length\""));
    assert!(text.contains("\"output_tokens\":3"));
    assert!(text.contains("\"mtp_verify_cycles\":3"));
    assert!(text.contains("\"mtp_accepted_draft_lengths\":[1,1,1]"));
    assert!(text.contains("\"dspark_joint_cycles\":3"));
    assert!(text.contains("\"dspark_max_joint_batch_width\":4"));
    assert!(text.contains("\"dspark_max_joint_cohort_width\":2"));
    assert!(text.contains("\"dspark_max_joint_cohort_count\":2"));
    assert!(text.contains("\"dspark_2x2_wavefront_cycles\":3"));
    assert!(text.contains("\"prefill_chunk_count\":1"));
    assert!(text.contains("\"layerwave_prefill_rows\":3"));
    assert!(text.contains("\"layerwave_decode_rows\":3"));
    assert!(text.contains(
        "\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":3,\"total_tokens\":6,\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens_details\":{\"reasoning_tokens\":0}}"
    ));
    assert!(text.contains("[DONE]"));

    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 3);
    assert_eq!(captured[1].sequence_id, captured[0].sequence_id);
    assert_eq!(captured[2].sequence_id, captured[0].sequence_id);
    assert_eq!(captured[0].max_tokens, 1);
    assert!(captured[0].generated_token_ids.is_empty());
    assert_eq!(captured[0].decode_step_index, 0);
    assert_eq!(captured[0].decode_budget, 3);
    assert_eq!(captured[1].generated_token_ids, vec![42]);
    assert_eq!(captured[1].decode_step_index, 1);
    assert_eq!(captured[2].generated_token_ids, vec![42, 43]);
    assert_eq!(captured[2].decode_step_index, 2);
}

#[tokio::test]
async fn real_ds4_full_route_streams_executor_failure_as_error_not_content() {
    let sequences = Arc::new(Mutex::new(Vec::new()));
    let finishes = Arc::new(Mutex::new(Vec::new()));
    let mut config = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc).config;
    config.real_full = Some(blocked_real_full_info());
    config.real_full_executor = Some(Arc::new(FailingFinishingRealFullExecutor {
        sequences: Arc::clone(&sequences),
        finishes: Arc::clone(&finishes),
    }));

    let (status, text) = request_text_with_config(
        config,
        Method::POST,
        "/v1/chat/completions",
        json!({
            "model": format!("{}-full", DEFAULT_MODEL_ID),
            "stream": true,
            "stream_options": {"include_usage": true},
            "messages": [{"role": "user", "content": "Use real full."}],
            "max_tokens": 8
        }),
    )
    .await;

    // The response headers are already committed once an SSE stream begins,
    // so a runtime failure remains HTTP 200 and is carried by an error event.
    assert_eq!(status, StatusCode::OK);
    assert!(text.contains("\"error\":{\"message\":\"real-full streaming executor error: intentional real-full execution failure\""));
    assert!(text.contains("\"type\":\"server_error\""));
    assert!(text.contains("\"code\":\"backend_error\""));
    assert!(!text.contains("\"delta\":{\"content\":\"real-full streaming executor error"));
    assert!(!text.contains("\"finish_reason\""));
    assert!(!text.contains("\"usage\""));
    assert!(!text.contains("\"metrics\""));
    assert!(text.contains("[DONE]"));
    assert_eq!(
        finishes.lock().unwrap().as_slice(),
        sequences.lock().unwrap().as_slice()
    );
}

#[tokio::test]
async fn real_ds4_full_route_streams_reasoning_separately_in_real_time() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut base = ready_runtime_sample_real_full_info(42, "unused");
    base.request_decode_budget = 1;
    let mut config = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc).config;
    config.real_full = Some(base.clone());
    config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base,
        tokens: vec![
            (42, "working".to_owned()),
            (128_822, "</think>".to_owned()),
            (43, "answer".to_owned()),
        ],
    }));

    let (status, text) = request_text_with_config(
        config,
        Method::POST,
        "/v1/chat/completions",
        json!({
            "model": format!("{}-full", DEFAULT_MODEL_ID),
            "stream": true,
            "stream_options": {"include_usage": true},
            "messages": [{"role": "user", "content": "Think, then answer."}],
            "thinking": {"type": "enabled", "clear_thinking": false},
            "max_tokens": 3
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let reasoning = text.find("\"reasoning_content\":\"working\"").unwrap();
    let answer = text.find("\"content\":\"answer\"").unwrap();
    assert!(reasoning < answer);
    assert!(!text.contains("</think>"));
    assert!(text.contains("\"completion_tokens\":3"));
    assert!(text.contains("\"completion_tokens_details\":{\"reasoning_tokens\":1}"));
    assert_eq!(requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn real_ds4_full_route_streaming_stops_at_ds4_chat_marker() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut base = blocked_real_full_info();
    base.status = "ready".to_owned();
    base.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    base.request_decode_budget = 1;
    base.scheduler_full_context_device_attention_complete = true;
    base.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    base.scheduler_terminal_lm_head_sample_passed = true;
    base.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    base.scheduler_terminal_lm_head_logits_evaluated = base.scheduler_terminal_lm_head_vocab_size;
    base.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    base.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    base.scheduler_terminal_lm_head_blocker = None;
    base.blocker.clear();
    base.failed_requirements.clear();

    let mut config = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc).config;
    config.real_full = Some(base.clone());
    config.real_full_executor = Some(Arc::new(StepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base,
        tokens: vec![
            (42, "A".to_owned()),
            (43, "B".to_owned()),
            (128_803, "<｜User｜>".to_owned()),
            (44, "C".to_owned()),
        ],
    }));

    let (status, text) = request_text_with_config(
        config,
        Method::POST,
        "/v1/chat/completions",
        json!({
            "model": format!("{}-full", DEFAULT_MODEL_ID),
            "stream": true,
            "messages": [{"role": "user", "content": "Use real full."}],
            "max_tokens": 4
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(text.matches("\"content\":").count(), 2);
    assert!(text.contains("\"content\":\"A\""));
    assert!(text.contains("\"content\":\"B\""));
    assert!(!text.contains("<｜User｜>"));
    assert!(!text.contains("\"content\":\"C\""));
    assert!(text.contains("\"finish_reason\":\"stop\""));
    assert!(text.contains("\"output_tokens\":2"));
    assert!(text.contains("\"layerwave_decode_rows\":3"));
    assert!(!text.contains("\"usage\":"));
    assert!(text.contains("[DONE]"));
    assert_eq!(requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn real_ds4_full_streaming_waits_for_first_decode_result_before_role() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut base = blocked_real_full_info();
    base.status = "ready".to_owned();
    base.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    base.request_decode_budget = 1;
    base.scheduler_full_context_device_attention_complete = true;
    base.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    base.scheduler_terminal_lm_head_sample_passed = true;
    base.scheduler_terminal_lm_head_uses_final_decode_device_hidden = true;
    base.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    base.scheduler_terminal_lm_head_logits_evaluated = base.scheduler_terminal_lm_head_vocab_size;
    base.scheduler_terminal_lm_head_argmax_backend =
        Some("cuda-full-vocab-lm-head-argmax-bf16".to_owned());
    base.scheduler_terminal_lm_head_sampler_backend =
        Some("cuda-full-vocab-lm-head-sample-topk-topp-bf16".to_owned());
    base.scheduler_terminal_lm_head_blocker = None;
    base.blocker.clear();
    base.failed_requirements.clear();

    let mut config = test_state(ApiBackend::RealDs4Full, ApiTransport::Tcp).config;
    config.real_full = Some(base.clone());
    config.real_full_executor = Some(Arc::new(BlockingStepSamplingRealFullExecutor {
        requests: Arc::clone(&requests),
        base,
        token: (42, "A".to_owned()),
        entered: entered_tx,
        release: Mutex::new(release_rx),
    }));

    let app = router_with_config(config);
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": format!("{}-full", DEFAULT_MODEL_ID),
                "stream": true,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1
            })
            .to_string(),
        ))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let mut body = response.into_body().into_data_stream();
    let first = body.next();
    tokio::pin!(first);
    assert!(
        tokio::time::timeout(Duration::from_millis(250), &mut first)
            .await
            .is_err(),
        "stream must not emit a progress SSE event while the first decode is still running"
    );
    assert_eq!(entered_rx.recv_timeout(Duration::from_secs(1)).unwrap(), ());

    release_tx.send(()).unwrap();
    let first = tokio::time::timeout(Duration::from_secs(2), &mut first)
        .await
        .expect("streaming role frame should arrive with the first decode result")
        .expect("stream should yield a first frame")
        .expect("first stream frame should be ok");
    let first_text = String::from_utf8(first.to_vec()).unwrap();
    assert!(first_text.contains("\"role\":\"assistant\""));

    let remaining = tokio::time::timeout(Duration::from_secs(2), async {
        let mut text = String::new();
        while let Some(chunk) = body.next().await {
            text.push_str(&String::from_utf8(chunk.unwrap().to_vec()).unwrap());
            if text.contains("[DONE]") {
                break;
            }
        }
        text
    })
    .await
    .expect("stream should finish after decode executor is released");
    assert!(remaining.contains("\"content\":\"A\""));
    assert!(remaining.contains("[DONE]"));
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert!(captured[0].greedy_sampling);
}

#[tokio::test]
async fn real_ds4_full_backend_uses_request_executor_result_when_available() {
    let mut state = test_state(ApiBackend::RealDs4Full, ApiTransport::Inproc);
    state.config.real_full = Some(blocked_real_full_info());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut executed = blocked_real_full_info();
    executed.status = "ready".to_owned();
    executed.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    executed.request_prefill_tokens = 33;
    executed.request_prefill_chunks = 3;
    executed.request_decode_budget = 1;
    executed.request_mtp_verify_rows = 2;
    executed.request_mtp_accepted_rows = 1;
    executed.request_candidate_layerwaves = 19;
    executed.request_deferred_layerwaves = 2;
    executed.scheduler_iterations = 9;
    executed.selected_layerwaves = 17;
    executed.sparse_expert_batches = 5;
    executed.scheduler_sparse_tcp_dispatch_batches = 5;
    executed.request_expert_batch_rows = 101;
    executed.request_expert_batch_routes = 808;
    executed.request_expert_prefill_rows = 11;
    executed.request_expert_decode_rows = 12;
    executed.request_expert_mtp_verify_rows = 78;
    executed.request_expert_prefill_routes = 88;
    executed.request_expert_decode_routes = 96;
    executed.request_expert_mtp_verify_routes = 624;
    executed.kv_read_blocks = 11;
    executed.committed_kv_writes = 13;
    executed.tentative_kv_writes = 7;
    executed.request_committed_mtp_writes = 14;
    executed.request_discarded_mtp_writes = 15;
    executed.request_backed_kv_writes = 27;
    executed.request_backed_kv_bytes = 28_672;
    executed.request_kv_reservation_bytes = 32_768;
    executed.request_byte_backed_scheduler_trace = true;
    executed.scheduler_numeric_progression_source_rows = 99;
    executed.scheduler_numeric_progression_hidden_dim = 6_144;
    executed.scheduler_numeric_progression_visible_checksum = 12_345.0;
    executed.scheduler_numeric_progression_rejected_mtp_checksum = 678.0;
    executed.request_numeric_progression_selected_prefill_rows = 21;
    executed.request_numeric_progression_selected_decode_rows = 22;
    executed.request_numeric_progression_selected_mtp_rows = 23;
    executed.request_numeric_progression_attention_value_updates = 24;
    executed.request_numeric_progression_mlp_value_updates = 25;
    executed.scheduler_full_context_device_attention_complete = true;
    executed.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
    executed.scheduler_terminal_lm_head_sample_passed = true;
    executed.scheduler_terminal_lm_head_covers_full_vocabulary = true;
    executed.scheduler_terminal_lm_head_logits_evaluated =
        executed.scheduler_terminal_lm_head_vocab_size;
    executed.scheduler_terminal_lm_head_top_token_id =
        executed.scheduler_terminal_lm_head_sampled_token_id;
    executed.scheduler_terminal_lm_head_sampled_text = Some("live-scheduler-token".to_owned());
    executed.scheduler_terminal_lm_head_blocker = None;
    executed.blocker.clear();
    executed.failed_requirements.clear();
    state.config.real_full_executor = Some(Arc::new(CapturingRealFullExecutor {
        requests: Arc::clone(&requests),
        info: executed,
    }));

    let mut request = base_request("Use real full.");
    request.model = format!("{}-full", DEFAULT_MODEL_ID);
    request.max_tokens = Some(1);
    let output = build_completion(&state, request).await.unwrap();

    assert_eq!(output.content.as_deref(), Some("live-scheduler-token"));
    assert_eq!(output.metrics.prefill_chunk_count, 3);
    assert_eq!(output.metrics.layerwave_prefill_rows, 33);
    assert_eq!(output.metrics.layerwave_decode_rows, 1);
    let captured = requests.lock().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].request_index, 1);
    assert_eq!(captured[0].request_id, "real-ds4-full-api-1");
    assert_eq!(captured[0].sequence_id, "real-ds4-full-api-1-sequence");
    assert_eq!(
        captured[0].prompt,
        "<｜begin▁of▁sentence｜><｜User｜>Use real full.<｜Assistant｜></think>"
    );
    assert_eq!(captured[0].prompt_tokens, 3);
    assert_eq!(captured[0].max_tokens, 1);
    assert!(captured[0].generated_token_ids.is_empty());
    assert_eq!(captured[0].decode_step_index, 0);
    assert_eq!(captured[0].decode_budget, 1);
    let diagnostics = output.metrics.real_full.as_ref().unwrap();
    assert_eq!(diagnostics.status, "ready");
    assert!(diagnostics.request_scheduler_summary_runtime_reported);
    assert_eq!(diagnostics.request_prefill_tokens, 33);
    assert_eq!(diagnostics.request_prefill_chunks, 3);
    assert_eq!(diagnostics.request_decode_budget, 1);
    assert_eq!(diagnostics.request_layerwaves, 17);
    assert_eq!(diagnostics.scheduler_sparse_tcp_dispatch_batches, 5);
    assert!(diagnostics.scheduler_full_context_device_attention_complete);
    assert!(diagnostics.scheduler_terminal_lm_head_sample_passed);
}

mod real_slice;

#[tokio::test]
async fn invalid_role_returns_openai_error_metadata() {
    let state = test_state(ApiBackend::Tiny, ApiTransport::Inproc);
    let mut request = base_request("hello");
    request.messages[0].role = "invalid".to_owned();
    let err = build_completion(&state, request).await.unwrap_err();
    assert_eq!(err.status, StatusCode::BAD_REQUEST);
    assert_eq!(err.param.as_deref(), Some("messages[0].role"));
}

mod upstream_protocol_errors;
mod upstream_stop_strings;
mod upstream_streaming;
mod upstream_tool_calls;
mod upstream_sampler;
mod upstream_native_v41;

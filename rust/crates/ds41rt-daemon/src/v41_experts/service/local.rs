//! The GPU owner polls admitted QPs and sends results without work queues.
use super::*;
use ds41rt_transport::{LocalVerbsExpertConnection, ProtocolV2ExecutorResponseRef};
use std::{
    net::TcpListener,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

struct Admission {
    stop: Arc<AtomicBool>,
}
impl Drop for Admission {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
    }
}

pub(super) fn run(config: NativeExpertServiceConfig, listen: &str) -> Result<()> {
    ensure!(config.rank < 4, "native rank must be 0..3");
    ensure!(
        matches!(config.capacity, 1 | 16 | 80 | 256 | 1024 | 4096),
        "unsupported native capacity"
    );
    ensure!(
        (128 + 10240 + 40 + 6 * 12..=64 * 1024 * 1024).contains(&config.max_frame_bytes),
        "invalid native frame budget"
    );
    let library = unsafe { NativeLibrary::load(&config.library) }?;
    let (weights, remaining) = load_weights(&library, &config)?;
    let mut execution = weights.execution(&library, &config, remaining)?;
    let mut exchange = HostExpertExchange::new(config.capacity)?;
    let mut row_indices = vec![0; config.capacity as usize];
    let listener = TcpListener::bind(listen)?;
    listener.set_nonblocking(true)?;
    let (admit, incoming) = mpsc::sync_channel(2);
    let stop = Arc::new(AtomicBool::new(false));
    let _guard = Admission { stop: stop.clone() };
    let max_frame_bytes = config.max_frame_bytes;
    // Resolved once here: the admission thread and the poll loop must not read
    // the process environment per connection or per poll.
    let protocol_v2_timing = ds41rt_transport::protocol_v2_timing_from_env();
    thread::Builder::new()
        .name("v41-roce-bootstrap".into())
        .spawn(move || {
            while !stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        match LocalVerbsExpertConnection::accept(stream, max_frame_bytes, protocol_v2_timing) {
                            Ok(connection) => {
                                if admit.try_send(connection).is_err() {
                                    tracing::warn!("native RoCE admission queue full or stopped");
                                }
                            }
                            Err(error) => tracing::warn!(%error, "native RoCE bootstrap failed"),
                        }
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => {
                        tracing::error!(%error, "native RoCE listener failed");
                        break;
                    }
                }
            }
        })?;
    let executor_id = ds41rt_transport::v41_expert::v41_spark_executor_id(config.world, config.rank)?;
    let mut connections = Vec::<LocalVerbsExpertConnection>::with_capacity(16);
    tracing::info!(
        rank = config.rank,
        world = config.world,
        capacity = config.capacity,
        first_layer = config.first_layer,
        layers = weights.len(),
        "native local RoCE expert worker ready"
    );
    loop {
        if connections.is_empty() {
            connections.push(incoming.recv().context("native RoCE admission stopped")?);
        } else if let Ok(connection) = incoming.try_recv() {
            if connections.len() < 16 {
                connections.push(connection);
            } else {
                tracing::warn!("native RoCE active connection limit reached");
            }
        }
        let mut index = 0;
        while index < connections.len() {
            let mut execution_failed = false;
            let result = connections[index].poll(|view, mapped, emit| {
                let request = if execution.is_paired() {
                    V41BackboneRequest::parse_paired(view.frame_bytes(), config.capacity)?
                } else {
                    V41BackboneRequest::parse(view.frame_bytes(), config.capacity)?
                };
                let layer = (request.layer() as usize).checked_sub(config.first_layer)
                    .context("requested expert layer is not resident on this Spark")?;
                execution.bind_layer(&weights, layer)?;
                if let Some(slot) = mapped.response_slot {
                    let response = unsafe { execution.execute_mapped_request(&request,
                        executor_id, &mut exchange, slot) };
                    let response = match response {
                        Ok(response) => response,
                        Err(error) => { execution_failed = true; return Err(error); }
                    };
                    if let Some(response) = response {
                        return emit(ProtocolV2ExecutorResponseRef::Device(response));
                    }
                }
                let mut emit_failed = false;
                let result = execution.execute_host_chunks(
                    &request,
                    executor_id,
                    &mut exchange,
                    &mut row_indices,
                    config.max_frame_bytes,
                    |response| {
                        let result = emit(ProtocolV2ExecutorResponseRef::Host(response));
                        emit_failed |= result.is_err();
                        result
                    },
                );
                execution_failed = result.is_err() && !emit_failed;
                result
            });
            match result {
                Ok(_) => index += 1,
                Err(error) => {
                    if execution_failed {
                        return Err(error).context("native GPU execution failed");
                    }
                    tracing::warn!(%error, "native RoCE peer removed");
                    connections.swap_remove(index);
                }
            }
        }
        std::hint::spin_loop();
    }
}

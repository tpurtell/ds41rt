use super::{tests::request, V41BackboneRequest, V41Tp4Tcp, V41_PARTIAL_ROW_BYTES};
use crate::{ExpertProtocolV2Request, TcpTransportConfig, EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN};
use anyhow::Result;
use std::{sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{mpsc, Semaphore},
    time::timeout,
};

const FRAME: usize = 200_000;
const EXECUTORS: [u64; 4] = [11, 12, 13, 14];
async fn read_request(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut frame = vec![0; EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN];
    stream.read_exact(&mut frame).await?;
    let bytes = ExpertProtocolV2Request::wire_bytes_from_header(&frame)?;
    assert!(bytes <= FRAME);
    frame.resize(bytes, 0);
    stream
        .read_exact(&mut frame[EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN..])
        .await?;
    Ok(frame)
}
async fn respond(stream: &mut TcpStream, frame: &[u8], rank: usize) -> Result<()> {
    let native = V41BackboneRequest::parse(frame, 2)?;
    let id = ExpertProtocolV2Request::decode(frame)?.header.request_id;
    for row in 0..native.rows() {
        let partials = vec![(id + rank as u64 + row as u64) as u8; V41_PARTIAL_ROW_BYTES as usize];
        let mut indices = [0u32];
        let response =
            native.response_chunk(EXECUTORS[rank], row, &partials, &mut indices, FRAME)?;
        stream.write_all(&response.to_owned()?.encode()?).await?;
    }
    Ok(())
}
fn config() -> TcpTransportConfig { timing: false,
    TcpTransportConfig { timing: false,
        timeout: Duration::from_secs(3),
        max_frame_bytes: FRAME,
    }
}
fn check_chunk(id: u64, rank: usize, row: u32, bytes: &[u8]) -> Result<()> {
    assert_eq!(bytes.len(), V41_PARTIAL_ROW_BYTES as usize);
    assert!(bytes
        .iter()
        .all(|&b| b == (id + rank as u64 + row as u64) as u8));
    Ok(())
}

#[tokio::test]
async fn native_tcp_dispatch_finishes_before_responses_and_reuses_connections() -> Result<()> {
    let mut peers = Vec::new();
    let mut servers = Vec::new();
    let gates = Arc::new(Semaphore::new(0));
    let (seen, mut observed) = mpsc::channel(8);
    for rank in 0..4 {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        peers.push(listener.local_addr()?);
        let gate = gates.clone();
        let seen = seen.clone();
        servers.push(tokio::spawn(async move {
            // Both requests must arrive on this same persistent connection.
            let (mut stream, _) = listener.accept().await?;
            for cycle in 0..2 {
                let frame = read_request(&mut stream).await?;
                seen.send((rank, cycle)).await?;
                gate.acquire().await?.forget();
                respond(&mut stream, &frame, rank).await?;
            }
            Ok::<_, anyhow::Error>(())
        }));
    }
    let mut transport = V41Tp4Tcp::new(peers.try_into().unwrap(), EXECUTORS, 2, config())?;
    for cycle in 0..2 {
        let mut req = request(2);
        req.header.request_id += cycle as u64;
        let pending = timeout(Duration::from_secs(3), transport.dispatch(&req)).await??;
        let mut ranks = Vec::new();
        for _ in 0..4 {
            let (rank, received_cycle) = timeout(Duration::from_secs(3), observed.recv())
                .await?
                .unwrap();
            assert_eq!(received_cycle, cycle);
            ranks.push(rank);
        }
        ranks.sort();
        assert_eq!(ranks, [0, 1, 2, 3]);
        // No worker is allowed to reply until dispatch has returned. A combined
        // write/read implementation would deadlock and fail the timeout above.
        gates.add_permits(4);
        let mut chunks = 0;
        pending
            .receive(|rank, row, bytes| {
                chunks += 1;
                check_chunk(req.header.request_id, rank, row, bytes)
            })
            .await?;
        assert_eq!(chunks, 8);
    }
    for server in servers {
        timeout(Duration::from_secs(3), server).await???;
    }
    Ok(())
}

#[tokio::test]
async fn native_tcp_dropped_dispatch_closes_unread_responses_and_reconnects() -> Result<()> {
    let mut peers = Vec::new();
    let mut servers = Vec::new();
    let (seen, mut observed) = mpsc::channel(8);
    let (closed, mut closures) = mpsc::channel(4);
    for rank in 0..4 {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        peers.push(listener.local_addr()?);
        let seen = seen.clone();
        let closed = closed.clone();
        servers.push(tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await?;
            let frame = read_request(&mut stream).await?;
            respond(&mut stream, &frame, rank).await?;
            seen.send(rank).await?;
            let mut byte = [0u8];
            match stream.read(&mut byte).await {
                Ok(0) => {}
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => {}
                other => panic!("abandoned socket must close: {other:?}"),
            }
            closed.send(rank).await?;
            let (mut next, _) = listener.accept().await?;
            let frame = read_request(&mut next).await?;
            respond(&mut next, &frame, rank).await?;
            Ok::<_, anyhow::Error>(())
        }));
    }
    let mut transport = V41Tp4Tcp::new(peers.try_into().unwrap(), EXECUTORS, 2, config())?;
    let first = request(1);
    let pending = transport.dispatch(&first).await?;
    for _ in 0..4 {
        timeout(Duration::from_secs(3), observed.recv())
            .await?
            .unwrap();
    }
    drop(pending);
    for _ in 0..4 {
        timeout(Duration::from_secs(3), closures.recv())
            .await?
            .unwrap();
    }
    let mut next = request(2);
    next.header.request_id += 10;
    let mut chunks = 0;
    transport
        .execute(&next, |rank, row, bytes| {
            chunks += 1;
            check_chunk(next.header.request_id, rank, row, bytes)
        })
        .await?;
    assert_eq!(chunks, 8);
    for server in servers {
        timeout(Duration::from_secs(3), server).await???;
    }
    Ok(())
}

#[tokio::test]
async fn native_tcp_cancelled_receive_discards_partial_wave() -> Result<()> {
    let mut peers = Vec::new();
    let mut servers = Vec::new();
    let (sent, mut written) = mpsc::channel(4);
    let (closed, mut closures) = mpsc::channel(4);
    for rank in 0..4 {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        peers.push(listener.local_addr()?);
        let sent = sent.clone();
        let closed = closed.clone();
        servers.push(tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await?;
            let frame = read_request(&mut stream).await?;
            let native = V41BackboneRequest::parse(&frame, 2)?;
            let partials = vec![0; V41_PARTIAL_ROW_BYTES as usize];
            let mut indices = [0];
            let first =
                native.response_chunk(EXECUTORS[rank], 0, &partials, &mut indices, FRAME)?;
            // The first chunk is valid, but no final chunk can arrive.
            stream.write_all(&first.to_owned()?.encode()?).await?;
            sent.send(rank).await?;
            let mut byte = [0];
            match stream.read(&mut byte).await {
                Ok(0) => {}
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => {}
                other => panic!("cancelled receive must close socket: {other:?}"),
            }
            closed.send(rank).await?;
            let (mut next, _) = listener.accept().await?;
            let frame = read_request(&mut next).await?;
            respond(&mut next, &frame, rank).await?;
            Ok::<_, anyhow::Error>(())
        }));
    }
    let mut transport = V41Tp4Tcp::new(peers.try_into().unwrap(), EXECUTORS, 2, config())?;
    let first = request(2);
    let pending = transport.dispatch(&first).await?;
    for _ in 0..4 {
        timeout(Duration::from_secs(3), written.recv())
            .await?
            .unwrap();
    }
    let (progress, mut copied) = mpsc::unbounded_channel();
    {
        let receive = pending.receive(|rank, row, _| {
            assert_eq!(row, 0);
            progress.send(rank)?;
            Ok(())
        });
        tokio::pin!(receive);
        tokio::select! {
            result = &mut receive => panic!("incomplete wave must remain pending: {result:?}"),
            notification = timeout(Duration::from_secs(3), copied.recv()) => { notification?.unwrap(); }
        }
        // Scope exit cancels after at least one validated destination write.
    }
    for _ in 0..4 {
        timeout(Duration::from_secs(3), closures.recv())
            .await?
            .unwrap();
    }
    let mut next = request(2);
    next.header.request_id += 20;
    let mut chunks = 0;
    transport
        .execute(&next, |rank, row, bytes| {
            chunks += 1;
            check_chunk(next.header.request_id, rank, row, bytes)
        })
        .await?;
    assert_eq!(chunks, 8);
    for server in servers {
        timeout(Duration::from_secs(3), server).await???;
    }
    Ok(())
}

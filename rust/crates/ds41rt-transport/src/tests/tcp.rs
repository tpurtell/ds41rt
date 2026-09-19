use anyhow::Result;
use std::time::Duration;
use tokio::net::TcpListener;

use super::common::{request_with_hidden_len, spawn_server};
use crate::{tcp_roundtrip, TcpTransportConfig, DEFAULT_MAX_FRAME_BYTES};

#[tokio::test]
async fn small_and_large_payloads_roundtrip() -> Result<()> {
    let (addr, shutdown) = spawn_server().await?;
    for (request_id, hidden_len) in [(1, 1), (2, 16_384)] {
        let request = request_with_hidden_len(request_id, hidden_len);
        let response = tcp_roundtrip(addr, &request, TcpTransportConfig::default()).await?;
        assert_eq!(response.partial_outputs.len(), request.rows.len());
        assert_eq!(response.partial_outputs[0].len(), hidden_len);
    }
    let _ = shutdown.send(());
    Ok(())
}

#[tokio::test]
async fn timeout_behavior_is_reported() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        let Ok((_stream, _peer)) = listener.accept().await else {
            return;
        };
        tokio::time::sleep(Duration::from_millis(250)).await;
    });
    let request = request_with_hidden_len(3, 4);
    let err = tcp_roundtrip(
        addr,
        &request,
        TcpTransportConfig { timing: false,
            timeout: Duration::from_millis(25),
            max_frame_bytes: DEFAULT_MAX_FRAME_BYTES,
        },
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("timed out"));
    Ok(())
}

#[tokio::test]
async fn connection_close_is_reported() -> Result<()> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        let _ = listener.accept().await;
    });
    let request = request_with_hidden_len(4, 4);
    let err = tcp_roundtrip(addr, &request, TcpTransportConfig::default())
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("reading transport frame") || err.contains("reset"));
    Ok(())
}

#[tokio::test]
async fn concurrent_requests_are_queued_by_tcp_listener() -> Result<()> {
    let (addr, shutdown) = spawn_server().await?;
    let mut tasks = Vec::new();
    for request_id in 10..26 {
        tasks.push(tokio::spawn(async move {
            let request = request_with_hidden_len(request_id, 32);
            tcp_roundtrip(addr, &request, TcpTransportConfig::default()).await
        }));
    }
    for task in tasks {
        let response = task.await??;
        assert_eq!(response.status, "ok");
    }
    let _ = shutdown.send(());
    Ok(())
}

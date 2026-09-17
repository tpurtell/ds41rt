//! Upstream-ported container/quant-format validation invariants.
//!
//! Ported from:
//! - `llama.cpp/gguf-py/tests/test_gguf_reader_validation.py` (adversarial
//!   header cases: absurd dimension-count bound, dims-product uint64
//!   wraparound).
//! - `llama.cpp/tests/test-gguf.cpp` (property tests over handcrafted files:
//!   roundtrip reads in multiple read modes, randomized header parsing).
//!
//! ds41rt-loader's container surface is `read_safetensors_metadata` (safetensors
//! header parsing in `catalog.rs`), the contract byte-size derivation
//! `NativeDeepseekV4AttentionTensorSpec::byte_length` (`attention_format.rs`),
//! and the tensor row readers (`tensors.rs`). The invariants are ported against
//! those parsers.
//!
//! Mapping notes / gaps:
//! - GGUF's `GGML_MAX_DIMS` upper bound has no ds41rt analog: the safetensors
//!   header parser accepts arbitrarily high tensor rank (see the ignored
//!   BUG(candidate) test below). The "must not read past EOF" half of that
//!   invariant IS covered: declared extents are validated against the real
//!   file length before any data read.
//! - The dims-product wraparound invariant maps directly onto
//!   `byte_length()`'s checked arithmetic and the row readers' checked window
//!   math.
//! - The upstream "set-kv roundtrip" property has no ds41rt analog: ds41rt-loader
//!   never writes containers or mutates KV metadata, so there is no writer
//!   surface to roundtrip. Roundtrip coverage instead pins that all read modes
//!   over one handcrafted file agree byte-for-byte.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use ds41rt_core::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole};
use ds41rt_loader::{
    load_tensor_bytes, load_tensor_rows, read_safetensors_metadata, read_tensor_bytes_into,
    read_tensor_row_window_into, NativeDeepseekV4AttentionTensorFamily,
    NativeDeepseekV4AttentionTensorSpec,
};

/// Write a minimal but valid single-file safetensors container.
///
/// `entries` are (name, dtype string, shape, data bytes); data_offsets are
/// assigned sequentially in the given order.
fn write_safetensors(
    path: &Path,
    entries: &[(&str, &str, Vec<usize>, Vec<u8>)],
) -> (u64, Vec<u8>) {
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, dtype, shape, bytes) in entries {
        let start = data.len() as u64;
        data.extend_from_slice(bytes);
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, data.len() as u64],
            }),
        );
    }
    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }
    let data_start = 8 + header_bytes.len() as u64;
    let mut file = File::create(path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&data).unwrap();
    (data_start, data)
}

fn minimal_facts() -> ModelFacts {
    ModelFacts::default()
}

fn catalog_for(dir: &Path, _file_name: &str, tensors: Vec<TensorInfo>) -> TensorCatalog {
    TensorCatalog {
        model_id: "upstream-invariants".to_owned(),
        snapshot_path: dir.display().to_string(),
        facts: minimal_facts(),
        tensors,
    }
}

fn tensor_info(name: &str, file: &str, dtype: DType, shape: Vec<usize>, data_start: u64, bytes: &[u8]) -> TensorInfo {
    TensorInfo {
        name: name.to_owned(),
        file: file.to_owned(),
        dtype,
        shape,
        byte_offset: data_start,
        byte_length: bytes.len() as u64,
        role: TensorRole::Other,
        layer_id: None,
        expert_id: None,
        is_quantization_metadata: false,
    }
}

// ---------------------------------------------------------------------------
// Adversarial cases (ported from test_gguf_reader_validation.py)
// ---------------------------------------------------------------------------

#[test]
fn header_length_past_eof_is_rejected_without_overread() {
    // Ported invariant: a crafted container whose declared header extent
    // exceeds the actual file must be rejected, not read past EOF.
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("evil_header_len.safetensors");
    let mut file = File::create(&path).unwrap();
    // Claim a 1 MiB header in a file that only contains the length prefix.
    file.write_all(&(1_048_576_u64).to_le_bytes()).unwrap();

    let error = read_safetensors_metadata(&path).unwrap_err();
    let message = format!("{error:#}").to_lowercase();
    assert!(
        message.contains("header") || message.contains("eof") || message.contains("past end"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn header_length_over_64mib_is_rejected_before_allocation() {
    // Ported invariant: an absurd declared bound must be rejected up front
    // (the parser caps the header at 64 MiB) rather than driving a huge
    // allocation/read.
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("evil_huge_header.safetensors");
    let mut file = File::create(&path).unwrap();
    file.write_all(&(64_u64 * 1024 * 1024 + 1).to_le_bytes()).unwrap();

    let error = read_safetensors_metadata(&path).unwrap_err();
    assert!(
        format!("{error:#}").contains("64 MiB"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn tensor_extent_past_eof_is_rejected_without_overread() {
    // Ported invariant (GGUF "absurd n_dims" analog): a tensor whose declared
    // byte extent reaches beyond the end of the file must be rejected at
    // header-parse time, so no data read can overrun the file.
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("evil_extent.safetensors");
    let header = serde_json::json!({
        "bad_tensor": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 1_000_000_000],
        },
    });
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }
    let mut file = File::create(&path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&[0_u8; 16]).unwrap();

    let error = read_safetensors_metadata(&path).unwrap_err();
    assert!(
        format!("{error:#}").contains("extends beyond"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn inverted_data_offsets_are_rejected() {
    // data_offsets[0] > data_offsets[1] would underflow byte-length math.
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("evil_inverted.safetensors");
    let header = serde_json::json!({
        "bad_tensor": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [16, 0],
        },
    });
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }
    let mut file = File::create(&path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&[0_u8; 16]).unwrap();

    let error = read_safetensors_metadata(&path).unwrap_err();
    assert!(
        format!("{error:#}").contains("invalid safetensors offsets"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn attention_spec_dims_product_has_no_uint64_wraparound() {
    // Direct port of test_dims_product_no_uint64_wraparound: dims whose true
    // product overflows uint64 (np.prod wraps them to 4) must be rejected by
    // the contract byte-size derivation instead of silently passing an
    // undersized byte count.
    let dims = vec![4_194_305_usize, 4_194_305, 211_106_198_978_564];
    // Sanity-check the port: unchecked u64 multiplication wraps to 4, which
    // is exactly the upstream bug being pinned.
    let wrapped = dims
        .iter()
        .fold(1_u64.wrapping_mul(1), |acc, dim| acc.wrapping_mul(*dim as u64));
    assert_eq!(wrapped, 4);

    let spec = NativeDeepseekV4AttentionTensorSpec {
        name: "evil.adversarial".to_owned(),
        family: NativeDeepseekV4AttentionTensorFamily::Main,
        dtype: DType::F32,
        shape: dims,
        role: TensorRole::Attention,
        logical_layer_id: 0,
        is_quantization_metadata: false,
    };
    let error = spec.byte_length().unwrap_err();
    assert!(
        format!("{error:#}").contains("overflow"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn attention_spec_dims_product_at_boundary_is_accepted() {
    // Counterpart invariant: a dims product that exactly fits in u64 must not
    // be falsely rejected.
    let dims = vec![(u64::MAX / 16) as usize, 4];
    let spec = NativeDeepseekV4AttentionTensorSpec {
        name: "ok.boundary".to_owned(),
        family: NativeDeepseekV4AttentionTensorFamily::Main,
        dtype: DType::F32,
        shape: dims.clone(),
        role: TensorRole::Attention,
        logical_layer_id: 0,
        is_quantization_metadata: false,
    };
    let expected: u64 = dims
        .iter()
        .map(|dim| *dim as u128)
        .product::<u128>()
        .checked_mul(4)
        .unwrap()
        .try_into()
        .expect("dims product times dtype size fits in u64");
    assert_eq!(spec.byte_length().unwrap(), expected);
}

#[test]
fn row_window_byte_math_has_no_usize_wraparound() {
    // Container-level analog of the dims-product wraparound: a rank-2 tensor
    // whose declared shape would wrap usize when converted to byte offsets
    // must be rejected by the row readers, not read with a wrapped length.
    let tempdir = tempfile::tempdir().unwrap();
    let file_name = "rows.safetensors";
    let path = tempdir.path().join(file_name);
    // Craft a real catalog entry with adversarial shape; only a handful of
    // real bytes exist on disk.
    let (data_start, data) = write_safetensors(
        &path,
        &[("w", "F32", vec![2, 2], vec![0_u8; 16])],
    );
    let mut info = tensor_info("w", file_name, DType::F32, vec![2, 2], data_start, &data);
    // Lie about the geometry: row width overflows usize when multiplied by
    // the dtype byte width, and the row count multiplication would wrap too.
    info.shape = vec![usize::MAX / 2, usize::MAX / 2];
    info.byte_length = u64::MAX;
    let catalog = catalog_for(tempdir.path(), file_name, vec![info]);

    let error = load_tensor_rows(&catalog, "w", 0, 1).unwrap_err();
    assert!(
        format!("{error:#}").contains("overflow"),
        "unexpected error: {error:#}"
    );

    let mut dst = vec![0_u8; 64];
    let error = read_tensor_row_window_into(&catalog, "w", 0, 1, 0, 1, &mut dst).unwrap_err();
    assert!(
        format!("{error:#}").contains("overflow"),
        "unexpected error: {error:#}"
    );
}

fn write_rank_header(path: &std::path::Path, rank: usize) {
    let header = serde_json::json!({
        "bad_tensor": {
            "dtype": "F32",
            "shape": vec![1_u64; rank],
            "data_offsets": [0, 4],
        },
    });
    let mut header_bytes = serde_json::to_vec(&header).unwrap();
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }
    let mut file = File::create(path).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&[0_u8; 16]).unwrap();
}

#[test]
fn tensor_rank_policy_boundaries_and_diagnostic() {
    // Rank 32 is ds41rt's policy ceiling (ported from the upstream gguf-py
    // n_dims-bound class); 33 must be rejected with a diagnostic naming the
    // tensor, the observed rank, and the maximum. Review 2026-09-15: the
    // original test only exercised rank 1_000_000 and discarded the error.
    let tempdir = tempfile::tempdir().unwrap();

    let ok_path = tempdir.path().join("rank32.safetensors");
    write_rank_header(&ok_path, 32);
    read_safetensors_metadata(&ok_path).expect("rank 32 must be accepted");

    let bad_path = tempdir.path().join("rank33.safetensors");
    write_rank_header(&bad_path, 33);
    let err = read_safetensors_metadata(&bad_path).unwrap_err().to_string();
    assert!(err.contains("bad_tensor"), "diagnostic names the tensor: {err}");
    assert!(err.contains("absurd rank 33"), "diagnostic carries the observed rank: {err}");
    assert!(err.contains("max 32"), "diagnostic carries the bound: {err}");

    let absurd_path = tempdir.path().join("absurd.safetensors");
    write_rank_header(&absurd_path, 1_000_000);
    assert!(read_safetensors_metadata(&absurd_path).is_err());
}

// ---------------------------------------------------------------------------
// Property tests (ported from test-gguf.cpp handcrafted/randomized suites)
// ---------------------------------------------------------------------------

#[test]
fn handcrafted_file_roundtrips_across_three_read_modes() {
    // Ported invariant: bytes written into a handcrafted container must come
    // back byte-identical through every read mode the loader exposes
    // (whole-tensor load, into-buffer read, row read). ds41rt has no GGUF
    // writer/set-kv surface, so the roundtrip is read-side only.
    let tempdir = tempfile::tempdir().unwrap();
    let file_name = "handcrafted.safetensors";
    let path = tempdir.path().join(file_name);
    let weights: Vec<u8> = (0..96_u8).map(|value| value.wrapping_mul(37)).collect();
    let (data_start, data) = write_safetensors(
        &path,
        &[
            ("alpha", "F32", vec![4, 6], weights[..96].to_vec()),
            ("beta", "BF16", vec![3], weights[..6].to_vec()),
        ],
    );
    assert_eq!(data.len(), 96 + 6);

    let catalog = catalog_for(
        tempdir.path(),
        file_name,
        vec![
            tensor_info("alpha", file_name, DType::F32, vec![4, 6], data_start, &data[..96]),
            tensor_info("beta", file_name, DType::Bf16, vec![3], data_start + 96, &data[96..]),
        ],
    );

    // Read mode 1: whole-tensor load.
    let whole = load_tensor_bytes(&catalog, "alpha").unwrap();
    assert_eq!(whole.bytes, weights[..96]);

    // Read mode 2: caller-provided buffer.
    let mut dst = vec![0_u8; 96];
    let summary = read_tensor_bytes_into(&catalog, "alpha", &mut dst).unwrap();
    assert_eq!(dst, weights[..96]);
    assert_eq!(summary.bytes_read, 96);

    // Read mode 3: row-based read covering the full tensor.
    let rows = load_tensor_rows(&catalog, "alpha", 0, 4).unwrap();
    assert_eq!(rows.bytes, weights[..96]);
    assert_eq!(rows.row_width, 6);

    // Rank-1 tensor via whole-tensor read as well.
    let beta = load_tensor_bytes(&catalog, "beta").unwrap();
    assert_eq!(beta.bytes, weights[..6]);
}

#[test]
fn randomized_handcrafted_headers_parse_consistently() {
    // Ported invariant (randomized suite): for many seeded handcrafted
    // containers, parsed metadata must agree exactly with what was written:
    // names, dtypes, shapes, absolute offsets, and byte_length == dtype size
    // x dims product. Deterministic xorshift PRNG, no external crates.
    let mut state = 0x9E3779B97F4A7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let dtypes: &[(&str, usize, DType)] = &[
        ("BF16", 2, DType::Bf16),
        ("F16", 2, DType::F16),
        ("F32", 4, DType::F32),
        ("I8", 1, DType::I8),
        ("I64", 8, DType::I64),
        ("U8", 1, DType::U8),
    ];

    for round in 0..24 {
        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path().join(format!("random_{round}.safetensors"));
        let tensor_count = 1 + (next() % 6) as usize;
        let mut written: Vec<(String, DType, usize, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut total = 0_usize;
        for index in 0..tensor_count {
            let (_, dtype_bytes, dtype) = &dtypes[(next() as usize) % dtypes.len()];
            let dtype = dtype.clone();
            let rank = 1 + (next() % 3) as usize;
            let shape: Vec<usize> = (0..rank)
                .map(|_| 1 + (next() % 8) as usize)
                .collect();
            let element_count: usize = shape.iter().product();
            let nbytes = element_count * dtype_bytes;
            let bytes: Vec<u8> = (0..nbytes).map(|_| next() as u8).collect();
            written.push((
                format!("tensor_{index}"),
                dtype,
                *dtype_bytes,
                shape.clone(),
                bytes.clone(),
            ));
            total += nbytes;
        }

        let entries: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = written
            .iter()
            .map(|(name, dtype, _, shape, bytes)| {
                (
                    name.as_str(),
                    dtype_str(dtype.clone()),
                    shape.clone(),
                    bytes.clone(),
                )
            })
            .collect();
        let (data_start, file_data) = write_safetensors(&path, &entries);
        assert_eq!(file_data.len(), total);

        let metadata = read_safetensors_metadata(&path).unwrap();
        assert_eq!(metadata.len(), written.len());
        // BTreeMap ordering: names come back sorted.
        let mut expected = written.clone();
        expected.sort_by(|a, b| a.0.cmp(&b.0));
        for (parsed, (name, dtype, dtype_bytes, shape, bytes)) in
            metadata.iter().zip(expected.iter())
        {
            assert_eq!(&parsed.name, name);
            assert_eq!(parsed.dtype, *dtype);
            assert_eq!(parsed.shape, *shape);
            assert_eq!(
                parsed.byte_length as usize,
                shape.iter().product::<usize>() * dtype_bytes
            );
            // Absolute offset points at the right slice of the data region.
            let rel = (parsed.byte_offset - data_start) as usize;
            assert_eq!(&file_data[rel..rel + bytes.len()], bytes.as_slice());
        }
    }
}

fn dtype_str(dtype: DType) -> &'static str {
    match dtype {
        DType::Bf16 => "BF16",
        DType::F16 => "F16",
        DType::F32 => "F32",
        DType::I8 => "I8",
        DType::I64 => "I64",
        DType::U8 => "U8",
        other => panic!("unsupported in randomized test: {other:?}"),
    }
}

#[test]
fn randomized_row_windows_read_back_written_bytes() {
    // Companion randomized property: every row window of a handcrafted rank-2
    // tensor reads back exactly the bytes that were written there.
    let mut state = 0xDEADBEEFCAFEF00D_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    for round in 0..12 {
        let tempdir = tempfile::tempdir().unwrap();
        let file_name = format!("rows_{round}.safetensors");
        let path = tempdir.path().join(&file_name);
        let rows = 1 + (next() % 8) as usize;
        let width = 1 + (next() % 16) as usize;
        let bytes: Vec<u8> = (0..rows * width * 4)
            .map(|_| next() as u8)
            .collect();
        let (data_start, data) = write_safetensors(
            &path,
            &[("matrix", "F32", vec![rows, width], bytes.clone())],
        );
        let catalog = catalog_for(
            tempdir.path(),
            &file_name,
            vec![tensor_info(
                "matrix",
                &file_name,
                DType::F32,
                vec![rows, width],
                data_start,
                &data,
            )],
        );

        let start_row = (next() as usize) % rows;
        let row_count = 1 + (next() as usize) % (rows - start_row);
        let start_col = (next() as usize) % width;
        let col_count = 1 + (next() as usize) % (width - start_col);

        let mut dst = vec![0_u8; row_count * col_count * 4];
        let summary = read_tensor_row_window_into(
            &catalog,
            "matrix",
            start_row,
            row_count,
            start_col,
            col_count,
            &mut dst,
        )
        .unwrap();
        assert_eq!(summary.bytes_read as usize, dst.len());

        for (row_index, window) in dst.chunks(col_count * 4).enumerate() {
            let source_row = start_row + row_index;
            let row_start = (source_row * width + start_col) * 4;
            assert_eq!(
                window,
                &bytes[row_start..row_start + col_count * 4],
                "round {round} row {source_row}"
            );
        }
    }
}

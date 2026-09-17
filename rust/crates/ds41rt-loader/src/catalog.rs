use crate::snapshot::resolve_snapshot;
use crate::{
    exl3_format::exl3_recipe_from_quantization_config, is_deepseek_v4_exl3_recipe,
    validate_exl3_expert_catalog, validate_native_deepseek_v4_attention_catalog,
    validate_native_deepseek_v4_dspark_catalog, validate_native_fp4_expert_catalog,
    DEEPSEEK_V4_EXL3_RECIPE,
};
use anyhow::{Context, Result};
use ds41rt_core::{
    DType, ModelFacts, ModelVariant, TensorCatalog, TensorInfo, TensorRole, DS4_FLASH_HIDDEN_SIZE,
    DS4_FLASH_NUM_HIDDEN_LAYERS, DS4_FLASH_ROUTED_EXPERTS, DS4_PRO_HIDDEN_SIZE,
    DS4_PRO_NUM_HIDDEN_LAYERS, DS4_PRO_ROUTED_EXPERTS,
};
use serde::Deserialize;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const EXTERNAL_QUANTIZATION_CONFIG_FILES: [&str; 2] =
    ["quantize_config.json", "quantization_config.json"];

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsTensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsTensorMetadata {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Deserialize)]
struct ConfigFile {
    model_type: String,
    architectures: Vec<String>,
    hidden_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f32,
    num_hidden_layers: usize,
    num_hash_layers: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    n_shared_experts: usize,
    vocab_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    q_lora_rank: usize,
    o_lora_rank: usize,
    o_groups: usize,
    qk_rope_head_dim: usize,
    index_head_dim: usize,
    index_n_heads: usize,
    index_topk: usize,
    sliding_window: usize,
    max_position_embeddings: usize,
    rope_theta: f32,
    compress_rope_theta: f32,
    rope_scaling: RopeScalingConfig,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
    compress_ratios: Vec<usize>,
    routed_scaling_factor: f32,
    scoring_func: String,
    topk_method: String,
    swiglu_limit: f32,
    expert_dtype: String,
    dspark_block_size: usize,
    dspark_markov_rank: usize,
    dspark_noise_token_id: usize,
    dspark_target_layer_ids: Vec<usize>,
    quantization_config: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct RopeScalingConfig {
    #[serde(rename = "type")]
    kind: String,
    factor: f32,
    original_max_position_embeddings: usize,
    beta_fast: f32,
    beta_slow: f32,
}

fn default_rms_norm_eps() -> f32 {
    1.0e-6
}

pub fn build_catalog(model_id: &str, hf_home: Option<&Path>) -> Result<TensorCatalog> {
    let resolution = resolve_snapshot(model_id, hf_home)?;
    let snapshot_path = resolution
        .snapshot_path
        .as_ref()
        .with_context(|| format!("no local snapshot found for {model_id}"))?;
    build_catalog_for_snapshot(model_id, snapshot_path)
}

pub fn build_catalog_for_snapshot(model_id: &str, snapshot_path: &Path) -> Result<TensorCatalog> {
    let facts = read_model_facts(model_id, snapshot_path)?;
    anyhow::ensure!(
        facts.variant != ModelVariant::Pro || facts.quantization_recipe == DEEPSEEK_V4_EXL3_RECIPE,
        "DeepSeek V4 Pro expert loading requires calibrated recipe {DEEPSEEK_V4_EXL3_RECIPE}; got {}",
        facts.quantization_recipe
    );
    let index_path = snapshot_path.join("model.safetensors.index.json");
    let index: SafetensorsIndex = serde_json::from_reader(
        File::open(&index_path).with_context(|| format!("opening {}", index_path.display()))?,
    )
    .with_context(|| format!("parsing {}", index_path.display()))?;

    let files = index
        .weight_map
        .values()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let header_slots = (0..files.len())
        .map(|_| Mutex::new(None))
        .collect::<Vec<OptionSlot<BTreeMap<String, SafetensorsTensorHeader>>>>();
    let worker_count = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(32)
        .min(files.len().max(1));
    let next_file = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| loop {
                let index = next_file.fetch_add(1, Ordering::Relaxed);
                let Some(file_name) = files.get(index) else {
                    break;
                };
                let path = snapshot_path.join(file_name);
                let parsed = parse_safetensors_header(&path)
                    .with_context(|| format!("parsing safetensors header {}", path.display()));
                *header_slots[index]
                    .lock()
                    .expect("safetensors header result slot is poisoned") = Some(parsed);
            });
        }
    });
    let mut header_by_tensor = BTreeMap::new();
    for (file_name, slot) in files.into_iter().zip(header_slots) {
        let entries = slot
            .into_inner()
            .map_err(|_| anyhow::anyhow!("safetensors header result slot is poisoned"))?
            .with_context(|| format!("safetensors header worker did not visit {file_name}"))??;
        for (name, header) in entries {
            header_by_tensor.insert(name, (file_name.clone(), header));
        }
    }

    let mut tensors = Vec::with_capacity(index.weight_map.len());
    for (name, file_name) in index.weight_map {
        let (header_file, header) = header_by_tensor
            .get(&name)
            .with_context(|| format!("tensor {name} missing from safetensors header"))?;
        if header_file != &file_name {
            anyhow::bail!(
                "index/header file mismatch for {name}: index={file_name} header={header_file}"
            );
        }
        let layer_id = tensor_layer_id(&name, &facts);
        let expert_id = extract_number_after(&name, ".ffn.experts.")
            .or_else(|| extract_number_after(&name, ".mlp.experts."))
            .map(|v| v as u32);
        let dtype = DType::from_safetensors(&header.dtype);
        let is_quantization_metadata = is_quantization_tensor(&name);
        let role = classify_tensor(&name, layer_id, expert_id, is_quantization_metadata, &facts);
        tensors.push(TensorInfo {
            name,
            file: file_name,
            dtype,
            shape: header.shape.clone(),
            byte_offset: header.data_offsets[0],
            byte_length: header.data_offsets[1] - header.data_offsets[0],
            role,
            layer_id,
            expert_id,
            is_quantization_metadata,
        });
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let catalog = TensorCatalog {
        model_id: model_id.to_owned(),
        snapshot_path: snapshot_path.display().to_string(),
        facts,
        tensors,
    };
    validate_native_deepseek_v4_attention_catalog(&catalog)
        .context("validating checkpoint-native DeepSeek V4 attention tensors")?;
    validate_native_deepseek_v4_dspark_catalog(&catalog)
        .context("validating checkpoint-native DeepSeek V4 dSpark tensors")?;
    if catalog.facts.quantization_recipe == "deepseek_v4_native_fp4_fp8_mixed_v1" {
        validate_native_fp4_expert_catalog(&catalog)
            .context("validating checkpoint-native DeepSeek V4 expert tensors")?;
    } else if is_deepseek_v4_exl3_recipe(&catalog.facts.quantization_recipe) {
        validate_exl3_expert_catalog(&catalog)
            .context("validating calibrated DeepSeek V4 EXL3 routed experts")?;
    }
    Ok(catalog)
}

type OptionSlot<T> = Mutex<Option<Result<T>>>;

pub fn read_safetensors_metadata(path: &Path) -> Result<Vec<SafetensorsTensorMetadata>> {
    let entries = parse_safetensors_header(path)?;
    Ok(entries
        .into_iter()
        .map(|(name, header)| SafetensorsTensorMetadata {
            name,
            dtype: DType::from_safetensors(&header.dtype),
            shape: header.shape,
            byte_offset: header.data_offsets[0],
            byte_length: header.data_offsets[1] - header.data_offsets[0],
        })
        .collect())
}

pub fn read_model_facts(model_id: &str, snapshot_path: &Path) -> Result<ModelFacts> {
    let config_path = snapshot_path.join("config.json");
    let config: ConfigFile = serde_json::from_reader(
        File::open(&config_path).with_context(|| format!("opening {}", config_path.display()))?,
    )
    .with_context(|| format!("parsing {}", config_path.display()))?;
    anyhow::ensure!(
        config.model_type == "deepseek_v4",
        "unsupported model_type {:?} for {model_id}; DS41RT requires deepseek_v4",
        config.model_type
    );
    anyhow::ensure!(
        config
            .architectures
            .iter()
            .any(|architecture| architecture == "DeepseekV4ForCausalLM"),
        "unsupported architectures {:?} for {model_id}",
        config.architectures
    );
    anyhow::ensure!(
        config.compress_ratios.len() >= config.num_hidden_layers,
        "compress_ratios has {} entries but {model_id} has {} hidden layers",
        config.compress_ratios.len(),
        config.num_hidden_layers
    );
    anyhow::ensure!(
        config
            .compress_ratios
            .iter()
            .all(|ratio| matches!(ratio, 0 | 4 | 128)),
        "unsupported attention compression schedule {:?} for {model_id}",
        config.compress_ratios
    );
    anyhow::ensure!(
        config.num_experts_per_tok > 0 && config.num_experts_per_tok <= config.n_routed_experts,
        "invalid routed expert top-k {} for {} experts",
        config.num_experts_per_tok,
        config.n_routed_experts
    );
    anyhow::ensure!(
        config
            .dspark_target_layer_ids
            .iter()
            .all(|layer_id| *layer_id < config.num_hidden_layers),
        "dSpark target layers {:?} exceed the {}-layer target",
        config.dspark_target_layer_ids,
        config.num_hidden_layers
    );
    anyhow::ensure!(
        config.rope_scaling.kind == "yarn",
        "unsupported RoPE scaling {:?} for {model_id}; DeepSeek V4 requires yarn",
        config.rope_scaling.kind
    );
    anyhow::ensure!(
        config.rope_theta.is_finite()
            && config.rope_theta > 0.0
            && config.compress_rope_theta.is_finite()
            && config.compress_rope_theta > 0.0
            && config.rope_scaling.factor.is_finite()
            && config.rope_scaling.factor > 0.0
            && config.rope_scaling.beta_fast.is_finite()
            && config.rope_scaling.beta_slow.is_finite()
            && config.rope_scaling.original_max_position_embeddings > 0,
        "invalid DeepSeek V4 RoPE/YaRN geometry for {model_id}"
    );
    anyhow::ensure!(
        config.hc_eps.is_finite() && config.hc_eps > 0.0,
        "invalid mHC epsilon {} for {model_id}",
        config.hc_eps
    );
    anyhow::ensure!(
        config.dspark_noise_token_id < config.vocab_size,
        "dSpark noise token {} exceeds vocabulary size {} for {model_id}",
        config.dspark_noise_token_id,
        config.vocab_size
    );
    let quantization_config =
        resolve_model_quantization_config(snapshot_path, config.quantization_config.as_ref())?;
    let quant_method = quantization_config
        .as_deref()
        .and_then(|value| {
            value
                .get("quant_method")
                .or_else(|| value.get("quant_algo"))
        })
        .and_then(Value::as_str)
        .unwrap_or("unquantized");
    let recipe = if let Some(recipe) =
        exl3_recipe_from_quantization_config(quantization_config.as_deref())?
    {
        recipe.to_owned()
    } else if quant_method.eq_ignore_ascii_case("fp8") && config.expert_dtype == "fp4" {
        "deepseek_v4_native_fp4_fp8_mixed_v1".to_owned()
    } else {
        format!("deepseek_v4_{}_{}_v1", config.expert_dtype, quant_method)
    };
    let variant = match (
        config.hidden_size,
        config.num_hidden_layers,
        config.n_routed_experts,
    ) {
        (DS4_FLASH_HIDDEN_SIZE, DS4_FLASH_NUM_HIDDEN_LAYERS, DS4_FLASH_ROUTED_EXPERTS) => {
            ModelVariant::Flash
        }
        (DS4_PRO_HIDDEN_SIZE, DS4_PRO_NUM_HIDDEN_LAYERS, DS4_PRO_ROUTED_EXPERTS) => {
            ModelVariant::Pro
        }
        _ => ModelVariant::Custom,
    };
    Ok(ModelFacts {
        model_id: model_id.to_owned(),
        model_type: config.model_type,
        variant,
        hidden_size: config.hidden_size,
        rms_norm_eps: config.rms_norm_eps,
        num_hidden_layers: config.num_hidden_layers,
        num_hash_layers: config.num_hash_layers,
        first_k_dense_replace: 0,
        routed_experts: config.n_routed_experts,
        top_k: config.num_experts_per_tok,
        moe_intermediate_size: config.moe_intermediate_size,
        shared_experts: config.n_shared_experts,
        vocab_size: config.vocab_size,
        attention_heads: config.num_attention_heads,
        kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        q_lora_rank: config.q_lora_rank,
        o_lora_rank: config.o_lora_rank,
        o_groups: config.o_groups,
        qk_rope_head_dim: config.qk_rope_head_dim,
        index_head_dim: config.index_head_dim,
        index_heads: config.index_n_heads,
        index_top_k: config.index_topk,
        sliding_window: config.sliding_window,
        max_position_embeddings: config.max_position_embeddings,
        rope_theta: config.rope_theta,
        compress_rope_theta: config.compress_rope_theta,
        rope_scaling_factor: config.rope_scaling.factor,
        original_max_position_embeddings: config.rope_scaling.original_max_position_embeddings,
        rope_beta_fast: config.rope_scaling.beta_fast,
        rope_beta_slow: config.rope_scaling.beta_slow,
        hyper_connection_multiplier: config.hc_mult,
        hyper_connection_sinkhorn_iters: config.hc_sinkhorn_iters,
        hyper_connection_eps: config.hc_eps,
        compress_ratios: config.compress_ratios,
        routed_scaling_factor: config.routed_scaling_factor,
        scoring_function: config.scoring_func,
        topk_method: config.topk_method,
        swiglu_limit: config.swiglu_limit,
        expert_dtype: config.expert_dtype,
        dspark_block_size: config.dspark_block_size,
        dspark_markov_rank: config.dspark_markov_rank,
        dspark_noise_token_id: config.dspark_noise_token_id,
        dspark_target_layer_ids: config.dspark_target_layer_ids,
        quantization_recipe: recipe,
    })
}

pub(crate) fn resolve_model_quantization_config<'a>(
    snapshot_path: &Path,
    embedded: Option<&'a Value>,
) -> Result<Option<Cow<'a, Value>>> {
    let Some(embedded) = embedded else {
        return Ok(None);
    };
    let quant_method = embedded
        .get("quant_method")
        .or_else(|| embedded.get("quant_algo"))
        .and_then(Value::as_str)
        .unwrap_or("unquantized");
    let is_compact_gptqmodel_exl3 = quant_method.eq_ignore_ascii_case("exl3")
        && embedded.get("tensor_storage").is_none()
        && embedded.get("ds41rt").is_none();
    if !is_compact_gptqmodel_exl3 {
        return Ok(Some(Cow::Borrowed(embedded)));
    }
    let embedded_object = embedded
        .as_object()
        .context("config.json quantization_config must be a JSON object")?;

    let mut resolved: Option<(String, Value)> = None;
    for filename in EXTERNAL_QUANTIZATION_CONFIG_FILES {
        let path = snapshot_path.join(filename);
        if !path.is_file() {
            continue;
        }
        let full: Value = serde_json::from_reader(
            File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))?;
        anyhow::ensure!(
            full.is_object(),
            "external EXL3 quantization config {} must be a JSON object",
            path.display()
        );

        let full_object = full.as_object().expect("object checked above");
        anyhow::ensure!(
            full_object.get("tensor_storage").is_some()
                && full_object.len() == embedded_object.len() + 1
                && embedded_object
                    .iter()
                    .all(|(key, value)| full_object.get(key) == Some(value)),
            "config.json quantization_config differs from compact {}",
            path.display()
        );
        if let Some((prior_filename, prior)) = resolved.as_ref() {
            anyhow::ensure!(
                prior == &full,
                "external EXL3 quantization configs {prior_filename} and {filename} differ"
            );
        } else {
            resolved = Some((filename.to_owned(), full));
        }
    }

    let (_, full) = resolved.with_context(|| {
        format!(
            "compact EXL3 quantization_config requires {} or {} in {}",
            EXTERNAL_QUANTIZATION_CONFIG_FILES[0],
            EXTERNAL_QUANTIZATION_CONFIG_FILES[1],
            snapshot_path.display()
        )
    })?;
    Ok(Some(Cow::Owned(full)))
}

fn parse_safetensors_header(path: &Path) -> Result<BTreeMap<String, SafetensorsTensorHeader>> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("reading metadata for {}", path.display()))?
        .len();
    let mut len_bytes = [0_u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    anyhow::ensure!(header_len <= 64 * 1024 * 1024, "safetensors header exceeds 64 MiB");
    let header_len_usize: usize = header_len
        .try_into()
        .context("safetensors header length does not fit in memory")?;
    let data_start = 8_u64
        .checked_add(header_len)
        .context("safetensors data offset overflow")?;
    anyhow::ensure!(
        data_start <= file_len,
        "safetensors header extends beyond {}: data starts at {data_start}, file length is {file_len}",
        path.display()
    );
    let mut header_bytes = vec![0_u8; header_len_usize];
    file.read_exact(&mut header_bytes)?;
    let raw: BTreeMap<String, Value> = serde_json::from_slice(&header_bytes)?;
    let mut entries = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let mut header: SafetensorsTensorHeader = serde_json::from_value(value)
            .with_context(|| format!("parsing header entry {name} in {}", path.display()))?;
        // Hardening bound ported from upstream llama.cpp gguf-py
        // test_gguf_reader_validation.py: reject absurd tensor rank before any
        // downstream allocation/arithmetic on the dims vector.
        anyhow::ensure!(
            header.shape.len() <= 32,
            "safetensors tensor {name} in {} has absurd rank {} (max 32)",
            path.display(),
            header.shape.len()
        );
        anyhow::ensure!(
            header.data_offsets[0] <= header.data_offsets[1],
            "invalid safetensors offsets for {name} in {}: {:?}",
            path.display(),
            header.data_offsets
        );
        header.data_offsets[0] = header.data_offsets[0]
            .checked_add(data_start)
            .context("safetensors tensor start offset overflow")?;
        header.data_offsets[1] = header.data_offsets[1]
            .checked_add(data_start)
            .context("safetensors tensor end offset overflow")?;
        anyhow::ensure!(
            header.data_offsets[1] <= file_len,
            "safetensors tensor {name} extends beyond {}: end {}, file length {file_len}",
            path.display(),
            header.data_offsets[1]
        );
        entries.insert(name, header);
    }
    Ok(entries)
}

fn extract_number_after(name: &str, marker: &str) -> Option<usize> {
    let start = name.find(marker)? + marker.len();
    let digits = name[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

pub(crate) fn tensor_layer_id(name: &str, facts: &ModelFacts) -> Option<u32> {
    extract_number_after(name, "layers.")
        .or_else(|| extract_number_after(name, "model.layers."))
        .or_else(|| {
            extract_number_after(name, "mtp.").map(|block_id| facts.num_hidden_layers + block_id)
        })
        .map(|layer_id| layer_id as u32)
}

pub(crate) fn is_quantization_tensor(name: &str) -> bool {
    name.ends_with(".input_scale")
        || name.ends_with(".weight_scale")
        || name.ends_with(".weight_scale_2")
        || name.contains(".input_scale.")
        || name.contains(".weight_scale.")
        || name.ends_with(".scale")
        || name.ends_with(".trellis_mcg")
        || name.ends_with(".suh")
        || name.ends_with(".svh")
        || name.ends_with(".su")
        || name.ends_with(".sv")
        || name.ends_with(".mcg")
}

pub(crate) fn classify_tensor(
    name: &str,
    layer_id: Option<u32>,
    expert_id: Option<u32>,
    _is_quantization_metadata: bool,
    facts: &ModelFacts,
) -> TensorRole {
    if expert_id.is_some() && (name.contains(".ffn.experts.") || name.contains(".mlp.experts.")) {
        return TensorRole::RoutedExpert;
    }
    if name.starts_with("mtp.") {
        return TensorRole::Dspark;
    }
    if name.contains(".ffn.shared_experts.") || name.contains(".mlp.shared_experts.") {
        return TensorRole::SharedExpert;
    }
    if name.contains(".ffn.gate.") || name.contains(".mlp.gate.") {
        return TensorRole::Router;
    }
    if name == "embed.weight" || name == "model.embed_tokens.weight" {
        return TensorRole::Embedding;
    }
    if name == "head.weight" || name == "lm_head.weight" {
        return TensorRole::LmHead;
    }
    if name.contains(".attn.indexer.") {
        return TensorRole::AttentionIndexer;
    }
    if name.contains(".attn.compressor.") {
        return TensorRole::AttentionCompressor;
    }
    if name.contains(".attn.") || name.contains(".self_attn.") {
        return TensorRole::Attention;
    }
    if name.starts_with("hc_head_") || name.contains(".hc_") {
        return TensorRole::HyperConnection;
    }
    if name.contains("layernorm")
        || name == "norm.weight"
        || name.ends_with("_norm.weight")
        || name.ends_with(".norm.weight")
        || name.contains(".norm.")
    {
        return TensorRole::Norm;
    }
    if name.contains(".ffn.") || name.contains(".mlp.") {
        return TensorRole::DenseMlp;
    }
    if layer_id
        .map(|layer| layer as usize >= facts.num_hidden_layers)
        .unwrap_or(false)
    {
        return if facts.model_type == "deepseek_v4" {
            TensorRole::Dspark
        } else {
            TensorRole::Mtp
        };
    }
    TensorRole::Other
}

pub fn classification_summary_markdown(catalog: &TensorCatalog) -> String {
    let mut by_role = BTreeMap::<String, usize>::new();
    let mut routed_weight = 0_usize;
    let mut routed_quant = 0_usize;
    let mut shared = 0_usize;
    let mut files = BTreeSet::new();
    for tensor in &catalog.tensors {
        *by_role.entry(format!("{:?}", tensor.role)).or_default() += 1;
        files.insert(tensor.file.clone());
        if tensor.role == TensorRole::RoutedExpert && tensor.is_quantization_metadata {
            routed_quant += 1;
        } else if tensor.role == TensorRole::RoutedExpert {
            routed_weight += 1;
        }
        if tensor.role == TensorRole::SharedExpert {
            shared += 1;
        }
    }
    let mut out = String::new();
    out.push_str("# Tensor Classification Summary\n\n");
    out.push_str(&format!("- Model: `{}`\n", catalog.model_id));
    out.push_str(&format!("- Snapshot: `{}`\n", catalog.snapshot_path));
    out.push_str(&format!("- Catalog hash: `{}`\n", catalog.content_hash()));
    out.push_str(&format!("- Tensor count: `{}`\n", catalog.tensors.len()));
    out.push_str(&format!("- Safetensors files: `{}`\n", files.len()));
    out.push_str(&format!("- Variant: `{:?}`\n", catalog.facts.variant));
    out.push_str(&format!("- Hidden size: `{}`\n", catalog.facts.hidden_size));
    out.push_str(&format!(
        "- Hidden layers: `{}`\n",
        catalog.facts.num_hidden_layers
    ));
    out.push_str(&format!(
        "- Hash-routed MoE layers: `{}`\n",
        catalog.facts.num_hash_layers
    ));
    out.push_str(&format!(
        "- Routed experts per MoE layer: `{}`\n",
        catalog.facts.routed_experts
    ));
    out.push_str(&format!(
        "- Top-k experts per token: `{}`\n",
        catalog.facts.top_k
    ));
    out.push_str(&format!(
        "- MoE intermediate size: `{}`\n",
        catalog.facts.moe_intermediate_size
    ));
    out.push_str(&format!(
        "- dSpark target layers: `{:?}`\n",
        catalog.facts.dspark_target_layer_ids
    ));
    out.push_str(&format!(
        "- Quantization recipe: `{}`\n\n",
        catalog.facts.quantization_recipe
    ));
    out.push_str("## Role Counts\n\n");
    out.push_str("| Role | Tensors |\n| --- | ---: |\n");
    for (role, count) in by_role {
        out.push_str(&format!("| {role} | {count} |\n"));
    }
    out.push_str("\n## Routed Expert Detail\n\n");
    out.push_str(&format!(
        "- Routed expert non-scale tensors: `{routed_weight}`\n"
    ));
    out.push_str(&format!(
        "- Routed expert quantization tensors: `{routed_quant}`\n"
    ));
    out.push_str(&format!("- Shared expert tensors: `{shared}`\n"));
    out
}

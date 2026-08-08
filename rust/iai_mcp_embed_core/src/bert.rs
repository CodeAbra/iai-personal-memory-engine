//! BertEmbedder — bge-small-en-v1.5 pure-Rust forward pass via candle-nn.
//!
//! Implements the full BERT inference pipeline:
//!   tokenize → BertEmbeddings → 12× BertLayer → CLS pool → L2 normalize → Vec<f32>
//!
//! Numeric-parity constraints honored (each one a silent-drift bug if broken):
//!   - gelu_erf() (exact erf formula, not tanh approximation)
//!   - LayerNorm eps = 1e-12 (BERT config, not candle default 1e-5)
//!   - Raw CLS pooling: last_hidden[:, 0, :] — NOT BertPooler dense+tanh
//!   - Apple Accelerate BLAS via extern crate in lib.rs root (not here)

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};
use hf_hub::{api::sync::ApiBuilder, Cache, Repo, RepoType};
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use crate::error::EmbedError;

// ---------------------------------------------------------------------------
// Model spec + geometry
// ---------------------------------------------------------------------------

const MODEL_ID: &str = "BAAI/bge-small-en-v1.5";
const REVISION: &str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a";

// The default model's exact architecture, asserted against its config.json
// at load: the P[384, 10000] projection is SHA-locked to 384-d input, so a
// silent geometry drift in the default snapshot must fail loud, never
// re-derive.
const DEFAULT_GEOMETRY: Geometry = Geometry {
    vocab_size: 30522,
    hidden_size: 384,
    num_hidden_layers: 12,
    num_attention_heads: 12,
    intermediate_size: 1536,
    max_position: 512,
    type_vocab_size: 2,
    layer_norm_eps: 1e-12,
};

/// Sentence pooling over the final hidden states. CLS is the historical
/// default (bge family); MEAN is required by mean-pooled retrieval models
/// (e5 family) — serving a mean-pooled model through CLS silently degrades
/// every vector it emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Cls,
    Mean,
}

impl Pooling {
    pub fn parse(raw: &str) -> Result<Self, EmbedError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "cls" => Ok(Pooling::Cls),
            "mean" => Ok(Pooling::Mean),
            other => Err(EmbedError::HfHub(format!(
                "invalid pooling mode {other:?}; valid: cls, mean"
            ))),
        }
    }
}

/// Which model snapshot to load. The default is the pinned
/// bge-small-en-v1.5 revision; `IAI_MCP_EMBED_MODEL_ID` selects another
/// BERT-architecture model (with `IAI_MCP_EMBED_REVISION`, default `main`).
fn model_spec() -> (String, String) {
    match std::env::var("IAI_MCP_EMBED_MODEL_ID") {
        Ok(id) if !id.trim().is_empty() => {
            let rev = std::env::var("IAI_MCP_EMBED_REVISION")
                .ok()
                .filter(|r| !r.trim().is_empty())
                .unwrap_or_else(|| "main".to_string());
            (id, rev)
        }
        _ => (MODEL_ID.to_string(), REVISION.to_string()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geometry {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
}

impl Geometry {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Parse the model's own config.json. The forward pass implemented here
    /// is plain BERT, so any other `model_type` fails loud; hidden size must
    /// stay 384 — the SHA-locked SimHash projection maps exactly 384-d input.
    fn from_config_file(path: &std::path::Path, model_id: &str) -> Result<Self, EmbedError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| EmbedError::HfHub(format!("config.json read failed: {e}")))?;
        let cfg: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| EmbedError::HfHub(format!("config.json parse failed: {e}")))?;
        let model_type = cfg.get("model_type").and_then(|v| v.as_str()).unwrap_or("");
        if model_type != "bert" {
            return Err(EmbedError::HfHub(format!(
                "{model_id}: model_type {model_type:?} is not \"bert\" — this \
                 embedder implements the plain BERT forward pass only"
            )));
        }
        let field = |name: &str| -> Result<usize, EmbedError> {
            cfg.get(name)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| EmbedError::HfHub(format!("{model_id}: config.json missing {name}")))
        };
        let geometry = Geometry {
            vocab_size: field("vocab_size")?,
            hidden_size: field("hidden_size")?,
            num_hidden_layers: field("num_hidden_layers")?,
            num_attention_heads: field("num_attention_heads")?,
            intermediate_size: field("intermediate_size")?,
            max_position: field("max_position_embeddings")?,
            type_vocab_size: field("type_vocab_size")?,
            layer_norm_eps: cfg
                .get("layer_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-12),
        };
        if geometry.hidden_size != DEFAULT_GEOMETRY.hidden_size {
            return Err(EmbedError::HfHub(format!(
                "{model_id}: hidden_size {} != 384 — the locked SimHash \
                 projection maps exactly 384-d embeddings",
                geometry.hidden_size
            )));
        }
        if geometry.hidden_size % geometry.num_attention_heads != 0 {
            return Err(EmbedError::HfHub(format!(
                "{model_id}: hidden_size {} not divisible by attention heads {}",
                geometry.hidden_size, geometry.num_attention_heads
            )));
        }
        Ok(geometry)
    }
}

// ---------------------------------------------------------------------------
// HF cache resolver
// ---------------------------------------------------------------------------

/// Resolve model files from HF cache (or trigger lazy download on cache miss).
///
/// Cache location honors the `HF_HOME` environment variable (via the hf-hub
/// `from_env` resolvers), falling back to the default `~/.cache/huggingface`
/// when it is unset. This keeps model resolution independent of `$HOME`, so a
/// relocated or redirected home directory does not break model loading. When no
/// HF environment variable is set, behavior is identical to the prior default.
///
/// If `IAI_MCP_EMBED_OFFLINE=1` is set, fails loudly when any file is missing
/// rather than attempting a network download.
///
/// Stats `<hub>/models--<org>--<name>/snapshots/<revision>/<filename>`
/// directly. The hf-hub ref-based lookup needs `refs/<revision>`, which only
/// an online fetch writes — a sha-pinned snapshot already on disk must still
/// resolve offline.
// "0", "false", and empty must read as off — presence alone is not consent.
fn offline_mode_enabled() -> bool {
    matches!(
        std::env::var("IAI_MCP_EMBED_OFFLINE").ok().as_deref(),
        Some(v) if !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
    )
}

fn cached_file_direct(
    model_id: &str,
    revision: &str,
    filename: &str,
) -> Option<std::path::PathBuf> {
    // The revision joins a filesystem path below — separators or parent
    // traversal must never escape the snapshots directory.
    if revision.is_empty() || revision.contains(['/', '\\']) || revision.contains("..") {
        return None;
    }
    let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
    let path = Cache::from_env()
        .path()
        .join(repo.folder_name())
        .join("snapshots")
        .join(revision)
        .join(filename);
    path.exists().then_some(path)
}

fn resolve_model_files_for(
    model_id_arg: Option<&str>,
    revision_arg: Option<&str>,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf), EmbedError> {
    let (model_id, revision) = match model_id_arg {
        Some(id) => (
            id.to_string(),
            revision_arg.unwrap_or("main").to_string(),
        ),
        None => model_spec(),
    };
    let repo = Repo::with_revision(model_id.clone(), RepoType::Model, revision.clone());

    // Complete sha-pinned snapshots on disk serve without touching the
    // network OR the hf-hub ref indirection, in every mode.
    if let (Some(weights), Some(tokenizer), Some(config)) = (
        cached_file_direct(&model_id, &revision, "model.safetensors"),
        cached_file_direct(&model_id, &revision, "tokenizer.json"),
        cached_file_direct(&model_id, &revision, "config.json"),
    ) {
        return Ok((weights, tokenizer, config));
    }

    let offline = offline_mode_enabled();

    if offline {
        let cache = Cache::from_env().repo(repo);
        let get = |filename: &str| -> Result<std::path::PathBuf, EmbedError> {
            cached_file_direct(&model_id, &revision, filename)
                .or_else(|| cache.get(filename))
                .ok_or_else(|| {
                    EmbedError::HfHub(format!("{filename} not in HF cache (offline mode)"))
                })
        };
        Ok((
            get("model.safetensors")?,
            get("tokenizer.json")?,
            get("config.json")?,
        ))
    } else {
        let api = ApiBuilder::from_env()
            .build()
            .map_err(|e| EmbedError::HfHub(e.to_string()))?
            .repo(repo);
        let weights = api
            .get("model.safetensors")
            .map_err(|e| EmbedError::HfHub(e.to_string()))?;
        let tokenizer = api
            .get("tokenizer.json")
            .map_err(|e| EmbedError::HfHub(e.to_string()))?;
        let config = api
            .get("config.json")
            .map_err(|e| EmbedError::HfHub(e.to_string()))?;
        Ok((weights, tokenizer, config))
    }
}

// ---------------------------------------------------------------------------
// BertEmbeddings
// ---------------------------------------------------------------------------

struct BertEmbeddings {
    word_emb: Embedding,
    pos_emb: Embedding,
    type_emb: Embedding,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, g: &Geometry) -> Result<Self, EmbedError> {
        let vb_emb = vb.pp("embeddings");
        let word_emb = embedding(g.vocab_size, g.hidden_size, vb_emb.pp("word_embeddings"))?;
        let pos_emb = embedding(
            g.max_position,
            g.hidden_size,
            vb_emb.pp("position_embeddings"),
        )?;
        let type_emb = embedding(
            g.type_vocab_size,
            g.hidden_size,
            vb_emb.pp("token_type_embeddings"),
        )?;
        // Pitfall 3: must pass the config eps explicitly — candle default is 1e-5
        let layer_norm = layer_norm(g.hidden_size, g.layer_norm_eps, vb_emb.pp("LayerNorm"))?;
        Ok(Self {
            word_emb,
            pos_emb,
            type_emb,
            layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor, EmbedError> {
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device();

        // Generate position ids at runtime (buffer in safetensors is ignored per Pattern 4)
        let position_ids = Tensor::arange(0u32, seq_len as u32, device)?.unsqueeze(0)?;

        let word_out = self.word_emb.forward(input_ids)?;
        let pos_out = self.pos_emb.forward(&position_ids)?;
        let type_out = self.type_emb.forward(token_type_ids)?;

        let combined = (word_out + pos_out)?.add(&type_out)?;
        Ok(self.layer_norm.forward(&combined)?)
    }
}

// ---------------------------------------------------------------------------
// BertSelfAttention
// ---------------------------------------------------------------------------

struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, g: &Geometry) -> Result<Self, EmbedError> {
        let vb_self = vb.pp("attention").pp("self");
        let query = linear(g.hidden_size, g.hidden_size, vb_self.pp("query"))?;
        let key = linear(g.hidden_size, g.hidden_size, vb_self.pp("key"))?;
        let value = linear(g.hidden_size, g.hidden_size, vb_self.pp("value"))?;
        Ok(Self {
            query,
            key,
            value,
            num_heads: g.num_attention_heads,
            head_dim: g.head_dim(),
            hidden_size: g.hidden_size,
        })
    }

    fn forward(&self, hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EmbedError> {
        let (batch, seq_len, _hidden) = hidden.dims3()?;

        // Project Q, K, V
        let q = self.query.forward(hidden)?;
        let k = self.key.forward(hidden)?;
        let v = self.value.forward(hidden)?;

        // Reshape to (batch, seq_len, num_heads, head_dim) then transpose
        // → (batch, num_heads, seq_len, head_dim)
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product: Q @ K^T / sqrt(head_dim)
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)?.affine(1.0 / scale, 0.0)?;

        // Additive mask: attention_mask is (1,1,1,seq_len), broadcast over (batch,heads,seq,seq)
        let scores = scores.broadcast_add(attention_mask)?;

        // Softmax over last dim (seq dimension)
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;

        // Weighted sum of values
        let context = probs.matmul(&v)?;

        // Transpose back: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, hidden_size)
        Ok(context
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.hidden_size))?)
    }
}

// ---------------------------------------------------------------------------
// BertLayer
// ---------------------------------------------------------------------------

struct BertLayer {
    attention: BertSelfAttention,
    attn_output_dense: Linear,
    attn_output_ln: LayerNorm,
    ffn_intermediate: Linear,
    ffn_output: Linear,
    output_ln: LayerNorm,
}

impl BertLayer {
    fn load(vb: VarBuilder, layer_idx: usize, g: &Geometry) -> Result<Self, EmbedError> {
        let vb_layer = vb.pp("encoder").pp("layer").pp(layer_idx.to_string());

        let attention = BertSelfAttention::load(vb_layer.clone(), g)?;

        let attn_output_dense = linear(
            g.hidden_size,
            g.hidden_size,
            vb_layer.pp("attention").pp("output").pp("dense"),
        )?;
        let attn_output_ln = layer_norm(
            g.hidden_size,
            g.layer_norm_eps,
            vb_layer.pp("attention").pp("output").pp("LayerNorm"),
        )?;

        // FFN: hidden → intermediate then intermediate → hidden
        let ffn_intermediate = linear(
            g.hidden_size,
            g.intermediate_size,
            vb_layer.pp("intermediate").pp("dense"),
        )?;
        let ffn_output = linear(
            g.intermediate_size,
            g.hidden_size,
            vb_layer.pp("output").pp("dense"),
        )?;
        let output_ln = layer_norm(
            g.hidden_size,
            g.layer_norm_eps,
            vb_layer.pp("output").pp("LayerNorm"),
        )?;

        Ok(Self {
            attention,
            attn_output_dense,
            attn_output_ln,
            ffn_intermediate,
            ffn_output,
            output_ln,
        })
    }

    fn forward(&self, hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EmbedError> {
        // --- Attention sub-layer ---
        let attn_out = self.attention.forward(hidden, attention_mask)?;
        let attn_out = self.attn_output_dense.forward(&attn_out)?;
        // Residual + LayerNorm
        let attn_out = self.attn_output_ln.forward(&attn_out.add(hidden)?)?;

        // --- FFN sub-layer ---
        // Pitfall 2: gelu_erf (exact erf), NOT gelu (tanh approximation)
        let ffn_inter = self.ffn_intermediate.forward(&attn_out)?.gelu_erf()?;
        let ffn_out = self.ffn_output.forward(&ffn_inter)?;
        // Residual + LayerNorm
        Ok(self.output_ln.forward(&ffn_out.add(&attn_out)?)?)
    }
}

// ---------------------------------------------------------------------------
// BertEncoder (12-layer stack)
// ---------------------------------------------------------------------------

struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn load(vb: VarBuilder, g: &Geometry) -> Result<Self, EmbedError> {
        let layers = (0..g.num_hidden_layers)
            .map(|i| BertLayer::load(vb.clone(), i, g))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { layers })
    }

    fn forward(&self, hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor, EmbedError> {
        let mut h = hidden.clone();
        for layer in &self.layers {
            h = layer.forward(&h, attention_mask)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// BertEmbedder — public facade
// ---------------------------------------------------------------------------

/// Pure-Rust BERT embedder for bge-small-en-v1.5.
///
/// Loads weights from the HF cache at construction time.
/// `encode()` returns a 384-dim L2-normalized `Vec<f32>`.
pub struct BertEmbedder {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    tokenizer: Tokenizer,
    device: Device,
    // Dedicated rayon pool: candle's CPU kernels par_iter on whatever pool is
    // current, and the process-global registry is shared with every other
    // native component — a wedged/poisoned global pool would freeze encode()
    // forever (callers park on a LockLatch while global workers idle).
    // install() scopes all kernel parallelism to this private pool.
    pool: rayon::ThreadPool,
    text_prefix: String,
    pooling: Pooling,
}

impl BertEmbedder {
    /// Load bge-small-en-v1.5 from the HF cache. The cache location honors
    /// `HF_HOME` (falling back to `~/.cache/huggingface/hub/` when unset).
    /// Triggers a lazy download from HF Hub if the snapshot is absent
    /// (unless `IAI_MCP_EMBED_OFFLINE=1` is set, in which case this fails loudly).
    pub fn load() -> Result<Self, EmbedError> {
        Self::load_with(None, None, None)
    }

    /// Load an explicit model spec. `None` fields fall back to the env
    /// override / pinned default resolution of `load()`. An explicit spec
    /// wins over the environment — the caller (the Python registry) is the
    /// single source of truth for which model serves, so a stray env var
    /// cannot silently redirect a spec-driven construction.
    pub fn load_with(
        model_id_arg: Option<&str>,
        revision_arg: Option<&str>,
        pooling_arg: Option<Pooling>,
    ) -> Result<Self, EmbedError> {
        let (weights_path, tokenizer_path, config_path) =
            resolve_model_files_for(model_id_arg, revision_arg)?;
        let model_id = match model_id_arg {
            Some(id) => id.to_string(),
            None => model_spec().0,
        };

        // Geometry comes from the model's own config.json. For the pinned
        // default model any drift from the historical architecture fails
        // loud — the SHA-locked SimHash projection depends on it.
        let geometry = Geometry::from_config_file(&config_path, &model_id)?;
        if model_id == MODEL_ID && geometry != DEFAULT_GEOMETRY {
            return Err(EmbedError::HfHub(format!(
                "{MODEL_ID}: config.json geometry drifted from the pinned                  architecture: {geometry:?}"
            )));
        }

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        // Pitfall 7: must configure truncation; encode() alone does NOT truncate
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: geometry.max_position,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        #[cfg(feature = "metal")]
        let device = Device::new_metal(0)?;
        #[cfg(not(feature = "metal"))]
        let device = Device::Cpu;

        // Pattern 4: unsafe scoped to mmap call; file integrity assumed via HF CDN + revision pin
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device) }?;

        let embeddings = BertEmbeddings::load(vb.clone(), &geometry)?;
        let encoder = BertEncoder::load(vb, &geometry)?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("embed-rayon-{i}"))
            .build()
            .map_err(|e| EmbedError::Pool(e.to_string()))?;

        // Some retrieval models expect an instruction prefix on every input
        // (e.g. the E5 family). Empty default = byte-identical behavior.
        let text_prefix = std::env::var("IAI_MCP_EMBED_TEXT_PREFIX").unwrap_or_default();

        // Pooling comes ONLY from the explicit spec — an env knob here would
        // let the service environment flip pooling behind the identity stamp.
        let pooling = pooling_arg.unwrap_or(Pooling::Cls);

        Ok(Self {
            embeddings,
            encoder,
            tokenizer,
            device,
            pool,
            text_prefix,
            pooling,
        })
    }

    /// Encode a single text string to a 384-dim L2-normalized embedding.
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        // The whole pass runs inside the private pool so no tokenizer or
        // candle kernel can ever park on the shared global registry.
        self.pool.install(|| self.encode_inner(text))
    }

    fn encode_inner(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let prefixed;
        let text = if self.text_prefix.is_empty() {
            text
        } else {
            prefixed = format!("{}{}", self.text_prefix, text);
            prefixed.as_str()
        };
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let seq_len = ids.len();

        // Pattern 9: additive attention mask — 0.0 for real tokens, f32::MIN for padding
        let mask: Vec<f32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| if m == 1 { 0.0_f32 } else { f32::MIN })
            .collect();

        let input_ids = Tensor::from_vec(ids, (1, seq_len), &self.device)?;
        let token_type_ids = Tensor::zeros((1, seq_len), DType::I64, &self.device)?;
        // Shape (1, 1, 1, seq_len) broadcasts over (batch, heads, seq_q, seq_k)
        let attention_mask = Tensor::from_vec(mask, (1, 1, 1, seq_len), &self.device)?;

        let embedded = self.embeddings.forward(&input_ids, &token_type_ids)?;
        let encoded = self.encoder.forward(&embedded, &attention_mask)?;

        let pooled = match self.pooling {
            // Pattern 8 / Pitfall 1: raw CLS token — NOT BertPooler dense+tanh
            Pooling::Cls => encoded.i((0, 0))?,
            Pooling::Mean => {
                // Attention-weighted mean over real tokens only; the additive
                // mask above is for attention scores, so a fresh binary mask
                // weights the hidden states here.
                let bin: Vec<f32> = encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&m| if m == 1 { 1.0_f32 } else { 0.0_f32 })
                    .collect();
                let count: f32 = bin.iter().sum::<f32>().max(1.0);
                let bin_t = Tensor::from_vec(bin, (seq_len, 1), &self.device)?;
                let hidden = encoded.i((0,))?;
                let summed = hidden.broadcast_mul(&bin_t)?.sum(0)?;
                (summed / count as f64)?
            }
        };

        // L2 normalize
        let norm = pooled.sqr()?.sum_keepdim(0)?.sqrt()?;
        let normalized = pooled.broadcast_div(&norm)?;

        Ok(normalized.to_vec1::<f32>()?)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_geometry_unchanged() {
        assert_eq!(DEFAULT_GEOMETRY.hidden_size, 384);
        assert_eq!(DEFAULT_GEOMETRY.num_hidden_layers, 12);
        assert_eq!(
            DEFAULT_GEOMETRY.head_dim(),
            DEFAULT_GEOMETRY.hidden_size / DEFAULT_GEOMETRY.num_attention_heads
        );
        assert_eq!(DEFAULT_GEOMETRY.layer_norm_eps, 1e-12_f64);
    }

    #[test]
    fn geometry_parses_and_gates_config() {
        let dir = std::env::temp_dir().join("iai-embed-geom-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            r#"{"model_type":"bert","vocab_size":30522,"hidden_size":384,
                "num_hidden_layers":12,"num_attention_heads":12,
                "intermediate_size":1536,"max_position_embeddings":512,
                "type_vocab_size":2,"layer_norm_eps":1e-12}"#,
        )
        .unwrap();
        let g = Geometry::from_config_file(&path, "test/model").unwrap();
        assert_eq!(g, DEFAULT_GEOMETRY);

        std::fs::write(
            &path,
            r#"{"model_type":"xlm-roberta","vocab_size":250002,"hidden_size":384,
                "num_hidden_layers":12,"num_attention_heads":12,
                "intermediate_size":1536,"max_position_embeddings":512,
                "type_vocab_size":1}"#,
        )
        .unwrap();
        assert!(Geometry::from_config_file(&path, "test/model").is_err());

        std::fs::write(
            &path,
            r#"{"model_type":"bert","vocab_size":30522,"hidden_size":768,
                "num_hidden_layers":12,"num_attention_heads":12,
                "intermediate_size":3072,"max_position_embeddings":512,
                "type_vocab_size":2}"#,
        )
        .unwrap();
        assert!(Geometry::from_config_file(&path, "test/model").is_err());
    }

    fn cache_present() -> bool {
        dirs::home_dir()
            .map(|h| {
                h.join(".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5")
                    .join(format!("snapshots/{REVISION}/model.safetensors"))
                    .exists()
            })
            .unwrap_or(false)
    }

    #[test]
    fn load_succeeds_when_cache_present() {
        if !cache_present() {
            eprintln!("HF cache absent — skipping");
            return;
        }
        let _ = BertEmbedder::load().expect("load");
    }

    #[test]
    fn encode_returns_384_dim_vector() {
        if !cache_present() {
            eprintln!("HF cache absent — skipping");
            return;
        }
        let e = BertEmbedder::load().unwrap();
        let v = e.encode("hello").unwrap();
        assert_eq!(v.len(), 384);
    }

    #[test]
    fn output_is_l2_normalized() {
        if !cache_present() {
            eprintln!("HF cache absent — skipping");
            return;
        }
        let e = BertEmbedder::load().unwrap();
        let v = e.encode("hello").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "L2 norm = {norm}");
    }

    #[test]
    fn encode_is_deterministic() {
        if !cache_present() {
            eprintln!("HF cache absent — skipping");
            return;
        }
        let e = BertEmbedder::load().unwrap();
        let a = e.encode("the quick brown fox").unwrap();
        let b = e.encode("the quick brown fox").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn truncation_handles_oversized_input() {
        if !cache_present() {
            eprintln!("HF cache absent — skipping");
            return;
        }
        let e = BertEmbedder::load().unwrap();
        let long = "word ".repeat(1000);
        let v = e.encode(&long).unwrap();
        assert_eq!(v.len(), 384);
    }

    const E5_ID: &str = "intfloat/multilingual-e5-small";
    const E5_REV: &str = "614241f622f53c4eeff9890bdc4f31cfecc418b3";

    // Stat the snapshot path directly: the hf-hub ref-based lookup needs
    // refs/<sha>, which only an online fetch writes, and a silent skip here
    // would let the golden read as verified without ever running.
    fn require_e5() {
        assert!(
            cached_file_direct(E5_ID, E5_REV, "model.safetensors").is_some(),
            "e5 snapshot absent — golden cannot run; fetch {E5_ID}@{E5_REV} into the HF cache first"
        );
    }

    /// Golden: the mean arm against the reference implementation (masked
    /// mean of last hidden states, then L2). Reference head computed with
    /// the transformers library on the same pinned snapshot.
    #[test]
    fn mean_pooling_matches_reference_on_e5() {
        require_e5();
        let e = BertEmbedder::load_with(Some(E5_ID), Some(E5_REV), Some(Pooling::Mean)).unwrap();
        let v = e.encode("query: hello world").unwrap();
        assert_eq!(v.len(), 384);
        let expected = [
            0.046284_f32, 0.031391, -0.038663, -0.055825, 0.066460, -0.043633, 0.011182, 0.087065,
        ];
        for (i, exp) in expected.iter().enumerate() {
            assert!(
                (v[i] - exp).abs() < 1e-3,
                "dim {i}: got {} expected {exp}",
                v[i]
            );
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn mean_and_cls_pooling_disagree_on_multi_token_input() {
        require_e5();
        let mean = BertEmbedder::load_with(Some(E5_ID), Some(E5_REV), Some(Pooling::Mean))
            .unwrap()
            .encode("query: hello world")
            .unwrap();
        let cls = BertEmbedder::load_with(Some(E5_ID), Some(E5_REV), Some(Pooling::Cls))
            .unwrap()
            .encode("query: hello world")
            .unwrap();
        let dot: f32 = mean.iter().zip(&cls).map(|(a, b)| a * b).sum();
        assert!(dot < 0.999, "pooling arms must be distinguishable, cos={dot}");
    }

    /// Restores the prior value on drop, so a panicking assertion cannot
    /// leak a mutated env var into the rest of the test run.
    struct EnvVarGuard {
        key: &'static str,
        prior: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prior = std::env::var(key).ok();
            std::env::set_var(key, value);
            EnvVarGuard { key, prior }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prior {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    // Sibling tests only READ these vars, through loads that resolve from
    // the on-disk snapshot in every mode — a concurrently observed mutation
    // changes which branch resolves them, never whether they resolve.
    #[test]
    fn env_knobs_stay_inert_and_offline_serves_from_cache() {
        require_e5();
        {
            let _off = EnvVarGuard::set("IAI_MCP_EMBED_OFFLINE", "0");
            assert!(!offline_mode_enabled());
        }
        {
            let _off = EnvVarGuard::set("IAI_MCP_EMBED_OFFLINE", "false");
            assert!(!offline_mode_enabled());
        }
        let _off = EnvVarGuard::set("IAI_MCP_EMBED_OFFLINE", "1");
        assert!(offline_mode_enabled());

        // Pooling comes only from the spec — a revived env knob must fail here.
        let _pool = EnvVarGuard::set("IAI_MCP_EMBED_POOL", "mean");
        let default_pool = BertEmbedder::load_with(Some(E5_ID), Some(E5_REV), None)
            .unwrap()
            .pooling;
        assert!(matches!(default_pool, Pooling::Cls));

        // Offline load of a sha-pinned snapshot must resolve from disk even
        // when the cache carries no refs entry for that sha.
        let mean_pool = BertEmbedder::load_with(Some(E5_ID), Some(E5_REV), Some(Pooling::Mean))
            .unwrap()
            .pooling;
        assert!(matches!(mean_pool, Pooling::Mean));
    }

    #[test]
    fn cached_file_direct_rejects_traversal_revisions() {
        require_e5();
        // Each escaping revision resolves to a path that DOES exist on
        // disk — only the reject stands between it and a hit, so deleting
        // the guard fails this test instead of passing vacuously.
        let dotted = format!("./{E5_REV}");
        for (rev, file) in [
            ("..", "refs"),
            ("", E5_REV),
            (dotted.as_str(), "model.safetensors"),
        ] {
            assert!(
                cached_file_direct(E5_ID, rev, file).is_none(),
                "revision {rev:?} must be rejected"
            );
        }
        assert!(cached_file_direct(E5_ID, E5_REV, "model.safetensors").is_some());
    }

    #[test]
    fn argless_load_matches_explicit_default_spec() {
        if !cache_present() {
            eprintln!("HF cache absent — skipping");
            return;
        }
        let a = BertEmbedder::load().unwrap().encode("stability probe").unwrap();
        let b = BertEmbedder::load_with(None, None, None)
            .unwrap()
            .encode("stability probe")
            .unwrap();
        assert_eq!(a, b);
    }
}

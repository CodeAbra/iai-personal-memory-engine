//! BertEmbedder — bge-small-en-v1.5 pure-Rust forward pass via candle-nn.
//!
//! Implements the full BERT inference pipeline:
//!   tokenize → BertEmbeddings → 12× BertLayer → CLS pool → L2 normalize → Vec<f32>
//!
//! Constitutional constraints honored:
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
// Architecture constants — bge-small-en-v1.5 config.json
// ---------------------------------------------------------------------------

const VOCAB_SIZE: usize = 30522;
const HIDDEN_SIZE: usize = 384;
const NUM_HIDDEN_LAYERS: usize = 12;
const NUM_ATTENTION_HEADS: usize = 12;
const INTERMEDIATE_SIZE: usize = 1536;
const MAX_POSITION: usize = 512;
const TYPE_VOCAB_SIZE: usize = 2;
const LAYER_NORM_EPS: f64 = 1e-12;
const HEAD_DIM: usize = HIDDEN_SIZE / NUM_ATTENTION_HEADS; // = 32

const MODEL_ID: &str = "BAAI/bge-small-en-v1.5";
const REVISION: &str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a";

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
fn resolve_model_files(
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf), EmbedError> {
    let repo = Repo::with_revision(MODEL_ID.to_string(), RepoType::Model, REVISION.to_string());
    let offline = std::env::var("IAI_MCP_EMBED_OFFLINE").is_ok();

    if offline {
        let cache = Cache::from_env().repo(repo);
        let weights = cache.get("model.safetensors").ok_or_else(|| {
            EmbedError::HfHub("model.safetensors not in HF cache (offline mode)".into())
        })?;
        let tokenizer = cache.get("tokenizer.json").ok_or_else(|| {
            EmbedError::HfHub("tokenizer.json not in HF cache (offline mode)".into())
        })?;
        let config = cache.get("config.json").ok_or_else(|| {
            EmbedError::HfHub("config.json not in HF cache (offline mode)".into())
        })?;
        Ok((weights, tokenizer, config))
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
    fn load(vb: VarBuilder) -> Result<Self, EmbedError> {
        let vb_emb = vb.pp("embeddings");
        let word_emb = embedding(VOCAB_SIZE, HIDDEN_SIZE, vb_emb.pp("word_embeddings"))?;
        let pos_emb = embedding(MAX_POSITION, HIDDEN_SIZE, vb_emb.pp("position_embeddings"))?;
        let type_emb = embedding(
            TYPE_VOCAB_SIZE,
            HIDDEN_SIZE,
            vb_emb.pp("token_type_embeddings"),
        )?;
        // Pitfall 3: must pass 1e-12 explicitly — candle default is 1e-5
        let layer_norm = layer_norm(HIDDEN_SIZE, LAYER_NORM_EPS, vb_emb.pp("LayerNorm"))?;
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
}

impl BertSelfAttention {
    fn load(vb: VarBuilder) -> Result<Self, EmbedError> {
        let vb_self = vb.pp("attention").pp("self");
        let query = linear(HIDDEN_SIZE, HIDDEN_SIZE, vb_self.pp("query"))?;
        let key = linear(HIDDEN_SIZE, HIDDEN_SIZE, vb_self.pp("key"))?;
        let value = linear(HIDDEN_SIZE, HIDDEN_SIZE, vb_self.pp("value"))?;
        Ok(Self { query, key, value })
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
            .reshape((batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product: Q @ K^T / sqrt(head_dim)
        let scale = (HEAD_DIM as f64).sqrt();
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
            .reshape((batch, seq_len, HIDDEN_SIZE))?)
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
    fn load(vb: VarBuilder, layer_idx: usize) -> Result<Self, EmbedError> {
        let vb_layer = vb.pp("encoder").pp("layer").pp(layer_idx.to_string());

        let attention = BertSelfAttention::load(vb_layer.clone())?;

        let attn_output_dense = linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            vb_layer.pp("attention").pp("output").pp("dense"),
        )?;
        let attn_output_ln = layer_norm(
            HIDDEN_SIZE,
            LAYER_NORM_EPS,
            vb_layer.pp("attention").pp("output").pp("LayerNorm"),
        )?;

        // FFN: 384 → 1536 (intermediate) then 1536 → 384 (output)
        let ffn_intermediate = linear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            vb_layer.pp("intermediate").pp("dense"),
        )?;
        let ffn_output = linear(
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            vb_layer.pp("output").pp("dense"),
        )?;
        let output_ln = layer_norm(
            HIDDEN_SIZE,
            LAYER_NORM_EPS,
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
    fn load(vb: VarBuilder) -> Result<Self, EmbedError> {
        let layers = (0..NUM_HIDDEN_LAYERS)
            .map(|i| BertLayer::load(vb.clone(), i))
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
}

impl BertEmbedder {
    /// Load bge-small-en-v1.5 from the HF cache. The cache location honors
    /// `HF_HOME` (falling back to `~/.cache/huggingface/hub/` when unset).
    /// Triggers a lazy download from HF Hub if the snapshot is absent
    /// (unless `IAI_MCP_EMBED_OFFLINE=1` is set, in which case this fails loudly).
    pub fn load() -> Result<Self, EmbedError> {
        let (weights_path, tokenizer_path, _config_path) = resolve_model_files()?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        // Pitfall 7: must configure truncation to 512; encode() alone does NOT truncate
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_POSITION,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        #[cfg(feature = "metal")]
        let device = Device::new_metal(0)?;
        #[cfg(not(feature = "metal"))]
        let device = Device::Cpu;

        // Pattern 4: unsafe scoped to mmap call; file integrity assumed via HF CDN + REVISION pin
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device) }?;

        let embeddings = BertEmbeddings::load(vb.clone())?;
        let encoder = BertEncoder::load(vb)?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("embed-rayon-{i}"))
            .build()
            .map_err(|e| EmbedError::Pool(e.to_string()))?;

        Ok(Self {
            embeddings,
            encoder,
            tokenizer,
            device,
            pool,
        })
    }

    /// Encode a single text string to a 384-dim L2-normalized embedding.
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        // The whole pass runs inside the private pool so no tokenizer or
        // candle kernel can ever park on the shared global registry.
        self.pool.install(|| self.encode_inner(text))
    }

    fn encode_inner(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
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

        // Pattern 8 / Pitfall 1: raw CLS token — NOT BertPooler dense+tanh
        let cls = encoded.i((0, 0))?;

        // L2 normalize
        let norm = cls.sqr()?.sum_keepdim(0)?.sqrt()?;
        let normalized = cls.broadcast_div(&norm)?;

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
    fn constants_unchanged() {
        assert_eq!(HIDDEN_SIZE, 384);
        assert_eq!(NUM_HIDDEN_LAYERS, 12);
        assert_eq!(HEAD_DIM, HIDDEN_SIZE / NUM_ATTENTION_HEADS);
        assert_eq!(LAYER_NORM_EPS, 1e-12_f64);
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
}

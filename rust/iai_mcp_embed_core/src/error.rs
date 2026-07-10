//! Typed errors that map to Python exceptions via PyO3.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("candle tensor error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("hf-hub error: {0}")]
    HfHub(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("thread pool error: {0}")]
    Pool(String),
}

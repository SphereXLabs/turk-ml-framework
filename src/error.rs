use thiserror::Error;
use std::fmt;

#[derive(Debug, Error)]
pub enum TurkError {
    #[error("Geçersiz katman tanımı: {0}")]
    InvalidLayer(String),
    
    #[error("Eğitim hatası: loss {0} limiti aştı")]
    TrainingDiverged(f32),
    
    #[error("Optimizer hatası: {0}")]
    OptimizerError(String),
    
    #[error("Şekil uyuşmazlığı: {0}")]
    ShapeMismatch(String),
    
    #[error("Model hatası: {0}")]
    ModelError(String),
    
    #[error("Girdi hatası: {0}")]
    InvalidInput(String),
    
    #[error("Ayrıştırma hatası: {0}")]
    ParseError(#[from] crate::parser::error::ParseError),
    
    #[error("Sözlüksel analiz hatası: {0}")]
    LexerError(String),
    
    #[error("Tür hatası: {0}")]
    TypeError(String),
    
    #[error("IO hatası: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization hatası: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Ndarray hatası: {0}")]
    NdarrayError(#[from] ndarray::ShapeError),
    
    #[error("Dinamik katman yapılandırma hatası: {0}")]
    DynamicLayerConfigError(String),
    
    #[error("GPU hesaplama hatası: {0}")]
    GpuComputeError(String),
    
    #[error("Bellek yetersiz: {0}")]
    OutOfMemoryError(String),
    
    #[error("Ağ geri yayılım hatası: {0}")]
    BackpropagationError(String),
}

pub type Result<T> = std::result::Result<T, TurkError>;

impl fmt::Display for TurkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TurkError::InvalidLayer(msg) => write!(f, "Geçersiz katman tanımı: {}", msg),
            TurkError::TrainingDiverged(loss) => write!(f, "Eğitim hatası: loss {} limiti aştı", loss),
            TurkError::OptimizerError(msg) => write!(f, "Optimizer hatası: {}", msg),
            TurkError::ShapeMismatch(msg) => write!(f, "Şekil uyuşmazlığı: {}", msg),
            TurkError::ModelError(msg) => write!(f, "Model hatası: {}", msg),
            TurkError::InvalidInput(msg) => write!(f, "Girdi hatası: {}", msg),
            TurkError::ParseError(err) => write!(f, "Ayrıştırma hatası: {}", err),
            TurkError::LexerError(msg) => write!(f, "Sözlüksel analiz hatası: {}", msg),
            TurkError::TypeError(msg) => write!(f, "Tür hatası: {}", msg),
            TurkError::IoError(err) => write!(f, "IO hatası: {}", err),
            TurkError::SerializationError(err) => write!(f, "Serialization hatası: {}", err),
            TurkError::NdarrayError(err) => write!(f, "Ndarray hatası: {}", err),
            TurkError::DynamicLayerConfigError(msg) => write!(f, "Dinamik katman yapılandırma hatası: {}", msg),
            TurkError::GpuComputeError(msg) => write!(f, "GPU hesaplama hatası: {}", msg),
            TurkError::OutOfMemoryError(msg) => write!(f, "Bellek yetersiz: {}", msg),
            TurkError::BackpropagationError(msg) => write!(f, "Ağ geri yayılım hatası: {}", msg),
        }
    }
} 
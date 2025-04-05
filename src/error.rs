use thiserror::Error;

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
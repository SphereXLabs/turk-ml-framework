use ndarray::ShapeError;
use serde::{Deserialize, Serialize};
use serde_json::Error as SerdeError;
use std::io;
use thiserror::Error;

/// Türk hataları
#[derive(Error, Debug)]
pub enum TurkError {
    /// Sözdizimi hatası
    #[error("Sözdizimi hatası: {0}")]
    SyntaxError(String),

    /// Değişken bulunamadı
    #[error("Değişken bulunamadı: {0}")]
    VariableNotFound(String),

    /// Tip hatası
    #[error("Tip hatası: {0}")]
    TypeError(String),

    /// Değer hatası
    #[error("Değer hatası: {0}")]
    ValueError(String),

    /// Ayrıştırma hatası
    #[error("Ayrıştırma hatası: {0}")]
    ParseError(String),

    /// Çalıştırma hatası
    #[error("Çalıştırma hatası: {0}")]
    RuntimeError(String),

    /// Model hatası
    #[error("Model hatası: {0}")]
    ModelError(String),

    /// Katman hatası
    #[error("Layer hatası: {0}")]
    LayerError(String),

    /// Optimizer hatası
    #[error("Optimizer hatası: {0}")]
    OptimizerError(String),

    /// Kayıp fonksiyonu hatası
    #[error("Kayıp fonksiyonu hatası: {0}")]
    LossError(String),

    /// Metrik hatası
    #[error("Metrik hatası: {0}")]
    MetricError(String),

    /// Veri hatası
    #[error("Veri hatası: {0}")]
    DataError(String),

    /// G/Ç hatası
    #[error("G/Ç hatası: {0}")]
    IOError(String),

    /// Serileştirme hatası
    #[error("Serileştirme hatası: {0}")]
    SerializationError(String),

    /// Deserileştirme hatası
    #[error("Deserileştirme hatası: {0}")]
    DeserializationError(String),

    /// İç hata
    #[error("İç hata: {0}")]
    InternalError(String),

    /// Boyut uyuşmazlığı
    #[error("Boyut uyuşmazlığı: {0}")]
    DimensionMismatch(String),

    /// Şekil uyuşmazlığı
    #[error("Şekil uyuşmazlığı: {0}")]
    ShapeMismatch(String),

    /// Geçersiz değer
    #[error("Geçersiz değer: {0}")]
    InvalidValue(String),

    /// Dosya hatası
    #[error("Dosya hatası: {0}")]
    FileError(String),

    /// ONNX dönüşüm hatası
    #[error("ONNX hatası: {0}")]
    OnnxError(String),

    /// Uygulanmamış
    #[error("Bu özellik henüz uygulanmamış")]
    NotImplemented,
}

impl TurkError {
    /// Sözdizimi hatası
    pub fn syntax_error(message: &str) -> Self {
        Self::SyntaxError(message.to_string())
    }

    /// Değişken bulunamadı
    pub fn variable_not_found(name: String) -> Self {
        Self::VariableNotFound(name)
    }

    /// Tip hatası
    pub fn type_error(message: &str) -> Self {
        Self::TypeError(message.to_string())
    }

    /// Ayrıştırma hatası
    pub fn parse_error(message: &str) -> Self {
        Self::ParseError(message.to_string())
    }

    /// Çalıştırma hatası
    pub fn runtime_error(message: &str) -> Self {
        Self::RuntimeError(message.to_string())
    }

    /// Model hatası
    pub fn model_error(message: &str) -> Self {
        Self::ModelError(message.to_string())
    }

    /// Katman hatası
    pub fn layer_error(message: &str) -> Self {
        Self::LayerError(message.to_string())
    }

    /// Optimizer hatası
    pub fn optimizer_error(message: &str) -> Self {
        Self::OptimizerError(message.to_string())
    }

    /// Kayıp fonksiyonu hatası
    pub fn loss_error(message: &str) -> Self {
        Self::LossError(message.to_string())
    }

    /// Metrik hatası
    pub fn metric_error(message: &str) -> Self {
        Self::MetricError(message.to_string())
    }

    /// Veri hatası
    pub fn data_error(message: &str) -> Self {
        Self::DataError(message.to_string())
    }

    /// G/Ç hatası
    pub fn io_error(message: &str) -> Self {
        Self::IOError(message.to_string())
    }

    /// Serileştirme hatası
    pub fn serialization_error(message: &str) -> Self {
        Self::SerializationError(message.to_string())
    }

    /// Deserileştirme hatası
    pub fn deserialization_error(message: &str) -> Self {
        Self::DeserializationError(message.to_string())
    }

    /// İç hata
    pub fn internal_error(message: &str) -> Self {
        Self::InternalError(message.to_string())
    }

    /// Boyut uyuşmazlığı
    pub fn dimension_mismatch(message: &str) -> Self {
        Self::DimensionMismatch(message.to_string())
    }

    /// Şekil uyuşmazlığı
    pub fn shape_mismatch(message: &str) -> Self {
        Self::ShapeMismatch(message.to_string())
    }

    /// Geçersiz değer
    pub fn invalid_value(message: &str) -> Self {
        Self::InvalidValue(message.to_string())
    }

    /// Dosya hatası
    pub fn file_error(message: &str) -> Self {
        Self::FileError(message.to_string())
    }

    /// ONNX hatası
    pub fn onnx_error(message: impl Into<String>) -> Self {
        Self::OnnxError(message.into())
    }
}

/// Sonuç tipi
pub type Result<T> = std::result::Result<T, TurkError>;

impl From<io::Error> for TurkError {
    fn from(error: io::Error) -> Self {
        TurkError::IOError(error.to_string())
    }
}

impl From<SerdeError> for TurkError {
    fn from(error: SerdeError) -> Self {
        TurkError::SerializationError(error.to_string())
    }
}

impl From<ShapeError> for TurkError {
    fn from(error: ShapeError) -> Self {
        TurkError::DimensionMismatch(error.to_string())
    }
}

/// Aktivasyon
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Aktivasyon {
    /// ReLU
    ReLU,
    /// Sigmoid
    Sigmoid,
    /// Tanh
    Tanh,
    /// Softmax
    Softmax,
    /// LeakyReLU
    LeakyReLU(f64),
    /// Swish
    Swish,
    /// Hiçbiri
    Yok,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_string() {
        let err = TurkError::SyntaxError("hatalı girdi".to_string());
        assert_eq!(err.to_string(), "Sözdizimi hatası: hatalı girdi");

        let err = TurkError::DimensionMismatch("10x10 vs 5x5".to_string());
        assert_eq!(err.to_string(), "Boyut uyuşmazlığı: 10x10 vs 5x5");

        let err = TurkError::LayerError("geçersiz katman".to_string());
        assert_eq!(err.to_string(), "Layer hatası: geçersiz katman");
    }

    #[test]
    fn test_error_display() {
        let err = TurkError::SyntaxError("Eksik parantez".to_string());
        assert_eq!(err.to_string(), "Sözdizimi hatası: Eksik parantez");

        let err = TurkError::TypeError("Beklenen tip: sayı, bulunan: metin".to_string());
        assert_eq!(
            err.to_string(),
            "Tip hatası: Beklenen tip: sayı, bulunan: metin"
        );

        let io_err = io::Error::new(io::ErrorKind::NotFound, "Dosya bulunamadı");
        let err = TurkError::from(io_err);
        assert!(err.to_string().contains("G/Ç hatası"));
        assert!(err.to_string().contains("Dosya bulunamadı"));
    }

    #[test]
    fn test_error_messages() {
        let shape_error = TurkError::dimension_mismatch("Beklenen: (2,3), Alınan: (3,2)");
        assert!(format!("{}", shape_error).contains("Boyut uyuşmazlığı"));

        let data_error = TurkError::data_error("Veri formatı hatalı");
        assert!(format!("{}", data_error).contains("Veri hatası"));

        let ser_error = TurkError::serialization_error("JSON hatası");
        assert!(format!("{}", ser_error).contains("Serileştirme hatası"));

        let file_error = TurkError::IOError("Dosya bulunamadı".to_string());
        assert!(format!("{}", file_error).contains("G/Ç hatası"));
        assert!(format!("{}", file_error).contains("Dosya bulunamadı"));

        let model_error = TurkError::model_error("Model yüklenemedi");
        assert!(format!("{}", model_error).contains("Model hatası"));

        let opt_error = TurkError::optimizer_error("Gradyan hesaplanamadı");
        assert!(format!("{}", opt_error).contains("Optimizer hatası"));

        let invalid_input = TurkError::IOError("Geçersiz giriş".to_string());
        assert!(format!("{}", invalid_input).contains("G/Ç hatası"));
        assert!(format!("{}", invalid_input).contains("Geçersiz giriş"));
    }
}

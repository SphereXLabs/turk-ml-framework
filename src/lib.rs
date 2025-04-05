pub mod ast;
pub mod parser;
pub mod token;
pub mod model;
pub mod layer;
pub mod optimizer;
pub mod loss;
pub mod utils;
pub mod error;
pub mod initializers;
pub mod metrics;
pub mod preprocessing;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use crate::ast::{Deyim, Ifade, Program, YapayZekaModeli, DegerTipi};
pub use crate::error::{TurkError, Result};
pub use crate::initializers::{Initializer, Xavier, HeNormal, Zeros, Ones, RandomNormal};
pub use crate::layer::{Dense, Activation, Layer, Dropout, BatchNorm};
pub use crate::loss::{Loss, CrossEntropy, MSE};
pub use crate::metrics::{Metric, Accuracy, Precision, Recall, F1Score};
pub use crate::model::{Sequential, Model};
pub use crate::optimizer::{Optimizer, SGD, Adam, AdamW, RMSProp};
pub use crate::parser::Parser;
pub use crate::preprocessing::{StandardScaler, MinMaxScaler, Normalizer};
pub use crate::token::Token;
pub use crate::utils::{one_hot_encode, normalize, train_test_split, accuracy, confusion_matrix};

/// Versiyon bilgisi
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Kütüphane bilgilerini döndürür
pub fn library_info() -> String {
    format!(
        "TURK Dili v{}\nTürkçe yapay zeka programlama dili\n\nÖzellikler:\n- Katmanlar: Dense, Dropout, BatchNorm\n- Optimizerlar: SGD, Adam, AdamW, RMSProp\n- Python Binding: {}\n- WASM Desteği: {}", 
        VERSION,
        cfg!(feature = "python"),
        cfg!(feature = "wasm")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_type() {
        let result: Result<()> = Ok(());
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
    
    #[test]
    fn test_library_info() {
        let info = library_info();
        assert!(info.contains("TURK Dili"));
        assert!(info.contains("Türkçe yapay zeka programlama dili"));
    }
} 
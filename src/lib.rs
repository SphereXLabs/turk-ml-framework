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

// Re-export items using the module name directly
pub use ast::{Deyim, Ifade, Program, YapayZekaModeli, DegerTipi};
pub use error::{TurkError, Result};
pub use initializers::{Initializer, Xavier, HeNormal, Zeros, Ones, RandomNormal};
pub use layer::{Dense, Activation, Layer, Dropout, BatchNorm};
pub use loss::{Loss, CrossEntropy, MSE};
pub use metrics::{Metric, Accuracy, Precision, Recall, F1Score};
pub use model::{Sequential, Model};
pub use optimizer::{Optimizer, SGD, Adam, AdamW, RMSProp};
pub use parser::Parser;
pub use parser::model_parser::ModelParser;
pub use parser::expr_parser::ExprParser;
pub use preprocessing::{StandardScaler, MinMaxScaler, Normalizer};
pub use token::Token;
pub use utils::{one_hot_encode, normalize, train_test_split, accuracy, confusion_matrix};

/// Versiyon bilgisi
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Kütüphane bilgilerini döndürür
pub fn library_info() -> String {
    format!(
        "TURK Dili v{}\nTürkçe yapay zeka programlama dili\n\nÖzellikler:\n- Katmanlar: Dense, Dropout, BatchNorm\n- Optimizer'lar: SGD, Adam, AdamW, RMSProp\n- Python Binding: {}\n- WASM Desteği: {}",
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
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

use ndarray::Array2;
use std::fmt::Debug;

pub use crate::ast::{Deyim, Ifade, Program, YapayZekaModeli, DegerTipi};
pub use crate::error::{TurkError, Result};
pub use crate::initializers::{Initializer, Xavier, HeNormal, Zeros, Ones, RandomNormal};
pub use crate::layer::{Dense, Activation, Layer, Dropout, BatchNorm};
pub use crate::loss::{Loss, CrossEntropy, MSE};
pub use crate::metrics::{Metric, Accuracy, Precision, Recall, F1Score};
pub use crate::model::{Sequential, Model};
pub use crate::optimizer::{Optimizer, SGD, Adam, AdamW};
pub use crate::parser::Parser;
pub use crate::preprocessing::{StandardScaler, MinMaxScaler, Normalizer};
pub use crate::token::Token;
pub use crate::utils::{one_hot_encode, normalize, train_test_split, accuracy, confusion_matrix};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_result_type() {
        let result: Result<()> = Ok(());
        assert!(result.is_ok());
    }
} 
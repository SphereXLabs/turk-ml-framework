pub mod ast;
pub mod parser;
pub mod token;
pub mod model;
pub mod layer;
pub mod optimizer;
pub mod loss;
pub mod utils;

use std::error::Error;
use std::fmt;
use std::io;
use serde_json;
use ndarray::ShapeError;
use std::fmt::Debug;
use ndarray::Array2;

pub use model::Sequential;
pub use layer::{Dense, Activation};
pub use optimizer::{Optimizer, SGD, Adam};
pub use loss::{Loss, CrossEntropy, MSE};
pub use utils::{one_hot_encode, normalize, train_test_split, accuracy, confusion_matrix};

pub type Result<T> = std::result::Result<T, TinyMLError>;

#[derive(Debug)]
pub enum TinyMLError {
    ShapeMismatch(String),
    InvalidInput(String),
    ModelError(String),
    LayerError(String),
    OptimizerError(String),
    LossError(String),
    IOError(io::Error),
    SerializationError(serde_json::Error),
    ShapeError(ShapeError),
}

impl fmt::Display for TinyMLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TinyMLError::ShapeMismatch(msg) => write!(f, "Şekil uyuşmazlığı: {}", msg),
            TinyMLError::InvalidInput(msg) => write!(f, "Geçersiz girdi: {}", msg),
            TinyMLError::ModelError(msg) => write!(f, "Model hatası: {}", msg),
            TinyMLError::LayerError(msg) => write!(f, "Katman hatası: {}", msg),
            TinyMLError::OptimizerError(msg) => write!(f, "Optimizer hatası: {}", msg),
            TinyMLError::LossError(msg) => write!(f, "Kayıp fonksiyonu hatası: {}", msg),
            TinyMLError::IOError(e) => write!(f, "IO hatası: {}", e),
            TinyMLError::SerializationError(e) => write!(f, "Serileştirme hatası: {}", e),
            TinyMLError::ShapeError(e) => write!(f, "Şekil hatası: {}", e),
        }
    }
}

impl Error for TinyMLError {}

impl From<io::Error> for TinyMLError {
    fn from(err: io::Error) -> Self {
        TinyMLError::IOError(err)
    }
}

impl From<serde_json::Error> for TinyMLError {
    fn from(err: serde_json::Error) -> Self {
        TinyMLError::SerializationError(err)
    }
}

impl From<ShapeError> for TinyMLError {
    fn from(err: ShapeError) -> Self {
        TinyMLError::ShapeError(err)
    }
}

pub trait Layer: Debug {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>>;
    fn backward(&mut self, grad: &Array2<f32>) -> Result<Array2<f32>>;
    fn get_weights(&self) -> &Array2<f32>;
    fn get_bias(&self) -> &Array2<f32>;
    fn get_weight_grads(&self) -> &Array2<f32>;
    fn get_bias_grads(&self) -> &Array2<f32>;
    fn update(&mut self) -> Result<()>;
    fn activation_type(&self) -> Activation;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
    fn weights_as_vec(&self) -> Vec<f32>;
    fn bias_as_vec(&self) -> Vec<f32>;
    fn set_weights(&mut self, weights: Array2<f32>) -> Result<()>;
    fn set_bias(&mut self, bias: Array2<f32>) -> Result<()>;
}

pub trait Model: Debug {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>>;
    fn backward(&mut self, grad: &Array2<f32>) -> Result<()>;
    fn update(&mut self) -> Result<()>;
    fn save(&self, path: &str) -> Result<()>;
    fn load(&mut self, path: &str) -> Result<()>;
}

pub use crate::ast::{Deyim, Ifade, Program, YapayZekaModeli, DegerTipi};
pub use crate::parser::Parser;
pub use crate::token::Token;

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
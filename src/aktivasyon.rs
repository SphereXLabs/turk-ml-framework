//! Aktivasyon fonksiyonları modülü
//!
//! Bu modül, sinir ağı katmanlarında kullanılan aktivasyon fonksiyonlarını içerir.
//! Her aktivasyon fonksiyonu için ileri yayılım (forward) ve geri yayılım (backward)
//! hesaplamaları burada tanımlanmıştır.

use crate::error::{Result, TurkError};
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Aktivasyon fonksiyonları
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    Relu,
    LeakyRelu(f64), // alpha değeri
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
    Elu(f64), // alpha değeri
    Selu,
    Swish,
}

impl Activation {
    /// Metinden aktivasyon oluşturur
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "relu" => Ok(Self::Relu),
            "leakyrelu" => Ok(Self::LeakyRelu(0.01)),
            "sigmoid" => Ok(Self::Sigmoid),
            "tanh" => Ok(Self::Tanh),
            "softmax" => Ok(Self::Softmax),
            "linear" => Ok(Self::Linear),
            "elu" => Ok(Self::Elu(1.0)),
            "selu" => Ok(Self::Selu),
            "swish" => Ok(Self::Swish),
            _ => Err(TurkError::LayerError(format!(
                "Bilinmeyen aktivasyon: {}",
                s
            ))),
        }
    }

    /// İleri hesaplama
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Relu => x.mapv(|v| if v > 0.0 { v } else { 0.0 }),
            Self::LeakyRelu(alpha) => x.mapv(|v| if v > 0.0 { v } else { v * alpha }),
            Self::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Self::Tanh => x.mapv(|v| v.tanh()),
            Self::Softmax => {
                let mut result = Array2::zeros(x.dim());
                for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
                    let x_row = x.row(i);
                    let max = x_row.fold(f64::NEG_INFINITY, |max, &v| max.max(v));
                    let exp_x = x_row.mapv(|v| (v - max).exp());
                    let sum_exp = exp_x.sum();
                    for (j, v) in row.iter_mut().enumerate() {
                        *v = exp_x[j] / sum_exp;
                    }
                }
                result
            }
            Self::Linear => x.to_owned(),
            Self::Elu(alpha) => x.mapv(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }),
            Self::Selu => x.mapv(|v| {
                let alpha = 1.6732632423543772;
                let scale = 1.0507009873554804;
                if v > 0.0 {
                    scale * v
                } else {
                    scale * alpha * (v.exp() - 1.0)
                }
            }),
            Self::Swish => {
                let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                sigmoid.clone() + x * &sigmoid.mapv(|v| v * (1.0 - v))
            }
        }
    }

    /// Geri hesaplama (türev)
    pub fn backward(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            Self::Relu => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Self::LeakyRelu(alpha) => x.mapv(|v| if v > 0.0 { 1.0 } else { *alpha }),
            Self::Sigmoid => {
                let sigmoid = self.forward(x);
                sigmoid.mapv(|v| v * (1.0 - v))
            }
            Self::Tanh => {
                let tanh = x.mapv(|v| v.tanh());
                tanh.mapv(|v| 1.0 - v * v)
            }
            Self::Softmax => {
                // Softmax türevi karmaşıktır, genellikle cross-entropy ile birlikte kullanılır
                // Bu durumda türev daha basittir (y_pred - y_true)
                Array2::ones(x.dim())
            }
            Self::Linear => Array2::ones(x.dim()),
            Self::Elu(alpha) => x.mapv(|v| if v > 0.0 { 1.0 } else { alpha * v.exp() }),
            Self::Selu => x.mapv(|v| {
                let alpha = 1.6732632423543772;
                let scale = 1.0507009873554804;
                if v > 0.0 {
                    scale
                } else {
                    scale * alpha * v.exp()
                }
            }),
            Self::Swish => {
                let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                sigmoid.clone() + x * &sigmoid.mapv(|v| v * (1.0 - v))
            }
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Relu => write!(f, "relu"),
            Self::LeakyRelu(alpha) => write!(f, "leaky_relu(alpha={})", alpha),
            Self::Sigmoid => write!(f, "sigmoid"),
            Self::Tanh => write!(f, "tanh"),
            Self::Softmax => write!(f, "softmax"),
            Self::Linear => write!(f, "linear"),
            Self::Elu(alpha) => write!(f, "elu(alpha={})", alpha),
            Self::Selu => write!(f, "selu"),
            Self::Swish => write!(f, "swish"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_activation_functions() {
        let input = arr2(&[[1.0, -1.0], [0.5, -0.5]]);

        // ReLU testi
        let relu_output = Activation::Relu.forward(&input);
        assert_eq!(relu_output[[0, 0]], 1.0);
        assert_eq!(relu_output[[0, 1]], 0.0);

        // Sigmoid testi
        let sigmoid_output = Activation::Sigmoid.forward(&input);
        assert!(sigmoid_output[[0, 0]] > 0.5);
        assert!(sigmoid_output[[0, 1]] < 0.5);

        // Tanh testi
        let tanh_output = Activation::Tanh.forward(&input);
        assert!(tanh_output[[0, 0]] > 0.0);
        assert!(tanh_output[[0, 1]] < 0.0);
    }
}

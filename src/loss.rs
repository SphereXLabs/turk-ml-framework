use crate::{Result, TurkError};
use ndarray::prelude::*;
use ndarray::Array2;
use serde::{Serialize, Deserialize};

/// Kayıp fonksiyonu trait'i
pub trait Loss: Send + Sync {
    /// Kayıp değerini hesaplar
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32>;
    
    /// Kayıp fonksiyonunun gradyanını hesaplar
    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>>;
    
    /// Kayıp fonksiyonunun adını döndürür
    fn name(&self) -> &'static str;
    
    /// Kayıp fonksiyonunun klonunu oluşturur
    fn clone_box(&self) -> Box<dyn Loss>;
}

/// Çapraz Entropi kayıp fonksiyonu
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEntropy {
    pub epsilon: f32,
    pub from_logits: bool,
}

impl CrossEntropy {
    pub fn new(from_logits: bool) -> Self {
        Self {
            epsilon: 1e-7,
            from_logits,
        }
    }
}

impl Default for CrossEntropy {
    fn default() -> Self {
        Self {
            epsilon: 1e-7,
            from_logits: false,
        }
    }
}

impl Loss for CrossEntropy {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "CrossEntropy: Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let predictions = if self.from_logits {
            // Logits'ten olasılıklara dönüşüm (softmax)
            let max_vals = predictions.map_axis(Axis(1), |row| {
                row.fold(f32::NEG_INFINITY, |max, &val| max.max(val))
            });
            
            let exp_preds = predictions.outer_iter()
                .zip(max_vals.iter())
                .map(|(row, &max_val)| {
                    row.mapv(|val| (val - max_val).exp())
                })
                .collect::<Vec<_>>();
                
            let sum_exp = exp_preds.iter()
                .map(|row| row.sum())
                .collect::<Vec<_>>();
                
            let mut softmax_probs = Array2::zeros(predictions.raw_dim());
            
            for ((i, row), &sum) in exp_preds.iter().enumerate().zip(sum_exp.iter()) {
                for (j, &val) in row.iter().enumerate() {
                    softmax_probs[[i, j]] = val / sum;
                }
            }
            
            softmax_probs
        } else {
            predictions.clone()
        };
        
        let clipped_predictions = predictions.mapv(|x| x.max(self.epsilon).min(1.0 - self.epsilon));
        
        let batch_size = predictions.shape()[0] as f32;
        let loss = -targets.iter()
            .zip(clipped_predictions.iter())
            .map(|(&t, &p)| t * p.ln())
            .sum::<f32>();

        Ok(loss / batch_size)
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        if predictions.shape() != targets.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "CrossEntropy gradient: Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let batch_size = predictions.shape()[0] as f32;
        
        if self.from_logits {
            // Softmax + Çapraz entropi birleşik gradient:
            // Bu durumda softmax çıktıları - hedefler formülü kullanılır
            let max_vals = predictions.map_axis(Axis(1), |row| {
                row.fold(f32::NEG_INFINITY, |max, &val| max.max(val))
            });
            
            let mut softmax_probs = Array2::zeros(predictions.raw_dim());
            
            for (i, row) in predictions.outer_iter().enumerate() {
                let max_val = max_vals[i];
                let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exp: f32 = exp_vals.iter().sum();
                
                for (j, &exp_val) in exp_vals.iter().enumerate() {
                    softmax_probs[[i, j]] = exp_val / sum_exp;
                }
            }
            
            // softmax çıktılar - hedefler
            Ok((softmax_probs - targets) / batch_size)
        } else {
            let clipped_predictions = predictions.mapv(|x| x.max(self.epsilon).min(1.0 - self.epsilon));
            
            let grad = Array2::from_shape_fn(predictions.raw_dim(), |(i, j)| {
                if targets[[i, j]] > 0.0 {
                    -targets[[i, j]] / clipped_predictions[[i, j]]
                } else {
                    0.0
                }
            });
            
            Ok(grad / batch_size)
        }
    }
    
    fn name(&self) -> &'static str {
        "cross_entropy"
    }
    
    fn clone_box(&self) -> Box<dyn Loss> {
        Box::new(self.clone())
    }
}

/// Ortalama Kare Hata (MSE) kayıp fonksiyonu
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MSE;

impl Default for MSE {
    fn default() -> Self {
        Self
    }
}

impl Loss for MSE {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "MSE: Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let batch_size = predictions.shape()[0] as f32;
        let diff = predictions - targets;
        let squared_diff = diff.mapv(|x| x * x);
        let sum_squared_diff = squared_diff.sum();
        
        Ok(sum_squared_diff / batch_size)
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        if predictions.shape() != targets.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "MSE gradient: Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let batch_size = predictions.shape()[0] as f32;
        Ok(2.0 * (predictions - targets) / batch_size)
    }
    
    fn name(&self) -> &'static str {
        "mse"
    }
    
    fn clone_box(&self) -> Box<dyn Loss> {
        Box::new(self.clone())
    }
}

/// Ortalama Mutlak Hata (MAE) kayıp fonksiyonu
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MAE;

impl Default for MAE {
    fn default() -> Self {
        Self
    }
}

impl Loss for MAE {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "MAE: Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let batch_size = predictions.shape()[0] as f32;
        let abs_diff = (predictions - targets).mapv(|x| x.abs());
        
        Ok(abs_diff.sum() / batch_size)
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        if predictions.shape() != targets.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "MAE gradient: Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let batch_size = predictions.shape()[0] as f32;
        let gradient = Array2::from_shape_fn(predictions.raw_dim(), |(i, j)| {
            if predictions[[i, j]] > targets[[i, j]] {
                1.0
            } else if predictions[[i, j]] < targets[[i, j]] {
                -1.0
            } else {
                0.0
            }
        });
        
        Ok(gradient / batch_size)
    }
    
    fn name(&self) -> &'static str {
        "mae"
    }
    
    fn clone_box(&self) -> Box<dyn Loss> {
        Box::new(self.clone())
    }
}

/// Huber kayıp fonksiyonu - MSE ve MAE'nin bir kombinasyonu
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Huber {
    pub delta: f32,
}

impl Huber {
    pub fn new(delta: f32) -> Self {
        Self { delta }
    }
}

impl Default for Huber {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl Loss for Huber {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Huber: Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let batch_size = predictions.shape()[0] as f32;
        let mut loss_sum = 0.0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let error = (pred - target).abs();
            
            if error <= self.delta {
                // MSE bölgesi
                loss_sum += 0.5 * error * error;
            } else {
                // MAE bölgesi
                loss_sum += self.delta * (error - 0.5 * self.delta);
            }
        }
        
        Ok(loss_sum / batch_size)
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        if predictions.shape() != targets.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Huber gradient: Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let batch_size = predictions.shape()[0] as f32;
        let gradient = Array2::from_shape_fn(predictions.raw_dim(), |(i, j)| {
            let error = predictions[[i, j]] - targets[[i, j]];
            
            if error.abs() <= self.delta {
                // MSE bölgesi
                error
            } else {
                // MAE bölgesi
                self.delta * error.signum()
            }
        });
        
        Ok(gradient / batch_size)
    }
    
    fn name(&self) -> &'static str {
        "huber"
    }
    
    fn clone_box(&self) -> Box<dyn Loss> {
        Box::new(self.clone())
    }
}

/// Kategorik Çapraz Entropi (Categorical Cross Entropy) kayıp fonksiyonu
/// Çok sınıflı sınıflandırma için
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalCrossEntropy {
    pub epsilon: f32,
    pub from_logits: bool,
}

impl CategoricalCrossEntropy {
    pub fn new(from_logits: bool) -> Self {
        Self {
            epsilon: 1e-7,
            from_logits,
        }
    }
}

impl Default for CategoricalCrossEntropy {
    fn default() -> Self {
        Self {
            epsilon: 1e-7,
            from_logits: false,
        }
    }
}

impl Loss for CategoricalCrossEntropy {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "CategoricalCrossEntropy: Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let batch_size = predictions.shape()[0] as f32;
        
        // Softmax uygula (eğer gerekiyorsa)
        let predictions = if self.from_logits {
            // Logits'ten olasılıklara dönüşüm (softmax)
            let max_vals = predictions.map_axis(Axis(1), |row| {
                row.fold(f32::NEG_INFINITY, |max, &val| max.max(val))
            });
            
            let mut softmax_probs = Array2::zeros(predictions.raw_dim());
            
            for (i, row) in predictions.outer_iter().enumerate() {
                let max_val = max_vals[i];
                let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exp: f32 = exp_vals.iter().sum();
                
                for (j, &exp_val) in exp_vals.iter().enumerate() {
                    softmax_probs[[i, j]] = exp_val / sum_exp;
                }
            }
            
            softmax_probs
        } else {
            predictions.clone()
        };
        
        let clipped_preds = predictions.mapv(|x| x.max(self.epsilon).min(1.0 - self.epsilon));
        
        // Her örnek için çapraz entropi hesapla ve topla
        let mut loss_sum = 0.0;
        
        for (i, target_row) in targets.outer_iter().enumerate() {
            for (j, &target_val) in target_row.iter().enumerate() {
                if target_val > 0.0 {
                    loss_sum -= target_val * clipped_preds[[i, j]].ln();
                }
            }
        }
        
        Ok(loss_sum / batch_size)
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        if predictions.shape() != targets.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "CategoricalCrossEntropy gradient: Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let batch_size = predictions.shape()[0] as f32;
        
        if self.from_logits {
            // Softmax'in türevi + çapraz entropi basit bir forma indirgenebilir
            // Sonuç: softmax(logits) - targets
            let max_vals = predictions.map_axis(Axis(1), |row| {
                row.fold(f32::NEG_INFINITY, |max, &val| max.max(val))
            });
            
            let mut softmax_probs = Array2::zeros(predictions.raw_dim());
            
            for (i, row) in predictions.outer_iter().enumerate() {
                let max_val = max_vals[i];
                let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exp: f32 = exp_vals.iter().sum();
                
                for (j, &exp_val) in exp_vals.iter().enumerate() {
                    softmax_probs[[i, j]] = exp_val / sum_exp;
                }
            }
            
            Ok((softmax_probs - targets) / batch_size)
        } else {
            let clipped_preds = predictions.mapv(|x| x.max(self.epsilon).min(1.0 - self.epsilon));
            
            let mut gradient = Array2::zeros(predictions.raw_dim());
            
            for i in 0..predictions.shape()[0] {
                for j in 0..predictions.shape()[1] {
                    if targets[[i, j]] > 0.0 {
                        gradient[[i, j]] = -targets[[i, j]] / clipped_preds[[i, j]];
                    }
                }
            }
            
            Ok(gradient / batch_size)
        }
    }
    
    fn name(&self) -> &'static str {
        "categorical_cross_entropy"
    }
    
    fn clone_box(&self) -> Box<dyn Loss> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_mse_loss() {
        let mse = MSE::default();
        
        let predictions = Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let targets = Array::from_shape_vec((2, 1), vec![1.5, 1.0]).unwrap();
        
        let loss = mse.compute(&predictions, &targets).unwrap();
        assert!((loss - 0.625).abs() < 1e-6);
        
        let gradient = mse.gradient(&predictions, &targets).unwrap();
        assert_eq!(gradient.shape(), &[2, 1]);
        assert!((gradient[[0, 0]] + 1.0).abs() < 1e-6);
        assert!((gradient[[1, 0]] - 2.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_mae_loss() {
        let mae = MAE::default();
        
        let predictions = Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let targets = Array::from_shape_vec((2, 1), vec![1.5, 1.0]).unwrap();
        
        let loss = mae.compute(&predictions, &targets).unwrap();
        assert!((loss - 0.75).abs() < 1e-6);
    }
    
    #[test]
    fn test_cross_entropy_loss() {
        let ce = CrossEntropy::new(false);
        
        let predictions = Array::from_shape_vec((2, 3), vec![
            0.7, 0.2, 0.1,
            0.1, 0.8, 0.1,
        ]).unwrap();
        
        let targets = Array::from_shape_vec((2, 3), vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]).unwrap();
        
        let loss = ce.compute(&predictions, &targets).unwrap();
        assert!(loss > 0.0);
    }
    
    #[test]
    fn test_huber_loss() {
        let huber = Huber::new(1.0);
        
        let predictions = Array::from_shape_vec((3, 1), vec![1.0, 2.0, 5.0]).unwrap();
        let targets = Array::from_shape_vec((3, 1), vec![1.5, 1.0, 2.0]).unwrap();
        
        let loss = huber.compute(&predictions, &targets).unwrap();
        assert!(loss > 0.0);
        
        let gradient = huber.gradient(&predictions, &targets).unwrap();
        assert_eq!(gradient.shape(), &[3, 1]);
    }
} 
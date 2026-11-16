use crate::{Result, TurkError};
use ndarray::prelude::*;
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use std::fmt;

/// Loss trait'i
pub trait Loss: fmt::Debug + Send + Sync {
    /// Kayıp değerini hesaplar
    fn loss(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<f64>;
    
    /// Kayıp fonksiyonunun gradyanını hesaplar
    fn gradient(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<Array2<f64>>;
    
    /// Kayıp fonksiyonunun adını döndürür
    fn name(&self) -> &'static str;
    
    /// Kayıp fonksiyonunu klonlar
    fn clone_box(&self) -> Box<dyn Loss>;
}

/// Çapraz Entropi kayıp fonksiyonu
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossEntropy {
    pub epsilon: f64,
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
    fn loss(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<f64> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::DimensionMismatch(format!(
                "Gerçek ve tahmin değerlerin şekilleri uyuşmuyor: {:?} vs {:?}",
                y_true.shape(), y_pred.shape()
            )));
        }
        
        let predictions = if self.from_logits {
            // Logits'ten olasılıklara dönüşüm (softmax)
            let max_vals = y_pred.map_axis(Axis(1), |row| {
                row.fold(f64::NEG_INFINITY, |max, &val| max.max(val))
            });
            
            let mut softmax_probs = Array2::zeros(y_pred.raw_dim());
            
            for i in 0..y_pred.nrows() {
                let row = y_pred.row(i);
                let max_val = max_vals[i];
                let mut exp_sum = 0.0;
                
                // e^(x_i - max) hesapla ve topla
                for j in 0..row.len() {
                    let exp_val = (row[j] - max_val).exp();
                    softmax_probs[[i, j]] = exp_val;
                    exp_sum += exp_val;
                }
                
                // Normalizasyon
                for j in 0..row.len() {
                    softmax_probs[[i, j]] /= exp_sum;
                }
            }
            
            softmax_probs
        } else {
            y_pred.to_owned()
        };
        
        // Epsilon ile clipping
        let predictions = predictions.mapv(|p| p.max(self.epsilon).min(1.0 - self.epsilon));
        
        // Cross entropy hesapla: -sum(y_true * log(y_pred))
        let mut loss = 0.0;
        for i in 0..y_true.nrows() {
            for j in 0..y_true.ncols() {
                if y_true[[i, j]] > 0.0 {
                    loss -= y_true[[i, j]] * predictions[[i, j]].ln();
                }
            }
        }
        
        Ok(loss / y_true.nrows() as f64)
    }
    
    fn gradient(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::DimensionMismatch(format!(
                "Gerçek ve tahmin değerlerin şekilleri uyuşmuyor: {:?} vs {:?}",
                y_true.shape(), y_pred.shape()
            )));
        }
        
        let predictions = if self.from_logits {
            // Logits'ten olasılıklara dönüşüm (softmax)
            let max_vals = y_pred.map_axis(Axis(1), |row| {
                row.fold(f64::NEG_INFINITY, |max, &val| max.max(val))
            });
            
            let mut softmax_probs = Array2::zeros(y_pred.raw_dim());
            
            for i in 0..y_pred.nrows() {
                let row = y_pred.row(i);
                let max_val = max_vals[i];
                let mut exp_sum = 0.0;
                
                // e^(x_i - max) hesapla ve topla
                for j in 0..row.len() {
                    let exp_val = (row[j] - max_val).exp();
                    softmax_probs[[i, j]] = exp_val;
                    exp_sum += exp_val;
                }
                
                // Normalizasyon
                for j in 0..row.len() {
                    softmax_probs[[i, j]] /= exp_sum;
                }
            }
            
            softmax_probs
        } else {
            y_pred.to_owned()
        };
        
        // Softmax CE için gradient hesapla
        let batch_size = y_true.nrows() as f64;
        let mut grad = predictions.clone();
        
        // Gradient: (y_pred - y_true) / batch_size
        grad = grad - y_true;
        grad = grad / batch_size;
        
        Ok(grad)
    }

    fn name(&self) -> &'static str {
        "crossentropy"
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
    fn loss(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<f64> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::DimensionMismatch(format!(
                "Gerçek ve tahmin değerlerin şekilleri uyuşmuyor: {:?} vs {:?}",
                y_true.shape(), y_pred.shape()
            )));
        }
        
        let diff = y_pred - y_true;
        let mse = (diff.clone() * diff).mean().unwrap();
        Ok(mse)
    }
    
    fn gradient(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::DimensionMismatch(format!(
                "Gerçek ve tahmin değerlerin şekilleri uyuşmuyor: {:?} vs {:?}",
                y_true.shape(), y_pred.shape()
            )));
        }
        
        let grad = 2.0 * (y_pred - y_true) / y_true.len() as f64;
        Ok(grad)
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
    fn loss(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<f64> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::DimensionMismatch(format!(
                "Gerçek ve tahmin değerlerin şekilleri uyuşmuyor: {:?} vs {:?}",
                y_true.shape(), y_pred.shape()
            )));
        }
        
        let diff = (y_pred - y_true).mapv(|x| x.abs());
        let mae = diff.sum() / y_true.len() as f64;
        Ok(mae)
    }

    fn gradient(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::DimensionMismatch(format!(
                "Gerçek ve tahmin değerlerin şekilleri uyuşmuyor: {:?} vs {:?}",
                y_true.shape(), y_pred.shape()
            )));
        }
        
        let diff = (y_pred - y_true).mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        });
        
        Ok(diff / y_true.len() as f64)
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
    pub delta: f64,
}

impl Huber {
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl Default for Huber {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl Loss for Huber {
    fn loss(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<f64> {
        let mut loss_sum = 0.0;
        
        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            let error = (t - p).abs();
            
            if error <= self.delta {
                // MSE bölgesi
                loss_sum += 0.5 * error * error;
            } else {
                // MAE bölgesi
                loss_sum += self.delta * (error - 0.5 * self.delta);
            }
        }
        
        Ok(loss_sum / y_true.len() as f64)
    }

    fn gradient(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut gradient = Array2::zeros(y_true.raw_dim());
        
        for (i, (t, p)) in y_true.iter().zip(y_pred.iter()).enumerate() {
            let error = t - p;
            
            if error.abs() <= self.delta {
                // MSE bölgesi
                gradient[[i, 0]] = error;
            } else {
                // MAE bölgesi
                gradient[[i, 0]] = self.delta * error.signum();
            }
        }
        
        Ok(gradient / y_true.len() as f64)
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
pub struct CategoricalCrossEntropy;

impl Loss for CategoricalCrossEntropy {
    fn loss(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<f64> {
        let epsilon = 1e-7;
        let y_pred = y_pred.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        Ok(-(y_true * y_pred.mapv(|v| v.ln())).sum() / y_true.len() as f64)
    }

    fn gradient(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let epsilon = 1e-7;
        let y_pred = y_pred.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        Ok((y_pred - y_true) / y_true.len() as f64)
    }

    fn name(&self) -> &'static str {
        "categorical_crossentropy"
    }
    
    fn clone_box(&self) -> Box<dyn Loss> {
        Box::new(self.clone())
    }
}

/// Binary Crossentropy
#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryCrossentropy;

impl Loss for BinaryCrossentropy {
    fn loss(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<f64> {
        let epsilon = 1e-7;
        let y_pred = y_pred.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        Ok(-(y_true * y_pred.mapv(|v| v.ln()) + (1.0 - y_true) * y_pred.mapv(|v| (1.0 - v).ln()))
            .mean()
            .unwrap())
    }
    
    fn gradient(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let epsilon = 1e-7;
        let y_pred = y_pred.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        let y_pred_clone = y_pred.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        Ok((y_pred_clone.clone() - y_true) / (y_pred_clone.clone() * (1.0 - y_pred_clone)) / y_true.len() as f64)
    }

    fn name(&self) -> &'static str {
        "binary_crossentropy"
    }
    
    fn clone_box(&self) -> Box<dyn Loss> {
        Box::new(BinaryCrossentropy)
    }
}

/// Sparse Categorical Crossentropy
#[derive(Debug, Serialize, Deserialize)]
pub struct SparseCategoricalCrossentropy;

impl Loss for SparseCategoricalCrossentropy {
    fn loss(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<f64> {
        let epsilon = 1e-7;
        let y_pred = y_pred.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        let mut loss = 0.0;
        for i in 0..y_true.nrows() {
            let class = y_true[[i, 0]] as usize;
            loss -= y_pred[[i, class]].ln();
        }
        Ok(loss / y_true.len() as f64)
    }
    
    fn gradient(&self, y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let epsilon = 1e-7;
        let y_pred = y_pred.mapv(|v| v.max(epsilon).min(1.0 - epsilon));
        let mut grad = Array2::zeros(y_pred.raw_dim());
        for i in 0..y_true.nrows() {
            let class = y_true[[i, 0]] as usize;
            grad[[i, class]] = y_pred[[i, class]] - 1.0;
        }
        Ok(grad / y_true.len() as f64)
    }

    fn name(&self) -> &'static str {
        "sparse_categorical_crossentropy"
    }
    
    fn clone_box(&self) -> Box<dyn Loss> {
        Box::new(SparseCategoricalCrossentropy)
    }
}

/// Kayıp fonksiyonu oluştur
pub fn create_loss(loss: &str) -> Result<Box<dyn Loss>> {
    match loss.to_lowercase().as_str() {
        "mse" => Ok(Box::new(MSE::default())),
        "mae" => Ok(Box::new(MAE::default())),
        "huber" => Ok(Box::new(Huber::default())),
        "binary_crossentropy" | "binarycrossentropy" => Ok(Box::new(BinaryCrossentropy)),
        "categorical_crossentropy" | "categoricalcrossentropy" => Ok(Box::new(CategoricalCrossEntropy)),
        "sparse_categorical_crossentropy" | "sparsecategoricalcrossentropy" => Ok(Box::new(SparseCategoricalCrossentropy)),
        _ => Err(TurkError::LossError(format!("Geçersiz kayıp fonksiyonu: {}", loss))),
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
        
        let loss = mse.loss(&targets.view(), &predictions.view()).unwrap();
        assert!((loss - 0.625).abs() < 1e-6);
        
        let gradient = mse.gradient(&targets.view(), &predictions.view()).unwrap();
        assert_eq!(gradient.shape(), &[2, 1]);
        assert!((gradient[[0, 0]] + 0.5).abs() < 1e-6);  // -0.5 değerini kontrol et
        assert!((gradient[[1, 0]] - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_mae_loss() {
        let mae = MAE::default();
        
        let predictions = Array::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let targets = Array::from_shape_vec((2, 1), vec![1.5, 1.0]).unwrap();
        
        let loss = mae.loss(&targets.view(), &predictions.view()).unwrap();
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
        
        let loss = ce.loss(&targets.view(), &predictions.view()).unwrap();
        assert!(loss > 0.0);
    }
    
    #[test]
    fn test_huber_loss() {
        let huber = Huber::new(1.0);
        
        let predictions = Array::from_shape_vec((3, 1), vec![1.0, 2.0, 5.0]).unwrap();
        let targets = Array::from_shape_vec((3, 1), vec![1.5, 1.0, 2.0]).unwrap();
        
        let loss = huber.loss(&targets.view(), &predictions.view()).unwrap();
        assert!(loss > 0.0);
    }
} 
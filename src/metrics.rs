use ndarray::{Array1, Array2, Axis};
use crate::error::{Result, TurkError};
use serde::{Serialize, Deserialize};

/// Performans metriği trait'i
pub trait Metric: Send + Sync {
    fn compute(&self, y_true: &Array2<f32>, y_pred: &Array2<f32>) -> Result<f32>;
    fn name(&self) -> &'static str;
}

/// Doğruluk (Accuracy) ölçümü
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Accuracy;

impl Metric for Accuracy {
    fn compute(&self, y_true: &Array2<f32>, y_pred: &Array2<f32>) -> Result<f32> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Gerçek ve tahmin edilen çıktıların şekilleri uyuşmuyor: {:?} vs {:?}",
                y_true.shape(), y_pred.shape()
            )));
        }
        
        let true_classes: Vec<usize> = y_true
            .axis_iter(Axis(1))
            .map(|row| row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0)
            .collect();
            
        let pred_classes: Vec<usize> = y_pred
            .axis_iter(Axis(1))
            .map(|row| row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0)
            .collect();
            
        let correct = true_classes.iter().zip(pred_classes.iter()).filter(|&(a, b)| a == b).count();
        let total = y_true.shape()[0];
        
        Ok(correct as f32 / total as f32)
    }
    
    fn name(&self) -> &'static str {
        "accuracy"
    }
}

/// Hassasiyet (Precision) ölçümü
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Precision {
    pub average: String, // 'micro', 'macro', 'weighted', veya 'None' (belirli bir sınıf için)
    pub class_idx: Option<usize>, // 'None' olmadığında, belirli bir sınıfın indeksi
}

impl Default for Precision {
    fn default() -> Self {
        Self {
            average: "macro".to_string(),
            class_idx: None,
        }
    }
}

impl Metric for Precision {
    fn compute(&self, y_true: &Array2<f32>, y_pred: &Array2<f32>) -> Result<f32> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Gerçek ve tahmin edilen çıktıların şekilleri uyuşmuyor: {:?} vs {:?}",
                y_true.shape(), y_pred.shape()
            )));
        }
        
        let n_classes = y_true.shape()[1];
        let mut class_precision = Vec::new();
        
        let true_classes: Vec<usize> = y_true
            .axis_iter(Axis(1))
            .map(|row| row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0)
            .collect();
            
        let pred_classes: Vec<usize> = y_pred
            .axis_iter(Axis(1))
            .map(|row| row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0)
            .collect();
            
        // Belirli bir sınıf için hassasiyet hesaplama
        if let Some(idx) = self.class_idx {
            if idx >= n_classes {
                return Err(TurkError::InvalidInput(format!(
                    "Geçersiz sınıf indeksi: {}. Toplam sınıf sayısı: {}", idx, n_classes
                )));
            }
            
            let true_positives = true_classes.iter().zip(pred_classes.iter())
                .filter(|&(t, p)| *t == idx && *p == idx).count() as f32;
                
            let predicted_positives = pred_classes.iter().filter(|&&p| p == idx).count() as f32;
            
            return if predicted_positives > 0.0 {
                Ok(true_positives / predicted_positives)
            } else {
                Ok(0.0)
            };
        }
        
        // Tüm sınıflar için hassasiyet hesaplama
        for i in 0..n_classes {
            let true_positives = true_classes.iter().zip(pred_classes.iter())
                .filter(|&(t, p)| *t == i && *p == i).count() as f32;
                
            let predicted_positives = pred_classes.iter().filter(|&&p| p == i).count() as f32;
            
            let precision = if predicted_positives > 0.0 {
                true_positives / predicted_positives
            } else {
                0.0
            };
            
            class_precision.push(precision);
        }
        
        // Ortalama hesaplama
        match self.average.as_str() {
            "micro" => {
                let total_true_positives: f32 = (0..n_classes).map(|i| {
                    true_classes.iter().zip(pred_classes.iter())
                        .filter(|&(t, p)| *t == i && *p == i).count() as f32
                }).sum();
                
                let total_predicted_positives: f32 = pred_classes.len() as f32;
                
                if total_predicted_positives > 0.0 {
                    Ok(total_true_positives / total_predicted_positives)
                } else {
                    Ok(0.0)
                }
            },
            "macro" => {
                let avg_precision: f32 = class_precision.iter().sum::<f32>() / n_classes as f32;
                Ok(avg_precision)
            },
            "weighted" => {
                let class_counts: Vec<f32> = (0..n_classes).map(|i| {
                    true_classes.iter().filter(|&&t| t == i).count() as f32
                }).collect();
                
                let total_samples: f32 = class_counts.iter().sum();
                
                let weighted_precision: f32 = class_precision.iter().enumerate()
                    .map(|(i, &p)| p * (class_counts[i] / total_samples))
                    .sum();
                    
                Ok(weighted_precision)
            },
            _ => Err(TurkError::InvalidInput(format!(
                "Geçersiz ortalama tipi: {}. 'micro', 'macro' veya 'weighted' olmalı.", self.average
            )))
        }
    }
    
    fn name(&self) -> &'static str {
        "precision"
    }
}

/// Duyarlılık (Recall) ölçümü
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recall {
    pub average: String, // 'micro', 'macro', 'weighted', veya 'None' (belirli bir sınıf için)
    pub class_idx: Option<usize>, // 'None' olmadığında, belirli bir sınıfın indeksi
}

impl Default for Recall {
    fn default() -> Self {
        Self {
            average: "macro".to_string(),
            class_idx: None,
        }
    }
}

impl Metric for Recall {
    fn compute(&self, y_true: &Array2<f32>, y_pred: &Array2<f32>) -> Result<f32> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Gerçek ve tahmin edilen çıktıların şekilleri uyuşmuyor: {:?} vs {:?}",
                y_true.shape(), y_pred.shape()
            )));
        }
        
        let n_classes = y_true.shape()[1];
        let mut class_recall = Vec::new();
        
        let true_classes: Vec<usize> = y_true
            .axis_iter(Axis(1))
            .map(|row| row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0)
            .collect();
            
        let pred_classes: Vec<usize> = y_pred
            .axis_iter(Axis(1))
            .map(|row| row.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0)
            .collect();
            
        // Belirli bir sınıf için duyarlılık hesaplama
        if let Some(idx) = self.class_idx {
            if idx >= n_classes {
                return Err(TurkError::InvalidInput(format!(
                    "Geçersiz sınıf indeksi: {}. Toplam sınıf sayısı: {}", idx, n_classes
                )));
            }
            
            let true_positives = true_classes.iter().zip(pred_classes.iter())
                .filter(|&(t, p)| *t == idx && *p == idx).count() as f32;
                
            let actual_positives = true_classes.iter().filter(|&&t| t == idx).count() as f32;
            
            return if actual_positives > 0.0 {
                Ok(true_positives / actual_positives)
            } else {
                Ok(0.0)
            };
        }
        
        // Tüm sınıflar için duyarlılık hesaplama
        for i in 0..n_classes {
            let true_positives = true_classes.iter().zip(pred_classes.iter())
                .filter(|&(t, p)| *t == i && *p == i).count() as f32;
                
            let actual_positives = true_classes.iter().filter(|&&t| t == i).count() as f32;
            
            let recall = if actual_positives > 0.0 {
                true_positives / actual_positives
            } else {
                0.0
            };
            
            class_recall.push(recall);
        }
        
        // Ortalama hesaplama
        match self.average.as_str() {
            "micro" => {
                let total_true_positives: f32 = (0..n_classes).map(|i| {
                    true_classes.iter().zip(pred_classes.iter())
                        .filter(|&(t, p)| *t == i && *p == i).count() as f32
                }).sum();
                
                let total_actual_positives: f32 = true_classes.len() as f32;
                
                if total_actual_positives > 0.0 {
                    Ok(total_true_positives / total_actual_positives)
                } else {
                    Ok(0.0)
                }
            },
            "macro" => {
                let avg_recall: f32 = class_recall.iter().sum::<f32>() / n_classes as f32;
                Ok(avg_recall)
            },
            "weighted" => {
                let class_counts: Vec<f32> = (0..n_classes).map(|i| {
                    true_classes.iter().filter(|&&t| t == i).count() as f32
                }).collect();
                
                let total_samples: f32 = class_counts.iter().sum();
                
                let weighted_recall: f32 = class_recall.iter().enumerate()
                    .map(|(i, &r)| r * (class_counts[i] / total_samples))
                    .sum();
                    
                Ok(weighted_recall)
            },
            _ => Err(TurkError::InvalidInput(format!(
                "Geçersiz ortalama tipi: {}. 'micro', 'macro' veya 'weighted' olmalı.", self.average
            )))
        }
    }
    
    fn name(&self) -> &'static str {
        "recall"
    }
}

/// F1 Skoru ölçümü
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F1Score {
    pub average: String, // 'micro', 'macro', 'weighted', veya 'None' (belirli bir sınıf için)
    pub class_idx: Option<usize>, // 'None' olmadığında, belirli bir sınıfın indeksi
}

impl Default for F1Score {
    fn default() -> Self {
        Self {
            average: "macro".to_string(),
            class_idx: None,
        }
    }
}

impl Metric for F1Score {
    fn compute(&self, y_true: &Array2<f32>, y_pred: &Array2<f32>) -> Result<f32> {
        let precision = Precision {
            average: self.average.clone(),
            class_idx: self.class_idx,
        };
        
        let recall = Recall {
            average: self.average.clone(),
            class_idx: self.class_idx,
        };
        
        let p = precision.compute(y_true, y_pred)?;
        let r = recall.compute(y_true, y_pred)?;
        
        if p + r > 0.0 {
            Ok(2.0 * p * r / (p + r))
        } else {
            Ok(0.0)
        }
    }
    
    fn name(&self) -> &'static str {
        "f1_score"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_accuracy_perfect() {
        let y_true = Array::from_shape_vec((3, 3), vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]).unwrap();
        
        let y_pred = Array::from_shape_vec((3, 3), vec![
            0.9, 0.05, 0.05,
            0.05, 0.9, 0.05,
            0.05, 0.05, 0.9,
        ]).unwrap();
        
        let metric = Accuracy;
        let result = metric.compute(&y_true, &y_pred).unwrap();
        assert_eq!(result, 1.0);
    }
    
    #[test]
    fn test_accuracy_imperfect() {
        let y_true = Array::from_shape_vec((3, 3), vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]).unwrap();
        
        let y_pred = Array::from_shape_vec((3, 3), vec![
            0.9, 0.05, 0.05,
            0.7, 0.2, 0.1,
            0.05, 0.05, 0.9,
        ]).unwrap();
        
        let metric = Accuracy;
        let result = metric.compute(&y_true, &y_pred).unwrap();
        assert_eq!(result, 2.0 / 3.0);
    }
    
    #[test]
    fn test_precision_recall_f1() {
        let y_true = Array::from_shape_vec((5, 3), vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ]).unwrap();
        
        let y_pred = Array::from_shape_vec((5, 3), vec![
            0.9, 0.05, 0.05,
            0.7, 0.2, 0.1,
            0.05, 0.05, 0.9,
            0.6, 0.3, 0.1,
            0.1, 0.6, 0.3,
        ]).unwrap();
        
        let precision = Precision::default();
        let recall = Recall::default();
        let f1 = F1Score::default();
        
        let p = precision.compute(&y_true, &y_pred).unwrap();
        let r = recall.compute(&y_true, &y_pred).unwrap();
        let f = f1.compute(&y_true, &y_pred).unwrap();
        
        assert!(p >= 0.0 && p <= 1.0);
        assert!(r >= 0.0 && r <= 1.0);
        assert!(f >= 0.0 && f <= 1.0);
    }
} 
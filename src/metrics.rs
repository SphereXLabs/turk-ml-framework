use ndarray::Array2;
use crate::error::{Result, TurkError};

/// Metrik trait'i - tüm metrikler bu trait'i implemente etmelidir
pub trait Metric: Send + Sync {
    /// Metrik adını döndürür
    fn name(&self) -> &str;
    
    /// Metrik değerini hesaplar
    fn compute(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> Result<f32>;
}

/// Doğruluk (Accuracy) metriği
#[derive(Debug, Default)]
pub struct Accuracy;

impl Accuracy {
    /// Yeni bir doğruluk metriği oluşturur
    pub fn new() -> Self {
        Self
    }
}

impl Metric for Accuracy {
    fn name(&self) -> &str {
        "accuracy"
    }
    
    fn compute(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> Result<f32> {
        if y_pred.shape() != y_true.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Doğruluk hesaplaması için şekiller uyuşmuyor: {:?} != {:?}",
                y_pred.shape(),
                y_true.shape()
            )));
        }
        
        let mut correct = 0;
        let mut total = 0;
        
        for ((_pred, _true_), (pred_row, true_row)) in y_pred.iter().zip(y_true.iter())
            .zip(y_pred.rows().into_iter().zip(y_true.rows().into_iter())) {
            let pred_class = pred_row.iter()
                .position(|&x| x == *pred_row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
                .unwrap();
            let true_class = true_row.iter()
                .position(|&x| x == *true_row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
                .unwrap();
            
            if pred_class == true_class {
                correct += 1;
            }
            total += 1;
        }
        
        Ok(correct as f32 / total as f32)
    }
}

/// Hassasiyet (Precision) metriği
#[derive(Debug, Default)]
pub struct Precision {
    threshold: f32,
}

impl Precision {
    /// Yeni bir hassasiyet metriği oluşturur
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Metric for Precision {
    fn name(&self) -> &str {
        "precision"
    }
    
    fn compute(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> Result<f32> {
        if y_pred.shape() != y_true.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Hassasiyet hesaplaması için şekiller uyuşmuyor: {:?} != {:?}",
                y_pred.shape(),
                y_true.shape()
            )));
        }
        
        let mut true_positives = 0;
        let mut false_positives = 0;
        
        for (pred, true_) in y_pred.iter().zip(y_true.iter()) {
            if *pred >= self.threshold {
                if *true_ >= self.threshold {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
            }
        }
        
        let precision = if true_positives + false_positives > 0 {
            true_positives as f32 / (true_positives + false_positives) as f32
        } else {
            0.0
        };
        
        Ok(precision)
    }
}

/// Duyarlılık (Recall) metriği
#[derive(Debug, Default)]
pub struct Recall {
    threshold: f32,
}

impl Recall {
    /// Yeni bir duyarlılık metriği oluşturur
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Metric for Recall {
    fn name(&self) -> &str {
        "recall"
    }
    
    fn compute(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> Result<f32> {
        if y_pred.shape() != y_true.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Duyarlılık hesaplaması için şekiller uyuşmuyor: {:?} != {:?}",
                y_pred.shape(),
                y_true.shape()
            )));
        }
        
        let mut true_positives = 0;
        let mut false_negatives = 0;
        
        for (pred, true_) in y_pred.iter().zip(y_true.iter()) {
            if *true_ >= self.threshold {
                if *pred >= self.threshold {
                    true_positives += 1;
                } else {
                    false_negatives += 1;
                }
            }
        }
        
        let recall = if true_positives + false_negatives > 0 {
            true_positives as f32 / (true_positives + false_negatives) as f32
        } else {
            0.0
        };
        
        Ok(recall)
    }
}

/// F1-Score metriği
#[derive(Debug, Default)]
pub struct F1Score {
    threshold: f32,
}

impl F1Score {
    /// Yeni bir F1-Score metriği oluşturur
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl Metric for F1Score {
    fn name(&self) -> &str {
        "f1_score"
    }
    
    fn compute(&self, y_pred: &Array2<f32>, y_true: &Array2<f32>) -> Result<f32> {
        let precision = Precision::new(self.threshold).compute(y_pred, y_true)?;
        let recall = Recall::new(self.threshold).compute(y_pred, y_true)?;
        
        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };
        
        Ok(f1_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_accuracy() {
        let y_pred = arr2(&[
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ]);
        
        let y_true = arr2(&[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        
        let accuracy = Accuracy::new();
        let result = accuracy.compute(&y_pred, &y_true).unwrap();
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_precision() {
        let y_pred = arr2(&[
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6],
        ]);
        
        let y_true = arr2(&[
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]);
        
        let precision = Precision::new(0.5);
        let result = precision.compute(&y_pred, &y_true).unwrap();
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_recall() {
        let y_pred = arr2(&[
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6],
        ]);
        
        let y_true = arr2(&[
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]);
        
        let recall = Recall::new(0.5);
        let result = recall.compute(&y_pred, &y_true).unwrap();
        assert_eq!(result, 0.5);
    }

    #[test]
    fn test_f1_score() {
        let y_pred = arr2(&[
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6],
        ]);
        
        let y_true = arr2(&[
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]);
        
        let f1_score = F1Score::new(0.5);
        let result = f1_score.compute(&y_pred, &y_true).unwrap();
        assert_eq!(result, 0.5);
    }
} 
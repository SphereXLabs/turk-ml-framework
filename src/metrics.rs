use crate::error::{Result, TurkError};
use ndarray::ArrayView2;
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};

/// Metrik trait'i - object safety için trait bound'lar kaldırıldı
use crate::dtype::Float;
pub trait Metric: Send + Sync {
    /// Metrik hesapla
    fn compute(&self, y_true: &ArrayView2<Float>, y_pred: &ArrayView2<Float>) -> Float;

    /// Metriğin adını döndürür
    fn name(&self) -> &'static str;

    /// Metriği klonlayarak yeni bir Box olarak döndürür
    fn box_clone(&self) -> Box<dyn Metric>;
}

/// Metric nesneleri için Clone uygulaması
impl Clone for Box<dyn Metric> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// Accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Accuracy;

impl Accuracy {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for Accuracy {
    fn compute(&self, y_true: &ArrayView2<Float>, y_pred: &ArrayView2<Float>) -> Float {
        let n = y_true.nrows();
        if n == 0 {
            return 0.0;
        }

        let mut correct = 0;
        // Tek çıkışlı (XOR / binary MSE+sigmoid): argmax her zaman 0 döner —
        // 0.5 eşiği ile sınıf karşılaştır.
        if y_pred.ncols() == 1 {
            for i in 0..n {
                let t = y_true[[i, 0]] >= 0.5;
                let p = y_pred[[i, 0]] >= 0.5;
                if t == p {
                    correct += 1;
                }
            }
        } else {
            for i in 0..n {
                let true_class = if y_true.ncols() == 1 {
                    y_true[[i, 0]] as usize
                } else {
                    y_true.row(i).argmax().unwrap_or(0)
                };
                let pred_class = y_pred.row(i).argmax().unwrap_or(0);
                if true_class == pred_class {
                    correct += 1;
                }
            }
        }
        correct as Float / n as Float
    }

    fn name(&self) -> &'static str {
        "accuracy"
    }

    fn box_clone(&self) -> Box<dyn Metric> {
        Box::new(self.clone())
    }
}

/// Precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Precision;

impl Precision {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for Precision {
    fn compute(&self, y_true: &ArrayView2<Float>, y_pred: &ArrayView2<Float>) -> Float {
        let mut true_positives = 0;
        let mut false_positives = 0;
        for i in 0..y_true.nrows() {
            let true_class = y_true[[i, 0]] as usize;
            let pred_class = y_pred.row(i).argmax().unwrap();
            if true_class == pred_class {
                true_positives += 1;
            } else {
                false_positives += 1;
            }
        }
        if true_positives + false_positives == 0 {
            return 0.0;
        }
        true_positives as Float / (true_positives + false_positives) as Float
    }

    fn name(&self) -> &'static str {
        "precision"
    }

    fn box_clone(&self) -> Box<dyn Metric> {
        Box::new(self.clone())
    }
}

/// Recall
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recall;

impl Recall {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for Recall {
    fn compute(&self, y_true: &ArrayView2<Float>, y_pred: &ArrayView2<Float>) -> Float {
        let mut true_positives = 0;
        let mut false_negatives = 0;
        for i in 0..y_true.nrows() {
            let true_class = y_true[[i, 0]] as usize;
            let pred_class = y_pred.row(i).argmax().unwrap();
            if true_class == pred_class {
                true_positives += 1;
            } else if true_class != pred_class {
                false_negatives += 1;
            }
        }
        if true_positives + false_negatives == 0 {
            return 0.0;
        }
        true_positives as Float / (true_positives + false_negatives) as Float
    }

    fn name(&self) -> &'static str {
        "recall"
    }

    fn box_clone(&self) -> Box<dyn Metric> {
        Box::new(self.clone())
    }
}

/// F1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F1;

impl F1 {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for F1 {
    fn compute(&self, y_true: &ArrayView2<Float>, y_pred: &ArrayView2<Float>) -> Float {
        let precision = Precision.compute(y_true, y_pred);
        let recall = Recall.compute(y_true, y_pred);
        if precision + recall == 0.0 {
            return 0.0;
        }
        2.0 * precision * recall / (precision + recall)
    }

    fn name(&self) -> &'static str {
        "f1"
    }

    fn box_clone(&self) -> Box<dyn Metric> {
        Box::new(self.clone())
    }
}

/// MAE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MAE;

impl MAE {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for MAE {
    fn compute(&self, y_true: &ArrayView2<Float>, y_pred: &ArrayView2<Float>) -> Float {
        let diff = y_true - y_pred;
        diff.mapv(|v| v.abs()).mean().unwrap_or(0.0)
    }

    fn name(&self) -> &'static str {
        "mae"
    }

    fn box_clone(&self) -> Box<dyn Metric> {
        Box::new(self.clone())
    }
}

/// MSE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MSE;

impl MSE {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for MSE {
    fn compute(&self, y_true: &ArrayView2<Float>, y_pred: &ArrayView2<Float>) -> Float {
        let diff = y_true - y_pred;
        let squared = &diff * &diff;
        squared.mean().unwrap_or(0.0)
    }

    fn name(&self) -> &'static str {
        "mse"
    }

    fn box_clone(&self) -> Box<dyn Metric> {
        Box::new(self.clone())
    }
}

/// RMSE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSE;

impl RMSE {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for RMSE {
    fn compute(&self, y_true: &ArrayView2<Float>, y_pred: &ArrayView2<Float>) -> Float {
        let mse = MSE.compute(y_true, y_pred);
        mse.sqrt()
    }

    fn name(&self) -> &'static str {
        "rmse"
    }

    fn box_clone(&self) -> Box<dyn Metric> {
        Box::new(self.clone())
    }
}

/// R2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R2;

impl R2 {
    pub fn new() -> Self {
        Self
    }
}

impl Metric for R2 {
    fn compute(&self, y_true: &ArrayView2<Float>, y_pred: &ArrayView2<Float>) -> Float {
        let diff = y_true - y_pred;
        let ss_res = (&diff * &diff).sum();
        let y_mean = y_true.mean().unwrap_or(0.0);
        let diff_mean = y_true.mapv(|v| v - y_mean);
        let ss_tot = (&diff_mean * &diff_mean).sum();
        if ss_tot == 0.0 {
            return 0.0;
        }
        1.0 - ss_res / ss_tot
    }

    fn name(&self) -> &'static str {
        "r2"
    }

    fn box_clone(&self) -> Box<dyn Metric> {
        Box::new(self.clone())
    }
}

/// Metrik oluştur
pub fn create_metric(metric: &str) -> Result<Box<dyn Metric>> {
    match metric {
        "accuracy" => Ok(Box::new(Accuracy::new())),
        "precision" => Ok(Box::new(Precision::new())),
        "recall" => Ok(Box::new(Recall::new())),
        "f1" => Ok(Box::new(F1::new())),
        "mae" => Ok(Box::new(MAE::new())),
        "mse" => Ok(Box::new(MSE::new())),
        "rmse" => Ok(Box::new(RMSE::new())),
        "r2" => Ok(Box::new(R2::new())),
        _ => Err(TurkError::model_error(&format!(
            "Geçersiz metrik: {}",
            metric
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_binary_accuracy_threshold() {
        // XOR benzeri tek çıkış: argmax her zaman 0 olurdu; 0.5 eşiği gerekir.
        let y_true = array![[0.0], [1.0], [1.0], [0.0]];
        let y_pred = array![[0.01], [0.99], [0.97], [0.02]];
        let acc = Accuracy::new().compute(&y_true.view(), &y_pred.view());
        assert!((acc - 1.0).abs() < 1e-6, "beklenen 100%, gelen {acc}");
    }
}

use crate::{Result, TinyMLError};
use ndarray::Array2;

pub trait Loss {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32>;
    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>>;
}

#[derive(Debug, Clone)]
pub struct CrossEntropy;

impl CrossEntropy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for CrossEntropy {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(TinyMLError::ShapeMismatch(format!(
                "Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let epsilon = 1e-7;
        let clipped_predictions = predictions.map(|&x| x.max(epsilon).min(1.0 - epsilon));
        
        let loss = -targets.iter()
            .zip(clipped_predictions.iter())
            .map(|(&t, &p)| t * p.ln())
            .sum::<f32>();

        Ok(loss / predictions.len() as f32)
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        if predictions.shape() != targets.shape() {
            return Err(TinyMLError::ShapeMismatch(format!(
                "Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let epsilon = 1e-7;
        let clipped_predictions = predictions.map(|&x| x.max(epsilon).min(1.0 - epsilon));
        
        Ok((&clipped_predictions - targets) / predictions.len() as f32)
    }
}

#[derive(Debug, Clone)]
pub struct MSE;

impl Default for MSE {
    fn default() -> Self {
        Self
    }
}

impl Loss for MSE {
    fn compute(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(TinyMLError::ShapeMismatch(format!(
                "Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        let diff = predictions - targets;
        let squared_diff = diff.map(|&x| x * x);
        Ok(squared_diff.mean().unwrap())
    }

    fn gradient(&self, predictions: &Array2<f32>, targets: &Array2<f32>) -> Result<Array2<f32>> {
        if predictions.shape() != targets.shape() {
            return Err(TinyMLError::ShapeMismatch(format!(
                "Tahmin şekli: {:?}, Hedef şekli: {:?}",
                predictions.shape(),
                targets.shape()
            )));
        }

        Ok(2.0 * (predictions - targets) / predictions.len() as f32)
    }
} 
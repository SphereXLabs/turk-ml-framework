use ndarray::{Array2, Axis};
use crate::error::{Result, TurkError};
use serde::{Serialize, Deserialize};
use std::ops::Range;

/// Veri önişleme trait'i
pub trait Preprocessor: Send + Sync {
    fn fit(&mut self, data: &Array2<f32>) -> Result<()>;
    fn transform(&self, data: &Array2<f32>) -> Result<Array2<f32>>;
    fn fit_transform(&mut self, data: &Array2<f32>) -> Result<Array2<f32>> {
        self.fit(data)?;
        self.transform(data)
    }
    fn inverse_transform(&self, data: &Array2<f32>) -> Result<Array2<f32>>;
    fn name(&self) -> &'static str;
}

/// Standart Ölçeklendirici (StandardScaler)
/// Veriyi ortalama 0, standart sapma 1 olacak şekilde dönüştürür
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    pub means: Option<Vec<f32>>,
    pub std_devs: Option<Vec<f32>>,
    pub epsilon: f32,
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self {
            means: None,
            std_devs: None,
            epsilon: 1e-10,
        }
    }
}

impl Preprocessor for StandardScaler {
    fn fit(&mut self, data: &Array2<f32>) -> Result<()> {
        let n_features = data.shape()[1];
        let mut means = vec![0.0; n_features];
        let mut std_devs = vec![0.0; n_features];
        
        for (i, feature) in data.axis_iter(Axis(1)).enumerate() {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let n_samples = feature.len() as f32;
            
            for &value in feature.iter() {
                sum += value;
                sum_sq += value * value;
            }
            
            let mean = sum / n_samples;
            let variance = (sum_sq / n_samples) - (mean * mean);
            let std_dev = variance.sqrt();
            
            means[i] = mean;
            std_devs[i] = if std_dev < self.epsilon { 1.0 } else { std_dev };
        }
        
        self.means = Some(means);
        self.std_devs = Some(std_devs);
        
        Ok(())
    }
    
    fn transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let means = self.means.as_ref().ok_or_else(|| {
            TurkError::InvalidInput("StandardScaler önce fit() ile eğitilmelidir".to_string())
        })?;
        
        let std_devs = self.std_devs.as_ref().ok_or_else(|| {
            TurkError::InvalidInput("StandardScaler önce fit() ile eğitilmelidir".to_string())
        })?;
        
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        
        if n_features != means.len() {
            return Err(TurkError::ShapeMismatch(format!(
                "Veri şekli uyuşmuyor: veri {} özellik içeriyor, ancak ölçekleyici {} özellik için eğitilmiş",
                n_features, means.len()
            )));
        }
        
        let mut result = Array2::zeros((n_samples, n_features));
        
        for i in 0..n_samples {
            for j in 0..n_features {
                result[[i, j]] = (data[[i, j]] - means[j]) / std_devs[j];
            }
        }
        
        Ok(result)
    }
    
    fn inverse_transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let means = self.means.as_ref().ok_or_else(|| {
            TurkError::InvalidInput("StandardScaler önce fit() ile eğitilmelidir".to_string())
        })?;
        
        let std_devs = self.std_devs.as_ref().ok_or_else(|| {
            TurkError::InvalidInput("StandardScaler önce fit() ile eğitilmelidir".to_string())
        })?;
        
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        
        if n_features != means.len() {
            return Err(TurkError::ShapeMismatch(format!(
                "Veri şekli uyuşmuyor: veri {} özellik içeriyor, ancak ölçekleyici {} özellik için eğitilmiş",
                n_features, means.len()
            )));
        }
        
        let mut result = Array2::zeros((n_samples, n_features));
        
        for i in 0..n_samples {
            for j in 0..n_features {
                result[[i, j]] = data[[i, j]] * std_devs[j] + means[j];
            }
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &'static str {
        "standard_scaler"
    }
}

/// Min-Max Ölçeklendirici (MinMaxScaler)
/// Veriyi belirtilen min-max aralığına ölçeklendirir (varsayılan: 0-1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScaler {
    pub min_vals: Option<Vec<f32>>,
    pub max_vals: Option<Vec<f32>>,
    pub feature_range: Range<f32>,
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self {
            min_vals: None,
            max_vals: None,
            feature_range: 0.0..1.0,
        }
    }
}

impl MinMaxScaler {
    pub fn new(feature_range: Range<f32>) -> Self {
        Self {
            min_vals: None,
            max_vals: None,
            feature_range,
        }
    }
}

impl Preprocessor for MinMaxScaler {
    fn fit(&mut self, data: &Array2<f32>) -> Result<()> {
        let n_features = data.shape()[1];
        let mut min_vals = vec![f32::MAX; n_features];
        let mut max_vals = vec![f32::MIN; n_features];
        
        for (i, feature) in data.axis_iter(Axis(1)).enumerate() {
            for &value in feature.iter() {
                if value < min_vals[i] {
                    min_vals[i] = value;
                }
                if value > max_vals[i] {
                    max_vals[i] = value;
                }
            }
        }
        
        self.min_vals = Some(min_vals);
        self.max_vals = Some(max_vals);
        
        Ok(())
    }
    
    fn transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let min_vals = self.min_vals.as_ref().ok_or_else(|| {
            TurkError::InvalidInput("MinMaxScaler önce fit() ile eğitilmelidir".to_string())
        })?;
        
        let max_vals = self.max_vals.as_ref().ok_or_else(|| {
            TurkError::InvalidInput("MinMaxScaler önce fit() ile eğitilmelidir".to_string())
        })?;
        
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        
        if n_features != min_vals.len() {
            return Err(TurkError::ShapeMismatch(format!(
                "Veri şekli uyuşmuyor: veri {} özellik içeriyor, ancak ölçekleyici {} özellik için eğitilmiş",
                n_features, min_vals.len()
            )));
        }
        
        let mut result = Array2::zeros((n_samples, n_features));
        let scale = self.feature_range.end - self.feature_range.start;
        
        for i in 0..n_samples {
            for j in 0..n_features {
                let range = max_vals[j] - min_vals[j];
                if range.abs() < 1e-10 {
                    result[[i, j]] = self.feature_range.start;
                } else {
                    let scaled_value = (data[[i, j]] - min_vals[j]) / range;
                    result[[i, j]] = scaled_value * scale + self.feature_range.start;
                }
            }
        }
        
        Ok(result)
    }
    
    fn inverse_transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let min_vals = self.min_vals.as_ref().ok_or_else(|| {
            TurkError::InvalidInput("MinMaxScaler önce fit() ile eğitilmelidir".to_string())
        })?;
        
        let max_vals = self.max_vals.as_ref().ok_or_else(|| {
            TurkError::InvalidInput("MinMaxScaler önce fit() ile eğitilmelidir".to_string())
        })?;
        
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        
        if n_features != min_vals.len() {
            return Err(TurkError::ShapeMismatch(format!(
                "Veri şekli uyuşmuyor: veri {} özellik içeriyor, ancak ölçekleyici {} özellik için eğitilmiş",
                n_features, min_vals.len()
            )));
        }
        
        let mut result = Array2::zeros((n_samples, n_features));
        let scale = self.feature_range.end - self.feature_range.start;
        
        for i in 0..n_samples {
            for j in 0..n_features {
                let orig_value = (data[[i, j]] - self.feature_range.start) / scale;
                let range = max_vals[j] - min_vals[j];
                result[[i, j]] = orig_value * range + min_vals[j];
            }
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &'static str {
        "min_max_scaler"
    }
}

/// Normalizer - Veriyi satır bazında normalize eder
/// L1, L2 veya max norm kullanılabilir
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalizer {
    pub norm: String, // "l1", "l2", veya "max"
}

impl Default for Normalizer {
    fn default() -> Self {
        Self {
            norm: "l2".to_string(),
        }
    }
}

impl Normalizer {
    pub fn new(norm: &str) -> Result<Self> {
        match norm {
            "l1" | "l2" | "max" => Ok(Self { norm: norm.to_string() }),
            _ => Err(TurkError::InvalidInput(format!(
                "Geçersiz norm tipi: {}. 'l1', 'l2' veya 'max' olmalı.", norm
            ))),
        }
    }
}

impl Preprocessor for Normalizer {
    fn fit(&mut self, _data: &Array2<f32>) -> Result<()> {
        // Normalizer eğitim gerektirmez, her örneği bağımsız olarak işler
        Ok(())
    }
    
    fn transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];
        let mut result = data.clone();
        
        for i in 0..n_samples {
            let mut norm_value = 0.0;
            
            match self.norm.as_str() {
                "l1" => {
                    // L1 norm: sum(|x_i|)
                    for j in 0..n_features {
                        norm_value += data[[i, j]].abs();
                    }
                },
                "l2" => {
                    // L2 norm: sqrt(sum(x_i^2))
                    for j in 0..n_features {
                        norm_value += data[[i, j]] * data[[i, j]];
                    }
                    norm_value = norm_value.sqrt();
                },
                "max" => {
                    // Max norm: max(|x_i|)
                    for j in 0..n_features {
                        let abs_val = data[[i, j]].abs();
                        if abs_val > norm_value {
                            norm_value = abs_val;
                        }
                    }
                },
                _ => {
                    return Err(TurkError::InvalidInput(format!(
                        "Geçersiz norm tipi: {}. 'l1', 'l2' veya 'max' olmalı.", self.norm
                    )));
                }
            }
            
            if norm_value > 1e-10 {
                for j in 0..n_features {
                    result[[i, j]] = data[[i, j]] / norm_value;
                }
            }
        }
        
        Ok(result)
    }
    
    fn inverse_transform(&self, _data: &Array2<f32>) -> Result<Array2<f32>> {
        Err(TurkError::InvalidInput(
            "Normalizer ters dönüşüm desteklemez çünkü işlem tersinir değildir".to_string()
        ))
    }
    
    fn name(&self) -> &'static str {
        "normalizer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    
    #[test]
    fn test_standard_scaler() {
        let data = Array::from_shape_vec((3, 2), vec![
            0.0, 0.0,
            1.0, 1.0,
            2.0, 2.0,
        ]).unwrap();
        
        let mut scaler = StandardScaler::default();
        let transformed = scaler.fit_transform(&data).unwrap();
        
        // Dönüştürülmüş veri ortalamasının yaklaşık olarak 0 olduğunu kontrol et
        let col_means = transformed.mean_axis(Axis(0)).unwrap();
        assert!(col_means.iter().all(|&x| x.abs() < 1e-10));
        
        // Dönüştürülmüş veri standart sapmasının yaklaşık olarak 1 olduğunu kontrol et
        let var = transformed.map_axis(Axis(0), |view| {
            let mean = view.mean().unwrap();
            view.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / view.len() as f32
        });
        assert!(var.iter().all(|&x| (x - 1.0).abs() < 1e-10));
        
        // Ters dönüşümü kontrol et
        let inverse = scaler.inverse_transform(&transformed).unwrap();
        assert!(data.iter().zip(inverse.iter()).all(|(&a, &b)| (a - b).abs() < 1e-5));
    }
    
    #[test]
    fn test_min_max_scaler() {
        let data = Array::from_shape_vec((3, 2), vec![
            0.0, 10.0,
            5.0, 15.0,
            10.0, 20.0,
        ]).unwrap();
        
        let mut scaler = MinMaxScaler::new(0.0..1.0);
        let transformed = scaler.fit_transform(&data).unwrap();
        
        // İlk sütunun 0-1 aralığında olduğunu kontrol et
        assert_eq!(transformed[[0, 0]], 0.0);
        assert_eq!(transformed[[1, 0]], 0.5);
        assert_eq!(transformed[[2, 0]], 1.0);
        
        // İkinci sütunun 0-1 aralığında olduğunu kontrol et
        assert_eq!(transformed[[0, 1]], 0.0);
        assert_eq!(transformed[[1, 1]], 0.5);
        assert_eq!(transformed[[2, 1]], 1.0);
        
        // Farklı bir aralık kullanarak kontrol et
        let mut scaled_range = MinMaxScaler::new(-1.0..1.0);
        let transformed_range = scaled_range.fit_transform(&data).unwrap();
        
        assert_eq!(transformed_range[[0, 0]], -1.0);
        assert_eq!(transformed_range[[1, 0]], 0.0);
        assert_eq!(transformed_range[[2, 0]], 1.0);
    }
    
    #[test]
    fn test_normalizer() {
        let data = Array::from_shape_vec((2, 3), vec![
            3.0, 4.0, 0.0,
            1.0, 2.0, 2.0,
        ]).unwrap();
        
        // L2 norm test
        let normalizer_l2 = Normalizer::new("l2").unwrap();
        let transformed_l2 = normalizer_l2.transform(&data).unwrap();
        
        // İlk satırın L2 norm'u 5.0, ikinci satırın 3.0
        assert!((transformed_l2[[0, 0]] - 0.6).abs() < 1e-5); // 3/5
        assert!((transformed_l2[[0, 1]] - 0.8).abs() < 1e-5); // 4/5
        assert_eq!(transformed_l2[[0, 2]], 0.0);
        
        // L1 norm test
        let normalizer_l1 = Normalizer::new("l1").unwrap();
        let transformed_l1 = normalizer_l1.transform(&data).unwrap();
        
        // İlk satırın L1 norm'u 7.0, ikinci satırın 5.0
        assert!((transformed_l1[[0, 0]] - 3.0/7.0).abs() < 1e-5);
        assert!((transformed_l1[[0, 1]] - 4.0/7.0).abs() < 1e-5);
        assert_eq!(transformed_l1[[0, 2]], 0.0);
        
        // Max norm test
        let normalizer_max = Normalizer::new("max").unwrap();
        let transformed_max = normalizer_max.transform(&data).unwrap();
        
        // İlk satırın max değeri 4.0, ikinci satırın 2.0
        assert!((transformed_max[[0, 0]] - 0.75).abs() < 1e-5); // 3/4
        assert_eq!(transformed_max[[0, 1]], 1.0); // 4/4
        assert_eq!(transformed_max[[0, 2]], 0.0);
    }
} 
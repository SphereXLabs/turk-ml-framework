use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use serde::{Serialize, Deserialize};
use crate::error::{Result, TurkError};

/// Ağırlık başlangıç değerleri için trait
pub trait Initializer: Send + Sync + std::fmt::Debug {
    fn initialize(&self, shape: &[usize]) -> Result<Array2<f32>>;
    fn name(&self) -> &'static str;
}

/// Xavier/Glorot başlangıç değerleri
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Xavier {
    pub gain: f32,
}

impl Default for Xavier {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl Initializer for Xavier {
    fn initialize(&self, shape: &[usize]) -> Result<Array2<f32>> {
        if shape.len() != 2 {
            return Err(TurkError::InvalidInput(format!(
                "Xavier başlangıç değerleri için 2 boyutlu şekil gerekli, alınan: {:?}", shape
            )));
        }
        
        let fan_in = shape[0] as f32;
        let fan_out = shape[1] as f32;
        let std_dev = self.gain * (2.0 / (fan_in + fan_out)).sqrt();
        
        let dist = Normal::new(0.0, std_dev)
            .map_err(|e| TurkError::InvalidInput(format!("Normal dağılım oluşturulurken hata: {}", e)))?;
        
        Ok(Array::random((shape[0], shape[1]), dist))
    }
    
    fn name(&self) -> &'static str {
        "xavier"
    }
}

/// He başlangıç değerleri (Kaiming He)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeNormal {
    pub gain: f32,
}

impl Default for HeNormal {
    fn default() -> Self {
        Self { gain: 2.0 }
    }
}

impl Initializer for HeNormal {
    fn initialize(&self, shape: &[usize]) -> Result<Array2<f32>> {
        if shape.len() != 2 {
            return Err(TurkError::InvalidInput(format!(
                "He başlangıç değerleri için 2 boyutlu şekil gerekli, alınan: {:?}", shape
            )));
        }
        
        let fan_in = shape[0] as f32;
        let std_dev = self.gain * (1.0 / fan_in).sqrt();
        
        let dist = Normal::new(0.0, std_dev)
            .map_err(|e| TurkError::InvalidInput(format!("Normal dağılım oluşturulurken hata: {}", e)))?;
        
        Ok(Array::random((shape[0], shape[1]), dist))
    }
    
    fn name(&self) -> &'static str {
        "he_normal"
    }
}

/// Sıfır başlangıç değerleri
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Zeros;

impl Initializer for Zeros {
    fn initialize(&self, shape: &[usize]) -> Result<Array2<f32>> {
        if shape.len() != 2 {
            return Err(TurkError::InvalidInput(format!(
                "Zeros başlangıç değerleri için 2 boyutlu şekil gerekli, alınan: {:?}", shape
            )));
        }
        
        Ok(Array2::<f32>::zeros((shape[0], shape[1])))
    }
    
    fn name(&self) -> &'static str {
        "zeros"
    }
}

/// Bir başlangıç değerleri
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ones;

impl Initializer for Ones {
    fn initialize(&self, shape: &[usize]) -> Result<Array2<f32>> {
        if shape.len() != 2 {
            return Err(TurkError::InvalidInput(format!(
                "Ones başlangıç değerleri için 2 boyutlu şekil gerekli, alınan: {:?}", shape
            )));
        }
        
        Ok(Array2::<f32>::ones((shape[0], shape[1])))
    }
    
    fn name(&self) -> &'static str {
        "ones"
    }
}

/// Rastgele normal dağılım başlangıç değerleri
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomNormal {
    pub mean: f32,
    pub std_dev: f32,
}

impl Default for RandomNormal {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.05,
        }
    }
}

impl Initializer for RandomNormal {
    fn initialize(&self, shape: &[usize]) -> Result<Array2<f32>> {
        if shape.len() != 2 {
            return Err(TurkError::InvalidInput(format!(
                "RandomNormal başlangıç değerleri için 2 boyutlu şekil gerekli, alınan: {:?}", shape
            )));
        }
        
        let dist = Normal::new(self.mean, self.std_dev)
            .map_err(|e| TurkError::InvalidInput(format!("Normal dağılım oluşturulurken hata: {}", e)))?;
        
        Ok(Array::random((shape[0], shape[1]), dist))
    }
    
    fn name(&self) -> &'static str {
        "random_normal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_xavier_initializer() {
        let init = Xavier::default();
        let shape = [10, 5];
        let result = init.initialize(&shape);
        assert!(result.is_ok());
        
        let weights = result.unwrap();
        assert_eq!(weights.shape(), &[10, 5]);
    }
    
    #[test]
    fn test_he_normal_initializer() {
        let init = HeNormal::default();
        let shape = [20, 10];
        let result = init.initialize(&shape);
        assert!(result.is_ok());
        
        let weights = result.unwrap();
        assert_eq!(weights.shape(), &[20, 10]);
    }
    
    #[test]
    fn test_zeros_initializer() {
        let init = Zeros;
        let shape = [5, 5];
        let result = init.initialize(&shape);
        assert!(result.is_ok());
        
        let weights = result.unwrap();
        assert_eq!(weights.shape(), &[5, 5]);
        assert!(weights.iter().all(|&x| x == 0.0));
    }
    
    #[test]
    fn test_ones_initializer() {
        let init = Ones;
        let shape = [3, 3];
        let result = init.initialize(&shape);
        assert!(result.is_ok());
        
        let weights = result.unwrap();
        assert_eq!(weights.shape(), &[3, 3]);
        assert!(weights.iter().all(|&x| x == 1.0));
    }
    
    #[test]
    fn test_random_normal_initializer() {
        let init = RandomNormal::default();
        let shape = [15, 8];
        let result = init.initialize(&shape);
        assert!(result.is_ok());
        
        let weights = result.unwrap();
        assert_eq!(weights.shape(), &[15, 8]);
    }
}
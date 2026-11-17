use crate::error::{Result, TurkError};
use ndarray::Array2;
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Ağırlık başlangıç değerleri için trait
use crate::dtype::Float;
pub trait Initializer: fmt::Debug + Send + Sync {
    /// Verilen boyutlarda rastgele değerlerle bir array oluşturur
    fn initialize(&self, shape: (usize, usize)) -> Result<Array2<Float>>;

    /// Başlatıcının adını döndürür
    fn name(&self) -> &'static str;
}

/// Xavier/Glorot başlangıç değerleri
#[derive(Debug, Clone)]
pub struct Xavier {
    pub gain: Float,
}

impl Default for Xavier {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl Initializer for Xavier {
    fn initialize(&self, shape: (usize, usize)) -> Result<Array2<Float>> {
        let fan_in = shape.0 as Float;
        let fan_out = shape.1 as Float;
        let std = self.gain * (2.0 / (fan_in + fan_out)).sqrt();

        let normal = Normal::new(0.0, std).map_err(|e| {
            TurkError::InvalidValue(format!("Normal dağılım oluşturulurken hata: {}", e))
        })?;

        let mut rng = thread_rng();
        let values: Vec<Float> = (0..shape.0 * shape.1)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Array2::from_shape_vec(shape, values).map_err(|e| {
            TurkError::ShapeMismatch(format!("Xavier başlatılırken şekil uyuşmazlığı: {}", e))
        })
    }

    fn name(&self) -> &'static str {
        "xavier"
    }
}

/// Glorot uniform başlangıç değerleri
#[derive(Debug, Serialize, Deserialize)]
pub struct GlorotUniform;

impl GlorotUniform {
    /// Yeni bir GlorotUniform initializer oluşturur
    pub fn new() -> Self {
        Self
    }
}

impl Initializer for GlorotUniform {
    fn initialize(&self, shape: (usize, usize)) -> Result<Array2<Float>> {
        let fan_in = shape.0 as Float;
        let fan_out = shape.1 as Float;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();

        let mut rng = thread_rng();
        let dist = Uniform::new(-limit, limit);

        let values: Vec<Float> = (0..shape.0 * shape.1)
            .map(|_| dist.sample(&mut rng))
            .collect();

        Array2::from_shape_vec(shape, values).map_err(|e| {
            TurkError::ShapeMismatch(format!(
                "GlorotUniform başlatılırken şekil uyuşmazlığı: {}",
                e
            ))
        })
    }

    fn name(&self) -> &'static str {
        "glorot_uniform"
    }
}

/// Glorot normal başlangıç değerleri
#[derive(Debug, Clone)]
pub struct GlorotNormal;

impl GlorotNormal {
    /// Yeni bir GlorotNormal initializer oluşturur
    pub fn new() -> Self {
        Self
    }
}

impl Initializer for GlorotNormal {
    fn initialize(&self, shape: (usize, usize)) -> Result<Array2<Float>> {
        let fan_in = shape.0 as Float;
        let fan_out = shape.1 as Float;
        let std = (2.0 / (fan_in + fan_out)).sqrt();

        let normal = Normal::new(0.0, std).map_err(|e| {
            TurkError::InvalidValue(format!("Normal dağılım oluşturulurken hata: {}", e))
        })?;

        let mut rng = thread_rng();
        let values: Vec<Float> = (0..shape.0 * shape.1)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Array2::from_shape_vec(shape, values).map_err(|e| {
            TurkError::ShapeMismatch(format!(
                "GlorotNormal başlatılırken şekil uyuşmazlığı: {}",
                e
            ))
        })
    }

    fn name(&self) -> &'static str {
        "glorot_normal"
    }
}

/// He normal başlangıç değerleri
#[derive(Debug, Clone)]
pub struct HeNormal {
    pub gain: Float,
}

impl Default for HeNormal {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl HeNormal {
    /// Yeni bir HeNormal initializer oluşturur
    pub fn new() -> Self {
        Self { gain: 1.0 }
    }
}

impl Initializer for HeNormal {
    fn initialize(&self, shape: (usize, usize)) -> Result<Array2<Float>> {
        let fan_in = shape.0 as Float;
        let std = self.gain * (2.0 / fan_in).sqrt();

        let normal = Normal::new(0.0, std).map_err(|e| {
            TurkError::InvalidValue(format!("Normal dağılım oluşturulurken hata: {}", e))
        })?;

        let mut rng = thread_rng();
        let values: Vec<Float> = (0..shape.0 * shape.1)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Array2::from_shape_vec(shape, values).map_err(|e| {
            TurkError::ShapeMismatch(format!("HeNormal başlatılırken şekil uyuşmazlığı: {}", e))
        })
    }

    fn name(&self) -> &'static str {
        "he_normal"
    }
}

/// He uniform başlangıç değerleri
#[derive(Debug, Clone)]
pub struct HeUniform;

impl HeUniform {
    /// Yeni bir HeUniform initializer oluşturur
    pub fn new() -> Self {
        Self
    }
}

impl Initializer for HeUniform {
    fn initialize(&self, shape: (usize, usize)) -> Result<Array2<Float>> {
        let fan_in = shape.0 as Float;
        let limit = (6.0 / fan_in).sqrt();

        let mut rng = thread_rng();
        let dist = Uniform::new(-limit, limit);

        let values: Vec<Float> = (0..shape.0 * shape.1)
            .map(|_| dist.sample(&mut rng))
            .collect();

        Array2::from_shape_vec(shape, values).map_err(|e| {
            TurkError::ShapeMismatch(format!("HeUniform başlatılırken şekil uyuşmazlığı: {}", e))
        })
    }

    fn name(&self) -> &'static str {
        "he_uniform"
    }
}

/// Sıfır başlangıç değerleri
#[derive(Debug, Clone)]
pub struct Zeros;

impl Zeros {
    /// Yeni bir Zeros initializer oluşturur
    pub fn new() -> Self {
        Self
    }
}

impl Initializer for Zeros {
    fn initialize(&self, shape: (usize, usize)) -> Result<Array2<Float>> {
        Ok(Array2::zeros(shape))
    }

    fn name(&self) -> &'static str {
        "zeros"
    }
}

/// Bir başlangıç değerleri
#[derive(Debug, Clone)]
pub struct Ones;

impl Ones {
    /// Yeni bir Ones initializer oluşturur
    pub fn new() -> Self {
        Self
    }
}

impl Initializer for Ones {
    fn initialize(&self, shape: (usize, usize)) -> Result<Array2<Float>> {
        Ok(Array2::ones(shape))
    }

    fn name(&self) -> &'static str {
        "ones"
    }
}

/// Rastgele normal dağılım başlangıç değerleri
#[derive(Debug, Clone)]
pub struct RandomNormal {
    pub mean: Float,
    pub stddev: Float,
}

impl Default for RandomNormal {
    fn default() -> Self {
        Self {
            mean: 0.0,
            stddev: 0.05,
        }
    }
}

impl RandomNormal {
    pub fn new(mean: Float, stddev: Float) -> Result<Self> {
        if stddev <= 0.0 {
            return Err(TurkError::InvalidValue(format!(
                "Standart sapma pozitif olmalıdır: {}",
                stddev
            )));
        }
        Ok(Self { mean, stddev })
    }
}

impl Initializer for RandomNormal {
    fn initialize(&self, shape: (usize, usize)) -> Result<Array2<Float>> {
        let normal = Normal::new(self.mean, self.stddev).map_err(|e| {
            TurkError::InvalidValue(format!("Normal dağılım oluşturulurken hata: {}", e))
        })?;

        let mut rng = thread_rng();
        let values: Vec<Float> = (0..shape.0 * shape.1)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Array2::from_shape_vec(shape, values).map_err(|e| {
            TurkError::ShapeMismatch(format!(
                "RandomNormal başlatılırken şekil uyuşmazlığı: {}",
                e
            ))
        })
    }

    fn name(&self) -> &'static str {
        "random_normal"
    }
}

/// Başlatıcı oluştur
pub fn create_initializer(initializer: &str) -> Result<Box<dyn Initializer>> {
    match initializer {
        "glorot_uniform" => Ok(Box::new(GlorotUniform::new())),
        "glorot_normal" => Ok(Box::new(GlorotNormal::new())),
        "he_uniform" => Ok(Box::new(HeUniform::new())),
        "he_normal" => Ok(Box::new(HeNormal::new())),
        "zeros" => Ok(Box::new(Zeros::new())),
        "ones" => Ok(Box::new(Ones::new())),
        "random_normal" => Ok(Box::new(RandomNormal::default())),
        _ => Err(TurkError::ModelError(format!(
            "Geçersiz başlatıcı: {}",
            initializer
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_initializer() {
        let init = Xavier::default();
        let shape = (10, 5);
        let result = init.initialize(shape);
        assert!(result.is_ok());

        let weights = result.unwrap();
        assert_eq!(weights.shape(), &[10, 5]);
    }

    #[test]
    fn test_he_normal_initializer() {
        let init = HeNormal::default();
        let shape = (20, 10);
        let result = init.initialize(shape);
        assert!(result.is_ok());

        let weights = result.unwrap();
        assert_eq!(weights.shape(), &[20, 10]);
    }

    #[test]
    fn test_zeros_initializer() {
        let init = Zeros;
        let shape = (5, 5);
        let result = init.initialize(shape);
        assert!(result.is_ok());

        let weights = result.unwrap();
        assert_eq!(weights.shape(), &[5, 5]);
        assert!(weights.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones_initializer() {
        let init = Ones;
        let shape = (3, 3);
        let result = init.initialize(shape);
        assert!(result.is_ok());

        let weights = result.unwrap();
        assert_eq!(weights.shape(), &[3, 3]);
        assert!(weights.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_random_normal_initializer() {
        let init = RandomNormal::default();
        let shape = (15, 8);
        let result = init.initialize(shape);
        assert!(result.is_ok());

        let weights = result.unwrap();
        assert_eq!(weights.shape(), &[15, 8]);
    }
}

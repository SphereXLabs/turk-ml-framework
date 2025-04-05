use crate::error::{Result, TurkError};
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use std::fmt;
use std::collections::HashMap;

/// Optimizer trait - ağırlık güncellemeleri için
pub trait Optimizer: fmt::Debug + Send + Sync {
    /// Parametreleri ve gradyanları alarak güncelleme yapar
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> Result<()>;
    
    /// Öğrenme oranını döndürür
    fn learning_rate(&self) -> f32;
    
    /// Optimizer tipini döndürür
    fn optimizer_type(&self) -> &'static str;
    
    /// Optimizer'ı klonlar
    fn clone_box(&self) -> Box<dyn Optimizer>;
    
    /// Parametre ismi için state'i kaldırır
    fn reset_state_for(&mut self, param_name: &str);
    
    /// Tüm state'i kaldırır
    fn reset_state(&mut self);
}

/// Stokastik Gradient İniş (SGD) optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocity: HashMap<String, Array2<f32>>,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            velocity: HashMap::new(),
        }
    }
    
    pub fn with_momentum(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> Result<()> {
        if params.shape() != grads.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "SGD step: Parametre şekli: {:?}, Gradyan şekli: {:?}",
                params.shape(),
                grads.shape()
            )));
        }
        
        let param_key = format!("{:p}", params.as_ptr());
        
        if self.momentum > 0.0 {
            // Momentum kullanıyoruz
            let velocity = self.velocity.entry(param_key)
                .or_insert_with(|| Array2::zeros(params.raw_dim()));
            
            // v = momentum * v - learning_rate * grads
            *velocity = &*velocity * self.momentum - grads * self.learning_rate;
            
            // params += v
            *params += velocity;
        } else {
            // Standart SGD
            *params -= &(grads * self.learning_rate);
        }
        
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
    
    fn optimizer_type(&self) -> &'static str {
        if self.momentum > 0.0 {
            "sgd_momentum"
        } else {
            "sgd"
        }
    }
    
    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
    
    fn reset_state_for(&mut self, param_name: &str) {
        self.velocity.remove(param_name);
    }
    
    fn reset_state(&mut self) {
        self.velocity.clear();
    }
}

/// Adam optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m: HashMap<String, Array2<f32>>,     // Birinci moment (gradyan ortalaması)
    v: HashMap<String, Array2<f32>>,     // İkinci moment (gradyan karelerinin ortalaması)
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
    
    pub fn with_betas(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> Result<()> {
        if params.shape() != grads.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Adam step: Parametre şekli: {:?}, Gradyan şekli: {:?}",
                params.shape(),
                grads.shape()
            )));
        }

        let param_key = format!("{:p}", params.as_ptr());
        self.t += 1;
        let t = self.t as f32;

        // Momenti başlat veya al
        let m = self.m.entry(param_key.clone())
            .or_insert_with(|| Array2::zeros(params.raw_dim()));
        
        // Hızı başlat veya al
        let v = self.v.entry(param_key)
            .or_insert_with(|| Array2::zeros(params.raw_dim()));
        
        // Moment ve hız güncellemeleri
        *m = m * self.beta1 + grads * (1.0 - self.beta1);
        *v = v * self.beta2 + grads.map(|x| x * x) * (1.0 - self.beta2);

        // Bias düzeltmeleri
        let m_hat = m / (1.0 - self.beta1.powf(t));
        let v_hat = v / (1.0 - self.beta2.powf(t));

        // Parametre güncellemesi
        *params -= &(m_hat.clone() * self.learning_rate / (v_hat.map(|x| x.sqrt()) + self.epsilon));
        
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
    
    fn optimizer_type(&self) -> &'static str {
        "adam"
    }
    
    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
    
    fn reset_state_for(&mut self, param_name: &str) {
        self.m.remove(param_name);
        self.v.remove(param_name);
    }
    
    fn reset_state(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
    }
}

/// AdamW optimizer - Ağırlık aşınması (weight decay) ile Adam
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    t: usize,
    m: HashMap<String, Array2<f32>>,
    v: HashMap<String, Array2<f32>>,
}

impl AdamW {
    pub fn new(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
    
    pub fn with_betas(learning_rate: f32, weight_decay: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> Result<()> {
        if params.shape() != grads.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "AdamW step: Parametre şekli: {:?}, Gradyan şekli: {:?}",
                params.shape(),
                grads.shape()
            )));
        }

        let param_key = format!("{:p}", params.as_ptr());
        self.t += 1;
        let t = self.t as f32;

        // Momenti başlat veya al
        let m = self.m.entry(param_key.clone())
            .or_insert_with(|| Array2::zeros(params.raw_dim()));
        
        // Hızı başlat veya al
        let v = self.v.entry(param_key)
            .or_insert_with(|| Array2::zeros(params.raw_dim()));

        // Moment ve hız güncellemeleri - standart Adam gibi
        *m = m * self.beta1 + grads * (1.0 - self.beta1);
        *v = v * self.beta2 + grads.map(|x| x * x) * (1.0 - self.beta2);

        // Bias düzeltmeleri
        let m_hat = m / (1.0 - self.beta1.powf(t));
        let v_hat = v / (1.0 - self.beta2.powf(t));
        
        // AdamW: Önce ağırlık aşınması uygula, sonra Adam güncellemesi
        *params *= 1.0 - self.learning_rate * self.weight_decay;
        *params -= &(m_hat.clone() * self.learning_rate / (v_hat.map(|x| x.sqrt()) + self.epsilon));
        
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
    
    fn optimizer_type(&self) -> &'static str {
        "adamw"
    }
    
    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
    
    fn reset_state_for(&mut self, param_name: &str) {
        self.m.remove(param_name);
        self.v.remove(param_name);
    }
    
    fn reset_state(&mut self) {
        self.t = 0;
        self.m.clear();
        self.v.clear();
    }
}

/// RMSProp optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSProp {
    learning_rate: f32,
    decay_rate: f32,
    epsilon: f32,
    cache: HashMap<String, Array2<f32>>,
}

impl RMSProp {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            decay_rate: 0.9,
            epsilon: 1e-8,
            cache: HashMap::new(),
        }
    }
    
    pub fn with_decay(learning_rate: f32, decay_rate: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            decay_rate,
            epsilon,
            cache: HashMap::new(),
        }
    }
}

impl Optimizer for RMSProp {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> Result<()> {
        if params.shape() != grads.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "RMSProp step: Parametre şekli: {:?}, Gradyan şekli: {:?}",
                params.shape(),
                grads.shape()
            )));
        }

        let param_key = format!("{:p}", params.as_ptr());
        
        // Cache başlat veya al
        let cache = self.cache.entry(param_key)
            .or_insert_with(|| Array2::zeros(params.raw_dim()));

        // Cache güncelleme: decay_rate * cache + (1 - decay_rate) * grads^2
        *cache = cache * self.decay_rate + grads.map(|x| x * x) * (1.0 - self.decay_rate);

        // Parametre güncellemesi: params -= learning_rate * grads / sqrt(cache + epsilon)
        *params -= &(grads * self.learning_rate / cache.map(|x| (x + self.epsilon).sqrt()));
        
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
    
    fn optimizer_type(&self) -> &'static str {
        "rmsprop"
    }
    
    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
    
    fn reset_state_for(&mut self, param_name: &str) {
        self.cache.remove(param_name);
    }
    
    fn reset_state(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sgd_step() {
        let mut optimizer = SGD::new(0.01);
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        
        // Standart SGD
        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(params.shape(), &[2, 2]);
        
        // Momentum ile SGD
        let mut optimizer_momentum = SGD::with_momentum(0.01, 0.9);
        let mut params2 = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        
        optimizer_momentum.step(&mut params2, &grads).unwrap();
        assert_eq!(params2.shape(), &[2, 2]);
    }

    #[test]
    fn test_adam_step() {
        let mut optimizer = Adam::new(0.01);
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        
        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(params.shape(), &[2, 2]);
        
        // Özel beta değerleri ile
        let mut optimizer_custom = Adam::with_betas(0.01, 0.8, 0.99, 1e-10);
        let mut params2 = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        
        optimizer_custom.step(&mut params2, &grads).unwrap();
        assert_eq!(params2.shape(), &[2, 2]);
    }
    
    #[test]
    fn test_adamw_step() {
        let mut optimizer = AdamW::new(0.01, 0.01);
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        
        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(params.shape(), &[2, 2]);
    }
    
    #[test]
    fn test_rmsprop_step() {
        let mut optimizer = RMSProp::new(0.01);
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        
        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(params.shape(), &[2, 2]);
    }
} 
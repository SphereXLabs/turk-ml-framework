use crate::error::{Result, TurkError};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Optimizer trait'i - Tüm optimizasyon algoritmaları bu trait'i implemente etmelidir
use crate::dtype::Float;
pub trait Optimizer: fmt::Debug + Send + Sync {
    /// Ağırlık ve bias değerlerini gradyanlarına göre günceller
    fn update(
        &self,
        weights: &mut Array2<Float>,
        biases: &mut Array2<Float>,
        grad_weights: &ArrayView2<Float>,
        grad_biases: &ArrayView2<Float>,
    ) -> Result<()>;

    /// Öğrenme oranını döndürür
    fn learning_rate(&self) -> Float;

    /// Öğrenme oranını ayarlar
    fn set_learning_rate(&mut self, learning_rate: Float);

    /// Optimizer durumunu sıfırlar
    fn reset(&mut self);

    /// Optimizer'ı klonlar
    fn clone_box(&self) -> Box<dyn Optimizer>;

    /// Optimizer'ın tipini döndürür
    fn optimizer_type(&self) -> &'static str;
}

/// Optimizer için Clone trait'i implementasyonu
impl Clone for Box<dyn Optimizer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Stokastik Gradyan İnişi (SGD) optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGD {
    /// Öğrenme oranı
    learning_rate: Float,
    /// Momentum değeri (isteğe bağlı)
    momentum: Option<Float>,
    /// Nesterov momentum'u kullan
    nesterov: bool,
    /// Hızlar (momentum için)
    velocity_weights: Option<Array2<Float>>,
    /// Hızlar (bias için)
    velocity_biases: Option<Array2<Float>>,
    /// İterasyon sayısı
    iterations: usize,
    /// Öğrenme oranının azalma oranı
    decay: Float,
}

impl SGD {
    /// Yeni bir SGD optimizer oluşturur
    pub fn new(learning_rate: Float, momentum: Option<Float>, nesterov: bool) -> Self {
        Self {
            learning_rate,
            momentum,
            nesterov,
            velocity_weights: None,
            velocity_biases: None,
            iterations: 0,
            decay: 0.0,
        }
    }

    /// Varsayılan parametrelerle yeni bir SGD optimizer oluşturur
    pub fn default() -> Self {
        Self::new(0.01, None, false)
    }
}

impl Optimizer for SGD {
    fn update(
        &self,
        weights: &mut Array2<Float>,
        biases: &mut Array2<Float>,
        grad_weights: &ArrayView2<Float>,
        grad_biases: &ArrayView2<Float>,
    ) -> Result<()> {
        // Boyut kontrolü
        if weights.shape() != grad_weights.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "Ağırlık ve gradyan boyutları uyuşmuyor: {:?} vs {:?}",
                weights.shape(),
                grad_weights.shape()
            )));
        }

        if biases.shape() != grad_biases.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "Bias ve gradyan boyutları uyuşmuyor: {:?} vs {:?}",
                biases.shape(),
                grad_biases.shape()
            )));
        }

        // Öğrenme oranını hesapla (decay ile)
        let actual_lr = self.learning_rate / (1.0 + self.decay * self.iterations as Float);

        // Gradyanları öğrenme oranıyla çarp
        let weight_updates = grad_weights * actual_lr;
        let bias_updates = grad_biases * actual_lr;

        // Momentum kullanılıyorsa
        if let Some(momentum) = self.momentum {
            let mut self_mut = self.clone();

            // Velocity'i ilk kez oluştur (lazy initialization)
            if self_mut.velocity_weights.is_none() {
                self_mut.velocity_weights = Some(Array2::zeros(weights.dim()));
            }

            if self_mut.velocity_biases.is_none() {
                self_mut.velocity_biases = Some(Array2::zeros(biases.dim()));
            }

            let v_w = self_mut.velocity_weights.as_mut().unwrap();
            let v_b = self_mut.velocity_biases.as_mut().unwrap();

            // Nesterov momentum kullanılıyorsa
            if self.nesterov {
                // v = momentum * v - lr * grad
                *v_w = &(v_w.clone() * momentum) - &weight_updates;
                *v_b = &(v_b.clone() * momentum) - &bias_updates;

                // params = params + momentum * v - lr * grad
                *weights = &*weights + &(&*v_w * momentum) - &weight_updates;
                *biases = &*biases + &(&*v_b * momentum) - &bias_updates;
            } else {
                // v = momentum * v - lr * grad
                *v_w = &(v_w.clone() * momentum) - &weight_updates;
                *v_b = &(v_b.clone() * momentum) - &bias_updates;

                // params = params + v
                *weights = &*weights + &*v_w;
                *biases = &*biases + &*v_b;
            }
        } else {
            // Normal SGD: params = params - lr * grad
            *weights = &*weights - &weight_updates;
            *biases = &*biases - &bias_updates;
        }

        Ok(())
    }

    fn optimizer_type(&self) -> &'static str {
        "sgd"
    }

    fn learning_rate(&self) -> Float {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: Float) {
        self.learning_rate = learning_rate;
    }

    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }

    fn reset(&mut self) {
        self.velocity_weights = None;
        self.velocity_biases = None;
    }
}

/// Adam optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam {
    /// Öğrenme oranı
    learning_rate: Float,
    /// Beta1 (ilk moment için momentum faktörü)
    beta1: Float,
    /// Beta2 (ikinci moment için momentum faktörü)
    beta2: Float,
    /// Epsilon (sayısal stabilite için)
    epsilon: Float,
    /// İlk moment (weights)
    m_weights: Option<Array2<Float>>,
    /// İlk moment (biases)
    m_biases: Option<Array2<Float>>,
    /// İkinci moment (weights)
    v_weights: Option<Array2<Float>>,
    /// İkinci moment (biases)
    v_biases: Option<Array2<Float>>,
    /// Adım sayacı
    t: usize,
}

impl Adam {
    /// Yeni bir Adam optimizer oluşturur
    pub fn new(learning_rate: Float, beta1: Float, beta2: Float, epsilon: Float) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m_weights: None,
            m_biases: None,
            v_weights: None,
            v_biases: None,
            t: 0,
        }
    }

    /// Varsayılan parametrelerle yeni bir Adam optimizer oluşturur
    pub fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8)
    }
}

impl Optimizer for Adam {
    fn update(
        &self,
        weights: &mut Array2<Float>,
        biases: &mut Array2<Float>,
        grad_weights: &ArrayView2<Float>,
        grad_biases: &ArrayView2<Float>,
    ) -> Result<()> {
        // Boyut kontrolü
        if weights.shape() != grad_weights.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "Ağırlık ve gradyan boyutları uyuşmuyor: {:?} vs {:?}",
                weights.shape(),
                grad_weights.shape()
            )));
        }

        if biases.shape() != grad_biases.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "Bias ve gradyan boyutları uyuşmuyor: {:?} vs {:?}",
                biases.shape(),
                grad_biases.shape()
            )));
        }

        let mut self_mut = self.clone();
        self_mut.t += 1;

        // İlk momenti ilk kez oluştur
        if self_mut.m_weights.is_none() {
            self_mut.m_weights = Some(Array2::zeros(weights.dim()));
        }

        if self_mut.m_biases.is_none() {
            self_mut.m_biases = Some(Array2::zeros(biases.dim()));
        }

        // İkinci momenti ilk kez oluştur
        if self_mut.v_weights.is_none() {
            self_mut.v_weights = Some(Array2::zeros(weights.dim()));
        }

        if self_mut.v_biases.is_none() {
            self_mut.v_biases = Some(Array2::zeros(biases.dim()));
        }

        let m_w = self_mut.m_weights.as_mut().unwrap();
        let m_b = self_mut.m_biases.as_mut().unwrap();
        let v_w = self_mut.v_weights.as_mut().unwrap();
        let v_b = self_mut.v_biases.as_mut().unwrap();

        // İlk moment güncellemesi (momentumlu gradyan)
        *m_w = &(m_w.clone() * self.beta1) + &(grad_weights * (1.0 - self.beta1));
        *m_b = &(m_b.clone() * self.beta1) + &(grad_biases * (1.0 - self.beta1));

        // İkinci moment güncellemesi (gradyanın karesi)
        *v_w = &(v_w.clone() * self.beta2) + &(grad_weights.mapv(|g| g * g) * (1.0 - self.beta2));
        *v_b = &(v_b.clone() * self.beta2) + &(grad_biases.mapv(|g| g * g) * (1.0 - self.beta2));

        // Bias düzeltmesi
        let m_w_hat = &*m_w / (1.0 - self.beta1.powi(self_mut.t as i32));
        let m_b_hat = &*m_b / (1.0 - self.beta1.powi(self_mut.t as i32));
        let v_w_hat = &*v_w / (1.0 - self.beta2.powi(self_mut.t as i32));
        let v_b_hat = &*v_b / (1.0 - self.beta2.powi(self_mut.t as i32));

        // Parametre güncellemesi
        *weights = &*weights
            - (self.learning_rate * &m_w_hat / &(v_w_hat.mapv(|v| v.sqrt()) + self.epsilon));
        *biases = &*biases
            - (self.learning_rate * &m_b_hat / &(v_b_hat.mapv(|v| v.sqrt()) + self.epsilon));

        Ok(())
    }

    fn learning_rate(&self) -> Float {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: Float) {
        self.learning_rate = learning_rate;
    }

    fn reset(&mut self) {
        self.m_weights = None;
        self.m_biases = None;
        self.v_weights = None;
        self.v_biases = None;
        self.t = 0;
    }

    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }

    fn optimizer_type(&self) -> &'static str {
        "Adam"
    }
}

/// AdamW optimizer (Adam + Ağırlık bozunumu)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdamW {
    /// Öğrenme oranı
    learning_rate: Float,
    /// Beta1 (ilk moment için momentum faktörü)
    beta1: Float,
    /// Beta2 (ikinci moment için momentum faktörü)
    beta2: Float,
    /// Epsilon (sayısal stabilite için)
    epsilon: Float,
    /// Ağırlık bozunumu
    weight_decay: Float,
    /// İlk moment (weights)
    m_weights: Option<Array2<Float>>,
    /// İlk moment (biases)
    m_biases: Option<Array2<Float>>,
    /// İkinci moment (weights)
    v_weights: Option<Array2<Float>>,
    /// İkinci moment (biases)
    v_biases: Option<Array2<Float>>,
    /// Adım sayacı
    t: usize,
}

impl AdamW {
    /// Yeni bir AdamW optimizer oluşturur
    pub fn new(
        learning_rate: Float,
        beta1: Float,
        beta2: Float,
        epsilon: Float,
        weight_decay: Float,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            m_weights: None,
            m_biases: None,
            v_weights: None,
            v_biases: None,
            t: 0,
        }
    }

    /// Varsayılan parametrelerle yeni bir AdamW optimizer oluşturur
    pub fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, 0.01)
    }
}

impl Optimizer for AdamW {
    fn update(
        &self,
        weights: &mut Array2<Float>,
        biases: &mut Array2<Float>,
        grad_weights: &ArrayView2<Float>,
        grad_biases: &ArrayView2<Float>,
    ) -> Result<()> {
        // Boyut kontrolü
        if weights.shape() != grad_weights.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "Ağırlık ve gradyan boyutları uyuşmuyor: {:?} vs {:?}",
                weights.shape(),
                grad_weights.shape()
            )));
        }

        if biases.shape() != grad_biases.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "Bias ve gradyan boyutları uyuşmuyor: {:?} vs {:?}",
                biases.shape(),
                grad_biases.shape()
            )));
        }

        let mut self_mut = self.clone();
        self_mut.t += 1;

        // İlk momenti ilk kez oluştur
        if self_mut.m_weights.is_none() {
            self_mut.m_weights = Some(Array2::zeros(weights.dim()));
        }

        if self_mut.m_biases.is_none() {
            self_mut.m_biases = Some(Array2::zeros(biases.dim()));
        }

        // İkinci momenti ilk kez oluştur
        if self_mut.v_weights.is_none() {
            self_mut.v_weights = Some(Array2::zeros(weights.dim()));
        }

        if self_mut.v_biases.is_none() {
            self_mut.v_biases = Some(Array2::zeros(biases.dim()));
        }

        let m_w = self_mut.m_weights.as_mut().unwrap();
        let m_b = self_mut.m_biases.as_mut().unwrap();
        let v_w = self_mut.v_weights.as_mut().unwrap();
        let v_b = self_mut.v_biases.as_mut().unwrap();

        // Ağırlık bozunumu (weight decay) uygulaması - AdamW'nin özelliği
        let weight_decay_term = &*weights * self.weight_decay;

        // İlk moment güncellemesi (momentumlu gradyan)
        *m_w = &(m_w.clone() * self.beta1) + &(grad_weights * (1.0 - self.beta1));
        *m_b = &(m_b.clone() * self.beta1) + &(grad_biases * (1.0 - self.beta1));

        // İkinci moment güncellemesi (gradyanın karesi)
        *v_w = &(v_w.clone() * self.beta2) + &(grad_weights.mapv(|g| g * g) * (1.0 - self.beta2));
        *v_b = &(v_b.clone() * self.beta2) + &(grad_biases.mapv(|g| g * g) * (1.0 - self.beta2));

        // Bias düzeltmesi
        let m_w_hat = &*m_w / (1.0 - self.beta1.powi(self_mut.t as i32));
        let m_b_hat = &*m_b / (1.0 - self.beta1.powi(self_mut.t as i32));
        let v_w_hat = &*v_w / (1.0 - self.beta2.powi(self_mut.t as i32));
        let v_b_hat = &*v_b / (1.0 - self.beta2.powi(self_mut.t as i32));

        // Parametre güncellemesi (AdamW'ye özel olarak weight decay terimi eklenmiş)
        *weights = &*weights
            - (self.learning_rate
                * (&m_w_hat / &(v_w_hat.mapv(|v| v.sqrt()) + self.epsilon) + &weight_decay_term));
        *biases = &*biases
            - (self.learning_rate * &m_b_hat / &(v_b_hat.mapv(|v| v.sqrt()) + self.epsilon));

        Ok(())
    }

    fn learning_rate(&self) -> Float {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: Float) {
        self.learning_rate = learning_rate;
    }

    fn reset(&mut self) {
        self.m_weights = None;
        self.m_biases = None;
        self.v_weights = None;
        self.v_biases = None;
        self.t = 0;
    }

    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }

    fn optimizer_type(&self) -> &'static str {
        "AdamW"
    }
}

/// RMSProp optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSProp {
    /// Öğrenme oranı
    learning_rate: Float,
    /// Bozunum oranı
    decay: Float,
    /// Epsilon (sayısal stabilite için)
    epsilon: Float,
    /// Kare gradyan ortalaması (weights)
    cache_weights: Option<Array2<Float>>,
    /// Kare gradyan ortalaması (biases)
    cache_biases: Option<Array2<Float>>,
}

impl RMSProp {
    /// Yeni bir RMSProp optimizer oluşturur
    pub fn new(learning_rate: Float, decay: Float, epsilon: Float) -> Self {
        Self {
            learning_rate,
            decay,
            epsilon,
            cache_weights: None,
            cache_biases: None,
        }
    }

    /// Varsayılan parametrelerle yeni bir RMSProp optimizer oluşturur
    pub fn default() -> Self {
        Self::new(0.001, 0.9, 1e-8)
    }
}

impl Optimizer for RMSProp {
    fn update(
        &self,
        weights: &mut Array2<Float>,
        biases: &mut Array2<Float>,
        grad_weights: &ArrayView2<Float>,
        grad_biases: &ArrayView2<Float>,
    ) -> Result<()> {
        // Boyut kontrolü
        if weights.shape() != grad_weights.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "Ağırlık ve gradyan boyutları uyuşmuyor: {:?} vs {:?}",
                weights.shape(),
                grad_weights.shape()
            )));
        }

        if biases.shape() != grad_biases.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "Bias ve gradyan boyutları uyuşmuyor: {:?} vs {:?}",
                biases.shape(),
                grad_biases.shape()
            )));
        }

        // Cache'i ilk kez oluştur (lazy initialization)
        let mut self_mut = self.clone();

        if self_mut.cache_weights.is_none() {
            self_mut.cache_weights = Some(Array2::zeros(weights.dim()));
        }

        if self_mut.cache_biases.is_none() {
            self_mut.cache_biases = Some(Array2::zeros(biases.dim()));
        }

        let cache_w = self_mut.cache_weights.as_mut().unwrap();
        let cache_b = self_mut.cache_biases.as_mut().unwrap();

        // Cache güncellemesi (gradyanın karesi)
        *cache_w =
            &(cache_w.clone() * self.decay) + &(grad_weights.mapv(|g| g * g) * (1.0 - self.decay));
        *cache_b =
            &(cache_b.clone() * self.decay) + &(grad_biases.mapv(|g| g * g) * (1.0 - self.decay));

        // Parametre güncellemesi
        *weights = &*weights
            - &(self.learning_rate * grad_weights / &(cache_w.mapv(|c| (c + self.epsilon).sqrt())));
        *biases = &*biases
            - &(self.learning_rate * grad_biases / &(cache_b.mapv(|c| (c + self.epsilon).sqrt())));

        Ok(())
    }

    fn learning_rate(&self) -> Float {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: Float) {
        self.learning_rate = learning_rate;
    }

    fn reset(&mut self) {
        self.cache_weights = None;
        self.cache_biases = None;
    }

    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }

    fn optimizer_type(&self) -> &'static str {
        "RMSProp"
    }
}

/// Optimizer tipine göre optimizer oluşturur
pub fn create_optimizer(optimizer_type: &str) -> Result<Box<dyn Optimizer>> {
    match optimizer_type.to_lowercase().as_str() {
        "sgd" => Ok(Box::new(SGD::default())),
        "adam" => Ok(Box::new(Adam::default())),
        "adamw" => Ok(Box::new(AdamW::default())),
        "rmsprop" => Ok(Box::new(RMSProp::default())),
        _ => Err(TurkError::ValueError(format!(
            "Bilinmeyen optimizer: {}",
            optimizer_type
        ))),
    }
}

/// Model yükleme için parametrelerden optimizer oluşturur
pub fn create_optimizer_from_params(
    optimizer_type: &str,
    learning_rate: Float,
    momentum: Option<Float>,
    beta1: Option<Float>,
    beta2: Option<Float>,
    epsilon: Option<Float>,
    weight_decay: Option<Float>,
) -> Result<Box<dyn Optimizer>> {
    match optimizer_type.to_lowercase().as_str() {
        "sgd" => {
            let momentum_val = momentum.unwrap_or(0.0);
            let nesterov = momentum_val > 0.0;
            Ok(Box::new(SGD::new(learning_rate, momentum, nesterov)))
        }
        "adam" => {
            let beta1_val = beta1.unwrap_or(0.9);
            let beta2_val = beta2.unwrap_or(0.999);
            let epsilon_val = epsilon.unwrap_or(1e-8);
            Ok(Box::new(Adam::new(
                learning_rate,
                beta1_val,
                beta2_val,
                epsilon_val,
            )))
        }
        "adamw" => {
            let beta1_val = beta1.unwrap_or(0.9);
            let beta2_val = beta2.unwrap_or(0.999);
            let epsilon_val = epsilon.unwrap_or(1e-8);
            let weight_decay_val = weight_decay.unwrap_or(0.01);
            Ok(Box::new(AdamW::new(
                learning_rate,
                beta1_val,
                beta2_val,
                epsilon_val,
                weight_decay_val,
            )))
        }
        "rmsprop" => {
            let decay = beta1.unwrap_or(0.9);
            let epsilon_val = epsilon.unwrap_or(1e-8);
            Ok(Box::new(RMSProp::new(learning_rate, decay, epsilon_val)))
        }
        _ => Err(TurkError::ValueError(format!(
            "Bilinmeyen optimizer: {}",
            optimizer_type
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_sgd() {
        let mut weights = array![[1.0, 2.0], [3.0, 4.0]];
        let mut biases = array![[0.5, 0.5]];
        let grad_weights = array![[0.1, 0.2], [0.3, 0.4]];
        let grad_biases = array![[0.05, 0.05]];

        let sgd = SGD::new(0.1, None, false);
        sgd.update(
            &mut weights,
            &mut biases,
            &grad_weights.view(),
            &grad_biases.view(),
        )
        .unwrap();

        // weights -= lr * grad_weights
        assert_abs_diff_eq!(weights[[0, 0]], 0.99, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[[0, 1]], 1.98, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[[1, 0]], 2.97, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[[1, 1]], 3.96, epsilon = 1e-5);

        // biases -= lr * grad_biases
        assert_abs_diff_eq!(biases[[0, 0]], 0.495, epsilon = 1e-5);
        assert_abs_diff_eq!(biases[[0, 1]], 0.495, epsilon = 1e-5);
    }

    #[test]
    fn test_sgd_momentum() {
        let mut weights = array![[1.0, 2.0], [3.0, 4.0]];
        let mut biases = array![[0.5, 0.5]];
        let grad_weights = array![[0.1, 0.2], [0.3, 0.4]];
        let grad_biases = array![[0.05, 0.05]];

        let sgd = SGD::new(0.1, Some(0.9), false);
        sgd.update(
            &mut weights,
            &mut biases,
            &grad_weights.view(),
            &grad_biases.view(),
        )
        .unwrap();

        // Momentum için ilk adım: v = -lr * grad
        // weights += v
        assert_abs_diff_eq!(weights[[0, 0]], 0.99, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[[0, 1]], 1.98, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[[1, 0]], 2.97, epsilon = 1e-5);
        assert_abs_diff_eq!(weights[[1, 1]], 3.96, epsilon = 1e-5);
    }

    #[test]
    fn test_adam() {
        let mut weights = array![[1.0, 2.0], [3.0, 4.0]];
        let mut biases = array![[0.5, 0.5]];
        let grad_weights = array![[0.1, 0.2], [0.3, 0.4]];
        let grad_biases = array![[0.05, 0.05]];

        let adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
        adam.update(
            &mut weights,
            &mut biases,
            &grad_weights.view(),
            &grad_biases.view(),
        )
        .unwrap();

        // Adam ilk adımda değişiklik küçük olacaktır
        // Sadece bir şeylerin değiştiğini kontrol edelim
        assert!(weights[[0, 0]] < 1.0);
        assert!(weights[[0, 1]] < 2.0);
        assert!(weights[[1, 0]] < 3.0);
        assert!(weights[[1, 1]] < 4.0);
    }

    #[test]
    fn test_adamw() {
        let mut weights = array![[1.0, 2.0], [3.0, 4.0]];
        let mut biases = array![[0.5, 0.5]];
        let grad_weights = array![[0.1, 0.2], [0.3, 0.4]];
        let grad_biases = array![[0.05, 0.05]];

        let adamw = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
        adamw
            .update(
                &mut weights,
                &mut biases,
                &grad_weights.view(),
                &grad_biases.view(),
            )
            .unwrap();

        // AdamW ilk adımda değişiklik küçük olacaktır
        // Sadece bir şeylerin değiştiğini kontrol edelim
        assert!(weights[[0, 0]] < 1.0);
        assert!(weights[[0, 1]] < 2.0);
        assert!(weights[[1, 0]] < 3.0);
        assert!(weights[[1, 1]] < 4.0);
    }

    #[test]
    fn test_rmsprop() {
        let mut weights = array![[1.0, 2.0], [3.0, 4.0]];
        let mut biases = array![[0.5, 0.5]];
        let grad_weights = array![[0.1, 0.2], [0.3, 0.4]];
        let grad_biases = array![[0.05, 0.05]];

        let rmsprop = RMSProp::new(0.001, 0.9, 1e-8);
        rmsprop
            .update(
                &mut weights,
                &mut biases,
                &grad_weights.view(),
                &grad_biases.view(),
            )
            .unwrap();

        // RMSProp ilk adımda değişiklik küçük olacaktır
        // Sadece bir şeylerin değiştiğini kontrol edelim
        assert!(weights[[0, 0]] < 1.0);
        assert!(weights[[0, 1]] < 2.0);
        assert!(weights[[1, 0]] < 3.0);
        assert!(weights[[1, 1]] < 4.0);
    }

    #[test]
    fn test_optimizer_factory() {
        let sgd = create_optimizer("sgd").unwrap();
        assert_eq!(sgd.learning_rate(), 0.01);

        let adam = create_optimizer("adam").unwrap();
        assert_eq!(adam.learning_rate(), 0.001);

        let adamw = create_optimizer("adamw").unwrap();
        assert_eq!(adamw.learning_rate(), 0.001);

        let rmsprop = create_optimizer("rmsprop").unwrap();
        assert_eq!(rmsprop.learning_rate(), 0.001);

        let result = create_optimizer("unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_optimizer_from_params() {
        let sgd =
            create_optimizer_from_params("sgd", 0.05, Some(0.9), None, None, None, None).unwrap();
        assert_eq!(sgd.learning_rate(), 0.05);

        let adam = create_optimizer_from_params(
            "adam",
            0.01,
            None,
            Some(0.8),
            Some(0.99),
            Some(1e-6),
            None,
        )
        .unwrap();
        assert_eq!(adam.learning_rate(), 0.01);

        let adamw =
            create_optimizer_from_params("adamw", 0.02, None, None, None, None, Some(0.005))
                .unwrap();
        assert_eq!(adamw.learning_rate(), 0.02);

        let rmsprop =
            create_optimizer_from_params("rmsprop", 0.03, None, Some(0.85), None, Some(1e-7), None)
                .unwrap();
        assert_eq!(rmsprop.learning_rate(), 0.03);
    }
}

use crate::error::{Result, TurkError};
use crate::initializers::{Initializer, Xavier, HeNormal, Zeros};
use ndarray::{Array, Array2, Axis};
use rand::{thread_rng, Rng};
use serde::{Serialize, Deserialize};
use std::fmt;

/// Aktivasyon fonksiyonu türleri
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    Relu,
    LeakyRelu(f32),
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
    Elu(f32),
    Selu,
}

/// Layer trait'i - tüm katman türleri bu trait'i implemente etmelidir
pub trait Layer: fmt::Debug + Send + Sync {
    /// İleri yayılım
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>>;
    
    /// Geri yayılım
    fn backward(&mut self, grad: &Array2<f32>) -> Result<Array2<f32>>;
    
    /// Ağırlıkları döndürür (eğer varsa)
    fn get_weights(&self) -> Option<&Array2<f32>>;
    
    /// Bias değerlerini döndürür (eğer varsa)
    fn get_bias(&self) -> Option<&Array2<f32>>;
    
    /// Ağırlık gradyanlarını döndürür (eğer varsa)
    fn get_weight_grads(&self) -> Option<&Array2<f32>>;
    
    /// Bias gradyanlarını döndürür (eğer varsa)
    fn get_bias_grads(&self) -> Option<&Array2<f32>>;
    
    /// Ağırlıkları ve bias değerlerini optimizer kullanarak günceller
    fn update(&mut self) -> Result<()>;
    
    /// Aktivasyon fonksiyonunu döndürür
    fn activation_type(&self) -> Option<Activation>;
    
    /// Katman tipini döndürür
    fn layer_type(&self) -> &'static str;
    
    /// Giriş boyutunu döndürür
    fn input_size(&self) -> usize;
    
    /// Çıkış boyutunu döndürür
    fn output_size(&self) -> usize;
    
    /// Eğitilebilir parametrelerin sayısını döndürür
    fn param_count(&self) -> usize;
    
    /// Eğitim/çıkarım modunu ayarlar
    fn set_trainable(&mut self, trainable: bool);
    
    /// Katmanın eğitilebilir olup olmadığını döndürür
    fn is_trainable(&self) -> bool;
    
    /// Katmanı klonlar
    fn clone_box(&self) -> Box<dyn Layer>;
    
    /// Katmanı sıfırlar
    fn reset(&mut self);
}

/// Makro: şekil kontrolü yapar
macro_rules! check_shape {
    ($input:expr, $expected:expr, $name:expr) => {
        if $input.shape() != $expected.as_slice() {
            return Err(TurkError::ShapeMismatch(format!(
                "{} katmanı için beklenen şekil: {:?}, alınan şekil: {:?}",
                $name, $expected, $input.shape()
            )));
        }
    };
}

/// Tam bağlantılı katman
#[derive(Debug, Clone)]
pub struct Dense {
    weights: Array2<f32>,
    bias: Array2<f32>,
    input: Array2<f32>,
    output: Array2<f32>,
    z: Array2<f32>,  // aktivasyon öncesi değerler
    weight_grads: Array2<f32>,
    bias_grads: Array2<f32>,
    activation: Activation,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    trainable: bool,
}

impl Dense {
    /// Yeni bir tam bağlantılı katman oluşturur
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let initializer = match activation {
            Activation::Relu | Activation::LeakyRelu(_) | Activation::Elu(_) | Activation::Selu => {
                HeNormal::default()
            },
            _ => Xavier::default(),
        };
        
        let weights = initializer.initialize(&[input_size, output_size])
            .unwrap_or_else(|_| Array2::from_shape_fn((input_size, output_size), 
                            |_| thread_rng().gen_range::<f32, _>(-0.1..0.1)));
        
        let bias = Zeros.initialize(&[output_size, 1])
            .unwrap_or_else(|_| Array2::zeros((output_size, 1)));
            
        let weight_grads = Array2::zeros((input_size, output_size));
        let bias_grads = Array2::zeros((output_size, 1));

        Self {
            weights,
            bias,
            input: Array2::zeros((1, input_size)),
            output: Array2::zeros((1, output_size)),
            z: Array2::zeros((1, output_size)),
            weight_grads,
            bias_grads,
            activation,
            input_shape: vec![1, input_size],
            output_shape: vec![1, output_size],
            trainable: true,
        }
    }
    
    /// Özel ağırlık başlangıç değerleri ile yeni bir tam bağlantılı katman oluşturur
    pub fn with_initializer<I: Initializer>(
        input_size: usize, 
        output_size: usize, 
        activation: Activation,
        weight_initializer: &I,
        bias_initializer: &Zeros
    ) -> Result<Self> {
        let weights = weight_initializer.initialize(&[input_size, output_size])?;
        let bias = bias_initializer.initialize(&[output_size, 1])?;
        let weight_grads = Array2::zeros((input_size, output_size));
        let bias_grads = Array2::zeros((output_size, 1));

        Ok(Self {
            weights,
            bias,
            input: Array2::zeros((1, input_size)),
            output: Array2::zeros((1, output_size)),
            z: Array2::zeros((1, output_size)),
            weight_grads,
            bias_grads,
            activation,
            input_shape: vec![1, input_size],
            output_shape: vec![1, output_size],
            trainable: true,
        })
    }

    /// Aktivasyon fonksiyonu uygular
    fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            Activation::Relu => x.mapv(|v| v.max(0.0)),
            Activation::LeakyRelu(alpha) => x.mapv(|v| if v > 0.0 { v } else { alpha * v }),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Softmax => {
                let max_val = x.fold(f32::NEG_INFINITY, |max, &val| max.max(val));
                let exp = x.mapv(|v| (v - max_val).exp());
                let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
                exp / sum
            },
            Activation::Linear => x.clone(),
            Activation::Elu(alpha) => x.mapv(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }),
            Activation::Selu => {
                let alpha = 1.6732632423543772848170429916717;
                let scale = 1.0507009873554804934193349852946;
                x.mapv(|v| {
                    scale * if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }
                })
            }
        }
    }

    /// Aktivasyon fonksiyonunun türevini uygular
    fn activate_gradient(&self, x: &Array2<f32>, output: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            Activation::Relu => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::LeakyRelu(alpha) => x.mapv(|v| if v > 0.0 { 1.0 } else { alpha }),
            Activation::Sigmoid => output.mapv(|v| v * (1.0 - v)),
            Activation::Tanh => output.mapv(|v| 1.0 - v.powi(2)),
            Activation::Softmax => {
                // Softmax türevi matrisinin hesabı çok karmaşık, çapraz entropi kaybı ile birlikte
                // kullanıldığında genellikle sadece (output - target) formülü kullanılır.
                // Burada basit bir yaklaşım kullanacağız:
                Array2::ones(output.raw_dim())
            },
            Activation::Linear => Array2::ones(output.raw_dim()),
            Activation::Elu(alpha) => x.mapv(|v| if v > 0.0 { 1.0 } else { alpha * v.exp() }),
            Activation::Selu => {
                let alpha = 1.6732632423543772848170429916717;
                let scale = 1.0507009873554804934193349852946;
                x.mapv(|v| {
                    scale * if v > 0.0 { 1.0 } else { alpha * v.exp() }
                })
            }
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let batch_size = input.shape()[0];
        
        if input.shape()[1] != self.input_shape[1] {
            return Err(TurkError::ShapeMismatch(format!(
                "Dense katmanı için beklenen giriş boyutu: {}, alınan: {}",
                self.input_shape[1], input.shape()[1]
            )));
        }
        
        self.input = input.clone();
        self.input_shape[0] = batch_size;
        self.output_shape[0] = batch_size;
        
        // z = x * W^T + b
        self.z = input.dot(&self.weights) + &self.bias.t();
        self.output = self.activate(&self.z);
        
        Ok(self.output.clone())
    }

    fn backward(&mut self, grad: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.is_trainable() {
            // Eğitilebilir değilse gradyanları hesaplamadan geç
            return Ok(grad.dot(&self.weights.t()));
        }
        
        if grad.shape() != self.output.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Dense backward için beklenen gradyan şekli: {:?}, alınan: {:?}",
                self.output.shape(), grad.shape()
            )));
        }

        let activation_grad = self.activate_gradient(&self.z, &self.output);
        let delta = grad * &activation_grad;

        // Ağırlık ve bias gradyanlarını hesapla
        self.weight_grads = self.input.t().dot(&delta);
        self.bias_grads = delta.sum_axis(Axis(0)).insert_axis(Axis(1));

        // Önceki katman için gradyanları hesapla
        Ok(delta.dot(&self.weights.t()))
    }

    fn get_weights(&self) -> Option<&Array2<f32>> {
        Some(&self.weights)
    }

    fn get_bias(&self) -> Option<&Array2<f32>> {
        Some(&self.bias)
    }

    fn get_weight_grads(&self) -> Option<&Array2<f32>> {
        Some(&self.weight_grads)
    }

    fn get_bias_grads(&self) -> Option<&Array2<f32>> {
        Some(&self.bias_grads)
    }

    fn update(&mut self) -> Result<()> {
        // Ağırlık ve bias güncellemeleri optimizer tarafından yapılacak
        Ok(())
    }

    fn activation_type(&self) -> Option<Activation> {
        Some(self.activation)
    }
    
    fn layer_type(&self) -> &'static str {
        "dense"
    }

    fn input_size(&self) -> usize {
        self.input_shape[1]
    }

    fn output_size(&self) -> usize {
        self.output_shape[1]
    }
    
    fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
    
    fn set_trainable(&mut self, trainable: bool) {
        self.trainable = trainable;
    }
    
    fn is_trainable(&self) -> bool {
        self.trainable
    }
    
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    
    fn reset(&mut self) {
        // Gradyanları sıfırla
        self.weight_grads.fill(0.0);
        self.bias_grads.fill(0.0);
    }
}

impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let activation_str = match self.activation {
            Activation::Relu => "ReLU",
            Activation::LeakyRelu(alpha) => return write!(f, "Dense({} -> {}, LeakyReLU(alpha={}))", 
                self.input_shape[1], self.output_shape[1], alpha),
            Activation::Sigmoid => "Sigmoid",
            Activation::Tanh => "Tanh",
            Activation::Softmax => "Softmax",
            Activation::Linear => "Linear",
            Activation::Elu(alpha) => return write!(f, "Dense({} -> {}, ELU(alpha={}))", 
                self.input_shape[1], self.output_shape[1], alpha),
            Activation::Selu => "SELU",
        };
        
        write!(f, "Dense({} -> {}, {})", 
            self.input_shape[1], 
            self.output_shape[1], 
            activation_str)
    }
}

/// Dropout katmanı - eğitim sırasında rastgele nöronları devre dışı bırakır
#[derive(Debug, Clone)]
pub struct Dropout {
    rate: f32,
    mask: Array2<f32>,
    input_shape: Vec<usize>,
    trainable: bool,
}

impl Dropout {
    pub fn new(rate: f32) -> Self {
        if rate < 0.0 || rate >= 1.0 {
            panic!("Dropout oranı 0 ile 1 arasında olmalıdır (0 dahil, 1 hariç)");
        }
        
        Self {
            rate,
            mask: Array2::zeros((1, 1)),
            input_shape: vec![1, 1],
            trainable: true,
        }
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let batch_size = input.shape()[0];
        let feature_size = input.shape()[1];
        
        self.input_shape = vec![batch_size, feature_size];
        
        if self.is_trainable() {
            // Eğitim modu - dropout uygula
            let mut rng = thread_rng();
            let scale = 1.0 / (1.0 - self.rate);
            
            // Rastgele maske oluştur
            self.mask = Array::from_shape_fn((batch_size, feature_size), |_| {
                if rng.gen::<f32>() > self.rate { scale } else { 0.0 }
            });
            
            Ok(input * &self.mask)
        } else {
            // Çıkarım modu - doğrudan geçir
            Ok(input.clone())
        }
    }
    
    fn backward(&mut self, grad: &Array2<f32>) -> Result<Array2<f32>> {
        if grad.shape() != self.input_shape.as_slice() {
            return Err(TurkError::ShapeMismatch(format!(
                "Dropout backward için beklenen gradyan şekli: {:?}, alınan: {:?}",
                self.input_shape, grad.shape()
            )));
        }
        
        // Aynı maskeyi gradyanlara uygula
        Ok(grad * &self.mask)
    }
    
    fn get_weights(&self) -> Option<&Array2<f32>> {
        None
    }
    
    fn get_bias(&self) -> Option<&Array2<f32>> {
        None
    }
    
    fn get_weight_grads(&self) -> Option<&Array2<f32>> {
        None
    }
    
    fn get_bias_grads(&self) -> Option<&Array2<f32>> {
        None
    }
    
    fn update(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn activation_type(&self) -> Option<Activation> {
        None
    }
    
    fn layer_type(&self) -> &'static str {
        "dropout"
    }
    
    fn input_size(&self) -> usize {
        self.input_shape[1]
    }
    
    fn output_size(&self) -> usize {
        self.input_shape[1]
    }
    
    fn param_count(&self) -> usize {
        0
    }
    
    fn set_trainable(&mut self, trainable: bool) {
        self.trainable = trainable;
    }
    
    fn is_trainable(&self) -> bool {
        self.trainable
    }
    
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    
    fn reset(&mut self) {
        // Do nothing
    }
}

impl fmt::Display for Dropout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dropout(rate={})", self.rate)
    }
}

/// Toplu Normalizasyon (Batch Normalization) katmanı
#[derive(Debug, Clone)]
pub struct BatchNorm {
    gamma: Array2<f32>,     // ölçeklendirme parametresi
    beta: Array2<f32>,      // kaydırma parametresi
    input: Array2<f32>,
    normalized: Array2<f32>,
    xhat: Array2<f32>,      // normalize edilmiş giriş
    std: Array2<f32>,       // standart sapma
    var: Array2<f32>,       // varyans
    mean: Array2<f32>,      // ortalama
    gamma_grad: Array2<f32>,
    beta_grad: Array2<f32>,
    epsilon: f32,           // sayısal stabilite için küçük değer
    momentum: f32,          // hareketli ortalama için momentum
    moving_mean: Array2<f32>,
    moving_var: Array2<f32>,
    input_shape: Vec<usize>,
    trainable: bool,
}

impl BatchNorm {
    pub fn new(feature_size: usize, epsilon: Option<f32>, momentum: Option<f32>) -> Self {
        let epsilon = epsilon.unwrap_or(1e-5);
        let momentum = momentum.unwrap_or(0.99);
        
        Self {
            gamma: Array2::ones((1, feature_size)),
            beta: Array2::zeros((1, feature_size)),
            input: Array2::zeros((1, feature_size)),
            normalized: Array2::zeros((1, feature_size)),
            xhat: Array2::zeros((1, feature_size)),
            std: Array2::zeros((1, feature_size)),
            var: Array2::zeros((1, feature_size)),
            mean: Array2::zeros((1, feature_size)),
            gamma_grad: Array2::zeros((1, feature_size)),
            beta_grad: Array2::zeros((1, feature_size)),
            epsilon,
            momentum,
            moving_mean: Array2::zeros((1, feature_size)),
            moving_var: Array2::zeros((1, feature_size)),
            input_shape: vec![1, feature_size],
            trainable: true,
        }
    }
}

impl Layer for BatchNorm {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let batch_size = input.shape()[0];
        let feature_size = input.shape()[1];
        
        if feature_size != self.input_shape[1] {
            return Err(TurkError::ShapeMismatch(format!(
                "BatchNorm katmanı için beklenen özellik boyutu: {}, alınan: {}",
                self.input_shape[1], feature_size
            )));
        }
        
        self.input = input.clone();
        self.input_shape[0] = batch_size;
        
        if self.is_trainable() {
            // Eğitim modu
            // Batch ortalaması ve varyansını hesapla
            self.mean = input.mean_axis(Axis(0)).insert_axis(Axis(0));
            self.var = input.var_axis(Axis(0), 0.0).insert_axis(Axis(0));
            self.std = (&self.var + self.epsilon).mapv(f32::sqrt);
            
            // Hareketli ortalamaları güncelle
            if self.moving_mean.shape()[1] != feature_size {
                self.moving_mean = Array2::zeros((1, feature_size));
            }
            if self.moving_var.shape()[1] != feature_size {
                self.moving_var = Array2::zeros((1, feature_size));
            }
            
            self.moving_mean = &self.moving_mean * self.momentum + &self.mean * (1.0 - self.momentum);
            self.moving_var = &self.moving_var * self.momentum + &self.var * (1.0 - self.momentum);
            
            // Normalize et
            self.xhat = (input - &self.mean) / &self.std;
            self.normalized = &self.xhat * &self.gamma + &self.beta;
        } else {
            // Çıkarım modu
            self.normalized = (input - &self.moving_mean) / (&self.moving_var + self.epsilon).mapv(f32::sqrt) * &self.gamma + &self.beta;
        }
        
        Ok(self.normalized.clone())
    }
    
    fn backward(&mut self, grad: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.is_trainable() {
            // Eğitim modunda değilse
            return Ok(grad.clone());
        }
        
        let batch_size = self.input.shape()[0] as f32;
        
        if grad.shape() != self.normalized.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "BatchNorm backward için beklenen gradyan şekli: {:?}, alınan: {:?}",
                self.normalized.shape(), grad.shape()
            )));
        }
        
        // Parametrelere göre gradyanları hesapla
        self.beta_grad = grad.sum_axis(Axis(0)).insert_axis(Axis(0));
        self.gamma_grad = (grad * &self.xhat).sum_axis(Axis(0)).insert_axis(Axis(0));
        
        // Girişe göre gradyanı hesapla (zincir kuralı)
        let dxhat = grad * &self.gamma;
        let dvar = -0.5 * &(dxhat * (&self.input - &self.mean) * (&self.var + self.epsilon).mapv(|v| v.powf(-1.5))).sum_axis(Axis(0)).insert_axis(Axis(0));
        let dmean = -(&dxhat / &self.std).sum_axis(Axis(0)).insert_axis(Axis(0)) - 2.0 * dvar * (&self.input - &self.mean).mean_axis(Axis(0)).insert_axis(Axis(0));
        
        let dx = dxhat / &self.std + 2.0 * dvar * (&self.input - &self.mean) / batch_size + dmean / batch_size;
        
        Ok(dx)
    }
    
    fn get_weights(&self) -> Option<&Array2<f32>> {
        Some(&self.gamma)
    }
    
    fn get_bias(&self) -> Option<&Array2<f32>> {
        Some(&self.beta)
    }
    
    fn get_weight_grads(&self) -> Option<&Array2<f32>> {
        Some(&self.gamma_grad)
    }
    
    fn get_bias_grads(&self) -> Option<&Array2<f32>> {
        Some(&self.beta_grad)
    }
    
    fn update(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn activation_type(&self) -> Option<Activation> {
        None
    }
    
    fn layer_type(&self) -> &'static str {
        "batch_norm"
    }
    
    fn input_size(&self) -> usize {
        self.input_shape[1]
    }
    
    fn output_size(&self) -> usize {
        self.input_shape[1]
    }
    
    fn param_count(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }
    
    fn set_trainable(&mut self, trainable: bool) {
        self.trainable = trainable;
    }
    
    fn is_trainable(&self) -> bool {
        self.trainable
    }
    
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    
    fn reset(&mut self) {
        self.gamma_grad.fill(0.0);
        self.beta_grad.fill(0.0);
    }
}

impl fmt::Display for BatchNorm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BatchNorm({})", self.input_shape[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_dense_layer() {
        let mut layer = Dense::new(784, 128, Activation::Relu);
        
        // Test input
        let mut rng = rand::thread_rng();
        let input = Array2::from_shape_fn((5, 784), |_| {
            rng.gen_range::<f32, _>(-1.0..1.0)
        });
        
        // Forward pass
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[5, 128]);
        
        // Test shape mismatch
        let wrong_input = Array2::from_shape_fn((5, 100), |_| {
            rng.gen_range::<f32, _>(-1.0..1.0)
        });
        assert!(layer.forward(&wrong_input).is_err());
    }

    #[test]
    fn test_dropout_layer() {
        let mut dropout = Dropout::new(0.5);
        
        // Test input
        let input = Array2::ones((10, 20));
        
        // Forward pass (training mode)
        let output_train = dropout.forward(&input).unwrap();
        assert_eq!(output_train.shape(), &[10, 20]);
        
        // Forward pass (inference mode)
        dropout.set_trainable(false);
        let output_inference = dropout.forward(&input).unwrap();
        assert_eq!(output_inference, input);
        
        // Backward pass
        let grad = Array2::ones((10, 20));
        let grad_back = dropout.backward(&grad).unwrap();
        assert_eq!(grad_back.shape(), &[10, 20]);
    }

    #[test]
    fn test_batch_norm_layer() {
        let mut batch_norm = BatchNorm::new(10, None, None);
        
        // Test input
        let input = Array2::from_shape_fn((5, 10), |_| {
            thread_rng().gen_range::<f32, _>(-1.0..1.0)
        });
        
        // Forward pass (training mode)
        let output_train = batch_norm.forward(&input).unwrap();
        assert_eq!(output_train.shape(), &[5, 10]);
        
        // Calculate mean and variance of output
        let output_mean = output_train.mean_axis(Axis(0));
        let output_var = output_train.var_axis(Axis(0), 0.0);
        
        // Mean should be close to 0 and variance close to 1
        for mean in output_mean.iter() {
            assert!(mean.abs() < 1e-5);
        }
        
        // Forward pass (inference mode)
        batch_norm.set_trainable(false);
        let output_inference = batch_norm.forward(&input).unwrap();
        assert_eq!(output_inference.shape(), &[5, 10]);
        
        // Backward pass
        batch_norm.set_trainable(true);
        let grad = Array2::ones((5, 10));
        let _ = batch_norm.forward(&input);
        let grad_back = batch_norm.backward(&grad).unwrap();
        assert_eq!(grad_back.shape(), &[5, 10]);
    }

    #[test]
    fn test_dense_activations() {
        let activations = [
            Activation::Relu,
            Activation::LeakyRelu(0.1),
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::Linear,
            Activation::Elu(1.0),
            Activation::Selu,
        ];
        
        for activation in activations.iter() {
            let mut layer = Dense::new(5, 3, *activation);
            let input = Array2::from_shape_fn((2, 5), |_| {
                thread_rng().gen_range::<f32, _>(-1.0..1.0)
            });
            
            let output = layer.forward(&input).unwrap();
            assert_eq!(output.shape(), &[2, 3]);
            
            let grad = Array2::ones((2, 3));
            let grad_back = layer.backward(&grad).unwrap();
            assert_eq!(grad_back.shape(), &[2, 5]);
        }
    }
} 
use crate::error::{Result, TurkError};
use crate::initializers::{Initializer, HeNormal, Xavier, Zeros};
use ndarray::{Array2, Axis, s};
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

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Activation::Relu => write!(f, "ReLU"),
            Activation::LeakyRelu(alpha) => write!(f, "LeakyReLU({})", alpha),
            Activation::Sigmoid => write!(f, "Sigmoid"),
            Activation::Tanh => write!(f, "Tanh"),
            Activation::Softmax => write!(f, "Softmax"),
            Activation::Linear => write!(f, "Linear"),
            Activation::Elu(alpha) => write!(f, "ELU({})", alpha),
            Activation::Selu => write!(f, "SELU"),
        }
    }
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
    fn activation_type(&self) -> Activation;
    
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
    
    /// Ağırlıkları vektör olarak döndürür (serileştirme için)
    fn weights_as_vec(&self) -> Vec<f32> {
        if let Some(weights) = self.get_weights() {
            weights.iter().cloned().collect()
        } else {
            Vec::new()
        }
    }
    
    /// Bias değerlerini vektör olarak döndürür (serileştirme için)
    fn bias_as_vec(&self) -> Vec<f32> {
        if let Some(bias) = self.get_bias() {
            bias.iter().cloned().collect()
        } else {
            Vec::new()
        }
    }
    
    /// Dropout oranını döndürür (eğer varsa)
    fn dropout_rate(&self) -> Option<f32> {
        None
    }
    
    /// Batch normalizasyon kullanılıp kullanılmadığını döndürür
    fn use_batch_norm(&self) -> Option<bool> {
        None
    }
    
    /// Ağırlık değerlerini ayarlar
    fn set_weights(&mut self, weights: Array2<f32>) -> Result<()>;
    
    /// Bias değerlerini ayarlar
    fn set_bias(&mut self, bias: Array2<f32>) -> Result<()>;
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
        let initializer: Box<dyn Initializer> = match activation {
            Activation::Relu | Activation::LeakyRelu(_) | Activation::Elu(_) | Activation::Selu => {
                Box::new(HeNormal::default())
            },
            _ => Box::new(Xavier::default()),
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

    /// İleri yayılımı optimize edilmiş şekilde gerçekleştirir
    ///
    /// Bu metod, matris çarpımları için BLAS operasyonlarını kullanarak
    /// ve büyük batch boyutları için paralelleştirme yaparak performansı artırır.
    fn optimize_forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
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
        
        // Büyük batch'ler için paralelleştirme
        if batch_size > 32 {
            // Paralelleştirme için rayon kullan
            use rayon::prelude::*;
            
            // Önce lineer dönüşüm (dot product) uygula
            // Her satır için ayrı thread'de işlem yapılabilir
            let batch_results: Vec<Array2<f32>> = (0..batch_size)
                .into_par_iter()  // Parallel iterator
                .map(|batch_idx| {
                    // Tek bir örnek için işlem
                    let single_input = input.slice(s![batch_idx..batch_idx+1, ..]);
                    
                    // Matris çarpımı
                    let z = single_input.dot(&self.weights) + &self.bias.t();
                    
                    // Sonucu döndür
                    z
                })
                .collect();
            
            // Sonuçları birleştir
            let mut z = Array2::zeros((batch_size, self.output_shape[1]));
            for (idx, batch_result) in batch_results.iter().enumerate() {
                z.slice_mut(s![idx..idx+1, ..]).assign(batch_result);
            }
            
            self.z = z;
        } else {
            // Küçük batch'ler için standart matris çarpımı
            // BLAS operasyonları ndarray'in dot metodu tarafından kullanılır
            self.z = input.dot(&self.weights) + &self.bias.t();
        }
        
        // Aktivasyon fonksiyonunu uygula
        self.output = self.activate(&self.z);
        
        Ok(self.output.clone())
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // Optimize edilmiş forward metodu kullan
        self.optimize_forward(input)
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

    fn activation_type(&self) -> Activation {
        self.activation
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
        let initializer: Box<dyn Initializer> = match self.activation {
            Activation::Relu | Activation::LeakyRelu(_) | Activation::Elu(_) | Activation::Selu => {
                Box::new(HeNormal::default())
            },
            _ => Box::new(Xavier::default()),
        };
        
        if let Ok(weights) = initializer.initialize(&[self.input_size(), self.output_size()]) {
            self.weights = weights;
        }
        
        self.bias = Array2::zeros((self.output_size(), 1));
        self.weight_grads = Array2::zeros(self.weights.raw_dim());
        self.bias_grads = Array2::zeros(self.bias.raw_dim());
    }
    
    fn set_weights(&mut self, weights: Array2<f32>) -> Result<()> {
        if weights.shape() != self.weights.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Ağırlık şekli uyuşmazlığı: beklenen {:?}, alınan {:?}",
                self.weights.shape(), weights.shape()
            )));
        }
        self.weights = weights;
        Ok(())
    }
    
    fn set_bias(&mut self, bias: Array2<f32>) -> Result<()> {
        if bias.shape() != self.bias.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "Bias şekli uyuşmazlığı: beklenen {:?}, alınan {:?}",
                self.bias.shape(), bias.shape()
            )));
        }
        self.bias = bias;
        Ok(())
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
    output_shape: Vec<usize>,
    trainable: bool,
    training: bool,
    input: Array2<f32>,
    output: Array2<f32>,
}

impl Dropout {
    /// Yeni bir dropout katmanı oluşturur
    pub fn new(rate: f32) -> Self {
        let rate = rate.max(0.0).min(1.0); // 0-1 arasında olduğundan emin ol
        
        Self {
            rate,
            mask: Array2::zeros((1, 1)),
            input_shape: vec![1, 1],
            output_shape: vec![1, 1],
            trainable: false, // Dropout eğitilebilir parametre içermez
            training: true,   // Varsayılan olarak eğitim modunda
            input: Array2::zeros((1, 1)),
            output: Array2::zeros((1, 1)),
        }
    }
    
    /// Eğitim modunu ayarlar
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        self.input = input.clone();
        self.input_shape = input.shape().to_vec();
        self.output_shape = input.shape().to_vec();

        if self.training {
            let mut rng = thread_rng();
            
            // Yeni mask oluştur
            let input_shape_array = [input.shape()[0], input.shape()[1]];
            self.mask = Array2::from_shape_fn(input_shape_array, |_| {
                if rng.gen::<f32>() > self.rate { 1.0 / (1.0 - self.rate) } else { 0.0 }
            });
            
            // Dropout mask uygula
            let output = input * &self.mask;
            self.output = output.clone();
            Ok(output)
        } else {
            // Test modunda mask uygulanmaz
            self.output = input.clone();
            Ok(input.clone())
        }
    }
    
    fn backward(&mut self, grad: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.training {
            return Ok(grad.clone());
        }
        
        // Aynı maskeyi geriye de uygula
        Ok(grad * &self.mask)
    }
    
    fn get_weights(&self) -> Option<&Array2<f32>> {
        None // Dropout'un ağırlığı yok
    }
    
    fn get_bias(&self) -> Option<&Array2<f32>> {
        None // Dropout'un bias'ı yok
    }
    
    fn get_weight_grads(&self) -> Option<&Array2<f32>> {
        None
    }
    
    fn get_bias_grads(&self) -> Option<&Array2<f32>> {
        None
    }
    
    fn update(&mut self) -> Result<()> {
        // Dropout'un güncellenecek parametresi yok
        Ok(())
    }
    
    fn activation_type(&self) -> Activation {
        Activation::Linear // Dropout'un aktivasyonu yok
    }
    
    fn layer_type(&self) -> &'static str {
        "dropout"
    }
    
    fn input_size(&self) -> usize {
        if self.input_shape.len() > 1 {
            self.input_shape[1]
        } else {
            0 // veya uygun bir varsayılan değer
        }
    }
    
    fn output_size(&self) -> usize {
        if self.output_shape.len() > 1 {
            self.output_shape[1]
        } else {
            0 // veya uygun bir varsayılan değer
        }
    }
    
    fn param_count(&self) -> usize {
        0 // Dropout'un parametresi yok
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
        // Dropout için reset işlemi yok
    }
    
    fn dropout_rate(&self) -> Option<f32> {
        Some(self.rate)
    }
    
    fn set_weights(&mut self, _weights: Array2<f32>) -> Result<()> {
        Err(TurkError::InvalidLayer("Dropout katmanının ağırlığı yoktur".to_string()))
    }
    
    fn set_bias(&mut self, _bias: Array2<f32>) -> Result<()> {
        Err(TurkError::InvalidLayer("Dropout katmanının bias'ı yoktur".to_string()))
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
    gamma: Array2<f32>,        // Ölçekleme parametresi
    beta: Array2<f32>,         // Kaydırma parametresi
    gamma_grad: Array2<f32>,   // Gamma gradyanı
    beta_grad: Array2<f32>,    // Beta gradyanı
    running_mean: Array2<f32>, // Koşu ortalaması (eğitim modunda güncellenir)
    running_var: Array2<f32>,  // Koşu varyansı (eğitim modunda güncellenir)
    momentum: f32,             // Koşu istatistiklerini güncelleme hızı
    epsilon: f32,              // Sayısal stabilite için küçük sabit
    input: Array2<f32>,        // Giriş değeri
    norm_input: Array2<f32>,   // Normalize edilmiş giriş
    std: Array2<f32>,          // Standart sapma
    var: Array2<f32>,          // Batch varyansı
    mean: Array2<f32>,         // Batch ortalaması
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    trainable: bool,
    training: bool,
}

impl BatchNorm {
    /// Yeni bir batch normalizasyon katmanı oluşturur
    pub fn new(features: usize) -> Self {
        Self {
            gamma: Array2::ones((1, features)),
            beta: Array2::zeros((1, features)),
            gamma_grad: Array2::zeros((1, features)),
            beta_grad: Array2::zeros((1, features)),
            running_mean: Array2::zeros((1, features)),
            running_var: Array2::ones((1, features)),
            momentum: 0.9,
            epsilon: 1e-5,
            input: Array2::zeros((1, features)),
            norm_input: Array2::zeros((1, features)),
            std: Array2::zeros((1, features)),
            var: Array2::zeros((1, features)),
            mean: Array2::zeros((1, features)),
            input_shape: vec![1, features],
            output_shape: vec![1, features],
            trainable: true,
            training: true,
        }
    }
    
    /// Eğitim modunu ayarlar
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Layer for BatchNorm {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let batch_size = input.shape()[0];
        let features = input.shape()[1];
        
        if features != self.input_shape[1] {
            return Err(TurkError::ShapeMismatch(format!(
                "BatchNorm katmanı için beklenen özellik sayısı: {}, alınan: {}",
                self.input_shape[1], features
            )));
        }
        
        self.input = input.clone();
        self.input_shape[0] = batch_size;
        self.output_shape[0] = batch_size;
        
        if self.training {
            // Option<T> kullanarak unwrap hatalarını önle
            if let Some(mean) = input.mean_axis(Axis(0)) {
                self.mean = mean.insert_axis(Axis(0));
                
                let centered = input - &self.mean;
                if let Some(var) = centered.mapv(|x| x * x).mean_axis(Axis(0)) {
                    self.var = var.insert_axis(Axis(0));
                    self.std = (&self.var + self.epsilon).mapv(f32::sqrt);
                    
                    self.running_mean = &self.running_mean * self.momentum + &self.mean * (1.0 - self.momentum);
                    self.running_var = &self.running_var * self.momentum + &self.var * (1.0 - self.momentum);
                    
                    self.norm_input = (input - &self.mean) / &self.std;
                } else {
                    return Err(TurkError::ModelError("Varyans hesaplanamadı".to_string()));
                }
            } else {
                return Err(TurkError::ModelError("Ortalama hesaplanamadı".to_string()));
            }
        } else {
            self.norm_input = (input - &self.running_mean) / (&self.running_var + self.epsilon).mapv(f32::sqrt);
        }
        
        Ok(&self.norm_input * &self.gamma + &self.beta)
    }
    
    fn backward(&mut self, grad: &Array2<f32>) -> Result<Array2<f32>> {
        if !self.is_trainable() || !self.training {
            // Eğitilebilir değilse veya test modundaysa basitçe gradyanı geçir
            return Ok(grad.clone());
        }
        
        let batch_size = grad.shape()[0] as f32;
        
        // Gamma ve beta için gradyanları hesapla
        self.beta_grad = grad.sum_axis(Axis(0)).insert_axis(Axis(0));
        self.gamma_grad = (grad * &self.norm_input.clone()).sum_axis(Axis(0)).insert_axis(Axis(0));
        
        // Normalize edilmiş giriş için gradyan
        let dnorm = grad * &self.gamma.clone();
        
        // Varyans için gradyan
        let dvar_term1 = dnorm.clone() * &(&self.input - &self.mean);
        let dvar_term2 = -0.5 * (&self.std).mapv(|x| x.powi(-3));
        let dvar = (dvar_term1 * dvar_term2).sum_axis(Axis(0)).insert_axis(Axis(0));
        
        // Ortalama için gradyan
        let dmu1 = (dnorm.clone() * (-1.0 / &self.std)).sum_axis(Axis(0)).insert_axis(Axis(0));
        let dmu2_term1 = -2.0 * &(&self.input - &self.mean);
        let dmu2 = (dvar.clone() * dmu2_term1).sum_axis(Axis(0)).insert_axis(Axis(0)) / batch_size;
        let dmu = dmu1 + dmu2;
        
        // Giriş için gradyan
        let dx1 = dnorm / &self.std;
        let dx2 = dvar * (2.0 * &(&self.input - &self.mean)) / batch_size;
        let dx3 = dmu / batch_size;
        
        let dx = dx1 + dx2 + dx3;
        
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
        // BatchNorm güncellemesi optimizer'a bırakılır
        Ok(())
    }
    
    fn activation_type(&self) -> Activation {
        Activation::Linear // BatchNorm'un aktivasyonu yok
    }
    
    fn layer_type(&self) -> &'static str {
        "batch_norm"
    }
    
    fn input_size(&self) -> usize {
        self.input_shape[1]
    }
    
    fn output_size(&self) -> usize {
        self.output_shape[1]
    }
    
    fn param_count(&self) -> usize {
        // Gamma + Beta
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
        self.gamma = Array2::ones((1, self.input_shape[1]));
        self.beta = Array2::zeros((1, self.input_shape[1]));
        self.gamma_grad = Array2::zeros((1, self.input_shape[1]));
        self.beta_grad = Array2::zeros((1, self.input_shape[1]));
        self.running_mean = Array2::zeros((1, self.input_shape[1]));
        self.running_var = Array2::ones((1, self.input_shape[1]));
    }
    
    fn use_batch_norm(&self) -> Option<bool> {
        Some(true)
    }
    
    fn set_weights(&mut self, weights: Array2<f32>) -> Result<()> {
        if weights.shape() != self.gamma.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "BatchNorm gamma şekli uyuşmazlığı: beklenen {:?}, alınan {:?}",
                self.gamma.shape(), weights.shape()
            )));
        }
        self.gamma = weights;
        Ok(())
    }
    
    fn set_bias(&mut self, bias: Array2<f32>) -> Result<()> {
        if bias.shape() != self.beta.shape() {
            return Err(TurkError::ShapeMismatch(format!(
                "BatchNorm beta şekli uyuşmazlığı: beklenen {:?}, alınan {:?}",
                self.beta.shape(), bias.shape()
            )));
        }
        self.beta = bias;
        Ok(())
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
    use ndarray::Axis;

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
        dropout.set_training(false);
        let output_inference = dropout.forward(&input).unwrap();
        assert_eq!(output_inference, input);
        
        // Backward pass
        let grad = Array2::ones((10, 20));
        let grad_back = dropout.backward(&grad).unwrap();
        assert_eq!(grad_back.shape(), &[10, 20]);
    }

    #[test]
    fn test_batch_norm_layer() {
        let mut batch_norm = BatchNorm::new(10);
        
        // Test input
        let input = Array2::from_shape_fn((5, 10), |_| {
            thread_rng().gen_range::<f32, _>(-1.0..1.0)
        });
        
        // Forward pass (training mode)
        let output_train = batch_norm.forward(&input).unwrap();
        assert_eq!(output_train.shape(), &[5, 10]);
        
        // Test that output is normalized (test one by one to avoid problem with array types)
        let output_mean = output_train.mean_axis(Axis(0)).expect("Mean hesaplanamadı");
        for i in 0..output_mean.shape()[0] {
            assert!(output_mean[i].abs() < 1e-5);
        }
        
        let output_var = output_train.var_axis(Axis(0), 0.0);
        for i in 0..output_var.shape()[0] {
            assert!((output_var[i] - 1.0).abs() < 1e-4);
        }
        
        // Forward pass (inference mode)
        batch_norm.set_training(false);
        let output_inference = batch_norm.forward(&input).unwrap();
        assert_eq!(output_inference.shape(), &[5, 10]);
        
        // Backward pass
        batch_norm.set_training(true);
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
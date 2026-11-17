use crate::aktivasyon::Activation;
use crate::error::{Result, TurkError};
use crate::initializers::{GlorotUniform, Initializer, Zeros};
use log::{debug, warn};
use ndarray::{Array2, ArrayView2, Axis, Zip};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Katman özellikleri
pub trait Layer: fmt::Debug + Send + Sync {
    /// İleri yayılım
    fn forward(&mut self, input: &ArrayView2<f64>) -> Result<Array2<f64>>;

    /// Geri yayılım
    fn backward(
        &self,
        input: &ArrayView2<f64>,
        grad_output: &ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)>;

    /// Ağırlıkları ve bias değerlerini günceller
    fn update(
        &mut self,
        grad_weights: &ArrayView2<f64>,
        grad_biases: &ArrayView2<f64>,
        learning_rate: f64,
    ) -> Result<()>;

    /// Ağırlıkları döndürür
    fn get_weights(&self) -> Result<Array2<f64>>;

    /// Bias değerlerini döndürür
    fn get_biases(&self) -> Result<Array2<f64>>;

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
    fn weights_as_vec(&self) -> Vec<f64>;

    /// Bias değerlerini vektör olarak döndürür (serileştirme için)
    fn biases_as_vec(&self) -> Vec<f64>;

    /// Dropout oranını döndürür (eğer varsa)
    fn dropout_rate(&self) -> Option<f64>;

    /// Batch normalizasyon kullanılıp kullanılmadığını döndürür
    fn use_batch_norm(&self) -> Option<bool>;

    /// Ağırlık değerlerini ayarlar
    fn set_weights(&mut self, weights: &ArrayView2<f64>) -> Result<()>;

    /// Bias değerlerini ayarlar
    fn set_biases(&mut self, biases: &ArrayView2<f64>) -> Result<()>;

    /// Display için kullanılacak metod
    fn to_string(&self) -> String {
        format!("Layer: {}", self.layer_type())
    }

    /// Serileştirme için ek sayısal metadata (CNN boyutları vb.)
    fn layer_metadata(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

impl Clone for Box<dyn Layer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Tam bağlantılı (Dense) katman
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dense {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: Activation,
    pub input_shape: Option<(usize, usize)>,
    pub output_shape: Option<(usize, usize)>,
    pub kernel_initializer: String,
    pub bias_initializer: String,
    pub dropout_rate: f64,
    pub trainable: bool,
    pub batch_norm: bool,
}

impl Dense {
    /// Yeni bir Dense katmanı oluşturur
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        kernel_initializer: Box<dyn Initializer>,
        bias_initializer: Box<dyn Initializer>,
        use_bias: bool,
    ) -> Result<Self> {
        let weights = kernel_initializer.initialize((input_size, output_size))?;
        let biases = if use_bias {
            bias_initializer.initialize((1, output_size))?
        } else {
            Array2::zeros((1, output_size))
        };

        Ok(Self {
            weights,
            biases,
            activation,
            input_shape: None,
            output_shape: None,
            kernel_initializer: "glorot_uniform".to_string(),
            bias_initializer: "zeros".to_string(),
            dropout_rate: 0.0,
            trainable: true,
            batch_norm: false,
        })
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &ArrayView2<f64>) -> Result<Array2<f64>> {
        // Boyut kontrolü
        if input.ncols() != self.weights.nrows() {
            return Err(TurkError::shape_mismatch(&format!(
                "Dense katmanı forward geçişinde boyut uyuşmazlığı: giriş = {}, ağırlıklar = {}",
                input.ncols(),
                self.weights.nrows()
            )));
        }

        // X * W + b - ndarray-rayon özelliği etkinleştirildiğinde, dot işlemi otomatik olarak paralelleştirilir
        let output = input.dot(&self.weights) + &self.biases;

        // Aktivasyon
        let activated = self.activation.forward(&output);

        Ok(activated)
    }

    fn backward(
        &self,
        input: &ArrayView2<f64>,
        grad_output: &ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        // Aktivasyon girdisi hesapla - paralel dot
        let z = input.dot(&self.weights) + &self.biases;

        // Aktivasyon türevi - paralelleştirelim
        let mut dz = Array2::zeros(z.dim());

        match self.activation {
            // Basit aktivasyonlar için doğrudan paralelleştirilmiş hesaplama
            Activation::Relu => {
                Zip::from(&mut dz).and(&z).and(grad_output).for_each(
                    |dz_val, &z_val, &grad_val| {
                        *dz_val = if z_val > 0.0 { grad_val } else { 0.0 };
                    },
                );
            }
            Activation::Sigmoid => {
                Zip::from(&mut dz).and(&z).and(grad_output).for_each(
                    |dz_val, &z_val, &grad_val| {
                        let sigmoid = 1.0 / (1.0 + (-z_val).exp());
                        *dz_val = sigmoid * (1.0 - sigmoid) * grad_val;
                    },
                );
            }
            Activation::Tanh => {
                Zip::from(&mut dz).and(&z).and(grad_output).for_each(
                    |dz_val, &z_val, &grad_val| {
                        let tanh_val = z_val.tanh();
                        *dz_val = (1.0 - tanh_val * tanh_val) * grad_val;
                    },
                );
            }
            Activation::Linear => {
                dz.assign(grad_output);
            }
            // Diğer aktivasyonlar için normal hesaplayıp sonra çarpalım
            _ => {
                let activation_grad = self.activation.backward(&z);
                Zip::from(&mut dz)
                    .and(&activation_grad)
                    .and(grad_output)
                    .for_each(|dz_val, &act_grad, &grad_val| {
                        *dz_val = act_grad * grad_val;
                    });
            }
        }

        // Gradyanlar - paralel dot işlemleri
        let grad_weights = input.t().dot(&dz);
        let grad_biases = dz.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Giriş gradyanı - paralel dot
        let grad_input = dz.dot(&self.weights.t());

        Ok((grad_input, grad_weights, grad_biases))
    }

    fn update(
        &mut self,
        grad_weights: &ArrayView2<f64>,
        grad_biases: &ArrayView2<f64>,
        learning_rate: f64,
    ) -> Result<()> {
        if !self.is_trainable() {
            return Ok(());
        }

        // Boyut kontrolü
        if grad_weights.dim() != self.weights.dim() {
            return Err(TurkError::shape_mismatch(&format!(
                "Ağırlık güncelleme boyut uyuşmazlığı: gradyan = {:?}, ağırlıklar = {:?}",
                grad_weights.dim(),
                self.weights.dim()
            )));
        }

        if grad_biases.dim() != self.biases.dim() {
            return Err(TurkError::shape_mismatch(&format!(
                "Bias güncelleme boyut uyuşmazlığı: gradyan = {:?}, bias = {:?}",
                grad_biases.dim(),
                self.biases.dim()
            )));
        }

        // Güncelleme (SGD)
        self.weights -= &(learning_rate * grad_weights);
        self.biases -= &(learning_rate * grad_biases);

        Ok(())
    }

    fn get_weights(&self) -> Result<Array2<f64>> {
        Ok(self.weights.clone())
    }

    fn get_biases(&self) -> Result<Array2<f64>> {
        Ok(self.biases.clone())
    }

    fn activation_type(&self) -> Activation {
        self.activation
    }

    fn layer_type(&self) -> &'static str {
        "Dense"
    }

    fn input_size(&self) -> usize {
        self.weights.nrows()
    }

    fn output_size(&self) -> usize {
        self.weights.ncols()
    }

    fn param_count(&self) -> usize {
        self.weights.len() + self.biases.len()
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
        debug!("Dense katmanı sıfırlanıyor");
        let kernel_initializer = Box::new(GlorotUniform::new());
        let bias_initializer = Box::new(Zeros::new());

        if let Ok(weights) =
            kernel_initializer.initialize((self.weights.nrows(), self.weights.ncols()))
        {
            self.weights = weights;
        }

        if let Ok(biases) = bias_initializer.initialize((1, self.biases.ncols())) {
            self.biases = biases;
        }
    }

    fn weights_as_vec(&self) -> Vec<f64> {
        self.weights.iter().cloned().collect()
    }

    fn biases_as_vec(&self) -> Vec<f64> {
        self.biases.iter().cloned().collect()
    }

    fn dropout_rate(&self) -> Option<f64> {
        if self.dropout_rate > 0.0 {
            Some(self.dropout_rate)
        } else {
            None
        }
    }

    fn use_batch_norm(&self) -> Option<bool> {
        None
    }

    fn set_weights(&mut self, weights: &ArrayView2<f64>) -> Result<()> {
        if weights.dim() != self.weights.dim() {
            return Err(TurkError::shape_mismatch(&format!(
                "Ağırlık ayarlama boyut uyuşmazlığı: yeni = {:?}, mevcut = {:?}",
                weights.dim(),
                self.weights.dim()
            )));
        }

        self.weights.assign(weights);
        Ok(())
    }

    fn set_biases(&mut self, biases: &ArrayView2<f64>) -> Result<()> {
        if biases.dim() != self.biases.dim() {
            return Err(TurkError::shape_mismatch(&format!(
                "Bias ayarlama boyut uyuşmazlığı: yeni = {:?}, mevcut = {:?}",
                biases.dim(),
                self.biases.dim()
            )));
        }

        self.biases.assign(biases);
        Ok(())
    }
}

impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Dense({}, {}, {})",
            self.weights.nrows(),
            self.weights.ncols(),
            self.activation
        )
    }
}

/// Dropout katmanı
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dropout {
    pub rate: f64,
    pub mask: Option<Array2<f64>>,
    pub input_shape: Option<(usize, usize)>,
    pub output_shape: Option<(usize, usize)>,
    pub trainable: bool,
    pub training: bool,
}

impl Dropout {
    /// Yeni bir Dropout katmanı oluşturur
    pub fn new(rate: f64) -> Self {
        if !(0.0..1.0).contains(&rate) {
            warn!(
                "Dropout oranı [0, 1) aralığında olmalıdır, alınan: {}",
                rate
            );
        }

        let rate = rate.clamp(0.0, 0.999);

        Self {
            rate,
            mask: None,
            input_shape: None,
            output_shape: None,
            trainable: false,
            training: true,
        }
    }

    /// Eğitim modunu ayarlar
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if !self.training || self.rate <= 0.0 {
            return Ok(input.to_owned());
        }

        use ndarray_rand::{rand_distr::Uniform, RandomExt};

        // Rastgele maske oluştur
        let mut mask = Array2::random(input.dim(), Uniform::new(0.0, 1.0));

        mask.mapv_inplace(|v| if v > self.rate { 1.0 } else { 0.0 });

        // Ölçeklendirme (eğitim ve çıkarım arasındaki değerleri dengelemek için)
        let scale = 1.0 / (1.0 - self.rate);

        let mut output = Array2::zeros(input.dim());
        Zip::from(&mut output)
            .and(input)
            .and(&mask)
            .for_each(|o, &i, &m| {
                *o = i * m * scale;
            });

        // Maske referansını kaydet
        self.mask = Some(mask);

        Ok(output)
    }

    fn backward(
        &self,
        _input: &ArrayView2<f64>,
        grad_output: &ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        if !self.training || self.rate <= 0.0 {
            return Ok((
                grad_output.to_owned(),
                Array2::zeros((0, 0)),
                Array2::zeros((0, 0)),
            ));
        }

        if let Some(ref mask) = self.mask {
            // Aynı maskeyi kullan - paralelleştir
            let scale = 1.0 / (1.0 - self.rate);

            let mut grad_input = Array2::zeros(grad_output.dim());
            Zip::from(&mut grad_input)
                .and(grad_output)
                .and(mask)
                .for_each(|g_i, &g_o, &m| {
                    *g_i = g_o * m * scale;
                });

            // Dropout katmanında eğitilebilir parametre olmadığından,
            // ağırlık ve bias gradyanları boş dizilerdir
            Ok((grad_input, Array2::zeros((0, 0)), Array2::zeros((0, 0))))
        } else {
            Err(TurkError::LayerError(
                "Dropout maskesi oluşturulmamış".to_string(),
            ))
        }
    }

    fn update(
        &mut self,
        _grad_weights: &ArrayView2<f64>,
        _grad_biases: &ArrayView2<f64>,
        _learning_rate: f64,
    ) -> Result<()> {
        // Dropout katmanında eğitilebilir parametre olmadığından güncelleme yapmaz
        Ok(())
    }

    fn get_weights(&self) -> Result<Array2<f64>> {
        // Dropout katmanında ağırlık yoktur
        Ok(Array2::zeros((0, 0)))
    }

    fn get_biases(&self) -> Result<Array2<f64>> {
        // Dropout katmanında bias yoktur
        Ok(Array2::zeros((0, 0)))
    }

    fn activation_type(&self) -> Activation {
        Activation::Linear
    }

    fn layer_type(&self) -> &'static str {
        "Dropout"
    }

    fn input_size(&self) -> usize {
        if let Some((_, cols)) = self.input_shape {
            cols
        } else {
            0
        }
    }

    fn output_size(&self) -> usize {
        if let Some((_, cols)) = self.output_shape {
            cols
        } else {
            0
        }
    }

    fn param_count(&self) -> usize {
        0 // Dropout katmanında eğitilebilir parametre yoktur
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
        self.mask = None;
    }

    fn weights_as_vec(&self) -> Vec<f64> {
        Vec::new() // Dropout katmanında ağırlık yoktur
    }

    fn biases_as_vec(&self) -> Vec<f64> {
        Vec::new() // Dropout katmanında bias yoktur
    }

    fn dropout_rate(&self) -> Option<f64> {
        Some(self.rate)
    }

    fn use_batch_norm(&self) -> Option<bool> {
        None
    }

    fn set_weights(&mut self, _weights: &ArrayView2<f64>) -> Result<()> {
        // Dropout katmanında ağırlık olmadığından, işlem yapmaz
        Ok(())
    }

    fn set_biases(&mut self, _biases: &ArrayView2<f64>) -> Result<()> {
        // Dropout katmanında bias olmadığından, işlem yapmaz
        Ok(())
    }
}

impl fmt::Display for Dropout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dropout(rate={})", self.rate)
    }
}

/// Batch Normalizasyon katmanı
///
/// İleri yayılımda her mini-batch'in özellik bazlı ortalamasını ve varyansını
/// hesaplayarak girdileri normalleştirir, ardından öğrenilebilir gamma (ölçek)
/// ve beta (kaydırma) parametreleri uygular.
///
/// Formül: y = gamma * (x - mu) / sqrt(var + eps) + beta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormalization {
    /// Özellik sayısı (features)
    pub num_features: usize,
    /// Sayısal kararlılık için epsilon
    pub epsilon: f64,
    /// Hareketli ortalama momentum (çıkarım için)
    pub momentum: f64,
    /// Ölçekleme parametresi (gamma), shape: (1, num_features)
    pub gamma: Array2<f64>,
    /// Kaydırma parametresi (beta), shape: (1, num_features)
    pub beta: Array2<f64>,
    /// Hareketli ortalama (çıkarım modu)
    pub running_mean: Array2<f64>,
    /// Hareketli varyans (çıkarım modu)
    pub running_var: Array2<f64>,
    /// Eğitim modu mu?
    pub training: bool,
    /// Eğitilebilir mi?
    pub trainable: bool,
    /// Son forward geçişinin normalize edilmiş çıktısı (backward için)
    #[serde(skip)]
    pub last_x_norm: Option<Array2<f64>>,
    /// Son forward geçişinin standart sapması (backward için)
    #[serde(skip)]
    pub last_std: Option<Array2<f64>>,
    /// Son forward geçişinin batch ortalaması
    #[serde(skip)]
    pub last_mean: Option<Array2<f64>>,
}

impl BatchNormalization {
    /// `num_features` özellik boyutu için yeni bir BatchNormalization katmanı oluşturur.
    pub fn new(num_features: usize) -> Self {
        Self::with_params(num_features, 1e-5, 0.1)
    }

    pub fn with_params(num_features: usize, epsilon: f64, momentum: f64) -> Self {
        Self {
            num_features,
            epsilon,
            momentum,
            gamma: Array2::ones((1, num_features)),
            beta: Array2::zeros((1, num_features)),
            running_mean: Array2::zeros((1, num_features)),
            running_var: Array2::ones((1, num_features)),
            training: true,
            trainable: true,
            last_x_norm: None,
            last_std: None,
            last_mean: None,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Layer for BatchNormalization {
    fn forward(&mut self, input: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let (n, d) = input.dim();
        if d != self.num_features {
            return Err(TurkError::shape_mismatch(&format!(
                "BatchNorm: beklenen {d2} özellik, gelen {d}",
                d2 = self.num_features
            )));
        }

        if self.training && n > 1 {
            // Batch ortalaması: shape (1, d)
            let mean = input.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0));

            // Batch varyansı: shape (1, d)
            let diff = input - &mean;
            let var = (&diff * &diff)
                .mean_axis(Axis(0))
                .unwrap()
                .insert_axis(Axis(0));

            let std = (&var + self.epsilon).mapv(f64::sqrt);
            let x_norm = &diff / &std;

            // Hareketli istatistikleri güncelle
            self.running_mean = &self.running_mean * (1.0 - self.momentum) + &mean * self.momentum;
            self.running_var = &self.running_var * (1.0 - self.momentum) + &var * self.momentum;

            // Geri yayılım için sakla
            self.last_x_norm = Some(x_norm.clone());
            self.last_std = Some(std);
            self.last_mean = Some(mean);

            Ok(&x_norm * &self.gamma + &self.beta)
        } else {
            // Çıkarım modu: hareketli istatistikleri kullan
            let std = (&self.running_var + self.epsilon).mapv(f64::sqrt);
            let x_norm = (input - &self.running_mean) / &std;
            Ok(&x_norm * &self.gamma + &self.beta)
        }
    }

    fn backward(
        &self,
        _input: &ArrayView2<f64>,
        grad_output: &ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        let n = grad_output.nrows() as f64;

        let x_norm = self.last_x_norm.as_ref().ok_or_else(|| {
            TurkError::LayerError("BatchNorm: forward çağrılmadan backward yapılamaz".into())
        })?;
        let std = self.last_std.as_ref().ok_or_else(|| {
            TurkError::LayerError("BatchNorm: standart sapma kaydedilmemiş".into())
        })?;

        // gamma gradyanı: sum(grad * x_norm, axis=0), shape (1, d)
        let d_gamma = (grad_output * x_norm)
            .sum_axis(Axis(0))
            .insert_axis(Axis(0));

        // beta gradyanı: sum(grad, axis=0), shape (1, d)
        let d_beta = grad_output.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Giriş gradyanı: standart BN backward formülü
        // d_x_norm = grad * gamma
        let d_x_norm = grad_output * &self.gamma;

        // d_var = sum(d_x_norm * (x - mu) * -0.5 * std^{-3}, axis=0)
        let d_var = (&d_x_norm * x_norm * (-0.5 / std))
            .sum_axis(Axis(0))
            .insert_axis(Axis(0));

        // d_mu = sum(d_x_norm * -1/std, axis=0) + d_var * mean(-2*(x-mu))
        let d_mu = (&d_x_norm / (-std)).sum_axis(Axis(0)).insert_axis(Axis(0));

        // grad_input = d_x_norm/std + d_var * 2*(x-mu)/n + d_mu/n
        let grad_input = &d_x_norm / std + &d_var * x_norm * (2.0 / n) + &d_mu / n;

        // d_gamma shape: (1, d) — biases slot olarak d_beta kullanıyoruz
        Ok((grad_input, d_gamma, d_beta))
    }

    fn update(
        &mut self,
        grad_gamma: &ArrayView2<f64>,
        grad_beta: &ArrayView2<f64>,
        learning_rate: f64,
    ) -> Result<()> {
        if !self.trainable {
            return Ok(());
        }
        self.gamma = &self.gamma - &(grad_gamma * learning_rate);
        self.beta = &self.beta - &(grad_beta * learning_rate);
        Ok(())
    }

    fn get_weights(&self) -> Result<Array2<f64>> {
        Ok(self.gamma.clone())
    }
    fn get_biases(&self) -> Result<Array2<f64>> {
        Ok(self.beta.clone())
    }

    fn set_weights(&mut self, w: &ArrayView2<f64>) -> Result<()> {
        self.gamma = w.to_owned();
        Ok(())
    }
    fn set_biases(&mut self, b: &ArrayView2<f64>) -> Result<()> {
        self.beta = b.to_owned();
        Ok(())
    }

    fn activation_type(&self) -> Activation {
        Activation::Linear
    }
    fn layer_type(&self) -> &'static str {
        "BatchNormalization"
    }

    fn input_size(&self) -> usize {
        self.num_features
    }
    fn output_size(&self) -> usize {
        self.num_features
    }

    fn param_count(&self) -> usize {
        self.num_features * 2
    }

    fn set_trainable(&mut self, t: bool) {
        self.trainable = t;
    }
    fn is_trainable(&self) -> bool {
        self.trainable
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    fn reset(&mut self) {
        self.last_x_norm = None;
        self.last_std = None;
        self.last_mean = None;
    }

    fn weights_as_vec(&self) -> Vec<f64> {
        self.gamma.iter().copied().collect()
    }
    fn biases_as_vec(&self) -> Vec<f64> {
        self.beta.iter().copied().collect()
    }

    fn dropout_rate(&self) -> Option<f64> {
        None
    }
    fn use_batch_norm(&self) -> Option<bool> {
        Some(true)
    }
}

impl fmt::Display for BatchNormalization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BatchNormalization(features={}, eps={}, momentum={})",
            self.num_features, self.epsilon, self.momentum
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array;

    #[test]
    fn test_dense_layer() {
        use crate::initializers::{GlorotUniform, Zeros};
        let mut dense = Dense::new(
            2,
            3,
            Activation::Relu,
            Box::new(GlorotUniform::new()),
            Box::new(Zeros::new()),
            true,
        )
        .unwrap();
        let input = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let output = dense.forward(&input.view()).unwrap();

        // Çıktı boyutları kontrol et
        assert_eq!(output.dim(), (2, 3));

        // Geri yayılım
        let grad_output = Array::ones((2, 3));
        let (grad_input, grad_weights, grad_biases) =
            dense.backward(&input.view(), &grad_output.view()).unwrap();

        // Gradyan boyutları kontrol et
        assert_eq!(grad_input.dim(), (2, 2));
        assert_eq!(grad_weights.dim(), (2, 3));
        assert_eq!(grad_biases.dim(), (1, 3));
    }

    #[test]
    fn test_dropout_layer() {
        let mut dropout = Dropout::new(0.5);
        let input = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Eğitim modu
        dropout.set_training(true);
        let output_train = dropout.forward(&input.view()).unwrap();

        // Çıktı boyutları kontrol et
        assert_eq!(output_train.dim(), (2, 3));

        // Çıkarım modu
        dropout.set_training(false);
        let output_inference = dropout.forward(&input.view()).unwrap();

        // Çıkarımda dropout etkisi olmamalı
        assert_eq!(output_inference, input);
    }

    #[test]
    fn test_activation_functions() {
        let x = Array::from_shape_vec((2, 2), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();

        // ReLU
        let relu = Activation::Relu;
        let relu_output = relu.forward(&x);
        assert_eq!(relu_output[[0, 0]], 0.0); // -1 -> 0
        assert_eq!(relu_output[[0, 1]], 0.0); // 0 -> 0
        assert_eq!(relu_output[[1, 0]], 1.0); // 1 -> 1
        assert_eq!(relu_output[[1, 1]], 2.0); // 2 -> 2

        // Sigmoid
        let sigmoid = Activation::Sigmoid;
        let sigmoid_output = sigmoid.forward(&x);
        assert_abs_diff_eq!(sigmoid_output[[0, 0]], 0.26894142, epsilon = 1e-6); // sigmoid(-1)
        assert_abs_diff_eq!(sigmoid_output[[0, 1]], 0.5, epsilon = 1e-6); // sigmoid(0)
        assert_abs_diff_eq!(sigmoid_output[[1, 0]], 0.7310586, epsilon = 1e-6); // sigmoid(1)
        assert_abs_diff_eq!(sigmoid_output[[1, 1]], 0.8807971, epsilon = 1e-6); // sigmoid(2)

        // Tanh
        let tanh = Activation::Tanh;
        let tanh_output = tanh.forward(&x);
        assert_abs_diff_eq!(tanh_output[[0, 0]], -0.7615942, epsilon = 1e-6); // tanh(-1)
        assert_abs_diff_eq!(tanh_output[[0, 1]], 0.0, epsilon = 1e-6); // tanh(0)
        assert_abs_diff_eq!(tanh_output[[1, 0]], 0.7615942, epsilon = 1e-6); // tanh(1)
        assert_abs_diff_eq!(tanh_output[[1, 1]], 0.9640276, epsilon = 1e-6); // tanh(2)
    }
}

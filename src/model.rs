use crate::{Result, TurkError};
use crate::layer::Layer;
use crate::optimizer::Optimizer;
use crate::loss::Loss;
use turk_lang::metrics::Metric;
use ndarray::{Array2, Axis};
use std::fs;
use serde::{Serialize, Deserialize};
use std::fmt;
use std::path::Path;
use std::io::{BufReader, BufWriter};
use log::{warn};
use rand::Rng;

/// Model trait - Makine öğrenmesi modelleri için arayüz
pub trait Model: Send + Sync + fmt::Debug {
    /// İleri yayılım (forward propagation)
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>>;
    
    /// Geri yayılım (backpropagation)
    fn backward(&mut self, grad: &Array2<f32>) -> Result<()>;
    
    /// Ağırlık güncellemesi
    fn update(&mut self) -> Result<()>;
    
    /// Modeli dosyaya kaydet
    fn save(&self, path: &str) -> Result<()>;
    
    /// Modeli dosyadan yükle
    fn load(&mut self, path: &str) -> Result<()>;
    
    /// Modeli özetini döndürür
    fn summary(&self) -> String;
}

/// Sıralı model sınıfı.
/// 
/// Katmanları sıralı bir şekilde tutarak ileri ve geri yayılımı gerçekleştirir.
/// Her katman, önceki katmanın çıktısını girdi olarak alır.
#[derive(Debug)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Option<Box<dyn Optimizer>>,
    compiled: bool,
}

/// Katman verisi - Serileştirme/Deserileştirme için
#[derive(Debug, Serialize, Deserialize)]
struct LayerData {
    weights: Vec<f32>,
    bias: Vec<f32>,
    activation: crate::layer::Activation,
    input_size: usize,
    output_size: usize,
    layer_type: String,
    dropout_rate: Option<f32>,
    use_batch_norm: Option<bool>,
}

/// Model verisi - Serileştirme/Deserileştirme için
#[derive(Debug, Serialize, Deserialize)]
struct ModelData {
    layers: Vec<LayerData>,
    optimizer_type: String,
    optimizer_params: OptimizerParams,
    schema_version: String,
}

/// Optimizer parametreleri
#[derive(Debug, Serialize, Deserialize)]
struct OptimizerParams {
    learning_rate: f32,
    momentum: Option<f32>,
    beta1: Option<f32>,
    beta2: Option<f32>,
    epsilon: Option<f32>,
    weight_decay: Option<f32>,
    decay_rate: Option<f32>,
}

/// Model verisini doğrulayan yardımcı fonksiyon
fn validate_model_data(model_data: &ModelData) -> Result<()> {
    // Boş model kontrolü
    if model_data.layers.is_empty() {
        return Err(TurkError::ModelError("Boş model yüklenemez".to_string()));
    }
    
    // Şema versiyonu kontrolü
    if model_data.schema_version != "1.0" {
        return Err(TurkError::ModelError(
            format!("Desteklenmeyen model şema versiyonu: {}", model_data.schema_version)
        ));
    }
    
    // Optimizer tipi kontrolü
    match model_data.optimizer_type.as_str() {
        "sgd" | "adam" | "adamw" | "rmsprop" => {},
        _ => return Err(TurkError::ModelError(
            format!("Bilinmeyen optimizer tipi: {}", model_data.optimizer_type)
        )),
    }
    
    // Optimizer parametreleri kontrolü
    let params = &model_data.optimizer_params;
    if params.learning_rate <= 0.0 {
        return Err(TurkError::ModelError(
            format!("Geçersiz öğrenme oranı: {}", params.learning_rate)
        ));
    }
    
    // Katmanlar arası boyut tutarlılığı kontrolü
    for i in 1..model_data.layers.len() {
        let prev_layer = &model_data.layers[i-1];
        let curr_layer = &model_data.layers[i];
        
        if curr_layer.layer_type != "dropout" && 
           curr_layer.layer_type != "batch_norm" && 
           prev_layer.output_size != curr_layer.input_size {
            return Err(TurkError::ShapeMismatch(format!(
                "Katmanlar arası boyut uyuşmazlığı: {} ({}) -> {} ({})",
                prev_layer.layer_type, prev_layer.output_size, 
                curr_layer.layer_type, curr_layer.input_size
            )));
        }
    }
    
    Ok(())
}

impl Sequential {
    /// Yeni bir boş sıralı model oluşturur
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: None,
            compiled: false,
        }
    }

    /// Modele yeni bir katman ekler
    ///
    /// # Parametreler
    /// - `layer`: Eklenecek katman
    pub fn add<L: Layer + 'static>(&mut self, layer: L) -> &mut Self {
        self.layers.push(Box::new(layer));
        self.compiled = false; // Yeni katman eklendiğinde derlenmiş durumu sıfırla
        self
    }

    /// Modeli verilen optimizer ile derler
    ///
    /// # Parametreler
    /// - `optimizer`: Kullanılacak optimizer
    pub fn compile<O: Optimizer + 'static>(&mut self, optimizer: O) -> &mut Self {
        self.optimizer = Some(Box::new(optimizer));
        self.compiled = true;
        self
    }

    /// Modelin derlenmiş olduğunu kontrol eder
    fn ensure_compiled(&self) -> Result<()> {
        if !self.compiled || self.optimizer.is_none() {
            return Err(TurkError::ModelError("Model derlenmemiş. Önce compile() methodunu çağırın.".to_string()));
        }
        Ok(())
    }

    /// İleri yayılım (forward propagation)
    ///
    /// # Parametreler
    /// - `input`: Girdi tensörü
    ///
    /// # Dönüş
    /// Model çıktısı
    pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if self.layers.is_empty() {
            return Err(TurkError::ModelError("Model boş, en az bir katman ekleyin.".to_string()));
        }
        
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    /// Geri yayılım (backpropagation)
    ///
    /// # Parametreler
    /// - `grad`: Gradyan tensörü
    pub fn backward(&mut self, grad: &Array2<f32>) -> Result<()> {
        self.ensure_compiled()?;
        
        let mut current_grad = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad)?;
        }
        Ok(())
    }

    /// Ağırlık güncellemesi
    pub fn update(&mut self) -> Result<()> {
        self.ensure_compiled()?;
        
        // Optimizer'ı al (compile kontrolünden sonra güvenli)
        let optimizer = self.optimizer.as_mut().unwrap();
        
        // Her katmanı güncelleyelim
        for layer in &mut self.layers {
            // Ağırlık güncellemeleri
            if let (Some(weights), Some(weight_grads)) = (layer.get_weights(), layer.get_weight_grads()) {
                let mut weights_clone = weights.clone();
                optimizer.step(&mut weights_clone, weight_grads)?;
                layer.set_weights(weights_clone)?;
            }
            
            // Bias güncellemeleri
            if let (Some(bias), Some(bias_grads)) = (layer.get_bias(), layer.get_bias_grads()) {
                let mut bias_clone = bias.clone();
                optimizer.step(&mut bias_clone, bias_grads)?;
                layer.set_bias(bias_clone)?;
            }
            
            // Katman iç güncellemeleri
            layer.update()?;
        }
        
        Ok(())
    }

    /// Modeli dosyaya kaydeder
    ///
    /// # Parametreler
    /// - `path`: Kaydedilecek dosya yolu
    pub fn save(&self, path: &str) -> Result<()> {
        // Model doğrulaması
        if self.layers.is_empty() {
            return Err(TurkError::ModelError("Boş model kaydedilemez.".to_string()));
        }
        
        // Optimizer bilgisini kontrol et
        let optimizer = self.optimizer.as_ref().ok_or_else(|| 
            TurkError::ModelError("Derlenmiş bir model olmadan kaydetme yapılamaz.".to_string())
        )?;
        
        // Katman verilerini topla
        let layers_data: Vec<LayerData> = self.layers.iter().map(|layer| {
            LayerData {
                weights: layer.weights_as_vec(),
                bias: layer.bias_as_vec(),
                activation: layer.activation_type(),
                input_size: layer.input_size(),
                output_size: layer.output_size(),
                layer_type: layer.layer_type().to_string(),
                dropout_rate: layer.dropout_rate(),
                use_batch_norm: layer.use_batch_norm(),
            }
        }).collect();

        // Optimizer parametrelerini topla
        let optimizer_params = OptimizerParams {
            learning_rate: optimizer.learning_rate(),
            momentum: None, // Bu bilgileri optimizer tipine göre doldurabiliriz
            beta1: None,
            beta2: None,
            epsilon: None,
            weight_decay: None,
            decay_rate: None,
        };

        // Model verisini oluştur
        let model_data = ModelData {
            layers: layers_data,
            optimizer_type: optimizer.optimizer_type().to_string(),
            optimizer_params,
            schema_version: "1.0".to_string(), // Şema versiyonunu ekle
        };

        // Dosyaya yaz
        let file = fs::File::create(path).map_err(|e| 
            TurkError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other, 
                format!("Dosya oluşturulamadı: {}, Hata: {}", path, e)
            ))
        )?;
        
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &model_data)
            .map_err(|e| TurkError::SerializationError(e))?;
            
        Ok(())
    }

    /// Modeli dosyadan yükler
    ///
    /// # Parametreler
    /// - `path`: Yüklenecek dosya yolu
    pub fn load(&mut self, path: &str) -> Result<()> {
        // Dosya varlığını kontrol et
        if !Path::new(path).exists() {
            return Err(TurkError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Dosya bulunamadı: {}", path)
            )));
        }
        
        // Dosyayı oku
        let file = fs::File::open(path).map_err(|e| 
            TurkError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other, 
                format!("Dosya açılamadı: {}, Hata: {}", path, e)
            ))
        )?;
        
        // Güvenlik: dosya boyutu kontrolü
        let metadata = file.metadata().map_err(|e| 
            TurkError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other, 
                format!("Dosya metadata okunamadı: {}, Hata: {}", path, e)
            ))
        )?;
        
        if metadata.len() > 50_000_000 { // 50MB üst sınır
            return Err(TurkError::IoError(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Dosya çok büyük ({}MB), maksimum 50MB destekleniyor", metadata.len() / 1_000_000)
            )));
        }
        
        let reader = BufReader::new(file);
        let model_data: ModelData = serde_json::from_reader(reader)
            .map_err(|e| TurkError::SerializationError(e))?;
        
        // Model verisini doğrula
        validate_model_data(&model_data)?;

        // Katmanları yeniden oluştur
        self.layers.clear();
        for layer_data in model_data.layers {
            let layer: Box<dyn Layer> = match layer_data.layer_type.as_str() {
                "dense" => {
                    let mut dense = crate::layer::Dense::new(
                        layer_data.input_size,
                        layer_data.output_size,
                        layer_data.activation,
                    );
                    
                    // Ağırlıkları ve bias'ı yükle
                    let weights = Array2::from_shape_vec(
                        (layer_data.output_size, layer_data.input_size),
                        layer_data.weights,
                    ).map_err(|e| TurkError::NdarrayError(e))?;
                    
                    let bias = Array2::from_shape_vec(
                        (layer_data.output_size, 1),
                        layer_data.bias,
                    ).map_err(|e| TurkError::NdarrayError(e))?;
                    
                    // Ağırlıkları ve bias'ı ayarla
                    dense.set_weights(weights)?;
                    dense.set_bias(bias)?;
                    
                    Box::new(dense)
                },
                "dropout" => {
                    if let Some(rate) = layer_data.dropout_rate {
                        Box::new(crate::layer::Dropout::new(rate))
                    } else {
                        return Err(TurkError::ModelError(
                            "Dropout katmanı için dropout_rate gerekli".to_string()
                        ));
                    }
                },
                "batch_norm" => {
                    Box::new(crate::layer::BatchNorm::new(layer_data.input_size))
                },
                _ => return Err(TurkError::ModelError(
                    format!("Bilinmeyen katman tipi: {}", layer_data.layer_type)
                )),
            };
            
            self.layers.push(layer);
        }

        // Optimizer'ı yeniden oluştur
        let params = &model_data.optimizer_params;
        self.optimizer = match model_data.optimizer_type.as_str() {
            "sgd" => {
                if let Some(momentum) = params.momentum {
                    Some(Box::new(crate::optimizer::SGD::with_momentum(
                        params.learning_rate, momentum
                    )))
                } else {
                    Some(Box::new(crate::optimizer::SGD::new(params.learning_rate)))
                }
            },
            "adam" => {
                Some(Box::new(crate::optimizer::Adam::new(params.learning_rate)))
            },
            "adamw" => {
                if let Some(weight_decay) = params.weight_decay {
                    Some(Box::new(crate::optimizer::AdamW::new(
                        params.learning_rate, weight_decay
                    )))
                } else {
                    return Err(TurkError::ModelError(
                        "AdamW optimizeri için weight_decay parametresi gerekli".to_string()
                    ));
                }
            },
            "rmsprop" => {
                Some(Box::new(crate::optimizer::RMSProp::new(params.learning_rate)))
            },
            _ => return Err(TurkError::ModelError(
                format!("Bilinmeyen optimizer tipi: {}", model_data.optimizer_type)
            )),
        };
        
        self.compiled = true;
        log::info!("Model başarıyla yüklendi: {}", path);
        Ok(())
    }
    
    /// Model özetini döndürür
    pub fn summary(&self) -> String {
        let mut result = String::new();
        result.push_str(&format!("Model: Sequential\n"));
        result.push_str("-----------------------------\n");
        result.push_str(&format!("{:<5} {:<15} {:<15} {:<20} {:<10}\n", 
            "No", "Layer Type", "Output Shape", "Param Count", "Activation"));
        result.push_str("-----------------------------\n");
        
        let mut total_params = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_type = layer.layer_type();
            let output_shape = format!("[{}, {}]", layer.output_size(), 1);
            let params_count = layer.param_count();
            total_params += params_count;
            
            result.push_str(&format!("{:<5} {:<15} {:<15} {:<20} {:<10}\n", 
                i, layer_type, output_shape, params_count, layer.activation_type().to_string()));
        }
        
        result.push_str("-----------------------------\n");
        result.push_str(&format!("Toplam parametre sayısı: {}\n", total_params));
        
        if let Some(optimizer) = &self.optimizer {
            result.push_str(&format!("Optimizer: {} (lr={})\n", 
                optimizer.optimizer_type(), optimizer.learning_rate()));
        } else {
            result.push_str("Model henüz derlenmemiş.\n");
        }
        
        result
    }
    
    /// Modeli eğitir
    ///
    /// # Parametreler
    /// - `x`: Girdi verileri
    /// - `y`: Hedef değerler
    /// - `epochs`: Epoch sayısı
    /// - `batch_size`: Batch boyutu
    /// - `validation_data`: Doğrulama verileri (isteğe bağlı)
    pub fn fit(
        &mut self, 
        x: &Array2<f32>, 
        y: &Array2<f32>, 
        epochs: usize,
        batch_size: usize,
        validation_data: Option<(&Array2<f32>, &Array2<f32>)>
    ) -> Result<Vec<(f32, Option<f32>)>> {
        self.ensure_compiled()?;
        
        let mut loss_history = Vec::with_capacity(epochs);
        let n_samples = x.len_of(Axis(0));
        
        if n_samples == 0 {
            return Err(TurkError::InvalidInput("Eğitim verisinde örnek yok".to_string()));
        }
        
        let effective_batch_size = batch_size.min(n_samples);
        let n_batches = (n_samples + effective_batch_size - 1) / effective_batch_size;
        
        let loss_fn: &dyn Loss = &crate::loss::MSE::default();
        
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            
            let mut indices: Vec<usize> = (0..n_samples).collect();
            let mut rng = rand::thread_rng();
            
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
            
            for batch_idx in 0..n_batches {
                let start_idx = batch_idx * effective_batch_size;
                let end_idx = (start_idx + effective_batch_size).min(n_samples);
                let batch_indices = &indices[start_idx..end_idx];
                
                let mut batch_loss = 0.0;
                
                for &idx in batch_indices {
                    let x_sample = x.index_axis(Axis(0), idx).insert_axis(Axis(0)).to_owned();
                    let y_true = y.index_axis(Axis(0), idx).insert_axis(Axis(0)).to_owned();
                    
                    let y_pred = self.forward(&x_sample)?;
                    
                    let loss_val = loss_fn.compute(&y_pred, &y_true)?;
                    batch_loss += loss_val;
                    
                    let loss_grad = loss_fn.gradient(&y_pred, &y_true)?;
                    self.backward(&loss_grad)?;
                }
                
                self.update()?;
                
                batch_loss /= batch_indices.len() as f32;
                epoch_loss += batch_loss;
            }
            
            epoch_loss /= n_batches as f32;
            
            let validation_loss = if let Some((x_val, y_val)) = validation_data {
                let mut val_loss = 0.0;
                let n_val_samples = x_val.len_of(Axis(0));
                
                for i in 0..n_val_samples {
                    let x_sample = x_val.index_axis(Axis(0), i).insert_axis(Axis(0)).to_owned();
                    let y_true = y_val.index_axis(Axis(0), i).insert_axis(Axis(0)).to_owned();
                    
                    let y_pred = self.forward(&x_sample)?;
                    val_loss += loss_fn.compute(&y_pred, &y_true)?;
                }
                
                val_loss /= n_val_samples as f32;
                Some(val_loss)
            } else {
                None
            };
            
            loss_history.push((epoch_loss, validation_loss));
            
            if let Some(val_loss) = validation_loss {
                log::info!("Epoch {}/{} - loss: {:.4} - val_loss: {:.4}", 
                    epoch + 1, epochs, epoch_loss, val_loss);
            } else {
                log::info!("Epoch {}/{} - loss: {:.4}", 
                    epoch + 1, epochs, epoch_loss);
            }
        }
        
        Ok(loss_history)
    }
    
    /// Tahmin yapar
    ///
    /// # Parametreler
    /// - `x`: Girdi verileri
    ///
    /// # Dönüş
    /// Model çıktısı
    pub fn predict(&mut self, x: &Array2<f32>) -> Result<Array2<f32>> {
        self.forward(x)
    }
    
    /// Modeli test eder ve metrikleri döndürür
    ///
    /// # Parametreler
    /// - `x`: Test verileri
    /// - `y`: Gerçek etiketler
    /// - `metrics`: Hesaplanacak metrikler
    ///
    /// # Dönüş
    /// Metrik adı ve değeri çiftleri
    pub fn evaluate(
        &mut self, 
        x: &Array2<f32>, 
        y: &Array2<f32>,
        metrics: &[&dyn Metric]
    ) -> Result<Vec<(String, f32)>> {
        let y_pred = self.predict(x)?;
        
        let mut results = Vec::with_capacity(metrics.len());
        
        for metric in metrics {
            let score = metric.compute(&y_pred, y)?;
            results.push((metric.name().to_string(), score));
        }
        
        Ok(results)
    }

    /// AST'den model oluşturur
    ///
    /// # Örnek
    /// ```no_run
    /// use turk_lang::parser::{ModelParser, Parser};
    /// use turk_lang::ast::Program;
    /// use turk_lang::model::Sequential;
    /// 
    /// let dsl = r#"
    ///   Model mnist_model {
    ///     Katman(784, 128, ReLU),
    ///     Katman(128, 10, Softmax)
    ///   }
    /// "#;
    /// 
    /// let mut parser = ModelParser::new(dsl);
    /// let program = parser.parse().expect("DSL ayrıştırma hatası");
    /// let model = Sequential::from_ast(&program).expect("Model oluşturma hatası");
    /// ```
    pub fn from_ast(ast: &crate::ast::Program) -> Result<Self> {
        use crate::ast::{Deyim, KatmanOzellikleri, OptimizerTipi, AktivasyonFonksiyonu};
        use crate::layer::{Dense, Dropout, Activation};
        use crate::optimizer::{Adam, SGD, AdamW};
        use crate::parser::error::ParseError;
        
        let mut model = Self::new();
        
        let mut found_model = false;
        for deyim in &ast.deyimler {
            if let Deyim::ModelTanimi { isim: _, ozellikler: model_ozellikleri } = deyim {
                found_model = true;
                
                for katman_ozellik in &model_ozellikleri.katmanlar {
                    match katman_ozellik {
                        KatmanOzellikleri { noron_sayisi, aktivasyon, dropout_rate, .. } => {
                            let activation_enum = match aktivasyon {
                                AktivasyonFonksiyonu::Relu => Activation::Relu,
                                AktivasyonFonksiyonu::Sigmoid => Activation::Sigmoid,
                                AktivasyonFonksiyonu::Tanh => Activation::Tanh,
                                AktivasyonFonksiyonu::Softmax => Activation::Softmax,
                            };
                            
                            model.add(Dense::new(
                                model.layers.last().map_or(0, |l| l.output_size()),
                                *noron_sayisi as usize,
                                activation_enum
                            ));
                            
                            if let Some(rate) = dropout_rate {
                                if *rate > 0.0 {
                                    model.add(Dropout::new(*rate as f32));
                                }
                            }
                        }
                    }
                }
                
                let optimizer_tipi = &model_ozellikleri.optimizer;
                match optimizer_tipi {
                    OptimizerTipi::SGD(ogrenme_orani) => {
                        model.compile(SGD::new(*ogrenme_orani as f32));
                    },
                    OptimizerTipi::Adam(ogrenme_orani) => {
                        model.compile(Adam::new(*ogrenme_orani as f32));
                    },
                    OptimizerTipi::AdamW { lr, weight_decay, beta1: _, beta2: _ } => {
                        model.compile(AdamW::new(*lr as f32, *weight_decay as f32));
                    },
                }
                
                break;
            }
        }
        
        if !found_model {
            return Err(TurkError::ParseError(
                ParseError::syntax_error("AST içinde ModelTanimi bulunamadı".to_string())
            ));
        }
        
        if model.optimizer.is_none() {
            warn!("AST'den model oluşturuldu ancak optimizer belirtilmedi, varsayılan SGD (lr=0.01) kullanılıyor.");
            model.compile(SGD::new(0.01));
        }
        
        Ok(model)
    }
}

impl Model for Sequential {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        self.forward(input)
    }

    fn backward(&mut self, grad: &Array2<f32>) -> Result<()> {
        self.backward(grad)
    }

    fn update(&mut self) -> Result<()> {
        self.update()
    }

    fn save(&self, path: &str) -> Result<()> {
        self.save(path)
    }

    fn load(&mut self, path: &str) -> Result<()> {
        self.load(path)
    }
    
    fn summary(&self) -> String {
        self.summary()
    }
}

impl fmt::Display for Sequential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::{Dense, Activation, Dropout, BatchNorm};
    use crate::optimizer::{SGD, Adam, AdamW, RMSProp};
    use ndarray::{Array2, arr2};
    use tempfile::tempdir;
    use std::path::PathBuf;
    use crate::metrics::{Accuracy, MSE as MetricsMSE};
    
    fn create_test_model() -> Sequential {
        let mut model = Sequential::new();
        model.add(Dense::new(10, 5, Activation::Relu))
             .add(Dropout::new(0.2))
             .add(Dense::new(5, 2, Activation::Softmax))
             .compile(Adam::new(0.001));
        model
    }

    #[test]
    fn test_sequential_model() {
        let mut model = Sequential::new();
        model.add(Dense::new(784, 128, Activation::Relu))
             .add(BatchNorm::new(128))
             .add(Dense::new(128, 10, Activation::Softmax))
             .compile(SGD::new(0.01));

        let input = Array2::zeros((784, 1));
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[10, 1]);
    }

    #[test]
    fn test_model_compile() {
        let mut model = Sequential::new();
        model.add(Dense::new(10, 5, Activation::Relu))
             .compile(Adam::new(0.001));
        
        let input = Array2::zeros((10, 1));
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[5, 1]);
    }

    #[test]
    fn test_model_serialization() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model.json");
        let path_str = model_path.to_str().unwrap();
        
        let mut model = create_test_model();
        model.save(path_str).unwrap();
        
        let mut loaded = Sequential::new();
        loaded.load(path_str).unwrap();
        
        let input = Array2::zeros((10, 1));
        let output = loaded.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 1]);
    }
    
    #[test]
    fn test_model_adamw() {
        let mut model = Sequential::new();
        model.add(Dense::new(10, 5, Activation::Relu))
             .compile(AdamW::new(0.001, 0.01));
        
        let x = Array2::zeros((10, 1));
        let output = model.forward(&x).unwrap();
        assert_eq!(output.shape(), &[5, 1]);
    }
    
    #[test]
    fn test_model_rmsprop() {
        let mut model = Sequential::new();
        model.add(Dense::new(10, 5, Activation::Relu))
             .compile(RMSProp::new(0.001));
        
        let x = Array2::zeros((10, 1));
        let output = model.forward(&x).unwrap();
        assert_eq!(output.shape(), &[5, 1]);
    }
    
    #[test]
    fn test_model_fit() {
        let mut model = Sequential::new();
        model.add(Dense::new(2, 3, Activation::Relu))
             .add(Dense::new(3, 1, Activation::Sigmoid))
             .compile(SGD::new(0.1));
        
        let x = arr2(&[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]);
        
        let y = arr2(&[
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ]);
        
        let history = model.fit(&x, &y, 100, 1, None).unwrap();
        
        assert!(history.last().unwrap().0 < history.first().unwrap().0);
        
        let y_pred = model.predict(&x).unwrap();
        assert_eq!(y_pred.shape(), y.shape());
    }
    
    #[test]
    fn test_model_evaluate() {
        let mut model = Sequential::new();
        model.add(Dense::new(2, 3, Activation::Relu))
             .add(Dense::new(3, 1, Activation::Sigmoid))
             .compile(SGD::new(0.1));
        
        let x = arr2(&[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]);
        
        let y = arr2(&[
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ]);
        
        let mse = MetricsMSE::default();
        let accuracy = Accuracy::new(0.5);
        
        let metrics = model.evaluate(&x, &y, &[&mse, &accuracy]).unwrap();
        
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].0, "MSE");
        assert_eq!(metrics[1].0, "Accuracy");
    }
    
    #[test]
    fn test_model_summary() {
        let model = create_test_model();
        let summary = model.summary();
        
        assert!(summary.contains("Sequential"));
        assert!(summary.contains("Dense"));
        assert!(summary.contains("Dropout"));
        assert!(summary.contains("Relu"));
        assert!(summary.contains("Softmax"));
        assert!(summary.contains("Optimizer"));
    }

    #[test]
    fn test_model_compilation_from_ast() {
        use crate::ast::{KatmanOzellikleri, OptimizerTipi};
        use crate::layer::{Dense, Dropout, Activation};
        use crate::optimizer::{Adam, SGD, AdamW};
    }
} 
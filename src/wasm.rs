//! WebAssembly desteği modülü
//!
//! Bu modül, turk-lang kütüphanesinin WebAssembly üzerinden kullanılabilmesi için
//! wasm-bindgen kullanarak gerekli bağlantıları sağlar.
//!
//! Kullanım için Cargo.toml'da `wasm` feature'ını etkinleştirin:
//! ```toml
//! [dependencies]
//! turk-lang = { version = "0.1.0", features = ["wasm"] }
//! ```

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
#[cfg(feature = "wasm")]
use js_sys::{Array, Float32Array, Object, Reflect};
#[cfg(feature = "wasm")]
use crate::{
    Sequential, Dense, Activation, Dropout, BatchNorm,
    SGD, Adam, AdamW, RMSProp,
    TurkError, Result
};
#[cfg(feature = "wasm")]
use ndarray::Array2;
#[cfg(feature = "wasm")]
use serde::{Serialize, Deserialize};

#[cfg(feature = "wasm")]
#[wasm_bindgen(typescript_custom_section)]
const TS_ACTIVATION: &'static str = r#"
export enum Activation {
    Relu = "relu",
    Sigmoid = "sigmoid",
    Tanh = "tanh",
    Softmax = "softmax",
    Linear = "linear"
}
"#;

#[cfg(feature = "wasm")]
#[wasm_bindgen(typescript_custom_section)]
const TS_SEQUENTIAL_MODEL: &'static str = r#"
export interface LayerConfig {
    type: string;
    inputSize?: number;
    outputSize?: number;
    activation?: Activation;
    dropoutRate?: number;
    features?: number;
}

export interface OptimizerConfig {
    type: string;
    learningRate: number;
    momentum?: number;
    weightDecay?: number;
}

export interface FitConfig {
    epochs: number;
    batchSize: number;
    validationSplit?: number;
}

export interface TrainingHistory {
    loss: number[];
    valLoss?: number[];
}
"#;

/// WASM için Sequential model
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmModel {
    inner: Sequential,
}

/// WASM için katman konfigürasyonu
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct LayerConfig {
    layer_type: String,
    input_size: Option<usize>,
    output_size: Option<usize>,
    activation: Option<String>,
    dropout_rate: Option<f32>,
    features: Option<usize>,
}

/// WASM için optimizer konfigürasyonu
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct OptimizerConfig {
    optimizer_type: String,
    learning_rate: f32,
    momentum: Option<f32>,
    weight_decay: Option<f32>,
}

/// WASM için eğitim konfigürasyonu
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct FitConfig {
    epochs: usize,
    batch_size: usize,
    validation_split: Option<f32>,
}

/// WASM için eğitim geçmişi
#[cfg(feature = "wasm")]
#[derive(Serialize, Deserialize)]
pub struct TrainingHistory {
    loss: Vec<f32>,
    val_loss: Option<Vec<f32>>,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmModel {
    /// Yeni bir Sequential model oluşturur
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Panic mesajlarını yakalamak için
        console_error_panic_hook::set_once();
        
        Self {
            inner: Sequential::new()
        }
    }
    
    /// Modele katman ekler
    #[wasm_bindgen]
    pub fn add(&mut self, config_js: JsValue) -> Result<(), JsValue> {
        let config: LayerConfig = config_js.into_serde()
            .map_err(|e| JsValue::from_str(&format!("Katman konfigürasyonu çözülemedi: {}", e)))?;
        
        match config.layer_type.as_str() {
            "dense" => {
                let input_size = config.input_size
                    .ok_or_else(|| JsValue::from_str("Dense katmanı için input_size gerekli"))?;
                let output_size = config.output_size
                    .ok_or_else(|| JsValue::from_str("Dense katmanı için output_size gerekli"))?;
                let activation_str = config.activation
                    .ok_or_else(|| JsValue::from_str("Dense katmanı için activation gerekli"))?;
                
                let activation = match activation_str.to_lowercase().as_str() {
                    "relu" => Activation::Relu,
                    "sigmoid" => Activation::Sigmoid,
                    "tanh" => Activation::Tanh,
                    "softmax" => Activation::Softmax,
                    "linear" => Activation::Linear,
                    _ => return Err(JsValue::from_str(&format!("Bilinmeyen aktivasyon fonksiyonu: {}", activation_str)))
                };
                
                self.inner.add(Dense::new(input_size, output_size, activation));
            },
            "dropout" => {
                let rate = config.dropout_rate
                    .ok_or_else(|| JsValue::from_str("Dropout katmanı için dropout_rate gerekli"))?;
                
                self.inner.add(Dropout::new(rate));
            },
            "batchnorm" => {
                let features = config.features
                    .ok_or_else(|| JsValue::from_str("BatchNorm katmanı için features gerekli"))?;
                
                self.inner.add(BatchNorm::new(features));
            },
            _ => return Err(JsValue::from_str(&format!("Bilinmeyen katman tipi: {}", config.layer_type)))
        }
        
        Ok(())
    }
    
    /// Modeli optimizer ile derler
    #[wasm_bindgen]
    pub fn compile(&mut self, config_js: JsValue) -> Result<(), JsValue> {
        let config: OptimizerConfig = config_js.into_serde()
            .map_err(|e| JsValue::from_str(&format!("Optimizer konfigürasyonu çözülemedi: {}", e)))?;
        
        match config.optimizer_type.to_lowercase().as_str() {
            "sgd" => {
                if let Some(momentum) = config.momentum {
                    self.inner.compile(SGD::with_momentum(config.learning_rate, momentum));
                } else {
                    self.inner.compile(SGD::new(config.learning_rate));
                }
            },
            "adam" => {
                self.inner.compile(Adam::new(config.learning_rate));
            },
            "adamw" => {
                let weight_decay = config.weight_decay
                    .ok_or_else(|| JsValue::from_str("AdamW optimizeri için weight_decay gerekli"))?;
                
                self.inner.compile(AdamW::new(config.learning_rate, weight_decay));
            },
            "rmsprop" => {
                self.inner.compile(RMSProp::new(config.learning_rate));
            },
            _ => return Err(JsValue::from_str(&format!("Bilinmeyen optimizer tipi: {}", config.optimizer_type)))
        }
        
        Ok(())
    }
    
    /// Model özeti döndürür
    #[wasm_bindgen]
    pub fn summary(&self) -> String {
        self.inner.summary()
    }
    
    /// Float32Array'i ndarray'e dönüştürür
    fn float32array_to_ndarray(array: &Float32Array, shape: Vec<usize>) -> Result<Array2<f32>, JsValue> {
        if shape.len() != 2 {
            return Err(JsValue::from_str(&format!("Sadece 2-boyutlu array desteklenir, şu anki: {}-boyutlu", shape.len())));
        }
        
        let mut data = vec![0.0; array.length() as usize];
        array.copy_to(&mut data);
        
        let ndarray = Array2::from_shape_vec((shape[0], shape[1]), data)
            .map_err(|e| JsValue::from_str(&format!("Array dönüşüm hatası: {}", e)))?;
        
        Ok(ndarray)
    }
    
    /// ndarray'i Float32Array'e dönüştürür
    fn ndarray_to_float32array(array: &Array2<f32>) -> Float32Array {
        let data = array.iter().cloned().collect::<Vec<f32>>();
        let js_array = Float32Array::new_with_length(data.len() as u32);
        js_array.copy_from(&data);
        
        js_array
    }
    
    /// Modeli eğitir
    #[wasm_bindgen]
    pub fn fit(&mut self, x: Float32Array, y: Float32Array, 
               x_shape: Box<[usize]>, y_shape: Box<[usize]>,
               config_js: JsValue) -> Result<JsValue, JsValue> {
        let config: FitConfig = config_js.into_serde()
            .map_err(|e| JsValue::from_str(&format!("Eğitim konfigürasyonu çözülemedi: {}", e)))?;
        
        let x_train = Self::float32array_to_ndarray(&x, x_shape.to_vec())?;
        let y_train = Self::float32array_to_ndarray(&y, y_shape.to_vec())?;
        
        // Eğitim/doğrulama ayrımı
        let (x_train_final, y_train_final, x_val, y_val) = if let Some(val_split) = config.validation_split {
            if val_split <= 0.0 || val_split >= 1.0 {
                return Err(JsValue::from_str("validation_split 0 ile 1 arasında olmalıdır"));
            }
            
            let n_samples = x_train.shape()[0];
            let n_val = (n_samples as f32 * val_split).ceil() as usize;
            let n_train = n_samples - n_val;
            
            let x_t = x_train.slice(ndarray::s![0..n_train, ..]).to_owned();
            let y_t = y_train.slice(ndarray::s![0..n_train, ..]).to_owned();
            let x_v = x_train.slice(ndarray::s![n_train.., ..]).to_owned();
            let y_v = y_train.slice(ndarray::s![n_train.., ..]).to_owned();
            
            (x_t, y_t, Some((x_v, y_v)))
        } else {
            (x_train, y_train, None)
        };
        
        // Doğrulama verisi
        let val_data = match &x_val {
            Some((x_v, y_v)) => Some((x_v, y_v)),
            None => None,
        };
        
        // Modeli eğit
        let history = self.inner.fit(&x_train_final, &y_train_final, config.epochs, config.batch_size, val_data)
            .map_err(|e| JsValue::from_str(&format!("Eğitim hatası: {}", e)))?;
        
        // Eğitim geçmişi
        let mut train_loss = Vec::new();
        let mut val_loss = if val_data.is_some() { Some(Vec::new()) } else { None };
        
        for (t_loss, v_loss) in history {
            train_loss.push(t_loss);
            if let Some(v_loss) = v_loss {
                if let Some(val_loss) = &mut val_loss {
                    val_loss.push(v_loss);
                }
            }
        }
        
        let training_history = TrainingHistory { loss: train_loss, val_loss };
        
        // JSON'a dönüştür
        JsValue::from_serde(&training_history)
            .map_err(|e| JsValue::from_str(&format!("Eğitim geçmişi dönüştürülemedi: {}", e)))
    }
    
    /// Tahmin yapar
    #[wasm_bindgen]
    pub fn predict(&mut self, x: Float32Array, x_shape: Box<[usize]>) -> Result<Float32Array, JsValue> {
        let x_input = Self::float32array_to_ndarray(&x, x_shape.to_vec())?;
        let y_pred = self.inner.predict(&x_input)
            .map_err(|e| JsValue::from_str(&format!("Tahmin hatası: {}", e)))?;
        
        Ok(Self::ndarray_to_float32array(&y_pred))
    }
    
    /// Modeli kaydeder (JSON formatında)
    #[wasm_bindgen]
    pub fn save(&self) -> Result<JsValue, JsValue> {
        let mut writer = Vec::new();
        self.inner.save_to_writer(&mut writer)
            .map_err(|e| JsValue::from_str(&format!("Model kaydetme hatası: {}", e)))?;
        
        let json_str = String::from_utf8(writer)
            .map_err(|e| JsValue::from_str(&format!("UTF-8 dönüşüm hatası: {}", e)))?;
        
        Ok(JsValue::from_str(&json_str))
    }
    
    /// Modeli yükler (JSON formatından)
    #[wasm_bindgen]
    pub fn load(&mut self, json: &str) -> Result<(), JsValue> {
        self.inner.load_from_reader(json.as_bytes())
            .map_err(|e| JsValue::from_str(&format!("Model yükleme hatası: {}", e)))
    }
    
    /// Kütüphane versiyonunu döndürür
    #[wasm_bindgen(js_name = getVersion)]
    pub fn get_version() -> String {
        crate::VERSION.to_string()
    }
}

/// Sequential modelin varsayılan implementasyonuna metotlar eklemek için
impl Sequential {
    /// Modeli yazıcıya kaydeder (JSON)
    #[cfg(feature = "wasm")]
    pub fn save_to_writer<W: std::io::Write>(&self, writer: W) -> crate::Result<()> {
        // Mevcut save metodu dosyaya yazıyor, burada yazıcıya yazıyoruz
        let model_data = self.prepare_model_data()?;
        serde_json::to_writer_pretty(writer, &model_data)
            .map_err(|e| crate::TurkError::SerializationError(e))
    }
    
    /// Model verilerini hazırlar (JSON serileştirme için)
    #[cfg(feature = "wasm")]
    fn prepare_model_data(&self) -> crate::Result<crate::model::ModelData> {
        // Model data'yı hazırla
        unimplemented!("Bu metot model.rs içinde tanımlanmalı")
    }
    
    /// Modeli okuyucudan yükler (JSON)
    #[cfg(feature = "wasm")]
    pub fn load_from_reader<R: std::io::Read>(&mut self, reader: R) -> crate::Result<()> {
        // Mevcut load metodu dosyadan okuyor, burada okuyucudan okuyoruz
        let model_data: crate::model::ModelData = serde_json::from_reader(reader)
            .map_err(|e| crate::TurkError::SerializationError(e))?;
        
        // Model verisini doğrula ve modele yükle
        unimplemented!("Bu metot model.rs içinde tanımlanmalı")
    }
} 
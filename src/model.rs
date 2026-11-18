use crate::aktivasyon::Activation;
use crate::cihaz::Cihaz;
use crate::dtype::{self, Float};
use crate::error::{Result, TurkError};
use crate::initializers::{GlorotUniform, Zeros};
use crate::katman_kayit::{KatmanKayit, KatmanYukleVerisi};
use crate::layer::{BatchNormalization, Dense, Dropout, Layer};
use crate::loss::{Loss, MSE};
use crate::metrics::Metric;
use crate::optimizer::{Adam, AdamW, Optimizer, RMSProp, SGD};
use crate::utils::stack_selected_rows;
use crate::utils::tensor_ops;
use crate::utils::validate_tensor_shapes;
use log::{debug, info, warn};
use ndarray::{Array2, ArrayView2};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::io::{Read, Write};

// ─────────────────────────────────────────────
// Eğitim Geçmişi
// ─────────────────────────────────────────────

/// Epoch sonunda GUI / CLI geri bildirimi için callback.
pub type EpochCallback = Box<dyn FnMut(usize, usize, Float, Option<Float>) + Send>;

/// Eğitim süreci boyunca toplanan metrik ve kayıp değerleri.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct History {
    /// Her epoch'taki eğitim kaybı
    pub train_losses: Vec<Float>,
    /// Her epoch'taki doğrulama kaybı (doğrulama verisi yoksa boş)
    pub val_losses: Vec<Float>,
    /// Her epoch'taki eğitim metrikleri (henüz: MSE)
    pub train_metrics: Vec<Float>,
    /// Her epoch'taki doğrulama metrikleri
    pub val_metrics: Vec<Float>,
}

impl History {
    pub fn new() -> Self {
        Self::default()
    }

    /// Son eğitim kaybını döndürür.
    pub fn last_train_loss(&self) -> Option<Float> {
        self.train_losses.last().copied()
    }

    /// Son doğrulama kaybını döndürür.
    pub fn last_val_loss(&self) -> Option<Float> {
        self.val_losses.last().copied()
    }

    /// Toplam epoch sayısı.
    pub fn epochs_run(&self) -> usize {
        self.train_losses.len()
    }

    /// val_loss'un en düşük olduğu epoch'taki kayıtları döndürür:
    /// `(train_loss, val_loss, epoch_no)` — epoch_no 1 tabanlıdır.
    pub fn en_iyi_val_kayitlari(&self) -> Option<(Float, Float, usize)> {
        let (idx, &best_val) = self
            .val_losses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        let train = self.train_losses.get(idx).copied()?;
        Some((train, best_val, idx + 1))
    }

    /// Eğitim geçmişini JSON formatında dosyaya kaydeder.
    ///
    /// Her epoch için `train_loss`, `val_loss`, `train_metric`, `val_metric`
    /// değerlerini içeren bir dizi ve özet istatistikleri yazar.
    ///
    /// # Örnek
    /// ```no_run
    /// # use turk::model::History;
    /// let history = History::new();
    /// history.dosyaya_kaydet("egitim_gecmisi.json").unwrap();
    /// ```
    pub fn dosyaya_kaydet(&self, dosya_yolu: &str) -> Result<()> {
        use serde_json::json;

        let epoch_sayisi = self.epochs_run();

        // Her epoch'u tablo halinde oluştur
        let epochlar: Vec<serde_json::Value> = (0..epoch_sayisi)
            .map(|i| {
                let mut kayit = serde_json::Map::new();
                kayit.insert("epoch".into(), json!(i + 1));
                kayit.insert("train_loss".into(), json!(self.train_losses.get(i)));
                kayit.insert("val_loss".into(), json!(self.val_losses.get(i)));
                kayit.insert("train_metric".into(), json!(self.train_metrics.get(i)));
                kayit.insert("val_metric".into(), json!(self.val_metrics.get(i)));
                serde_json::Value::Object(kayit)
            })
            .collect();

        // Özet istatistikler
        let ozet = json!({
            "toplam_epoch": epoch_sayisi,
            "ilk_train_loss": self.train_losses.first(),
            "son_train_loss": self.train_losses.last(),
            "en_dusuk_train_loss": self.train_losses.iter().cloned().reduce(Float::min),
            "ilk_val_loss": self.val_losses.first(),
            "son_val_loss": self.val_losses.last(),
            "en_dusuk_val_loss": self.val_losses.iter().cloned().reduce(Float::min),
        });

        let json_cikti = json!({
            "ozet": ozet,
            "epochlar": epochlar,
        });

        let json_str = serde_json::to_string_pretty(&json_cikti)
            .map_err(|e| TurkError::IOError(format!("JSON serileştirme hatası: {e}")))?;

        #[cfg(not(target_arch = "wasm32"))]
        {
            fs::write(dosya_yolu, json_str)
                .map_err(|e| TurkError::IOError(format!("'{dosya_yolu}' yazılamadı: {e}")))?;
            Ok(())
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (dosya_yolu, json_str);
            Err(TurkError::IOError(
                "Eğitim geçmişi kaydı WASM ortamında desteklenmiyor.".into(),
            ))
        }
    }
}

// ─────────────────────────────────────────────
// Erken Durdurma
// ─────────────────────────────────────────────

/// Doğrulama kaybı iyileşmediğinde eğitimi erken durduran yapı.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStopping {
    /// Kaç epoch boyunca iyileşme bekleneceği
    pub patience: usize,
    /// Minimum iyileşme miktarı (bu kadar düşmezse sayaç artar)
    pub min_delta: Float,
    /// Mevcut en iyi kayıp
    best_loss: Float,
    /// İyileşme olmayan epoch sayacı
    counter: usize,
}

impl EarlyStopping {
    /// Yeni bir EarlyStopping oluşturur.
    ///
    /// `patience`: iyileşme gözlemlenmeden beklenecek epoch sayısı.  
    /// `min_delta`: iyileşme sayılabilmesi için gereken minimum düşüş.
    pub fn new(patience: usize, min_delta: Float) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: Float::INFINITY,
            counter: 0,
        }
    }

    /// Mevcut en iyi val_loss değerini döndürür (henüz adım yoksa +∞).
    pub fn best_val_loss(&self) -> Float {
        self.best_loss
    }

    /// Verilen val_loss ile sayacı günceller; durdurulması gerekiyorsa `true` döner.
    pub fn step(&mut self, val_loss: Float) -> bool {
        if val_loss < self.best_loss - self.min_delta {
            self.best_loss = val_loss;
            self.counter = 0;
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }

    /// İç durumu sıfırlar (modeli yeniden eğitmeden önce çağırın).
    pub fn reset(&mut self) {
        self.best_loss = Float::INFINITY;
        self.counter = 0;
    }
}

// ─────────────────────────────────────────────

/// # Model Modülü
///
/// Bu modül, yapay sinir ağı modelleri oluşturmak ve eğitmek için gerekli sınıfları ve
/// fonksiyonları içerir. Modüldeki temel bileşenler:
///
/// - `Model` trait'i: Tüm model tipleri için ortak bir arayüz sağlar
/// - `Sequential` sınıfı: Ardışık (Sequential) modellerini temsil eder
/// - Yardımcı fonksiyonlar: Kayıp hesaplama, gradient hesaplama vb.
///
/// ## Örnek Kullanım
///
/// ```no_run
/// use turk::model::Sequential;
/// use turk::aktivasyon::Activation;
/// use turk::initializers::{GlorotUniform, Zeros};
/// use turk::optimizer::SGD;
/// use turk::Float;
/// use ndarray::Array2;
///
/// fn dense(i: usize, o: usize, act: Activation) -> turk::layer::Dense {
///     turk::layer::Dense::new(i, o, act, Box::new(GlorotUniform::new()), Box::new(Zeros::new()), true).unwrap()
/// }
///
/// let mut model = Sequential::new();
/// model.add(dense(784, 128, Activation::Relu))
///      .add(dense(128, 10, Activation::Softmax))
///      .compile(SGD::new(0.01, None, false));
///
/// let x_train = Array2::<Float>::zeros((100, 784));
/// let y_train = Array2::<Float>::zeros((100, 10));
/// model.fit(&x_train.view(), &y_train.view(), 10, 32, None).unwrap();
///
/// let x_test = Array2::<Float>::zeros((10, 784));
/// let _predictions = model.predict(&x_test.view()).unwrap();
/// ```
/// Model trait - Makine öğrenmesi modelleri için arayüz
///
/// Bu trait, farklı model türleri (Sequential, Functional vb.) için ortak
/// bir arayüz tanımlar. Temel model işlevlerini (eğitim, tahmin, kaydedip yükleme)
/// bu trait üzerinden erişilebilir hale getirir.
pub trait Model: Send + Sync + fmt::Debug {
    /// İleri yayılım (forward propagation)
    ///
    /// Girdi verilerini modelin içinden geçirerek çıktı üretir.
    ///
    /// # Parametreler
    /// - `input`: Girdi tensörü, boyut (batch_size, input_features)
    ///
    /// # Dönüş
    /// - `Result<Array2<Float>>`: Başarılı ise model çıktısı, hata durumunda ilgili hata
    fn forward<'a>(&mut self, input: &'a ArrayView2<'a, Float>) -> Result<Array2<Float>>;

    /// Geri yayılım (backpropagation)
    ///
    /// Çıktıdan girdiye doğru gradyan hesaplar ve parametre güncellemeleri için
    /// bu bilgiyi kullanır.
    ///
    /// # Parametreler
    /// - `grad`: Çıktı gradyanı tensörü
    ///
    /// # Dönüş
    /// - `Result<Array2<Float>>`: Girdiye göre gradyan, hata durumunda ilgili hata
    fn backward<'a>(&mut self, grad: &'a ArrayView2<'a, Float>) -> Result<Array2<Float>>;

    /// Ağırlık güncellemesi
    ///
    /// Modelin optimizer'ı ile parametreleri günceller.
    ///
    /// # Dönüş
    /// - `Result<()>`: Başarılı ise Ok(()), hata durumunda ilgili hata
    fn update(&mut self) -> Result<()>;

    /// Modeli dosyaya kaydet
    ///
    /// # Parametreler
    /// - `path`: Kaydedilecek dosya yolu
    ///
    /// # Dönüş
    /// - `Result<()>`: Başarılı ise Ok(()), hata durumunda ilgili hata
    fn save(&self, path: &str) -> Result<()>;

    /// Modeli dosyadan yükle
    ///
    /// # Parametreler
    /// - `path`: Yüklenecek dosya yolu
    ///
    /// # Dönüş
    /// - `Result<()>`: Başarılı ise Ok(()), hata durumunda ilgili hata
    fn load(&mut self, path: &str) -> Result<()>;

    /// Model özetini döndürür
    ///
    /// Model yapısı, katmanlar ve parametreler hakkında bilgi içeren
    /// formatlanmış bir metin döndürür.
    ///
    /// # Dönüş
    /// - `String`: Model özeti
    fn summary(&self) -> String;

    /// Modeli eğitir
    ///
    /// Verilen eğitim verileri üzerinde modeli belirtilen sayıda epoch
    /// ve batch boyutu ile eğitir.
    ///
    /// # Parametreler
    /// - `x_train`: Eğitim girdi verileri
    /// - `y_train`: Eğitim hedef değerleri
    /// - `epochs`: Eğitim tekrar sayısı
    /// - `batch_size`: Her iterasyonda kullanılacak örnek sayısı
    /// - `validation_data`: Doğrulama verileri (isteğe bağlı)
    ///
    /// # Dönüş
    /// - `Result<History>`: Her epoch için toplanan eğitim/doğrulama kayıp ve metrikleri
    fn fit<'a>(
        &mut self,
        x_train: &'a ArrayView2<'a, Float>,
        y_train: &'a ArrayView2<'a, Float>,
        epochs: usize,
        batch_size: usize,
        validation_data: Option<(&'a ArrayView2<'a, Float>, &'a ArrayView2<'a, Float>)>,
    ) -> Result<History>;

    /// Tahmin yapar
    ///
    /// Model üzerinde ileri yayılım yaparak tahminleri hesaplar.
    ///
    /// # Parametreler
    /// - `x`: Girdi verileri
    ///
    /// # Dönüş
    /// - `Result<Array2<Float>>`: Model tahminleri
    fn predict<'a>(&mut self, x: &'a ArrayView2<'a, Float>) -> Result<Array2<Float>>;

    /// Modeli değerlendirir
    ///
    /// Verilen test verileri üzerinde modelin performansını değerlendirir.
    ///
    /// # Parametreler
    /// - `x_val`: Test girdi verileri
    /// - `y_val`: Test hedef değerleri
    ///
    /// # Dönüş
    /// - `Result<Float>`: Değerlendirme skoru (genellikle kayıp değeri)
    fn evaluate<'a>(
        &mut self,
        x_val: &'a ArrayView2<'a, Float>,
        y_val: &'a ArrayView2<'a, Float>,
    ) -> Result<Float>;
}

/// Sıralı model sınıfı (Sequential model).
///
/// Katmanları sıralı bir şekilde tutarak ileri ve geri yayılımı gerçekleştirir.
/// Her katman, önceki katmanın çıktısını girdi olarak alır. TURK dilinde
/// en yaygın kullanılan model tipidir.
///
/// ## Önemli Özellikler
///
/// - Sıralı yapı: Katmanlar birbirini lineer olarak takip eder
/// - Kolay kullanım: Basit bir API ile karmaşık modeller oluşturulabilir
/// - Tam eğitim döngüsü: fit, forward, backward ve update işlemleri
/// - Model kaydetme/yükleme: Eğitilen modeller JSON formatında kaydedilip yüklenebilir
///
/// ## Örnek
///
/// ```no_run
/// use turk::model::Sequential;
/// use turk::aktivasyon::Activation;
/// use turk::initializers::{GlorotUniform, Zeros};
/// use turk::optimizer::SGD;
/// use ndarray::arr2;
///
/// fn dense(i: usize, o: usize, act: Activation) -> turk::layer::Dense {
///     turk::layer::Dense::new(i, o, act, Box::new(GlorotUniform::new()), Box::new(Zeros::new()), true).unwrap()
/// }
///
/// let mut model = Sequential::new();
/// model.add(dense(2, 4, Activation::Relu))
///      .add(dense(4, 1, Activation::Sigmoid))
///      .compile(SGD::new(0.1, None, false));
///
/// let x = arr2(&[[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]);
/// let y = arr2(&[[0.0],[1.0],[1.0],[0.0]]);
/// model.fit(&x.view(), &y.view(), 1000, 4, None).unwrap();
/// ```
#[derive(Debug)]
pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
    pub optimizer: Option<Box<dyn Optimizer>>,
    pub loss: Option<Box<dyn Loss>>,
    pub compiled: bool,
    pub input_shape: Option<(usize, usize)>,
    pub output_shape: Option<(usize, usize)>,
    pub loss_history: Vec<Float>,
    pub val_loss_history: Vec<Float>,
    pub inputs: Option<Array2<Float>>,
    pub cihaz: Cihaz,
}

/// Sequential için manuel Clone uygulaması
impl Clone for Sequential {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.iter().map(|layer| layer.clone_box()).collect(),
            optimizer: self.optimizer.as_ref().map(|opt| opt.clone_box()),
            loss: self.loss.as_ref().map(|l| l.clone_box()),
            compiled: self.compiled,
            input_shape: self.input_shape,
            output_shape: self.output_shape,
            loss_history: self.loss_history.clone(),
            val_loss_history: self.val_loss_history.clone(),
            inputs: self.inputs.clone(),
            cihaz: self.cihaz,
        }
    }
}

/// Katman verisi - Serileştirme/Deserileştirme için
#[derive(Debug, Serialize, Deserialize)]
struct LayerData {
    layer_type: String,
    weights: Vec<Float>,
    biases: Vec<Float>,
    activation: Activation,
    input_size: usize,
    output_size: usize,
    dropout_rate: Option<Float>,
    /// BatchNormalization: özellik sayısı
    #[serde(default)]
    num_features: usize,
    /// BatchNormalization: epsilon
    #[serde(default)]
    bn_epsilon: Float,
    /// BatchNormalization: momentum
    #[serde(default)]
    bn_momentum: Float,
    #[serde(default)]
    extra: std::collections::HashMap<String, Float>,
}

/// Model verisi - Serileştirme/Deserileştirme için
#[derive(Debug, Serialize, Deserialize)]
struct ModelData {
    layers: Vec<LayerData>,
    optimizer_type: String,
    optimizer_params: OptimizerParams,
    schema_version: String,
    #[serde(default = "default_cihaz_str")]
    cihaz: String,
}

fn default_cihaz_str() -> String {
    "cpu".to_string()
}

/// Optimizer parametreleri
#[derive(Debug, Serialize, Deserialize)]
struct OptimizerParams {
    learning_rate: Float,
    momentum: Option<Float>,
    beta1: Option<Float>,
    beta2: Option<Float>,
    epsilon: Option<Float>,
    weight_decay: Option<Float>,
}

impl Sequential {
    /// Yeni bir boş sıralı model oluşturur
    ///
    /// Henüz hiçbir katman veya optimizer içermeyen, derlenme gereği olan
    /// boş bir Sequential model instance'ı oluşturur.
    ///
    /// # Örnek
    ///
    /// ```
    /// use turk::model::Sequential;
    /// let model = Sequential::new();
    /// assert!(model.layers.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: None,
            loss: None,
            compiled: false,
            input_shape: None,
            output_shape: None,
            loss_history: Vec::new(),
            val_loss_history: Vec::new(),
            inputs: None,
            cihaz: Cihaz::default(),
        }
    }

    /// Hesaplama cihazını ayarlar
    pub fn set_cihaz(&mut self, cihaz: Cihaz) -> &mut Self {
        self.cihaz = cihaz;
        self
    }

    /// Aktif cihazı döndürür
    pub fn cihaz(&self) -> Cihaz {
        self.cihaz
    }

    /// Modele yeni bir katman ekler
    ///
    /// Sıralı modele yeni bir katman ekler. Katman, Layer trait'ini
    /// gerçekleştiren herhangi bir yapı olabilir (Dense, Dropout, BatchNorm, vb.)
    ///
    /// # Parametreler
    /// - `layer`: Eklenecek katman (Layer trait'i gerçekleştiren bir nesne)
    ///
    /// # Dönüş Değeri
    /// Akışkan API için kendisine referans döndürür (builder pattern)
    ///
    /// # Notlar
    /// - Eğer model zaten derlenmiş ise, katman ekledikten sonra tekrar
    ///   derlenmesi gerektiğini kullanıcıya bildirir.
    ///
    /// # Örnek
    ///
    /// ```no_run
    /// use turk::model::Sequential;
    /// use turk::aktivasyon::Activation;
    /// use turk::initializers::{GlorotUniform, Zeros};
    ///
    /// fn dense(i: usize, o: usize, act: Activation) -> turk::layer::Dense {
    ///     turk::layer::Dense::new(i, o, act, Box::new(GlorotUniform::new()), Box::new(Zeros::new()), true).unwrap()
    /// }
    ///
    /// let mut model = Sequential::new();
    /// model.add(dense(784, 128, Activation::Relu))
    ///      .add(dense(128, 64, Activation::Relu))
    ///      .add(dense(64, 10, Activation::Softmax));
    /// ```
    /// Kutulanmış (`Box<dyn Layer>`) bir katmanı modele ekler.
    /// `KatmanKayit` gibi fabrikalar `Box<dyn Layer>` döndürdüğünden bu yöntem gereklidir.
    pub fn add_boxed(&mut self, layer: Box<dyn Layer>) -> &mut Self {
        if self.compiled {
            warn!("Derlenmiş bir modele katman ekleniyor. Yeniden derleme gerekecek.");
            self.compiled = false;
        }
        self.layers.push(layer);
        self
    }

    pub fn add<L: Layer + 'static>(&mut self, layer: L) -> &mut Self {
        if self.compiled {
            warn!("Derlenmiş bir modele katman ekleniyor. Bu, modelin yeniden derlenmesini gerektirecek.");
            self.compiled = false;
        }

        self.layers.push(Box::new(layer));
        self
    }

    /// Modeli verilen optimizer ile derler
    ///
    /// Modeli eğitim ve tahmin için hazırlar. Bu adım, fit(), forward() veya
    /// predict() gibi işlevler çağrılmadan önce zorunludur.
    ///
    /// # Parametreler
    /// - `optimizer`: Kullanılacak optimizer (SGD, Adam, RMSProp vb.)
    ///
    /// # Dönüş Değeri
    /// Akışkan API için kendisine referans döndürür (builder pattern)
    ///
    /// # Örnek
    ///
    /// ```no_run
    /// use turk::model::Sequential;
    /// use turk::optimizer::SGD;
    ///
    /// let mut model = Sequential::new();
    /// // ... katmanları ekle
    /// model.compile(SGD::new(0.01, None, false));
    /// ```
    pub fn compile<O: Optimizer + 'static>(&mut self, optimizer: O) -> &mut Self {
        self.optimizer = Some(Box::new(optimizer));
        if self.loss.is_none() {
            self.loss = Some(Box::new(MSE));
        }
        self.compiled = true;
        self
    }

    /// Modeli `Box<dyn Optimizer>` ile derler
    ///
    /// Modeli kutulanmış bir optimizer ile derler. Bu metot, from_ast veya harici
    /// bir optimizer oluşturma işlemi sırasında kullanılır.
    ///
    /// # Parametreler
    /// - `optimizer`: Kutulanmış optimizer
    ///
    /// # Dönüş Değeri
    /// Akışkan API için kendisine referans döndürür (builder pattern)
    pub fn compile_boxed(&mut self, optimizer: Box<dyn Optimizer>) -> &mut Self {
        self.optimizer = Some(optimizer);
        if self.loss.is_none() {
            self.loss = Some(Box::new(MSE));
        }
        self.compiled = true;
        self
    }

    /// Optimizer ve kayıp fonksiyonu ile modeli derler.
    pub fn compile_with_loss(
        &mut self,
        optimizer: Box<dyn Optimizer>,
        loss: Box<dyn Loss>,
    ) -> &mut Self {
        self.optimizer = Some(optimizer);
        self.loss = Some(loss);
        self.compiled = true;
        self
    }

    /// Aktif kayıp fonksiyonunun adını döndürür.
    pub fn loss_name(&self) -> &str {
        self.loss.as_ref().map(|l| l.name()).unwrap_or("mse")
    }

    /// Model mimarisi özetini metin olarak döndürür.
    pub fn summary(&self) -> String {
        let mut out = String::from("╔══════════════════════════════════════════════════╗\n");
        out.push_str("║              TÜRK Model Özeti                    ║\n");
        out.push_str("╠══════════════════════════════════════════════════╣\n");

        if self.layers.is_empty() {
            out.push_str("║  (Henüz katman eklenmemiş)                       ║\n");
        } else {
            let mut toplam_param = 0usize;
            let mut egitilebilir = 0usize;
            for (i, katman) in self.layers.iter().enumerate() {
                let param = katman.param_count();
                toplam_param += param;
                if katman.is_trainable() {
                    egitilebilir += param;
                }
                out.push_str(&format!(
                    "║  {:>2}. {:<20} girdi={:<4} cikti={:<4} param={:<6} ║\n",
                    i + 1,
                    katman.layer_type(),
                    katman.input_size(),
                    katman.output_size(),
                    param,
                ));
            }
            if let (Some(ilk), Some(son)) = (self.layers.first(), self.layers.last()) {
                out.push_str("╠══════════════════════════════════════════════════╣\n");
                out.push_str(&format!("║  Giriş boyutu  : {:<32} ║\n", ilk.input_size()));
                out.push_str(&format!("║  Çıkış boyutu  : {:<32} ║\n", son.output_size()));
                out.push_str(&format!("║  Toplam param  : {:<32} ║\n", toplam_param));
                out.push_str(&format!("║  Eğitilebilir  : {:<32} ║\n", egitilebilir));
            }
        }

        out.push_str("╠══════════════════════════════════════════════════╣\n");
        out.push_str(&format!(
            "║  Derlenmiş     : {:<32} ║\n",
            if self.compiled { "evet" } else { "hayır" }
        ));
        out.push_str(&format!(
            "║  Optimizer     : {:<32} ║\n",
            self.optimizer
                .as_ref()
                .map(|o| o.optimizer_type())
                .unwrap_or("-")
        ));
        out.push_str(&format!("║  Kayıp         : {:<32} ║\n", self.loss_name()));
        out.push_str(&format!("║  Cihaz         : {:<32} ║\n", self.cihaz.ad()));
        out.push_str("╚══════════════════════════════════════════════════╝\n");
        out
    }

    fn loss_fn(&self) -> Result<&dyn Loss> {
        self.loss
            .as_deref()
            .ok_or_else(|| TurkError::ModelError("Kayıp fonksiyonu tanımlı değil".into()))
    }

    fn batch_loss_and_grad(
        &self,
        y_true: &ArrayView2<Float>,
        y_pred: &ArrayView2<Float>,
    ) -> Result<(Float, Array2<Float>)> {
        let loss_fn = self.loss_fn()?;
        let batch_loss = loss_fn.loss(y_true, y_pred)?;
        let grad = loss_fn.gradient(y_true, y_pred)?;
        Ok((batch_loss, grad))
    }

    /// Modelin derlenip derlenmediğini kontrol eder
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Modelin derlenmiş olduğunu kontrol eder
    fn ensure_compiled(&self) -> Result<()> {
        if !self.compiled {
            return Err(TurkError::ModelError("Model derlenmemiş".to_string()));
        }

        if self.layers.is_empty() {
            return Err(TurkError::ModelError("Modelde hiç katman yok".to_string()));
        }

        Ok(())
    }

    /// İleri yayılım (forward propagation)
    ///
    /// Girdi verilerini modelin içinden geçirerek çıktı üretir.
    ///
    /// # Parametreler
    /// - `input`: Girdi tensörü
    ///
    /// # Dönüş
    /// - `Result<Array2<Float>>`: Modelin çıktısı, hata durumunda ilgili hata
    pub fn forward<'a>(&mut self, input: &'a ArrayView2<'a, Float>) -> Result<Array2<Float>> {
        self.ensure_compiled()?;

        self.inputs = Some(input.to_owned());
        let mut output = input.to_owned();

        for layer in &mut self.layers {
            output = layer.forward(&output.view())?;
        }

        Ok(output)
    }

    /// Geri yayılım (backpropagation)
    ///
    /// # Parametreler
    /// - `grad`: Gradyan tensörü
    pub fn backward<'a>(&mut self, grad: &'a ArrayView2<'a, Float>) -> Result<Array2<Float>> {
        self.ensure_compiled()?;

        // Kayıt ettiğimiz ileri yayılım sonuçlarını kullanalım

        // Geri yayılım için ters sırayla katmanları dolaşmalıyız
        let n_layers = self.layers.len();

        let mut inputs = Vec::with_capacity(n_layers);

        // İleri yayılım sırasında tüm ara girdileri hesapla
        let mut current_output = self.inputs.clone().unwrap_or_else(|| Array2::zeros((0, 0)));

        for layer in &mut self.layers {
            inputs.push(current_output.clone());
            let output = layer.forward(&current_output.view())?;
            current_output = output;
        }

        // Geri yayılım
        let mut grad_output = grad.to_owned();

        for i in (0..n_layers).rev() {
            let layer = &mut self.layers[i];
            let layer_input = &inputs[i];

            // Gradyanları hesapla
            let (grad_input, grad_weights, grad_biases) =
                layer.backward(&layer_input.view(), &grad_output.view())?;

            // Ağırlık ve bias'ları al
            let mut _weights = layer.get_weights()?.to_owned();
            let mut _biases = layer.get_biases()?.to_owned();

            // Optimizer ile güncellemeleri yap
            if let Some(optimizer) = &mut self.optimizer {
                optimizer.update(
                    &mut _weights,
                    &mut _biases,
                    &grad_weights.view(),
                    &grad_biases.view(),
                )?;

                // Güncellenmiş ağırlık ve bias'ları katmana geri aktar
                layer.set_weights(&_weights.view())?;
                layer.set_biases(&_biases.view())?;
            }

            grad_output = grad_input;
        }

        Ok(grad_output)
    }

    /// Ağırlık güncellemesi
    pub fn update(&mut self) -> Result<()> {
        // Her katmanı güncelle
        for layer_idx in 0..self.layers.len() {
            let layer = &mut self.layers[layer_idx];

            // Katman ağırlıklarını ve bias'larını al
            let weights_view = layer.get_weights()?;
            let biases_view = layer.get_biases()?;

            // Katman gradyanlarını al (eğer uygulanabilirse)
            let mut weights = weights_view.to_owned();
            let mut biases = biases_view.to_owned();

            // Optimizeri kullanarak güncelleme yap (sadece optimizer varsa)
            if let Some(optimizer) = &mut self.optimizer {
                // Dummy gradient değerleri oluştur
                let grad_weights = Array2::<Float>::zeros(weights.dim());
                let grad_biases = Array2::<Float>::zeros(biases.dim());

                optimizer.update(
                    &mut weights,
                    &mut biases,
                    &grad_weights.view(),
                    &grad_biases.view(),
                )?;

                // Güncellenmiş ağırlıkları ve bias'ları geri ayarla
                layer.set_weights(&weights.view())?;
                layer.set_biases(&biases.view())?;
            }
        }

        Ok(())
    }

    /// Model ağırlıklarını JSON metnine serileştirir (disk gerekmez — WASM uyumlu).
    pub fn to_json_string(&self) -> Result<String> {
        let mut layer_data = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let _weights = layer.get_weights()?;
            let _biases = layer.get_biases()?;

            // BatchNormalization için ek meta
            let (bn_eps, bn_mom) = if layer.layer_type() == "BatchNormalization" {
                // epsilon/momentum varsayılan değerleri (BatchNorm struct'tan alınamıyorsa)
                (1e-5, 0.1)
            } else {
                (0.0, 0.0)
            };

            layer_data.push(LayerData {
                layer_type: layer.layer_type().to_string(),
                weights: layer.weights_as_vec(),
                biases: layer.biases_as_vec(),
                activation: layer.activation_type(),
                input_size: layer.input_size(),
                output_size: layer.output_size(),
                dropout_rate: layer.dropout_rate(),
                num_features: layer.input_size(),
                bn_epsilon: bn_eps,
                bn_momentum: bn_mom,
                extra: layer.layer_metadata(),
            });
        }

        let optimizer_type = if let Some(optimizer) = &self.optimizer {
            optimizer.optimizer_type().to_string()
        } else {
            "sgd".to_string()
        };

        let optimizer_params = if let Some(optimizer) = &self.optimizer {
            OptimizerParams {
                learning_rate: optimizer.learning_rate(),
                momentum: None, // Burada tipe göre doğru değerleri set etmelisiniz
                beta1: None,
                beta2: None,
                epsilon: None,
                weight_decay: None,
            }
        } else {
            OptimizerParams {
                learning_rate: 0.01,
                momentum: None,
                beta1: None,
                beta2: None,
                epsilon: None,
                weight_decay: None,
            }
        };

        let model_data = ModelData {
            layers: layer_data,
            optimizer_type,
            optimizer_params,
            schema_version: "1.0".to_string(),
            cihaz: self.cihaz.kisa().to_string(),
        };

        Ok(serde_json::to_string_pretty(&model_data)?)
    }

    /// Modeli dosyaya kaydeder
    ///
    /// # Parametreler
    /// - `path`: Kaydedilecek dosya yolu
    pub fn save(&self, path: &str) -> Result<()> {
        let json = self.to_json_string()?;

        let mut file = fs::File::create(path)?;
        file.write_all(json.as_bytes())?;

        info!("Model başarıyla kaydedildi: {}", path);
        Ok(())
    }

    /// Model ağırlıklarını JSON metninden yükler (WASM / localStorage önbellek uyumlu).
    pub fn load_from_json_str(&mut self, contents: &str) -> Result<()> {
        let model_data: ModelData = serde_json::from_str(contents)?;

        // Eski modeli temizle
        self.layers.clear();
        self.optimizer = None;
        self.loss = None;
        self.compiled = false;
        self.cihaz = match model_data.cihaz.as_str() {
            "paralel" | "gpu" if Cihaz::paralel_secilebilir() => Cihaz::Paralel,
            _ => Cihaz::Cpu,
        };

        // Katmanları yükle — KatmanKayit üzerinden (artık match bloğu yok)
        let kayit = KatmanKayit::varsayilan();
        for ld in model_data.layers {
            let girdi_boyutu = if ld.input_size > 0 {
                ld.input_size
            } else if self.layers.is_empty() {
                10
            } else {
                self.layers.last().unwrap().output_size()
            };

            let mut ek = std::collections::HashMap::new();
            ek.insert("num_features".into(), ld.num_features as Float);
            ek.insert("bn_epsilon".into(), ld.bn_epsilon);
            ek.insert("bn_momentum".into(), ld.bn_momentum);
            for (k, v) in &ld.extra {
                ek.insert(k.clone(), *v);
            }

            let veri = KatmanYukleVerisi {
                agirliklar: ld.weights,
                biaslar: ld.biases,
                aktivasyon: ld.activation,
                girdi_boyutu,
                cikti_boyutu: ld.output_size,
                dropout_orani: ld.dropout_rate,
                ek_veriler: ek,
            };
            let katman = kayit.json_yukle(&ld.layer_type, &veri)?;
            self.add_boxed(katman);
        }

        // Optimizer'ı yükle
        let optimizer = crate::optimizer::create_optimizer_from_params(
            &model_data.optimizer_type,
            model_data.optimizer_params.learning_rate,
            model_data.optimizer_params.momentum,
            model_data.optimizer_params.beta1,
            model_data.optimizer_params.beta2,
            model_data.optimizer_params.epsilon,
            model_data.optimizer_params.weight_decay,
        )?;

        self.compile_boxed(optimizer);
        Ok(())
    }

    /// Modeli dosyadan yükler
    ///
    /// # Parametreler
    /// - `path`: Yüklenecek dosya yolu
    pub fn load(&mut self, path: &str) -> Result<()> {
        let mut file = fs::File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        self.load_from_json_str(&contents)?;
        info!("Model başarıyla yüklendi: {}", path);
        Ok(())
    }

    /// Girdi verilerinin boyutlarını doğrular
    ///
    /// Eğitim veya değerlendirme verilerinin boyutlarının uyumlu olup olmadığını
    /// kontrol eder. Özellikle x ve y veri setlerinin örnek sayılarının (satır sayısı)
    /// aynı olduğunu doğrular.
    ///
    /// # Parametreler
    /// - `x`: Girdi verileri (boyut: [örnek sayısı, girdi özellik sayısı])
    /// - `y`: Hedef değerler (boyut: [örnek sayısı, çıktı özellik sayısı])
    ///
    /// # Dönüş Değeri
    /// - `Result<()>`: Başarılı ise Ok(()), hata durumunda ilgili hata
    ///
    /// # Hatalar
    /// - Eğer x ve y aynı sayıda örneğe (satır) sahip değilse ShapeMismatch hatası döner
    fn validate_inputs(&self, x: &ArrayView2<Float>, y: &ArrayView2<Float>) -> Result<()> {
        validate_tensor_shapes(x, y, true, false, "Eğitim verileri")
    }

    /// Veri kümesini batch'lere ayırır
    ///
    /// Eğitim veri kümesini belirtilen boyutta mini-batch'lere böler.
    /// Bu, stokastik gradyan inişi (SGD) ve türevleri için gereklidir.
    ///
    /// # Parametreler
    /// - `x`: Girdi verileri (boyut: [örnek sayısı, girdi özellik sayısı])
    /// - `y`: Hedef değerler (boyut: [örnek sayısı, çıktı özellik sayısı])
    /// - `batch_size`: Her batch'teki örnek sayısı
    ///
    /// # Dönüş Değeri
    /// - Batch'lerin (x_batch, y_batch) çiftlerini içeren bir vektör
    ///
    /// # Notlar
    /// - Son batch, tam dolmayabilir (son batch boyutu < batch_size olabilir)
    ///
    /// Tek bir epoch için eğitim yapar.
    ///
    /// # Parametreler
    /// * `x_train` - Giriş veri seti
    /// * `y_train` - Çıkış veri seti
    /// * `batch_size` - Mini batch boyutu
    /// * `metrics` - Kullanılacak metrikler
    /// * `validation_data` - Opsiyonel doğrulama veri seti
    fn train_epoch<'a>(
        &mut self,
        x_train: &'a ArrayView2<'a, Float>,
        y_train: &'a ArrayView2<'a, Float>,
        batch_size: usize,
        _metrics: &[Box<dyn Metric>],
        validation_data: Option<(&'a ArrayView2<'a, Float>, &'a ArrayView2<'a, Float>)>,
    ) -> Result<(Float, Option<Float>)> {
        // Toplam kaybı tutacak değişken
        let mut total_loss = 0.0;

        // Mini batch'ler oluştur
        let n_samples = x_train.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut thread_rng());

        let batch_count = n_samples.div_ceil(batch_size);

        // Her batch için eğitim yap
        for (batch_idx, batch_start) in indices.chunks(batch_size).enumerate() {
            // Batch indeksleri kullanarak batch örnek verilerini oluştur
            let batch_indices: Vec<usize> = batch_start.to_vec();

            // Batch tensörleri oluştur - sahiplenilmiş değerler (owned values)
            let x_batch = stack_selected_rows(x_train, &batch_indices)?;
            let y_batch = stack_selected_rows(y_train, &batch_indices)?;

            // İleri yayılım
            let y_pred = self.forward(&x_batch.view())?;

            let (batch_loss, grad) = self.batch_loss_and_grad(&y_batch.view(), &y_pred.view())?;
            total_loss += batch_loss;

            self.backward(&grad.view())?;

            // Ağırlıkları ve bias'ları güncelle
            self.update()?;

            // Debug: Her 10 batch'te veya son batch'te ilerlemeyi göster
            if batch_idx.is_multiple_of(10) || batch_idx == batch_count - 1 {
                debug!(
                    "Batch {}/{} işlendi, batch kaybı: {:.6}",
                    batch_idx + 1,
                    batch_count,
                    batch_loss
                );
            }
        }

        // Ortalama eğitim kaybını hesapla
        let avg_train_loss = total_loss / batch_count as Float;

        // Doğrulama kaybını hesapla (eğer varsa) — çıkarım modu
        let val_loss = if let Some((x_val, y_val)) = validation_data {
            Some(self.validation_loss(x_val, y_val)?)
        } else {
            None
        };

        Ok((avg_train_loss, val_loss))
    }

    /// Modeli eğitir.
    ///
    /// # Parametreler
    /// - `x_train` / `y_train`: Eğitim verileri
    /// - `epochs`: Epoch sayısı
    /// - `batch_size`: Batch boyutu
    /// - `validation_data`: Doğrulama verileri (isteğe bağlı)
    ///
    /// # Dönüş
    /// `Result<History>` — kayıp ve metrik geçmişi
    pub fn fit<'a>(
        &mut self,
        x_train: &'a ArrayView2<'a, Float>,
        y_train: &'a ArrayView2<'a, Float>,
        epochs: usize,
        batch_size: usize,
        validation_data: Option<(&'a ArrayView2<'a, Float>, &'a ArrayView2<'a, Float>)>,
    ) -> Result<History> {
        self.fit_with_early_stopping(x_train, y_train, epochs, batch_size, validation_data, None)
    }

    /// Erken durdurma destekli eğitim.
    pub fn fit_with_early_stopping<'a>(
        &mut self,
        x_train: &'a ArrayView2<'a, Float>,
        y_train: &'a ArrayView2<'a, Float>,
        epochs: usize,
        batch_size: usize,
        validation_data: Option<(&'a ArrayView2<'a, Float>, &'a ArrayView2<'a, Float>)>,
        early_stopping: Option<&mut EarlyStopping>,
    ) -> Result<History> {
        self.fit_with_checkpoint(
            x_train,
            y_train,
            epochs,
            batch_size,
            validation_data,
            early_stopping,
            None,
            None,
        )
    }

    /// Checkpoint + erken durdurma destekli tam eğitim döngüsü.
    ///
    /// `checkpoint_yolu` verilirse, val_loss (doğrulama yoksa train loss) en iyi
    /// değere ulaştığında model otomatik olarak o dosyaya kaydedilir.
    /// Eğitim bittiğinde veya erken durdurma tetiklendiğinde en iyi checkpoint geri yüklenir.
    pub fn fit_with_checkpoint<'a>(
        &mut self,
        x_train: &'a ArrayView2<'a, Float>,
        y_train: &'a ArrayView2<'a, Float>,
        epochs: usize,
        batch_size: usize,
        validation_data: Option<(&'a ArrayView2<'a, Float>, &'a ArrayView2<'a, Float>)>,
        mut early_stopping: Option<&mut EarlyStopping>,
        checkpoint_yolu: Option<&str>,
        mut epoch_cb: Option<EpochCallback>,
    ) -> Result<History> {
        self.cihaz.hazirla()?;
        self.ensure_compiled()?;
        self.validate_inputs(x_train, y_train)?;

        let mut history = History::new();
        let mut en_iyi_metrik = Float::MAX;
        let val_kullaniliyor = validation_data.is_some();

        for epoch in 0..epochs {
            let (epoch_loss, val_loss) =
                self.train_epoch(x_train, y_train, batch_size, &[], validation_data)?;

            history.train_losses.push(epoch_loss);
            history.train_metrics.push(epoch_loss);
            if let Some(vl) = val_loss {
                history.val_losses.push(vl);
                history.val_metrics.push(vl);
            }

            info!(
                "Epoch {}/{}: loss = {:.6}{}",
                epoch + 1,
                epochs,
                epoch_loss,
                val_loss
                    .map(|v| format!(", val_loss = {:.6}", v))
                    .unwrap_or_default()
            );

            if let Some(cb) = &mut epoch_cb {
                cb(epoch + 1, epochs, epoch_loss, val_loss);
            }

            if let Some(yol) = checkpoint_yolu {
                Self::checkpoint_guncelle(
                    self,
                    yol,
                    epoch,
                    epoch_loss,
                    val_loss,
                    &mut en_iyi_metrik,
                );
            }

            if let (Some(es), Some(vl)) = (early_stopping.as_deref_mut(), val_loss) {
                if es.step(vl) {
                    info!(
                        "Erken durdurma tetiklendi! (epoch={}, val_loss={:.6}, en_iyi_val_loss={:.6})",
                        epoch + 1,
                        vl,
                        es.best_val_loss()
                    );
                    break;
                }
            }
        }

        if let Some(yol) = checkpoint_yolu {
            self.en_iyi_checkpoint_yukle(yol, val_kullaniliyor, en_iyi_metrik)?;
        }

        Ok(history)
    }

    /// Checkpoint metriği: val_loss varsa onu, yoksa train loss kullan.
    fn checkpoint_metrik(epoch_loss: Float, val_loss: Option<Float>) -> Float {
        val_loss.unwrap_or(epoch_loss)
    }

    fn checkpoint_metrik_adi(val_loss: Option<Float>) -> &'static str {
        if val_loss.is_some() {
            "val_loss"
        } else {
            "loss"
        }
    }

    /// val_loss (veya loss) iyileştiyse checkpoint kaydeder.
    fn checkpoint_guncelle(
        &mut self,
        yol: &str,
        epoch: usize,
        epoch_loss: Float,
        val_loss: Option<Float>,
        en_iyi: &mut Float,
    ) {
        let metrik = Self::checkpoint_metrik(epoch_loss, val_loss);
        let metrik_adi = Self::checkpoint_metrik_adi(val_loss);
        if metrik < *en_iyi {
            *en_iyi = metrik;
            match self.save(yol) {
                Ok(_) => info!(
                    "▶ En iyi model güncellendi! \
                     Epoch={} │ en düşük {metrik_adi}: {metrik:.6} │ → '{yol}'",
                    epoch + 1
                ),
                Err(e) => warn!("Checkpoint kaydedilemedi ({yol}): {e}"),
            }
        }
    }

    /// Eğitim sonunda diskteki en iyi checkpoint'i belleğe yükler.
    fn en_iyi_checkpoint_yukle(
        &mut self,
        yol: &str,
        val_kullaniliyor: bool,
        en_iyi_metrik: Float,
    ) -> Result<()> {
        if !std::path::Path::new(yol).exists() {
            return Ok(());
        }
        if val_kullaniliyor && en_iyi_metrik < Float::MAX {
            info!("En iyi model geri yükleniyor: '{yol}' (val_loss={en_iyi_metrik:.6})");
            self.load(yol)?;
        }
        Ok(())
    }

    /// Diskten gelen batch iterator'ü ile tek epoch eğitir.
    ///
    /// Bellekte tüm veri seti tutulmadan, her batch ayrı ayrı işlenir.
    pub fn train_epoch_from_batches<I>(
        &mut self,
        batches: I,
        validation_data: Option<(&Array2<Float>, &Array2<Float>)>,
    ) -> Result<(Float, Option<Float>)>
    where
        I: Iterator<Item = Result<(Array2<Float>, Array2<Float>)>>,
    {
        let mut total_loss = 0.0;
        let mut batch_count = 0usize;

        for batch_result in batches {
            let (x_batch, y_batch) = batch_result?;
            let y_pred = self.forward(&x_batch.view())?;
            let (batch_loss, grad) = self.batch_loss_and_grad(&y_batch.view(), &y_pred.view())?;
            total_loss += batch_loss;

            self.backward(&grad.view())?;
            self.update()?;

            batch_count += 1;
            if batch_count == 1 || batch_count.is_multiple_of(10) {
                debug!("Streaming batch {batch_count} işlendi, kayıp: {batch_loss:.6}");
            }
        }

        if batch_count == 0 {
            return Err(TurkError::DataError(
                "Eğitim verisi batch'i bulunamadı — CSV boş olabilir.".into(),
            ));
        }

        let avg_train_loss = total_loss / batch_count as Float;

        let val_loss = if let Some((x_val, y_val)) = validation_data {
            Some(self.validation_loss(&x_val.view(), &y_val.view())?)
        } else {
            None
        };

        Ok((avg_train_loss, val_loss))
    }

    /// Lazy veri akışı ile checkpoint destekli eğitim.
    ///
    /// `batch_kaynak` her epoch başında yeni bir batch iterator döndürmelidir
    /// (dosya her epoch'ta baştan okunur).
    pub fn fit_streaming_with_checkpoint<I, F>(
        &mut self,
        mut batch_kaynak: F,
        epochs: usize,
        validation_data: Option<(Array2<Float>, Array2<Float>)>,
        mut early_stopping: Option<&mut EarlyStopping>,
        checkpoint_yolu: Option<&str>,
        mut epoch_cb: Option<EpochCallback>,
    ) -> Result<History>
    where
        F: FnMut() -> Result<I>,
        I: Iterator<Item = Result<(Array2<Float>, Array2<Float>)>>,
    {
        self.cihaz.hazirla()?;
        self.ensure_compiled()?;

        let val_ref = validation_data.as_ref().map(|pair| (&pair.0, &pair.1));
        let mut history = History::new();
        let mut en_iyi_metrik = Float::MAX;
        let val_kullaniliyor = validation_data.is_some();

        for epoch in 0..epochs {
            let batches = batch_kaynak()?;
            let (epoch_loss, val_loss) = self.train_epoch_from_batches(batches, val_ref)?;

            history.train_losses.push(epoch_loss);
            history.train_metrics.push(epoch_loss);
            if let Some(vl) = val_loss {
                history.val_losses.push(vl);
                history.val_metrics.push(vl);
            }

            info!(
                "Epoch {}/{}: loss = {:.6}{}",
                epoch + 1,
                epochs,
                epoch_loss,
                val_loss
                    .map(|v| format!(", val_loss = {v:.6}"))
                    .unwrap_or_default()
            );

            if let Some(cb) = &mut epoch_cb {
                cb(epoch + 1, epochs, epoch_loss, val_loss);
            }

            if let Some(yol) = checkpoint_yolu {
                Self::checkpoint_guncelle(
                    self,
                    yol,
                    epoch,
                    epoch_loss,
                    val_loss,
                    &mut en_iyi_metrik,
                );
            }

            if let (Some(es), Some(vl)) = (early_stopping.as_deref_mut(), val_loss) {
                if es.step(vl) {
                    info!(
                        "Erken durdurma tetiklendi! (epoch={}, val_loss={vl:.6}, en_iyi_val_loss={:.6})",
                        epoch + 1,
                        es.best_val_loss()
                    );
                    break;
                }
            }
        }

        if let Some(yol) = checkpoint_yolu {
            self.en_iyi_checkpoint_yukle(yol, val_kullaniliyor, en_iyi_metrik)?;
        }

        Ok(history)
    }

    /// Dropout / BatchNorm eğitim bayrağını tüm katmanlarda ayarlar.
    pub fn set_training_mode(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    /// Doğrulama kaybı — geçici olarak çıkarım moduna geçer.
    fn validation_loss<'a>(
        &mut self,
        x_val: &'a ArrayView2<'a, Float>,
        y_val: &'a ArrayView2<'a, Float>,
    ) -> Result<Float> {
        self.set_training_mode(false);
        let sonuc = (|| {
            let y_val_pred = self.forward(x_val)?;
            let loss_fn = self.loss_fn()?;
            loss_fn.loss(y_val, &y_val_pred.view())
        })();
        self.set_training_mode(true);
        sonuc
    }

    /// Tahmin yapar
    ///
    /// # Parametreler
    /// - `x`: Girdi verileri
    ///
    /// # Dönüş
    /// Model çıktısı
    pub fn predict<'a>(&mut self, x: &'a ArrayView2<'a, Float>) -> Result<Array2<Float>> {
        self.set_training_mode(false);
        let out = self.forward(x);
        self.set_training_mode(true);
        out
    }

    /// Modeli değerlendirir
    ///
    /// # Parametreler
    /// - `x`: Test verileri
    /// - `y`: Gerçek etiketler
    ///
    /// # Dönüş
    /// Model çıktısı
    pub fn evaluate<'a>(
        &mut self,
        x_val: &'a ArrayView2<'a, Float>,
        y_val: &'a ArrayView2<'a, Float>,
    ) -> Result<Float> {
        self.set_training_mode(false);
        let sonuc = (|| {
            let y_pred = self.forward(x_val)?;
            tensor_ops::mse(&y_val.view(), &y_pred.view())
        })();
        self.set_training_mode(true);
        sonuc
    }

    /// AST içerisinden bir katmanı işler ve Sequential modeline ekler
    ///
    /// AST (Abstract Syntax Tree) içinde tanımlanan bir katmanı, Sequential
    /// modeline eklenecek şekilde dönüştürür. Katman tipine göre uygun Dense
    /// veya Dropout nesnesi oluşturur.
    ///
    /// # Parametreler
    /// - `katman`: AST içinde tanımlanmış katman
    ///
    /// # Dönüş Değeri
    /// - `Result<()>`: Başarılı ise Ok(()), hata durumunda ilgili hata
    fn create_layer_from_katman(&mut self, katman: &crate::ast::Katman) -> Result<()> {
        match katman {
            crate::ast::Katman::Dense {
                birimler,
                aktivasyon,
                ..
            } => {
                let activation = match aktivasyon {
                    crate::ast::Aktivasyon::ReLU => Activation::Relu,
                    crate::ast::Aktivasyon::Sigmoid => Activation::Sigmoid,
                    crate::ast::Aktivasyon::Tanh => Activation::Tanh,
                    crate::ast::Aktivasyon::LeakyReLU(alpha) => {
                        Activation::LeakyRelu(dtype::from_f64(*alpha))
                    }
                    crate::ast::Aktivasyon::Softmax => Activation::Softmax,
                    crate::ast::Aktivasyon::Swish => Activation::Swish,
                    crate::ast::Aktivasyon::Yok => Activation::Linear,
                };

                // Katmanı oluştur
                let input_size = if self.layers.is_empty() {
                    // İlk katman için varsayılan giriş boyutu, gerçek uygulamada düzeltilmesi gerekir
                    10
                } else {
                    self.layers.last().unwrap().output_size()
                };

                let kernel_init = Box::new(GlorotUniform::new());
                let bias_init = Box::new(Zeros::new());

                let layer = Dense::new(
                    input_size,
                    *birimler,
                    activation,
                    kernel_init,
                    bias_init,
                    true,
                )?;

                self.add(layer);
            }
            crate::ast::Katman::Dropout { orani } => {
                self.add(Dropout::new(dtype::from_f64(*orani)));
            }
            crate::ast::Katman::BatchNormalization { momentum, epsilon } => {
                let num_features = if self.layers.is_empty() {
                    10
                } else {
                    self.layers.last().unwrap().output_size()
                };
                self.add(BatchNormalization::with_params(
                    num_features,
                    dtype::from_f64(*epsilon),
                    dtype::from_f64(*momentum),
                ));
            }
            _ => {
                warn!("Desteklenmeyen katman tipi: {:?}", katman);
            }
        }

        Ok(())
    }

    /// Optimizer türüne göre optimizer oluşturur ve modeli derler
    ///
    /// AST'de belirtilen optimizer tipine göre uygun optimizer nesnesini
    /// oluşturur ve modeli derler.
    ///
    /// # Parametreler
    /// - `optimizer_tipi`: AST içinde tanımlanmış optimizer tipi ve parametreleri
    fn configure_optimizer_from_type(&mut self, optimizer_tipi: &crate::ast::OptimizerTipi) {
        match optimizer_tipi {
            crate::ast::OptimizerTipi::SGD {
                lr,
                momentum,
                nesterov,
            } => {
                let sgd = SGD::new(
                    dtype::from_f64(*lr),
                    momentum.map(dtype::from_f64),
                    nesterov.unwrap_or(false),
                );
                self.compile_boxed(Box::new(sgd));
            }
            crate::ast::OptimizerTipi::Adam {
                lr,
                beta1,
                beta2,
                epsilon,
            } => {
                let adam = Adam::new(
                    dtype::from_f64(*lr),
                    dtype::from_f64(beta1.unwrap_or(0.9)),
                    dtype::from_f64(beta2.unwrap_or(0.999)),
                    dtype::from_f64(epsilon.unwrap_or(1e-8)),
                );
                self.compile_boxed(Box::new(adam));
            }
            crate::ast::OptimizerTipi::AdamW {
                lr,
                weight_decay,
                beta1,
                beta2,
                epsilon,
            } => {
                let adamw = AdamW::new(
                    dtype::from_f64(*lr),
                    dtype::from_f64(beta1.unwrap_or(0.9)),
                    dtype::from_f64(beta2.unwrap_or(0.999)),
                    dtype::from_f64(epsilon.unwrap_or(1e-8)),
                    dtype::from_f64(weight_decay.unwrap_or(0.01)),
                );
                self.compile_boxed(Box::new(adamw));
            }
            crate::ast::OptimizerTipi::RMSProp { lr, rho, epsilon } => {
                let rmsprop = RMSProp::new(
                    dtype::from_f64(*lr),
                    dtype::from_f64(rho.unwrap_or(0.9)),
                    dtype::from_f64(epsilon.unwrap_or(1e-8)),
                );
                self.compile_boxed(Box::new(rmsprop));
            }
        }
    }

    /// Model tanımını işler
    ///
    /// AST içindeki model özelliklerini işleyerek Sequential modeli yapılandırır.
    /// Katmanlar ve optimizer'ı ayarlar.
    ///
    /// # Parametreler
    /// - `ozellikler`: AST içinde tanımlanmış model özellikleri
    ///
    /// # Dönüş Değeri
    /// - `Result<()>`: Başarılı ise Ok(()), hata durumunda ilgili hata
    fn process_model_definition(
        &mut self,
        ozellikler: &crate::ast::ModelOzellikleri,
    ) -> Result<()> {
        // Katmanları işle
        for katman in &ozellikler.katmanlar {
            self.create_layer_from_katman(katman)?;
        }

        // Optimizer'ı yapılandır
        self.configure_optimizer_from_type(&ozellikler.optimizer);

        Ok(())
    }

    /// AST'den model oluşturur
    ///
    /// TURK dili AST'si (Abstract Syntax Tree) içinden bir Sequential model
    /// oluşturur. AST içindeki model tanımını, katmanları ve optimizer'ı
    /// çözümleyerek uygun bir model yapılandırır.
    ///
    /// # Parametreler
    /// - `ast`: TURK dili program AST'si
    ///
    /// # Dönüş Değeri
    /// - `Result<Self>`: Başarılı ise oluşturulan model, hata durumunda ilgili hata
    ///
    /// # Örnek
    ///
    /// ```no_run
    /// use turk::model::Sequential;
    /// use turk::ast::{Program, Deyim, ModelOzellikleri, Katman, Aktivasyon, OptimizerTipi, KayipFonksiyonu};
    ///
    /// let program = Program {
    ///     deyimler: vec![Deyim::ModelTanimi {
    ///         isim: "net".into(),
    ///         ozellikler: ModelOzellikleri {
    ///             katmanlar: vec![Katman::Dense {
    ///                 birimler: 10,
    ///                 aktivasyon: Aktivasyon::ReLU,
    ///                 kernel_initializer: None,
    ///                 bias_initializer: None,
    ///             }],
    ///             optimizer: OptimizerTipi::SGD { lr: 0.01, momentum: None, nesterov: None },
    ///             kayip_fonksiyonu: KayipFonksiyonu::MSE,
    ///             batch_boyutu: 32,
    ///             epoch_sayisi: 10,
    ///             metrikler: None,
    ///             erken_durdurma: None,
    ///         },
    ///     }],
    /// };
    /// let model = Sequential::from_ast(&program).unwrap();
    /// ```
    pub fn from_ast(ast: &crate::ast::Program) -> Result<Self> {
        let mut model = Sequential::new();
        let mut found_model = false;

        // Deyimleri işle
        for deyim in &ast.deyimler {
            use crate::ast::Deyim;

            if let Deyim::ModelTanimi {
                isim: _,
                ozellikler,
            } = deyim
            {
                found_model = true;
                model.process_model_definition(ozellikler)?;
                break;
            }
        }

        if !found_model {
            warn!("AST içinde ModelTanimi bulunamadı, boş model döndürülüyor.");
        }

        if model.optimizer.is_none() {
            warn!("AST'den model oluşturuldu ancak optimizer belirtilmedi, varsayılan SGD (lr=0.01) kullanılıyor.");
            model.compile_boxed(Box::new(SGD::new(0.01, None, false)));
        }

        Ok(model)
    }
}

impl Model for Sequential {
    fn forward<'a>(&mut self, input: &'a ArrayView2<'a, Float>) -> Result<Array2<Float>> {
        self.forward(input)
    }

    fn backward<'a>(&mut self, grad: &'a ArrayView2<'a, Float>) -> Result<Array2<Float>> {
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

    fn fit<'a>(
        &mut self,
        x_train: &'a ArrayView2<'a, Float>,
        y_train: &'a ArrayView2<'a, Float>,
        epochs: usize,
        batch_size: usize,
        validation_data: Option<(&'a ArrayView2<'a, Float>, &'a ArrayView2<'a, Float>)>,
    ) -> Result<History> {
        self.fit(x_train, y_train, epochs, batch_size, validation_data)
    }

    fn predict<'a>(&mut self, x: &'a ArrayView2<'a, Float>) -> Result<Array2<Float>> {
        self.predict(x)
    }

    fn evaluate<'a>(
        &mut self,
        x_val: &'a ArrayView2<'a, Float>,
        y_val: &'a ArrayView2<'a, Float>,
    ) -> Result<Float> {
        self.evaluate(x_val, y_val)
    }
}

impl fmt::Display for Sequential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Sequential Model")?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "  Layer {}: {}", i, layer.to_string())?;
        }
        Ok(())
    }
}

/// Ortalama Kare Hata (MSE) hesaplama fonksiyonu
///
/// Gerçek değerler ile tahmin edilen değerler arasındaki ortalama kare hatasını hesaplar.
/// MSE değeri, eğitim için yaygın kullanılan bir kayıp fonksiyonudur.
///
/// # Formül
/// MSE = (1/n) * Σ(y_true - y_pred)²
///
/// # Parametreler
/// - `y_true`: Gerçek değerler
#[cfg(test)]
mod tests {
    use super::*;
    use crate::aktivasyon::Activation;
    use crate::initializers::{GlorotUniform, Zeros};
    use crate::layer::BatchNormalization;
    use crate::metrics::Accuracy;
    use ndarray::{arr2, Array2};

    fn dense(i: usize, o: usize, act: Activation) -> Dense {
        Dense::new(
            i,
            o,
            act,
            Box::new(GlorotUniform::new()),
            Box::new(Zeros::new()),
            true,
        )
        .unwrap()
    }

    fn sgd(lr: Float) -> SGD {
        SGD::new(lr, None, false)
    }

    fn create_test_model() -> Sequential {
        let mut model = Sequential::new();
        model
            .add(dense(2, 3, Activation::Relu))
            .add(dense(3, 1, Activation::Sigmoid))
            .compile(sgd(0.1));
        model
    }

    #[test]
    fn test_sequential_model() {
        let mut model = Sequential::new();
        model
            .add(dense(784, 128, Activation::Relu))
            .add(dense(128, 10, Activation::Softmax))
            .compile(sgd(0.01));

        let input = Array2::zeros((1, 784));
        let output = model.forward(&input.view()).unwrap();
        assert_eq!(output.shape(), &[1, 10]);
    }

    #[test]
    fn test_model_compile() {
        let mut model = Sequential::new();
        model
            .add(dense(10, 5, Activation::Relu))
            .compile(sgd(0.001));

        let input = Array2::zeros((1, 10));
        let output = model.forward(&input.view()).unwrap();
        assert_eq!(output.shape(), &[1, 5]);
    }

    #[test]
    fn test_model_serialization() {
        // Geçici bir dosya yolu oluştur
        let temp_file = std::env::temp_dir().join("test_model.json");
        let path_str = temp_file.to_str().unwrap();

        // İlk modeli oluştur ve kaydet
        let mut model = Sequential::new();
        model
            .add(dense(3, 2, Activation::Relu))
            .add(dense(2, 1, Activation::Sigmoid))
            .compile(sgd(0.1));
        model.save(path_str).unwrap();

        // Yeni bir model oluştur ve kaydedilen modeli yükle
        let mut loaded = Sequential::new();
        // Önce aynı topolojideki katmanları ekle, yükleme sırasında ağırlıklar değiştirilecek
        loaded
            .add(dense(3, 2, Activation::Relu))
            .add(dense(2, 1, Activation::Sigmoid))
            .compile(sgd(0.1));
        loaded.load(path_str).unwrap();

        let input = Array2::zeros((1, 3));
        let output = loaded.forward(&input.view()).unwrap();
        assert_eq!(output.shape(), &[1, 1]);

        // Test sonrası dosyayı temizle
        let _ = std::fs::remove_file(temp_file);
    }

    #[test]
    fn test_model_adamw() {
        let mut model = Sequential::new();
        model
            .add(dense(10, 5, Activation::Relu))
            .compile(sgd(0.001));

        let x = Array2::zeros((1, 10));
        let output = model.forward(&x.view()).unwrap();
        assert_eq!(output.shape(), &[1, 5]);
    }

    #[test]
    fn test_model_rmsprop() {
        let mut model = Sequential::new();
        model
            .add(dense(10, 5, Activation::Relu))
            .compile(sgd(0.001));

        let x = Array2::zeros((1, 10));
        let output = model.forward(&x.view()).unwrap();
        assert_eq!(output.shape(), &[1, 5]);
    }

    #[test]
    fn test_model_fit() {
        let mut model = Sequential::new();
        model
            .add(dense(2, 3, Activation::Relu))
            .add(dense(3, 1, Activation::Sigmoid))
            .compile(sgd(0.01));
        let x = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y = arr2(&[[0.0], [1.0], [1.0], [0.0]]);
        let history = model.fit(&x.view(), &y.view(), 100, 1, None).unwrap();
        let first = history.train_losses.first().copied().unwrap_or(0.0);
        let last = history.train_losses.last().copied().unwrap_or(0.0);
        assert!(last <= first + 0.5);
        let y_pred = model.predict(&x.view()).unwrap();
        assert_eq!(y_pred.shape(), y.shape());
    }

    #[test]
    fn test_model_evaluate() {
        let mut model = create_test_model();
        let x = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y = arr2(&[[0.0], [1.0], [1.0], [0.0]]);
        let mse = model.evaluate(&x.view(), &y.view()).unwrap();
        assert!(mse >= 0.0);
    }

    #[test]
    fn test_model_summary() {
        let model = create_test_model();
        let summary = model.summary();
        assert!(summary.contains("Dense"));
    }

    #[test]
    fn test_model_compilation_from_ast() {
        use crate::ast::{
            Aktivasyon, Deyim, Katman, KayipFonksiyonu, ModelOzellikleri, OptimizerTipi, Program,
        };
        let program = Program {
            deyimler: vec![Deyim::ModelTanimi {
                isim: "test_model".to_string(),
                ozellikler: ModelOzellikleri {
                    katmanlar: vec![Katman::Dense {
                        birimler: 10,
                        aktivasyon: Aktivasyon::ReLU,
                        kernel_initializer: None,
                        bias_initializer: None,
                    }],
                    optimizer: OptimizerTipi::SGD {
                        lr: 0.01,
                        momentum: None,
                        nesterov: None,
                    },
                    kayip_fonksiyonu: KayipFonksiyonu::MSE,
                    batch_boyutu: 32,
                    epoch_sayisi: 10,
                    metrikler: None,
                    erken_durdurma: None,
                },
            }],
        };
        let model = Sequential::from_ast(&program);
        assert!(model.is_ok());
    }

    #[test]
    fn test_metrics() {
        let mut model = create_test_model();
        let input = Array2::<Float>::ones((5, 2));
        let output = model.forward(&input.view()).unwrap();
        let accuracy = Accuracy::new();
        let targets = Array2::<Float>::ones((5, 1));
        let acc_value = accuracy.compute(&targets.view(), &output.view());
        assert!((0.0..=1.0).contains(&acc_value));
    }

    #[test]
    fn test_model_initialization() {
        let empty_model = Sequential::new();
        assert!(empty_model.layers.is_empty());
        assert!(empty_model.optimizer.is_none());
        assert!(!empty_model.compiled);
        let mut model = Sequential::new();
        model.add(dense(10, 5, Activation::Relu));
        assert_eq!(model.layers.len(), 1);
        model.compile(sgd(0.01));
        assert!(model.optimizer.is_some());
        assert!(model.compiled);
    }

    #[test]
    fn test_model_forward() {
        let mut model = create_test_model();
        let input = Array2::zeros((1, 2));
        let output = model.forward(&input.view()).unwrap();
        assert_eq!(output.shape(), &[1, 1]);
    }

    #[test]
    fn test_model_backward() {
        // Sigmoid her zaman türevlendirilebilir (s*(1-s) > 0), sıfır gradyan riski yok.
        // Linear da güvenli: dz = grad_output, ağırlıklar sıfırdan çok uzak.
        let mut model = Sequential::new();
        model
            .add(dense(2, 4, Activation::Sigmoid))
            .add(dense(4, 1, Activation::Sigmoid))
            .compile(sgd(0.1));

        // Büyük pozitif girdiler → ağırlıksız bölgede bile Sigmoid türevi sıfırdan uzak
        let input = arr2(&[[2.0, -2.0]]);
        let output = model.forward(&input.view()).unwrap();

        // Hedef çıktıyla tam zıt: kayıp gradyanı (output - target) kesinlikle sıfır değil
        let target = if output[[0, 0]] > 0.5 {
            arr2(&[[0.0]])
        } else {
            arr2(&[[1.0]])
        };
        let loss_grad = &output - &target.view();

        let grad = model.backward(&loss_grad.view()).unwrap();

        // Şekil korunmalı
        assert_eq!(grad.shape(), input.shape());
        // Sigmoid → her katmanda türev > 0, giriş gradyanı kesinlikle sıfırdan büyük
        assert!(
            grad.iter().any(|&x| x.abs() > 1e-12),
            "Geri yayılım gradyanı sıfır çıktı; değerler: {:?}",
            grad
        );
    }

    #[test]
    fn test_optimizer_updates() {
        // Sigmoid: her zaman türevlendirilebilir → dead neuron riski yok.
        // Yüksek lr (1.0) → ağırlık değişimi kesin görülür.
        let mut model = Sequential::new();
        model
            .add(dense(2, 4, Activation::Sigmoid))
            .add(dense(4, 1, Activation::Sigmoid))
            .compile(sgd(1.0));

        // backward() içinde zaten optimizer.update() çağrılıyor;
        // başlangıç anlık görüntüsünü forward'dan ÖNCE al.
        let initial_weights: Vec<Array2<Float>> = model
            .layers
            .iter()
            .map(|l| l.get_weights().unwrap_or_else(|_| Array2::zeros((0, 0))))
            .collect();

        // [[2.0, -2.0]]: Sigmoid'in doygunluk bölgesinden uzak, türev maksimuma yakın.
        let input = arr2(&[[2.0, -2.0]]);
        let output = model.forward(&input.view()).unwrap();

        // Hedefi çıktıyla zıt yönde seç → loss gradyanı (output - target) sıfır olmaz.
        let target = if output[[0, 0]] > 0.5 {
            arr2(&[[0.0]])
        } else {
            arr2(&[[1.0]])
        };
        let loss_grad = &output - &target.view();

        // backward() içinde real gradyanlarla optimizer.update() uygulanıyor.
        model.backward(&loss_grad.view()).unwrap();

        // backward sonrası en az bir katmanın ağırlıkları değişmiş olmalı.
        let changed_after_backward = model.layers.iter().enumerate().any(|(i, l)| {
            if l.param_count() == 0 {
                return false;
            }
            let w = l.get_weights().unwrap_or_else(|_| Array2::zeros((0, 0)));
            (&w - &initial_weights[i])
                .mapv(Float::abs)
                .iter()
                .any(|&d| d > 1e-12)
        });

        assert!(
            changed_after_backward,
            "backward() sonrası ağırlıklar güncellenmemiş. \
             Gradyan sıfır olabilir; katman çıktıları: {:?}",
            model
                .layers
                .iter()
                .map(|l| l.output_size())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_invalid_input_dimensions() {
        let mut model = create_test_model();
        let invalid_input = Array2::<Float>::zeros((1, 4));
        let result = model.forward(&invalid_input.view());
        assert!(result.is_err());
        if let Err(TurkError::ShapeMismatch(_)) = result {
        } else {
            panic!("ShapeMismatch bekleniyordu");
        }
    }

    #[test]
    fn test_uncompiled_model_errors() {
        let mut model = Sequential::new();
        model.add(dense(2, 3, Activation::Relu));
        let input = Array2::<Float>::zeros((1, 2));
        assert!(model.forward(&input.view()).is_err());
        let target = Array2::<Float>::zeros((1, 1));
        assert!(model
            .fit(&input.view(), &target.view(), 1, 1, None)
            .is_err());
    }

    #[test]
    fn test_error_types() {
        let temp_path = std::env::temp_dir().join("test_model_error.json");
        std::fs::write(&temp_path, r#"{"invalid":"data"}"#).unwrap();
        let mut model = Sequential::new();
        assert!(model.load(temp_path.to_str().unwrap()).is_err());
        let _ = std::fs::remove_file(&temp_path);

        let mut model = create_test_model();
        let input = Array2::<Float>::zeros((2, 2));
        let bad_target = Array2::<Float>::zeros((3, 1));
        let result = model.fit(&input.view(), &bad_target.view(), 1, 1, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_hyperparameter_sensitivity() {
        let x = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y = arr2(&[[0.0], [1.0], [1.0], [0.0]]);
        let lrs = [0.001f64, 0.5];
        let losses: Vec<Float> = lrs
            .iter()
            .map(|&lr| {
                let mut m = Sequential::new();
                m.add(dense(2, 4, Activation::Relu))
                    .add(dense(4, 1, Activation::Sigmoid))
                    .compile(sgd(lr));
                m.fit(&x.view(), &y.view(), 50, 2, None)
                    .unwrap()
                    .last_train_loss()
                    .unwrap_or(0.0)
            })
            .collect();
        assert!(
            (losses[0] - losses[1]).abs() > 1e-6,
            "farklı lr farklı kayıp vermeli"
        );
    }

    #[test]
    fn test_full_training_cycle() {
        let x = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y = arr2(&[[0.0], [1.0], [1.0], [0.0]]);
        let mut model = Sequential::new();
        model
            .add(dense(2, 8, Activation::Relu))
            .add(dense(8, 1, Activation::Sigmoid))
            .compile(sgd(0.3));
        let initial_loss =
            tensor_ops::mse(&y.view(), &model.forward(&x.view()).unwrap().view()).unwrap();
        let history = model.fit(&x.view(), &y.view(), 3000, 4, None).unwrap();
        let final_loss = history.last_train_loss().unwrap_or(Float::INFINITY);
        assert!(
            final_loss < initial_loss,
            "kayıp düşmeli: {final_loss} < {initial_loss}"
        );
        assert!(final_loss < 0.25, "kayıp yeterince düşmeli: {final_loss}");
    }

    #[test]
    fn test_model_save_load() {
        let model_path = std::env::temp_dir().join("test_integration_model.json");
        let path_str = model_path.to_str().unwrap();
        let mut original = create_test_model();
        original.save(path_str).unwrap();
        let mut loaded = Sequential::new();
        loaded
            .add(dense(2, 3, Activation::Relu))
            .add(dense(3, 1, Activation::Sigmoid))
            .compile(sgd(0.1));
        loaded.load(path_str).unwrap();
        let input = arr2(&[[0.5, 0.5]]);
        let out_orig = original.forward(&input.view()).unwrap();
        let out_load = loaded.forward(&input.view()).unwrap();
        assert_eq!(out_orig.shape(), out_load.shape());
        let _ = std::fs::remove_file(model_path);
    }

    // ──── Yeni özellik testleri: History, EarlyStopping, BatchNormalization ────

    #[test]
    fn test_history_struct() {
        let mut model = create_test_model();
        let x = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y = arr2(&[[0.0], [1.0], [1.0], [0.0]]);
        let history = model.fit(&x.view(), &y.view(), 5, 2, None).unwrap();
        assert_eq!(history.epochs_run(), 5);
        assert_eq!(history.train_losses.len(), 5);
        assert_eq!(history.train_metrics.len(), 5);
        assert!(history.val_losses.is_empty());
        assert!(history.last_train_loss().is_some());
        for &l in &history.train_losses {
            assert!(l >= 0.0, "negatif kayıp: {l}");
        }
    }

    #[test]
    fn test_history_with_validation() {
        let mut model = create_test_model();
        let x = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y = arr2(&[[0.0], [1.0], [1.0], [0.0]]);
        let xv = arr2(&[[0.0, 0.0], [1.0, 1.0]]);
        let yv = arr2(&[[0.0], [0.0]]);
        let history = model
            .fit(&x.view(), &y.view(), 3, 2, Some((&xv.view(), &yv.view())))
            .unwrap();
        assert_eq!(history.val_losses.len(), 3);
        for &v in &history.val_losses {
            assert!(v >= 0.0);
        }
        let (tr, vl, ep) = history.en_iyi_val_kayitlari().unwrap();
        assert!((1..=3).contains(&ep));
        assert_eq!(
            vl,
            history
                .val_losses
                .iter()
                .cloned()
                .reduce(Float::min)
                .unwrap()
        );
        assert_eq!(tr, history.train_losses[ep - 1]);
    }

    #[test]
    fn test_en_iyi_val_kayitlari() {
        let h = History {
            train_losses: vec![0.5, 0.3, 0.1],
            val_losses: vec![0.4, 0.25, 0.9],
            train_metrics: vec![],
            val_metrics: vec![],
        };
        let (tr, vl, ep) = h.en_iyi_val_kayitlari().unwrap();
        assert_eq!(ep, 2);
        assert!((tr - 0.3).abs() < 1e-9);
        assert!((vl - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_early_stopping_triggers() {
        let mut es = EarlyStopping::new(3, 1e-4);
        assert!(!es.step(0.5));
        assert!(!es.step(0.3));
        assert!(!es.step(0.2));
        assert!(!es.step(0.2001));
        assert!(!es.step(0.2001));
        assert!(es.step(0.2001), "patience=3 dolunca tetiklenmeli");
    }

    #[test]
    fn test_early_stopping_resets() {
        let mut es = EarlyStopping::new(2, 0.0);
        es.step(1.0);
        es.step(1.0);
        es.reset();
        assert!(!es.step(1.0));
    }

    #[test]
    fn test_checkpoint_val_loss_not_train() {
        use tempfile::NamedTempFile;

        let mut model = Sequential::new();
        model
            .add(dense(1, 1, Activation::Sigmoid))
            .compile(sgd(0.5));

        // 4 satir: train ve val ayri — overfit senaryosu icin yuksek lr
        let x = arr2(&[[0.0], [0.0], [1.0], [1.0]]);
        let y = arr2(&[[0.0], [1.0], [1.0], [0.0]]);
        let xv = arr2(&[[0.5]]);
        let yv = arr2(&[[0.5]]);

        let ckpt = NamedTempFile::new().unwrap();
        let yol = ckpt.path().to_str().unwrap();

        let mut es = EarlyStopping::new(2, 0.0);
        let history = model
            .fit_with_checkpoint(
                &x.view(),
                &y.view(),
                20,
                2,
                Some((&xv.view(), &yv.view())),
                Some(&mut es),
                Some(yol),
                None,
            )
            .unwrap();

        assert!(history.epochs_run() >= 1);
        // En iyi checkpoint val_loss'a gore kaydedilmeli (epoch 1 genelde en iyi val)
        let best_val = history
            .val_losses
            .iter()
            .cloned()
            .reduce(Float::min)
            .unwrap();
        assert!(
            es.best_val_loss() <= best_val + 1e-9,
            "EarlyStopping en iyi val_loss takip etmeli"
        );
    }

    #[test]
    fn test_fit_with_early_stopping() {
        let mut model = Sequential::new();
        model
            .add(dense(2, 4, Activation::Sigmoid))
            .add(dense(4, 1, Activation::Sigmoid))
            .compile(sgd(0.01));
        let x = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y = arr2(&[[0.0], [1.0], [1.0], [0.0]]);
        let xv = x.clone();
        let yv = y.clone();
        let mut es = EarlyStopping::new(2, 1e-6);
        let history = model
            .fit_with_early_stopping(
                &x.view(),
                &y.view(),
                100,
                4,
                Some((&xv.view(), &yv.view())),
                Some(&mut es),
            )
            .unwrap();
        assert!(history.epochs_run() >= 1);
        assert!(history.epochs_run() <= 100);
    }

    #[test]
    fn test_batch_normalization_forward() {
        let mut bn = BatchNormalization::new(4);
        bn.set_training(true);
        let input = arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]);
        let output = bn.forward(&input.view()).unwrap();
        assert_eq!(output.shape(), input.shape());
        for col in 0..4 {
            let mu: Float = output.column(col).iter().sum::<Float>() / 3.0;
            assert!(
                mu.abs() < 1e-6,
                "normalize sonrası ortalama ≈0 olmalı: {mu}"
            );
        }
    }

    #[test]
    fn test_batch_normalization_backward() {
        let mut bn = BatchNormalization::new(3);
        let input = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        bn.forward(&input.view()).unwrap();
        let grad_out = Array2::ones((3, 3));
        let (grad_in, d_gamma, d_beta) = bn.backward(&input.view(), &grad_out.view()).unwrap();
        assert_eq!(grad_in.shape(), input.shape());
        assert_eq!(d_gamma.shape(), &[1, 3]);
        assert_eq!(d_beta.shape(), &[1, 3]);
    }

    #[test]
    fn test_batch_normalization_meta() {
        let bn = BatchNormalization::new(16);
        assert_eq!(bn.param_count(), 32);
        assert_eq!(bn.input_size(), 16);
        assert_eq!(bn.output_size(), 16);
        assert_eq!(bn.layer_type(), "BatchNormalization");
    }

    #[test]
    fn test_model_with_batch_norm() {
        let mut model = Sequential::new();
        model
            .add(dense(2, 4, Activation::Relu))
            .add(BatchNormalization::new(4))
            .add(dense(4, 1, Activation::Sigmoid))
            .compile(sgd(0.05));
        let x = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let y = arr2(&[[0.0], [1.0], [1.0], [0.0]]);
        let h = model.fit(&x.view(), &y.view(), 5, 4, None).unwrap();
        assert_eq!(h.epochs_run(), 5);
        assert!(h.last_train_loss().unwrap() >= 0.0);
    }
}

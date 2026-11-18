//! WASM Browser Playground / ALPER Kokpit köprüsü.
//!
//! Disk erişimi yoktur; `veri_yukle("xor")` gibi çağrılar sentetik bellek verisine
//! yönlendirilir. Epoch metrikleri ve model özeti JSON olarak döner.
//! Eğitim sonrası `tahmin_et` ile canlı çıkarım yapılabilir.
//! `AlperBeyin` — harici LLM olmadan niyet sınıflandırma ağı (TF-IDF/BoW vektörleri).

use crate::aktivasyon::Activation;
use crate::dil::{EgitimIlerleme, TurkDil};
use crate::dtype::{self, Float};
use crate::initializers::{GlorotUniform, Zeros};
use crate::layer::Dense;
use crate::loss::CrossEntropy;
use crate::metrics::{Accuracy, Metric};
use crate::model::Sequential;
use crate::optimizer::Adam;
use ndarray::Array2;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Serialize)]
struct EpochMetrik {
    epoch: usize,
    loss: f64,
    accuracy: f64,
}

#[derive(Debug, Clone, Serialize)]
struct WasmCalistirSonuc {
    logs: Vec<String>,
    metrics: Vec<EpochMetrik>,
    model_summary: String,
    input_dim: usize,
    output_dim: usize,
}

#[derive(Debug, Clone, Serialize)]
struct WasmTahminSonuc {
    outputs: Vec<f64>,
    predicted_class: usize,
    confidence: f64,
    label: String,
}

/// Tarayıcıda TÜRK DSL çalıştırıcı.
#[wasm_bindgen]
pub struct TurkWasmRunner {
    dil: TurkDil,
}

#[wasm_bindgen]
impl TurkWasmRunner {
    /// Yeni WASM koşucu.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            dil: TurkDil::new(),
        }
    }

    /// `.turk` DSL kodunu satır satır çalıştırır.
    ///
    /// Dönüş: `{ logs, metrics, model_summary, input_dim, output_dim }`
    #[wasm_bindgen]
    pub fn calistir_kod(&mut self, kod: &str) -> Result<JsValue, JsValue> {
        let (tx, rx) = std::sync::mpsc::channel::<EgitimIlerleme>();
        self.dil.egitim_kanali_ayarla(tx);

        let mut logs: Vec<String> = Vec::new();
        let mut metrics: Vec<EpochMetrik> = Vec::new();
        let mut satir_no = 0usize;

        for satir in kod.lines() {
            satir_no += 1;
            let satir = satir.trim();
            if satir.is_empty() || satir.starts_with('#') || satir.starts_with("//") {
                continue;
            }

            logs.push(format!("› {satir}"));
            match self.dil.calistir_satir(satir, satir_no) {
                Ok(val) => {
                    if let Some(metin) = value_log(&val) {
                        logs.push(metin);
                    }
                }
                Err(e) => {
                    let mesaj = format!("Hata (satır {satir_no}): {e}");
                    logs.push(mesaj.clone());
                    // Kalan ilerleme mesajlarını boşalt
                    while let Ok(olay) = rx.try_recv() {
                        logs.push(ilerleme_log(&olay));
                    }
                    return Err(JsValue::from_str(&mesaj));
                }
            }

            while let Ok(olay) = rx.try_recv() {
                if let EgitimIlerleme::Epoch {
                    epoch, train_loss, ..
                } = &olay
                {
                    metrics.push(EpochMetrik {
                        epoch: *epoch,
                        loss: *train_loss,
                        accuracy: 0.0, // eğitim sonrası doldurulur
                    });
                }
                logs.push(ilerleme_log(&olay));
            }
        }

        // Son kalan kanal mesajları
        while let Ok(olay) = rx.try_recv() {
            if let EgitimIlerleme::Epoch {
                epoch, train_loss, ..
            } = &olay
            {
                metrics.push(EpochMetrik {
                    epoch: *epoch,
                    loss: *train_loss,
                    accuracy: 0.0,
                });
            }
            logs.push(ilerleme_log(&olay));
        }

        let accuracy = self.dataset_accuracy().unwrap_or(0.0);
        for m in &mut metrics {
            m.accuracy = accuracy;
        }

        let (input_dim, output_dim) = self.model_dims();
        let model_summary = self
            .dil
            .model
            .as_ref()
            .map(|m| m.summary())
            .unwrap_or_else(|| "(model yok)".into());

        let sonuc = WasmCalistirSonuc {
            logs,
            metrics,
            model_summary,
            input_dim,
            output_dim,
        };

        serde_wasm_bindgen::to_value(&sonuc)
            .map_err(|e| JsValue::from_str(&format!("JSON serileştirme hatası: {e}")))
    }

    /// Eğitilmiş modelin girdi boyutu (yoksa 0).
    #[wasm_bindgen]
    pub fn girdi_boyutu(&self) -> usize {
        self.model_dims().0
    }

    /// Eğitilmiş modelin çıktı boyutu (yoksa 0).
    #[wasm_bindgen]
    pub fn cikti_boyutu(&self) -> usize {
        self.model_dims().1
    }

    /// Kullanıcı girdileri üzerinde canlı çıkarım (`set_training(false)`).
    ///
    /// `inputs_json`: örn. `"[1.0, 0.0]"` — model girdi boyutuna eşit uzunlukta sayı dizisi.
    /// Dönüş JSON: `{ outputs, predicted_class, confidence, label }`
    #[wasm_bindgen]
    pub fn tahmin_et(&mut self, inputs_json: &str) -> Result<String, JsValue> {
        let inputs: Vec<f64> = serde_json::from_str(inputs_json).map_err(|e| {
            JsValue::from_str(&format!("Girdi JSON parse hatası: {e} (örnek: [1.0, 0.0])"))
        })?;

        let model =
            self.dil.model.as_mut().ok_or_else(|| {
                JsValue::from_str("Önce bir model eğitin (Modeli Eğit / Çalıştır).")
            })?;

        let in_dim = model.layers.first().map(|l| l.input_size()).unwrap_or(0);
        if in_dim == 0 {
            return Err(JsValue::from_str("Modelde girdi katmanı yok."));
        }
        if inputs.len() != in_dim {
            return Err(JsValue::from_str(&format!(
                "Girdi boyutu uyuşmuyor: beklenen {in_dim}, gelen {}",
                inputs.len()
            )));
        }

        let data: Vec<Float> = inputs.iter().copied().map(dtype::from_f64).collect();
        let x = Array2::from_shape_vec((1, in_dim), data)
            .map_err(|e| JsValue::from_str(&format!("Tensor şekil hatası: {e}")))?;

        let pred = model
            .predict(&x.view())
            .map_err(|e| JsValue::from_str(&format!("Tahmin hatası: {e}")))?;

        let outputs: Vec<f64> = pred.iter().copied().map(dtype::to_f64).collect();
        let (predicted_class, confidence) = classify_outputs(&outputs);
        let label = format!("Sınıf {predicted_class} ({confidence:.4})");

        let sonuc = WasmTahminSonuc {
            outputs,
            predicted_class,
            confidence,
            label,
        };
        serde_json::to_string(&sonuc)
            .map_err(|e| JsValue::from_str(&format!("JSON serileştirme hatası: {e}")))
    }

    /// JS uyumluluk takma adı — `tahmin_et` ile aynı.
    #[wasm_bindgen(js_name = predict)]
    pub fn predict_js(&mut self, inputs_json: &str) -> Result<String, JsValue> {
        self.tahmin_et(inputs_json)
    }

    /// Eğitilmiş model ağırlıklarını pretty JSON olarak döndürür (tarayıcıda indirme için).
    #[wasm_bindgen]
    pub fn model_json(&self) -> Result<String, JsValue> {
        let model = self
            .dil
            .model
            .as_ref()
            .ok_or_else(|| JsValue::from_str("İndirilecek model yok — önce eğitin."))?;
        model
            .to_json_string()
            .map_err(|e| JsValue::from_str(&format!("Model JSON hatası: {e}")))
    }
}

impl Default for TurkWasmRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkWasmRunner {
    fn model_dims(&self) -> (usize, usize) {
        match self.dil.model.as_ref() {
            Some(m) if !m.layers.is_empty() => {
                let i = m.layers.first().map(|l| l.input_size()).unwrap_or(0);
                let o = m.layers.last().map(|l| l.output_size()).unwrap_or(0);
                (i, o)
            }
            _ => (0, 0),
        }
    }

    /// Aktif sentetik veri (xor / kume) üzerinde ikili doğruluk.
    fn dataset_accuracy(&mut self) -> Option<f64> {
        let (x, y) = self.dil.aktif_degerlendirme_verisi()?;
        let model = self.dil.model.as_mut()?;
        let pred = model.predict(&x.view()).ok()?;
        let acc = Accuracy::new().compute(&y.view(), &pred.view());
        Some(crate::dtype::to_f64(acc))
    }
}

/// Tek çıkış → 0.5 eşiği; çoklu çıkış → argmax.
fn classify_outputs(outputs: &[f64]) -> (usize, f64) {
    if outputs.is_empty() {
        return (0, 0.0);
    }
    if outputs.len() == 1 {
        let p = outputs[0];
        let cls = if p >= 0.5 { 1 } else { 0 };
        let conf = if cls == 1 { p } else { 1.0 - p };
        (cls, conf)
    } else {
        let (idx, &val) = outputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        (idx, val)
    }
}

fn value_log(val: &crate::dil::Value) -> Option<String> {
    match val {
        crate::dil::Value::Text(t) => Some(t.clone()),
        crate::dil::Value::Number(n) => Some(format!("→ {n:.6}")),
        crate::dil::Value::Boolean(b) => Some(format!("→ {b}")),
        crate::dil::Value::None | crate::dil::Value::Model(_) => None,
    }
}

fn ilerleme_log(olay: &EgitimIlerleme) -> String {
    match olay {
        EgitimIlerleme::Indeksleme { satir } => format!("indeksleme: {satir} satır"),
        EgitimIlerleme::Durum { mesaj } => mesaj.clone(),
        EgitimIlerleme::Epoch {
            epoch,
            toplam,
            train_loss,
            val_loss,
        } => match val_loss {
            Some(v) => format!("epoch {epoch}/{toplam}: loss={train_loss:.6}, val={v:.6}"),
            None => format!("epoch {epoch}/{toplam}: loss={train_loss:.6}"),
        },
        EgitimIlerleme::Tamamlandi {
            epoch_sayisi,
            en_iyi_train,
            en_iyi_val,
        } => match en_iyi_val {
            Some(v) => format!(
                "tamamlandı: {epoch_sayisi} epoch, en iyi loss={en_iyi_train:.6}, val={v:.6}"
            ),
            None => format!("tamamlandı: {epoch_sayisi} epoch, en iyi loss={en_iyi_train:.6}"),
        },
        EgitimIlerleme::Hata(e) => format!("eğitim hatası: {e}"),
    }
}

#[derive(Debug, Clone, Serialize)]
struct AlperEgitimSonuc {
    metrics: Vec<EpochMetrik>,
    input_dim: usize,
    output_dim: usize,
    sample_count: usize,
    final_accuracy: f64,
}

#[derive(Debug, Clone, Serialize)]
struct AlperNiyetSonuc {
    intent_index: usize,
    confidence: f64,
    probabilities: Vec<f64>,
}

/// ALPER niyet / teşhis beyni — `Sequential` çok sınıflı sınıflandırıcı (WASM).
///
/// Harici LLM API’si yoktur; BoW/TF-IDF vektörleri doğrudan bu ağa beslenir.
#[wasm_bindgen]
pub struct AlperBeyin {
    model: Sequential,
    input_dim: usize,
    output_dim: usize,
}

#[wasm_bindgen]
impl AlperBeyin {
    /// `input_dim` özellik, `output_dim` sınıf (one-hot) için MLP oluşturur.
    #[wasm_bindgen(constructor)]
    pub fn new(input_dim: usize, output_dim: usize) -> Result<AlperBeyin, JsValue> {
        if input_dim == 0 || output_dim < 2 {
            return Err(JsValue::from_str(
                "AlperBeyin: input_dim > 0 ve output_dim >= 2 olmalı",
            ));
        }

        // Sabit kapasite: girdi → 64 → 32 → sınıf (softmax)
        let mut model = Sequential::new();
        model
            .add(dense_layer(input_dim, 64, Activation::Relu)?)
            .add(dense_layer(64, 32, Activation::Relu)?)
            .add(dense_layer(32, output_dim, Activation::Softmax)?);

        let lr = dtype::from_f64(0.035);
        model.compile_with_loss(
            Box::new(Adam::new(lr, 0.9, 0.999, dtype::from_f64(1e-8))),
            Box::new(CrossEntropy::new(false)),
        );

        Ok(Self {
            model,
            input_dim,
            output_dim,
        })
    }

    #[wasm_bindgen]
    pub fn girdi_boyutu(&self) -> usize {
        self.input_dim
    }

    #[wasm_bindgen]
    pub fn cikti_boyutu(&self) -> usize {
        self.output_dim
    }

    /// Düz JSON matrisleriyle eğit: `x` = [n][input_dim], `y` = [n][output_dim] one-hot.
    #[wasm_bindgen]
    pub fn egit(
        &mut self,
        x_json: &str,
        y_json: &str,
        epochs: usize,
        batch_size: usize,
    ) -> Result<JsValue, JsValue> {
        let x_raw: Vec<Vec<f64>> = serde_json::from_str(x_json)
            .map_err(|e| JsValue::from_str(&format!("x JSON hatası: {e}")))?;
        let y_raw: Vec<Vec<f64>> = serde_json::from_str(y_json)
            .map_err(|e| JsValue::from_str(&format!("y JSON hatası: {e}")))?;

        if x_raw.is_empty() || x_raw.len() != y_raw.len() {
            return Err(JsValue::from_str(
                "AlperBeyin.egit: boş veya boyut uyuşmayan veri",
            ));
        }

        let n = x_raw.len();
        let mut x_data = Vec::with_capacity(n * self.input_dim);
        let mut y_data = Vec::with_capacity(n * self.output_dim);

        for (row_x, row_y) in x_raw.iter().zip(y_raw.iter()) {
            if row_x.len() != self.input_dim {
                return Err(JsValue::from_str(&format!(
                    "x satır boyutu {} != input_dim {}",
                    row_x.len(),
                    self.input_dim
                )));
            }
            if row_y.len() != self.output_dim {
                return Err(JsValue::from_str(&format!(
                    "y satır boyutu {} != output_dim {}",
                    row_y.len(),
                    self.output_dim
                )));
            }
            x_data.extend(row_x.iter().copied().map(dtype::from_f64));
            y_data.extend(row_y.iter().copied().map(dtype::from_f64));
        }

        let x = Array2::from_shape_vec((n, self.input_dim), x_data)
            .map_err(|e| JsValue::from_str(&format!("x shape: {e}")))?;
        let y = Array2::from_shape_vec((n, self.output_dim), y_data)
            .map_err(|e| JsValue::from_str(&format!("y shape: {e}")))?;

        let epochs = epochs.max(1);
        let batch = batch_size.clamp(1, n);
        let history = self
            .model
            .fit(&x.view(), &y.view(), epochs, batch, None)
            .map_err(|e| JsValue::from_str(&format!("ALPER eğitim hatası: {e}")))?;

        let mut metrics = Vec::with_capacity(history.train_losses.len());
        for (i, &loss) in history.train_losses.iter().enumerate() {
            metrics.push(EpochMetrik {
                epoch: i + 1,
                loss: dtype::to_f64(loss),
                accuracy: 0.0,
            });
        }

        let pred = self
            .model
            .predict(&x.view())
            .map_err(|e| JsValue::from_str(&format!("doğruluk tahmini: {e}")))?;
        let final_accuracy = dtype::to_f64(Accuracy::new().compute(&y.view(), &pred.view()));
        for m in &mut metrics {
            m.accuracy = final_accuracy;
        }

        let sonuc = AlperEgitimSonuc {
            metrics,
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            sample_count: n,
            final_accuracy,
        };
        serde_wasm_bindgen::to_value(&sonuc).map_err(|e| JsValue::from_str(&format!("JSON: {e}")))
    }

    /// Tek BoW/TF-IDF vektörü → niyet indeksi + softmax güven.
    #[wasm_bindgen]
    pub fn niyet_tahmin(&mut self, x_json: &str) -> Result<String, JsValue> {
        let row: Vec<f64> = serde_json::from_str(x_json)
            .map_err(|e| JsValue::from_str(&format!("vektör JSON hatası: {e}")))?;
        if row.len() != self.input_dim {
            return Err(JsValue::from_str(&format!(
                "vektör boyutu {} != {}",
                row.len(),
                self.input_dim
            )));
        }

        let data: Vec<Float> = row.iter().copied().map(dtype::from_f64).collect();
        let x = Array2::from_shape_vec((1, self.input_dim), data)
            .map_err(|e| JsValue::from_str(&format!("shape: {e}")))?;
        let pred = self
            .model
            .predict(&x.view())
            .map_err(|e| JsValue::from_str(&format!("niyet tahmini: {e}")))?;

        let probabilities: Vec<f64> = pred.iter().copied().map(dtype::to_f64).collect();
        let (intent_index, confidence) = classify_outputs(&probabilities);
        let sonuc = AlperNiyetSonuc {
            intent_index,
            confidence,
            probabilities,
        };
        serde_json::to_string(&sonuc).map_err(|e| JsValue::from_str(&format!("JSON: {e}")))
    }
}

fn dense_layer(i: usize, o: usize, act: Activation) -> Result<Dense, JsValue> {
    Dense::new(
        i,
        o,
        act,
        Box::new(GlorotUniform::new()),
        Box::new(Zeros::new()),
        true,
    )
    .map_err(|e| JsValue::from_str(&format!("Dense katman hatası: {e}")))
}

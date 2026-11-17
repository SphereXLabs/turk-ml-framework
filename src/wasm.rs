//! WASM Browser Playground köprüsü — `.turk` DSL'ini tarayıcıda çalıştırır.
//!
//! Disk erişimi yoktur; `veri_yukle("xor")` gibi çağrılar sentetik bellek verisine
//! yönlendirilir. Epoch metrikleri ve model özeti JSON olarak döner.

use crate::dil::{EgitimIlerleme, TurkDil};
use crate::metrics::{Accuracy, Metric};
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
    /// Dönüş: `{ logs, metrics: [{epoch, loss, accuracy}], model_summary }`
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
        };

        serde_wasm_bindgen::to_value(&sonuc)
            .map_err(|e| JsValue::from_str(&format!("JSON serileştirme hatası: {e}")))
    }
}

impl Default for TurkWasmRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl TurkWasmRunner {
    /// Aktif sentetik veri (xor / kume) üzerinde ikili doğruluk.
    fn dataset_accuracy(&mut self) -> Option<f64> {
        let (x, y) = self.dil.aktif_degerlendirme_verisi()?;
        let model = self.dil.model.as_mut()?;
        let pred = model.predict(&x.view()).ok()?;
        let acc = Accuracy::new().compute(&y.view(), &pred.view());
        Some(crate::dtype::to_f64(acc))
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

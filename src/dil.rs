use crate::aktivasyon::Activation;
use crate::cihaz::{cihaz_listesi_metin, Cihaz};
use crate::error::{Result, TurkError};
use crate::initializers::{GlorotUniform, Zeros};
use crate::katman_kayit::KatmanKayit;
use crate::layer::Dense;
use crate::loss::create_loss;
use crate::model::{EarlyStopping, Sequential};
#[cfg(not(target_arch = "wasm32"))]
use crate::model_zoo::{model_bilgi_metin, model_indir, zoo_listesi_metin};
use crate::optimizer::{Adam, AdamW, RMSProp, SGD};
use crate::veri::CsvVeriAkisi;
use log::{debug, info, warn};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

/// Türkçe DSL değer türleri
#[derive(Debug, Clone)]
pub enum Value {
    Number(f64),
    Text(String),
    Boolean(bool),
    Model(Box<dyn ModelExt>),
    None,
}

impl Value {
    #[allow(dead_code)]
    fn as_number(&self) -> Result<f64> {
        match self {
            Value::Number(n) => Ok(*n),
            _ => Err(TurkError::TypeError("Sayı değil".into())),
        }
    }

    #[allow(dead_code)]
    fn as_text(&self) -> Result<String> {
        match self {
            Value::Text(s) => Ok(s.clone()),
            _ => Err(TurkError::TypeError("Metin değil".into())),
        }
    }
    #[allow(dead_code)]
    fn as_bool(&self) -> Result<bool> {
        match self {
            Value::Boolean(b) => Ok(*b),
            _ => Err(TurkError::TypeError("Boolean değil".into())),
        }
    }
    #[allow(dead_code)]
    fn as_model(&self) -> Result<&dyn ModelExt> {
        match self {
            Value::Model(model) => Ok(model.as_ref()),
            _ => Err(TurkError::TypeError("Model değil".into())),
        }
    }
}

/// Model trait'i için uzantı
pub trait ModelExt: Any + Send + Sync + Debug {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_box(&self) -> Box<dyn ModelExt>;
}

impl Clone for Box<dyn ModelExt> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl ModelExt for Sequential {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn ModelExt> {
        Box::new(self.clone())
    }
}

/// Yüklü veri seti — lazy CSV akışı (RAM'de yalnızca batch tutulur)
pub type YukluVeri = CsvVeriAkisi;

/// GUI / arka plan eğitim geri bildirimi
#[derive(Debug, Clone)]
pub enum EgitimIlerleme {
    /// CSV satir indeksleme ilerlemesi (MNIST gibi buyuk dosyalar)
    Indeksleme {
        satir: usize,
    },
    /// Ara durum (veri indeksleme, model kurulumu vb.)
    Durum {
        mesaj: String,
    },
    Epoch {
        epoch: usize,
        toplam: usize,
        train_loss: f64,
        val_loss: Option<f64>,
    },
    Tamamlandi {
        epoch_sayisi: usize,
        en_iyi_train: f64,
        en_iyi_val: Option<f64>,
    },
    Hata(String),
}

/// TÜRK dili yorumlayıcısı
pub struct TurkDil {
    pub model: Option<Sequential>,
    #[allow(dead_code)]
    variables: HashMap<String, Value>,
    /// Adlandırılmış veri setleri: veri_yukle ile yüklenenler
    veri_setleri: HashMap<String, YukluVeri>,
    /// Varsayılan aktif veri seti (en son yüklenen)
    aktif_veri: Option<String>,
    /// Son `kaydet(...)` ile yazılan dosya yolu (resume için)
    kaydet_yolu: Option<String>,
    /// Canlı epoch güncellemeleri (GUI kanalı)
    egitim_tx: Option<std::sync::mpsc::Sender<EgitimIlerleme>>,
}

impl TurkDil {
    /// Yeni bir TurkDil yorumlayıcısı oluşturur
    pub fn new() -> Self {
        Self {
            model: None,
            variables: HashMap::new(),
            veri_setleri: HashMap::new(),
            aktif_veri: None,
            kaydet_yolu: None,
            egitim_tx: None,
        }
    }

    /// GUI için epoch güncellemelerini gönderen kanal
    pub fn egitim_kanali_ayarla(&mut self, tx: std::sync::mpsc::Sender<EgitimIlerleme>) {
        self.egitim_tx = Some(tx);
    }

    fn egitim_durum_gonder(&self, mesaj: impl Into<String>) {
        if let Some(tx) = &self.egitim_tx {
            let _ = tx.send(EgitimIlerleme::Durum {
                mesaj: mesaj.into(),
            });
        }
    }

    /// Model Zoo betiğini çalıştırır; MNIST için `csv_yolu` zorunlu olabilir
    #[cfg(not(target_arch = "wasm32"))]
    pub fn zoo_betik_calistir(&mut self, id: &str, csv_yolu: Option<&str>) -> Result<()> {
        use std::fs;

        self.egitim_durum_gonder(format!("Model Zoo hazirlaniyor: {id}"));

        let yol = model_indir(id, None)?;
        let script = yol.join("model.turk");
        if !script.is_file() {
            return Err(TurkError::IOError(format!(
                "Betik bulunamadı: {}",
                script.display()
            )));
        }

        let onceki_cwd = std::env::current_dir().ok();
        std::env::set_current_dir(&yol)
            .map_err(|e| TurkError::IOError(format!("Model dizinine geçilemedi: {e}")))?;

        let sonuc = (|| {
            if let Some(csv) = csv_yolu {
                self.egitim_durum_gonder(format!(
                    "CSV indeksleniyor (lazy stream, ana thread serbest): {csv}"
                ));
                let veri_komut = format!("veri_yukle(\"{csv}\", bolme=0.1)");
                self.calistir_satir(&veri_komut, 1)?;
            }

            self.egitim_durum_gonder("Model mimarisi ve egitim baslatiliyor...");

            let icerik = fs::read_to_string(&script).map_err(|e| {
                TurkError::IOError(format!("'{}' okunamadı: {e}", script.display()))
            })?;

            let mut satir_no = 0usize;
            for satir in icerik.lines() {
                satir_no += 1;
                let satir = satir.trim();
                if satir.is_empty() || satir.starts_with('#') || satir.starts_with("//") {
                    continue;
                }
                if csv_yolu.is_some() && satir.starts_with("veri_yukle(") {
                    continue;
                }
                self.calistir_satir(satir, satir_no)?;
            }
            Ok(())
        })();

        if let Some(cwd) = onceki_cwd {
            let _ = std::env::set_current_dir(cwd);
        }

        sonuc
    }

    /// WASM: Model Zoo disk erişimi yok.
    #[cfg(target_arch = "wasm32")]
    pub fn zoo_betik_calistir(&mut self, _id: &str, _csv_yolu: Option<&str>) -> Result<()> {
        Err(TurkError::RuntimeError(
            "Model Zoo WASM ortamında desteklenmiyor; sentetik veri_yukle(\"xor\") kullanın."
                .into(),
        ))
    }

    /// Türkçe DSL komutunu çalıştırır
    pub fn calistir(&mut self, komut: &str) -> Result<Value> {
        self.calistir_satir(komut, 1)
    }

    /// Satır numarası ile komut çalıştırır (hata konumlandırma için).
    pub fn calistir_satir(&mut self, komut: &str, satir_no: usize) -> Result<Value> {
        self.komut_isle(komut, satir_no, false)
    }

    /// Statik analiz modu — eğitim başlatmadan script'i doğrular.
    ///
    /// `egit` komutu yalnızca ön koşulları kontrol eder; `fit` çağrılmaz.
    /// `kaydet` / `yukle` komutları sözdizimi düzeyinde doğrulanır, disk I/O yapılmaz.
    pub fn kontrol_et(&mut self, komut: &str) -> Result<()> {
        self.kontrol_et_satir(komut, 1)
    }

    /// Satır numarası ile statik analiz.
    pub fn kontrol_et_satir(&mut self, komut: &str, satir_no: usize) -> Result<()> {
        self.komut_isle(komut, satir_no, true).map(|_| ())
    }

    fn komut_isle(&mut self, komut: &str, satir_no: usize, kontrol_modu: bool) -> Result<Value> {
        let komut = komut.trim();
        if komut.is_empty() || komut.starts_with('#') || komut.starts_with("//") {
            return Ok(Value::None);
        }

        // Resmi parser — sözdizimi + satır/sütun doğrulama
        let parsed = crate::dsl_parser::komut_ayristir(komut, satir_no)?;
        debug!(
            "Komut {}: {}",
            if kontrol_modu {
                "kontrol"
            } else {
                "çalıştırma"
            },
            komut
        );

        match parsed {
            crate::dsl_parser::DslKomut::ModelOlustur => {
                info!("Yeni model oluşturuluyor");
                self.model = Some(Sequential::new());
                Ok(Value::None)
            }
            crate::dsl_parser::DslKomut::KatmanEkle { icerik } => {
                self.katman_ekle_icerik(icerik.as_str())
            }
            crate::dsl_parser::DslKomut::Derle { icerik } => self.derle_icerik(icerik.as_str()),
            crate::dsl_parser::DslKomut::ModelBilgisi => self.model_bilgisi_komutu(),
            crate::dsl_parser::DslKomut::Egit { icerik } => {
                if kontrol_modu {
                    self.egit_on_kontrol_icerik(icerik.as_str())
                } else {
                    self.egit_icerik(icerik.as_str())
                }
            }
            crate::dsl_parser::DslKomut::VeriYukle { icerik } => {
                self.veri_yukle_icerik(icerik.as_str())
            }
            crate::dsl_parser::DslKomut::Kaydet { icerik } => {
                if kontrol_modu {
                    self.kaydet_on_kontrol_icerik(icerik.as_str())
                } else {
                    self.kaydet_icerik(icerik.as_str())
                }
            }
            crate::dsl_parser::DslKomut::Yukle { icerik } => {
                if kontrol_modu {
                    self.yukle_on_kontrol_icerik(icerik.as_str())
                } else {
                    self.yukle_icerik(icerik.as_str())
                }
            }
            crate::dsl_parser::DslKomut::OnnxDisaAktar { dosya } => {
                if kontrol_modu {
                    self.onnx_on_kontrol(dosya.as_str())
                } else {
                    self.onnx_disa_aktar_komutu(dosya.as_str())
                }
            }
            crate::dsl_parser::DslKomut::CihazBilgisi => {
                let metin = cihaz_listesi_metin();
                println!("{metin}");
                Ok(Value::Text(metin))
            }
            crate::dsl_parser::DslKomut::ModelZoo => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let metin = zoo_listesi_metin();
                    println!("{metin}");
                    Ok(Value::Text(metin))
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let metin = "Model Zoo WASM'da kapalı. Örnek: veri_yukle(\"xor\")".to_string();
                    Ok(Value::Text(metin))
                }
            }
            crate::dsl_parser::DslKomut::ModelIndir { id, hedef } => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if kontrol_modu {
                        model_bilgi_metin(id.as_str())?;
                        Ok(Value::None)
                    } else {
                        self.model_indir_komutu(id.as_str(), hedef.as_deref())
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = (id, hedef, kontrol_modu);
                    Err(TurkError::RuntimeError(
                        "model_indir WASM ortamında desteklenmiyor.".into(),
                    ))
                }
            }
            crate::dsl_parser::DslKomut::ModelCalistir { id } => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if kontrol_modu {
                        model_bilgi_metin(id.as_str())?;
                        Ok(Value::None)
                    } else {
                        self.model_calistir_komutu(id.as_str())
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = (id, kontrol_modu);
                    Err(TurkError::RuntimeError(
                        "model_calistir WASM ortamında desteklenmiyor.".into(),
                    ))
                }
            }
        }
    }

    /// Katman türü, `KatmanKayit::varsayilan()` üzerinden çözülür.
    fn katman_ekle_icerik(&mut self, katman_str: &str) -> Result<Value> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| TurkError::RuntimeError("Önce model oluşturmalısınız".into()))?;

        info!("Katman ekleniyor: {katman_str}");

        // ── KatmanKayit üzerinden üret (artık match/if-else yok) ──────────
        let katman = KatmanKayit::varsayilan().dsl_olustur_tamam(katman_str)?;

        // ── Önceki katmanla boyut uyum kontrolü ───────────────────────────
        // (sadece boyutu bilinen katmanlar için — 0 dönenler atlanır)
        if let Some(onceki) = model.layers.last() {
            let beklenen = onceki.output_size();
            let gelen = katman.input_size();
            if beklenen != 0 && gelen != 0 && gelen != beklenen {
                return Err(TurkError::ShapeMismatch(format!(
                    "Katman boyut uyumsuzluğu!\n  \
                     Önceki katman çıkışı : {beklenen}\n  \
                     Yeni katman girişi   : {gelen}\n  \
                     İpucu: ilk parametreyi {beklenen} yapın."
                )));
            }
        }

        info!(
            "Katman eklendi: {} (girdi={}, çıktı={})",
            katman.layer_type(),
            katman.input_size(),
            katman.output_size()
        );
        model.add_boxed(katman);
        Ok(Value::None)
    }

    /// Derleme komutunu işler.
    ///
    /// ```text
    /// derle(adam)
    /// derle(adam, loss=mse)
    /// derle(optimizer=adam, loss=mse, ogrenme_orani=0.05)
    /// ```
    fn derle_icerik(&mut self, icerik: &str) -> Result<Value> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| TurkError::RuntimeError("Önce model oluşturmalısınız".into()))?;

        let params = derle_parametreleri_parse(icerik)?;
        let optimizer = optimizer_olustur(&params.optimizer, params.ogrenme_orani)?;
        let loss = create_loss(&params.loss)?;

        info!(
            "Model derleniyor: optimizer={}, loss={}, lr={}",
            params.optimizer,
            params.loss,
            params.ogrenme_orani.unwrap_or(f64::NAN)
        );

        model.compile_with_loss(optimizer, loss);

        if let Some(cihaz_str) = params.cihaz {
            let cihaz = Cihaz::from_str(&cihaz_str)?;
            cihaz.hazirla()?;
            model.set_cihaz(cihaz);
            info!("Cihaz ayarlandı: {}", cihaz.ad());
        }

        Ok(Value::None)
    }

    /// Model bilgisi komutunu işler — katman zinciri, parametre sayısı, optimizer/loss.
    fn model_bilgisi_komutu(&mut self) -> Result<Value> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| TurkError::RuntimeError("Önce model oluşturmalısınız".into()))?;

        let bilgi = model.summary();
        println!("{bilgi}");
        Ok(Value::Text(bilgi))
    }

    /// Eğitim komutunu işler.
    ///
    /// Sözdizimi:
    /// ```text
    /// egit(epoch, batch_boyutu)
    /// egit(200, 4, devam_et=true)
    /// egit(200, 4, sabir=10, min_delta=0.0001)
    /// egit(epoch=200, batch_boyutu=4, sabir=10)
    /// ```
    fn egit_icerik(&mut self, icerik: &str) -> Result<Value> {
        let p = egit_parametreleri_parse(icerik)?;

        let veri_ismi = self.aktif_veri.clone().ok_or_else(|| {
            TurkError::DataError(
                "Eğitim verisi yüklenmemiş. Önce veri_yukle() komutunu kullanın.".into(),
            )
        })?;
        let veri = self
            .veri_setleri
            .get(&veri_ismi)
            .ok_or_else(|| TurkError::DataError(format!("Veri seti bulunamadı: {veri_ismi}")))?
            .clone();

        if p.devam_et {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let kayit_yolu = self.kaydet_yolu.clone().unwrap_or_else(|| {
                    warn!(
                        "devam_et=true verildi ancak kaydet_yolu bilinmiyor; \
                           'xor_model.json' varsayılanı kullanılıyor."
                    );
                    "xor_model.json".to_string()
                });

                if std::path::Path::new(&kayit_yolu).exists() {
                    info!("▶ Checkpoint yükleniyor: '{kayit_yolu}'");
                    let model = self.model.as_mut().ok_or_else(|| {
                        TurkError::RuntimeError("Model henüz oluşturulmamış".into())
                    })?;
                    model.load(&kayit_yolu)?;
                    info!("Checkpoint başarıyla yüklendi — eğitime kaldığı yerden devam ediliyor.");
                } else {
                    warn!(
                        "devam_et=true verildi ancak '{kayit_yolu}' bulunamadı; \
                         sıfırdan eğitim başlatılıyor."
                    );
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                warn!("devam_et WASM'da yok sayıldı (disk checkpoint yok).");
            }
        }

        let model = self.model.as_mut().ok_or_else(|| {
            TurkError::RuntimeError("Önce model oluşturmalı ve derlemelisiniz".into())
        })?;

        let veri_ozellik = veri.ozellik_sayisi();
        if let Some(ilk_katman) = model.layers.first() {
            let model_girdi = ilk_katman.input_size();
            if model_girdi != 0 && veri_ozellik != model_girdi {
                return Err(TurkError::ShapeMismatch(format!(
                    "Veri–Model giriş boyutu uyumsuzluğu!\n  \
                     CSV özellik sayısı   : {veri_ozellik}\n  \
                     Modelin ilk katmanı  : {model_girdi} giriş bekliyor\n  \
                     Kaynak               : {}\n  \
                     İpucu: CSV'deki sütun sayısını veya modelin ilk katmanını güncelleyin.",
                    veri.kaynak()
                )));
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let checkpoint_yolu = {
            let temel = self
                .kaydet_yolu
                .clone()
                .unwrap_or_else(|| "model.json".into());
            let temel = temel.trim_end_matches(".json");
            format!("{temel}_en_iyi.json")
        };

        let validation_data = veri.val_veri().map(|(x, y)| (x.clone(), y.clone()));

        let mut early_stopping = p
            .sabir
            .map(|patience| EarlyStopping::new(patience, p.min_delta.unwrap_or(0.0)));

        info!(
            "Model eğitiliyor (lazy stream): kaynak='{}', egitim={}, val={}, {} özellik, \
             {} epoch, batch={} | loss={} | devam_et={}",
            veri.kaynak(),
            veri.egitim_satir_sayisi(),
            veri.val_satir_sayisi(),
            veri_ozellik,
            p.epochs,
            p.batch_size,
            model.loss_name(),
            p.devam_et
        );

        let batch_boyutu = p.batch_size;

        let mut epoch_cb: Option<crate::model::EpochCallback> = None;
        if let Some(tx) = self.egitim_tx.clone() {
            epoch_cb = Some(Box::new(move |epoch, toplam, loss, val| {
                let _ = tx.send(EgitimIlerleme::Epoch {
                    epoch,
                    toplam,
                    train_loss: loss,
                    val_loss: val,
                });
            }));
        }

        let history = {
            let batch_kaynak = || veri.epoch_batchleri(batch_boyutu);
            #[cfg(not(target_arch = "wasm32"))]
            {
                model.fit_streaming_with_checkpoint(
                    batch_kaynak,
                    p.epochs,
                    validation_data,
                    early_stopping.as_mut(),
                    Some(&checkpoint_yolu),
                    epoch_cb,
                )?
            }
            #[cfg(target_arch = "wasm32")]
            {
                model.fit_streaming_with_checkpoint(
                    batch_kaynak,
                    p.epochs,
                    validation_data,
                    early_stopping.as_mut(),
                    None,
                    epoch_cb,
                )?
            }
        };

        let son_kayip = if veri.val_satir_sayisi() > 0 {
            if let Some((en_iyi_train, en_iyi_val, en_iyi_ep)) = history.en_iyi_val_kayitlari() {
                info!(
                    "Eğitim tamamlandı — {} epoch çalıştırıldı | \
                     en iyi model geri yüklendi (epoch {en_iyi_ep}): \
                     loss={en_iyi_train:.6}, val_loss={en_iyi_val:.6}",
                    history.epochs_run(),
                );
                en_iyi_train
            } else {
                let son = history.last_train_loss().unwrap_or(f64::NAN);
                info!(
                    "Eğitim tamamlandı — {} epoch, son kayıp: {son:.6}",
                    history.epochs_run(),
                );
                son
            }
        } else {
            let son = history.last_train_loss().unwrap_or(f64::NAN);
            info!(
                "Eğitim tamamlandı — {} epoch, son kayıp: {son:.6}",
                history.epochs_run(),
            );
            son
        };

        if let Some(tx) = &self.egitim_tx {
            let (en_iyi_train, en_iyi_val) = history
                .en_iyi_val_kayitlari()
                .map(|(t, v, _)| (t, Some(v)))
                .unwrap_or_else(|| {
                    (
                        history.last_train_loss().unwrap_or(f64::NAN),
                        history.last_val_loss(),
                    )
                });
            let _ = tx.send(EgitimIlerleme::Tamamlandi {
                epoch_sayisi: history.epochs_run(),
                en_iyi_train,
                en_iyi_val,
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            const GECMIS_DOSYA: &str = "egitim_gecmisi.json";
            match history.dosyaya_kaydet(GECMIS_DOSYA) {
                Ok(_) => info!("Eğitim geçmişi '{GECMIS_DOSYA}' dosyasına kaydedildi."),
                Err(e) => warn!("Eğitim geçmişi kaydedilemedi: {e}"),
            }
        }

        Ok(Value::Number(son_kayip))
    }

    /// `check` modunda egit — parametreleri ve boyut uyumunu doğrular, eğitim başlatmaz.
    fn egit_on_kontrol_icerik(&mut self, icerik: &str) -> Result<Value> {
        let p = egit_parametreleri_parse(icerik)?;

        let veri_ismi = self.aktif_veri.clone().ok_or_else(|| {
            TurkError::DataError(
                "Eğitim verisi yüklenmemiş. Önce veri_yukle() komutunu kullanın.".into(),
            )
        })?;
        let veri = self
            .veri_setleri
            .get(&veri_ismi)
            .ok_or_else(|| TurkError::DataError(format!("Veri seti bulunamadı: {veri_ismi}")))?;

        let model = self.model.as_ref().ok_or_else(|| {
            TurkError::RuntimeError("Önce model oluşturmalı ve derlemelisiniz".into())
        })?;

        if model.optimizer.is_none() {
            return Err(TurkError::RuntimeError(
                "Model derlenmemiş. Önce derle(adam) veya derle(sgd) kullanın.".into(),
            ));
        }

        let veri_ozellik = veri.ozellik_sayisi();
        if let Some(ilk_katman) = model.layers.first() {
            let model_girdi = ilk_katman.input_size();
            if model_girdi != 0 && veri_ozellik != model_girdi {
                return Err(TurkError::ShapeMismatch(format!(
                    "Veri–Model giriş boyutu uyumsuzluğu!\n  \
                     CSV özellik sayısı   : {veri_ozellik}\n  \
                     Modelin ilk katmanı  : {model_girdi} giriş bekliyor\n  \
                     Kaynak               : {}",
                    veri.kaynak()
                )));
            }
        }

        info!(
            "✓ egit() statik kontrol geçti — epoch={}, batch={}, kaynak='{}'{}{}",
            p.epochs,
            p.batch_size,
            veri.kaynak(),
            p.sabir.map(|s| format!(", sabir={s}")).unwrap_or_default(),
            if p.devam_et { ", devam_et=true" } else { "" }
        );
        Ok(Value::None)
    }

    fn kaydet_on_kontrol_icerik(&self, icerik: &str) -> Result<Value> {
        self.model
            .as_ref()
            .ok_or_else(|| TurkError::RuntimeError("Kaydedilecek model bulunmuyor".into()))?;
        dosya_yolu_icerikten(icerik)?;
        Ok(Value::None)
    }

    fn yukle_on_kontrol_icerik(&self, icerik: &str) -> Result<Value> {
        dosya_yolu_icerikten(icerik)?;
        Ok(Value::None)
    }

    fn onnx_on_kontrol(&self, dosya: &str) -> Result<Value> {
        self.model
            .as_ref()
            .ok_or_else(|| TurkError::RuntimeError("Dışa aktarılacak model bulunmuyor".into()))?;
        if dosya.is_empty() {
            return Err(TurkError::SyntaxError(
                "onnx_disa_aktar: dosya yolu boş olamaz".into(),
            ));
        }
        Ok(Value::None)
    }

    fn onnx_disa_aktar_komutu(&mut self, dosya: &str) -> Result<Value> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = dosya;
            return Err(TurkError::RuntimeError(
                "ONNX dışa aktarma WASM ortamında desteklenmiyor.".into(),
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let model = self.model.as_ref().ok_or_else(|| {
                TurkError::RuntimeError("Dışa aktarılacak model bulunmuyor".into())
            })?;
            info!("ONNX dışa aktarılıyor: {dosya}");
            crate::onnx::export_onnx(model, dosya)?;
            let mesaj = format!("ONNX dışa aktarıldı: {dosya}");
            Ok(Value::Text(mesaj))
        }
    }

    /// Model Zoo'dan model indirir
    #[cfg(not(target_arch = "wasm32"))]
    fn model_indir_komutu(&mut self, id: &str, hedef: Option<&str>) -> Result<Value> {
        let hedef_yol = hedef.map(std::path::Path::new);
        let yol = model_indir(id, hedef_yol)?;
        let mesaj = format!("Model indirildi: {}", yol.display());
        info!("{mesaj}");
        println!("{mesaj}");
        Ok(Value::Text(mesaj))
    }

    /// İndirilmiş Model Zoo betiğini çalıştırır (model dizinine geçerek)
    #[cfg(not(target_arch = "wasm32"))]
    fn model_calistir_komutu(&mut self, id: &str) -> Result<Value> {
        use std::fs;

        let yol = model_indir(id, None)?;
        let script = yol.join("model.turk");
        if !script.is_file() {
            return Err(TurkError::IOError(format!(
                "Betik bulunamadı: {}",
                script.display()
            )));
        }

        let onceki_cwd = std::env::current_dir().ok();
        std::env::set_current_dir(&yol)
            .map_err(|e| TurkError::IOError(format!("Model dizinine geçilemedi: {e}")))?;

        let icerik = fs::read_to_string(&script)
            .map_err(|e| TurkError::IOError(format!("'{}' okunamadı: {e}", script.display())))?;

        info!("Model Zoo çalıştırılıyor: {} ({})", id, script.display());
        let mut satir_no = 0usize;
        let sonuc = (|| {
            for satir in icerik.lines() {
                satir_no += 1;
                let satir = satir.trim();
                if satir.is_empty() || satir.starts_with('#') || satir.starts_with("//") {
                    continue;
                }
                self.calistir_satir(satir, satir_no)?;
            }
            Ok(Value::None)
        })();

        if let Some(cwd) = onceki_cwd {
            let _ = std::env::set_current_dir(cwd);
        }

        sonuc
    }

    /// Veri yükleme — parser tarafından ayrıştırılmış içerik.
    fn veri_yukle_icerik(&mut self, icerik: &str) -> Result<Value> {
        let mut dosya_yolu: Option<String> = None;
        let mut hedef_kolon: Option<String> = None;
        let mut bolme: Option<f64> = None;

        // İlk tırnaklı string CSV yolu olarak kullanılır
        if let Some(yol) = ilk_tirnakli_deger(icerik) {
            dosya_yolu = Some(yol);
        }

        // key: "value" ya da key=value çiftleri
        for kisim in icerik.split(',') {
            let kisim = kisim.trim();
            for ayirici in [':', '='] {
                if let Some(pos) = kisim.find(ayirici) {
                    let anahtar = kisim[..pos].trim().to_lowercase();
                    let deger = kisim[pos + 1..].trim().trim_matches('"').trim().to_string();
                    match anahtar.as_str() {
                        "kaynak" | "dosya" | "source" => {
                            dosya_yolu = Some(deger);
                        }
                        "hedef_kolon" | "hedef" | "label" | "etiket" => {
                            hedef_kolon = Some(deger);
                        }
                        "bolme" | "val_oran" | "validation" => {
                            bolme = Some(deger.parse::<f64>().map_err(|_| {
                                TurkError::SyntaxError(format!(
                                    "bolme sayısal olmalı (örn. 0.2), gelen: '{deger}'"
                                ))
                            })?);
                        }
                        _ => {}
                    }
                    break;
                }
            }
        }

        let dosya_yolu = dosya_yolu.ok_or_else(|| {
            TurkError::SyntaxError(
                "veri_yukle: kaynak dosya belirtilmedi. Örnek: veri_yukle(\"dosya.csv\")".into(),
            )
        })?;

        // WASM / tarayıcı: disk yok — xor veya sentetik anahtarlar bellekten yüklenir
        #[cfg(target_arch = "wasm32")]
        let veri = {
            let _ = &hedef_kolon;
            let anahtar = dosya_yolu.to_lowercase();
            if anahtar.contains("xor") || anahtar == "sentetik" || anahtar.ends_with(".csv") {
                info!("WASM sentetik veri yükleniyor: '{dosya_yolu}' → XOR");
                CsvVeriAkisi::sentetik_xor(bolme)?
            } else {
                return Err(TurkError::DataError(format!(
                    "WASM'da yalnızca sentetik veri desteklenir. Örnek: veri_yukle(\"xor\") \
                     (istenen: '{dosya_yolu}')"
                )));
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        let veri = if let Some(tx) = self.egitim_tx.clone() {
            let mut on_satir = move |n: usize| {
                let _ = tx.send(EgitimIlerleme::Indeksleme { satir: n });
            };
            CsvVeriAkisi::yeni_bolme_with_progress(
                &dosya_yolu,
                hedef_kolon.as_deref(),
                bolme,
                Some(&mut on_satir),
            )?
        } else {
            CsvVeriAkisi::yeni_bolme(&dosya_yolu, hedef_kolon.as_deref(), bolme)?
        };

        if let Some(tx) = &self.egitim_tx {
            let _ = tx.send(EgitimIlerleme::Durum {
                mesaj: format!(
                    "CSV hazir: {} egitim + {} val satir ({} ozellik)",
                    veri.egitim_satir_sayisi(),
                    veri.val_satir_sayisi(),
                    veri.ozellik_sayisi()
                ),
            });
        }

        info!(
            "Veri akışı hazır: '{}' — {} özellik, egitim={}, val={}{}",
            dosya_yolu,
            veri.ozellik_sayisi(),
            veri.egitim_satir_sayisi(),
            veri.val_satir_sayisi(),
            bolme.map(|b| format!(", bolme={b}")).unwrap_or_default()
        );

        if veri.ozellik_sayisi() == 784 {
            info!("MNIST boyutu algılandı: 784 özellik (28×28 piksel)");
        }

        let isim = dosya_yolu.clone();
        self.veri_setleri.insert(isim.clone(), veri);
        self.aktif_veri = Some(isim);

        Ok(Value::None)
    }

    fn kaydet_icerik(&mut self, icerik: &str) -> Result<Value> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = icerik;
            return Err(TurkError::RuntimeError(
                "kaydet() WASM ortamında desteklenmiyor (disk yok).".into(),
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| TurkError::RuntimeError("Kaydedilecek model bulunmuyor".into()))?;
            let dosya_yolu = dosya_yolu_icerikten(icerik)?;
            info!("Model kaydediliyor: {dosya_yolu}");
            model.save(&dosya_yolu)?;
            self.kaydet_yolu = Some(dosya_yolu);
            Ok(Value::None)
        }
    }

    fn yukle_icerik(&mut self, icerik: &str) -> Result<Value> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = icerik;
            return Err(TurkError::RuntimeError(
                "yukle() WASM ortamında desteklenmiyor (disk yok).".into(),
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let dosya_yolu = dosya_yolu_icerikten(icerik)?;
            info!("Model yükleniyor: {dosya_yolu}");
            let mut model = Sequential::new();
            model.load(&dosya_yolu)?;
            self.model = Some(model);

            Ok(Value::None)
        }
    }

    /// Katman ekle - Dense
    pub fn dense(
        &mut self,
        girdi_boyutu: usize,
        cikti_boyutu: usize,
        aktivasyon: &str,
    ) -> Result<()> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| TurkError::RuntimeError("Önce model oluşturmalısınız".into()))?;

        let activation = Activation::from_str(aktivasyon)?;
        let kernel_init = Box::new(GlorotUniform::new());
        let bias_init = Box::new(Zeros::new());

        let dense = Dense::new(
            girdi_boyutu,
            cikti_boyutu,
            activation,
            kernel_init,
            bias_init,
            true, // use_bias
        )?;

        model.add(dense);
        Ok(())
    }
}

// ─── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────

/// İçerikteki ilk `"..."` değerini döndürür.
fn ilk_tirnakli_deger(s: &str) -> Option<String> {
    let start = s.find('"')?;
    let rest = &s[start + 1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// `"metin"` veya `'metin'` içindeki değeri döndürür (tüm string tırnaklı olmalı).
fn tirnak_ici(s: &str) -> Option<&str> {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        Some(&s[1..s.len() - 1])
    } else {
        None
    }
}

/// `kaydet("...")` / `yukle("...")` içindeki dosya yolunu çıkarır.
fn dosya_yolu_icerikten(icerik: &str) -> Result<String> {
    let dosya_yolu = icerik.trim();
    let dosya_yolu = if dosya_yolu.starts_with('"') && dosya_yolu.ends_with('"') {
        &dosya_yolu[1..dosya_yolu.len() - 1]
    } else {
        dosya_yolu
    };
    if dosya_yolu.is_empty() {
        return Err(TurkError::SyntaxError("Dosya yolu boş olamaz".into()));
    }
    Ok(dosya_yolu.to_string())
}

// ─── DSL Parametre Yardımcıları ─────────────────────────────────────────────

struct DerleParametreleri {
    optimizer: String,
    loss: String,
    ogrenme_orani: Option<f64>,
    cihaz: Option<String>,
}

struct EgitParametreleri {
    epochs: usize,
    batch_size: usize,
    devam_et: bool,
    sabir: Option<usize>,
    min_delta: Option<f64>,
}

fn dsl_anahtar_degerler(icerik: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for kisim in icerik.split(',') {
        let kisim = kisim.trim();
        for ayirici in [':', '='] {
            if let Some(pos) = kisim.find(ayirici) {
                let anahtar = kisim[..pos].trim().to_lowercase();
                let deger = kisim[pos + 1..].trim().trim_matches('"').trim().to_string();
                map.insert(anahtar, deger);
                break;
            }
        }
    }
    map
}

fn dsl_pozisyonel_tokenler(icerik: &str) -> Vec<String> {
    icerik
        .split(',')
        .map(str::trim)
        .filter(|t| {
            !t.is_empty() && !t.contains('=') && !t.contains(':') && tirnak_ici(t).is_none()
        })
        .map(|t| t.to_lowercase())
        .collect()
}

fn derle_parametreleri_parse(icerik: &str) -> Result<DerleParametreleri> {
    let map = dsl_anahtar_degerler(icerik);
    let poz = dsl_pozisyonel_tokenler(icerik);

    let optimizer = map
        .get("optimizer")
        .cloned()
        .or_else(|| poz.first().cloned())
        .ok_or_else(|| {
            TurkError::SyntaxError("derle: optimizer belirtilmedi. Örnek: derle(adam)".into())
        })?;

    let loss = map
        .get("loss")
        .or_else(|| map.get("kayip"))
        .cloned()
        .unwrap_or_else(|| "mse".to_string());

    let ogrenme_orani = map
        .get("ogrenme_orani")
        .or_else(|| map.get("lr"))
        .or_else(|| map.get("learning_rate"))
        .map(|s| {
            s.parse::<f64>()
                .map_err(|_| TurkError::SyntaxError(format!("ogrenme_orani sayısal olmalı: '{s}'")))
        })
        .transpose()?;

    let cihaz = map.get("cihaz").cloned();

    Ok(DerleParametreleri {
        optimizer,
        loss,
        ogrenme_orani,
        cihaz,
    })
}

fn optimizer_olustur(ad: &str, lr: Option<f64>) -> Result<Box<dyn crate::optimizer::Optimizer>> {
    match ad.to_lowercase().as_str() {
        "sgd" => Ok(Box::new(SGD::new(lr.unwrap_or(0.01), None, false))),
        "adam" => Ok(Box::new(Adam::new(lr.unwrap_or(0.001), 0.9, 0.999, 1e-8))),
        "adamw" => Ok(Box::new(AdamW::new(
            lr.unwrap_or(0.001),
            0.9,
            0.999,
            1e-8,
            0.01,
        ))),
        "rmsprop" => Ok(Box::new(RMSProp::new(lr.unwrap_or(0.001), 0.9, 1e-8))),
        _ => Err(TurkError::ValueError(format!(
            "Desteklenmeyen optimizasyon algoritması: {ad}"
        ))),
    }
}

fn egit_parametreleri_parse(icerik: &str) -> Result<EgitParametreleri> {
    let map = dsl_anahtar_degerler(icerik);
    let poz = dsl_pozisyonel_tokenler(icerik);

    let epochs = map
        .get("epoch")
        .or_else(|| map.get("epochs"))
        .cloned()
        .or_else(|| poz.first().cloned())
        .ok_or_else(|| TurkError::SyntaxError("egit: epoch belirtilmedi.".into()))?
        .parse::<usize>()
        .map_err(|_| TurkError::SyntaxError("Epoch sayısı tamsayı olmalıdır.".into()))?;

    let batch_size = map
        .get("batch_boyutu")
        .or_else(|| map.get("batch"))
        .cloned()
        .or_else(|| poz.get(1).cloned())
        .ok_or_else(|| TurkError::SyntaxError("egit: batch_boyutu belirtilmedi.".into()))?
        .parse::<usize>()
        .map_err(|_| TurkError::SyntaxError("Batch boyutu tamsayı olmalıdır.".into()))?;

    let devam_et = map
        .get("devam_et")
        .or_else(|| map.get("devam"))
        .map(|v| v.eq_ignore_ascii_case("true") || v == "1" || v.eq_ignore_ascii_case("evet"))
        .unwrap_or_else(|| {
            poz.get(2)
                .map(|v| {
                    v.eq_ignore_ascii_case("true") || v == "1" || v.eq_ignore_ascii_case("devam")
                })
                .unwrap_or(false)
        });

    let sabir = map
        .get("sabir")
        .or_else(|| map.get("patience"))
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| TurkError::SyntaxError(format!("sabir tamsayı olmalı: '{s}'")))
        })
        .transpose()?;

    let min_delta = map
        .get("min_delta")
        .or_else(|| map.get("esik"))
        .map(|s| {
            s.parse::<f64>()
                .map_err(|_| TurkError::SyntaxError(format!("min_delta sayısal olmalı: '{s}'")))
        })
        .transpose()?;

    Ok(EgitParametreleri {
        epochs,
        batch_size,
        devam_et,
        sabir,
        min_delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_olustur() {
        let mut dil = TurkDil::new();
        let result = dil.calistir("model_olustur()").unwrap();
        assert!(matches!(result, Value::None));
        assert!(dil.model.is_some());
    }

    #[test]
    fn test_katman_ekle() {
        let mut dil = TurkDil::new();
        dil.calistir("model_olustur()").unwrap();
        let result = dil.calistir("katman_ekle(yogun(10, 5, \"relu\"))").unwrap();
        assert!(matches!(result, Value::None));

        let model = dil.model.as_ref().unwrap();
        assert_eq!(model.layers.len(), 1);
    }

    #[test]
    fn test_csv_yukle_basit() {
        // Geçici CSV dosyası oluştur
        let dosya = "test_xor_dil.csv";
        std::fs::write(dosya, "x1,x2,label\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n").unwrap();

        let veri = CsvVeriAkisi::yeni(dosya, Some("label")).unwrap();
        assert_eq!(veri.ozellik_sayisi(), 2);

        std::fs::remove_file(dosya).ok();
    }

    #[test]
    fn test_veri_yukle_komutu() {
        let dosya = "test_veri_yukle_komutu.csv";
        std::fs::write(dosya, "x1,x2,y\n0,0,0\n1,1,0\n0,1,1\n1,0,1\n").unwrap();

        let mut dil = TurkDil::new();
        let sonuc = dil.calistir(&format!("veri_yukle(\"{dosya}\")"));
        assert!(sonuc.is_ok(), "veri_yukle komutu başarısız: {:?}", sonuc);
        assert!(dil.aktif_veri.is_some());

        std::fs::remove_file(dosya).ok();
    }

    #[test]
    fn test_egit_komutu_gercek_veri() {
        let dosya = "test_egit_gercek.csv";
        std::fs::write(dosya, "x1,x2,label\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n").unwrap();

        let mut dil = TurkDil::new();
        dil.calistir("model_olustur()").unwrap();
        dil.calistir("katman_ekle(yogun(2, 4, \"sigmoid\"))")
            .unwrap();
        dil.calistir("katman_ekle(yogun(4, 1, \"sigmoid\"))")
            .unwrap();
        dil.calistir("derle(adam)").unwrap();
        dil.calistir(&format!("veri_yukle(\"{dosya}\")")).unwrap();
        let sonuc = dil.calistir("egit(5, 4)");
        assert!(sonuc.is_ok(), "Eğitim başarısız: {:?}", sonuc);
        if let Ok(Value::Number(kayip)) = sonuc {
            assert!(kayip >= 0.0, "Kayıp negatif: {kayip}");
        }

        std::fs::remove_file(dosya).ok();
    }

    #[test]
    fn test_katman_boyut_kontrol() {
        let mut dil = TurkDil::new();
        dil.calistir("model_olustur()").unwrap();
        dil.calistir("katman_ekle(yogun(2, 4, \"relu\"))").unwrap();
        let sonuc = dil.kontrol_et("katman_ekle(yogun(8, 1, \"sigmoid\"))");
        assert!(sonuc.is_err());
    }

    #[test]
    fn test_egit_on_kontrol() {
        let dosya = "test_egit_on_kontrol.csv";
        std::fs::write(dosya, "x1,x2,label\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n").unwrap();

        let mut dil = TurkDil::new();
        dil.calistir("model_olustur()").unwrap();
        dil.calistir("katman_ekle(yogun(2, 4, \"sigmoid\"))")
            .unwrap();
        dil.calistir("katman_ekle(yogun(4, 1, \"sigmoid\"))")
            .unwrap();
        dil.calistir("derle(adam)").unwrap();
        dil.calistir(&format!("veri_yukle(\"{dosya}\")")).unwrap();
        assert!(dil.kontrol_et("egit(5, 4)").is_ok());

        std::fs::remove_file(dosya).ok();
    }

    #[test]
    fn test_derle_loss_parametresi() {
        let mut dil = TurkDil::new();
        dil.calistir("model_olustur()").unwrap();
        dil.calistir("katman_ekle(yogun(2, 1, \"sigmoid\"))")
            .unwrap();
        dil.calistir("derle(adam, loss=mse)").unwrap();
        let model = dil.model.as_ref().unwrap();
        assert!(model.is_compiled());
        assert_eq!(model.loss_name(), "mse");
    }

    #[test]
    fn test_veri_yukle_bolme() {
        let dosya = "test_veri_bolme.csv";
        std::fs::write(dosya, "x1,x2,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n").unwrap();

        let mut dil = TurkDil::new();
        dil.calistir(&format!("veri_yukle(\"{dosya}\", bolme=0.25)"))
            .unwrap();
        assert!(dil.aktif_veri.is_some());

        let veri = CsvVeriAkisi::yeni_bolme(dosya, Some("y"), Some(0.25)).unwrap();
        assert_eq!(veri.egitim_satir_sayisi(), 3);
        assert_eq!(veri.val_satir_sayisi(), 1);

        std::fs::remove_file(dosya).ok();
    }

    #[test]
    fn test_model_bilgisi_ozet() {
        let mut dil = TurkDil::new();
        dil.calistir("model_olustur()").unwrap();
        dil.calistir("katman_ekle(yogun(2, 4, \"relu\"))").unwrap();
        dil.calistir("derle(adam)").unwrap();
        let sonuc = dil.calistir("model_bilgisi()").unwrap();
        if let Value::Text(t) = sonuc {
            assert!(t.contains("TÜRK Model Özeti"));
            assert!(t.contains("Dense"));
            assert!(t.contains("Adam") || t.contains("adam"));
        } else {
            panic!("model_bilgisi Text döndürmeli");
        }
    }

    #[test]
    fn test_cnn_katman_zinciri_kontrol() {
        let mut dil = TurkDil::new();
        dil.calistir("model_olustur()").unwrap();
        dil.calistir("katman_ekle(conv2d(28, 28, 1, 8, 3, relu))")
            .unwrap();
        dil.calistir("katman_ekle(maxpool(26, 26, 8, 2))").unwrap();
        dil.calistir("katman_ekle(yogun(1352, 10, sigmoid))")
            .unwrap();
        dil.calistir("derle(adam)").unwrap();
        assert!(dil.kontrol_et("egit(1, 2)").is_err()); // veri yok — beklenen
    }

    #[test]
    fn test_mnist_784_boyut_esleme() {
        let dosya = "test_mnist_784.csv";
        let mut header = String::new();
        for i in 0..784 {
            header.push_str(&format!("p{i},"));
        }
        header.push_str("label\n");
        let satir = format!("{},0\n", "0.5,".repeat(784).trim_end_matches(','));
        std::fs::write(dosya, format!("{header}{satir}")).unwrap();

        let mut dil = TurkDil::new();
        dil.calistir("model_olustur()").unwrap();
        dil.calistir("katman_ekle(yogun(784, 128, relu))").unwrap();
        dil.calistir("derle(adam)").unwrap();
        dil.calistir(&format!("veri_yukle(\"{dosya}\", bolme=0.0)"))
            .unwrap();
        assert!(dil.kontrol_et("egit(1, 64)").is_ok());

        std::fs::remove_file(dosya).ok();
    }
}

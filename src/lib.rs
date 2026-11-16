// TÜRK dili projesi
// Türkçe yapay sinir ağı kütüphanesi

#![allow(dead_code)]

extern crate lazy_static;

// Standart modüller
pub mod error;
pub mod initializers;
pub mod layer;
pub mod loss;
pub mod metrics;
pub mod model;
pub mod optimizer;
pub mod preprocessing;
pub mod token;
pub mod utils;
pub mod aktivasyon;

// Lexer modülü
pub mod lexer;

// AST modülü
pub mod ast;

// TÜRK DSL modülü
pub mod dil;

// Katman kayıt sistemi (Katman trait + KatmanKayit fabrikası)
pub mod katman_kayit;

// Veri yükleme / lazy CSV akışı
pub mod veri;

// Evrişimli katmanlar
pub mod cnn;

// Kaynak konumu ve resmi DSL ayrıştırıcı
pub mod source_position;
pub mod dsl_parser;

// ONNX dışa aktarma
pub mod onnx;

// Hesaplama cihazı (CPU / GPU-paralel)
pub mod cihaz;

// Model Zoo
pub mod model_zoo;

// TÜRK Çekirdeği — yerel kişisel yapay zeka ekosistemi
#[cfg(feature = "turk_sistem")]
pub mod asistan;
#[cfg(feature = "turk_sistem")]
pub mod otomasyon;
#[cfg(feature = "turk_ses")]
pub mod ses;

// Re-export
pub use error::{Result, TurkError};
pub use dil::TurkDil;
pub use dil::{EgitimIlerleme, Value};
pub use dsl_parser::{DslKomut, komut_ayristir, hata_satirini_goster};
pub use source_position::SourcePosition;
pub use onnx::{export_onnx, import_onnx};
pub use model::{EarlyStopping, EpochCallback, History, Sequential};
pub use layer::{BatchNormalization, Layer, Dense, Dropout};
pub use cnn::{Conv2D, Flatten, MaxPool2D};
pub use optimizer::{Optimizer, SGD, Adam, AdamW, RMSProp};
pub use loss::{Loss, MSE, MAE, CrossEntropy};
pub use metrics::{Metric, Accuracy, Precision, Recall, F1};
pub use preprocessing::{StandardScaler, MinMaxScaler};
pub use initializers::{Initializer, GlorotUniform, GlorotNormal, HeUniform, HeNormal};
pub use aktivasyon::Activation;
pub use katman_kayit::{Katman, KatmanKayit, KatmanYukleVerisi};
pub use veri::{CsvVeriAkisi, CsvEpochOkuyucu};
pub use cihaz::{Cihaz, cihaz_listesi_metin, mevcut_cihazlar};
pub use model_zoo::{
    manifest_yukle, model_bul, model_bilgi_metin, model_indir, model_script_yolu,
    kullanici_zoo_dizini, zoo_kok, zoo_listesi_metin, ZooManifest, ZooModelMeta,
};

#[cfg(feature = "turk_sistem")]
pub use asistan::{
    AsistanBeyni, AsistanYanit, Beyin, BeyinTuru, Hafiza, HafizaHavuzu, HafizaMerkezi,
    SemantikKayit, TurkSistem, ZincirAdimi, ZincirSonuc, SISTEM_PROMPT, cok_adimli_mi,
    hafiza_tara, hafiza_tara_ozet, otonom_zincir_planla_ve_yurut, plan_olustur, zincir_yurut,
};
#[cfg(feature = "turk_sistem")]
pub use otomasyon::{
    AracCagrisiJson, AracKomut, SistemDurumu, SistemVekili, VekilKomut, VekilOlay, arac_calistir,
    canli_hava_durumu_cek, cpu_kullanim_guncelle, cpu_kullanim_oku, dosyalari_listele,
    dosya_olustur, dosya_yaz, proje_derle, proje_kontrol, sistem_durumu, terminal_komutu_calistir,
    TURK_VEKIL_OPTIMIZE_MESAJ, TURK_VEKIL_UYARI_MESAJ,
};

#[cfg(feature = "turk_ses")]
pub use ses::{KayitSonucu, Kulak, SesSinyal, konusmayi_metne_cevir, whisper_cli_yolu, whisper_hazir_mi, whisper_model_yolu};

/// Geriye uyumluluk alias'i
#[cfg(feature = "turk_sistem")]
pub type TurkCekirdek = TurkSistem;

// Kütüphane sürümü
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// Dosya açma yardımcı fonksiyonu
pub fn open_file<P: AsRef<Path>>(path: P) -> Result<File> {
    File::open(path.as_ref())
        .map_err(|e| TurkError::IOError(format!("{} dosyası açılamadı: {}", path.as_ref().display(), e)))
}

// Dosya okuma yardımcı fonksiyonu
pub fn read_file<P: AsRef<Path>>(path: P) -> Result<String> {
    let mut file = open_file(path.as_ref())?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|e| TurkError::IOError(format!("{} dosyası okunamadı: {}", path.as_ref().display(), e)))?;
    Ok(contents)
}

// Dosya oluşturma yardımcı fonksiyonu
pub fn create_file<P: AsRef<Path>>(path: P) -> Result<File> {
    File::create(path.as_ref())
        .map_err(|e| TurkError::IOError(format!("{} dosyası oluşturulamadı: {}", path.as_ref().display(), e)))
}

// Dosyaya yazma yardımcı fonksiyonu
pub fn write_file<P: AsRef<Path>>(path: P, contents: &str) -> Result<()> {
    let mut file = create_file(path.as_ref())?;
    file.write_all(contents.as_bytes())
        .map_err(|e| TurkError::IOError(format!("{} dosyasına yazılamadı: {}", path.as_ref().display(), e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        println!("TÜRK dili sürümü: {}", VERSION);
        assert!(!VERSION.is_empty());
    }
}

/// TÜRK bilgisi
pub struct TurkLang;

impl TurkLang {
    /// Versiyon bilgisini döndürür
    pub fn version() -> &'static str {
        VERSION
    }
    
    /// Kütüphane bilgilerini döndürür
    pub fn info() -> String {
        format!("TÜRK Dil Modeli ve Yapay Zeka Kütüphanesi v{}", VERSION)
    }
}

/// Ana TÜRK yapısı - Tüm TÜRK fonksiyonlarını içerir
#[derive(Debug)]
pub struct Turk {
    model: Option<Box<dyn model::Model>>,
    #[allow(dead_code)]
    variables: std::collections::HashMap<String, serde_json::Value>,
}

impl Turk {
    /// Yeni bir TÜRK nesnesi oluşturur
    pub fn new() -> Self {
        Self {
            model: None,
            variables: std::collections::HashMap::new(),
        }
    }

    /// Modeli dosyadan yükle
    pub fn model_yukle(&mut self, path: &Path) -> Result<()> {
        let mut model = model::Sequential::new();
        model.load(path.to_str().unwrap())?;
        self.model = Some(Box::new(model));
        Ok(())
    }
    
    /// Modeli dosyaya kaydet
    pub fn model_kaydet(&self, path: &Path) -> Result<()> {
        if let Some(model) = &self.model {
            model.save(path.to_str().unwrap())?;
            Ok(())
        } else {
            Err(TurkError::ModelError("Model tanımlanmamış".to_string()))
        }
    }
}

impl Default for Turk {
    fn default() -> Self {
        Self::new()
    }
}

use std::path::Path;
use std::io::{Read, Write};
use std::fs::File;

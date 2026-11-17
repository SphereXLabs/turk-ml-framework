//! Katman Kayıt Sistemi
//!
//! `Katman` trait'i ve `KatmanKayit` (factory registry) ile
//! yeni katman türleri eklemek için model.rs veya dil.rs'e dokunmak gerekmez.
//!
//! # Yeni Katman Ekleme Adımları
//!
//! 1. `Layer` trait'ini implement eden bir struct yaz.
//! 2. `KatmanKayit::global()` içinde `.kaydet(...)` ile iki fonksiyon sağla:
//!    - **DSL üreteci**: `fn(params: &str) -> Result<Box<dyn Layer>>`
//!    - **JSON yükleyici**: `fn(ld: &KatmanYukleVerisi) -> Result<Box<dyn Layer>>`
//! 3. Artık `.turk` dosyalarında o isimleri kullanabilirsin.
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::aktivasyon::Activation;
use crate::cnn::{Conv2D, Flatten, MaxPool2D};
use crate::error::{Result, TurkError};
use crate::initializers::{GlorotUniform, Zeros};
use crate::layer::{BatchNormalization, Dense, Dropout, Layer};

// ─── Katman Trait (Türkçe API) ─────────────────────────────────────────────

/// Ergonomik Türkçe katman arayüzü.
///
/// `Layer` trait'ini implemente eden her tür bu trait'i otomatik olarak kazanır
/// (blanket implementation).
use crate::dtype::Float;
pub trait Katman: Layer {
    /// İleri besleme — `Layer::forward`'ın Türkçe adı.
    fn ileri_besleme(
        &mut self,
        girdi: &ndarray::ArrayView2<Float>,
    ) -> Result<ndarray::Array2<Float>> {
        self.forward(girdi)
    }

    /// Geri besleme — `Layer::backward`'ın Türkçe adı.
    fn geri_besleme(
        &mut self,
        girdi: &ndarray::ArrayView2<Float>,
        gradyan: &ndarray::ArrayView2<Float>,
    ) -> Result<(
        ndarray::Array2<Float>,
        ndarray::Array2<Float>,
        ndarray::Array2<Float>,
    )> {
        self.backward(girdi, gradyan)
    }

    /// Giriş boyutu — `input_size`'ın Türkçe adı.
    fn girdi_boyutu(&self) -> usize {
        self.input_size()
    }

    /// Çıkış boyutu — `output_size`'ın Türkçe adı.
    fn cikti_boyutu(&self) -> usize {
        self.output_size()
    }

    /// Parametre sayısı — `param_count`'un Türkçe adı.
    fn parametre_sayisi(&self) -> usize {
        self.param_count()
    }

    /// Katman türü adı — `layer_type`'ın Türkçe adı.
    fn katman_turu(&self) -> &'static str {
        self.layer_type()
    }

    /// Ağırlık güncelleme — `update`'in Türkçe adı.
    fn parametreleri_guncelle(
        &mut self,
        grad_agirlik: &ndarray::ArrayView2<Float>,
        grad_bias: &ndarray::ArrayView2<Float>,
        lr: Float,
    ) -> Result<()> {
        self.update(grad_agirlik, grad_bias, lr)
    }
}

// Blanket impl: Layer'ı implemente eden her şey Katman da olur.
impl<T: Layer> Katman for T {}

// ─── Yükleme Verisi ────────────────────────────────────────────────────────

/// JSON'dan katman yüklerken ihtiyaç duyulan veri paketi.
/// Yeni katmanlar `ek_veriler` HashMap'ine kendi alanlarını yazabilir.
#[derive(Debug, Clone)]
pub struct KatmanYukleVerisi {
    pub agirliklar: Vec<Float>,
    pub biaslar: Vec<Float>,
    pub aktivasyon: Activation,
    pub girdi_boyutu: usize,
    pub cikti_boyutu: usize,
    pub dropout_orani: Option<Float>,
    /// Katmana özgü ek sayısal veriler (ör. "num_features", "bn_epsilon")
    pub ek_veriler: HashMap<String, Float>,
}

impl KatmanYukleVerisi {
    pub fn ek<T: Into<String>>(&self, anahtar: T) -> Float {
        self.ek_veriler.get(&anahtar.into()).copied().unwrap_or(0.0)
    }
}

// ─── Fabrika Fonksiyon Tipleri ─────────────────────────────────────────────

/// DSL parametrelerinden katman üretir.
///   Girdi: ham parametre string'i ("2, 8, relu" gibi)
pub type DslUreteciFn = fn(params: &str) -> Result<Box<dyn Layer>>;

/// JSON verisinden katman geri yükler.
pub type JsonYukleyiciFn = fn(veri: &KatmanYukleVerisi) -> Result<Box<dyn Layer>>;

// ─── KatmanKayit ───────────────────────────────────────────────────────────

/// Katman fabrikası — DSL ve JSON yükleme için tip bazında üretici fonksiyonlar tutar.
///
/// Yeni bir katman tipi eklemek için `kaydet()` çağırmak yeterli;
/// model.rs veya dil.rs çekirdeğine dokunmak gerekmez.
pub struct KatmanKayit {
    dsl: HashMap<String, DslUreteciFn>,
    json: HashMap<String, JsonYukleyiciFn>,
}

impl KatmanKayit {
    pub fn yeni() -> Self {
        Self {
            dsl: HashMap::new(),
            json: HashMap::new(),
        }
    }

    /// DSL adı ve fabrika fonksiyonlarıyla bir katman tipi kaydeder.
    /// `dsl_adlari`: boşluk ile ayrılmış takma adlar (ör. "yogun dense").
    pub fn kaydet(
        &mut self,
        dsl_adlari: &str,
        dsl_uretici: DslUreteciFn,
        json_adlari: &str,
        json_yukleyici: JsonYukleyiciFn,
    ) -> &mut Self {
        for ad in dsl_adlari.split_whitespace() {
            self.dsl.insert(ad.to_lowercase(), dsl_uretici);
        }
        for ad in json_adlari.split_whitespace() {
            self.json.insert(ad.to_string(), json_yukleyici);
        }
        self
    }

    /// DSL parametrelerinden katman üretir.
    /// `tip`: DSL anahtar kelimesi (ör. "yogun")
    /// `params`: parantez içindeki ham parametre dizesi
    pub fn dsl_olustur(&self, tip: &str, params: &str) -> Result<Box<dyn Layer>> {
        let anahtar = tip.trim().to_lowercase();
        match self.dsl.get(&anahtar) {
            Some(f) => f(params),
            None => Err(TurkError::SyntaxError(format!(
                "Bilinmeyen katman tipi: '{tip}'. \
                 Kayıtlı tipler: {}",
                self.dsl.keys().cloned().collect::<Vec<_>>().join(", ")
            ))),
        }
    }

    /// DSL string'ini tip ve parametrelere ayırarak katman üretir.
    /// Girdi: `"yogun(2, 8, relu)"` veya `"batch_norm(8)"`
    pub fn dsl_olustur_tamam(&self, katman_str: &str) -> Result<Box<dyn Layer>> {
        // "tip(params)" formatını ayır
        let paren = katman_str.find('(').ok_or_else(|| {
            TurkError::SyntaxError(format!(
                "Geçersiz katman formatı: '{katman_str}'. Beklenen: tip(parametre,...)"
            ))
        })?;
        let tip = katman_str[..paren].trim();
        let params =
            katman_str[paren + 1..katman_str.rfind(')').unwrap_or(katman_str.len())].trim();
        self.dsl_olustur(tip, params)
    }

    /// JSON verisinden katman geri yükler.
    /// `tip`: serileştirilmiş katman türü adı (ör. "Dense", "BatchNormalization")
    pub fn json_yukle(&self, tip: &str, veri: &KatmanYukleVerisi) -> Result<Box<dyn Layer>> {
        match self.json.get(tip) {
            Some(f) => f(veri),
            None => Err(TurkError::deserialization_error(&format!(
                "Bilinmeyen katman tipi JSON'da: '{tip}'. \
                 Kayıtlı tipler: {}",
                self.json.keys().cloned().collect::<Vec<_>>().join(", ")
            ))),
        }
    }

    /// Varsayılan katmanlar (Dense, Dropout, BatchNormalization) ile kayıtlı fabrika.
    pub fn varsayilan() -> &'static Self {
        static KAYIT: OnceLock<KatmanKayit> = OnceLock::new();
        KAYIT.get_or_init(|| {
            let mut k = KatmanKayit::yeni();
            k.kaydet("yogun dense", uret_yogun, "Dense", yukle_dense)
                .kaydet("dropout", uret_dropout, "Dropout", yukle_dropout)
                .kaydet(
                    "batch_norm batch_normalization batchnorm",
                    uret_batch_norm,
                    "BatchNormalization",
                    yukle_batch_norm,
                )
                .kaydet(
                    "conv2d conv evrim conv2",
                    uret_conv2d,
                    "Conv2D",
                    yukle_conv2d,
                )
                .kaydet(
                    "maxpool max_pool maxpool2d havuz",
                    uret_maxpool,
                    "MaxPool2D",
                    yukle_maxpool,
                )
                .kaydet("flatten duzlestir", uret_flatten, "Flatten", yukle_flatten);
            k
        })
    }
}

// ─── Dense üreteci / yükleyicisi ───────────────────────────────────────────

fn uret_yogun(params: &str) -> Result<Box<dyn Layer>> {
    let parts: Vec<&str> = params.split(',').map(str::trim).collect();
    if parts.len() < 2 {
        return Err(TurkError::SyntaxError(
            "yogun katman en az 2 parametre gerektirir: (girdi, cikti [, aktivasyon])".into(),
        ));
    }
    let girdi = parts[0].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("Girdi boyutu sayı olmalıdır: '{}'", parts[0]))
    })?;
    let cikti = parts[1].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("Çıktı boyutu sayı olmalıdır: '{}'", parts[1]))
    })?;
    let akt_str = parts
        .get(2)
        .map(|s| s.trim_matches('"').to_string())
        .unwrap_or_else(|| "linear".into());
    let akt = Activation::from_str(&akt_str)?;
    Ok(Box::new(Dense::new(
        girdi,
        cikti,
        akt,
        Box::new(GlorotUniform::new()),
        Box::new(Zeros::new()),
        true,
    )?))
}

fn yukle_dense(veri: &KatmanYukleVerisi) -> Result<Box<dyn Layer>> {
    let mut layer = Dense::new(
        veri.girdi_boyutu,
        veri.cikti_boyutu,
        veri.aktivasyon,
        Box::new(GlorotUniform::new()),
        Box::new(Zeros::new()),
        true,
    )?;
    if !veri.agirliklar.is_empty() {
        if let Ok(w) = ndarray::Array2::from_shape_vec(
            (veri.girdi_boyutu, veri.cikti_boyutu),
            veri.agirliklar.clone(),
        ) {
            layer.set_weights(&w.view())?;
        }
    }
    if !veri.biaslar.is_empty() {
        if let Ok(b) = ndarray::Array2::from_shape_vec((1, veri.cikti_boyutu), veri.biaslar.clone())
        {
            layer.set_biases(&b.view())?;
        }
    }
    Ok(Box::new(layer))
}

// ─── Dropout üreteci / yükleyicisi ─────────────────────────────────────────

fn uret_dropout(params: &str) -> Result<Box<dyn Layer>> {
    let oran = params
        .trim()
        .parse::<Float>()
        .map_err(|_| TurkError::SyntaxError(format!("Dropout oranı sayı olmalıdır: '{params}'")))?;
    if !(0.0..1.0).contains(&oran) {
        return Err(TurkError::ValueError(
            "Dropout oranı [0, 1) aralığında olmalıdır".into(),
        ));
    }
    Ok(Box::new(Dropout::new(oran)))
}

fn yukle_dropout(veri: &KatmanYukleVerisi) -> Result<Box<dyn Layer>> {
    let oran = veri.dropout_orani.ok_or_else(|| {
        TurkError::deserialization_error("Dropout katmanı için oran belirtilmemiş")
    })?;
    Ok(Box::new(Dropout::new(oran)))
}

// ─── BatchNormalization üreteci / yükleyicisi ───────────────────────────────

fn uret_batch_norm(params: &str) -> Result<Box<dyn Layer>> {
    let parts: Vec<&str> = params.split(',').map(str::trim).collect();
    let nf = parts[0].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!(
            "batch_norm: özellik sayısı tamsayı olmalıdır: '{}'",
            parts[0]
        ))
    })?;
    let bn = if parts.len() >= 3 {
        let eps = parts[1].parse::<Float>().unwrap_or(1e-5);
        let mom = parts[2].parse::<Float>().unwrap_or(0.1);
        BatchNormalization::with_params(nf, eps, mom)
    } else {
        BatchNormalization::new(nf)
    };
    Ok(Box::new(bn))
}

fn yukle_batch_norm(veri: &KatmanYukleVerisi) -> Result<Box<dyn Layer>> {
    let nf = if veri.ek("num_features") as usize > 0 {
        veri.ek("num_features") as usize
    } else {
        veri.cikti_boyutu
    };
    let eps = if veri.ek("bn_epsilon") > 0.0 {
        veri.ek("bn_epsilon")
    } else {
        1e-5
    };
    let mom = if veri.ek("bn_momentum") > 0.0 {
        veri.ek("bn_momentum")
    } else {
        0.1
    };

    let mut bn = BatchNormalization::with_params(nf, eps, mom);
    if !veri.agirliklar.is_empty() {
        if let Ok(g) = ndarray::Array2::from_shape_vec((1, nf), veri.agirliklar.clone()) {
            let _ = bn.set_weights(&g.view());
        }
    }
    if !veri.biaslar.is_empty() {
        if let Ok(b) = ndarray::Array2::from_shape_vec((1, nf), veri.biaslar.clone()) {
            let _ = bn.set_biases(&b.view());
        }
    }
    Ok(Box::new(bn))
}

// ─── Conv2D üreteci / yükleyicisi ───────────────────────────────────────────

/// `conv2d(in_h, in_w, in_c, filtre, kernel [, aktivasyon])`
fn uret_conv2d(params: &str) -> Result<Box<dyn Layer>> {
    let parts: Vec<&str> = params.split(',').map(str::trim).collect();
    if parts.len() < 5 {
        return Err(TurkError::SyntaxError(
            "conv2d: conv2d(yukseklik, genislik, kanal, filtre, cekirdek [, aktivasyon])".into(),
        ));
    }
    let in_h = parts[0].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("conv2d yükseklik tamsayı olmalı: '{}'", parts[0]))
    })?;
    let in_w = parts[1].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("conv2d genişlik tamsayı olmalı: '{}'", parts[1]))
    })?;
    let in_c = parts[2].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("conv2d kanal tamsayı olmalı: '{}'", parts[2]))
    })?;
    let filters = parts[3].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("conv2d filtre tamsayı olmalı: '{}'", parts[3]))
    })?;
    let kernel = parts[4].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("conv2d çekirdek tamsayı olmalı: '{}'", parts[4]))
    })?;
    let akt_str = parts
        .get(5)
        .map(|s| s.trim_matches('"').to_string())
        .unwrap_or_else(|| "relu".into());
    let akt = Activation::from_str(&akt_str)?;
    Ok(Box::new(Conv2D::new(
        in_h, in_w, in_c, filters, kernel, akt,
    )?))
}

fn yukle_conv2d(veri: &KatmanYukleVerisi) -> Result<Box<dyn Layer>> {
    let in_h = veri.ek("in_h") as usize;
    let in_w = veri.ek("in_w") as usize;
    let in_c = veri.ek("in_c") as usize;
    let out_c = veri.ek("out_c") as usize;
    let kernel = veri.ek("kernel") as usize;
    if in_h == 0 || in_w == 0 || in_c == 0 || out_c == 0 || kernel == 0 {
        return Err(TurkError::deserialization_error(
            "Conv2D metadata eksik (in_h, in_w, in_c, out_c, kernel)",
        ));
    }
    let mut layer = Conv2D::new(in_h, in_w, in_c, out_c, kernel, veri.aktivasyon)?;
    if !veri.agirliklar.is_empty() {
        let k = kernel * kernel * in_c;
        if let Ok(w) = ndarray::Array2::from_shape_vec((k, out_c), veri.agirliklar.clone()) {
            layer.set_weights(&w.view())?;
        }
    }
    if !veri.biaslar.is_empty() {
        if let Ok(b) = ndarray::Array2::from_shape_vec((1, out_c), veri.biaslar.clone()) {
            layer.set_biases(&b.view())?;
        }
    }
    Ok(Box::new(layer))
}

// ─── MaxPool2D üreteci / yükleyicisi ────────────────────────────────────────

/// `maxpool(in_h, in_w, kanal, havuz_boyutu)`
fn uret_maxpool(params: &str) -> Result<Box<dyn Layer>> {
    let parts: Vec<&str> = params.split(',').map(str::trim).collect();
    if parts.len() < 4 {
        return Err(TurkError::SyntaxError(
            "maxpool: maxpool(yukseklik, genislik, kanal, havuz)".into(),
        ));
    }
    let in_h = parts[0].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("maxpool yükseklik tamsayı olmalı: '{}'", parts[0]))
    })?;
    let in_w = parts[1].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("maxpool genişlik tamsayı olmalı: '{}'", parts[1]))
    })?;
    let ch = parts[2].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("maxpool kanal tamsayı olmalı: '{}'", parts[2]))
    })?;
    let pool = parts[3].parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("maxpool havuz tamsayı olmalı: '{}'", parts[3]))
    })?;
    Ok(Box::new(MaxPool2D::new(in_h, in_w, ch, pool)?))
}

fn yukle_maxpool(veri: &KatmanYukleVerisi) -> Result<Box<dyn Layer>> {
    let in_h = veri.ek("in_h") as usize;
    let in_w = veri.ek("in_w") as usize;
    let ch = veri.ek("in_c") as usize;
    let pool = veri.ek("pool") as usize;
    Ok(Box::new(MaxPool2D::new(in_h, in_w, ch, pool)?))
}

// ─── Flatten üreteci / yükleyicisi ──────────────────────────────────────────

fn uret_flatten(params: &str) -> Result<Box<dyn Layer>> {
    let size = params.trim().parse::<usize>().map_err(|_| {
        TurkError::SyntaxError(format!("flatten: boyut tamsayı olmalı: '{params}'"))
    })?;
    Ok(Box::new(Flatten::new(size)))
}

fn yukle_flatten(veri: &KatmanYukleVerisi) -> Result<Box<dyn Layer>> {
    let size = if veri.ek("size") as usize > 0 {
        veri.ek("size") as usize
    } else {
        veri.girdi_boyutu
    };
    Ok(Box::new(Flatten::new(size)))
}

// ─── Testler ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Layer;

    #[test]
    fn test_kayit_dsl_conv2d() {
        let k = KatmanKayit::varsayilan();
        let l = k
            .dsl_olustur_tamam("conv2d(28, 28, 1, 8, 3, relu)")
            .unwrap();
        assert_eq!(l.layer_type(), "Conv2D");
        assert_eq!(l.input_size(), 784);
        assert_eq!(l.output_size(), 26 * 26 * 8);
    }

    #[test]
    fn test_kayit_dsl_maxpool() {
        let k = KatmanKayit::varsayilan();
        let l = k.dsl_olustur_tamam("maxpool(26, 26, 8, 2)").unwrap();
        assert_eq!(l.layer_type(), "MaxPool2D");
        assert_eq!(l.output_size(), 13 * 13 * 8);
    }

    #[test]
    fn test_kayit_dsl_yogun() {
        let k = KatmanKayit::varsayilan();
        let l = k.dsl_olustur_tamam("yogun(2, 4, relu)").unwrap();
        assert_eq!(l.layer_type(), "Dense");
        assert_eq!(l.input_size(), 2);
        assert_eq!(l.output_size(), 4);
    }

    #[test]
    fn test_kayit_dsl_dropout() {
        let k = KatmanKayit::varsayilan();
        let l = k.dsl_olustur_tamam("dropout(0.3)").unwrap();
        assert_eq!(l.layer_type(), "Dropout");
    }

    #[test]
    fn test_kayit_dsl_batch_norm() {
        let k = KatmanKayit::varsayilan();
        let l = k.dsl_olustur_tamam("batch_norm(8)").unwrap();
        assert_eq!(l.layer_type(), "BatchNormalization");
    }

    #[test]
    fn test_kayit_bilinmeyen_hata() {
        let k = KatmanKayit::varsayilan();
        assert!(k.dsl_olustur_tamam("bilinmeyen(5)").is_err());
    }

    #[test]
    fn test_katman_trait_blanket() {
        use crate::initializers::{GlorotUniform, Zeros};
        // Dense direkt impl — Katman blanket impl burda çalışır
        let mut l = Dense::new(
            2,
            3,
            crate::aktivasyon::Activation::Sigmoid,
            Box::new(GlorotUniform::new()),
            Box::new(Zeros::new()),
            true,
        )
        .unwrap();
        assert_eq!(l.katman_turu(), "Dense");
        assert_eq!(l.girdi_boyutu(), 2);
        assert_eq!(l.cikti_boyutu(), 3);
        let x = ndarray::Array2::ones((1, 2));
        let out = l.ileri_besleme(&x.view()).unwrap();
        assert_eq!(out.shape(), &[1, 3]);
    }

    #[test]
    fn test_ozel_katman_kayit() {
        // Yeni katman = sadece Layer impl + kayıt. model.rs'e dokunma yok!
        use crate::error::Result as TResult;
        use ndarray::{Array2, ArrayView2};

        #[derive(Debug, Clone)]
        struct OzelKatman {
            boyut: usize,
        }

        impl Layer for OzelKatman {
            fn forward(&mut self, input: &ArrayView2<Float>) -> TResult<Array2<Float>> {
                Ok(input.to_owned())
            }
            fn backward(
                &self,
                _i: &ArrayView2<Float>,
                g: &ArrayView2<Float>,
            ) -> TResult<(Array2<Float>, Array2<Float>, Array2<Float>)> {
                Ok((g.to_owned(), Array2::zeros((0, 0)), Array2::zeros((0, 0))))
            }
            fn update(
                &mut self,
                _gw: &ArrayView2<Float>,
                _gb: &ArrayView2<Float>,
                _lr: Float,
            ) -> TResult<()> {
                Ok(())
            }
            fn get_weights(&self) -> TResult<Array2<Float>> {
                Ok(Array2::zeros((0, 0)))
            }
            fn get_biases(&self) -> TResult<Array2<Float>> {
                Ok(Array2::zeros((0, 0)))
            }
            fn activation_type(&self) -> crate::aktivasyon::Activation {
                crate::aktivasyon::Activation::Linear
            }
            fn layer_type(&self) -> &'static str {
                "OzelKatman"
            }
            fn input_size(&self) -> usize {
                self.boyut
            }
            fn output_size(&self) -> usize {
                self.boyut
            }
            fn param_count(&self) -> usize {
                0
            }
            fn set_trainable(&mut self, _t: bool) {}
            fn is_trainable(&self) -> bool {
                false
            }
            fn clone_box(&self) -> Box<dyn Layer> {
                Box::new(self.clone())
            }
            fn reset(&mut self) {}
            fn weights_as_vec(&self) -> Vec<Float> {
                vec![]
            }
            fn biases_as_vec(&self) -> Vec<Float> {
                vec![]
            }
            fn dropout_rate(&self) -> Option<Float> {
                None
            }
            fn use_batch_norm(&self) -> Option<bool> {
                None
            }
            fn set_weights(&mut self, _w: &ArrayView2<Float>) -> TResult<()> {
                Ok(())
            }
            fn set_biases(&mut self, _b: &ArrayView2<Float>) -> TResult<()> {
                Ok(())
            }
        }

        // Özel fabrika kayıt
        let mut ozel_kayit = KatmanKayit::yeni();
        ozel_kayit.kaydet(
            "ozel_katman",
            |p| {
                let boyut = p.trim().parse::<usize>().unwrap_or(4);
                Ok(Box::new(OzelKatman { boyut }))
            },
            "OzelKatman",
            |_veri| Ok(Box::new(OzelKatman { boyut: 4 })),
        );

        let l = ozel_kayit.dsl_olustur_tamam("ozel_katman(8)").unwrap();
        assert_eq!(l.layer_type(), "OzelKatman");
        assert_eq!(l.input_size(), 8);
    }
}

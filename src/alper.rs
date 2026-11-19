//! ALPER — yerli Türkçe niyet asistanı (masaüstü / Sequential).
//!
//! TF-IDF (unigram+bigram) + `Sequential` MLP; ağırlıklar `~/.turk/alper_brain.bin`
//! üzerinden ikili önbelleğe alınır.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::aktivasyon::Activation;
use crate::dtype::Float;
use crate::error::{Result, TurkError};
use crate::initializers::{GlorotUniform, Zeros};
use crate::layer::Dense;
use crate::loss::CrossEntropy;
use crate::model::Sequential;
use crate::optimizer::Adam;

pub const ALPER_VERSION: &str = "3.1-native";
const VOCAB_LIMIT: usize = 420;
const HIDDEN1: usize = 64;
const HIDDEN2: usize = 32;
const TRAIN_EPOCHS: usize = 55;
const TRAIN_BATCH: usize = 16;
const CACHE_SCHEMA: u32 = 1;

pub const INTENTS: &[&str] = &[
    "selamlama",
    "hal_hatir",
    "kimlik_sohbet",
    "kullanici_hafiza",
    "saat_tarih",
    "hesaplama",
    "model_egit_xor",
    "kod_uretme_dsl",
    "egitim_teshis",
    "hiperparametre_ayar",
    "kavram_aciklama",
    "sistem_durumu",
    "takip_detay",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabTerm {
    pub term: String,
    pub idf: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserProfile {
    pub name: Option<String>,
    pub role: Option<String>,
}

#[derive(Debug, Clone)]
pub struct IntentSonuc {
    pub intent: String,
    pub confidence: f64,
    pub index: usize,
}

#[derive(Debug, Clone)]
pub struct AlperYanit {
    pub intent: String,
    pub confidence: f64,
    pub metin: String,
    /// Editöre yazılacak `.turk` DSL (varsa)
    pub dsl: Option<String>,
    pub egitim_tetikle: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BrainCacheMeta {
    schema: u32,
    version: String,
    intents: Vec<String>,
    sample_count: usize,
    accuracy: f64,
    vocab: Vec<VocabTerm>,
}

/// Yerleşik eğitim cümleleri (niyet → örnekler).
fn dataset() -> HashMap<&'static str, Vec<&'static str>> {
    let mut m = HashMap::new();
    m.insert(
        "selamlama",
        vec![
            "selam",
            "merhaba",
            "selamlar",
            "günaydın",
            "iyi akşamlar",
            "hey alper",
            "merhaba alper",
            "selam alper",
            "hello",
            "hi",
            "iyi günler",
            "hoş geldin",
            "alper selam",
            "selam dostum",
            "günaydın alper",
        ],
    );
    m.insert(
        "hal_hatir",
        vec![
            "nasılsın",
            "naber",
            "ne haber",
            "iyi misin",
            "nasıl gidiyor",
            "keyfin nasıl",
            "alper nasılsın",
            "ne var ne yok",
            "durumlar nasıl",
            "işler nasıl",
            "moralin nasıl",
            "bugün nasılsın",
        ],
    );
    m.insert(
        "kimlik_sohbet",
        vec![
            "sen kimsin",
            "adın ne",
            "senin adın ne",
            "neler yapabilirsin",
            "kendini tanıt",
            "alper kim",
            "llm kullanıyor musun",
            "nasıl çalışıyorsun",
            "spherex labs mı yaptı",
            "harici api var mı",
            "sen alper misin",
            "ne işe yararsın",
        ],
    );
    m.insert(
        "kullanici_hafiza",
        vec![
            "benim adım sinan",
            "adım ayşe",
            "benim adım ne",
            "adımı hatırlıyor musun",
            "beni tanıyor musun",
            "ben kimim",
            "profilimi söyle",
            "adımı kaydet",
            "ben mühendisim",
            "ben geliştiriciyim",
            "ismim mehmet",
            "beni hatırla",
        ],
    );
    m.insert(
        "saat_tarih",
        vec![
            "saat kaç",
            "saat kaçtır",
            "bugün tarih ne",
            "tarih kaç",
            "hangi günüz",
            "şu an saat",
            "zaman söyle",
            "bugünün tarihi",
            "alper saat kaç",
            "saati söylesene",
            "tarihi söyle",
            "yerel saat",
        ],
    );
    m.insert(
        "hesaplama",
        vec![
            "hesapla 3+5",
            "hesapla 12*7",
            "karekök 16",
            "50 eksi 8",
            "100 bölü 4",
            "7*8 kaç eder",
            "hesapla 9*9",
            "15+27 kaç",
            "karekök 25",
            "2+2",
            "hesapla 18+7",
            "matematik 6*6",
        ],
    );
    m.insert(
        "model_egit_xor",
        vec![
            "xor modelini eğit",
            "xor eğit",
            "xor ağı kur",
            "xor çalıştır",
            "xor mlp eğit",
            "xor eğitim başlat",
            "xor ağını eğit",
            "basit xor eğitimi",
            "xor motorunu başlat",
            "xor öğren",
            "xor fit et",
            "eğit xor",
        ],
    );
    m.insert(
        "kod_uretme_dsl",
        vec![
            "xor için turk dsl üret",
            "xor dsl yaz",
            "dsl üret",
            "turk kod üret",
            ".turk script üret",
            "derin mlp dsl yaz",
            "dsl kod örneği",
            "xor script oluştur",
            "katman_ekle dsl",
            "turk dsl örneği",
            "egitim scripti üret",
            "dsl şablon xor",
        ],
    );
    m.insert(
        "egitim_teshis",
        vec![
            "eğitim teşhisi yap",
            "loss düşmüyor neden",
            "neden loss yüksek",
            "accuracy artmıyor",
            "model öğrenmiyor",
            "teşhis et",
            "loss salınıyor",
            "eğitim takıldı",
            "neden yakınsamıyor",
            "val loss yükseliyor",
            "teşhis alper",
            "eğitim debug",
        ],
    );
    m.insert(
        "hiperparametre_ayar",
        vec![
            "hiperparametre öner",
            "öğrenme oranı kaç olmalı",
            "lr ne olsun",
            "batch size öner",
            "epoch kaç olmalı",
            "adam lr öner",
            "hiperparametre ayarla",
            "lr düşür öner",
            "batch öner",
            "xor hiperparametre",
            "öğrenme oranı düşür",
            "tuning yardımı",
        ],
    );
    m.insert(
        "kavram_aciklama",
        vec![
            "relu nedir",
            "relu'yu anlat",
            "softmax nedir",
            "adam nedir",
            "dropout nedir",
            "wasm nedir",
            "overfitting nedir",
            "mse nedir",
            "cross entropy nedir",
            "mlp nedir",
            "backpropagation nedir",
            "öğrenme oranı nedir",
            "tf idf nedir",
            "embedding nedir",
            "conv1d nedir",
        ],
    );
    m.insert(
        "sistem_durumu",
        vec![
            "sistem durumu",
            "status",
            "sağlık kontrolü",
            "beyin eğitildi mi",
            "sistem raporu",
            "alper hazır mı",
            "vocab kaç",
            "niyet sayısı",
            "kokpit durumu",
            "sürüm ne",
            "checkup",
            "durum özeti",
        ],
    );
    m.insert(
        "takip_detay",
        vec![
            "neden",
            "detay ver",
            "detaylandır",
            "bunu açıkla",
            "bunu biraz daha aç",
            "peki nasıl",
            "örnek kod ver",
            "devam et",
            "uygula",
            "bunu uygula",
            "dezavantajı ne",
            "daha derin anlat",
        ],
    );
    m
}

fn kavram_makale(konu: &str) -> Option<&'static str> {
    let n = normalize_tr(konu);
    if n.contains("relu") {
        Some(
            "**ReLU**\n- `f(x)=max(0,x)`\n- Vanishing gradient’i azaltır\n- TÜRK: `yogun(..., \"relu\")`",
        )
    } else if n.contains("softmax") {
        Some("**Softmax**\n- Çok sınıflı olasılık; toplam 1\n- Cross-entropy ile çift\n- ALPER son katmanda Softmax kullanır")
    } else if n.contains("adam") {
        Some("**Adam**\n- Momentum + RMSprop\n- XOR `lr≈0.05`, ALPER `lr≈0.035`\n- `derle(adam, ...)`")
    } else if n.contains("dropout") {
        Some("**Dropout**\n- Eğitimde nöron kapatma → genelleme\n- `katman_ekle(dropout(0.1))`")
    } else if n.contains("wasm") {
        Some("**WASM**\n- Tarayıcıda native’e yakın baytkod\n- Kokpit/playground aynı paketi paylaşır")
    } else if n.contains("overfit") || n.contains("asiri uyum") || n.contains("aşırı") {
        Some("**Overfitting**\n- Train iyi, val kötü\n- Dropout, early stopping, daha fazla veri")
    } else if n.contains("mse") {
        Some("**MSE**\n- `(ŷ-y)²` ortalaması\n- XOR/regresyon demolarında klasik")
    } else if n.contains("cross") || n.contains("entropy") {
        Some("**Cross-Entropy**\n- Sınıflandırma loss’u\n- Softmax ile birlikte")
    } else if n.contains("mlp") || n.contains("sinir") {
        Some("**MLP**\n- Dense yığını\n- ALPER: girdi→64→32→niyet (softmax)")
    } else if n.contains("backprop") || n.contains("geri yay") {
        Some("**Backpropagation**\n- Zincir kuralı ile `∂L/∂W`\n- Forward → loss → backward → update")
    } else if n.contains("ogrenme") || n.contains("learning") || n.contains("lr") {
        Some("**Öğrenme oranı**\n- Adım büyüklüğü η\n- Yüksek → salınım; düşük → yavaş yakınsama")
    } else if n.contains("tf") || n.contains("idf") || n.contains("bigram") {
        Some("**TF-IDF + Bigram**\n- `benim_adim` vs `senin_adin`\n- ALPER giriş vektörü")
    } else if n.contains("embed") || n.contains("gomme") || n.contains("gömme") {
        Some("**Embedding**\n- `gommeli(kelime=..., boyut=...)`\n- Lookup tablosu; TF-IDF’siz kelime vektörleri")
    } else if n.contains("conv1") || n.contains("evrisim") {
        Some("**Conv1D**\n- `evrisim1d(giris=..., cikis=..., cekirdek=...)`\n- 1B kayan pencere + geri yayılım")
    } else {
        None
    }
}

fn xor_dsl() -> &'static str {
    r#"veri_yukle("xor")
model_olustur()
katman_ekle(yogun(2, 8, "relu"))
katman_ekle(yogun(8, 1, "sigmoid"))
derle(adam, loss=mse, ogrenme_orani=0.05)
model_bilgisi()
egit(120, 4)
"#
}

fn derin_dsl() -> &'static str {
    r#"veri_yukle("kume", bolme=0.25)
model_olustur()
katman_ekle(yogun(2, 16, "relu"))
katman_ekle(batch_norm(16))
katman_ekle(yogun(16, 8, "relu"))
katman_ekle(dropout(0.1))
katman_ekle(yogun(8, 1, "sigmoid"))
derle(adam, loss=mse, ogrenme_orani=0.03)
egit(200, 4, sabir=25, min_delta=0.0001)
"#
}

static STOP: &[&str] = &[
    "ve", "veya", "ile", "bir", "bu", "su", "o", "da", "de", "ki", "mi", "mu", "icin", "gibi",
    "kadar", "daha", "en", "cok", "ya", "ama", "var", "yok",
];

pub fn normalize_tr(s: &str) -> String {
    let lower = s.to_lowercase();
    let mut out = String::with_capacity(lower.len());
    for c in lower.chars() {
        let mapped = match c {
            'ı' | 'İ' => 'i',
            'ğ' | 'Ğ' => 'g',
            'ü' | 'Ü' => 'u',
            'ş' | 'Ş' => 's',
            'ö' | 'Ö' => 'o',
            'ç' | 'Ç' => 'c',
            _ => c,
        };
        if mapped.is_ascii_alphanumeric() || mapped == ' ' || mapped == '_' || mapped == '-' {
            out.push(mapped);
        } else if mapped.is_whitespace() {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn stem_tr(tok: &str) -> String {
    let suffixes = [
        "lariniz", "leriniz", "iyorlar", "uyorlar", "makta", "mekte", "lardan", "lerden", "iyor",
        "uyor", "arak", "erek", "dir", "dur", "tir", "tur", "dan", "den", "nin", "nun", "lar",
        "ler", "lik", "luk", "mak", "mek", "in", "un", "im", "um",
    ];
    let mut t = tok.to_string();
    for s in suffixes {
        if t.len() > s.len() + 2 && t.ends_with(s) {
            t.truncate(t.len() - s.len());
            break;
        }
    }
    t
}

pub fn tokenize(text: &str) -> Vec<String> {
    let n = normalize_tr(text);
    let uni: Vec<String> = n
        .split_whitespace()
        .map(stem_tr)
        .filter(|t| t.len() > 1 && !STOP.contains(&t.as_str()))
        .collect();
    let mut out = uni.clone();
    for i in 0..uni.len().saturating_sub(1) {
        out.push(format!("{}_{}", uni[i], uni[i + 1]));
    }
    out
}

fn turk_home() -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".turk")
}

fn brain_bin_path() -> PathBuf {
    turk_home().join("alper_brain.bin")
}

fn meta_path() -> PathBuf {
    turk_home().join("alper_brain_meta.json")
}

fn profile_path() -> PathBuf {
    turk_home().join("alper_user_profile.json")
}

fn dense(i: usize, o: usize, act: Activation) -> Result<Dense> {
    Dense::new(
        i,
        o,
        act,
        Box::new(GlorotUniform::new()),
        Box::new(Zeros::new()),
        true,
    )
}

/// Masaüstü ALPER beyni.
pub struct AlperBeyin {
    model: Sequential,
    vocab: Vec<VocabTerm>,
    pub profile: UserProfile,
    pub last_intent: Option<String>,
    pub last_topic: Option<String>,
    pub accuracy: f64,
    pub sample_count: usize,
    pub from_cache: bool,
}

impl AlperBeyin {
    /// Önbellekten yükler veya eğitir.
    pub fn ac() -> Result<Self> {
        let mut profile = UserProfile::default();
        if let Ok(raw) = fs::read_to_string(profile_path()) {
            if let Ok(p) = serde_json::from_str::<UserProfile>(&raw) {
                profile = p;
            }
        }

        if let Ok(beyin) = Self::onbellekten_yukle(profile.clone()) {
            return Ok(beyin);
        }

        let mut beyin = Self::bos(profile)?;
        beyin.egit_ve_kaydet()?;
        Ok(beyin)
    }

    fn bos(profile: UserProfile) -> Result<Self> {
        Ok(Self {
            model: Sequential::new(),
            vocab: Vec::new(),
            profile,
            last_intent: None,
            last_topic: None,
            accuracy: 0.0,
            sample_count: 0,
            from_cache: false,
        })
    }

    fn fingerprint_ok(meta: &BrainCacheMeta) -> bool {
        meta.schema == CACHE_SCHEMA
            && meta.version == ALPER_VERSION
            && meta.intents.len() == INTENTS.len()
            && meta.intents.iter().zip(INTENTS.iter()).all(|(a, b)| a == b)
    }

    fn onbellekten_yukle(profile: UserProfile) -> Result<Self> {
        let meta_raw = fs::read_to_string(meta_path())
            .map_err(|e| TurkError::io_error(&format!("ALPER meta okunamadı: {e}")))?;
        let meta: BrainCacheMeta = serde_json::from_str(&meta_raw)
            .map_err(|e| TurkError::deserialization_error(&format!("ALPER meta: {e}")))?;
        if !Self::fingerprint_ok(&meta) {
            return Err(TurkError::RuntimeError(
                "ALPER önbellek sürümü uyuşmuyor".into(),
            ));
        }
        let mut model = Sequential::new();
        model.modeli_ikili_yukle(brain_bin_path().to_str().unwrap_or("alper_brain.bin"))?;
        // Softmax sınıflandırma için CE bağla
        let lr: Float = 0.035;
        model.compile_with_loss(
            Box::new(Adam::new(lr, 0.9, 0.999, 1e-8)),
            Box::new(CrossEntropy::new(false)),
        );
        Ok(Self {
            model,
            vocab: meta.vocab,
            profile,
            last_intent: None,
            last_topic: None,
            accuracy: meta.accuracy,
            sample_count: meta.sample_count,
            from_cache: true,
        })
    }

    fn build_vocab(docs: &[String]) -> Vec<VocabTerm> {
        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in docs {
            let uniq: HashSet<String> = tokenize(doc).into_iter().collect();
            for t in uniq {
                *df.entry(t).or_default() += 1;
            }
        }
        let n = docs.len() as f64;
        let mut terms: Vec<_> = df.into_iter().collect();
        terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        terms.truncate(VOCAB_LIMIT);
        terms
            .into_iter()
            .map(|(term, d)| VocabTerm {
                idf: ((1.0 + n) / (1.0 + d as f64)).ln() + 1.0,
                term,
            })
            .collect()
    }

    fn vectorize(&self, text: &str) -> Vec<Float> {
        let toks = tokenize(text);
        let mut tf: HashMap<&str, f64> = HashMap::new();
        for t in &toks {
            *tf.entry(t.as_str()).or_default() += 1.0;
        }
        let len = toks.len().max(1) as f64;
        self.vocab
            .iter()
            .map(|v| {
                let c = tf.get(v.term.as_str()).copied().unwrap_or(0.0);
                (c / len * v.idf) as Float
            })
            .collect()
    }

    fn training_matrices(&self) -> Result<(Array2<Float>, Array2<Float>, usize)> {
        let ds = dataset();
        let mut phrases: Vec<(usize, String)> = Vec::new();
        for (i, intent) in INTENTS.iter().enumerate() {
            if let Some(list) = ds.get(intent) {
                for p in list {
                    phrases.push((i, (*p).to_string()));
                }
            }
        }
        let n = phrases.len();
        let d = self.vocab.len();
        let c = INTENTS.len();
        let mut x = Array2::zeros((n, d));
        let mut y = Array2::zeros((n, c));
        for (row, (idx, phrase)) in phrases.iter().enumerate() {
            let v = {
                // geçici vocab ile vektörle
                let toks = tokenize(phrase);
                let mut tf: HashMap<&str, f64> = HashMap::new();
                for t in &toks {
                    *tf.entry(t.as_str()).or_default() += 1.0;
                }
                let len = toks.len().max(1) as f64;
                self.vocab
                    .iter()
                    .map(|vt| {
                        let c = tf.get(vt.term.as_str()).copied().unwrap_or(0.0);
                        (c / len * vt.idf) as Float
                    })
                    .collect::<Vec<_>>()
            };
            for (j, val) in v.into_iter().enumerate() {
                x[[row, j]] = val;
            }
            y[[row, *idx]] = 1.0;
        }
        Ok((x, y, n))
    }

    /// Zorla yeniden eğit ve diske yaz.
    pub fn egit_ve_kaydet(&mut self) -> Result<f64> {
        let ds = dataset();
        let docs: Vec<String> = INTENTS
            .iter()
            .flat_map(|i| {
                ds.get(i)
                    .into_iter()
                    .flat_map(|v| v.iter().map(|s| (*s).to_string()))
            })
            .collect();
        self.vocab = Self::build_vocab(&docs);
        let (x, y, n) = self.training_matrices()?;
        let input_dim = self.vocab.len();
        let output_dim = INTENTS.len();

        let mut model = Sequential::new();
        model
            .add(dense(input_dim, HIDDEN1, Activation::Relu)?)
            .add(dense(HIDDEN1, HIDDEN2, Activation::Relu)?)
            .add(dense(HIDDEN2, output_dim, Activation::Softmax)?);
        let lr: Float = 0.035;
        model.compile_with_loss(
            Box::new(Adam::new(lr, 0.9, 0.999, 1e-8)),
            Box::new(CrossEntropy::new(false)),
        );
        let _hist = model.fit(&x.view(), &y.view(), TRAIN_EPOCHS, TRAIN_BATCH.min(n), None)?;

        // Doğruluk
        let pred = model.predict(&x.view())?;
        let mut correct = 0usize;
        for i in 0..n {
            let mut best = 0usize;
            let mut best_v = pred[[i, 0]];
            for j in 1..output_dim {
                if pred[[i, j]] > best_v {
                    best_v = pred[[i, j]];
                    best = j;
                }
            }
            let mut true_j = 0usize;
            for j in 0..output_dim {
                if y[[i, j]] > 0.5 {
                    true_j = j;
                    break;
                }
            }
            if best == true_j {
                correct += 1;
            }
        }
        let acc = correct as f64 / n as f64;
        self.model = model;
        self.accuracy = acc;
        self.sample_count = n;
        self.from_cache = false;
        self.kaydet_onbellek()?;
        Ok(acc)
    }

    fn kaydet_onbellek(&self) -> Result<()> {
        let dir = turk_home();
        fs::create_dir_all(&dir)
            .map_err(|e| TurkError::io_error(&format!("~/.turk oluşturulamadı: {e}")))?;
        let bin = brain_bin_path();
        self.model
            .modeli_ikili_kaydet(bin.to_str().unwrap_or("alper_brain.bin"))?;
        let meta = BrainCacheMeta {
            schema: CACHE_SCHEMA,
            version: ALPER_VERSION.into(),
            intents: INTENTS.iter().map(|s| (*s).to_string()).collect(),
            sample_count: self.sample_count,
            accuracy: self.accuracy,
            vocab: self.vocab.clone(),
        };
        fs::write(
            meta_path(),
            serde_json::to_string_pretty(&meta)
                .map_err(|e| TurkError::serialization_error(&e.to_string()))?,
        )
        .map_err(|e| TurkError::io_error(&format!("ALPER meta yazılamadı: {e}")))?;
        self.profil_kaydet()?;
        Ok(())
    }

    pub fn profil_kaydet(&self) -> Result<()> {
        let dir = turk_home();
        let _ = fs::create_dir_all(&dir);
        fs::write(
            profile_path(),
            serde_json::to_string_pretty(&self.profile)
                .map_err(|e| TurkError::serialization_error(&e.to_string()))?,
        )
        .map_err(|e| TurkError::io_error(&format!("profil yazılamadı: {e}")))?;
        Ok(())
    }

    pub fn siniflandir(&mut self, text: &str) -> Result<IntentSonuc> {
        if self.vocab.is_empty() || self.model.layers.is_empty() {
            return Err(TurkError::ModelError("ALPER beyni hazır değil".into()));
        }
        let v = self.vectorize(text);
        let x = Array2::from_shape_vec((1, v.len()), v)
            .map_err(|e| TurkError::ShapeMismatch(e.to_string()))?;
        let pred = self.model.predict(&x.view())?;
        let mut best = 0usize;
        let mut best_v = pred[[0, 0]];
        for j in 1..INTENTS.len() {
            if pred[[0, j]] > best_v {
                best_v = pred[[0, j]];
                best = j;
            }
        }
        Ok(IntentSonuc {
            intent: INTENTS[best].to_string(),
            confidence: best_v,
            index: best,
        })
    }

    fn extract_name(text: &str) -> Option<String> {
        let lower = text.to_lowercase();
        for key in ["benim adım ", "benim adim ", "adım ", "adim ", "ismim "] {
            if let Some(rest) = lower.split(key).nth(1) {
                let name = rest
                    .split_whitespace()
                    .next()?
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_string();
                if name.len() >= 2 {
                    let mut chars = name.chars();
                    let first = chars.next()?.to_uppercase().to_string();
                    return Some(first + chars.as_str());
                }
            }
        }
        None
    }

    fn safe_eval(text: &str) -> Option<f64> {
        // Operatörleri koru — normalize_tr +/* siler
        let mut expr = text.to_lowercase();
        for (a, b) in [
            ('ı', 'i'),
            ('ğ', 'g'),
            ('ü', 'u'),
            ('ş', 's'),
            ('ö', 'o'),
            ('ç', 'c'),
        ] {
            expr = expr.replace(a, &b.to_string());
        }
        for w in [
            "hesapla",
            "kaç eder",
            "kac eder",
            "sonuç",
            "sonuc",
            "matematik",
            "eşittir",
            "esittir",
        ] {
            expr = expr.replace(w, " ");
        }
        expr = expr.replace("karekök", " ").replace("karekok", " ");
        expr = expr.replace('×', "*").replace('÷', "/").replace(',', ".");
        let mut s = String::new();
        for c in expr.chars() {
            if c.is_ascii_digit() || matches!(c, '+' | '-' | '*' | '/' | '.') {
                s.push(c);
            }
        }
        if s.is_empty() || !s.chars().any(|c| c.is_ascii_digit()) {
            return None;
        }
        // çok basit sol-sağ değerlendirici (yalnızca * / sonra + -)
        fn parse_num(tok: &str) -> Option<f64> {
            tok.parse().ok()
        }
        let tokens: Vec<char> = s.chars().collect();
        let mut nums = Vec::new();
        let mut ops = Vec::new();
        let mut cur = String::new();
        for c in tokens {
            if c.is_ascii_digit() || c == '.' {
                cur.push(c);
            } else if matches!(c, '+' | '-' | '*' | '/') {
                if cur.is_empty() && c == '-' {
                    cur.push('-');
                    continue;
                }
                nums.push(parse_num(&cur)?);
                cur.clear();
                ops.push(c);
            }
        }
        nums.push(parse_num(&cur)?);
        // * /
        let mut i = 0;
        while i < ops.len() {
            if ops[i] == '*' || ops[i] == '/' {
                let a = nums[i];
                let b = nums[i + 1];
                let r = if ops[i] == '*' {
                    a * b
                } else if b.abs() < 1e-12 {
                    return None;
                } else {
                    a / b
                };
                nums[i] = r;
                nums.remove(i + 1);
                ops.remove(i);
            } else {
                i += 1;
            }
        }
        let mut acc = nums[0];
        for (op, n) in ops.iter().zip(nums.iter().skip(1)) {
            acc = match op {
                '+' => acc + n,
                '-' => acc - n,
                _ => return None,
            };
        }
        Some(acc)
    }

    /// Kullanıcı mesajını işle.
    pub fn yanitla(&mut self, text: &str) -> Result<AlperYanit> {
        let n = normalize_tr(text);
        let mut classified = self.siniflandir(text)?;

        // Kural güçlendirmeleri
        if n.contains("benim adim ne") || n.contains("beni tani") {
            classified.intent = "kullanici_hafiza".into();
            classified.confidence = classified.confidence.max(0.9);
        } else if n.contains("senin adin") || n == "adin ne" || n.contains("sen kimsin") {
            classified.intent = "kimlik_sohbet".into();
            classified.confidence = classified.confidence.max(0.9);
        } else if (n.starts_with("neden")
            || n.starts_with("detay")
            || n.starts_with("peki")
            || n.contains("bunu acikla")
            || n.contains("ornek kod"))
            && self.last_intent.is_some()
            && classified.confidence < 0.85
        {
            classified.intent = "takip_detay".into();
        }

        let intent = classified.intent.as_str();
        let conf = classified.confidence;
        let greet = self
            .profile
            .name
            .as_ref()
            .map(|n| format!(", **{n}**"))
            .unwrap_or_default();

        let mut dsl = None;
        let mut egitim_tetikle = None;
        let metin = match intent {
            "selamlama" => format!(
                "Selam{greet}!\nBen **ALPER** (yerel Sequential, v{ALPER_VERSION}).\n- {} niyet · {:.1}% doğruluk{}\nNasıl yardımcı olayım?",
                INTENTS.len(),
                self.accuracy * 100.0,
                if self.from_cache { " · önbellek" } else { "" }
            ),
            "hal_hatir" => format!(
                "Teşekkürler{greet} — yerelde hazırım.\n- Beyin: %{:.1}\n- Profil: {}\nSizin gününüz nasıl?",
                self.accuracy * 100.0,
                self.profile.name.as_deref().unwrap_or("—")
            ),
            "kimlik_sohbet" => format!(
                "**Ben ALPER.**\n- SphereX Labs × turk-ml-framework\n- TF-IDF+bigram → MLP 64→32→{}\n- Harici LLM yok\n- Önbellek: ~/.turk/alper_brain.bin",
                INTENTS.len()
            ),
            "kullanici_hafiza" => {
                if let Some(name) = Self::extract_name(text) {
                    self.profile.name = Some(name.clone());
                    let _ = self.profil_kaydet();
                    format!("Memnun oldum, **{name}**!\nAdınız `~/.turk/alper_user_profile.json` dosyasına kaydedildi.")
                } else if n.contains("adim ne") || n.contains("tani") || n.contains("kimim") {
                    if let Some(ref name) = self.profile.name {
                        format!("Sizi **{name}** olarak tanıyorum.")
                    } else {
                        "Henüz adınızı bilmiyorum. Örnek: «benim adım Sinan»".into()
                    }
                } else {
                    format!(
                        "Profil:\n- Ad: {}\n- Rol: {}",
                        self.profile.name.as_deref().unwrap_or("(yok)"),
                        self.profile.role.as_deref().unwrap_or("(yok)")
                    )
                }
            }
            "saat_tarih" => {
                use std::time::SystemTime;
                let _ = SystemTime::now();
                // chrono feature olmayabilir — basit yerel
                #[cfg(feature = "chrono")]
                {
                    let now = chrono::Local::now();
                    format!(
                        "**Yerel zaman**\n- {}",
                        now.format("%A %d %B %Y · %H:%M:%S")
                    )
                }
                #[cfg(not(feature = "chrono"))]
                {
                    "**Saat/tarih:** sistem saatine bakın (chrono özelliği kapalı). «hesapla» veya XOR eğitin.".into()
                }
            }
            "hesaplama" => {
                if let Some(v) = Self::safe_eval(text) {
                    format!("**Hesap:** `{v}`")
                } else {
                    "Anlaşılır aritmetik bulamadım. Örnek: `hesapla 12*7+3`".into()
                }
            }
            "model_egit_xor" => {
                egitim_tetikle = Some("xor");
                dsl = Some(xor_dsl().to_string());
                format!("XOR eğitimini başlatıyorum{greet}. DSL editöre de yazıldı.")
            }
            "kod_uretme_dsl" => {
                let deep = n.contains("derin") || n.contains("mlp") || n.contains("kume");
                let code = if deep { derin_dsl() } else { xor_dsl() };
                dsl = Some(code.to_string());
                format!(
                    "**Üretilen `.turk` DSL** ({})\nEditöre aktarıldı.",
                    if deep { "derin" } else { "xor" }
                )
            }
            "egitim_teshis" => {
                "**Eğitim teşhisi**\n- Loss↓ yoksa: lr 0.01–0.05, batch küçült\n- Val↑ train↓: dropout / sabır\n- Acc~%50: etiket/özellik kontrol\n- XOR: 80+ epoch; derin: sabır=25".into()
            }
            "hiperparametre_ayar" => {
                "**Hiperparametreler**\n- XOR: Adam lr=0.05, batch=4, ~120 epoch\n- Derin: lr=0.03, sabır=25\n- ALPER beyin: lr=0.035, batch=16, CE+Softmax".into()
            }
            "kavram_aciklama" => kavram_makale(text)
                .unwrap_or(
                    "**Kavram**\nReLU, Adam, Softmax, Dropout, WASM, Embedding, Conv1D deneyin.",
                )
                .to_string(),
            "sistem_durumu" => format!(
                "**Sistem**\n- ALPER v{ALPER_VERSION}\n- Doğruluk: %{:.1}\n- Örnek: {}\n- Vocab: {}\n- Önbellek: {}\n- Profil: {}",
                self.accuracy * 100.0,
                self.sample_count,
                self.vocab.len(),
                if self.from_cache { "evet" } else { "hayır (yeni eğitildi)" },
                self.profile.name.as_deref().unwrap_or("—")
            ),
            "takip_detay" => {
                let prev = self.last_topic.clone().unwrap_or_else(|| "genel".into());
                if n.contains("uygula") || n.contains("egit") {
                    egitim_tetikle = Some("xor");
                    dsl = Some(xor_dsl().to_string());
                    format!("Önceki konu `{prev}` → XOR eğitimi uygulanıyor.")
                } else if let Some(art) = kavram_makale(&prev) {
                    format!("**Takip (`{prev}`)**\n{art}")
                } else {
                    format!(
                        "**Takip:** `{prev}`\nDaha spesifik sorun veya «örnek kod ver» / «uygula» deyin."
                    )
                }
            }
            _ => format!("Niyet `{intent}` işlendi (güven %{:.0}).", conf * 100.0),
        };

        self.last_intent = Some(intent.to_string());
        self.last_topic = Some(if intent == "kavram_aciklama" {
            normalize_tr(text)
        } else {
            intent.to_string()
        });

        Ok(AlperYanit {
            intent: intent.to_string(),
            confidence: conf,
            metin,
            dsl,
            egitim_tetikle,
        })
    }

    pub fn ozet(&self) -> String {
        format!(
            "ALPER v{ALPER_VERSION} · niyet={} · örnek={} · vocab={} · acc%{:.1} · cache={}",
            INTENTS.len(),
            self.sample_count,
            self.vocab.len(),
            self.accuracy * 100.0,
            self.from_cache
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_bigram() {
        let t = tokenize("benim adım ne");
        assert!(t.iter().any(|x| x.contains('_')));
        assert!(t.iter().any(|x| x.contains("adim") || x.contains("benim")));
    }

    #[test]
    fn test_safe_eval() {
        assert!((AlperBeyin::safe_eval("hesapla 3+5").unwrap() - 8.0).abs() < 1e-9);
        assert!((AlperBeyin::safe_eval("12*7").unwrap() - 84.0).abs() < 1e-9);
    }

    #[test]
    fn test_dataset_intents() {
        let ds = dataset();
        for i in INTENTS {
            assert!(ds.get(i).map(|v| !v.is_empty()).unwrap_or(false), "{i}");
        }
    }

    #[test]
    fn test_alper_egit_siniflandir() {
        let mut b = AlperBeyin::bos(UserProfile::default()).unwrap();
        let acc = b.egit_ve_kaydet().unwrap();
        assert!(acc > 0.85, "acc={acc}");
        let r = b.siniflandir("selam alper").unwrap();
        assert_eq!(r.intent, "selamlama");
        let y = b.yanitla("xor için turk dsl üret").unwrap();
        assert!(y.dsl.is_some());
    }
}

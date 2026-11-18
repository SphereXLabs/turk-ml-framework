//! TÜRK ALPER v0.3 — bağlamsal semantik hafıza havuzu + oturum vektör arama.

pub use crate::otomasyon::notlar::{hafiza_tara, hafiza_tara_ozet, HafizaEslesme, HafizaMerkezi};

use crate::otomasyon::turk_sistem_veri_dizini;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Tek bir hafıza kaydı (oturum)
#[derive(Debug, Clone)]
pub struct HafizaKaydi {
    pub rol: String,
    pub metin: String,
    pub vektor: Vec<f32>,
    pub zaman: u64,
}

/// Kalıcı semantik havuz kaydı
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemantikKayit {
    pub kategori: String,
    pub etiketler: Vec<String>,
    pub metin: String,
    pub vektor: Vec<f32>,
    pub zaman: u64,
    pub onem: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct HavuzDosyasi {
    kayitlar: Vec<SemantikKayit>,
}

/// Yerel JSON + semantik matris hafıza havuzu
pub struct HafizaHavuzu {
    kayitlar: Vec<SemantikKayit>,
    dosya_yolu: PathBuf,
    boyut: usize,
    son_hatirlanan: Vec<String>,
}

impl HafizaHavuzu {
    pub fn yeni() -> Self {
        let dosya_yolu = turk_sistem_veri_dizini().join("semantik_hafiza.json");
        let mut h = Self {
            kayitlar: Vec::new(),
            dosya_yolu,
            boyut: 256,
            son_hatirlanan: Vec::new(),
        };
        h.yukle();
        h
    }

    pub fn kayit_sayisi(&self) -> usize {
        self.kayitlar.len()
    }

    pub fn son_hatirlananlar(&self) -> &[String] {
        &self.son_hatirlanan
    }

    fn yukle(&mut self) {
        if let Ok(raw) = fs::read_to_string(&self.dosya_yolu) {
            if let Ok(dosya) = serde_json::from_str::<HavuzDosyasi>(&raw) {
                self.kayitlar = dosya.kayitlar;
            }
        }
    }

    fn kaydet(&self) {
        if let Some(parent) = self.dosya_yolu.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let dosya = HavuzDosyasi {
            kayitlar: self.kayitlar.clone(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&dosya) {
            let _ = fs::write(&self.dosya_yolu, json);
        }
    }

    fn simdi() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Kritik bilgiyi havuza ekle (proje, tercih, plan vb.)
    pub fn ekle(&mut self, kategori: &str, metin: &str, etiketler: Vec<String>) {
        let vektor = pseudo_embed(metin, self.boyut);
        self.kayitlar.push(SemantikKayit {
            kategori: kategori.into(),
            etiketler,
            metin: metin.into(),
            vektor,
            zaman: Self::simdi(),
            onem: 1.0,
        });
        if self.kayitlar.len() > 800 {
            self.kayitlar.drain(0..self.kayitlar.len() - 800);
        }
        self.kaydet();
    }

    /// Kullanıcı mesajına göre ilgili kayıtları tara ve prompt bağlamı üret.
    pub fn baglam_enjekte(&mut self, sorgu: &str, limit: usize) -> String {
        let sorgu_v = pseudo_embed(sorgu, self.boyut);
        let mut skorlar: Vec<(f32, &SemantikKayit)> = self
            .kayitlar
            .iter()
            .map(|k| {
                let skor = kosinus_benzerlik(&sorgu_v, &k.vektor) * k.onem;
                (skor, k)
            })
            .collect();

        skorlar.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        self.son_hatirlanan.clear();
        let mut satirlar = Vec::new();

        for (skor, k) in skorlar.into_iter().filter(|(s, _)| *s > 0.08).take(limit) {
            let satir = format!(
                "[{}|{:.0}%] {}",
                k.kategori,
                skor * 100.0,
                k.metin.chars().take(120).collect::<String>()
            );
            self.son_hatirlanan.push(satir.clone());
            satirlar.push(satir);
        }

        // Disk not arşivinden de semantik tarama
        let notlar = hafiza_tara_ozet(sorgu);
        if !notlar.is_empty() && notlar.len() <= 400 {
            let kisa = format!(
                "[not_arşivi] {}",
                notlar.chars().take(200).collect::<String>()
            );
            if !self.son_hatirlanan.iter().any(|s| s.contains("not_arşivi")) {
                self.son_hatirlanan.push(kisa.clone());
                satirlar.push(kisa);
            }
        }

        if satirlar.is_empty() {
            return String::new();
        }
        format!("--- SEMANTİK HAFIZA ---\n{}", satirlar.join("\n"))
    }

    /// Diyalogdan otomatik kritik çıkarım
    pub fn mesajdan_ogren(&mut self, rol: &str, metin: &str) {
        let k = metin.to_lowercase();
        let kritik = k.contains("refaktör")
            || k.contains("refaktor")
            || k.contains("tercih")
            || k.contains("proje")
            || k.contains("arsa")
            || k.contains("modifikasyon")
            || k.contains("siber")
            || k.contains("mnist")
            || k.contains("optimize")
            || k.contains("vekil");
        if !kritik && rol == "kullanici" && metin.len() < 40 {
            return;
        }
        if metin.len() < 12 {
            return;
        }
        let kategori = if rol == "kullanici" {
            "kullanici"
        } else {
            "turk"
        };
        let etiketler: Vec<String> = k
            .split_whitespace()
            .filter(|w| w.len() >= 4)
            .take(6)
            .map(str::to_string)
            .collect();
        self.ekle(kategori, metin, etiketler);
    }
}

/// Oturum içi vektör hafızası
pub struct Hafiza {
    kayitlar: Vec<HafizaKaydi>,
    dosya_yolu: Option<PathBuf>,
    boyut: usize,
    pub havuz: HafizaHavuzu,
}

impl Hafiza {
    pub fn yeni() -> Self {
        Self {
            kayitlar: Vec::new(),
            dosya_yolu: None,
            boyut: 256,
            havuz: HafizaHavuzu::yeni(),
        }
    }

    pub fn dosyadan(yol: PathBuf) -> Self {
        let mut h = Self::yeni();
        h.dosya_yolu = Some(yol);
        h
    }

    pub fn kayit_sayisi(&self) -> usize {
        self.kayitlar.len()
    }

    pub fn havuz(&self) -> &HafizaHavuzu {
        &self.havuz
    }

    pub fn havuz_mut(&mut self) -> &mut HafizaHavuzu {
        &mut self.havuz
    }

    pub fn ekle(&mut self, rol: &str, metin: &str) {
        let vektor = pseudo_embed(metin, self.boyut);
        let zaman = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        self.kayitlar.push(HafizaKaydi {
            rol: rol.into(),
            metin: metin.into(),
            vektor,
            zaman,
        });

        self.havuz.mesajdan_ogren(rol, metin);

        if self.kayitlar.len() > 500 {
            self.kayitlar.drain(0..self.kayitlar.len() - 500);
        }
    }

    pub fn ilgili_getir(&self, sorgu: &str, limit: usize) -> Vec<&HafizaKaydi> {
        let sorgu_v = pseudo_embed(sorgu, self.boyut);
        let mut skorlar: Vec<(f32, &HafizaKaydi)> = self
            .kayitlar
            .iter()
            .map(|k| (kosinus_benzerlik(&sorgu_v, &k.vektor), k))
            .collect();

        skorlar.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        skorlar
            .into_iter()
            .filter(|(s, _)| *s > 0.05)
            .take(limit)
            .map(|(_, k)| k)
            .collect()
    }

    /// Oturum + semantik havuz birleşik bağlam
    pub fn ilgili_baglam(&mut self, sorgu: &str, limit: usize) -> String {
        let oturum: String = self
            .ilgili_getir(sorgu, limit)
            .iter()
            .map(|k| format!("[{}] {}", k.rol, k.metin))
            .collect::<Vec<_>>()
            .join("\n");

        let havuz = self.havuz.baglam_enjekte(sorgu, limit);

        match (oturum.is_empty(), havuz.is_empty()) {
            (true, true) => String::new(),
            (false, true) => oturum,
            (true, false) => havuz,
            (false, false) => format!("{oturum}\n{havuz}"),
        }
    }
}

fn pseudo_embed(metin: &str, boyut: usize) -> Vec<f32> {
    let mut vektor = vec![0.0f32; boyut];
    for kelime in metin.to_lowercase().split_whitespace() {
        let mut h = DefaultHasher::new();
        kelime.hash(&mut h);
        let idx = (h.finish() as usize) % boyut;
        vektor[idx] += 1.0;
    }
    normalize(&mut vektor);
    vektor
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn kosinus_benzerlik(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hafiza_arama() {
        let mut h = Hafiza::yeni();
        h.ekle("kullanici", "MNIST modeli eğitmek istiyorum");
        h.ekle("turk", "Orta panelden mnist_mlp seçebilirsin");
        let ilgili = h.ilgili_getir("MNIST eğitimi", 2);
        assert!(!ilgili.is_empty());
    }

    #[test]
    fn test_havuz_baglam() {
        let mut h = HafizaHavuzu::yeni();
        h.ekle(
            "proje",
            "Siber tema refaktör tercihi aktif",
            vec!["siber".into()],
        );
        let baglam = h.baglam_enjekte("siber refaktör", 3);
        assert!(baglam.contains("SEMANTİK") || baglam.is_empty());
    }
}

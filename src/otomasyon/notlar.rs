//! Uzun vadeli not arşivi taraması — ~/.turk/sistem/notlar/

use super::turk_sistem_veri_dizini;
use chrono::{Datelike, Local, TimeZone};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Not dizinindeki eşleşen kayıt
#[derive(Debug, Clone)]
pub struct HafizaEslesme {
    pub dosya_adi: String,
    pub yol: PathBuf,
    /// İnsan odaklı tarih: "24 Haziran 2026 tarihli notunuz"
    pub olusturulma: String,
    pub onizleme: String,
    pub zaman_unix: u64,
}

/// Uzun vadeli not arşivi
pub struct HafizaMerkezi;

impl HafizaMerkezi {
    pub fn notlar_dizini() -> PathBuf {
        turk_sistem_veri_dizini().join("notlar")
    }
}

/// Belirtilen anahtar kelimeyi not .txt dosyalarında arar (büyük/küçük harf duyarsız).
pub fn hafiza_tara(anahtar_kelime: &str) -> Vec<HafizaEslesme> {
    let anahtar = anahtar_kelime.trim();
    if anahtar.is_empty() {
        return Vec::new();
    }

    let dizin = HafizaMerkezi::notlar_dizini();
    if !dizin.is_dir() {
        return Vec::new();
    }

    let anahtar_kucuk = anahtar.to_lowercase();
    let kelimeler: Vec<String> = anahtar_kucuk
        .split_whitespace()
        .filter(|k| k.len() >= 2)
        .map(str::to_string)
        .collect();

    let mut sonuclar = Vec::new();

    let Ok(girdiler) = fs::read_dir(&dizin) else {
        return sonuclar;
    };

    for girdi in girdiler.flatten() {
        let yol = girdi.path();
        if !yol.is_file() {
            continue;
        }
        if yol.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }

        let dosya_adi = yol
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("not.txt")
            .to_string();

        let Ok(icerik) = fs::read_to_string(&yol) else {
            continue;
        };

        if !metin_eslesir(&dosya_adi, &icerik, &anahtar_kucuk, &kelimeler) {
            continue;
        }

        let zaman_unix = dosya_zaman_damgasi(&yol);
        let onizleme = icerik
            .lines()
            .find(|l| satir_eslesir(l, &anahtar_kucuk, &kelimeler))
            .or_else(|| icerik.lines().next())
            .unwrap_or("")
            .chars()
            .take(160)
            .collect::<String>();

        sonuclar.push(HafizaEslesme {
            dosya_adi: dosya_adi.clone(),
            olusturulma: insan_okunur_tarih(zaman_unix),
            onizleme,
            zaman_unix,
            yol,
        });
    }

    sonuclar.sort_by(|a, b| b.zaman_unix.cmp(&a.zaman_unix));
    sonuclar
}

/// Araç çıktısı için biçimlendirilmiş özet metin.
pub fn hafiza_tara_ozet(anahtar_kelime: &str) -> String {
    let eslesmeler = hafiza_tara(anahtar_kelime);
    if eslesmeler.is_empty() {
        return format!(
            "«{anahtar_kelime}» için not bulunamadı.\n\
             Dizin: {}",
            HafizaMerkezi::notlar_dizini().display()
        );
    }

    let mut satirlar = vec![format!(
        "«{anahtar_kelime}» — {} eşleşme bulundu:",
        eslesmeler.len()
    )];

    for (i, e) in eslesmeler.iter().take(8).enumerate() {
        satirlar.push(format!(
            "\n{}. 📄 {}\n   Dosya: {}\n   Özet: {}\n   Yol: {}",
            i + 1,
            e.olusturulma,
            e.dosya_adi,
            e.onizleme,
            e.yol.display()
        ));
    }

    satirlar.join("")
}

fn metin_eslesir(dosya_adi: &str, icerik: &str, anahtar_kucuk: &str, kelimeler: &[String]) -> bool {
    let dosya_kucuk = dosya_adi.to_lowercase();
    let icerik_kucuk = icerik.to_lowercase();

    if dosya_kucuk.contains(anahtar_kucuk) || icerik_kucuk.contains(anahtar_kucuk) {
        return true;
    }

    if kelimeler.is_empty() {
        return false;
    }

    kelimeler
        .iter()
        .any(|k| dosya_kucuk.contains(k.as_str()) || icerik_kucuk.contains(k.as_str()))
}

fn satir_eslesir(satir: &str, anahtar_kucuk: &str, kelimeler: &[String]) -> bool {
    let k = satir.to_lowercase();
    k.contains(anahtar_kucuk) || kelimeler.iter().any(|w| k.contains(w.as_str()))
}

fn dosya_zaman_damgasi(path: &Path) -> u64 {
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        if let Some(ts) = stem.strip_prefix("not_") {
            if let Ok(secs) = ts.parse::<u64>() {
                return secs;
            }
        }
    }

    fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn insan_okunur_tarih(zaman_unix: u64) -> String {
    if zaman_unix == 0 {
        return "tarihi bilinmeyen notunuz".into();
    }

    Local
        .timestamp_opt(zaman_unix as i64, 0)
        .single()
        .map(|dt| {
            format!(
                "{} {} {} tarihli notunuz",
                dt.day(),
                turkce_ay_adi(dt.month()),
                dt.year()
            )
        })
        .unwrap_or_else(|| "tarihi bilinmeyen notunuz".into())
}

fn turkce_ay_adi(ay: u32) -> &'static str {
    match ay {
        1 => "Ocak",
        2 => "Şubat",
        3 => "Mart",
        4 => "Nisan",
        5 => "Mayıs",
        6 => "Haziran",
        7 => "Temmuz",
        8 => "Ağustos",
        9 => "Eylül",
        10 => "Ekim",
        11 => "Kasım",
        12 => "Aralık",
        _ => "Ay",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insan_okunur_tarih() {
        // 21 Haziran 2026 UTC civarı
        let metin = insan_okunur_tarih(1_750_502_400);
        assert!(metin.contains("tarihli notunuz"));
    }

    #[test]
    fn test_metin_eslesir_dosya_adi() {
        assert!(metin_eslesir(
            "notlar_ozet.txt",
            "içerik",
            "notlar",
            &["notlar".into()]
        ));
        assert!(metin_eslesir(
            "a.txt",
            "selam dünya",
            "selam",
            &["selam".into()]
        ));
        assert!(!metin_eslesir("a.txt", "merhaba", "xyz", &["xyz".into()]));
    }
}

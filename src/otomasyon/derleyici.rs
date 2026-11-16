//! TÜRK terminal derleyici otomasyonu — cargo check ve kendi kendini onarma döngüsü.

use super::kod_ustasi::{cozumle_yol, dosya_oku, dosya_yaz};
use super::sistem::proje_kok_dizini;
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_ONARIM_DENEMESI: usize = 4;

/// Derleyici mesajı
#[derive(Debug, Clone)]
pub struct DerlemeHatasi {
    pub dosya: String,
    pub satir: u32,
    pub mesaj: String,
    pub seviye: String,
}

/// `cargo check --message-format=json` çıktısı
#[derive(Debug)]
pub struct CargoCheckSonuc {
    pub basarili: bool,
    pub ham_cikti: String,
    pub hatalar: Vec<DerlemeHatasi>,
    pub uyarilar: Vec<DerlemeHatasi>,
}

fn cargo_komutu() -> Command {
    #[cfg(windows)]
    if let Ok(home) = std::env::var("USERPROFILE") {
        let cargo = PathBuf::from(home)
            .join(".cargo")
            .join("bin")
            .join("cargo.exe");
        if cargo.is_file() {
            return Command::new(cargo);
        }
    }
    Command::new("cargo")
}

fn proje_dizini(yol: &str) -> PathBuf {
    if yol.trim().is_empty() || yol == "." {
        proje_kok_dizini()
    } else {
        PathBuf::from(yol)
    }
}

/// Proje kökünde cargo check çalıştırır (senkron — arka plan thread'inde çağrılır).
pub fn cargo_check_json(proje_yolu: &str) -> Result<CargoCheckSonuc, String> {
    let dir = proje_dizini(proje_yolu);
    if !dir.join("Cargo.toml").is_file() {
        return Err(format!("Cargo.toml bulunamadı: {}", dir.display()));
    }

    let out = cargo_komutu()
        .args([
            "check",
            "--message-format=json",
            "--features",
            "turk_sistem",
        ])
        .current_dir(&dir)
        .output()
        .map_err(|e| format!("cargo çalıştırılamadı: {e}"))?;

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let birlesik = format!("{stdout}{stderr}");
    let (hatalar, uyarilar) = json_mesajlari_ayikla(&birlesik);

    Ok(CargoCheckSonuc {
        basarili: out.status.success() && hatalar.is_empty(),
        ham_cikti: birlesik,
        hatalar,
        uyarilar,
    })
}

/// Asenkron sarmalayıcı — GUI thread'ini bloklamaz.
pub fn cargo_check_json_async(proje_yolu: String) -> Result<CargoCheckSonuc, String> {
    std::thread::Builder::new()
        .name("turk-cargo-check".into())
        .spawn(move || cargo_check_json(&proje_yolu))
        .map_err(|e| format!("cargo thread başlatılamadı: {e}"))?
        .join()
        .map_err(|_| "cargo thread panikledi.".to_string())?
}

fn json_mesajlari_ayikla(cikti: &str) -> (Vec<DerlemeHatasi>, Vec<DerlemeHatasi>) {
    let mut hatalar = Vec::new();
    let mut uyarilar = Vec::new();
    let mut gorulen = HashSet::new();

    for satir in cikti.lines() {
        let satir = satir.trim();
        if satir.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(satir) else {
            continue;
        };
        if v.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
            continue;
        }
        let Some(msg) = v.get("message") else {
            continue;
        };
        let seviye = msg
            .get("level")
            .and_then(|l| l.as_str())
            .unwrap_or("unknown")
            .to_string();

        if seviye != "error" && seviye != "warning" {
            continue;
        }

        let aciklama = msg
            .get("message")
            .and_then(|m| m.as_str())
            .or_else(|| msg.get("rendered").and_then(|m| m.as_str()))
            .unwrap_or("bilinmeyen mesaj")
            .lines()
            .next()
            .unwrap_or("bilinmeyen mesaj")
            .to_string();

        let (dosya, satir_no) = msg
            .get("spans")
            .and_then(|s| s.as_array())
            .and_then(|arr| arr.first())
            .map(|span| {
                (
                    span.get("file_name")
                        .and_then(|f| f.as_str())
                        .unwrap_or("?")
                        .to_string(),
                    span.get("line_start").and_then(|l| l.as_u64()).unwrap_or(0) as u32,
                )
            })
            .unwrap_or_else(|| ("?".into(), 0));

        let kayit = DerlemeHatasi {
            dosya,
            satir: satir_no,
            mesaj: aciklama,
            seviye: seviye.clone(),
        };

        let anahtar = (
            kayit.dosya.clone(),
            kayit.satir,
            kayit.mesaj.clone(),
            seviye.clone(),
        );
        if !gorulen.insert(anahtar) {
            continue;
        }

        if seviye == "error" {
            hatalar.push(kayit);
        } else {
            uyarilar.push(kayit);
        }
    }

    (hatalar, uyarilar)
}

/// Tek bir hatayı sezgisel olarak onarmaya çalışır.
pub fn hata_onar(hata: &DerlemeHatasi) -> Result<String, String> {
    if hata.dosya == "?" || hata.satir == 0 {
        return Err("Geçersiz hata konumu.".into());
    }

    let yol = cozumle_yol(&hata.dosya).unwrap_or_else(|_| proje_kok_dizini().join(&hata.dosya));
    if !yol.is_file() {
        return Err(format!("Dosya bulunamadı: {}", yol.display()));
    }

    let icerik = dosya_oku(&yol.to_string_lossy())?;
    let mut satirlar: Vec<String> = icerik.lines().map(String::from).collect();
    let idx = hata.satir.saturating_sub(1) as usize;
    if idx >= satirlar.len() {
        return Err("Satır aralık dışı.".into());
    }

    let mesaj_k = hata.mesaj.to_lowercase();
    let satir = &mut satirlar[idx];

    if mesaj_k.contains("expected `;`") || mesaj_k.contains("expected semicolon") {
        if !satir.trim_end().ends_with(';') && !satir.trim().is_empty() {
            satir.push(';');
            return kaydet_ve_bildir(&yol, &satirlar, &icerik, "noktalı virgül eklendi");
        }
    }

    if mesaj_k.contains("expected `}`") || mesaj_k.contains("unclosed delimiter") {
        satirlar.push("}".into());
        return kaydet_ve_bildir(&yol, &satirlar, &icerik, "kapanış süslü parantez eklendi");
    }

    if mesaj_k.contains("unused import") || mesaj_k.contains("is never used") {
        if satir.trim_start().starts_with("use ") {
            let yorum = format!("// TÜRK otomatik: {satir}");
            *satir = yorum;
            return kaydet_ve_bildir(&yol, &satirlar, &icerik, "kullanılmayan import yorumlandı");
        }
    }

    if mesaj_k.contains("cannot find") && mesaj_k.contains("main") {
        satirlar.push("fn main() {}".into());
        return kaydet_ve_bildir(&yol, &satirlar, &icerik, "fn main iskeleti eklendi");
    }

    Err(format!("Otomatik onarım uygulanamadı: {}", hata.mesaj))
}

fn kaydet_ve_bildir(
    yol: &Path,
    satirlar: &[String],
    orijinal: &str,
    islem: &str,
) -> Result<String, String> {
    let yeni = if orijinal.ends_with('\n') {
        format!("{}\n", satirlar.join("\n"))
    } else {
        satirlar.join("\n")
    };
    dosya_yaz(&yol.to_string_lossy(), &yeni)?;
    Ok(format!("▸ [KOD] {islem}: {}", yol.display()))
}

/// Otonom derleme + self-healing döngüsü.
pub fn otonom_derle_dongusu(proje_yolu: &str, uyarilari_temizle: bool) -> Result<String, String> {
    otonom_derle_dongusu_bagimlilar(proje_yolu, uyarilari_temizle, None)
}

/// Bağımlı dosyalar öncelikli otonom derleme döngüsü (refaktör hattı).
pub fn otonom_derle_dongusu_bagimlilar(
    proje_yolu: &str,
    uyarilari_temizle: bool,
    oncelikli_dosyalar: Option<&[String]>,
) -> Result<String, String> {
    let mut log = vec![format!(
        "▸ [DERLE] otonom cargo check başlatıldı ({})",
        proje_dizini(proje_yolu).display()
    )];

    if let Some(dosyalar) = oncelikli_dosyalar {
        if !dosyalar.is_empty() {
            log.push(format!(
                "▸ [DERLE] öncelikli bağımlı dosyalar: {}",
                dosyalar.join(", ")
            ));
        }
    }

    for deneme in 1..=MAX_ONARIM_DENEMESI {
        let sonuc = cargo_check_json(proje_yolu)?;

        if sonuc.basarili && (!uyarilari_temizle || sonuc.uyarilar.is_empty()) {
            log.push(format!(
                "✓ [DERLE] proje temiz — deneme {deneme}/{MAX_ONARIM_DENEMESI}"
            ));
            return Ok(log.join("\n"));
        }

        let hedefler: Vec<DerlemeHatasi> = if !sonuc.hatalar.is_empty() {
            hatalari_onceliklendir(sonuc.hatalar, oncelikli_dosyalar)
        } else if uyarilari_temizle {
            hatalari_onceliklendir(sonuc.uyarilar, oncelikli_dosyalar)
        } else {
            log.push(format!(
                "✓ [DERLE] hata yok ({} uyarı atlandı)",
                sonuc.uyarilar.len()
            ));
            return Ok(log.join("\n"));
        };

        log.push(format!(
            "▸ [DERLE] {} mesaj bulundu — deneme {deneme}/{MAX_ONARIM_DENEMESI}",
            hedefler.len()
        ));

        let mut onarildi = false;
        for hata in &hedefler {
            log.push(format!(
                "▸ [DERLE] {}:{} → {}",
                hata.dosya, hata.satir, hata.mesaj
            ));
            match hata_onar(hata) {
                Ok(bildirim) => {
                    log.push(bildirim);
                    onarildi = true;
                }
                Err(e) => log.push(format!("▸ [DERLE] onarım atlandı: {e}")),
            }
        }

        if !onarildi {
            log.push("✗ [DERLE] otomatik onarım mümkün değil — manuel müdahale gerekli.".into());
            break;
        }

        log.push("▸ [DERLE] onarım uygulandı, yeniden kontrol ediliyor...".into());
    }

    Ok(log.join("\n"))
}

fn hatalari_onceliklendir(
    hatalar: Vec<DerlemeHatasi>,
    oncelikli: Option<&[String]>,
) -> Vec<DerlemeHatasi> {
    let Some(oncelikli) = oncelikli else {
        return hatalar;
    };
    if oncelikli.is_empty() {
        return hatalar;
    }

    let mut once = Vec::new();
    let mut sonra = Vec::new();
    for hata in hatalar {
        let oncelikli_mi = oncelikli.iter().any(|p| dosya_eslesir(&hata.dosya, p));
        if oncelikli_mi {
            once.push(hata);
        } else {
            sonra.push(hata);
        }
    }
    once.extend(sonra);
    once
}

fn dosya_eslesir(hata_dosya: &str, oncelik: &str) -> bool {
    let h = hata_dosya.replace('\\', "/");
    let p = oncelik.replace('\\', "/");
    h.ends_with(&p) || p.ends_with(&h) || h.contains(&p) || p.contains(&h)
}

/// Araç çıktısı için özet rapor.
pub fn otonom_derle_rapor(proje_yolu: &str, uyarilari_temizle: bool) -> String {
    match otonom_derle_dongusu(proje_yolu, uyarilari_temizle) {
        Ok(rapor) => rapor,
        Err(e) => format!("✗ [DERLE] otonom derleme başarısız: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_ayikla_bos() {
        let (h, u) = json_mesajlari_ayikla("");
        assert!(h.is_empty());
        assert!(u.is_empty());
    }
}

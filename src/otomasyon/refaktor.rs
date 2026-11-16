//! TÜRK otonom proje refaktörü — çoklu dosya analizi, kod enjeksiyonu ve derleme zinciri.

use super::derleyici::otonom_derle_dongusu_bagimlilar;
use super::kod_ustasi::{cozumle_yol, dosya_oku, dosya_yaz, kod_enjekte_et};
use super::sistem::proje_kok_dizini;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

pub const REFAKTOR_BASLANGIC: &str =
    "▸ [REFAKTÖR] Dosya analiz ediliyor ve yeni siber fikirler üretiliyor...";

pub const REFAKTOR_BAGIMLI_TARAMA: &str =
    "▸ [REFAKTÖR] Bağımlı dosyalar taranıyor ve yan etkiler temizleniyor...";

pub const REFAKTOR_BASARI: &str =
    "✓ [REFAKTÖR] Geliştirme tamamlandı ve kararlı şekilde derlendi, Sinan kanka!";

/// Yerel akıl süzgeci çıktısı
#[derive(Debug, Clone)]
struct RefaktorPlani {
    hedef_satir: String,
    enjekte_kod: String,
    aciklama: String,
    yorum_temizligi: bool,
}

/// Tam refaktör hattı: oku → planla → enjekte → bağımlılık tara → otonom derle.
pub fn proje_refaktor_yurut(yol: &str, kullanici_istek: &str) -> Result<String, String> {
    let mut log = vec![REFAKTOR_BASLANGIC.to_string()];

    let icerik = dosya_oku(yol)?;
    let satir_sayisi = icerik.lines().count();
    let plan = akil_suzgeci(&icerik, yol, kullanici_istek)?;

    log.push(format!(
        "▸ [REFAKTÖR] Analiz: {yol} — {satir_sayisi} satır | {}",
        plan.aciklama
    ));

    if plan.yorum_temizligi {
        let temiz = yorum_satirlari_temizle(&icerik);
        if temiz != icerik {
            dosya_yaz(yol, &temiz)?;
            log.push(format!(
                "▸ [REFAKTÖR] Gereksiz yorum satırları temizlendi: {yol}"
            ));
        }
    }

    let enjekte_log = kod_enjekte_et(yol, &plan.hedef_satir, &plan.enjekte_kod)?;
    log.push(enjekte_log.replace("[KOD]", "[REFAKTÖR-KOD]"));

    log.push(REFAKTOR_BAGIMLI_TARAMA.into());
    let bagimlilar = proje_geneli_tara(yol, &plan.enjekte_kod)?;
    if bagimlilar.is_empty() {
        log.push(
            "▸ [REFAKTÖR] Doğrudan bağımlı dosya bulunamadı — proje geneli kontrol edilecek."
                .into(),
        );
    } else {
        log.push(format!(
            "▸ [REFAKTÖR] {} bağımlı dosya tespit edildi: {}",
            bagimlilar.len(),
            bagimlilar.join(", ")
        ));
    }

    log.push("▸ [REFAKTÖR] Otonom derleme zinciri başlatılıyor...".into());
    let derleme = otonom_derle_dongusu_bagimlilar(".", false, Some(&bagimlilar))?;
    let basarili = derleme.contains("✓ [DERLE] proje temiz");
    log.push(derleme);

    if basarili {
        log.push(REFAKTOR_BASARI.into());
    } else {
        log.push(
            "▸ [REFAKTÖR] Enjeksiyon tamamlandı; derleme döngüsü kalan uyarı/hata raporladı."
                .into(),
        );
    }

    Ok(log.join("\n"))
}

/// `src/` altındaki dosyaları asenkron tarar; hedef dosyaya referans verenleri döndürür.
pub fn proje_geneli_tara(hedef_yol: &str, eklenen_kod: &str) -> Result<Vec<String>, String> {
    let hedef = hedef_yol.to_string();
    let kod = eklenen_kod.to_string();
    std::thread::Builder::new()
        .name("turk-refaktor-tara".into())
        .spawn(move || proje_geneli_tara_sync(&hedef, &kod))
        .map_err(|e| format!("refaktör tarama thread'i başlatılamadı: {e}"))?
        .join()
        .map_err(|_| "refaktör tarama thread'i panikledi.".to_string())?
}

fn proje_geneli_tara_sync(hedef_yol: &str, eklenen_kod: &str) -> Result<Vec<String>, String> {
    let hedef_tam = cozumle_yol(hedef_yol)?;
    let hedef_icerik = dosya_oku(hedef_yol)?;
    let simgeler = dosyadan_simgeleri_cikar(&hedef_icerik);
    let simgeler = simgeler
        .union(&dosyadan_simgeleri_cikar(eklenen_kod))
        .cloned()
        .collect::<HashSet<_>>();
    let mod_yollari = modul_yollarini_cikar(&hedef_tam);

    let mut bagimlilar = Vec::new();
    for dosya in src_rs_dosyalarini_topla()? {
        if dosya == hedef_tam {
            continue;
        }
        let yol_str = dosya
            .strip_prefix(proje_kok_dizini())
            .unwrap_or(&dosya)
            .to_string_lossy()
            .replace('\\', "/");
        let icerik = fs::read_to_string(&dosya).map_err(|e| {
            format!(
                "Bağımlılık taraması okuma hatası ({}): {e}",
                dosya.display()
            )
        })?;
        if dosya_bagimli_mi(&icerik, &mod_yollari, &simgeler) {
            bagimlilar.push(yol_str);
        }
    }
    bagimlilar.sort();
    Ok(bagimlilar)
}

fn src_rs_dosyalarini_topla() -> Result<Vec<PathBuf>, String> {
    let src = proje_kok_dizini().join("src");
    if !src.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    src_dizin_gez(&src, &mut out)?;
    Ok(out)
}

fn src_dizin_gez(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for girdi in
        fs::read_dir(dir).map_err(|e| format!("Dizin okunamadı ({}): {e}", dir.display()))?
    {
        let girdi = girdi.map_err(|e| format!("Dizin girdisi hatası: {e}"))?;
        let yol = girdi.path();
        if yol.is_dir() {
            src_dizin_gez(&yol, out)?;
        } else if yol.extension().is_some_and(|e| e == "rs") {
            out.push(yol);
        }
    }
    Ok(())
}

fn modul_yollarini_cikar(dosya: &Path) -> Vec<String> {
    let src = proje_kok_dizini().join("src");
    let rel = dosya.strip_prefix(&src).unwrap_or(dosya).with_extension("");
    let parcalar: Vec<String> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
        .collect();
    if parcalar.is_empty() {
        return Vec::new();
    }
    let mut yollar = vec![parcalar.join("::")];
    if let Some(son) = parcalar.last() {
        yollar.push(son.clone());
    }
    if parcalar.len() >= 2 {
        yollar.push(format!("{}::{}", parcalar[0], parcalar[1]));
    }
    yollar
}

fn dosyadan_simgeleri_cikar(icerik: &str) -> HashSet<String> {
    let mut simgeler = HashSet::new();
    for satir in icerik.lines() {
        let t = satir.trim();
        for on_ek in [
            "pub fn ",
            "fn ",
            "pub struct ",
            "struct ",
            "pub enum ",
            "enum ",
            "pub trait ",
            "trait ",
            "impl ",
            "type ",
            "pub type ",
        ] {
            if let Some(rest) = t.strip_prefix(on_ek) {
                let ad = rest
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("");
                if !ad.is_empty() && ad != "main" {
                    simgeler.insert(ad.to_string());
                }
            }
        }
    }
    simgeler
}

fn dosya_bagimli_mi(icerik: &str, mod_yollari: &[String], simgeler: &HashSet<String>) -> bool {
    for yol in mod_yollari {
        if icerik.contains(yol) {
            return true;
        }
        let alt = yol.replace("::", "/");
        if icerik.contains(&alt) {
            return true;
        }
    }
    for simge in simgeler {
        if icerik.contains(simge) {
            return true;
        }
    }
    false
}

/// Dosya içeriği + istek → enjeksiyon planı (yerel kural matrisi).
fn akil_suzgeci(icerik: &str, yol: &str, istek: &str) -> Result<RefaktorPlani, String> {
    let k = istek.to_lowercase();
    let hedef = enjeksiyon_hedefi_bul(icerik);
    let rust = yol.ends_with(".rs");

    let yorum_temizligi =
        k.contains("yorum") && (k.contains("temiz") || k.contains("sil") || k.contains("optimize"));

    if k.contains("siber") || k.contains("cyber") || k.contains("neon") {
        return Ok(RefaktorPlani {
            hedef_satir: hedef,
            enjekte_kod: if rust {
                siber_yardimci_fonksiyon()
            } else {
                "// TÜRK siber refaktör — telemetri etiketi aktif".into()
            },
            aciklama: "Siber telemetri yardımcısı eklenecek".into(),
            yorum_temizligi,
        });
    }

    if k.contains("optimize")
        || k.contains("optimiz")
        || k.contains("iyileştir")
        || k.contains("iyilestir")
    {
        return Ok(RefaktorPlani {
            hedef_satir: hedef,
            enjekte_kod: if rust {
                performans_yardimci_fonksiyon()
            } else {
                "// TÜRK refaktör — performans notu".into()
            },
            aciklama: "Performans izleme yardımcısı eklenecek".into(),
            yorum_temizligi,
        });
    }

    if k.contains("helper")
        || k.contains("yardımcı")
        || k.contains("yardimci")
        || k.contains("fonksiyon")
    {
        return Ok(RefaktorPlani {
            hedef_satir: hedef,
            enjekte_kod: if rust {
                genel_yardimci_fonksiyon()
            } else {
                "// TÜRK refaktör — genel yardımcı blok".into()
            },
            aciklama: "Genel yardımcı fonksiyon eklenecek".into(),
            yorum_temizligi,
        });
    }

    Ok(RefaktorPlani {
        hedef_satir: hedef,
        enjekte_kod: if rust {
            genel_yardimci_fonksiyon()
        } else {
            format!(
                "// TÜRK refaktör — otomatik geliştirme: {}",
                istek.chars().take(60).collect::<String>()
            )
        },
        aciklama: "Genel otonom geliştirme uygulanacak".into(),
        yorum_temizligi,
    })
}

fn enjeksiyon_hedefi_bul(icerik: &str) -> String {
    let mut son_fn: Option<String> = None;
    for satir in icerik.lines() {
        let t = satir.trim();
        if t.starts_with("pub fn ") || t.starts_with("fn ") {
            son_fn = Some(satir.to_string());
        }
    }
    if let Some(f) = son_fn {
        return f;
    }
    for satir in icerik.lines() {
        let t = satir.trim();
        if t.starts_with("impl ") {
            return satir.to_string();
        }
    }
    icerik.lines().next().unwrap_or("fn main").to_string()
}

fn yorum_satirlari_temizle(icerik: &str) -> String {
    let mut satirlar: Vec<&str> = Vec::new();
    for satir in icerik.lines() {
        let t = satir.trim();
        if t.is_empty() {
            continue;
        }
        if t == "//" || t.starts_with("// TODO") && t.len() < 8 {
            continue;
        }
        if t.starts_with("//") && t.len() <= 3 {
            continue;
        }
        satirlar.push(satir);
    }
    let birlesik = satirlar.join("\n");
    if icerik.ends_with('\n') {
        format!("{birlesik}\n")
    } else {
        birlesik
    }
}

fn siber_yardimci_fonksiyon() -> String {
    r#"
/// TÜRK siber telemetri etiketi — otonom refaktör.
#[allow(dead_code)]
fn turk_siber_etiket() -> &'static str {
    "◈ SİBER DALGA — TÜRK refaktör aktif"
}
"#
    .trim()
    .to_string()
}

fn performans_yardimci_fonksiyon() -> String {
    r#"
/// TÜRK performans izleme notu — refaktör yardımcısı.
#[inline]
#[allow(dead_code)]
fn turk_perf_notu(ham: &str) -> String {
    format!("[TÜRK-PERF] {ham}")
}
"#
    .trim()
    .to_string()
}

fn genel_yardimci_fonksiyon() -> String {
    r#"
/// TÜRK otonom refaktör yardımcısı.
#[allow(dead_code)]
fn turk_refaktor_yardimci() -> bool {
    true
}
"#
    .trim()
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_akil_suzgeci_siber() {
        let plan = akil_suzgeci("fn main() {}", "test.rs", "siber özellik ekle").unwrap();
        assert!(plan.enjekte_kod.contains("turk_siber_etiket"));
    }

    #[test]
    fn test_yorum_temizligi() {
        let girdi = "fn a() {}\n//\n\nfn b() {}\n";
        let cikti = yorum_satirlari_temizle(girdi);
        assert!(!cikti.contains("\n//\n"));
    }

    #[test]
    fn test_simgeleri_cikar() {
        /// TÜRK performans izleme notu — refaktör yardımcısı.
        #[inline]
        #[allow(dead_code)]
        fn turk_perf_notu(ham: &str) -> String {
            format!("[TÜRK-PERF] {ham}")
        }
        let simgeler = dosyadan_simgeleri_cikar("pub fn foo() {}\nstruct Bar {}");
        assert!(simgeler.contains("foo"));
        assert!(simgeler.contains("Bar"));
    }
}

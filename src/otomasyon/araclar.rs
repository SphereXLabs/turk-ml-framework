//! TÜRK araç (tool calling) kaydı — JSON ve doğal dil → sistem eylemleri.

use super::derleyici::otonom_derle_rapor;
use super::kod_ustasi::{self, kod_yaz_uygula};
use super::notlar::hafiza_tara_ozet;
use super::refaktor::proje_refaktor_yurut;
use super::sistem::{
    dosya_olustur, dosya_yaz, dosyalari_listele, proje_derle, proje_kontrol, sistem_durumu,
    terminal_komutu_calistir,
};
use super::web::{canli_hava_durumu_cek, hava_sorgusu_mu, sehir_ayikla};
use crate::error::Result;
use serde::Deserialize;

/// LLM'den beklenen JSON araç çağrısı
#[derive(Debug, Clone, Deserialize)]
pub struct AracCagrisiJson {
    pub arac: String,
    #[serde(default)]
    pub parametre: Option<String>,
    /// `dosya_yaz` / `dosya_olustur` / `kod_yaz` için içerik
    #[serde(default)]
    pub parametre2: Option<String>,
    /// `kod_yaz` enjeksiyon hedefi (fonksiyon/satır)
    #[serde(default)]
    pub enjekte_hedef: Option<String>,
}

/// LLM veya kural motorunun çağırabileceği araçlar
#[derive(Debug, Clone)]
pub enum AracKomut {
    DosyaListele {
        yol: String,
    },
    TerminalKomut {
        komut: String,
    },
    SistemDurumu,
    ProjeDerle {
        yol: String,
    },
    ProjeKontrol {
        yol: String,
    },
    DosyaYaz {
        yol: String,
        icerik: String,
    },
    DosyaOlustur {
        yol: String,
        icerik: String,
    },
    HafizaTara {
        anahtar_kelime: String,
    },
    WebArama {
        sorgu: String,
    },
    KodYaz {
        yol: String,
        icerik: String,
        enjekte_hedef: Option<String>,
    },
    OtonomDerle {
        yol: String,
        uyarilari_temizle: bool,
    },
    ProjeRefaktor {
        yol: String,
        istek: String,
    },
}

impl AracKomut {
    pub fn etiket(&self) -> &'static str {
        match self {
            Self::DosyaListele { .. } => "dosya_listele",
            Self::TerminalKomut { .. } => "terminal_komutu_calistir",
            Self::SistemDurumu => "sistem_durumu",
            Self::ProjeDerle { .. } => "proje_derle",
            Self::ProjeKontrol { .. } => "proje_kontrol",
            Self::DosyaYaz { .. } => "dosya_yaz",
            Self::DosyaOlustur { .. } => "dosya_olustur",
            Self::HafizaTara { .. } => "hafiza_tara",
            Self::WebArama { .. } => "web_arama",
            Self::KodYaz { .. } => "kod_yaz",
            Self::OtonomDerle { .. } => "otonom_derle",
            Self::ProjeRefaktor { .. } => "proje_refaktor",
        }
    }

    pub fn json_den(cagri: &AracCagrisiJson) -> Option<Self> {
        let ad = cagri.arac.to_lowercase().replace('-', "_");
        match ad.as_str() {
            "dosya_listele" | "dosyalari_listele" | "dosya_list" => Some(Self::DosyaListele {
                yol: cagri.parametre.clone().unwrap_or_else(|| ".".into()),
            }),
            "terminal_komutu_calistir" | "terminal_komut" | "surec_baslat" | "komut_calistir" => {
                cagri
                    .parametre
                    .as_ref()
                    .map(|k| Self::TerminalKomut { komut: k.clone() })
            }
            "sistem_durumu" | "sistem" => Some(Self::SistemDurumu),
            "proje_derle" | "cargo_derle" | "derle" => Some(Self::ProjeDerle {
                yol: cagri.parametre.clone().unwrap_or_else(|| ".".into()),
            }),
            "proje_kontrol" | "cargo_check" | "kontrol" => Some(Self::ProjeKontrol {
                yol: cagri.parametre.clone().unwrap_or_else(|| ".".into()),
            }),
            "dosya_yaz" | "dosya_kaydet" => {
                let yol = cagri.parametre.clone()?;
                let icerik = cagri.parametre2.clone().unwrap_or_default();
                Some(Self::DosyaYaz { yol, icerik })
            }
            "dosya_olustur" | "not_al" => {
                let yol = cagri.parametre.clone().unwrap_or_default();
                let icerik = cagri.parametre2.clone().unwrap_or_default();
                Some(Self::DosyaOlustur { yol, icerik })
            }
            "hafiza_tara" | "hafiza_ara" | "not_ara" => {
                let anahtar = cagri.parametre.clone().unwrap_or_default();
                Some(Self::HafizaTara {
                    anahtar_kelime: anahtar,
                })
            }
            "web_arama" | "web_sorgu" | "hava_durumu" => {
                let sorgu = cagri.parametre.clone().unwrap_or_default();
                Some(Self::WebArama { sorgu })
            }
            "kod_yaz" | "kod_yazma" | "kod_ustasi" => {
                let yol = cagri.parametre.clone().unwrap_or_default();
                let icerik = cagri
                    .parametre2
                    .clone()
                    .unwrap_or_else(|| kod_ustasi::MERHABA_DUNYA_RS.to_string());
                Some(Self::KodYaz {
                    yol,
                    icerik,
                    enjekte_hedef: cagri.enjekte_hedef.clone(),
                })
            }
            "otonom_derle" | "otonom_derleme" | "self_healing" => Some(Self::OtonomDerle {
                yol: cagri.parametre.clone().unwrap_or_else(|| ".".into()),
                uyarilari_temizle: cagri
                    .parametre2
                    .as_deref()
                    .is_some_and(|p| p == "uyarilar" || p.contains("uyari")),
            }),
            "proje_refaktor" | "proje_refaktör" | "refaktor" | "refaktör" => {
                let yol = cagri.parametre.clone().unwrap_or_default();
                let istek = cagri
                    .parametre2
                    .clone()
                    .unwrap_or_else(|| "dosyayı optimize et".into());
                Some(Self::ProjeRefaktor { yol, istek })
            }
            _ => None,
        }
    }
}

/// Aracı çalıştır ve ham sonuç döndür
pub fn arac_calistir(arac: &AracKomut) -> Result<String> {
    Ok(match arac {
        AracKomut::DosyaListele { yol } => {
            let liste = dosyalari_listele(yol);
            if liste.is_empty() {
                format!("{yol} — boş dizin.")
            } else if liste.len() == 1 && liste[0].starts_with("Hata:") {
                liste[0].clone()
            } else {
                format!("{yol} — {} öğe:\n{}", liste.len(), liste.join("\n"))
            }
        }
        AracKomut::TerminalKomut { komut } => terminal_komutu_calistir(komut),
        AracKomut::SistemDurumu => {
            let d = sistem_durumu();
            let cpu = d.cpu_yuzde.round();
            format!(
                "Sistem: {} | CPU: {:.0}% | RAM: {:.1}/{:.1} GB ({:.0}%){}",
                d.os_ad,
                cpu,
                d.ram_kullanilan_gb,
                d.ram_toplam_gb,
                d.ram_yuzde,
                d.disk_yuzde
                    .map(|p| format!(" | Disk (dolu): {p:.0}%"))
                    .unwrap_or_default()
            )
        }
        AracKomut::ProjeDerle { yol } => proje_derle(yol),
        AracKomut::ProjeKontrol { yol } => proje_kontrol(yol),
        AracKomut::DosyaYaz { yol, icerik } => dosya_yaz(yol, icerik),
        AracKomut::DosyaOlustur { yol, icerik } => dosya_olustur(yol, icerik),
        AracKomut::HafizaTara { anahtar_kelime } => hafiza_tara_ozet(anahtar_kelime),
        AracKomut::WebArama { sorgu } => {
            if hava_sorgusu_mu(sorgu) {
                return canli_hava_durumu_cek(&sehir_ayikla(sorgu))
                    .map_err(|e| crate::error::TurkError::RuntimeError(e));
            }
            "Web araması henüz yalnızca hava durumu için aktif. \
             Örnek: «izmirde hava durumu kaç»"
                .into()
        }
        AracKomut::KodYaz {
            yol,
            icerik,
            enjekte_hedef,
        } => kod_yaz_uygula(yol, icerik, enjekte_hedef.as_deref())
            .map_err(|e| crate::error::TurkError::RuntimeError(e))?,
        AracKomut::OtonomDerle {
            yol,
            uyarilari_temizle,
        } => otonom_derle_rapor(yol, *uyarilari_temizle),
        AracKomut::ProjeRefaktor { yol, istek } => proje_refaktor_yurut(yol, istek)
            .map_err(|e| crate::error::TurkError::RuntimeError(e))?,
    })
}

pub fn json_icerik_ayikla(metin: &str) -> Option<AracKomut> {
    let trimmed = metin.trim();
    if let Ok(cagri) = serde_json::from_str::<AracCagrisiJson>(trimmed) {
        return AracKomut::json_den(&cagri);
    }
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if start < end {
            let blok = &trimmed[start..=end];
            if let Ok(cagri) = serde_json::from_str::<AracCagrisiJson>(blok) {
                return AracKomut::json_den(&cagri);
            }
        }
    }
    None
}

pub fn json_veya_metin_arac(metin: &str) -> Option<AracKomut> {
    json_icerik_ayikla(metin).or_else(|| metinden_arac_ayikla(metin))
}

/// Girdide belirgin bir araç komutu niyeti var mı? (gevşek alt-dize eşleşmesi yok)
pub fn metin_arac_niyeti(metin: &str) -> bool {
    if json_icerik_ayikla(metin).is_some() {
        return true;
    }
    let k = metin.to_lowercase();
    if kod_yaz_niyeti_mi(&k) || otonom_derle_niyeti_mi(&k) || refaktor_niyeti_mi(&k) {
        return true;
    }
    const TETIKLER: &[&str] = &[
        "sistem durumu",
        "sistem durum",
        "sistem bilgi",
        "cpu kullan",
        "ram kullan",
        "ram durum",
        "not al",
        "şunu kaydet",
        "sunu kaydet",
        "bunu kaydet",
        "dosya listele",
        "dosyalari listele",
        "dosya yaz",
        "dosya kaydet",
        "dosya oluştur",
        "dosya olustur",
        "cargo check",
        "cargo build",
        "cargo derle",
        "proje kontrol",
        "projeyi kontrol",
        "proje derle",
        "projeyi derle",
        "hata var mı",
        "hata var mi",
        "hata varmi",
        "çalıştır:",
        "calistir:",
        "başlat:",
        "baslat:",
        "komut:",
        "notumu bul",
        "notu bul",
        "notumu getir",
        "geçen haftaki not",
        "gecen haftaki not",
        "hafıza",
        "hafiza",
        "hafiza_tara",
        "hava durumu",
        "hava nasıl",
        "hava nasil",
        "kaç derece",
        "kac derece",
        "kod_yaz",
        "kod yaz",
        "kod_ustasi",
        "otonom derle",
        "otonom_derle",
        "kendi kendini onar",
        "uyarıları temizle",
        "uyarilari temizle",
        "proje_refaktor",
        "proje refaktor",
        "refaktör",
        "refaktor",
        "optimize et",
        "siber özellik",
        "siber ozellik",
    ];
    TETIKLER.iter().any(|t| k.contains(t))
}

pub fn metinden_arac_ayikla(metin: &str) -> Option<AracKomut> {
    if !metin_arac_niyeti(metin) {
        return None;
    }
    let k = metin.to_lowercase();

    if refaktor_niyeti_mi(&k) {
        if let Some(arac) = refaktor_ayikla(metin) {
            return Some(arac);
        }
    }

    if kod_yaz_niyeti_mi(&k) {
        if let Some(arac) = kod_yaz_ayikla(metin) {
            return Some(arac);
        }
    }

    if otonom_derle_niyeti_mi(&k) {
        let yol = yol_ayikla(metin).unwrap_or_else(|| ".".into());
        let uyarilari_temizle = k.contains("uyar") && k.contains("temizle");
        return Some(AracKomut::OtonomDerle {
            yol,
            uyarilari_temizle,
        });
    }

    if k.contains("sistem durum")
        || k.contains("sistem bilgi")
        || k.contains("cpu kullan")
        || k.contains("ram kullan")
        || k.contains("ram durum")
    {
        return Some(AracKomut::SistemDurumu);
    }

    if k.contains("not al")
        || k.contains("şunu kaydet")
        || k.contains("sunu kaydet")
        || k.contains("bunu kaydet")
        || ((k.contains("dosya") || k.contains("not"))
            && (k.contains("yaz")
                || k.contains("kaydet")
                || k.contains("olustur")
                || k.contains("oluştur")))
    {
        if let Some((yol, icerik)) = dosya_olustur_ayikla(metin) {
            return Some(AracKomut::DosyaOlustur { yol, icerik });
        }
    }

    if k.contains("dosya")
        && (k.contains("listele") || k.contains("göster") || k.contains("goster"))
    {
        let yol = yol_ayikla(metin).unwrap_or_else(|| ".".into());
        return Some(AracKomut::DosyaListele { yol });
    }

    if k.contains("cargo check")
        || (k.contains("projeyi") && k.contains("kontrol"))
        || (k.contains("proje") && k.contains("kontrol"))
        || (k.contains("hata")
            && (k.contains("var mı") || k.contains("varmi") || k.contains("var mi")))
    {
        let yol = yol_ayikla(metin).unwrap_or_else(|| ".".into());
        return Some(AracKomut::ProjeKontrol { yol });
    }

    if (k.contains("proje") || k.contains("cargo"))
        && (k.contains("derle") || k.contains("build"))
        && !k.contains("kontrol")
    {
        let yol = yol_ayikla(metin).unwrap_or_else(|| ".".into());
        return Some(AracKomut::ProjeDerle { yol });
    }

    if k.contains("cargo") && (k.contains("build") || k.contains("derle")) {
        return Some(AracKomut::TerminalKomut {
            komut: "cargo build".into(),
        });
    }

    if k.contains("çalıştır")
        || k.contains("calistir")
        || k.contains("başlat")
        || k.contains("baslat")
    {
        if let Some(komut) = komut_ayikla(metin) {
            return Some(AracKomut::TerminalKomut { komut });
        }
    }

    if hafiza_niyeti_mi(&k) {
        let anahtar = hafiza_anahtar_ayikla(metin);
        if !anahtar.is_empty() {
            return Some(AracKomut::HafizaTara {
                anahtar_kelime: anahtar,
            });
        }
    }

    if hava_sorgusu_mu(metin) {
        return Some(AracKomut::WebArama {
            sorgu: metin.to_string(),
        });
    }

    None
}

fn refaktor_niyeti_mi(k: &str) -> bool {
    k.contains("proje_refaktor")
        || k.contains("proje refaktor")
        || k.contains("refaktör")
        || k.contains("refaktor")
        || (k.contains("incele")
            && (k.contains("optimize")
                || k.contains("optimiz")
                || k.contains("siber")
                || k.contains("geliştir")
                || k.contains("gelistir")
                || k.contains("iyileştir")
                || k.contains("iyilestir")))
        || (k.contains("dosya") && k.contains("optimize"))
        || (k.contains("dosyasını") && k.contains("optimize"))
        || (k.contains("dosyasini") && k.contains("optimize"))
        || (k.contains("siber")
            && (k.contains("ekle") || k.contains("özellik") || k.contains("ozellik")))
        || (k.contains("dosya") && k.contains("siber"))
}

fn refaktor_ayikla(metin: &str) -> Option<AracKomut> {
    let yol = kod_ustasi::rs_yol_ayikla(metin).or_else(|| {
        yol_ayikla(metin).filter(|y| y.contains('.') || y.contains('/') || y.contains('\\'))
    })?;

    Some(AracKomut::ProjeRefaktor {
        yol,
        istek: metin.to_string(),
    })
}

fn kod_yaz_niyeti_mi(k: &str) -> bool {
    k.contains("kod_yaz")
        || k.contains("kod yaz")
        || (k.contains("kod_ustasi") && (k.contains("yaz") || k.contains("aç") || k.contains("ac")))
        || (k.contains(".rs")
            && (k.contains("aç")
                || k.contains("ac")
                || k.contains("oluştur")
                || k.contains("olustur")
                || k.contains("yaz")))
        || (k.contains("rust") && k.contains("dosya"))
        || (k.contains("merhaba")
            && (k.contains("dünya") || k.contains("dunya"))
            && (k.contains(".rs") || k.contains("rust")))
}

fn otonom_derle_niyeti_mi(k: &str) -> bool {
    k.contains("otonom derle")
        || k.contains("otonom_derle")
        || k.contains("kendi kendini onar")
        || k.contains("kendi kendine onar")
        || (k.contains("uyar") && k.contains("temizle") && k.contains("kod"))
        || (k.contains("kod_ustasi") && k.contains("temizle"))
}

fn kod_yaz_ayikla(metin: &str) -> Option<AracKomut> {
    let k = metin.to_lowercase();
    let mut yol = kod_ustasi::rs_yol_ayikla(metin)
        .or_else(|| yol_ayikla(metin).filter(|y| y.ends_with(".rs") || y.contains('/')))?;

    if !yol.contains('/') && (k.contains("src/bin") || k.contains("bin/") || k.contains("bin ")) {
        yol = format!("src/bin/{yol}");
    }
    if !yol.ends_with(".rs") {
        yol.push_str(".rs");
    }

    let icerik = tirnak_icerik(metin)
        .or_else(|| kod_ustasi::varsayilan_rust_icerik(metin))
        .unwrap_or_else(|| kod_ustasi::MERHABA_DUNYA_RS.to_string());

    Some(AracKomut::KodYaz {
        yol,
        icerik,
        enjekte_hedef: None,
    })
}

fn hafiza_niyeti_mi(k: &str) -> bool {
    k.contains("notumu bul")
        || k.contains("notu bul")
        || k.contains("notumu getir")
        || k.contains("notu getir")
        || k.contains("geçen haftaki not")
        || k.contains("gecen haftaki not")
        || k.contains("hafıza")
        || k.contains("hafiza")
        || k.contains("hafiza_tara")
        || (k.contains("dosya") && k.contains("getir") && k.contains("not"))
        || k.contains("selam verdiğim")
        || k.contains("selam verdigi")
}

fn hafiza_anahtar_ayikla(metin: &str) -> String {
    if let Some(icerik) = tirnak_icerik(metin) {
        return icerik;
    }

    let k = metin.to_lowercase();

    if k.contains("selam verdiğim") || k.contains("selam verdigi") {
        return "selam".into();
    }

    for tetik in [
        "notumu bul",
        "notu bul",
        "notumu getir",
        "notu getir",
        "geçen haftaki not",
        "gecen haftaki not",
        "dosyayı getir",
        "dosyayi getir",
        "hafızada ara",
        "hafizada ara",
        "hafızada",
        "hafizada",
    ] {
        if let Some(pos) = k.find(tetik) {
            let kalan = metin[pos + tetik.len()..].trim();
            let temiz = kalan
                .trim_matches(|c: char| c == '?' || c == '.' || c == '!' || c == ',')
                .trim();
            if !temiz.is_empty() {
                return temiz.into();
            }
        }
    }

    // Son anlamlı kelime grubu
    let kelimeler: Vec<&str> = metin.split_whitespace().collect();
    if kelimeler.len() >= 2 {
        return kelimeler[kelimeler.len() - 2..].join(" ");
    }

    metin.trim().into()
}

fn tirnak_icerik(metin: &str) -> Option<String> {
    if let Some(start) = metin.find('"') {
        if let Some(end) = metin[start + 1..].find('"') {
            return Some(metin[start + 1..start + 1 + end].to_string());
        }
    }
    if let Some(start) = metin.find('\'') {
        if let Some(end) = metin[start + 1..].find('\'') {
            return Some(metin[start + 1..start + 1 + end].to_string());
        }
    }
    None
}

fn dosya_olustur_ayikla(metin: &str) -> Option<(String, String)> {
    if let Some(icerik) = tirnak_icerik(metin) {
        let temiz = metin
            .replace(&format!("\"{icerik}\""), " ")
            .replace(&format!("'{icerik}'"), " ");
        if let Some(yol) = yol_ayikla(&temiz) {
            return Some((yol, icerik));
        }
        return Some((String::new(), icerik));
    }

    for ayirici in [": ", " -> ", " => ", " | "] {
        if let Some(pos) = metin.find(ayirici) {
            let (sol, sag) = metin.split_at(pos);
            let kelimeler: Vec<&str> = sol.split_whitespace().collect();
            if let Some(aday) = kelimeler.last() {
                if aday.contains('.') || aday.contains('/') || aday.contains('\\') {
                    let icerik = sag[ayirici.len()..].trim();
                    if !icerik.is_empty() {
                        return Some((aday.to_string(), icerik.to_string()));
                    }
                }
            }
        }
    }

    let k = metin.to_lowercase();
    for tetik in [
        "not al",
        "şunu kaydet",
        "sunu kaydet",
        "bunu kaydet",
        "dosya yaz",
        "dosya kaydet",
        "dosya oluştur",
        "dosya olustur",
    ] {
        if let Some(pos) = k.find(tetik) {
            let icerik = metin[pos + tetik.len()..].trim();
            if !icerik.is_empty() {
                if let Some(yol) = yol_ayikla(icerik) {
                    let metin_kalan = icerik
                        .split_whitespace()
                        .skip_while(|p| *p == yol.as_str() || yol.contains(p))
                        .collect::<Vec<_>>()
                        .join(" ");
                    if metin_kalan.is_empty() {
                        return Some((yol, String::new()));
                    }
                    return Some((yol, metin_kalan));
                }
                return Some((String::new(), icerik.to_string()));
            }
        }
    }

    None
}

fn yol_ayikla(metin: &str) -> Option<String> {
    if let Some(start) = metin.find('"') {
        if let Some(end) = metin[start + 1..].find('"') {
            return Some(metin[start + 1..start + 1 + end].to_string());
        }
    }
    for parca in metin.split_whitespace() {
        if parca.contains(':') || parca.starts_with('/') || parca.starts_with('.') {
            return Some(parca.trim_matches(|c| c == ',' || c == '.').to_string());
        }
        if parca.ends_with(".txt") || parca.ends_with(".rs") {
            return Some(parca.to_string());
        }
    }
    None
}

fn komut_ayikla(metin: &str) -> Option<String> {
    let tetikleyiciler = [
        "çalıştır:",
        "calistir:",
        "başlat:",
        "baslat:",
        "komut:",
        "run:",
    ];
    let k = metin.to_lowercase();
    for t in tetikleyiciler {
        if let Some(pos) = k.find(t) {
            let geri_kalan = metin[pos + t.len()..].trim();
            if !geri_kalan.is_empty() {
                return Some(geri_kalan.to_string());
            }
        }
    }
    None
}

pub fn yardim_metni() -> String {
    "\
TÜRK Komutları (yerel, API yok):
• sistem durumu — CPU, RAM, disk
• dosya listele . — dizin içeriği
• not al <metin> — ~/.turk/sistem/notlar/ altına .txt kaydeder
• dosya yaz not.rs: fn main() {} — .txt/.rs dosya oluşturur
• proje kontrol / cargo check / hata var mı — JSON ile hata satırlarını raporlar
• proje derle — cargo check (özet)
• JSON: {\"arac\":\"dosya_olustur\",\"parametre\":\"not.txt\",\"parametre2\":\"merhaba\"}
• JSON: {\"arac\":\"proje_kontrol\",\"parametre\":\".\"}
• geçen haftaki notumu bul selam — hafıza taraması
• izmirde hava durumu kaç — canlı hava (web_arama)
• kod_yaz / src/bin/deneme.rs merhaba dünya — Rust dosyası oluştur
• otonom derle / uyarıları kod_ustasi ile temizle — self-healing cargo check
• TÜRK, src/foo.rs incele ve optimize et — proje_refaktor (analiz + enjekte + derle)
• Merhaba / yardım — sohbet"
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_arac() {
        let json = r#"{"arac":"dosya_listele","parametre":"."}"#;
        let arac = json_icerik_ayikla(json).unwrap();
        assert!(matches!(arac, AracKomut::DosyaListele { .. }));
    }

    #[test]
    fn test_proje_derle_arac() {
        assert!(matches!(
            metinden_arac_ayikla("projeyi derle"),
            Some(AracKomut::ProjeDerle { .. })
        ));
    }

    #[test]
    fn test_hata_var_mi_arac() {
        assert!(matches!(
            metinden_arac_ayikla("hata var mı"),
            Some(AracKomut::ProjeKontrol { .. })
        ));
    }

    #[test]
    fn test_proje_kontrol_arac() {
        assert!(matches!(
            metinden_arac_ayikla("projeyi kontrol et"),
            Some(AracKomut::ProjeKontrol { .. })
        ));
        assert!(matches!(
            metinden_arac_ayikla("cargo check"),
            Some(AracKomut::ProjeKontrol { .. })
        ));
    }

    #[test]
    fn test_dosya_olustur_arac() {
        let arac = metinden_arac_ayikla("not al TÜRK test notu").unwrap();
        assert!(matches!(arac, AracKomut::DosyaOlustur { .. }));
        if let AracKomut::DosyaOlustur { yol, icerik } = arac {
            assert!(yol.is_empty());
            assert!(icerik.contains("TÜRK"));
        }
    }

    #[test]
    fn test_arac_sistem() {
        assert!(matches!(
            metinden_arac_ayikla("sistem durumunu göster"),
            Some(AracKomut::SistemDurumu)
        ));
    }

    #[test]
    fn test_proje_refaktor_arac() {
        let metin = "TÜRK, src/bin/turk_sistem.rs dosyasını incele ve optimize et";
        let arac = metinden_arac_ayikla(metin).unwrap();
        assert!(matches!(arac, AracKomut::ProjeRefaktor { .. }));
        if let AracKomut::ProjeRefaktor { yol, .. } = arac {
            assert!(yol.contains("turk_sistem.rs"));
        }
    }

    #[test]
    fn test_siber_refaktor_arac() {
        assert!(refaktor_niyeti_mi(
            "src/lib.rs dosyasına yeni siber özellik ekle"
        ));
    }

    #[test]
    fn test_kod_yaz_arac() {
        let metin = "TÜRK, src/bin/deneme.rs adında yeni bir Rust dosyası aç ve içine merhaba dünya kodu yaz";
        let arac = metinden_arac_ayikla(metin).unwrap();
        assert!(matches!(arac, AracKomut::KodYaz { .. }));
        if let AracKomut::KodYaz { yol, icerik, .. } = arac {
            assert!(yol.contains("deneme.rs"));
            assert!(icerik.contains("Merhaba dünya") || icerik.contains("println"));
        }
    }

    #[test]
    fn test_otonom_derle_arac() {
        assert!(matches!(
            metinden_arac_ayikla("Projedeki uyarıları kontrol et ve kod_ustasi ile temizle"),
            Some(AracKomut::OtonomDerle { .. })
        ));
    }

    #[test]
    fn test_sohbet_sistem_niyeti_yok() {
        assert!(!metin_arac_niyeti("Ne arıyorsunuz? Tutun elinden götürün"));
        assert!(metinden_arac_ayikla("Ne arıyorsunuz? Tutun elinden götürün").is_none());
    }
}

//! Jarvis v3 — çok adımlı otonom araç zincirleme (Chain of Thought).

use crate::error::Result;
use crate::otomasyon::araclar::{self, AracKomut};
use crate::otomasyon::kod_ustasi;
use crate::otomasyon::vekil::os_seviyesi_kontrol;

/// Tek zincir adımı
#[derive(Debug, Clone)]
pub struct ZincirAdimi {
    pub etiket: String,
    pub aciklama: String,
    pub yol: Option<String>,
    pub arac: Option<AracKomut>,
    pub vekil_temizlik: bool,
}

/// Zincir yürütme sonucu
#[derive(Debug, Clone)]
pub struct ZincirSonuc {
    pub basarili: bool,
    pub adim_loglari: Vec<String>,
    pub ham_ciktilar: Vec<String>,
    pub ozet: String,
}

/// Çok adımlı komut mu?
pub fn cok_adimli_mi(metin: &str) -> bool {
    let k = metin.to_lowercase();
    const BAGLACLAR: &[&str] = &[
        "ve ardından",
        "ve ardindan",
        " ardından ",
        " ardindan ",
        "ve sonra",
        " sonrasında",
        " sonrasinda ",
        " ardından,",
        " ardindan,",
    ];
    if BAGLACLAR.iter().any(|b| k.contains(b)) {
        return true;
    }
    let mut n = 0_u8;
    if k.contains("temizle") || k.contains("optimize") || k.contains("vekil") {
        n += 1;
    }
    if k.contains("refaktör") || k.contains("refaktor") || k.contains("siber") {
        n += 1;
    }
    if k.contains("derle") || k.contains("cargo") || k.contains("kararlı") || k.contains("kararli")
    {
        n += 1;
    }
    if k.contains("kontrol") || k.contains("sistem durumu") || k.contains("sistem kontrol") {
        n += 1;
    }
    if k.contains("dosya") && (k.contains("oku") || k.contains("refaktör") || k.contains("refaktor"))
    {
        n += 1;
    }
    n >= 2 && (k.contains(" ve ") || k.contains("sonra") || k.contains("ardın"))
}

fn dosya_yolu_cikar(metin: &str) -> Option<String> {
    for token in metin.split_whitespace() {
        let t = token.trim_matches(|c: char| ",.;:\"'()[]".contains(c));
        if t.contains('/') || t.contains('\\') || t.ends_with(".rs") || t.ends_with(".toml") {
            return Some(t.to_string());
        }
    }
    None
}

fn refaktor_istegi(metin: &str) -> String {
    let k = metin.to_lowercase();
    if k.contains("siber") {
        "siber temaya göre refaktör et".into()
    } else if k.contains("optimize") {
        "optimize et".into()
    } else {
        "refaktör uygula".into()
    }
}

/// Kullanıcı girdisinden otonom plan üret.
pub fn plan_olustur(metin: &str) -> Option<Vec<ZincirAdimi>> {
    if !cok_adimli_mi(metin) {
        return None;
    }
    let k = metin.to_lowercase();
    let yol = dosya_yolu_cikar(metin).unwrap_or_else(|| "src/bin/turk_sistem.rs".into());
    let proje_yolu = ".".to_string();

    let mut plan = Vec::new();

    if k.contains("kontrol") || k.contains("sistem") || k.contains("durum") {
        plan.push(ZincirAdimi {
            etiket: "sistem_kontrol".into(),
            aciklama: "Sistem durumu kontrol ediliyor".into(),
            yol: None,
            arac: Some(AracKomut::SistemDurumu),
            vekil_temizlik: false,
        });
    }

    if k.contains("temizle") || k.contains("optimize") || k.contains("vekil") {
        plan.push(ZincirAdimi {
            etiket: "vekil_temizle".into(),
            aciklama: "Sistem temizliği (vekil) tetikleniyor".into(),
            yol: None,
            arac: None,
            vekil_temizlik: true,
        });
    }

    if k.contains("refaktör") || k.contains("refaktor") || k.contains("siber") {
        plan.push(ZincirAdimi {
            etiket: "proje_refaktor".into(),
            aciklama: format!("Refaktör: {yol}"),
            yol: Some(yol.clone()),
            arac: Some(AracKomut::ProjeRefaktor {
                yol: yol.clone(),
                istek: refaktor_istegi(metin),
            }),
            vekil_temizlik: false,
        });
    } else if k.contains("dosya") && k.contains("oku") {
        plan.push(ZincirAdimi {
            etiket: "dosya_oku".into(),
            aciklama: format!("Dosya okunuyor: {yol}"),
            yol: Some(yol.clone()),
            arac: None,
            vekil_temizlik: false,
        });
    }

    if k.contains("kod") && (k.contains("enjekte") || k.contains("yaz")) {
        plan.push(ZincirAdimi {
            etiket: "kod_enjekte".into(),
            aciklama: format!("Kod enjeksiyonu: {yol}"),
            yol: Some(yol.clone()),
            arac: Some(AracKomut::KodYaz {
                yol: yol.clone(),
                icerik: kod_ustasi::MERHABA_DUNYA_RS.to_string(),
                enjekte_hedef: None,
            }),
            vekil_temizlik: false,
        });
    }

    if k.contains("derle")
        || k.contains("cargo")
        || k.contains("kararlı")
        || k.contains("kararli")
        || k.contains("kararlı derle")
    {
        plan.push(ZincirAdimi {
            etiket: "otonom_derle".into(),
            aciklama: "Kararlı otonom derleme döngüsü".into(),
            yol: Some(proje_yolu.clone()),
            arac: Some(AracKomut::OtonomDerle {
                yol: proje_yolu,
                uyarilari_temizle: k.contains("uyarı") || k.contains("uyari") || k.contains("temizle"),
            }),
            vekil_temizlik: false,
        });
    }

    if plan.len() >= 2 {
        Some(plan)
    } else {
        None
    }
}

fn adim_log(adim_no: usize, toplam: usize, aciklama: &str) -> String {
    format!("▸ [ZİNCİR] Adım {adim_no}/{toplam}: {aciklama}...")
}

fn adim_basarili_log(adim_no: usize, toplam: usize, aciklama: &str) -> String {
    format!("✓ [ZİNCİR] Adım {adim_no}/{toplam}: {aciklama} tamamlandı")
}

/// Planı sırayla yürüt — her adım başarılı olunca sonrakine geç.
pub fn zincir_yurut(plan: &[ZincirAdimi]) -> Result<ZincirSonuc> {
    let toplam = plan.len();
    let mut adim_loglari = Vec::new();
    let mut ham_ciktilar = Vec::new();

    adim_loglari.push(format!(
        "[Plan: {}]",
        plan.iter()
            .map(|a| a.etiket.as_str())
            .collect::<Vec<_>>()
            .join(" -> ")
    ));

    for (i, adim) in plan.iter().enumerate() {
        let no = i + 1;
        adim_loglari.push(adim_log(no, toplam, &adim.aciklama));

        let sonuc = if adim.vekil_temizlik {
            match os_seviyesi_kontrol() {
                Ok(liste) => {
                    if liste.is_empty() {
                        "Vekil: temizlenecek ağır süreç bulunamadı.".into()
                    } else {
                        format!("Vekil temizlik: sonlandırılan — {}", liste.join(", "))
                    }
                }
                Err(e) => format!("Vekil temizlik hatası: {e}"),
            }
        } else if adim.etiket == "dosya_oku" {
            let yol = adim.yol.as_deref().unwrap_or(".");
            kod_ustasi::dosya_oku(yol).unwrap_or_else(|e| format!("Okuma hatası: {e}"))
        } else if let Some(ref arac) = adim.arac {
            araclar::arac_calistir(arac)?
        } else {
            "Atlandı.".into()
        };

        let basarisiz = sonuc.contains('✗')
            || sonuc.to_lowercase().contains("hata:")
            || sonuc.to_lowercase().contains("error")
                && !sonuc.contains("✓");

        ham_ciktilar.push(sonuc.clone());

        if basarisiz {
            adim_loglari.push(format!(
                "✗ [ZİNCİR] Adım {no}/{toplam} başarısız — zincir durduruldu."
            ));
            return Ok(ZincirSonuc {
                basarili: false,
                adim_loglari,
                ham_ciktilar,
                ozet: format!(
                    "Zincir adım {no}/{toplam} başarısız oldu. Logları incele, Sinan."
                ),
            });
        }

        adim_loglari.push(adim_basarili_log(no, toplam, &adim.aciklama));
    }

    Ok(ZincirSonuc {
        basarili: true,
        adim_loglari,
        ham_ciktilar,
        ozet: "✓ Çoklu operasyon zinciri başarıyla kararlı sonuca ulaştırıldı, Sinan kanka!"
            .into(),
    })
}

/// Plan + yürütme — beyin giriş noktası.
pub fn otonom_zincir_planla_ve_yurut(metin: &str) -> Option<Result<ZincirSonuc>> {
    plan_olustur(metin).map(|p| zincir_yurut(&p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cok_adimli_algila() {
        let m = "Sistemi kontrol et, temizle ve ardından src/lib.rs dosyasını refaktör edip kararlı derle";
        assert!(cok_adimli_mi(m));
        let plan = plan_olustur(m).unwrap();
        assert!(plan.len() >= 3);
    }

    #[test]
    fn test_tek_adim_zincir_degil() {
        assert!(!cok_adimli_mi("sistem durumu"));
    }
}

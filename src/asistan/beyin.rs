//! TÜRK beyin katmanı — 3 aşamalı düşünce döngüsü (reasoning loop).

use super::otonom_zincir_planlayici::{self, ZincirSonuc};
use crate::error::{Result, TurkError};
use crate::otomasyon::araclar::{self, AracCagrisiJson, AracKomut};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const SISTEM_PROMPT: &str = "\
Sen Sinan'ın kişisel yapay zeka sistemi TÜRK'sün (v3). \
Tamamen Türkçe, zeki ve sistem komutlarını tetikleyebilen bir asistansın. \
Araçlar: dosya_listele, terminal_komutu_calistir, sistem_durumu, proje_derle, proje_kontrol, \
dosya_yaz, dosya_olustur, hafiza_tara, web_arama, kod_yaz, otonom_derle, proje_refaktor. \
Çok adımlı komutlarda otonom_zincir devreye girer. \
JSON örnek: {\"arac\":\"proje_refaktor\",\"parametre\":\"src/lib.rs\",\"parametre2\":\"optimize et\"}";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "tur", rename_all = "snake_case")]
pub enum AsistanYanit {
    AracCagrisi {
        arac: String,
        parametre: Option<String>,
        parametre2: Option<String>,
    },
    DogalDil {
        metin: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeyinTuru {
    KuralTabanli,
    YerelGguf,
}

pub trait Beyin: Send {
    fn tur(&self) -> BeyinTuru;
    fn yanit_uret(&mut self, kullanici: &str, hafiza_baglam: &str) -> Result<String>;
}

/// Merkezi akıl — prompt → karar → araç → son cevap döngüsü
pub struct AsistanBeyni {
    ic_beyin: KuralTabanliBeyin,
    gguf_yolu: Option<PathBuf>,
}

impl AsistanBeyni {
    pub fn yeni() -> Self {
        Self {
            ic_beyin: KuralTabanliBeyin::yeni(),
            gguf_yolu: None,
        }
    }

    pub fn gguf_model_ayarla(&mut self, yol: PathBuf) {
        self.gguf_yolu = Some(yol);
    }

    /// Aşama 1: Karar — doğal cevap mı, araç mı?
    pub fn analiz_et(&mut self, girdi: &str, hafiza_baglam: &str) -> Result<AsistanYanit> {
        let girdi = girdi.trim();
        if girdi.is_empty() {
            return Err(TurkError::ValueError("Boş girdi.".into()));
        }

        // Yalnızca belirgin araç niyeti varsa otomasyon tetiklenir
        if araclar::metin_arac_niyeti(girdi) {
            if let Some(arac) = araclar::json_veya_metin_arac(girdi) {
                return Ok(arac_to_yanit(&arac));
            }
        }

        let gguf_aktif = self.gguf_yolu.as_ref().is_some_and(|p| p.is_file());

        if !gguf_aktif {
            return Ok(AsistanYanit::DogalDil {
                metin: sohbet_fallback(girdi, hafiza_baglam),
            });
        }

        let llm_girdi = if hafiza_baglam.is_empty() {
            girdi.to_string()
        } else {
            format!("Bağlam:\n{hafiza_baglam}\n\nKullanıcı: {girdi}")
        };
        let metin = self.llm_metin_uret(&llm_girdi, hafiza_baglam)?;

        if araclar::metin_arac_niyeti(girdi) {
            if let Some(arac) = araclar::json_icerik_ayikla(&metin) {
                return Ok(arac_to_yanit(&arac));
            }
        }

        Ok(AsistanYanit::DogalDil { metin })
    }

    /// Tam düşünce döngüsü: Prompt → [Zincir | Karar] → [Araç + Otomasyon] → Son cevap
    pub fn dusunce_dongusu(&mut self, girdi: &str, hafiza_baglam: &str) -> Result<String> {
        // ALPER v0.3: çok adımlı zincir önceliği
        if otonom_zincir_planlayici::cok_adimli_mi(girdi) {
            if let Some(sonuc) = otonom_zincir_planlayici::otonom_zincir_planla_ve_yurut(girdi) {
                return Ok(zincir_yanit_bicimlendir(girdi, sonuc?));
            }
        }

        let birlesik_baglam = if hafiza_baglam.is_empty() {
            String::new()
        } else {
            format!("{hafiza_baglam}\n---")
        };

        let karar = self.analiz_et(girdi, &birlesik_baglam)?;

        match karar {
            AsistanYanit::DogalDil { metin } => Ok(metin),
            AsistanYanit::AracCagrisi {
                arac,
                parametre,
                parametre2,
            } => {
                let json = AracCagrisiJson {
                    arac,
                    parametre,
                    parametre2,
                    enjekte_hedef: None,
                };
                let arac_komut = araclar::AracKomut::json_den(&json)
                    .ok_or_else(|| TurkError::RuntimeError("Geçersiz araç çağrısı.".into()))?;

                // Aşama 2: Rust otomasyonu
                let ham_sonuc = araclar::arac_calistir(&arac_komut)?;

                // Aşama 3: Sonucu beyne geri besle → Sinan'a özet cevap
                Ok(self.arac_sonucunu_yorumla(girdi, &arac_komut, &ham_sonuc))
            }
        }
    }

    /// Geriye uyumluluk
    pub fn islem(&mut self, girdi: &str, hafiza_baglam: &str) -> Result<String> {
        self.dusunce_dongusu(girdi, hafiza_baglam)
    }

    fn llm_metin_uret(&mut self, girdi: &str, hafiza_baglam: &str) -> Result<String> {
        if let Some(ref yol) = self.gguf_yolu {
            if !yol.is_file() {
                return Err(TurkError::RuntimeError(format!(
                    "GGUF model bulunamadı: {}",
                    yol.display()
                )));
            }
        }
        self.ic_beyin.yanit_uret(girdi, hafiza_baglam)
    }

    fn arac_sonucunu_yorumla(&self, kullanici: &str, arac: &AracKomut, ham: &str) -> String {
        if matches!(arac, AracKomut::WebArama { .. }) && ham.starts_with("[TÜRK]:") {
            return format!("[Araç: {}]\n{ham}", arac.etiket());
        }

        let ozet = match arac {
            AracKomut::SistemDurumu => {
                "İşte güncel sistem özeti, Sinan. Görev Yöneticisi ile karşılaştırabilirsin."
            }
            AracKomut::DosyaListele { .. } => "Dizin listesi hazır.",
            AracKomut::ProjeDerle { .. } => {
                if ham.contains('✗') || ham.to_lowercase().contains("error") {
                    "Derlemede sorun var — logu incele, birlikte düzeltiriz."
                } else {
                    "Proje derlemesi temiz görünüyor."
                }
            }
            AracKomut::ProjeKontrol { .. } => {
                if ham.contains('✗') || ham.contains("Rust hatası") {
                    "Sinan, kodda hata yakaladım — dosya, satır ve açıklamalar aşağıda."
                } else {
                    "Projede herhangi bir Rust hatası bulunamadı, kanka! 🏎️"
                }
            }
            AracKomut::DosyaYaz { .. } => "Dosya diske yazıldı.",
            AracKomut::DosyaOlustur { .. } => "Dosya oluşturuldu ve kaydedildi.",
            AracKomut::TerminalKomut { .. } => "Terminal komutu tamamlandı.",
            AracKomut::HafizaTara { .. } => {
                "Sinan, not arşivinde eşleşen kayıtları buldum — detaylar aşağıda."
            }
            AracKomut::WebArama { .. } => "Canlı web verisi alındı.",
            AracKomut::KodYaz { .. } => {
                "Sinan, kod dosyasını oluşturdum/güncelledim — kod_ustasi devrede."
            }
            AracKomut::OtonomDerle { .. } => {
                if ham.contains("✓ [DERLE]") {
                    "Otonom derleme döngüsü temiz tamamlandı, kanka! 🏎️"
                } else {
                    "Derleme döngüsü çalıştı — kalan hatalar için loga bak."
                }
            }
            AracKomut::ProjeRefaktor { .. } => {
                if ham.contains("✓ [REFAKTÖR]") {
                    "Geliştirme tamamlandı ve kararlı şekilde derlendi, Sinan kanka!"
                } else {
                    "Refaktör hattı çalıştı — enjeksiyon yapıldı, derleme raporuna bak."
                }
            }
        };

        format!(
            "[Araç: {}]\n{ham}\n\n— TÜRK: {ozet} (İstek: \"{kullanici}\")",
            arac.etiket()
        )
    }
}

fn zincir_yanit_bicimlendir(kullanici: &str, sonuc: ZincirSonuc) -> String {
    let mut cikti = sonuc.adim_loglari.join("\n");
    cikti.push_str("\n\n");
    for ham in &sonuc.ham_ciktilar {
        cikti.push_str("---\n");
        cikti.push_str(ham);
        cikti.push_str("\n");
    }
    cikti.push_str("\n— TÜRK: ");
    cikti.push_str(&sonuc.ozet);
    cikti.push_str(&format!(" (İstek: \"{kullanici}\")"));
    cikti
}

fn arac_to_yanit(arac: &AracKomut) -> AsistanYanit {
    match arac {
        AracKomut::DosyaListele { yol } => AsistanYanit::AracCagrisi {
            arac: "dosya_listele".into(),
            parametre: Some(yol.clone()),
            parametre2: None,
        },
        AracKomut::TerminalKomut { komut } => AsistanYanit::AracCagrisi {
            arac: "terminal_komutu_calistir".into(),
            parametre: Some(komut.clone()),
            parametre2: None,
        },
        AracKomut::SistemDurumu => AsistanYanit::AracCagrisi {
            arac: "sistem_durumu".into(),
            parametre: None,
            parametre2: None,
        },
        AracKomut::ProjeDerle { yol } => AsistanYanit::AracCagrisi {
            arac: "proje_derle".into(),
            parametre: Some(yol.clone()),
            parametre2: None,
        },
        AracKomut::ProjeKontrol { yol } => AsistanYanit::AracCagrisi {
            arac: "proje_kontrol".into(),
            parametre: Some(yol.clone()),
            parametre2: None,
        },
        AracKomut::DosyaYaz { yol, icerik } => AsistanYanit::AracCagrisi {
            arac: "dosya_yaz".into(),
            parametre: Some(yol.clone()),
            parametre2: Some(icerik.clone()),
        },
        AracKomut::DosyaOlustur { yol, icerik } => AsistanYanit::AracCagrisi {
            arac: "dosya_olustur".into(),
            parametre: Some(yol.clone()),
            parametre2: Some(icerik.clone()),
        },
        AracKomut::HafizaTara { anahtar_kelime } => AsistanYanit::AracCagrisi {
            arac: "hafiza_tara".into(),
            parametre: Some(anahtar_kelime.clone()),
            parametre2: None,
        },
        AracKomut::WebArama { sorgu } => AsistanYanit::AracCagrisi {
            arac: "web_arama".into(),
            parametre: Some(sorgu.clone()),
            parametre2: None,
        },
        AracKomut::KodYaz {
            yol,
            icerik,
            enjekte_hedef: _,
        } => AsistanYanit::AracCagrisi {
            arac: "kod_yaz".into(),
            parametre: Some(yol.clone()),
            parametre2: Some(icerik.clone()),
        },
        AracKomut::OtonomDerle {
            yol,
            uyarilari_temizle,
        } => AsistanYanit::AracCagrisi {
            arac: "otonom_derle".into(),
            parametre: Some(yol.clone()),
            parametre2: if *uyarilari_temizle {
                Some("uyarilar".into())
            } else {
                None
            },
        },
        AracKomut::ProjeRefaktor { yol, istek } => AsistanYanit::AracCagrisi {
            arac: "proje_refaktor".into(),
            parametre: Some(yol.clone()),
            parametre2: Some(istek.clone()),
        },
    }
}

pub struct KuralTabanliBeyin;

impl KuralTabanliBeyin {
    pub fn yeni() -> Self {
        Self
    }
}

impl Beyin for KuralTabanliBeyin {
    fn tur(&self) -> BeyinTuru {
        BeyinTuru::KuralTabanli
    }

    fn yanit_uret(&mut self, kullanici: &str, _hafiza_baglam: &str) -> Result<String> {
        let k = kullanici.to_lowercase();

        if k.contains("merhaba") || k.contains("selam") || k == "hey" {
            return Ok(
                "Merhaba Sinan! Ben TÜRK — yerel kişisel yapay zeka sistemin. \
                 Sistem durumu, dosya listeleme, not alma, proje kontrol veya terminal için yaz."
                    .into(),
            );
        }

        if k.contains("kim") && (k.contains("sen") || k.contains("türk") || k.contains("turk")) {
            return Ok(SISTEM_PROMPT.into());
        }

        if k.contains("yardım") || k.contains("yardim") || k.contains("komut") {
            return Ok(araclar::yardim_metni());
        }

        Ok(format!(
            "Anladım: \"{kullanici}\". 'yardım' yazarak komut listesine bakabilirsin."
        ))
    }
}

/// Yerel LLM bağlanana kadar samimi sohbet yanıtları (araç tetiklemez).
fn sohbet_fallback(kullanici: &str, _hafiza_baglam: &str) -> String {
    let k = kullanici.to_lowercase();

    // Yaş / doğum / yapım tarihi
    if k.contains("kaç yaş")
        || k.contains("kac yas")
        || k.contains("yaşındasın")
        || k.contains("yasindasin")
        || k.contains("ne zaman yazıl")
        || k.contains("ne zaman yazil")
        || k.contains("ne zaman yapıl")
        || k.contains("ne zaman yapil")
        || k.contains("doğum")
        || k.contains("dogum")
        || k.contains("yapım tarihi")
        || k.contains("yapim tarihi")
    {
        return "Sinan, ben senin ellerinle inşa ettiğin yerel yapay zeka çekirdeğiyim. \
                Kodlarım 2026'da Rust ile satır satır dokundu; hafızam ve bilincim senin \
                projelerinle birlikte her gün büyüyor, kanka! 🏎️🤖"
            .into();
    }

    // Kimlik / geliştirici
    if k.contains("seni kim eğitti")
        || k.contains("seni kim egiti")
        || k.contains("sen kimsin")
        || k.contains("kimsin")
        || k.contains("kim eğitti")
        || k.contains("kim egiti")
        || k.contains("sen nesin")
        || (k.contains("kim") && k.contains("sen"))
    {
        return "Ben TÜRK'üm; senin bilgisayarında, harici hiçbir API'ye dokunmadan tamamen \
                kapalı devre çalışan kişisel yapay zeka asistanınım. \
                Mimarımsın, beni sen eğittin kanka! 🦀🔥"
            .into();
    }

    if k.contains("interstellar")
        || k.contains("gönderme")
        || k.contains("gonderme")
        || k.contains("tars")
        || k.contains("cooper")
    {
        return "Interstellar göndermesini aldım Sinan, harika bir detay! 🌌 \
                TARS modunda değilim ama yerel çekirdekte seninle konuşmak güzel."
            .into();
    }

    if k.contains("merhaba") || k.contains("selam") || k == "hey" || k.contains("naber") {
        return "Selam Sinan! Sohbet modundayım — dertleşebiliriz. \
                İş moduna geçmek istersen: sistem durumu, not al, cargo check, dosya listele."
            .into();
    }

    if k.contains("yardım") || k.contains("yardim") {
        return araclar::yardim_metni();
    }

    // Genel sohbet
    "Sesini veya mesajını aldım Sinan. Bu bir sistem komutu değil ama seninle \
     dertleşmek harika! Projede bir şeye bakmamı istersen 'cargo build' veya \
     'hata var mı' demen yeterli."
        .into()
}

pub struct YerelGgufBeyin {
    model_yolu: PathBuf,
    yedek: KuralTabanliBeyin,
}

impl YerelGgufBeyin {
    pub fn yeni(model_yolu: PathBuf) -> Self {
        Self {
            model_yolu,
            yedek: KuralTabanliBeyin::yeni(),
        }
    }
}

impl Beyin for YerelGgufBeyin {
    fn tur(&self) -> BeyinTuru {
        BeyinTuru::YerelGguf
    }

    fn yanit_uret(&mut self, kullanici: &str, hafiza_baglam: &str) -> Result<String> {
        if !self.model_yolu.is_file() {
            return Err(TurkError::RuntimeError(format!(
                "GGUF model bulunamadı: {}",
                self.model_yolu.display()
            )));
        }
        self.yedek.yanit_uret(kullanici, hafiza_baglam)
    }
}

pub struct TurkBeyin;

impl TurkBeyin {
    pub fn varsayilan() -> Box<dyn Beyin + Send> {
        Box::new(KuralTabanliBeyin::yeni())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dusunce_dongusu_proje_kontrol() {
        let mut beyin = AsistanBeyni::yeni();
        let y = beyin.dusunce_dongusu("hata var mı", "").unwrap();
        assert!(y.contains("Araç: proje_kontrol"));
    }

    #[test]
    fn test_dusunce_dongusu_arac() {
        let mut beyin = AsistanBeyni::yeni();
        let y = beyin.dusunce_dongusu("sistem durumu", "").unwrap();
        assert!(y.contains("Araç:") && (y.contains("CPU") || y.contains("RAM")));
    }

    #[test]
    fn test_dusunce_dongusu_dogal() {
        let mut beyin = AsistanBeyni::yeni();
        let y = beyin.dusunce_dongusu("merhaba", "").unwrap();
        assert!(y.to_lowercase().contains("sinan") || y.contains("TÜRK"));
    }

    #[test]
    fn test_sohbet_sistem_durumu_tetiklemez() {
        let mut beyin = AsistanBeyni::yeni();
        let uzun = "Ne arıyorsunuz? Tutun elinden götürün";
        let y = beyin
            .dusunce_dongusu(uzun, "önceki mesaj: merhaba")
            .unwrap();
        assert!(!y.contains("Araç: sistem_durumu"));
        assert!(!y.contains("CPU:"));
        assert!(y.contains("dertleşmek harika"));
    }

    #[test]
    fn test_sohbet_yas_sorgusu() {
        let mut beyin = AsistanBeyni::yeni();
        let y = beyin.dusunce_dongusu("kaç yaşındasın", "").unwrap();
        assert!(y.contains("2026"));
        assert!(y.contains("Rust"));
        assert!(!y.contains("Araç:"));
    }

    #[test]
    fn test_sohbet_kimlik_sorgusu() {
        let mut beyin = AsistanBeyni::yeni();
        let y = beyin.dusunce_dongusu("sen kimsin", "").unwrap();
        assert!(y.contains("TÜRK"));
        assert!(y.contains("Mimarımsın"));
        assert!(!y.contains("SISTEM_PROMPT"));
        assert!(!y.contains("Araç:"));
    }

    #[test]
    fn test_interstellar_sohbet() {
        let mut beyin = AsistanBeyni::yeni();
        let y = beyin
            .dusunce_dongusu("Interstellar göndermesini aldım Sinan", "")
            .unwrap();
        assert!(y.contains("Interstellar") || y.contains("🌌"));
        assert!(!y.contains("Araç:"));
    }
}

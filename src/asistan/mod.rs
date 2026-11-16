//! TÜRK Çekirdeği — yerel kişisel yapay zeka motoru.

mod beyin;
mod hafiza;
mod otonom_zincir_planlayici;

pub use beyin::{
    AsistanBeyni, AsistanYanit, Beyin, BeyinTuru, KuralTabanliBeyin, TurkBeyin, YerelGgufBeyin,
    SISTEM_PROMPT,
};
pub use hafiza::{
    hafiza_tara, hafiza_tara_ozet, Hafiza, HafizaEslesme, HafizaHavuzu, HafizaKaydi, HafizaMerkezi,
    SemantikKayit,
};
pub use otonom_zincir_planlayici::{
    cok_adimli_mi, otonom_zincir_planla_ve_yurut, plan_olustur, zincir_yurut, ZincirAdimi,
    ZincirSonuc,
};

use crate::error::{Result, TurkError};
use std::path::PathBuf;

/// TÜRK sistemi motoru: AsistanBeyni + hafıza
pub struct TurkSistem {
    asistan: AsistanBeyni,
    hafiza: Hafiza,
}

impl TurkSistem {
    pub fn yeni() -> Self {
        Self {
            asistan: AsistanBeyni::yeni(),
            hafiza: Hafiza::yeni(),
        }
    }

    pub fn asistan(&self) -> &AsistanBeyni {
        &self.asistan
    }

    pub fn asistan_mut(&mut self) -> &mut AsistanBeyni {
        &mut self.asistan
    }

    pub fn hafiza(&self) -> &Hafiza {
        &self.hafiza
    }

    pub fn hafiza_mut(&mut self) -> &mut Hafiza {
        &mut self.hafiza
    }

    /// Kullanıcı mesajını işler (GUI / CLI giriş noktası)
    pub fn mesaj_isle(&mut self, kullanici: &str) -> Result<String> {
        let kullanici = kullanici.trim();
        if kullanici.is_empty() {
            return Err(TurkError::ValueError("Boş mesaj gönderilemez.".into()));
        }

        self.hafiza.ekle("kullanici", kullanici);
        let baglam = self.hafiza.ilgili_baglam(kullanici, 4);
        let yanit = self.asistan.dusunce_dongusu(kullanici, &baglam)?;
        self.hafiza.ekle("turk", &yanit);
        Ok(yanit)
    }

    /// Semantik havuz son hatırlanan bağlamlar (HUD)
    pub fn semantik_son_hatirlanan(&self) -> Vec<String> {
        self.hafiza.havuz().son_hatirlananlar().to_vec()
    }

    pub fn gguf_model_ayarla(&mut self, yol: impl Into<PathBuf>) {
        self.asistan.gguf_model_ayarla(yol.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turk_selamlama() {
        let mut motor = TurkSistem::yeni();
        let yanit = motor.mesaj_isle("Merhaba TÜRK").unwrap();
        assert!(
            yanit.to_lowercase().contains("türk")
                || yanit.to_lowercase().contains("turk")
                || yanit.contains("Sinan")
        );
    }

    #[test]
    fn test_turk_sistem_araci() {
        let mut motor = TurkSistem::yeni();
        let yanit = motor.mesaj_isle("sistem durumu").unwrap();
        assert!(yanit.contains("CPU") || yanit.contains("RAM"));
    }

    #[test]
    fn test_turk_dosya_listele() {
        let mut motor = TurkSistem::yeni();
        let yanit = motor.mesaj_isle("dosya listele .").unwrap();
        assert!(yanit.contains("öğe") || yanit.contains("dosya") || yanit.contains("[klasör]"));
    }
}

//! TÜRK sistem otomasyonu — dosya, süreç ve sistem izleme araçları.

pub mod araclar;
pub mod derleyici;
pub mod kod_ustasi;
pub mod notlar;
pub mod refaktor;
pub mod sistem;
pub mod vekil;
pub mod web;

#[cfg(windows)]
mod win_cpu;

pub use araclar::{AracCagrisiJson, AracKomut, arac_calistir, json_veya_metin_arac, metin_arac_niyeti};
pub use sistem::{
    SistemDurumu, cpu_kullanim_guncelle, cpu_kullanim_oku, dosya_listele, dosyalari_listele,
    dosya_olustur, dosya_yaz, proje_derle, proje_kontrol, proje_kok_dizini, surec_baslat, sistem_durumu,
    terminal_komutu_calistir, turk_sistem_veri_dizini,
};
pub use refaktor::{
    REFAKTOR_BAGIMLI_TARAMA, REFAKTOR_BASARI, REFAKTOR_BASLANGIC, proje_geneli_tara,
    proje_refaktor_yurut,
};
pub use derleyici::{
    CargoCheckSonuc, DerlemeHatasi, cargo_check_json, otonom_derle_dongusu_bagimlilar,
    otonom_derle_rapor,
};
pub use kod_ustasi::{MERHABA_DUNYA_RS, dosya_oku as kod_dosya_oku, dosya_yaz as kod_dosya_yaz, kod_enjekte_et, kod_yaz_uygula};
pub use notlar::{HafizaEslesme, HafizaMerkezi, hafiza_tara, hafiza_tara_ozet};
pub use vekil::{
    SistemVekili, TURK_VEKIL_OPTIMIZE_MESAJ, TURK_VEKIL_UYARI_MESAJ, VekilKomut, VekilOlay,
    os_seviyesi_kontrol,
};
pub use web::{canli_hava_durumu_cek, hava_sorgusu_mu, sehir_ayikla};

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

pub use araclar::{
    arac_calistir, json_veya_metin_arac, metin_arac_niyeti, AracCagrisiJson, AracKomut,
};
pub use derleyici::{
    cargo_check_json, otonom_derle_dongusu_bagimlilar, otonom_derle_rapor, CargoCheckSonuc,
    DerlemeHatasi,
};
pub use kod_ustasi::{
    dosya_oku as kod_dosya_oku, dosya_yaz as kod_dosya_yaz, kod_enjekte_et, kod_yaz_uygula,
    MERHABA_DUNYA_RS,
};
pub use notlar::{hafiza_tara, hafiza_tara_ozet, HafizaEslesme, HafizaMerkezi};
pub use refaktor::{
    proje_geneli_tara, proje_refaktor_yurut, REFAKTOR_BAGIMLI_TARAMA, REFAKTOR_BASARI,
    REFAKTOR_BASLANGIC,
};
pub use sistem::{
    cpu_kullanim_guncelle, cpu_kullanim_oku, dosya_listele, dosya_olustur, dosya_yaz,
    dosyalari_listele, proje_derle, proje_kok_dizini, proje_kontrol, sistem_durumu, surec_baslat,
    terminal_komutu_calistir, turk_sistem_veri_dizini, SistemDurumu,
};
pub use vekil::{
    os_seviyesi_kontrol, SistemVekili, VekilKomut, VekilOlay, TURK_VEKIL_OPTIMIZE_MESAJ,
    TURK_VEKIL_UYARI_MESAJ,
};
pub use web::{canli_hava_durumu_cek, hava_sorgusu_mu, sehir_ayikla};

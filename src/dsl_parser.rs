//! TÜRK komut DSL'i için resmi ayrıştırıcı.
//!
//! Yorumlayıcı sözdizimini (`model_olustur()`, `katman_ekle(...)`, …) doğrular
//! ve hataları satır/sütun konumuyla bildirir.

use crate::error::{Result, TurkError};

/// Ayrıştırılmış DSL komutu
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DslKomut {
    ModelOlustur,
    KatmanEkle { icerik: String },
    Derle { icerik: String },
    ModelBilgisi,
    Egit { icerik: String },
    VeriYukle { icerik: String },
    Kaydet { icerik: String },
    Yukle { icerik: String },
    OnnxDisaAktar { dosya: String },
    CihazBilgisi,
    ModelZoo,
    ModelIndir { id: String, hedef: Option<String> },
    ModelCalistir { id: String },
}

/// Tek satırlık komutu ayrıştırır.
///
/// `satir_no` dosyadaki satır numarasıdır (1 tabanlı); hata mesajlarında kullanılır.
pub fn komut_ayristir(komut: &str, satir_no: usize) -> Result<DslKomut> {
    let ham = komut.trim();
    if ham.is_empty() || ham.starts_with('#') || ham.starts_with("//") {
        return Err(TurkError::SyntaxError("Boş komut".into()));
    }

    parantez_dengesi_kontrol(ham, satir_no)?;

    if ham == "model_olustur()" {
        return Ok(DslKomut::ModelOlustur);
    }
    if ham == "model_bilgisi()" {
        return Ok(DslKomut::ModelBilgisi);
    }
    if ham == "cihaz_bilgisi()" || ham == "cihazlar()" {
        return Ok(DslKomut::CihazBilgisi);
    }
    if ham == "model_zoo()" {
        return Ok(DslKomut::ModelZoo);
    }

    if let Some(icerik) = fonksiyon_ici(ham, "katman_ekle", satir_no)? {
        if icerik.trim().is_empty() {
            return konumlu_hata(
                ham,
                satir_no,
                ham.find('(').unwrap_or(0) + 1,
                "katman_ekle() içi boş olamaz",
            );
        }
        return Ok(DslKomut::KatmanEkle { icerik });
    }

    if let Some(icerik) = fonksiyon_ici(ham, "derle", satir_no)? {
        return Ok(DslKomut::Derle { icerik });
    }
    if let Some(icerik) = fonksiyon_ici(ham, "egit", satir_no)? {
        return Ok(DslKomut::Egit { icerik });
    }
    if let Some(icerik) = fonksiyon_ici(ham, "kaydet", satir_no)? {
        return Ok(DslKomut::Kaydet { icerik });
    }
    if let Some(icerik) = fonksiyon_ici(ham, "yukle", satir_no)? {
        return Ok(DslKomut::Yukle { icerik });
    }

    if let Some(icerik) = fonksiyon_ici(ham, "veri_yukle", satir_no)? {
        return Ok(DslKomut::VeriYukle { icerik });
    }
    if let Some(icerik) = fonksiyon_ici(ham, "onnx_disa_aktar", satir_no)? {
        let dosya = tirnakli_dosya_yolu(&icerik).ok_or_else(|| {
            konumlu_hata_msg(ham, satir_no, ham.find('(').unwrap_or(0) + 1, "onnx_disa_aktar(\"dosya.onnx.json\") bekleniyor")
        })?;
        return Ok(DslKomut::OnnxDisaAktar { dosya });
    }

    if let Some(icerik) = fonksiyon_ici(ham, "model_indir", satir_no)? {
        let (id, hedef) = zoo_arg_parse(&icerik, ham, satir_no)?;
        return Ok(DslKomut::ModelIndir { id, hedef });
    }
    if let Some(icerik) = fonksiyon_ici(ham, "model_calistir", satir_no)? {
        let (id, _) = zoo_arg_parse(&icerik, ham, satir_no)?;
        return Ok(DslKomut::ModelCalistir { id });
    }

    // Eski büyük harf sözdizimi: VERI_YUKLE: (...)
    if ham.to_uppercase().starts_with("VERI_YUKLE") {
        let icerik = ham
            .split_once(':')
            .map(|(_, r)| r.trim().trim_start_matches('(').trim_end_matches(')').to_string())
            .unwrap_or_default();
        return Ok(DslKomut::VeriYukle { icerik });
    }

    konumlu_hata(ham, satir_no, 0, &format!("Desteklenmeyen komut: {ham}"))
}

/// Parantez dengesini kontrol eder; hata varsa sütun konumunu bildirir.
pub fn parantez_dengesi_kontrol(komut: &str, satir_no: usize) -> Result<()> {
    let mut derinlik = 0i32;
    let mut string_icinde = false;
    let mut escape = false;

    for (byte_idx, ch) in komut.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match ch {
            '\\' if string_icinde => escape = true,
            '"' => string_icinde = !string_icinde,
            '(' if !string_icinde => derinlik += 1,
            ')' if !string_icinde => {
                derinlik -= 1;
                if derinlik < 0 {
                    return konumlu_hata_void(
                        komut,
                        satir_no,
                        byte_idx,
                        "Fazladan kapanış parantezi ')'",
                    );
                }
            }
            _ => {}
        }
    }

    if string_icinde {
        return konumlu_hata_void(komut, satir_no, komut.len().saturating_sub(1), "Kapatılmamış string");
    }
    if derinlik > 0 {
        return konumlu_hata_void(komut, satir_no, komut.len().saturating_sub(1), "Kapatılmamış '(' parantezi");
    }
    if derinlik < 0 {
        return konumlu_hata_void(komut, satir_no, 0, "Parantez dengesi bozuk");
    }
    Ok(())
}

/// Hata satırını okunabilir biçimde formatlar (caret ile).
pub fn hata_satirini_goster(ham_satir: &str, sutun: usize, mesaj: &str) -> String {
    let caret_pos = sutun.saturating_sub(1);
    let caret = " ".repeat(caret_pos) + "^";
    format!("Satır hatası — {mesaj}\n  {ham_satir}\n  {caret}")
}

fn fonksiyon_ici<'a>(
    komut: &'a str,
    ad: &str,
    satir_no: usize,
) -> Result<Option<String>> {
    let prefix = format!("{ad}(");
    if !komut.starts_with(&prefix) {
        return Ok(None);
    }
    if !komut.ends_with(')') {
        return Err(konumlu_hata_msg(
            komut,
            satir_no,
            komut.len().saturating_sub(1),
            &format!("{ad}(...) kapanış parantezi ')' eksik"),
        ));
    }
    let icerik = &komut[prefix.len()..komut.len() - 1];
    Ok(Some(icerik.to_string()))
}

fn tirnakli_dosya_yolu(icerik: &str) -> Option<String> {
    let t = icerik.trim();
    if t.starts_with('"') && t.ends_with('"') && t.len() >= 2 {
        Some(t[1..t.len() - 1].to_string())
    } else {
        None
    }
}

fn zoo_arg_parse(
    icerik: &str,
    ham: &str,
    satir_no: usize,
) -> Result<(String, Option<String>)> {
    let parcalar: Vec<&str> = icerik.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if parcalar.is_empty() {
        return Err(konumlu_hata_msg(
            ham,
            satir_no,
            ham.find('(').unwrap_or(0) + 1,
            "model id belirtilmedi",
        ));
    }
    let id = tirnakli_dosya_yolu(parcalar[0]).unwrap_or_else(|| parcalar[0].to_string());
    let hedef = parcalar
        .get(1)
        .map(|s| tirnakli_dosya_yolu(s).unwrap_or_else(|| (*s).to_string()));
    Ok((id, hedef))
}

fn byte_to_sutun(komut: &str, byte_idx: usize) -> usize {
    komut[..byte_idx.min(komut.len())].chars().count() + 1
}

fn konumlu_hata(
    komut: &str,
    satir_no: usize,
    byte_idx: usize,
    mesaj: &str,
) -> Result<DslKomut> {
    Err(konumlu_hata_msg(komut, satir_no, byte_idx, mesaj))
}

fn konumlu_hata_void(
    komut: &str,
    satir_no: usize,
    byte_idx: usize,
    mesaj: &str,
) -> Result<()> {
    Err(konumlu_hata_msg(komut, satir_no, byte_idx, mesaj))
}

fn konumlu_hata_msg(komut: &str, satir_no: usize, byte_idx: usize, mesaj: &str) -> TurkError {
    let sutun = byte_to_sutun(komut, byte_idx);
    TurkError::SyntaxError(format!(
        "Satır {satir_no}, sütun {sutun}: {mesaj}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_olustur() {
        let k = komut_ayristir("model_olustur()", 1).unwrap();
        assert_eq!(k, DslKomut::ModelOlustur);
    }

    #[test]
    fn test_katman_ekle() {
        let k = komut_ayristir("katman_ekle(yogun(2, 4, relu))", 3).unwrap();
        assert_eq!(
            k,
            DslKomut::KatmanEkle {
                icerik: "yogun(2, 4, relu)".into()
            }
        );
    }

    #[test]
    fn test_fazla_parantez_sutun() {
        let err = komut_ayristir("katman_ekle(yogun(2, 4)))", 5).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Satır 5"), "{msg}");
        assert!(msg.contains("sütun"), "{msg}");
    }

    #[test]
    fn test_eksik_parantez() {
        let err = komut_ayristir("derle(adam", 2).unwrap_err();
        assert!(err.to_string().contains("Satır 2"));
    }

    #[test]
    fn test_bilinmeyen_komut() {
        let err = komut_ayristir("foo(1,2)", 1).unwrap_err();
        assert!(err.to_string().contains("Desteklenmeyen komut"));
    }

    #[test]
    fn test_onnx_disa_aktar() {
        let k = komut_ayristir("onnx_disa_aktar(\"model.onnx.json\")", 10).unwrap();
        assert_eq!(
            k,
            DslKomut::OnnxDisaAktar {
                dosya: "model.onnx.json".into()
            }
        );
    }

    #[test]
    fn test_hata_satirini_goster() {
        let goster = hata_satirini_goster("katman_ekle(yogun(2, 8))", 14, "boyut uyumsuz");
        assert!(goster.contains('^'));
        assert!(goster.contains("boyut uyumsuz"));
    }

    #[test]
    fn test_model_indir_parse() {
        let k = komut_ayristir("model_indir(\"xor\")", 1).unwrap();
        assert_eq!(
            k,
            DslKomut::ModelIndir {
                id: "xor".into(),
                hedef: None
            }
        );
    }

    #[test]
    fn test_cihaz_bilgisi() {
        let k = komut_ayristir("cihaz_bilgisi()", 1).unwrap();
        assert_eq!(k, DslKomut::CihazBilgisi);
    }
}

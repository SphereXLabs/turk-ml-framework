//! TÜRK Model Zoo — hazır model şablonları ve indirme.
//!
//! `model_zoo/manifest.json` içindeki modeller depo kökünden veya
//! `TURK_ZOO_DIR` ortam değişkeninden okunur.

use crate::error::{Result, TurkError};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

const MANIFEST_JSON: &str = include_str!("../model_zoo/manifest.json");

/// Model Zoo manifest kökü
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZooManifest {
    pub version: String,
    pub models: Vec<ZooModelMeta>,
}

/// Tek model meta verisi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZooModelMeta {
    pub id: String,
    pub ad: String,
    pub aciklama: String,
    pub turk_dosya: String,
    pub agirlik: Option<String>,
    pub veri: Option<String>,
    #[serde(default)]
    pub etiketler: Vec<String>,
}

/// Gömülü manifest'i yükler
pub fn manifest_yukle() -> Result<ZooManifest> {
    serde_json::from_str(MANIFEST_JSON)
        .map_err(|e| TurkError::IOError(format!("Model Zoo manifest okunamadı: {e}")))
}

/// Kaynak model_zoo dizini (depo veya TURK_ZOO_DIR)
pub fn zoo_kok() -> PathBuf {
    if let Ok(dir) = std::env::var("TURK_ZOO_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model_zoo")
}

/// Kullanıcı önbellek dizini (~/.turk/zoo veya ./models)
pub fn kullanici_zoo_dizini() -> PathBuf {
    if let Ok(home) = std::env::var("USERPROFILE") {
        return PathBuf::from(home).join(".turk").join("zoo");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".turk").join("zoo");
    }
    PathBuf::from("models")
}

/// Model meta bilgisini id ile bulur
pub fn model_bul(id: &str) -> Result<ZooModelMeta> {
    let manifest = manifest_yukle()?;
    manifest
        .models
        .into_iter()
        .find(|m| m.id.eq_ignore_ascii_case(id))
        .ok_or_else(|| {
            TurkError::RuntimeError(format!(
                "Model Zoo'da '{id}' bulunamadı. `model_zoo()` ile listeyi görün."
            ))
        })
}

/// Model dosyalarını hedef dizine kopyalar
pub fn model_indir(id: &str, hedef_kok: Option<&Path>) -> Result<PathBuf> {
    let meta = model_bul(id)?;
    let kaynak = zoo_kok().join(&meta.id);
    if !kaynak.is_dir() {
        return Err(TurkError::IOError(format!(
            "Kaynak dizin yok: {}",
            kaynak.display()
        )));
    }

    let hedef = hedef_kok
        .map(Path::to_path_buf)
        .unwrap_or_else(kullanici_zoo_dizini)
        .join(&meta.id);

    if hedef.exists() {
        fs::remove_dir_all(&hedef).map_err(|e| {
            TurkError::IOError(format!("Eski model dizini silinemedi: {e}"))
        })?;
    }
    fs::create_dir_all(&hedef).map_err(|e| {
        TurkError::IOError(format!("Hedef dizin oluşturulamadı: {e}"))
    })?;
    dizin_kopyala(&kaynak, &hedef)?;

    Ok(hedef)
}

/// İndirilmiş modelin .turk betik yolunu döndürür (yoksa indirir)
pub fn model_script_yolu(id: &str) -> Result<PathBuf> {
    let meta = model_bul(id)?;
    let kok = kullanici_zoo_dizini().join(&meta.id);
    let script = kok.join("model.turk");
    if script.is_file() {
        return Ok(script);
    }
    let indirilen = model_indir(id, None)?;
    Ok(indirilen.join("model.turk"))
}

/// Model listesini metin tablosu olarak formatlar
pub fn zoo_listesi_metin() -> String {
    let manifest = manifest_yukle().unwrap_or_else(|_| ZooManifest {
        version: "?".into(),
        models: vec![],
    });

    let mut out = String::from("╔══════════════════════════════════════════════════════════════╗\n");
    out.push_str("║                    TÜRK Model Zoo                            ║\n");
    out.push_str("╠══════════════════════════════════════════════════════════════╣\n");

    if manifest.models.is_empty() {
        out.push_str("║  (Model bulunamadı)                                          ║\n");
    } else {
        for m in &manifest.models {
            out.push_str(&format!(
                "║  {:<12} — {:<44} ║\n",
                m.id,
                &m.ad[..m.ad.len().min(44)]
            ));
            let acik = if m.aciklama.len() > 58 {
                format!("{}…", &m.aciklama[..57])
            } else {
                m.aciklama.clone()
            };
            out.push_str(&format!("║               {acik:<58} ║\n"));
            if !m.etiketler.is_empty() {
                out.push_str(&format!(
                    "║               etiketler: {} ║\n",
                    m.etiketler.join(", ")
                ));
            }
        }
    }

    out.push_str("╠══════════════════════════════════════════════════════════════╣\n");
    out.push_str("║  model_indir(\"id\")  |  turk zoo indir <id>                   ║\n");
    out.push_str("╚══════════════════════════════════════════════════════════════╝\n");
    out
}

/// Tek model detay metni
pub fn model_bilgi_metin(id: &str) -> Result<String> {
    let meta = model_bul(id)?;
    let mut out = String::from("╔══════════════════════════════════════════════════╗\n");
    out.push_str(&format!("║  Model: {:<40} ║\n", meta.id));
    out.push_str("╠══════════════════════════════════════════════════╣\n");
    out.push_str(&format!("║  Ad        : {:<34} ║\n", meta.ad));
    out.push_str(&format!("║  Betik     : {:<34} ║\n", meta.turk_dosya));
    if let Some(v) = &meta.veri {
        out.push_str(&format!("║  Veri      : {:<34} ║\n", v));
    }
    out.push_str(&format!("║  Açıklama  : {:<34} ║\n", meta.aciklama));
    if !meta.etiketler.is_empty() {
        out.push_str(&format!(
            "║  Etiketler : {:<34} ║\n",
            meta.etiketler.join(", ")
        ));
    }
    out.push_str("╚══════════════════════════════════════════════════╝\n");
    Ok(out)
}

fn dizin_kopyala(kaynak: &Path, hedef: &Path) -> Result<()> {
    for entry in fs::read_dir(kaynak).map_err(|e| {
        TurkError::IOError(format!("Dizin okunamadı {}: {e}", kaynak.display()))
    })? {
        let entry = entry.map_err(|e| TurkError::IOError(e.to_string()))?;
        let tur = entry.path();
        let ad = tur.file_name().ok_or_else(|| {
            TurkError::IOError("Geçersiz dosya adı".into())
        })?;
        let hedef_yol = hedef.join(ad);
        if tur.is_dir() {
            fs::create_dir_all(&hedef_yol).map_err(|e| TurkError::IOError(e.to_string()))?;
            dizin_kopyala(&tur, &hedef_yol)?;
        } else {
            fs::copy(&tur, &hedef_yol).map_err(|e| {
                TurkError::IOError(format!(
                    "Kopyalanamadı {} → {}: {e}",
                    tur.display(),
                    hedef_yol.display()
                ))
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_yukle() {
        let m = manifest_yukle().unwrap();
        assert!(!m.models.is_empty());
        assert!(m.models.iter().any(|x| x.id == "xor"));
    }

    #[test]
    fn test_zoo_listesi() {
        let t = zoo_listesi_metin();
        assert!(t.contains("xor"));
    }

    #[test]
    fn test_model_indir_xor() {
        let tmp = tempfile::tempdir().unwrap();
        let yol = model_indir("xor", Some(tmp.path())).unwrap();
        assert!(yol.join("model.turk").is_file());
        assert!(yol.join("xor_data.csv").is_file());
    }
}

//! TÜRK otonom kod düzenleme — proje dosyalarını okuma, yazma ve enjeksiyon.

use super::sistem::proje_kok_dizini;
use std::path::{Path, PathBuf};

pub const MERHABA_DUNYA_RS: &str = r#"fn main() {
    println!("Merhaba dünya!");
}
"#;

/// Proje köküne göre dosya yolunu çözümler.
pub fn cozumle_yol(yol: &str) -> Result<PathBuf, String> {
    let yol = yol.trim().trim_matches('"');
    if yol.is_empty() {
        return Err("Dosya yolu boş.".into());
    }
    let kok = proje_kok_dizini();
    let hedef = Path::new(yol);
    let tam = if hedef.is_absolute() {
        hedef.to_path_buf()
    } else {
        kok.join(hedef)
    };
    Ok(tam)
}

/// Kaynak dosyasını okur.
pub fn dosya_oku(yol: &str) -> Result<String, String> {
    let path = cozumle_yol(yol)?;
    if !path.is_file() {
        return Err(format!("Dosya bulunamadı: {}", path.display()));
    }
    std::fs::read_to_string(&path).map_err(|e| format!("Okuma hatası ({}): {e}", path.display()))
}

/// Dosyaya yazar (üst dizinleri oluşturur).
pub fn dosya_yaz(yol: &str, icerik: &str) -> Result<(), String> {
    let path = cozumle_yol(yol)?;
    if let Some(ust) = path.parent() {
        if !ust.as_os_str().is_empty() {
            std::fs::create_dir_all(ust)
                .map_err(|e| format!("Dizin oluşturulamadı ({}): {e}", ust.display()))?;
        }
    }
    std::fs::write(&path, icerik).map_err(|e| format!("Yazma hatası ({}): {e}", path.display()))
}

/// Hedef satır/fonksiyon bulunduktan sonra altına yeni kod enjekte eder.
pub fn kod_enjekte_et(yol: &str, hedef_satir: &str, yeni_kod: &str) -> Result<String, String> {
    let hedef_satir = hedef_satir.trim();
    if hedef_satir.is_empty() {
        return Err("Enjeksiyon hedefi boş.".into());
    }

    let icerik = dosya_oku(yol)?;
    let hedef_kucuk = hedef_satir.to_lowercase();
    let mut satirlar: Vec<String> = icerik.lines().map(String::from).collect();

    let idx = satirlar
        .iter()
        .position(|l| l.to_lowercase().contains(&hedef_kucuk))
        .ok_or_else(|| format!("Hedef bulunamadı: '{hedef_satir}'"))?;

    for (i, kod_satir) in yeni_kod.lines().enumerate() {
        satirlar.insert(idx + 1 + i, kod_satir.to_string());
    }

    let yeni_icerik = if icerik.ends_with('\n') {
        format!("{}\n", satirlar.join("\n"))
    } else {
        satirlar.join("\n")
    };

    dosya_yaz(yol, &yeni_icerik)?;

    Ok(format!(
        "▸ [KOD] enjekte edildi: {yol} (satır {} sonrası, {} satır eklendi)",
        idx + 1,
        yeni_kod.lines().count()
    ))
}

/// Kod yazma aracı — dosya oluşturur veya üzerine yazar.
pub fn kod_yaz_uygula(yol: &str, icerik: &str, enjekte_hedef: Option<&str>) -> Result<String, String> {
    if let Some(hedef) = enjekte_hedef.filter(|h| !h.trim().is_empty()) {
        return kod_enjekte_et(yol, hedef, icerik);
    }

    let path = cozumle_yol(yol)?;
    let vardi = path.is_file();
    dosya_yaz(yol, icerik)?;
    let karakter = icerik.chars().count();
    if vardi {
        Ok(format!(
            "▸ [KOD] dosya güncellendi: {yol} ({karakter} karakter)\n\
             ✓ [KOD] kod_ustasi yazma tamamlandı"
        ))
    } else {
        Ok(format!(
            "▸ [KOD] yeni dosya oluşturuldu: {yol} ({karakter} karakter)\n\
             ✓ [KOD] kod_ustasi yazma tamamlandı"
        ))
    }
}

/// Doğal dil girdisinden Rust dosya yolu çıkarır.
pub fn rs_yol_ayikla(metin: &str) -> Option<String> {
    if let Some(start) = metin.find('"') {
        if let Some(end) = metin[start + 1..].find('"') {
            let y = &metin[start + 1..start + 1 + end];
            if y.contains(".rs") {
                return Some(y.to_string());
            }
        }
    }
    for parca in metin.split_whitespace() {
        let temiz = parca.trim_matches(|c: char| c == ',' || c == '.' || c == ';');
        if temiz.contains(".rs") {
            return Some(temiz.to_string());
        }
        if temiz.contains('/') || temiz.contains('\\') {
            if temiz.ends_with(".rs") || !temiz.contains('.') {
                continue;
            }
        }
    }
    None
}

/// Merhaba dünya veya benzeri istekler için varsayılan içerik.
pub fn varsayilan_rust_icerik(metin: &str) -> Option<String> {
    let k = metin.to_lowercase();
    if (k.contains("merhaba") && (k.contains("dünya") || k.contains("dunya")))
        || k.contains("hello world")
    {
        return Some(MERHABA_DUNYA_RS.to_string());
    }
    if k.contains("rust dosya") || k.contains(".rs") {
        return Some(MERHABA_DUNYA_RS.to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_kod_enjekte_et() {
        let tmp = std::env::temp_dir().join("turk_kod_test.rs");
        let _ = fs::write(&tmp, "fn main() {\n}\n");
        let yol = tmp.to_string_lossy();
        let sonuc = kod_enjekte_et(&yol, "fn main", "    println!(\"test\");");
        assert!(sonuc.is_ok());
        let icerik = fs::read_to_string(&tmp).unwrap();
        assert!(icerik.contains("println!(\"test\")"));
        let _ = fs::remove_file(tmp);
    }
}

//! Yerel Whisper STT — whisper.cpp CLI ile Türkçe konuşmayı metne çevirir.

use crate::otomasyon::turk_sistem_veri_dizini;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const HEDEF_HZ: u32 = 16_000;

/// Whisper GGML model dosyası yolu.
pub fn whisper_model_yolu() -> PathBuf {
    if let Ok(p) = std::env::var("TURK_WHISPER_MODEL") {
        return PathBuf::from(p);
    }
    turk_sistem_veri_dizini()
        .join("models")
        .join("ggml-small.bin")
}

/// whisper.cpp CLI yolu (`whisper-cli.exe` veya `main.exe`).
pub fn whisper_cli_yolu() -> PathBuf {
    if let Ok(p) = std::env::var("TURK_WHISPER_CLI") {
        return PathBuf::from(p);
    }
    let bin = turk_sistem_veri_dizini().join("bin");
    for ad in ["whisper-cli.exe", "whisper-cli", "main.exe", "main"] {
        let yol = bin.join(ad);
        if yol.is_file() {
            return yol;
        }
    }
    // PATH'te ara
    if let Ok(path_env) = std::env::var("PATH") {
        for ad in ["whisper-cli.exe", "whisper-cli", "main.exe"] {
            for dir in std::env::split_paths(&path_env) {
                let yol = dir.join(ad);
                if yol.is_file() {
                    return yol;
                }
            }
        }
    }
    bin.join("whisper-cli.exe")
}

/// Whisper bileşenlerinin kurulu olup olmadığını kontrol eder.
pub fn whisper_hazir_mi() -> bool {
    whisper_cli_yolu().is_file() && whisper_model_yolu().is_file()
}

fn yeniden_ornekle(ornekler: &[f32], kaynak_hz: u32, hedef_hz: u32) -> Vec<f32> {
    if kaynak_hz == hedef_hz || ornekler.is_empty() {
        return ornekler.to_vec();
    }
    let oran = hedef_hz as f64 / kaynak_hz as f64;
    let cikti_uzunluk = ((ornekler.len() as f64) * oran).ceil() as usize;
    (0..cikti_uzunluk)
        .map(|i| {
            let src = i as f64 / oran;
            let idx = src as usize;
            let frac = (src - idx as f64) as f32;
            let a = ornekler.get(idx).copied().unwrap_or(0.0);
            let b = ornekler.get(idx + 1).copied().unwrap_or(a);
            a * (1.0 - frac) + b * frac
        })
        .collect()
}

fn wav_yaz(yol: &Path, ornekler: &[f32], hz: u32) -> Result<(), String> {
    if let Some(ust) = yol.parent() {
        std::fs::create_dir_all(ust).map_err(|e| format!("Geçici dizin oluşturulamadı: {e}"))?;
    }
    let spec = WavSpec {
        channels: 1,
        sample_rate: hz,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(yol, spec).map_err(|e| format!("WAV yazılamadı: {e}"))?;
    for &s in ornekler {
        let clipped = s.clamp(-1.0, 1.0);
        let i16 = (clipped * i16::MAX as f32) as i16;
        writer
            .write_sample(i16)
            .map_err(|e| format!("WAV örneği yazılamadı: {e}"))?;
    }
    writer.finalize().map_err(|e| format!("WAV kapatılamadı: {e}"))?;
    Ok(())
}

fn cli_hazir_mi() -> Result<PathBuf, String> {
    let cli = whisper_cli_yolu();
    if !cli.is_file() {
        return Err(format!(
            "Whisper CLI bulunamadı: {}\n\
             Kurulum: .\\whisper-kur.ps1 çalıştırın\n\
             veya whisper-cli.exe dosyasını ~/.turk/sistem/bin/ dizinine kopyalayın.",
            cli.display()
        ));
    }
    Ok(cli)
}

fn model_hazir_mi() -> Result<PathBuf, String> {
    let yol = whisper_model_yolu();
    if !yol.is_file() {
        return Err(format!(
            "Whisper modeli bulunamadı: {}\n\
             ggml-small.bin dosyasını ~/.turk/sistem/models/ altına indirin\n\
             veya TURK_WHISPER_MODEL ortam değişkenini ayarlayın.",
            yol.display()
        ));
    }
    Ok(yol)
}

/// PCM kaydını Türkçe metne çevirir (16 kHz mono → whisper.cpp).
pub fn konusmayi_metne_cevir(ornekler: &[f32], ornekleme_hz: u32) -> Result<String, String> {
    if ornekler.len() < ornekleme_hz as usize / 20 {
        let sn = ornekler.len() as f32 / ornekleme_hz as f32;
        return Err(format!(
            "Kayıt çok kısa ({sn:.2} sn) — 🎙️ yeşil yanınca en az 0.5 sn konuşun."
        ));
    }

    let cli = cli_hazir_mi()?;
    let model = model_hazir_mi()?;
    let pcm = yeniden_ornekle(ornekler, ornekleme_hz, HEDEF_HZ);

    let tmp = turk_sistem_veri_dizini().join("tmp");
    let wav = tmp.join(format!("turk-ses-{}.wav", std::process::id()));
    wav_yaz(&wav, &pcm, HEDEF_HZ)?;

    let cikti = tmp.join(format!("turk-ses-{}.txt", std::process::id()));

    let sonuc = Command::new(&cli)
        .current_dir(cli.parent().unwrap_or_else(|| Path::new(".")))
        .arg("-m")
        .arg(&model)
        .arg("-f")
        .arg(&wav)
        .arg("-l")
        .arg("tr")
        .arg("-nt")
        .arg("-of")
        .arg(cikti.with_extension(""))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Whisper CLI çalıştırılamadı: {e}"));

    let _ = std::fs::remove_file(&wav);

    let out = sonuc?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&out.stdout).trim().to_string();
        let detay = if !stderr.is_empty() { stderr } else { stdout };
        return Err(if detay.is_empty() {
            "Whisper çıkış kodu sıfır değil — DLL eksik olabilir, .\\whisper-kur.ps1 tekrar çalıştırın.".into()
        } else {
            format!("Whisper hatası: {detay}")
        });
    }

    let metin = if cikti.is_file() {
        std::fs::read_to_string(&cikti).unwrap_or_default()
    } else {
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let _ = std::fs::remove_file(&cikti);

    let metin = metin.trim().to_string();
    if metin.is_empty() {
        return Err("Konuşma algılanamadı — tekrar deneyin.".into());
    }
    Ok(metin)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yeniden_ornekle() {
        let src: Vec<f32> = (0..480).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = yeniden_ornekle(&src, 48_000, 16_000);
        assert_eq!(out.len(), 160);
    }

    #[test]
    fn test_wav_yaz() {
        let dir = std::env::temp_dir().join("turk-ses-test");
        let wav = dir.join("test.wav");
        let ornekler = vec![0.0f32; 1600];
        wav_yaz(&wav, &ornekler, 16_000).expect("wav");
        assert!(wav.is_file());
        let _ = std::fs::remove_file(&wav);
    }
}

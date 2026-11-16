//! İşletim sistemi otomasyonu — dosya, süreç, sistem metrikleri.

use crate::error::{Result, TurkError};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use sysinfo::{MINIMUM_CPU_UPDATE_INTERVAL, System};

/// Task Manager benzeri yumuşatma penceresi
const CPU_GUNCELLEME_ARALIGI: Duration = Duration::from_millis(1000);
const CPU_EMA_ALPHA: f32 = 0.25;

struct CpuOlcumDurumu {
    #[cfg(not(windows))]
    sys: System,
    son_olcum: Instant,
    yuzde: f32,
    baslatildi: bool,
    ornek_sayisi: u32,
}

fn cpu_olcum_durumu() -> &'static Mutex<CpuOlcumDurumu> {
    static DURUM: OnceLock<Mutex<CpuOlcumDurumu>> = OnceLock::new();
    DURUM.get_or_init(|| {
        #[cfg(windows)]
        {
            super::win_cpu::isin();
            Mutex::new(CpuOlcumDurumu {
                son_olcum: Instant::now(),
                yuzde: 0.0,
                baslatildi: false,
                ornek_sayisi: 0,
            })
        }
        #[cfg(not(windows))]
        {
            let mut sys = System::new_all();
            sys.refresh_cpu();
            Mutex::new(CpuOlcumDurumu {
                sys,
                son_olcum: Instant::now(),
                yuzde: 0.0,
                baslatildi: false,
                ornek_sayisi: 0,
            })
        }
    })
}

#[cfg(not(windows))]
fn cpu_ham_deger(sys: &System) -> f32 {
    let cpus = sys.cpus();
    if cpus.is_empty() {
        sys.global_cpu_info().cpu_usage()
    } else {
        cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32
    }
}

fn cpu_ornek_guncelle(d: &mut CpuOlcumDurumu) {
    #[cfg(windows)]
    let ham = {
        if let Some(pct) = super::win_cpu::cpu_yuzde_ornek() {
            pct
        } else {
            return;
        }
    };
    #[cfg(not(windows))]
    let ham = {
        d.sys.refresh_cpu();
        cpu_ham_deger(&d.sys).clamp(0.0, 100.0)
    };

    d.ornek_sayisi += 1;
    if d.ornek_sayisi == 1 {
        d.yuzde = ham;
    } else {
        d.yuzde = d.yuzde * (1.0 - CPU_EMA_ALPHA) + ham * CPU_EMA_ALPHA;
    }
    d.son_olcum = Instant::now();
    d.baslatildi = true;
}

fn cpu_guncelleme_araligi() -> Duration {
    CPU_GUNCELLEME_ARALIGI.max(MINIMUM_CPU_UPDATE_INTERVAL)
}

/// GUI döngüsünden çağrılır — paylaşımlı CPU önbelleğini günceller.
pub fn cpu_kullanim_guncelle() -> f32 {
    let mut d = cpu_olcum_durumu().lock().expect("cpu ölçüm kilidi");
    if !d.baslatildi {
        #[cfg(windows)]
        super::win_cpu::isin();
        #[cfg(not(windows))]
        d.sys.refresh_cpu();
        d.baslatildi = true;
        d.son_olcum = Instant::now();
        return 0.0;
    }
    if d.son_olcum.elapsed() >= cpu_guncelleme_araligi() {
        cpu_ornek_guncelle(&mut d);
    }
    d.yuzde
}

/// Sohbet araçları — önbelleği okur, yeni ölçüm tetiklemez.
pub fn cpu_kullanim_oku() -> f32 {
    cpu_olcum_durumu()
        .lock()
        .map(|d| d.yuzde)
        .unwrap_or(0.0)
}

fn cargo_komutu() -> Command {
    #[cfg(windows)]
    if let Ok(home) = std::env::var("USERPROFILE") {
        let cargo = PathBuf::from(home)
            .join(".cargo")
            .join("bin")
            .join("cargo.exe");
        if cargo.exists() {
            return Command::new(cargo);
        }
    }
    Command::new("cargo")
}

/// Anlık sistem durumu özeti
#[derive(Debug, Clone)]
pub struct SistemDurumu {
    pub cpu_yuzde: f32,
    pub ram_kullanilan_gb: f64,
    pub ram_toplam_gb: f64,
    pub ram_yuzde: f32,
    pub disk_yuzde: Option<f32>,
    pub os_ad: String,
}

/// Dizindeki dosya ve klasörleri listele (hata durumunda `Result`)
pub fn dosya_listele(yol: &Path) -> Result<Vec<String>> {
    if !yol.exists() {
        return Err(TurkError::IOError(format!(
            "Yol bulunamadı: {}",
            yol.display()
        )));
    }

    let mut girdiler: Vec<String> = std::fs::read_dir(yol)
        .map_err(|e| TurkError::IOError(format!("{} okunamadı: {e}", yol.display())))?
        .filter_map(|e| e.ok())
        .map(|e| {
            let ad = e.file_name().to_string_lossy().into_owned();
            let tur = if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                "[klasör]"
            } else {
                "[dosya]"
            };
            format!("{tur} {ad}")
        })
        .collect();

    girdiler.sort();
    Ok(girdiler)
}

/// Belirtilen dizindeki dosya/klasör adlarını listeler (TÜRK otomasyon API)
pub fn dosyalari_listele(yol: &str) -> Vec<String> {
    match dosya_listele(Path::new(yol)) {
        Ok(liste) => liste,
        Err(e) => vec![format!("Hata: {e}")],
    }
}

/// Metni dosyaya yazar
pub fn dosya_yaz(yol: &str, icerik: &str) -> String {
    if yol.trim().is_empty() {
        return "Hata: dosya yolu boş.".into();
    }
    match std::fs::write(yol, icerik) {
        Ok(()) => format!(
            "Dosya kaydedildi: {yol} ({} karakter)",
            icerik.chars().count()
        ),
        Err(e) => format!("Yazma hatası ({yol}): {e}"),
    }
}

/// Not/dosya oluşturur — üst dizinleri açar, uzantı yoksa `.txt` ekler.
pub fn dosya_olustur(yol: &str, icerik: &str) -> String {
    let hedef = if yol.trim().is_empty() {
        varsayilan_not_yolu()
    } else {
        dosya_yolunu_hazirla(yol)
    };

    if let Some(ust) = Path::new(&hedef).parent() {
        if !ust.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(ust) {
                return format!("Dizin oluşturulamadı ({}): {e}", ust.display());
            }
        }
    }

    match std::fs::write(&hedef, icerik) {
        Ok(()) => format!(
            "✓ Dosya oluşturuldu: {hedef} ({} karakter)",
            icerik.chars().count()
        ),
        Err(e) => format!("Oluşturma hatası ({hedef}): {e}"),
    }
}

fn dosya_yolunu_hazirla(yol: &str) -> String {
    let yol = yol.trim().trim_matches('"');
    let path = Path::new(yol);
    if path.extension().is_some() {
        yol.to_string()
    } else {
        format!("{yol}.txt")
    }
}

fn varsayilan_not_yolu() -> String {
    let dir = turk_sistem_veri_dizini().join("notlar");
    let saniye = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    dir.join(format!("not_{saniye}.txt"))
        .to_string_lossy()
        .into_owned()
}

/// Proje kök dizinini bulur (`Cargo.toml` olan dizin).
pub fn proje_kok_dizini() -> PathBuf {
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest);
    }
    if let Ok(cwd) = std::env::current_dir() {
        let mut cur = cwd.as_path();
        loop {
            if cur.join("Cargo.toml").is_file() {
                return cur.to_path_buf();
            }
            cur = match cur.parent() {
                Some(p) => p,
                None => break,
            };
        }
        return cwd;
    }
    PathBuf::from(".")
}

pub const PROJE_KONTROL_TEMIZ_MESAJ: &str =
    "Projede herhangi bir Rust hatası bulunamadı, kanka! 🏎️";

/// `cargo check --message-format=json` — hataları satır ve açıklama ile raporlar.
pub fn proje_kontrol(yol: &str) -> String {
    let dir = if yol.trim().is_empty() || yol == "." {
        proje_kok_dizini()
    } else {
        PathBuf::from(yol)
    };

    if !dir.is_dir() {
        return format!("Proje dizini bulunamadı: {}", dir.display());
    }

    if !dir.join("Cargo.toml").is_file() {
        return format!(
            "Cargo.toml bulunamadı: {} — TÜRK proje kök dizininde değil.",
            dir.display()
        );
    }

    let cikti = cargo_komutu()
        .args(["check", "--message-format=json", "--features", "turk_sistem"])
        .current_dir(&dir)
        .output();

    match cikti {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let birlesik = format!("{stdout}{stderr}");
            let hatalar = cargo_json_hatalari(&birlesik);

            if out.status.success() && hatalar.is_empty() {
                return format!(
                    "✓ {}\n(Dizin: {})",
                    PROJE_KONTROL_TEMIZ_MESAJ,
                    dir.display()
                );
            }

            if hatalar.is_empty() {
                return format!(
                    "✗ Proje kontrolü başarısız ({}), ancak JSON çıktısı ayrıştırılamadı:\n{}",
                    dir.display(),
                    stderr.trim()
                );
            }

            let mut rapor = format!(
                "✗ {} Rust hatası bulundu ({}):\n\n",
                hatalar.len(),
                dir.display()
            );
            for (i, (dosya, satir, mesaj)) in hatalar.iter().enumerate() {
                rapor.push_str(&format!("{}. {dosya}:{satir}\n   → {mesaj}\n", i + 1));
            }
            rapor
        }
        Err(e) => format!("cargo çalıştırılamadı: {e}"),
    }
}

fn cargo_json_hatalari(cikti: &str) -> Vec<(String, u32, String)> {
    use serde_json::Value;
    use std::collections::HashSet;

    let mut hatalar = Vec::new();
    let mut gorulen = HashSet::new();

    for satir in cikti.lines() {
        let satir = satir.trim();
        if satir.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(satir) else {
            continue;
        };
        if v.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
            continue;
        }
        let Some(msg) = v.get("message") else {
            continue;
        };
        if msg.get("level").and_then(|l| l.as_str()) != Some("error") {
            continue;
        }
        let aciklama = msg
            .get("message")
            .and_then(|m| m.as_str())
            .or_else(|| msg.get("rendered").and_then(|m| m.as_str()))
            .unwrap_or("bilinmeyen hata")
            .lines()
            .next()
            .unwrap_or("bilinmeyen hata")
            .to_string();
        let (dosya, satir_no) = msg
            .get("spans")
            .and_then(|s| s.as_array())
            .and_then(|arr| arr.first())
            .map(|span| {
                (
                    span.get("file_name")
                        .and_then(|f| f.as_str())
                        .unwrap_or("?")
                        .to_string(),
                    span.get("line_start")
                        .and_then(|l| l.as_u64())
                        .unwrap_or(0) as u32,
                )
            })
            .unwrap_or_else(|| ("?".into(), 0));

        let anahtar = (dosya.clone(), satir_no, aciklama.clone());
        if gorulen.insert(anahtar) {
            hatalar.push((dosya, satir_no, aciklama));
        }
    }
    hatalar
}

/// Proje dizininde `cargo check` çalıştırır (hızlı derleme doğrulaması)
pub fn proje_derle(yol: &str) -> String {
    let yol = if yol.trim().is_empty() { "." } else { yol };
    let dir = Path::new(yol);
    if !dir.is_dir() {
        return format!("Proje dizini bulunamadı: {yol}");
    }

    let cikti = cargo_komutu()
        .args(["check", "--features", "turk_sistem"])
        .current_dir(dir)
        .output();

    match cikti {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let birlesik = if stderr.is_empty() {
                stdout.into_owned()
            } else {
                format!("{stdout}\n{stderr}")
            };
            if out.status.success() {
                format!("✓ Proje derlemesi başarılı ({yol}):\n{birlesik}")
            } else {
                format!("✗ Derleme hatası ({yol}):\n{birlesik}")
            }
        }
        Err(e) => format!("cargo çalıştırılamadı: {e}"),
    }
}

/// Shell komutu çalıştırır ve stdout/stderr çıktısını döndürür
pub fn terminal_komutu_calistir(komut: &str) -> String {
    match surec_baslat(komut) {
        Ok(cikti) => cikti,
        Err(e) => format!("Komut hatası: {e}"),
    }
}

/// Shell komutu başlat (geliştirici otomasyonu) — `Result` sürümü
pub fn surec_baslat(komut: &str) -> Result<String> {
    let komut = komut.trim();
    if komut.is_empty() {
        return Err(TurkError::ValueError("Boş komut.".into()));
    }

    #[cfg(windows)]
    let cikti = Command::new("cmd")
        .args(["/C", komut])
        .output()
        .map_err(|e| TurkError::RuntimeError(format!("Süreç başlatılamadı: {e}")))?;

    #[cfg(not(windows))]
    let cikti = Command::new("sh")
        .args(["-c", komut])
        .output()
        .map_err(|e| TurkError::RuntimeError(format!("Süreç başlatılamadı: {e}")))?;

    let stdout = String::from_utf8_lossy(&cikti.stdout);
    let stderr = String::from_utf8_lossy(&cikti.stderr);

    if cikti.status.success() {
        Ok(if stdout.is_empty() {
            "Komut başarıyla tamamlandı.".into()
        } else {
            stdout.into_owned()
        })
    } else {
        Err(TurkError::RuntimeError(format!(
            "Komut hata kodu {}: {stderr}",
            cikti.status.code().unwrap_or(-1)
        )))
    }
}

/// CPU, RAM ve disk kullanımını ölç
pub fn sistem_durumu() -> SistemDurumu {
    use sysinfo::Disks;

    let cpu_yuzde = cpu_kullanim_oku();

    let mut sys = System::new_all();
    sys.refresh_memory();
    let ram_toplam = sys.total_memory() as f64 / 1_073_741_824.0;
    let ram_kullanilan = sys.used_memory() as f64 / 1_073_741_824.0;
    let ram_yuzde = if sys.total_memory() > 0 {
        (sys.used_memory() as f32 / sys.total_memory() as f32) * 100.0
    } else {
        0.0
    };

    let disk_yuzde = Disks::new_with_refreshed_list()
        .iter()
        .find(|d| d.mount_point().to_str() == Some("/") || {
            #[cfg(windows)]
            {
                d.mount_point().to_str() == Some("C:\\")
            }
            #[cfg(not(windows))]
            {
                false
            }
        })
        .map(|d| {
            let total = d.total_space();
            if total > 0 {
                ((total - d.available_space()) as f32 / total as f32) * 100.0
            } else {
                0.0
            }
        });

    let os_ad = System::name().unwrap_or_else(|| "Bilinmiyor".into());

    SistemDurumu {
        cpu_yuzde,
        ram_kullanilan_gb: ram_kullanilan,
        ram_toplam_gb: ram_toplam,
        ram_yuzde,
        disk_yuzde,
        os_ad,
    }
}

/// TÜRK veri dizini (~/.turk/sistem)
pub fn turk_sistem_veri_dizini() -> PathBuf {
    if let Ok(home) = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")) {
        return PathBuf::from(home).join(".turk").join("sistem");
    }
    PathBuf::from(".turk").join("sistem")
}

/// Geriye uyumluluk — `turk_sistem_veri_dizini` kullanın.
#[deprecated(note = "turk_sistem_veri_dizini kullanın")]
pub fn jarvis_veri_dizini() -> PathBuf {
    turk_sistem_veri_dizini()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sistem_durumu() {
        let d = sistem_durumu();
        assert!(d.ram_toplam_gb > 0.0);
    }

    #[test]
    fn test_dosyalari_listele() {
        let liste = dosyalari_listele(".");
        assert!(!liste.is_empty());
    }

    #[test]
    fn test_dosya_olustur() {
        let dir = std::env::temp_dir().join("turk_test_not");
        let _ = std::fs::create_dir_all(&dir);
        let yol = dir.join("test_not.txt");
        let _ = std::fs::remove_file(&yol);
        let sonuc = dosya_olustur(yol.to_str().unwrap(), "TÜRK test");
        assert!(sonuc.contains("oluşturuldu"));
        assert!(yol.exists());
        let _ = std::fs::remove_file(yol);
    }

    #[test]
    fn test_cargo_json_hatalari() {
        let ornek = r#"{"reason":"compiler-message","message":{"level":"error","message":"expected `;`","spans":[{"file_name":"src/main.rs","line_start":10,"column_start":5}]}}"#;
        let h = cargo_json_hatalari(ornek);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].0, "src/main.rs");
        assert_eq!(h[0].1, 10);
    }
}

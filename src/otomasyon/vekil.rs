//! Proaktif sistem koruyucusu — TÜRK Vekil arka plan dinleyicisi ve OS süreç yönetimi.

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use sysinfo::{Disks, ProcessRefreshKind, System};
use tokio::sync::mpsc;
use tokio::time::{self, Duration};

const KONTROL_ARALIGI: Duration = Duration::from_secs(5);
const CPU_ESIK: f32 = 90.0;
const RAM_ESIK: f32 = 90.0;
const DISK_ESIK: f32 = 95.0;
const SUREC_CPU_ESIK: f32 = 25.0;
const SUREC_RAM_YUZDE_ESIK: f32 = 5.0;
const MAX_SONLANDIRMA: usize = 3;

/// Yük eşiği ilk kez aşıldığında tek uyarı — mükerrer mesajı engeller.
static YUK_UYARISI_AKTIF: AtomicBool = AtomicBool::new(false);

/// Otonom mod — onay beklemeden süreç optimizasyonu.
static OTONOM_MOD_AKTIF: AtomicBool = AtomicBool::new(false);

/// GUI → Vekil komut kanalı
#[derive(Debug, Clone)]
pub enum VekilKomut {
    OtonomMod(bool),
    OptimizasyonOnayla,
}

/// Vekil → GUI olay kanalı
#[derive(Debug, Clone)]
pub enum VekilOlay {
    /// CPU/RAM/disk eşiği ilk kez aşıldı
    KritikYuk { cpu: f32, ram: f32, disk: f32 },
    /// Metrikler normale döndü — flag sıfırlandı
    YukNormaleDondu,
    /// OS seviyesi süreç optimizasyonu tamamlandı
    OptimizasyonTamamlandi { sonlandirilan: Vec<String> },
}

/// Arka planda sistem metriklerini izleyen TÜRK vekili.
pub struct SistemVekili;

impl SistemVekili {
    /// Tokio döngüsünü ayrı thread'de başlatır; GUI donmaz.
    /// Dönüş: (komut gönderici, olay alıcı)
    pub fn baslat() -> (mpsc::Sender<VekilKomut>, mpsc::Receiver<VekilOlay>) {
        let (tx_olay, rx_olay) = mpsc::channel(16);
        let (tx_komut, rx_komut) = mpsc::channel(16);

        std::thread::Builder::new()
            .name("turk-vekil".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("vekil tokio runtime");
                rt.block_on(vekil_dongusu(tx_olay, rx_komut));
            })
            .expect("vekil thread");

        (tx_komut, rx_olay)
    }

    pub fn otonom_mod_ayarla(aktif: bool) {
        OTONOM_MOD_AKTIF.store(aktif, Ordering::SeqCst);
    }

    pub fn otonom_mod_aktif() -> bool {
        OTONOM_MOD_AKTIF.load(Ordering::SeqCst)
    }
}

async fn vekil_dongusu(tx: mpsc::Sender<VekilOlay>, mut rx_komut: mpsc::Receiver<VekilKomut>) {
    let mut sys = System::new_all();

    loop {
        tokio::select! {
            _ = time::sleep(KONTROL_ARALIGI) => {
                sys.refresh_cpu();
                sys.refresh_memory();

                let cpu = cpu_ortalama(&sys);
                let ram = ram_yuzde(&sys);
                let disk = ana_disk_yuzde();
                let kritik = cpu >= CPU_ESIK || ram >= RAM_ESIK || disk >= DISK_ESIK;

                if kritik {
                    if !YUK_UYARISI_AKTIF.swap(true, Ordering::SeqCst) {
                        if OTONOM_MOD_AKTIF.load(Ordering::SeqCst) {
                            optimizasyon_calistir(&tx).await;
                        } else if tx
                            .send(VekilOlay::KritikYuk { cpu, ram, disk })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                } else if YUK_UYARISI_AKTIF.swap(false, Ordering::SeqCst) {
                    if tx.send(VekilOlay::YukNormaleDondu).await.is_err() {
                        break;
                    }
                }
            }
            Some(komut) = rx_komut.recv() => {
                match komut {
                    VekilKomut::OtonomMod(aktif) => {
                        OTONOM_MOD_AKTIF.store(aktif, Ordering::SeqCst);
                    }
                    VekilKomut::OptimizasyonOnayla => {
                        optimizasyon_calistir(&tx).await;
                    }
                }
            }
        }
    }
}

async fn optimizasyon_calistir(tx: &mpsc::Sender<VekilOlay>) {
    let sonlandirilan = match std::thread::Builder::new()
        .name("turk-os-kontrol".into())
        .spawn(os_seviyesi_kontrol)
    {
        Ok(handle) => handle.join().ok().and_then(|r| r.ok()).unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    let _ = tx
        .send(VekilOlay::OptimizasyonTamamlandi { sonlandirilan })
        .await;
}

/// Windows/Linux süreç listesini tarayıp aşırı kaynak tüketenleri sonlandırır.
pub fn os_seviyesi_kontrol() -> Result<Vec<String>, String> {
    let mut sys = System::new_all();
    sys.refresh_all();
    sys.refresh_processes_specifics(ProcessRefreshKind::new().with_cpu().with_memory());

    let toplam_ram = sys.total_memory().max(1);
    let mut adaylar: Vec<(u32, String, f32, f32)> = Vec::new();

    for (pid, proc) in sys.processes() {
        let ad = proc.name().to_string();
        if surec_korumali(&ad) {
            continue;
        }
        let cpu = proc.cpu_usage();
        let ram_yuzde = (proc.memory() as f64 / toplam_ram as f64) * 100.0;
        if cpu >= SUREC_CPU_ESIK || ram_yuzde >= SUREC_RAM_YUZDE_ESIK as f64 {
            adaylar.push((pid.as_u32(), ad, cpu, ram_yuzde as f32));
        }
    }

    adaylar.sort_by(|a, b| {
        (b.2 + b.3)
            .partial_cmp(&(a.2 + a.3))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sonlandirilan = Vec::new();
    for (pid, ad, _, _) in adaylar.into_iter().take(MAX_SONLANDIRMA) {
        if surec_sonlandir(pid).is_ok() {
            sonlandirilan.push(format!("{ad} (PID {pid})"));
        }
    }

    Ok(sonlandirilan)
}

fn surec_korumali(ad: &str) -> bool {
    let k = ad.to_lowercase();
    const KORUMALI: &[&str] = &[
        "system",
        "idle",
        "registry",
        "csrss",
        "wininit",
        "services",
        "lsass",
        "smss",
        "dwm",
        "explorer",
        "turk_sistem",
        "turk",
        "cargo",
        "rustc",
        "rust-analyzer",
        "msmpeng",
        "winlogon",
        "fontdrvhost",
        "sihost",
        "taskhostw",
        "conhost",
    ];
    KORUMALI.iter().any(|p| k.contains(p))
}

#[cfg(windows)]
fn surec_sonlandir(pid: u32) -> Result<(), String> {
    let out = Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .output()
        .map_err(|e| format!("taskkill başlatılamadı: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "taskkill başarısız: {}",
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

#[cfg(not(windows))]
fn surec_sonlandir(pid: u32) -> Result<(), String> {
    let out = Command::new("kill")
        .args(["-9", &pid.to_string()])
        .output()
        .map_err(|e| format!("kill başlatılamadı: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(format!(
            "kill başarısız: {}",
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

fn cpu_ortalama(sys: &System) -> f32 {
    let cpus = sys.cpus();
    if cpus.is_empty() {
        sys.global_cpu_info().cpu_usage()
    } else {
        cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32
    }
    .clamp(0.0, 100.0)
}

fn ram_yuzde(sys: &System) -> f32 {
    let toplam = sys.total_memory();
    if toplam == 0 {
        return 0.0;
    }
    (sys.used_memory() as f32 / toplam as f32) * 100.0
}

fn ana_disk_yuzde() -> f32 {
    Disks::new_with_refreshed_list()
        .iter()
        .find(|d| {
            #[cfg(windows)]
            {
                d.mount_point().to_str() == Some("C:\\")
            }
            #[cfg(not(windows))]
            {
                d.mount_point().to_str() == Some("/")
            }
        })
        .map(|d| {
            let total = d.total_space();
            if total > 0 {
                ((total - d.available_space()) as f32 / total as f32) * 100.0
            } else {
                0.0
            }
        })
        .unwrap_or(0.0)
}

/// Sohbet paneline düşecek sabit TÜRK vekil uyarı metni.
pub const TURK_VEKIL_UYARI_MESAJ: &str = "\
[TÜRK Uyarısı]: Sinan, sistemde yüksek yük tespit ettim. \
Arka plandaki bazı süreçleri veya tarayıcı sekmelerini optimize etmemi ister misin? 🚨";

/// Süreç optimizasyonu tamamlandığında panel logu.
pub const TURK_VEKIL_OPTIMIZE_MESAJ: &str = "\
🚨 [TÜRK Vekil]: Kritik sistem yükü nedeniyle arka plan süreçleri otonom olarak optimize edildi, Sinan kanka!";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surec_korumali() {
        assert!(surec_korumali("turk_sistem.exe"));
        assert!(surec_korumali("csrss.exe"));
        assert!(!surec_korumali("chrome.exe"));
    }
}

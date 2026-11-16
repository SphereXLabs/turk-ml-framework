//! Hesaplama cihazı soyutlaması (CPU / Paralel).
//!
//! `--features parallel` (varsayilan) ile `cihaz=paralel` secilebilir; bu mod Rayon
//! ile paralel CPU matris islemlerini etkinlestirir. Gercek GPU (CUDA/wgpu) sonraki
//! sürümde `cihaz=gpu` olarak eklenecektir.

use crate::error::{Result, TurkError};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Hesaplama cihazı
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Cihaz {
    #[default]
    Cpu,
    /// Rayon ile cok cekirdekli CPU (varsayilan derlemede aktif)
    #[serde(alias = "gpu")]
    Paralel,
}

impl Cihaz {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "cpu" | "islemci" => Ok(Cihaz::Cpu),
            // "gpu" simdilik paralel CPU icin geriye uyumluluk alias'i
            "paralel" | "parallel" | "multi-thread" | "gpu" => {
                if paralel_secilebilir() {
                    Ok(Cihaz::Paralel)
                } else {
                    Err(TurkError::RuntimeError(
                        "Paralel mod bu derlemede yok. \
                         Derleyin: cargo build  \
                         veya .\\cargo.ps1 build  \
                         (tek cekirdek icin: --no-default-features --features csv)"
                            .into(),
                    ))
                }
            }
            "cuda" | "ekran_karti" => Err(TurkError::RuntimeError(
                "Gercek GPU (CUDA) henuz desteklenmiyor. Simdilik: cihaz=paralel".into(),
            )),
            other => Err(TurkError::SyntaxError(format!(
                "Bilinmeyen cihaz: '{other}'. Gecerli: cpu, paralel"
            ))),
        }
    }

    pub fn ad(self) -> &'static str {
        match self {
            Cihaz::Cpu => "CPU",
            Cihaz::Paralel => "Paralel CPU",
        }
    }

    pub fn kisa(self) -> &'static str {
        match self {
            Cihaz::Cpu => "cpu",
            Cihaz::Paralel => "paralel",
        }
    }

    /// Paralel cihazin secilebilir olup olmadigi
    pub fn paralel_secilebilir() -> bool {
        paralel_secilebilir()
    }

    /// Geriye uyumluluk
    pub fn gpu_secilebilir() -> bool {
        paralel_secilebilir()
    }

    /// Cihazi egitim oncesi hazirlar
    pub fn hazirla(self) -> Result<()> {
        match self {
            Cihaz::Cpu => Ok(()),
            Cihaz::Paralel => {
                #[cfg(feature = "parallel")]
                {
                    let threads = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4);
                    let _ = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build_global();
                    log::info!("Paralel CPU modu etkin — {threads} is parcacigi");
                    Ok(())
                }
                #[cfg(not(feature = "parallel"))]
                {
                    Err(TurkError::RuntimeError(
                        "Paralel mod derlenmemis (--features parallel)".into(),
                    ))
                }
            }
        }
    }
}

fn paralel_secilebilir() -> bool {
    cfg!(feature = "parallel")
}

/// Kullanilabilir cihazlarin listesini dondurur
pub fn mevcut_cihazlar() -> Vec<CihazDurumu> {
    let mut list = vec![CihazDurumu {
        cihaz: Cihaz::Cpu,
        mevcut: true,
        aciklama: "Tek is parcacikli CPU (varsayilan)".into(),
    }];
    list.push(CihazDurumu {
        cihaz: Cihaz::Paralel,
        mevcut: paralel_secilebilir(),
        aciklama: if paralel_secilebilir() {
            "Rayon cok cekirdekli CPU (varsayilan)".into()
        } else {
            "Tek cekirdek derleme (--no-default-features)".into()
        },
    });
    list
}

/// Cihaz durumu ozeti
#[derive(Debug, Clone)]
pub struct CihazDurumu {
    pub cihaz: Cihaz,
    pub mevcut: bool,
    pub aciklama: String,
}

/// Cihaz listesini metin tablosu olarak formatlar
pub fn cihaz_listesi_metin() -> String {
    let mut out = String::from("╔══════════════════════════════════════════════════╗\n");
    out.push_str("║           TURK Hesaplama Cihazlari                 ║\n");
    out.push_str("╠══════════════════════════════════════════════════╣\n");
    for d in mevcut_cihazlar() {
        let durum = if d.mevcut { "✓" } else { "—" };
        out.push_str(&format!(
            "║  [{durum}] {:<8} — {:<36} ║\n",
            d.cihaz.kisa(),
            &d.aciklama[..d.aciklama.len().min(36)]
        ));
    }
    out.push_str("╚══════════════════════════════════════════════════╝\n");
    out
}

impl fmt::Display for Cihaz {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ad())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_parse() {
        assert_eq!(Cihaz::from_str("cpu").unwrap(), Cihaz::Cpu);
    }

    #[test]
    fn test_paralel_parse_feature() {
        let r = Cihaz::from_str("paralel");
        #[cfg(feature = "parallel")]
        assert!(r.is_ok());
        #[cfg(not(feature = "parallel"))]
        assert!(r.is_err());
    }

    #[test]
    fn test_gpu_alias_parse() {
        let r = Cihaz::from_str("gpu");
        #[cfg(feature = "parallel")]
        assert_eq!(r.unwrap(), Cihaz::Paralel);
        #[cfg(not(feature = "parallel"))]
        assert!(r.is_err());
    }

    #[test]
    fn test_cihaz_listesi() {
        let t = cihaz_listesi_metin();
        assert!(t.contains("CPU"));
    }
}

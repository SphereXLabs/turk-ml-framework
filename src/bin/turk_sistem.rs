#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! TÜRK Çekirdeği — Kadim Siber Kokpit v0.3.0 (Göktürk Runik + Semantik Hafıza + Zincir)

#[path = "../kokpit_ui.rs"]
mod kokpit_ui;

use chrono::Local;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use kokpit_ui::{
    arka_plan_ciz, cam_panel_ciz, gokturk_tamga_filigrani, komut_satir_sayisi, log_filtre_uygun,
    log_satir_rengi, minimal_telemetry_ciz, olcekli, otuken_pulse_ciz, panel_baslik_ciz,
    pencere_simgesi, renk_lerp, runik_etiket_ciz, splash_ciz, tab_oneri, tema_uygula,
    turk_tamga_duzelt, vu_metre_ciz, LogFiltre, ACCENT_GREEN, ACCENT_ORANGE, DEEP_MOR, GLASS_FILL,
    KENAR_INCE, LOG_CYAN, LOG_NEON_ZINCIR, MICRO_CYAN, NEON_ALERT, NEON_CYAN, TERMINAL_DIM,
    TERMINAL_USER, TUNC_BRONZ, VEKIL_TARAMA_SURE,
};
use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use sysinfo::{Disks, MemoryRefreshKind, ProcessRefreshKind, RefreshKind, System};
use turk::{
    cpu_kullanim_guncelle, konusmayi_metne_cevir, whisper_hazir_mi, EgitimIlerleme, Kulak,
    SesSinyal, SistemVekili, TurkDil, TurkSistem, VekilKomut, VekilOlay, TURK_VEKIL_OPTIMIZE_MESAJ,
    TURK_VEKIL_UYARI_MESAJ,
};

const KOMUT_MERKEZI_YUKSEKLIK_TABAN: f32 = 88.0;
const TURK_CMD_PROMPT: &str = "𐰰𐰼𐰇𐱅_CMD > ";
const KOMUT_GIRDI_ID: &str = "turk_komut_merkezi_girdi";
const BASLIK_MUHUR: &str = "🇹🇷 𐰰𐰼𐰇𐱅 // SİBER KONTROL PANELİ v0.3.0";
const PANEL_SEMANTIK: &str = "🧠 SEMANTİK BELLEK";
const ZINCIR_MUHUR_METIN: &str =
    "✓ Çoklu operasyon zinciri başarıyla kararlı sonuca ulaştırıldı, Sinan kanka!";
const YEREL_AKIL_MUHUR: &str = "[ 𐰶𐰼𐰜 : YEREL AKIL ]";
const PANEL_TELEMETRI: &str = "📊 𐱃𐰸𐰆𐰾 : TELEMETRİ";
const PANEL_REZONANS: &str = "🏎️ 𐰄𐰍𐰃𐰾 : REZONANS";
const PANEL_TERMINAL: &str = "📜 𐰋𐰃𐱅𐰃𐰏 : TERMİNAL";
const VEKIL_BASLIK: &str = "[ 𐰯𐰃𐰏 : VEKİL ]";
const RUNE_CPU: &str = "𐰴";
const RUNE_RAM: &str = "𐰼";
const RUNE_DSK: &str = "𐰓";
const RUNE_IO: &str = "𐰇";

const CPU_OLCUM_ARALIGI: Duration = Duration::from_millis(1000);
const DISK_IO_ARALIGI: Duration = Duration::from_millis(1000);
const ANIMASYON_ARALIGI: Duration = Duration::from_millis(16);

struct DiskIoOlcum {
    son_zaman: Instant,
    baslatildi: bool,
    okuma_mbps: f32,
    yazma_mbps: f32,
}

impl Default for DiskIoOlcum {
    fn default() -> Self {
        Self {
            son_zaman: Instant::now(),
            baslatildi: false,
            okuma_mbps: 0.0,
            yazma_mbps: 0.0,
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    #[cfg(feature = "turk_sistem")]
    {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1360.0, 860.0])
                .with_min_inner_size([1100.0, 700.0])
                .with_icon(pencere_simgesi()),
            ..Default::default()
        };

        eframe::run_native(
            "TÜRK v0.3.0 — Göktürk Kokpit",
            options,
            Box::new(|cc| {
                tema_uygula(&cc.egui_ctx);
                Box::new(TurkSistemGui::default())
            }),
        )
    }

    #[cfg(not(feature = "turk_sistem"))]
    {
        println!("Lütfen projeyi '--features turk_sistem' flag'i ile derleyin.");
        Ok(())
    }
}

fn uptime_metni() -> String {
    let s = System::uptime();
    let gun = s / 86_400;
    let saat = (s % 86_400) / 3_600;
    let dakika = (s % 3_600) / 60;
    format!("{gun}g {saat}s {dakika}dk")
}

fn disk_io_guncelle(sistem: &mut System, olcum: &mut DiskIoOlcum) -> (f32, f32) {
    if olcum.son_zaman.elapsed() < DISK_IO_ARALIGI {
        return (olcum.okuma_mbps, olcum.yazma_mbps);
    }

    sistem.refresh_processes_specifics(ProcessRefreshKind::new().with_disk_usage());
    let mut okuma = 0u64;
    let mut yazma = 0u64;
    for p in sistem.processes().values() {
        let u = p.disk_usage();
        okuma += u.read_bytes;
        yazma += u.written_bytes;
    }

    let saniye = olcum.son_zaman.elapsed().as_secs_f32().max(0.001);
    if olcum.baslatildi {
        olcum.okuma_mbps = (okuma as f32 / 1_048_576.0) / saniye;
        olcum.yazma_mbps = (yazma as f32 / 1_048_576.0) / saniye;
    }
    olcum.baslatildi = true;
    olcum.son_zaman = Instant::now();
    (olcum.okuma_mbps, olcum.yazma_mbps)
}

// ── Rezonans dalga motoru ─────────────────────────────────────────────────────

fn catmull_rom(
    p0: egui::Pos2,
    p1: egui::Pos2,
    p2: egui::Pos2,
    p3: egui::Pos2,
    t: f32,
) -> egui::Pos2 {
    let t2 = t * t;
    let t3 = t2 * t;
    egui::pos2(
        0.5 * ((2.0 * p1.x)
            + (-p0.x + p2.x) * t
            + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
            + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3),
        0.5 * ((2.0 * p1.y)
            + (-p0.y + p2.y) * t
            + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
            + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3),
    )
}

fn akiskan_y_koordinati(t: f32, fazi: f32, genlik: f32, frekans: f32) -> f32 {
    let u = t * std::f32::consts::TAU;
    (fazi * frekans + u * 1.1).sin() * genlik * 0.55
        + (fazi * 0.55 + u * 2.3).sin() * genlik * 0.30
        + (fazi * 0.25 + u * 0.7).cos() * genlik * 0.15
}

fn bezier_dalga_yolu(
    rect: egui::Rect,
    fazi: f32,
    genlik: f32,
    frekans: f32,
    faz_ofset: f32,
) -> Vec<egui::Pos2> {
    let ic = rect.shrink(8.0);
    let n_ankor = 8_usize;
    let mut ankorlar: Vec<egui::Pos2> = Vec::with_capacity(n_ankor);
    for i in 0..n_ankor {
        let t = i as f32 / (n_ankor - 1) as f32;
        let x = ic.left() + t * ic.width();
        let y = ic.center().y + akiskan_y_koordinati(t, fazi + faz_ofset, genlik, frekans);
        ankorlar.push(egui::pos2(x, y));
    }
    let mut yol = Vec::new();
    if ankorlar.len() < 4 {
        return ankorlar;
    }
    for i in 0..ankorlar.len() - 1 {
        let p0 = ankorlar[i.saturating_sub(1)];
        let p1 = ankorlar[i];
        let p2 = ankorlar[i + 1];
        let p3 = ankorlar[(i + 2).min(ankorlar.len() - 1)];
        let adim = 12_usize;
        for s in 0..adim {
            let t = s as f32 / adim as f32;
            yol.push(catmull_rom(p0, p1, p2, p3, t));
        }
    }
    yol.push(*ankorlar.last().unwrap());
    yol
}

fn akiskan_nabiz_ciz(
    painter: &egui::Painter,
    rect: egui::Rect,
    fazi: f32,
    temel_genlik: f32,
    hiz_carp: f32,
    ana_renk: egui::Color32,
) {
    let katmanlar: [(f32, f32, f32, f32); 3] = [
        (0.0, 1.0, 1.0, 0.55),
        (1.8, 0.65, 1.25, 0.38),
        (3.2, 0.40, 1.55, 0.22),
    ];
    for (faz_ofset, gen_carp, frek, opak) in katmanlar {
        let yol = bezier_dalga_yolu(
            rect,
            fazi * hiz_carp,
            temel_genlik * gen_carp,
            frek,
            faz_ofset,
        );
        if yol.len() < 2 {
            continue;
        }
        let renk = if faz_ofset < 0.1 {
            ana_renk.linear_multiply(opak)
        } else if faz_ofset < 2.5 {
            MICRO_CYAN.linear_multiply(opak)
        } else {
            DEEP_MOR.linear_multiply(opak * 1.4)
        };
        painter.add(egui::Shape::Path(egui::epaint::PathShape {
            points: yol.clone(),
            closed: false,
            fill: egui::Color32::TRANSPARENT,
            stroke: egui::Stroke::new(2.5, renk.linear_multiply(0.12)),
        }));
        painter.add(egui::Shape::Path(egui::epaint::PathShape {
            points: yol,
            closed: false,
            fill: egui::Color32::TRANSPARENT,
            stroke: egui::Stroke::new(1.0, renk),
        }));
    }
}

fn sohbet_gosterim_satirlari(metin: &str) -> Vec<&str> {
    if !metin.contains("[ZİNCİR]") && !metin.starts_with("[Plan:") {
        return metin.lines().filter(|l| !l.trim().is_empty()).collect();
    }
    metin
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty()
                && !t.contains("[ZİNCİR]")
                && !t.starts_with("[Plan:")
                && !t.starts_with("---")
        })
        .collect()
}

fn sohbet_baloncuk_ciz(ui: &mut egui::Ui, gonderen: &str, metin: &str) {
    let kullanici = gonderen == "Sinan";
    let uyari = gonderen == "TÜRK" && metin.trim_start().starts_with("[TÜRK Uyarısı]");

    ui.with_layout(
        if kullanici {
            egui::Layout::right_to_left(egui::Align::TOP)
        } else {
            egui::Layout::left_to_right(egui::Align::TOP)
        },
        |ui| {
            let max_w = ui.available_width() * 0.88;
            egui::Frame::none()
                .fill(if uyari {
                    egui::Color32::from_rgba_premultiplied(60, 12, 24, 180)
                } else if kullanici {
                    egui::Color32::from_rgba_premultiplied(28, 32, 42, 200)
                } else {
                    egui::Color32::from_rgba_premultiplied(12, 36, 28, 190)
                })
                .rounding(6.0)
                .stroke(egui::Stroke::new(
                    0.5,
                    if kullanici {
                        TUNC_BRONZ.linear_multiply(0.35)
                    } else {
                        MICRO_CYAN.linear_multiply(0.25)
                    },
                ))
                .inner_margin(egui::Margin::symmetric(10.0, 6.0))
                .show(ui, |ui| {
                    ui.set_max_width(max_w);
                    if !kullanici {
                        ui.label(
                            egui::RichText::new(format!("[{gonderen}]"))
                                .small()
                                .strong()
                                .monospace()
                                .color(ACCENT_GREEN.linear_multiply(0.85)),
                        );
                    }
                    for satir in sohbet_gosterim_satirlari(metin) {
                        let trimmed = satir.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let renk = if uyari || trimmed.starts_with("[TÜRK Uyarısı]") {
                            NEON_ALERT
                        } else if trimmed.contains("[ZİNCİR]") || trimmed.starts_with("[Plan:") {
                            LOG_NEON_ZINCIR
                        } else if trimmed.contains("operasyon zinciri başarıyla") {
                            ACCENT_GREEN.linear_multiply(1.15)
                        } else if kullanici {
                            TERMINAL_USER
                        } else if trimmed.starts_with("[Araç:") {
                            LOG_CYAN
                        } else if trimmed.starts_with('✗') || trimmed.contains("Hata") {
                            egui::Color32::from_rgb(255, 80, 100)
                        } else if trimmed.starts_with('✓') {
                            ACCENT_GREEN
                        } else {
                            egui::Color32::from_rgb(200, 230, 210)
                        };
                        ui.label(
                            egui::RichText::new(turk_tamga_duzelt(trimmed))
                                .monospace()
                                .size(13.5)
                                .color(renk),
                        );
                    }
                });
        },
    );
    ui.add_space(6.0);
}

enum TurkSinyal {
    Tamamlandi {
        zincir_loglari: Vec<String>,
        jarvis_basarili: bool,
        semantik: Vec<String>,
    },
    Hata(String),
}

struct TurkSistemGui {
    secili_model: String,
    csv_yolu: String,
    egitim_calisiyor: bool,
    toplam_epoch: usize,
    guncel_epoch: usize,
    epoch_ilerleme: f32,
    indeksleme_satir: usize,
    egitim_asamasi: bool,
    loss_gecmisi: Vec<f64>,
    val_loss_gecmisi: Vec<f64>,
    durum_mesaji: String,
    hata: Option<String>,
    egitim_rx: Option<Receiver<EgitimIlerleme>>,
    loglar: Vec<String>,

    sohbet: Arc<Mutex<Vec<(String, String)>>>,
    girdi: String,
    turk_dusunuyor: bool,
    bekleyen_komutlar: VecDeque<String>,
    turk_rx: Option<Receiver<TurkSinyal>>,
    dalga_fazi: f32,

    sistem_bilgisi: System,
    os_ad: String,
    cpu_kullanim: f32,
    ram_kullanim_oran: f32,
    ram_kullanim_gb: f32,
    ram_toplam_gb: f32,
    disk_kullanim_oran: Option<f32>,
    disk_io: DiskIoOlcum,
    uptime_metin: String,
    son_cpu_olcumu: Instant,
    son_disk_olcumu: Instant,

    // Sesli komut (STT)
    kulak: Option<Kulak>,
    ses_dinliyor: bool,
    ses_ceviriyor: bool,
    ses_rx: Option<Receiver<SesSinyal>>,

    // TÜRK proaktif vekil (tokio mpsc — çift yönlü)
    kritik_yuk: bool,
    vekil_onay_bekliyor: bool,
    vekil_otonom_mod: bool,
    vekil_tx: tokio::sync::mpsc::Sender<VekilKomut>,
    vekil_rx: tokio::sync::mpsc::Receiver<VekilOlay>,

    anim_zaman: f32,
    splash_baslangic: Instant,
    grafik_opaklik: f32,
    onceki_grafik_modu: bool,
    log_filtre: LogFiltre,
    komut_gecmisi: VecDeque<String>,
    komut_gecmis_idx: Option<usize>,
    otuken_pulse: f32,
    terminal_stick: bool,
    terminal_onceki_len: usize,
    vekil_son_eylem: String,
    vekil_son_tarama: Instant,
    ses_seviye: f32,

    // v3 — semantik hafıza + zincir mührü
    turk_sistem: Arc<Mutex<TurkSistem>>,
    semantik_bellek: Vec<String>,
    zincir_muhur: Option<String>,
}

impl Default for TurkSistemGui {
    fn default() -> Self {
        let mut sistem_bilgisi = System::new_with_specifics(
            RefreshKind::new()
                .with_memory(MemoryRefreshKind::everything())
                .with_processes(ProcessRefreshKind::new().with_disk_usage()),
        );
        sistem_bilgisi.refresh_memory();

        let sohbet = Arc::new(Mutex::new(vec![(
            "TÜRK".into(),
            "Merhaba Sinan. Sistem durumu, dosya listeleme, not alma, proje kontrol veya sesli komut (🎙️) için yazabilirsin.".into(),
        )]));

        let mut loglar = Vec::new();
        let damga = Local::now().format("%H:%M:%S").to_string();
        loglar.push(format!("[{damga}] ▸ TÜRK çevrimiçi — yerel mod aktif."));
        if whisper_hazir_mi() {
            loglar.push(format!("[{damga}] ▸ Whisper STT hazır."));
            loglar.push(format!("[{damga}] ▸ Mikrofon: [ 🎤 MIC ] ile etkinleştir."));
        } else {
            loglar.push(format!(
                "[{damga}] ▸ Whisper: .\\whisper-kur.ps1 ile kurun."
            ));
        }
        loglar.push(format!(
            "[{damga}] ▸ TÜRK v0.3.0 — Göktürk Kokpit (Semantik Hafıza + Otonom Zincir) aktif."
        ));

        let (vekil_tx, vekil_rx) = SistemVekili::baslat();
        let turk_sistem = Arc::new(Mutex::new(TurkSistem::yeni()));
        let semantik_bellek = turk_sistem
            .lock()
            .map(|m| m.semantik_son_hatirlanan())
            .unwrap_or_default();

        Self {
            secili_model: "xor".into(),
            csv_yolu: String::new(),
            egitim_calisiyor: false,
            toplam_epoch: 200,
            guncel_epoch: 0,
            epoch_ilerleme: 0.0,
            indeksleme_satir: 0,
            egitim_asamasi: false,
            loss_gecmisi: Vec::new(),
            val_loss_gecmisi: Vec::new(),
            durum_mesaji: "TÜRK Model Zoo hazır.".into(),
            hata: None,
            egitim_rx: None,
            loglar,
            sohbet,
            girdi: String::new(),
            turk_dusunuyor: false,
            bekleyen_komutlar: VecDeque::new(),
            turk_rx: None,
            dalga_fazi: 0.0,
            sistem_bilgisi,
            os_ad: System::name().unwrap_or_else(|| "Bilinmiyor".into()),
            cpu_kullanim: 0.0,
            ram_kullanim_oran: 0.0,
            ram_kullanim_gb: 0.0,
            ram_toplam_gb: 0.0,
            disk_kullanim_oran: None,
            disk_io: DiskIoOlcum::default(),
            uptime_metin: uptime_metni(),
            son_cpu_olcumu: Instant::now(),
            son_disk_olcumu: Instant::now(),
            kulak: None,
            ses_dinliyor: false,
            ses_ceviriyor: false,
            ses_rx: None,
            kritik_yuk: false,
            vekil_onay_bekliyor: false,
            vekil_otonom_mod: false,
            vekil_tx,
            vekil_rx,
            anim_zaman: 0.0,
            splash_baslangic: Instant::now(),
            grafik_opaklik: 0.0,
            onceki_grafik_modu: false,
            log_filtre: LogFiltre::Tum,
            komut_gecmisi: VecDeque::new(),
            komut_gecmis_idx: None,
            otuken_pulse: 0.0,
            terminal_stick: true,
            terminal_onceki_len: 1,
            vekil_son_eylem: "Beklemede".into(),
            vekil_son_tarama: Instant::now(),
            ses_seviye: 0.0,
            turk_sistem,
            semantik_bellek,
            zincir_muhur: None,
        }
    }
}

impl TurkSistemGui {
    fn sohbet_sayisi(&self) -> usize {
        self.sohbet.lock().map(|g| g.len()).unwrap_or(0)
    }

    fn log_ekle(&mut self, satir: impl Into<String>) {
        let mesaj = satir.into();
        if mesaj.contains("[TÜRK Vekil]") || mesaj.contains('🚨') {
            self.vekil_son_eylem = mesaj
                .trim_start_matches('▸')
                .trim_start_matches('🚨')
                .trim()
                .to_string();
            self.vekil_son_tarama = Instant::now();
        }
        let damga = Local::now().format("%H:%M:%S").to_string();
        self.loglar.push(format!("[{damga}] {mesaj}"));
        if self.loglar.len() > 120 {
            self.loglar.remove(0);
        }
    }

    fn sistem_yuku_ortalama(&self) -> f32 {
        let disk = self.disk_kullanim_oran.unwrap_or(0.0);
        (self.cpu_kullanim + self.ram_kullanim_oran + disk) / 3.0
    }

    fn kritik_banner_goster(&self) -> bool {
        self.kritik_yuk || self.disk_kullanim_oran.is_some_and(|d| d >= 0.90)
    }

    fn otonom_arac_etiketi(&self) -> Option<&'static str> {
        for satir in self.loglar.iter().rev().take(8) {
            if satir.contains("[REFAKTÖR]") {
                return Some("[REFAKTÖR]");
            }
            if satir.contains("[DERLE]") {
                return Some("[DERLE]");
            }
            if satir.contains("[KOD]") {
                return Some("[KOD]");
            }
        }
        None
    }

    fn vekil_sonraki_tarama_kalan(&self) -> f32 {
        let gecen = self.vekil_son_tarama.elapsed().as_secs_f32();
        (VEKIL_TARAMA_SURE.as_secs_f32() - gecen).max(0.0)
    }

    fn mnist_modeli_mi(&self) -> bool {
        self.secili_model.starts_with("mnist")
    }

    fn egitimi_baslat(&mut self) {
        if self.egitim_calisiyor {
            return;
        }
        if self.mnist_modeli_mi() && self.csv_yolu.trim().is_empty() {
            self.hata = Some("MNIST için mnist_train.csv gerekli.".into());
            return;
        }

        self.hata = None;
        self.egitim_calisiyor = true;
        self.epoch_ilerleme = 0.0;
        self.guncel_epoch = 0;
        self.indeksleme_satir = 0;
        self.egitim_asamasi = false;
        self.loss_gecmisi.clear();
        self.val_loss_gecmisi.clear();
        self.durum_mesaji = format!("Eğitim: {}...", self.secili_model);
        self.log_ekle(format!("▸ Eğitim başlatıldı: {}", self.secili_model));

        let (tx, rx) = mpsc::channel();
        self.egitim_rx = Some(rx);
        let model_id = self.secili_model.clone();
        let csv = if self.mnist_modeli_mi() {
            Some(self.csv_yolu.clone())
        } else {
            None
        };

        thread::spawn(move || {
            let mut dil = TurkDil::new();
            dil.egitim_kanali_ayarla(tx.clone());
            if let Err(e) = dil.zoo_betik_calistir(&model_id, csv.as_deref()) {
                let _ = tx.send(EgitimIlerleme::Hata(e.to_string()));
            }
        });
    }

    fn dalga_hiz_carpani(&self) -> f32 {
        let yuk = self.sistem_yuku_ortalama();
        let yuk_carp = 1.0 + yuk * 2.2;
        let temel = if self.kritik_yuk {
            2.8
        } else if self.ses_dinliyor {
            3.4
        } else if self.ses_ceviriyor {
            2.6
        } else if self.turk_dusunuyor {
            3.2
        } else if self.egitim_calisiyor {
            1.6
        } else {
            1.0
        };
        temel * yuk_carp
    }

    fn rezonans_genligi(&self) -> f32 {
        let yuk = self.sistem_yuku_ortalama();
        let temel = 10.0 + yuk * 18.0;
        if self.turk_dusunuyor || self.otonom_rezonans_aktif() {
            temel * 1.35
        } else if self.kritik_yuk {
            temel * 1.15
        } else {
            temel
        }
    }

    fn otonom_rezonans_aktif(&self) -> bool {
        self.turk_dusunuyor
            || self
                .loglar
                .iter()
                .rev()
                .take(3)
                .any(|s| s.contains("[REFAKTÖR]") || s.contains("[DERLE]") || s.contains("[KOD]"))
    }

    fn dalga_rengi(&self) -> egui::Color32 {
        if self.kritik_yuk {
            return NEON_ALERT.linear_multiply(0.8);
        }
        if self.otonom_rezonans_aktif() {
            let t = (self.anim_zaman * 0.8).sin() * 0.5 + 0.5;
            return renk_lerp(TUNC_BRONZ, MICRO_CYAN, t).linear_multiply(0.88);
        }
        if self.ses_dinliyor || self.turk_dusunuyor {
            MICRO_CYAN.linear_multiply(0.85)
        } else {
            MICRO_CYAN.linear_multiply(0.65)
        }
    }

    fn vekil_guncelle(&mut self, ctx: &egui::Context) {
        loop {
            match self.vekil_rx.try_recv() {
                Ok(VekilOlay::KritikYuk { cpu, ram, disk }) => {
                    self.kritik_yuk = true;
                    self.vekil_onay_bekliyor = true;
                    self.log_ekle(format!(
                        "🚨 YÜKSEK YÜK — CPU:{cpu:.0}% RAM:{ram:.0}% Disk:{disk:.0}%"
                    ));
                    {
                        let mut g = self.sohbet.lock().expect("sohbet");
                        g.push(("TÜRK".into(), TURK_VEKIL_UYARI_MESAJ.into()));
                    }
                    ctx.request_repaint();
                }
                Ok(VekilOlay::YukNormaleDondu) => {
                    if self.kritik_yuk {
                        self.kritik_yuk = false;
                        self.vekil_onay_bekliyor = false;
                        self.log_ekle("▸ Sistem yükü normale döndü.");
                        ctx.request_repaint();
                    }
                }
                Ok(VekilOlay::OptimizasyonTamamlandi { sonlandirilan }) => {
                    self.vekil_onay_bekliyor = false;
                    self.log_ekle(TURK_VEKIL_OPTIMIZE_MESAJ.trim());
                    if !sonlandirilan.is_empty() {
                        self.log_ekle(format!(
                            "▸ [TÜRK Vekil] Sonlandırılan süreçler: {}",
                            sonlandirilan.join(", ")
                        ));
                    }
                    ctx.request_repaint();
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
    }

    fn vekil_komut_isle(&mut self, metin: &str) {
        let k = metin.to_lowercase();

        if k.contains("vekil otonom") || (k.contains("vekil") && k.contains("otonom")) {
            let aktif = k.contains("aç")
                || k.contains("ac")
                || k.contains("aktif")
                || k.contains("etkin")
                || !k.contains("kapat");
            self.vekil_otonom_mod = aktif;
            let _ = self.vekil_tx.try_send(VekilKomut::OtonomMod(aktif));
            self.log_ekle(format!(
                "▸ [TÜRK Vekil] Otonom mod {}",
                if aktif { "AÇIK" } else { "KAPALI" }
            ));
            return;
        }

        if self.vekil_onay_bekliyor
            && (k.contains("evet")
                || k.contains("onay")
                || k.contains("optimize")
                || k.contains("optimiz")
                || k.contains("hadi"))
        {
            let _ = self.vekil_tx.try_send(VekilKomut::OptimizasyonOnayla);
            self.log_ekle(
                "▸ [TÜRK Vekil] Süreç optimizasyonu onaylandı — os_seviyesi_kontrol başlatıldı.",
            );
        }
    }

    fn kulak_etkinlestir(&mut self) -> bool {
        if self.kulak.is_some() {
            return true;
        }
        match Kulak::ac() {
            Ok(k) => {
                self.kulak = Some(k);
                self.log_ekle("▸ Mikrofon (Kulak) hazır — 🎙️ aktif.");
                true
            }
            Err(e) => {
                self.log_ekle(format!("▸ Mikrofon açılamadı: {e}"));
                false
            }
        }
    }

    fn ses_mikrofon_toggle(&mut self, ctx: &egui::Context) {
        if !self.kulak_etkinlestir() {
            return;
        }
        let Some(kulak) = self.kulak.clone() else {
            return;
        };

        if !self.ses_dinliyor {
            if kulak.kayit_baslat().is_err() {
                self.log_ekle("Kayıt başlatılamadı.");
                return;
            }
            if !whisper_hazir_mi() {
                self.log_ekle("Whisper kurulu degil — .\\whisper-kur.ps1 calistirin.");
            }
            self.ses_dinliyor = true;
            self.log_ekle("🎙️ Dinleniyor — tekrar tıkla");
            ctx.request_repaint();
            return;
        }

        self.ses_dinliyor = false;
        self.ses_ceviriyor = true;
        self.log_ekle("🎙️ Whisper işleniyor...");

        let (tx, rx) = mpsc::channel();
        self.ses_rx = Some(rx);

        thread::Builder::new()
            .name("turk-whisper".into())
            .spawn(move || {
                let sonuc = match kulak.kayit_durdur() {
                    Ok(kayit) => konusmayi_metne_cevir(&kayit.ornekler, kayit.ornekleme_hz)
                        .map(|m| SesSinyal::Metin { metin: m }),
                    Err(e) => Ok(SesSinyal::Hata(e)),
                };
                match sonuc {
                    Ok(sinyal) => {
                        let _ = tx.send(sinyal);
                    }
                    Err(e) => {
                        let _ = tx.send(SesSinyal::Hata(e));
                    }
                }
            })
            .expect("whisper thread");

        ctx.request_repaint();
    }

    fn ses_metni_isle(&mut self, metin: String, ctx: &egui::Context) {
        let metin = metin.trim().to_string();
        if metin.is_empty() {
            return;
        }
        self.girdi = metin.clone();
        self.log_ekle(format!("🎙️ «{metin}»"));

        {
            let mut gecmis = self.sohbet.lock().expect("sohbet kilidi");
            gecmis.push(("Sinan".into(), format!("🎙️ {metin}")));
        }

        if self.turk_dusunuyor {
            self.log_ekle(format!(
                "⏳ Kuyruğa alındı ({}): {metin}",
                self.bekleyen_komutlar.len() + 1
            ));
            self.bekleyen_komutlar.push_back(metin);
        } else {
            self.turk_komut_baslat(metin);
        }
        ctx.request_repaint();
    }

    fn ses_guncelle(&mut self, ctx: &egui::Context) {
        let mut bitti = false;
        if let Some(rx) = self.ses_rx.as_ref() {
            if let Ok(msg) = rx.try_recv() {
                match msg {
                    SesSinyal::Metin { metin } => {
                        self.ses_metni_isle(metin, ctx);
                    }
                    SesSinyal::Hata(e) => {
                        self.log_ekle(format!("🎙️ Hata: {e}"));
                        {
                            let mut g = self.sohbet.lock().expect("sohbet");
                            g.push(("TÜRK".into(), format!("[Ses] {e}")));
                        }
                    }
                }
                bitti = true;
            }
        }
        if bitti {
            self.ses_ceviriyor = false;
            self.ses_rx = None;
            ctx.request_repaint();
        }
    }

    fn turk_gonder(&mut self) {
        let metin = self.girdi.trim().to_string();
        if metin.is_empty() {
            return;
        }
        self.otuken_pulse = 1.0;
        if self.komut_gecmisi.back().map(|s| s.as_str()) != Some(metin.as_str()) {
            self.komut_gecmisi.push_back(metin.clone());
            if self.komut_gecmisi.len() > 50 {
                self.komut_gecmisi.pop_front();
            }
        }
        self.komut_gecmis_idx = None;
        self.girdi.clear();

        {
            let mut gecmis = self.sohbet.lock().expect("sohbet kilidi");
            gecmis.push(("Sinan".into(), metin.clone()));
        }

        self.vekil_komut_isle(&metin);

        if self.turk_dusunuyor {
            self.bekleyen_komutlar.push_back(metin.clone());
            let sira = self.bekleyen_komutlar.len();
            self.log_ekle(format!("⏳ Kuyruk ({sira}): {metin}"));
            return;
        }

        self.turk_komut_baslat(metin);
    }

    fn turk_komut_baslat(&mut self, metin: String) {
        self.turk_dusunuyor = true;
        self.zincir_muhur = None;
        self.log_ekle(format!("→ {metin}"));

        let (tx, rx) = mpsc::channel();
        self.turk_rx = Some(rx);

        let sohbet = Arc::clone(&self.sohbet);
        let turk_sistem = Arc::clone(&self.turk_sistem);

        thread::Builder::new()
            .name("turk-asistan".into())
            .spawn(move || {
                let sonuc = {
                    let mut motor = turk_sistem.lock().expect("turk_sistem kilidi");
                    motor.mesaj_isle(&metin)
                };
                match sonuc {
                    Ok(yanit) => {
                        let zincir_loglari: Vec<String> = yanit
                            .lines()
                            .filter(|l| {
                                let t = l.trim();
                                t.contains("[ZİNCİR]") || t.starts_with("[Plan:")
                            })
                            .map(|l| l.trim().to_string())
                            .collect();
                        let zincir_basarili = yanit.contains("operasyon zinciri başarıyla");
                        let semantik = turk_sistem
                            .lock()
                            .map(|m| m.semantik_son_hatirlanan())
                            .unwrap_or_default();
                        if let Ok(mut g) = sohbet.lock() {
                            g.push(("TÜRK".into(), yanit));
                        }
                        let _ = tx.send(TurkSinyal::Tamamlandi {
                            zincir_loglari,
                            jarvis_basarili: zincir_basarili,
                            semantik,
                        });
                    }
                    Err(e) => {
                        if let Ok(mut g) = sohbet.lock() {
                            g.push(("TÜRK".into(), format!("Hata: {e}")));
                        }
                        let _ = tx.send(TurkSinyal::Hata(e.to_string()));
                    }
                }
            })
            .expect("asistan thread");
    }

    fn egitim_guncelle(&mut self, ctx: &egui::Context) {
        let mut bitti = false;
        let mut yenile = false;
        let mut log_satirlari: Vec<String> = Vec::new();

        if let Some(rx) = self.egitim_rx.as_ref() {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    EgitimIlerleme::Indeksleme { satir } => {
                        self.indeksleme_satir = satir;
                        self.durum_mesaji = if satir == 0 {
                            "Doğrulama seti yükleniyor...".into()
                        } else {
                            format!("CSV indeksleniyor: {satir} satır")
                        };
                        yenile = true;
                    }
                    EgitimIlerleme::Durum { mesaj } => {
                        self.durum_mesaji = mesaj.clone();
                        log_satirlari.push(mesaj);
                        yenile = true;
                    }
                    EgitimIlerleme::Epoch {
                        epoch,
                        toplam,
                        train_loss,
                        val_loss,
                    } => {
                        self.egitim_asamasi = true;
                        self.toplam_epoch = toplam;
                        self.guncel_epoch = epoch;
                        self.epoch_ilerleme = epoch as f32 / toplam.max(1) as f32;
                        self.loss_gecmisi.push(train_loss);
                        if let Some(v) = val_loss {
                            self.val_loss_gecmisi.push(v);
                        }
                        let satir = format!("Epoch {epoch}/{toplam} loss={train_loss:.4}");
                        self.durum_mesaji = satir.clone();
                        log_satirlari.push(satir);
                        yenile = true;
                    }
                    EgitimIlerleme::Tamamlandi {
                        epoch_sayisi,
                        en_iyi_train,
                        en_iyi_val,
                    } => {
                        self.durum_mesaji = format!("Tamamlandı — {epoch_sayisi} epoch");
                        log_satirlari.push(format!(
                            "En iyi: loss={en_iyi_train:.6}{}",
                            en_iyi_val
                                .map(|v| format!(", val={v:.6}"))
                                .unwrap_or_default()
                        ));
                        self.epoch_ilerleme = 1.0;
                        self.guncel_epoch = epoch_sayisi;
                        bitti = true;
                        yenile = true;
                    }
                    EgitimIlerleme::Hata(e) => {
                        self.hata = Some(e.clone());
                        log_satirlari.push(format!("HATA: {e}"));
                        bitti = true;
                    }
                }
            }
        }

        for satir in log_satirlari {
            self.log_ekle(satir);
        }

        if bitti {
            self.egitim_calisiyor = false;
            self.egitim_rx = None;
        }
        if yenile {
            ctx.request_repaint();
        }
        if self.egitim_calisiyor {
            ctx.request_repaint_after(ANIMASYON_ARALIGI);
        }
    }

    fn ajans_gunlugu_cikar(&mut self, yanit: &str) {
        for satir in yanit.lines() {
            let t = satir.trim();
            if t.starts_with("▸ [KOD]")
                || t.starts_with("✓ [KOD]")
                || t.starts_with("▸ [DERLE]")
                || t.starts_with("✓ [DERLE]")
                || t.starts_with("✗ [DERLE]")
                || t.starts_with("▸ [REFAKTÖR]")
                || t.starts_with("✓ [REFAKTÖR]")
                || t.contains("[ZİNCİR]")
                || t.starts_with("[Plan:")
                || t.contains("[TÜRK Vekil]")
            {
                self.log_ekle(t);
            }
        }
    }

    fn turk_guncelle(&mut self, ctx: &egui::Context) {
        let mut bitti = false;
        let mut log_satirlari: Vec<String> = Vec::new();
        let mut son_yanit: Option<String> = None;
        let mut jarvis_basarili = false;

        if let Some(rx) = self.turk_rx.as_ref() {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    TurkSinyal::Tamamlandi {
                        zincir_loglari,
                        jarvis_basarili: jarvis_ok,
                        semantik,
                    } => {
                        for z in &zincir_loglari {
                            log_satirlari.push(z.clone());
                        }
                        jarvis_basarili = jarvis_ok;
                        self.semantik_bellek = semantik;
                        if let Ok(g) = self.sohbet.lock() {
                            if let Some((_, son)) = g.last() {
                                son_yanit = Some(son.clone());
                                log_satirlari
                                    .push(format!("← {}", son.lines().next().unwrap_or("")));
                            }
                        }
                    }
                    TurkSinyal::Hata(e) => {
                        log_satirlari.push(format!("HATA: {e}"));
                    }
                }
                bitti = true;
            }
        }

        for satir in log_satirlari {
            self.log_ekle(satir);
        }

        if let Some(yanit) = son_yanit {
            self.ajans_gunlugu_cikar(&yanit);
        }

        if jarvis_basarili {
            self.zincir_muhur = Some(ZINCIR_MUHUR_METIN.into());
        }

        if bitti {
            self.turk_dusunuyor = false;
            self.turk_rx = None;
            ctx.request_repaint();

            if let Some(siradaki) = self.bekleyen_komutlar.pop_front() {
                self.turk_komut_baslat(siradaki);
            }
        }
    }

    fn telemetry_bar(
        &self,
        ui: &mut egui::Ui,
        etiket: &str,
        oran: f32,
        _dolgu: egui::Color32,
        metin: impl std::fmt::Display,
    ) {
        let disk_kritik = self.disk_kullanim_oran.is_some_and(|d| d >= 0.90);
        minimal_telemetry_ciz(
            ui,
            etiket,
            oran,
            &format!("{metin}"),
            self.kritik_yuk || disk_kritik,
            self.anim_zaman,
        );
    }

    fn sol_panel(&mut self, ui: &mut egui::Ui) {
        let panel_rect = ui.available_rect_before_wrap();
        cam_panel_ciz(ui.painter(), panel_rect, self.anim_zaman);
        ui.allocate_ui_at_rect(panel_rect.shrink(12.0), |ui| {
            ui.vertical(|ui| {
                ui.vertical(|ui| {
                    panel_baslik_ciz(ui, PANEL_TELEMETRI);
                    ui.separator();
                    ui.label(
                        egui::RichText::new(&self.os_ad)
                            .small()
                            .monospace()
                            .color(TERMINAL_DIM),
                    );
                    ui.label(
                        egui::RichText::new(format!("UPTIME  {}", self.uptime_metin))
                            .small()
                            .monospace()
                            .color(TUNC_BRONZ.linear_multiply(0.9)),
                    );

                    ui.add_space(8.0);
                    self.telemetry_bar(
                        ui,
                        &format!("{RUNE_CPU} CPU"),
                        self.cpu_kullanim,
                        ACCENT_GREEN,
                        format!("{:.0}%", self.cpu_kullanim * 100.0),
                    );
                    self.telemetry_bar(
                        ui,
                        &format!("{RUNE_RAM} RAM"),
                        self.ram_kullanim_oran,
                        ACCENT_GREEN,
                        format!("{:.1}%", self.ram_kullanim_oran * 100.0),
                    );
                    if let Some(disk) = self.disk_kullanim_oran {
                        self.telemetry_bar(
                            ui,
                            &format!("{RUNE_DSK} DSK"),
                            disk,
                            ACCENT_GREEN,
                            format!("{:.0}%", disk * 100.0),
                        );
                    }

                    ui.label(
                        egui::RichText::new(format!(
                            "{:.1} / {:.1} GB RAM",
                            self.ram_kullanim_gb, self.ram_toplam_gb
                        ))
                        .small()
                        .monospace()
                        .weak(),
                    );

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        runik_etiket_ciz(ui, &format!("{RUNE_IO} I/O"));
                        ui.label(
                            egui::RichText::new(format!(
                                "↓ {:.1}  |  ↑ {:.1} MB/s",
                                self.disk_io.okuma_mbps, self.disk_io.yazma_mbps
                            ))
                            .small()
                            .monospace()
                            .color(MICRO_CYAN.linear_multiply(0.7)),
                        );
                    });

                    ui.add_space(12.0);
                    runik_etiket_ciz(ui, VEKIL_BASLIK);
                    ui.horizontal(|ui| {
                        let durum = if self.vekil_otonom_mod {
                            "Otonom AÇIK"
                        } else {
                            "Otonom KAPALI"
                        };
                        ui.label(egui::RichText::new(durum).small().monospace().color(
                            if self.vekil_otonom_mod {
                                ACCENT_GREEN.linear_multiply(0.8)
                            } else {
                                TERMINAL_DIM
                            },
                        ));
                        if ui.checkbox(&mut self.vekil_otonom_mod, "toggle").changed() {
                            let _ = self
                                .vekil_tx
                                .try_send(VekilKomut::OtonomMod(self.vekil_otonom_mod));
                            self.log_ekle(format!(
                                "▸ [TÜRK Vekil] Otonom mod {}",
                                if self.vekil_otonom_mod {
                                    "AÇIK"
                                } else {
                                    "KAPALI"
                                }
                            ));
                        }
                    });
                    if self.vekil_onay_bekliyor {
                        ui.label(
                            egui::RichText::new("⚠ Onay bekleniyor — «evet optimize et»")
                                .small()
                                .color(NEON_ALERT.linear_multiply(0.85)),
                        );
                    }
                    if self.kritik_yuk {
                        ui.label(
                            egui::RichText::new("𐰴 Kritik yük")
                                .small()
                                .color(NEON_ALERT.linear_multiply(0.8)),
                        );
                    }
                    ui.label(
                        egui::RichText::new(format!("Son: {}", self.vekil_son_eylem))
                            .small()
                            .monospace()
                            .color(TERMINAL_DIM),
                    );
                    if self.vekil_otonom_mod {
                        ui.label(
                            egui::RichText::new(format!(
                                "Sonraki tarama: {:.0}s",
                                self.vekil_sonraki_tarama_kalan()
                            ))
                            .small()
                            .monospace()
                            .color(TUNC_BRONZ.linear_multiply(0.85)),
                        );
                    }
                });

                ui.add_space(8.0);
                ui.separator();
                runik_etiket_ciz(ui, PANEL_SEMANTIK);
                ui.label(
                    egui::RichText::new("Son hatırlanan bağlam")
                        .small()
                        .monospace()
                        .color(TERMINAL_DIM.linear_multiply(0.85)),
                );
                egui::Frame::none()
                    .fill(egui::Color32::from_rgba_premultiplied(8, 16, 24, 160))
                    .rounding(4.0)
                    .stroke(egui::Stroke::new(0.5, MICRO_CYAN.linear_multiply(0.25)))
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .max_height(120.0)
                            .show(ui, |ui| {
                                if self.semantik_bellek.is_empty() {
                                    ui.label(
                                        egui::RichText::new("Henüz semantik eşleşme yok.")
                                            .small()
                                            .monospace()
                                            .color(TERMINAL_DIM),
                                    );
                                } else {
                                    for satir in &self.semantik_bellek {
                                        ui.label(
                                            egui::RichText::new(satir)
                                                .small()
                                                .monospace()
                                                .color(MICRO_CYAN.linear_multiply(0.88)),
                                        );
                                    }
                                }
                            });
                    });
            });
        });
    }

    fn grafik_gosterilmeli(&self) -> bool {
        self.egitim_calisiyor || !self.loss_gecmisi.is_empty()
    }

    fn rezonans_alani_ciz(&self, ui: &mut egui::Ui, rect: egui::Rect, opaklik: f32) {
        let painter = ui.painter_at(rect);
        cam_panel_ciz(&painter, rect, self.anim_zaman);
        gokturk_tamga_filigrani(
            &painter,
            rect,
            self.anim_zaman,
            self.otonom_rezonans_aktif(),
        );
        let dalga_rect = rect.shrink(12.0);
        let hiz = self.dalga_hiz_carpani();
        let amp = self.rezonans_genligi()
            * if self.otonom_rezonans_aktif() {
                1.15
            } else {
                0.75
            };
        akiskan_nabiz_ciz(
            &painter,
            dalga_rect,
            self.dalga_fazi,
            amp,
            hiz,
            self.dalga_rengi().linear_multiply(opaklik.max(0.05)),
        );
        let durum = if self.kritik_banner_goster() {
            format!(
                "Kritik yük — Disk {:.0}%",
                self.disk_kullanim_oran.unwrap_or(0.0) * 100.0
            )
        } else if self.ses_dinliyor {
            "Dinleniyor".into()
        } else if self.ses_ceviriyor {
            "Whisper".into()
        } else if let Some(arac) = self.otonom_arac_etiketi() {
            format!("Otonom rezonans {arac}")
        } else if self.turk_dusunuyor {
            "Otonom rezonans".into()
        } else {
            "Bekleme".into()
        };
        painter.text(
            dalga_rect.center_bottom() + egui::vec2(0.0, -6.0),
            egui::Align2::CENTER_BOTTOM,
            durum,
            egui::FontId::monospace(12.0),
            TERMINAL_DIM,
        );
    }

    fn orta_panel(&mut self, ui: &mut egui::Ui) {
        panel_baslik_ciz(ui, PANEL_REZONANS);
        ui.separator();

        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("Model:")
                    .small()
                    .monospace()
                    .color(TERMINAL_DIM),
            );
            egui::ComboBox::from_id_source("turk_model")
                .selected_text(&self.secili_model)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.secili_model, "xor".into(), "XOR");
                    ui.selectable_value(&mut self.secili_model, "mnist_mlp".into(), "MNIST MLP");
                    ui.selectable_value(&mut self.secili_model, "mnist_cnn".into(), "MNIST CNN");
                });
            if ui
                .add_enabled(
                    !self.egitim_calisiyor,
                    egui::Button::new(egui::RichText::new("▶ EĞİT").strong().color(ACCENT_GREEN)),
                )
                .clicked()
            {
                self.egitimi_baslat();
            }
        });

        if self.mnist_modeli_mi() {
            ui.horizontal(|ui| {
                ui.label("CSV:");
                ui.text_edit_singleline(&mut self.csv_yolu);
                if ui.button("...").clicked() {
                    if let Some(p) = rfd::FileDialog::new()
                        .add_filter("CSV", &["csv"])
                        .pick_file()
                    {
                        self.csv_yolu = p.display().to_string();
                    }
                }
            });
        }

        if self.egitim_calisiyor || self.guncel_epoch > 0 {
            if self.egitim_calisiyor && !self.egitim_asamasi {
                let t = if self.indeksleme_satir == 0 {
                    "CSV hazırlanıyor...".into()
                } else {
                    format!("İndeks: {} satır", self.indeksleme_satir)
                };
                ui.add(
                    egui::ProgressBar::new(-1.0)
                        .animate(true)
                        .fill(NEON_CYAN)
                        .text(t),
                );
            }
            if self.egitim_asamasi || self.guncel_epoch > 0 {
                ui.add(
                    egui::ProgressBar::new(self.epoch_ilerleme)
                        .fill(ACCENT_GREEN)
                        .text(format!("Epoch {}/{}", self.guncel_epoch, self.toplam_epoch)),
                );
            }
        }

        ui.label(
            egui::RichText::new(&self.durum_mesaji)
                .small()
                .monospace()
                .color(NEON_CYAN.linear_multiply(0.9)),
        );
        if let Some(h) = &self.hata {
            ui.colored_label(egui::Color32::from_rgb(255, 70, 90), h);
        }

        let grafik_yukseklik = (ui.available_height() * 0.38).clamp(120.0, 200.0);
        let grafik_modu = self.grafik_gosterilmeli() && !self.loss_gecmisi.is_empty();

        egui::Frame::none()
            .fill(egui::Color32::TRANSPARENT)
            .inner_margin(0.0)
            .show(ui, |ui| {
                let w = ui.available_width();
                let (kutu, _) =
                    ui.allocate_exact_size(egui::vec2(w, grafik_yukseklik), egui::Sense::hover());
                let dalga_opak = if grafik_modu {
                    (1.0 - self.grafik_opaklik).max(0.08)
                } else {
                    1.0
                };
                self.rezonans_alani_ciz(ui, kutu, dalga_opak);

                if grafik_modu && self.grafik_opaklik > 0.02 {
                    let opak = self.grafik_opaklik;
                    ui.allocate_ui_at_rect(kutu, |ui| {
                        let pts: PlotPoints = self
                            .loss_gecmisi
                            .iter()
                            .enumerate()
                            .map(|(i, &y)| [i as f64 + 1.0, y])
                            .collect();
                        Plot::new("turk_loss")
                            .height(grafik_yukseklik)
                            .width(kutu.width())
                            .allow_zoom(false)
                            .allow_drag(false)
                            .allow_scroll(false)
                            .show_background(false)
                            .show_axes([true, true])
                            .legend(Legend::default())
                            .show(ui, |p| {
                                p.line(
                                    Line::new(pts)
                                        .color(ACCENT_GREEN.linear_multiply(opak))
                                        .width(2.0)
                                        .name("loss"),
                                );
                                if !self.val_loss_gecmisi.is_empty() {
                                    let vpts: PlotPoints = self
                                        .val_loss_gecmisi
                                        .iter()
                                        .enumerate()
                                        .map(|(i, &y)| [i as f64 + 1.0, y])
                                        .collect();
                                    p.line(
                                        Line::new(vpts)
                                            .color(ACCENT_ORANGE.linear_multiply(opak))
                                            .width(2.0)
                                            .name("val_loss"),
                                    );
                                }
                            });
                    });
                }
            });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("📜 SİSTEM LOG")
                    .small()
                    .strong()
                    .monospace()
                    .color(TUNC_BRONZ.linear_multiply(0.85)),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                for (etiket, filtre) in [
                    ("Sohbet", LogFiltre::Sohbet),
                    ("Derle", LogFiltre::Derle),
                    ("Vekil", LogFiltre::Vekil),
                    ("Tümü", LogFiltre::Tum),
                ] {
                    let secili = self.log_filtre == filtre;
                    if ui
                        .selectable_label(secili, egui::RichText::new(etiket).small().monospace())
                        .clicked()
                    {
                        self.log_filtre = filtre;
                    }
                }
            });
        });
        egui::Frame::none()
            .fill(egui::Color32::TRANSPARENT)
            .inner_margin(0.0)
            .show(ui, |ui| {
                let log_rect = ui.available_rect_before_wrap();
                cam_panel_ciz(ui.painter(), log_rect, self.anim_zaman);
                ui.allocate_ui_at_rect(log_rect.shrink(8.0), |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(110.0)
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for satir in self
                                .loglar
                                .iter()
                                .filter(|s| log_filtre_uygun(s, self.log_filtre))
                            {
                                ui.label(
                                    egui::RichText::new(turk_tamga_duzelt(satir))
                                        .monospace()
                                        .size(12.5)
                                        .color(log_satir_rengi(satir)),
                                );
                            }
                        });
                });
            });
    }

    fn sag_panel(&mut self, ui: &mut egui::Ui) {
        let panel_rect = ui.available_rect_before_wrap();
        cam_panel_ciz(ui.painter(), panel_rect, self.anim_zaman);
        ui.allocate_ui_at_rect(panel_rect.shrink(12.0), |ui| {
            panel_baslik_ciz(ui, PANEL_TERMINAL);
            ui.separator();
            ui.label(
                egui::RichText::new(format!("{} mesaj", self.sohbet_sayisi()))
                    .small()
                    .monospace()
                    .color(TERMINAL_DIM),
            );
            ui.add_space(6.0);

            let gecmis = self.sohbet.lock().map(|g| g.clone()).unwrap_or_default();

            self.terminal_onceki_len = gecmis.len();

            if !self.terminal_stick {
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .button(egui::RichText::new("↓ Yeni mesajlar").small())
                            .clicked()
                        {
                            self.terminal_stick = true;
                        }
                    });
                });
            }

            let scroll_yukseklik = ui.available_height().max(80.0);
            let (chat_rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), scroll_yukseklik),
                egui::Sense::hover(),
            );
            cam_panel_ciz(ui.painter(), chat_rect, self.anim_zaman);

            ui.allocate_ui_at_rect(chat_rect.shrink(8.0), |ui| {
                egui::ScrollArea::vertical()
                    .id_source("turk_terminal_scroll")
                    .max_height(scroll_yukseklik - 16.0)
                    .auto_shrink([false, false])
                    .stick_to_bottom(self.terminal_stick)
                    .show(ui, |ui| {
                        if ui.input(|i| i.smooth_scroll_delta.y > 1.0) {
                            self.terminal_stick = false;
                        }
                        for (gonderen, metin) in &gecmis {
                            sohbet_baloncuk_ciz(ui, gonderen, metin);
                        }
                        if self.turk_dusunuyor {
                            egui::Frame::none()
                                .fill(egui::Color32::from_rgba_premultiplied(12, 36, 28, 160))
                                .rounding(6.0)
                                .inner_margin(8.0)
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.spinner();
                                        ui.label(
                                            egui::RichText::new("_ düşünüyor...")
                                                .monospace()
                                                .italics()
                                                .color(ACCENT_GREEN.linear_multiply(0.65)),
                                        );
                                    });
                                });
                        }
                    });
            });
        });
    }

    fn komut_merkezi(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        otuken_pulse_ciz(ui, self.otuken_pulse);
        let panel = ui.max_rect();
        ui.painter().rect_filled(panel, 0.0, GLASS_FILL);
        ui.painter().hline(
            panel.x_range(),
            panel.top(),
            egui::Stroke::new(KENAR_INCE, renk_lerp(MICRO_CYAN, TUNC_BRONZ, 0.4)),
        );

        ui.allocate_ui_at_rect(panel.shrink2(egui::vec2(16.0, 10.0)), |ui| {
            let girdi_id = egui::Id::new(KOMUT_GIRDI_ID);
            ui.memory_mut(|mem| mem.request_focus(girdi_id));

            if let Some(ref muhur) = self.zincir_muhur {
                let pulse = 0.85 + (self.anim_zaman * 2.4).sin() * 0.15;
                ui.horizontal_wrapped(|ui| {
                    ui.set_max_width(ui.available_width());
                    ui.label(
                        egui::RichText::new(muhur)
                            .monospace()
                            .size(14.0)
                            .strong()
                            .color(ACCENT_GREEN.linear_multiply(pulse)),
                    );
                });
                ui.add_space(6.0);
            }

            if self.ses_dinliyor {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("🎤").small().color(ACCENT_ORANGE));
                    vu_metre_ciz(ui, self.ses_seviye, self.anim_zaman);
                });
            }

            let mut gonder = false;
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new(TURK_CMD_PROMPT)
                        .monospace()
                        .size(15.0)
                        .strong()
                        .color(ACCENT_GREEN),
                );

                let satirlar = komut_satir_sayisi(&self.girdi);
                let girdi = ui.add(
                    egui::TextEdit::multiline(&mut self.girdi)
                        .id(girdi_id)
                        .hint_text("sistem durumu | not al | cargo check | proje refaktör …")
                        .font(egui::TextStyle::Monospace)
                        .desired_width(ui.available_width().min(520.0))
                        .desired_rows(satirlar)
                        .margin(egui::vec2(12.0, 10.0)),
                );

                if ui.input(|i| i.key_pressed(egui::Key::Tab)) {
                    if let Some(oneri) = tab_oneri(&self.girdi) {
                        self.girdi = oneri;
                    }
                }

                if ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) && !self.komut_gecmisi.is_empty()
                {
                    let idx = self.komut_gecmis_idx.unwrap_or(self.komut_gecmisi.len());
                    if idx > 0 {
                        let yeni = idx - 1;
                        if let Some(k) = self.komut_gecmisi.get(yeni) {
                            self.girdi = k.clone();
                            self.komut_gecmis_idx = Some(yeni);
                        }
                    }
                }
                if ui.input(|i| i.key_pressed(egui::Key::ArrowDown))
                    && self.komut_gecmis_idx.is_some()
                {
                    let idx = self.komut_gecmis_idx.unwrap();
                    if idx + 1 < self.komut_gecmisi.len() {
                        let yeni = idx + 1;
                        if let Some(k) = self.komut_gecmisi.get(yeni) {
                            self.girdi = k.clone();
                            self.komut_gecmis_idx = Some(yeni);
                        }
                    } else {
                        self.girdi.clear();
                        self.komut_gecmis_idx = None;
                    }
                }

                let enter = ui.input(|i| i.key_pressed(egui::Key::Enter))
                    && ui.input(|i| !i.modifiers.shift);

                if let Some(oneri) = tab_oneri(&self.girdi) {
                    ui.label(
                        egui::RichText::new(format!("⇥ {}", &oneri[self.girdi.len()..]))
                            .small()
                            .monospace()
                            .color(TERMINAL_DIM.linear_multiply(0.5)),
                    );
                }

                let mic_aktif = self.ses_dinliyor || self.ses_ceviriyor;
                let mic_etiket = if self.ses_dinliyor {
                    "[ 🎤 MIC ● ]"
                } else if self.ses_ceviriyor {
                    "[ 🎤 MIC … ]"
                } else {
                    "[ 🎤 MIC ]"
                };
                let mic_renk = if mic_aktif {
                    ACCENT_ORANGE
                } else if whisper_hazir_mi() {
                    ACCENT_GREEN
                } else {
                    TERMINAL_DIM
                };
                if ui
                    .add_enabled(
                        whisper_hazir_mi() && !self.ses_ceviriyor,
                        egui::Button::new(egui::RichText::new(mic_etiket).color(mic_renk)),
                    )
                    .clicked()
                {
                    self.ses_mikrofon_toggle(ctx);
                }

                let otuken_renk = if self.otuken_pulse > 0.0 {
                    ACCENT_GREEN.linear_multiply(0.6 + self.otuken_pulse * 0.4)
                } else {
                    egui::Color32::WHITE
                };
                if ui
                    .button(
                        egui::RichText::new("[ 💥 ÖTÜKEN ]")
                            .small()
                            .strong()
                            .color(otuken_renk),
                    )
                    .clicked()
                    || (enter && !self.girdi.trim().is_empty())
                {
                    gonder = true;
                }

                if enter && gonder {
                    girdi.surrender_focus();
                }
            });

            if gonder {
                self.turk_gonder();
            }
        });
    }
}

impl eframe::App for TurkSistemGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let dt = ctx.input(|i| i.unstable_dt).min(0.05);
        self.anim_zaman += dt;
        if self.otuken_pulse > 0.0 {
            self.otuken_pulse = (self.otuken_pulse - dt * 2.5).max(0.0);
        }
        if self.ses_dinliyor {
            self.ses_seviye = (0.35 + (self.anim_zaman * 5.5).sin().abs() * 0.45).clamp(0.1, 0.95);
        } else {
            self.ses_seviye = (self.ses_seviye * 0.9).max(0.0);
        }

        let grafik_modu = self.grafik_gosterilmeli() && !self.loss_gecmisi.is_empty();
        if grafik_modu {
            self.grafik_opaklik = (self.grafik_opaklik + dt * 2.0).min(1.0);
        } else {
            self.grafik_opaklik = (self.grafik_opaklik - dt * 1.5).max(0.0);
        }
        self.onceki_grafik_modu = grafik_modu;

        arka_plan_ciz(ctx, self.anim_zaman);

        let rezonans = if self.otonom_rezonans_aktif() {
            1.4
        } else {
            1.0
        };
        self.dalga_fazi += dt * 2.8 * self.dalga_hiz_carpani() * rezonans;

        self.sistem_bilgisi.refresh_memory();
        let toplam_bytes = self.sistem_bilgisi.total_memory() as f64;
        let kullanilan_bytes = self.sistem_bilgisi.used_memory() as f64;
        if toplam_bytes > 0.0 {
            self.ram_toplam_gb = (toplam_bytes / (1024.0 * 1024.0 * 1024.0)) as f32;
            self.ram_kullanim_gb = (kullanilan_bytes / (1024.0 * 1024.0 * 1024.0)) as f32;
            self.ram_kullanim_oran = ((kullanilan_bytes / toplam_bytes) as f32).clamp(0.0, 1.0);
        }

        let cpu_yuku = cpu_kullanim_guncelle();
        self.cpu_kullanim = (cpu_yuku / 100.0).clamp(0.0, 1.0);
        if self.son_cpu_olcumu.elapsed() >= CPU_OLCUM_ARALIGI {
            self.son_cpu_olcumu = Instant::now();
            self.uptime_metin = uptime_metni();
        }

        let (_okuma, _yazma) = disk_io_guncelle(&mut self.sistem_bilgisi, &mut self.disk_io);

        if self.son_disk_olcumu.elapsed() >= Duration::from_secs(2) {
            self.disk_kullanim_oran = ana_disk_yuzdesi();
            self.son_disk_olcumu = Instant::now();
        }

        if self.vekil_otonom_mod && self.vekil_sonraki_tarama_kalan() <= 0.0 {
            self.vekil_son_tarama = Instant::now();
        }

        self.egitim_guncelle(ctx);
        self.turk_guncelle(ctx);
        self.ses_guncelle(ctx);
        self.vekil_guncelle(ctx);

        ctx.request_repaint_after(ANIMASYON_ARALIGI);

        let sol_gen = olcekli(ctx, 300.0);
        let sag_gen = olcekli(ctx, 400.0);
        let komut_yuk = olcekli(ctx, KOMUT_MERKEZI_YUKSEKLIK_TABAN)
            + (komut_satir_sayisi(&self.girdi).saturating_sub(2) as f32 * 18.0)
            + if self.zincir_muhur.is_some() {
                28.0
            } else {
                0.0
            };

        let header_yuk = if self.kritik_banner_goster() {
            olcekli(ctx, 76.0)
        } else {
            olcekli(ctx, 54.0)
        };

        egui::TopBottomPanel::top("turk_header")
            .frame(egui::Frame::none())
            .exact_height(header_yuk)
            .show(ctx, |ui| {
                let hdr = ui.max_rect();
                cam_panel_ciz(ui.painter(), hdr, self.anim_zaman);
                ui.allocate_ui_at_rect(hdr.shrink(14.0), |ui| {
                    if self.kritik_banner_goster() {
                        let alpha = 0.55 + (self.anim_zaman * 4.0).sin().abs() * 0.25;
                        ui.allocate_ui_with_layout(
                            ui.available_size(),
                            egui::Layout::top_down(egui::Align::Center),
                            |ui| {
                                ui.label(
                                    egui::RichText::new(
                                        "⚠ 𐰴 KRİTİK YÜK — Disk / sistem kaynakları sınırda",
                                    )
                                    .monospace()
                                    .size(13.0)
                                    .color(NEON_ALERT.linear_multiply(0.92 * alpha)),
                                );
                            },
                        );
                        ui.separator();
                    }
                    ui.horizontal(|ui| {
                        runik_etiket_ciz(ui, BASLIK_MUHUR);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let whisper_etiket = if whisper_hazir_mi() {
                                "[ WHISPER · AKTİF ]"
                            } else {
                                "[ WHISPER · KAPALI ]"
                            };
                            ui.label(
                                egui::RichText::new(whisper_etiket)
                                    .monospace()
                                    .size(13.0)
                                    .color(if whisper_hazir_mi() {
                                        ACCENT_GREEN.linear_multiply(0.8)
                                    } else {
                                        TERMINAL_DIM
                                    }),
                            );
                            ui.label(
                                egui::RichText::new(YEREL_AKIL_MUHUR)
                                    .monospace()
                                    .size(13.0)
                                    .color(TUNC_BRONZ.linear_multiply(0.95)),
                            );
                        });
                    });
                });
            });

        egui::TopBottomPanel::bottom("komut_merkezi")
            .frame(egui::Frame::none())
            .min_height(komut_yuk)
            .show(ctx, |ui| self.komut_merkezi(ui, ctx));

        egui::SidePanel::left("sol_panel")
            .resizable(true)
            .default_width(sol_gen)
            .frame(egui::Frame::none())
            .show(ctx, |ui| self.sol_panel(ui));

        egui::SidePanel::right("sag_panel")
            .resizable(true)
            .default_width(sag_gen)
            .frame(egui::Frame::none())
            .show(ctx, |ui| self.sag_panel(ui));

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| self.orta_panel(ui));

        splash_ciz(ctx, self.splash_baslangic, self.anim_zaman);
    }
}

fn ana_disk_yuzdesi() -> Option<f32> {
    /// TÜRK siber telemetri etiketi — otonom refaktör.
    #[allow(dead_code)]
    fn turk_siber_etiket() -> &'static str {
        /// TÜRK siber telemetri etiketi — otonom refaktör.
        #[allow(dead_code)]
        fn turk_siber_etiket() -> &'static str {
            "◈ SİBER DALGA — TÜRK refaktör aktif"
        }
        "◈ SİBER DALGA — TÜRK refaktör aktif"
    }
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
                ((total - d.available_space()) as f32 / total as f32).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
}

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;
use turk::{EgitimIlerleme, TurkDil};

const TURK_MAVI: egui::Color32 = egui::Color32::from_rgb(0, 120, 212);
const TURK_YESIL: egui::Color32 = egui::Color32::from_rgb(76, 175, 80);
const TURK_TURUNCU: egui::Color32 = egui::Color32::from_rgb(255, 152, 0);

fn main() -> Result<(), eframe::Error> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 720.0])
            .with_min_inner_size([720.0, 520.0]),
        ..Default::default()
    };

    eframe::run_native(
        "TURK — Gorsel Stüdyo",
        options,
        Box::new(|_cc| Box::new(TurkGui::default())),
    )
}

struct TurkGui {
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
}

impl Default for TurkGui {
    fn default() -> Self {
        Self {
            secili_model: "xor".to_string(),
            csv_yolu: String::new(),
            egitim_calisiyor: false,
            toplam_epoch: 200,
            guncel_epoch: 0,
            epoch_ilerleme: 0.0,
            indeksleme_satir: 0,
            egitim_asamasi: false,
            loss_gecmisi: Vec::new(),
            val_loss_gecmisi: Vec::new(),
            durum_mesaji: "Hazir — Model Zoo'dan egitim baslatabilirsiniz.".into(),
            hata: None,
            egitim_rx: None,
        }
    }
}

impl TurkGui {
    fn mnist_modeli_mi(&self) -> bool {
        self.secili_model.starts_with("mnist")
    }

    /// Egitim arka plan thread'inde — GUI thread'i bloklanmaz.
    fn egitimi_baslat(&mut self) {
        if self.egitim_calisiyor {
            return;
        }

        if self.mnist_modeli_mi() && self.csv_yolu.trim().is_empty() {
            self.hata = Some("MNIST modelleri icin mnist_train.csv yolu gerekli.".into());
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
        self.durum_mesaji = format!("Egitim baslatiliyor: {}...", self.secili_model);

        let (tx, rx) = mpsc::channel();
        self.egitim_rx = Some(rx);

        let model_id = self.secili_model.clone();
        let csv = if self.mnist_modeli_mi() {
            Some(self.csv_yolu.clone())
        } else {
            None
        };

        thread::Builder::new()
            .name(format!("turk-egitim-{model_id}"))
            .spawn(move || {
                let mut dil = TurkDil::new();
                dil.egitim_kanali_ayarla(tx.clone());

                if let Err(e) = dil.zoo_betik_calistir(&model_id, csv.as_deref()) {
                    let _ = tx.send(EgitimIlerleme::Hata(e.to_string()));
                }
            })
            .expect("egitim thread baslatilamadi");
    }

    fn csv_dosya_sec(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("CSV", &["csv"])
            .pick_file()
        {
            self.csv_yolu = path.display().to_string();
        }
    }

    fn egitim_guncellemelerini_isle(&mut self, ctx: &egui::Context) {
        let mut bitti = false;
        let mut yenile = false;

        if let Some(rx) = &self.egitim_rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    EgitimIlerleme::Indeksleme { satir } => {
                        self.indeksleme_satir = satir;
                        self.durum_mesaji = if satir == 0 {
                            "Dogrulama seti bellege aliniyor...".into()
                        } else {
                            format!("CSV indeksleniyor: {satir} satir tarandi...")
                        };
                        yenile = true;
                    }
                    EgitimIlerleme::Durum { mesaj } => {
                        self.durum_mesaji = mesaj;
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
                        self.durum_mesaji = format!(
                            "Epoch {epoch}/{toplam} — loss={train_loss:.6}{}",
                            val_loss
                                .map(|v| format!(", val_loss={v:.6}"))
                                .unwrap_or_default()
                        );
                        yenile = true;
                    }
                    EgitimIlerleme::Tamamlandi {
                        epoch_sayisi,
                        en_iyi_train,
                        en_iyi_val,
                    } => {
                        self.durum_mesaji = format!(
                            "Egitim tamamlandi — {epoch_sayisi} epoch | \
                             en iyi: loss={en_iyi_train:.6}{}",
                            en_iyi_val
                                .map(|v| format!(", val_loss={v:.6}"))
                                .unwrap_or_default()
                        );
                        self.epoch_ilerleme = 1.0;
                        self.guncel_epoch = epoch_sayisi;
                        bitti = true;
                        yenile = true;
                    }
                    EgitimIlerleme::Hata(e) => {
                        self.hata = Some(e);
                        bitti = true;
                    }
                }
            }
        }

        if bitti {
            self.egitim_calisiyor = false;
            self.egitim_rx = None;
        }

        // Kanaldan veri geldiyse hemen yeniden ciz; egitim devam ederken de periyodik repaint
        if yenile {
            ctx.request_repaint();
        }
        if self.egitim_calisiyor {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
    }

    fn loss_grafik_kutusu(&self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_gray(24))
            .inner_margin(12.0)
            .rounding(6.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Canli Egitim Grafigi")
                        .strong()
                        .color(TURK_MAVI),
                );
                ui.label(
                    egui::RichText::new("Yesil: loss  |  Turuncu: val_loss  |  Surukle/zoom")
                        .small()
                        .weak(),
                );

                if self.loss_gecmisi.is_empty() {
                    ui.add_space(80.0);
                    ui.vertical_centered(|ui| {
                        ui.label(
                            egui::RichText::new(
                                "Egitim baslayinca loss egrileri burada canli cizilir",
                            )
                            .weak(),
                        );
                    });
                    ui.add_space(80.0);
                    return;
                }

                let loss_pts: PlotPoints = self
                    .loss_gecmisi
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| [i as f64 + 1.0, y])
                    .collect();

                let loss_line = Line::new(loss_pts)
                    .name("loss")
                    .color(TURK_YESIL)
                    .width(2.0);

                Plot::new("egitim_loss_plot")
                    .height(ui.available_height().max(240.0))
                    .width(ui.available_width())
                    .x_axis_label("Epoch")
                    .y_axis_label("Kayip")
                    .legend(Legend::default())
                    .allow_zoom(true)
                    .allow_drag(true)
                    .allow_scroll(true)
                    .show(ui, |plot_ui| {
                        plot_ui.line(loss_line);
                        if !self.val_loss_gecmisi.is_empty() {
                            let val_pts: PlotPoints = self
                                .val_loss_gecmisi
                                .iter()
                                .enumerate()
                                .map(|(i, &y)| [i as f64 + 1.0, y])
                                .collect();
                            plot_ui.line(
                                Line::new(val_pts)
                                    .name("val_loss")
                                    .color(TURK_TURUNCU)
                                    .width(2.0),
                            );
                        }
                    });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if let Some(&loss) = self.loss_gecmisi.last() {
                        ui.label(format!("Son loss: {loss:.6}"));
                    }
                    if let Some(&val) = self.val_loss_gecmisi.last() {
                        ui.separator();
                        ui.label(format!("Son val_loss: {val:.6}"));
                    }
                    ui.separator();
                    ui.label(format!("N = {} epoch", self.loss_gecmisi.len()));
                });
            });
    }
}

impl eframe::App for TurkGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.egitim_guncellemelerini_isle(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(
                egui::RichText::new("TURK Model Zoo — Gorsel Egitim Paneli")
                    .strong()
                    .color(TURK_MAVI),
            );
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Model:");
                egui::ComboBox::from_id_source("model_sec")
                    .selected_text(&self.secili_model)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.secili_model,
                            "xor".into(),
                            "XOR Siniflandirici",
                        );
                        ui.selectable_value(
                            &mut self.secili_model,
                            "mnist_mlp".into(),
                            "MNIST MLP (784→128→10)",
                        );
                        ui.selectable_value(
                            &mut self.secili_model,
                            "mnist_cnn".into(),
                            "MNIST CNN",
                        );
                    });
            });

            if self.mnist_modeli_mi() {
                ui.horizontal(|ui| {
                    ui.label("CSV:");
                    ui.text_edit_singleline(&mut self.csv_yolu);
                    if ui.button("Dosya Sec...").clicked() {
                        self.csv_dosya_sec();
                    }
                });
                ui.label(
                    egui::RichText::new(
                        "mnist_train.csv — 784 piksel + label (lazy stream, batch=64)",
                    )
                    .small()
                    .weak(),
                );
            }

            ui.add_space(6.0);

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.egitim_calisiyor, egui::Button::new("Egitimi Baslat"))
                    .clicked()
                {
                    self.egitimi_baslat();
                }
                if self.egitim_calisiyor {
                    ui.spinner();
                    ui.label("Arka planda calisiyor (GUI serbest)...");
                }
            });

            if self.egitim_calisiyor || self.guncel_epoch > 0 {
                if self.egitim_calisiyor && !self.egitim_asamasi {
                    let metin = if self.indeksleme_satir == 0 {
                        "CSV hazirlaniyor...".to_string()
                    } else {
                        format!("Indeksleme: {} satir", self.indeksleme_satir)
                    };
                    ui.add(egui::ProgressBar::new(-1.0).animate(true).text(metin));
                }
                if self.egitim_asamasi || self.guncel_epoch > 0 {
                    ui.add(
                        egui::ProgressBar::new(self.epoch_ilerleme)
                            .text(format!("Epoch {}/{}", self.guncel_epoch, self.toplam_epoch)),
                    );
                }
            }

            ui.label(&self.durum_mesaji);

            if let Some(h) = &self.hata {
                ui.colored_label(egui::Color32::RED, format!("Hata: {h}"));
            }

            ui.add_space(6.0);
            ui.separator();

            // Grafik kutusu — kalan alani doldur
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), ui.available_height()),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    self.loss_grafik_kutusu(ui);
                },
            );
        });
    }
}

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;
use turk::{AlperBeyin, EgitimIlerleme, TurkDil};

const TURK_MAVI: egui::Color32 = egui::Color32::from_rgb(0, 120, 212);
const TURK_YESIL: egui::Color32 = egui::Color32::from_rgb(76, 175, 80);
const TURK_TURUNCU: egui::Color32 = egui::Color32::from_rgb(255, 152, 0);
const ALPER_CYAN: egui::Color32 = egui::Color32::from_rgb(110, 198, 212);

fn main() -> Result<(), eframe::Error> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([860.0, 560.0]),
        ..Default::default()
    };

    eframe::run_native(
        "TURK — Görsel Stüdyo + ALPER",
        options,
        Box::new(|_cc| Box::new(TurkGui::yeni())),
    )
}

struct ChatSatir {
    kullanici: bool,
    metin: String,
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
    // ALPER
    alper: Option<AlperBeyin>,
    alper_acik: bool,
    alper_girdi: String,
    alper_sohbet: Vec<ChatSatir>,
    alper_durum: String,
    dsl_editor: String,
    alper_egitiliyor: bool,
}

impl TurkGui {
    fn yeni() -> Self {
        let mut gui = Self {
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
            durum_mesaji: "Hazır — Model Zoo veya ALPER ile başlayın.".into(),
            hata: None,
            egitim_rx: None,
            alper: None,
            alper_acik: true,
            alper_girdi: String::new(),
            alper_sohbet: Vec::new(),
            alper_durum: "ALPER yükleniyor…".into(),
            dsl_editor: String::new(),
            alper_egitiliyor: false,
        };
        match AlperBeyin::ac() {
            Ok(beyin) => {
                gui.alper_durum = beyin.ozet();
                gui.alper_sohbet.push(ChatSatir {
                    kullanici: false,
                    metin: format!(
                        "Merhaba. Ben ALPER.\n{}\n\nDeneyin: «xor eğit», «dsl üret», «relu nedir», «benim adım Sinan»",
                        beyin.ozet()
                    ),
                });
                gui.alper = Some(beyin);
            }
            Err(e) => {
                gui.alper_durum = format!("ALPER açılamadı: {e}");
                gui.alper_sohbet.push(ChatSatir {
                    kullanici: false,
                    metin: format!("Beyin yüklenemedi: {e}"),
                });
            }
        }
        gui
    }

    fn mnist_modeli_mi(&self) -> bool {
        self.secili_model.starts_with("mnist")
    }

    fn egitimi_baslat(&mut self) {
        if self.egitim_calisiyor {
            return;
        }

        if self.mnist_modeli_mi() && self.csv_yolu.trim().is_empty() {
            self.hata = Some("MNIST modelleri için mnist_train.csv yolu gerekli.".into());
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
        self.durum_mesaji = format!("Eğitim başlatılıyor: {}...", self.secili_model);

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
                            "Doğrulama seti belleğe alınıyor...".into()
                        } else {
                            format!("CSV indeksleniyor: {satir} satır tarandı...")
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
                        self.guncel_epoch = epoch;
                        self.toplam_epoch = toplam.max(1);
                        self.epoch_ilerleme = epoch as f32 / self.toplam_epoch as f32;
                        self.loss_gecmisi.push(train_loss);
                        if let Some(v) = val_loss {
                            self.val_loss_gecmisi.push(v);
                        }
                        self.durum_mesaji = format!(
                            "Epoch {epoch}/{toplam} · loss={train_loss:.5}{}",
                            val_loss
                                .map(|v| format!(" · val={v:.5}"))
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
                            "Eğitim tamam · {epoch_sayisi} epoch · best loss={en_iyi_train:.5}{}",
                            en_iyi_val
                                .map(|v| format!(" · val={v:.5}"))
                                .unwrap_or_default()
                        );
                        bitti = true;
                        yenile = true;
                    }
                    EgitimIlerleme::Hata(e) => {
                        self.hata = Some(e);
                        self.durum_mesaji = "Eğitim hata ile durdu.".into();
                        bitti = true;
                        yenile = true;
                    }
                }
            }
        }

        if bitti {
            self.egitim_calisiyor = false;
            self.egitim_rx = None;
        }
        if yenile {
            ctx.request_repaint();
        }
        if self.egitim_calisiyor {
            ctx.request_repaint_after(Duration::from_millis(33));
        }
    }

    fn alper_gonder(&mut self) {
        let cmd = self.alper_girdi.trim().to_string();
        if cmd.is_empty() {
            return;
        }
        self.alper_girdi.clear();
        self.alper_sohbet.push(ChatSatir {
            kullanici: true,
            metin: cmd.clone(),
        });

        // Yerel kısayollar
        let n = cmd.to_lowercase();
        if n == "durum" || n == "özet" || n == "ozet" {
            let ozet = self
                .alper
                .as_ref()
                .map(|a| a.ozet())
                .unwrap_or_else(|| "ALPER yok".into());
            self.alper_sohbet.push(ChatSatir {
                kullanici: false,
                metin: ozet.clone(),
            });
            self.alper_durum = ozet;
            return;
        }
        if n == "eğit" || n == "egit" || n.contains("yeniden eğit") || n.contains("yeniden egit")
        {
            self.alper_yeniden_egit();
            return;
        }

        let (yanit, ozet) = {
            let Some(beyin) = self.alper.as_mut() else {
                self.alper_sohbet.push(ChatSatir {
                    kullanici: false,
                    metin: "ALPER beyni yok.".into(),
                });
                return;
            };
            let y = beyin.yanitla(&cmd);
            let o = beyin.ozet();
            (y, o)
        };

        match yanit {
            Ok(y) => {
                self.alper_sohbet.push(ChatSatir {
                    kullanici: false,
                    metin: format!("[{} · %{:.0}]\n{}", y.intent, y.confidence * 100.0, y.metin),
                });
                if let Some(code) = y.dsl {
                    self.dsl_editor = code;
                }
                let tetik = y.egitim_tetikle.map(|s| s.to_string());
                self.alper_durum = ozet;
                if let Some(preset) = tetik {
                    self.secili_model = preset;
                    if !self.egitim_calisiyor {
                        self.egitimi_baslat();
                    }
                }
            }
            Err(e) => {
                self.alper_sohbet.push(ChatSatir {
                    kullanici: false,
                    metin: format!("Hata: {e}"),
                });
            }
        }
    }

    fn alper_yeniden_egit(&mut self) {
        let Some(beyin) = self.alper.as_mut() else {
            return;
        };
        self.alper_egitiliyor = true;
        self.alper_sohbet.push(ChatSatir {
            kullanici: false,
            metin: "ALPER beyni yeniden eğitiliyor…".into(),
        });
        match beyin.egit_ve_kaydet() {
            Ok(acc) => {
                self.alper_durum = beyin.ozet();
                self.alper_sohbet.push(ChatSatir {
                    kullanici: false,
                    metin: format!(
                        "Yeniden eğitim tamam. Doğruluk %{:.1}\nÖnbellek: ~/.turk/alper_brain.bin",
                        acc * 100.0
                    ),
                });
            }
            Err(e) => {
                self.alper_sohbet.push(ChatSatir {
                    kullanici: false,
                    metin: format!("Eğitim hatası: {e}"),
                });
            }
        }
        self.alper_egitiliyor = false;
    }

    fn loss_grafik_kutusu(&self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_gray(24))
            .inner_margin(12.0)
            .rounding(6.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Canlı Eğitim Grafiği")
                        .strong()
                        .color(TURK_MAVI),
                );
                ui.label(
                    egui::RichText::new("Yeşil: loss  |  Turuncu: val_loss")
                        .small()
                        .weak(),
                );

                if self.loss_gecmisi.is_empty() {
                    ui.add_space(60.0);
                    ui.vertical_centered(|ui| {
                        ui.label(
                            egui::RichText::new(
                                "Eğitim başlayınca loss eğrileri burada canlı çizilir",
                            )
                            .weak(),
                        );
                    });
                    ui.add_space(60.0);
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
                    .width(2.0_f32);

                Plot::new("egitim_loss_plot")
                    .height(ui.available_height().max(200.0))
                    .width(ui.available_width())
                    .x_axis_label("Epoch")
                    .y_axis_label("Kayıp")
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
                                    .width(2.0_f32),
                            );
                        }
                    });

                ui.horizontal(|ui| {
                    if let Some(&loss) = self.loss_gecmisi.last() {
                        ui.label(format!("Son loss: {loss:.6}"));
                    }
                    if let Some(&val) = self.val_loss_gecmisi.last() {
                        ui.separator();
                        ui.label(format!("Son val_loss: {val:.6}"));
                    }
                });
            });
    }

    fn alper_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("ALPER Asistan")
                    .strong()
                    .color(ALPER_CYAN),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(!self.alper_egitiliyor, egui::Button::new("Yeniden Eğit"))
                    .clicked()
                {
                    self.alper_yeniden_egit();
                }
            });
        });
        ui.label(egui::RichText::new(&self.alper_durum).small().weak());
        ui.separator();

        egui::ScrollArea::vertical()
            .id_source("alper_chat_scroll")
            .max_height(ui.available_height() * 0.42)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for s in &self.alper_sohbet {
                    let (label, color) = if s.kullanici {
                        ("Siz", TURK_TURUNCU)
                    } else {
                        ("ALPER", ALPER_CYAN)
                    };
                    ui.label(egui::RichText::new(label).small().color(color).strong());
                    ui.label(&s.metin);
                    ui.add_space(6.0);
                }
            });

        ui.separator();
        ui.horizontal(|ui| {
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.alper_girdi)
                    .desired_width(ui.available_width() - 72.0)
                    .hint_text("xor eğit · dsl üret · relu nedir…"),
            );
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.alper_gonder();
            }
            if ui.button("Gönder").clicked() {
                self.alper_gonder();
            }
        });

        ui.add_space(4.0);
        ui.collapsing("İpuçları", |ui| {
            for h in [
                "selam",
                "benim adım Sinan",
                "xor modelini eğit",
                "xor dsl yaz",
                "relu nedir",
                "sistem durumu",
                "hesapla 12*7",
            ] {
                if ui.small_button(h).clicked() {
                    self.alper_girdi = h.to_string();
                    self.alper_gonder();
                }
            }
        });

        ui.separator();
        ui.label(
            egui::RichText::new(".turk DSL Editörü")
                .strong()
                .color(TURK_MAVI),
        );
        ui.label(
            egui::RichText::new("ALPER «dsl üret» deyince buraya yazar")
                .small()
                .weak(),
        );
        ui.add(
            egui::TextEdit::multiline(&mut self.dsl_editor)
                .code_editor()
                .desired_width(ui.available_width())
                .desired_rows(10),
        );
        ui.horizontal(|ui| {
            if ui.button("Editörü Temizle").clicked() {
                self.dsl_editor.clear();
            }
            if ui
                .add_enabled(
                    !self.dsl_editor.trim().is_empty() && !self.egitim_calisiyor,
                    egui::Button::new("DSL'yi Çalıştır"),
                )
                .clicked()
            {
                self.dsl_calistir();
            }
        });
    }

    fn dsl_calistir(&mut self) {
        let kod = self.dsl_editor.clone();
        if kod.trim().is_empty() {
            return;
        }
        self.hata = None;
        self.egitim_calisiyor = true;
        self.loss_gecmisi.clear();
        self.val_loss_gecmisi.clear();
        self.durum_mesaji = "DSL çalıştırılıyor…".into();
        let (tx, rx) = mpsc::channel();
        self.egitim_rx = Some(rx);
        thread::Builder::new()
            .name("turk-dsl-run".into())
            .spawn(move || {
                let mut dil = TurkDil::new();
                dil.egitim_kanali_ayarla(tx.clone());
                if let Err(e) = dil.calistir(&kod) {
                    let _ = tx.send(EgitimIlerleme::Hata(e.to_string()));
                } else {
                    let _ = tx.send(EgitimIlerleme::Tamamlandi {
                        epoch_sayisi: 0,
                        en_iyi_train: 0.0,
                        en_iyi_val: None,
                    });
                }
            })
            .expect("dsl thread");
    }
}

impl eframe::App for TurkGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.egitim_guncellemelerini_isle(ctx);

        egui::TopBottomPanel::top("turk_top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(
                    egui::RichText::new("TURK Model Zoo — Görsel Eğitim")
                        .strong()
                        .color(TURK_MAVI),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.toggle_value(&mut self.alper_acik, "ALPER Asistan");
                });
            });
        });

        egui::SidePanel::right("alper_side")
            .resizable(true)
            .default_width(380.0)
            .width_range(280.0..=520.0)
            .show_animated(ctx, self.alper_acik, |ui| {
                self.alper_panel(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Model:");
                egui::ComboBox::from_id_source("model_sec")
                    .selected_text(&self.secili_model)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.secili_model,
                            "xor".into(),
                            "XOR Sınıflandırıcı",
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
                    if ui.button("Dosya Seç...").clicked() {
                        self.csv_dosya_sec();
                    }
                });
            }

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!self.egitim_calisiyor, egui::Button::new("Eğitimi Başlat"))
                    .clicked()
                {
                    self.egitimi_baslat();
                }
                if self.egitim_calisiyor {
                    ui.spinner();
                    ui.label("Arka planda çalışıyor…");
                }
            });

            if self.egitim_calisiyor || self.guncel_epoch > 0 {
                if self.egitim_calisiyor && !self.egitim_asamasi {
                    ui.add(
                        egui::ProgressBar::new(-1.0)
                            .animate(true)
                            .text("Hazırlanıyor…"),
                    );
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

            ui.separator();
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

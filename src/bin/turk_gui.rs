use eframe::egui;
use egui::{Color32, RichText, Stroke, Vec2};
use poll_promise::Promise;
use rfd::FileDialog;
use std::fs;
use std::path::PathBuf;
use turk_lang::{
    ast::{AktivasyonFonksiyonu, Deyim, OptimizerTipi, Program},
    parser::model_parser::ModelParser,
    token::Token,
};
use logos::Logos;

const TURK_MAVİ: Color32 = Color32::from_rgb(0, 120, 212);
const TURK_KOYU_MAVİ: Color32 = Color32::from_rgb(0, 80, 170);
const TURK_AÇIK_MAVİ: Color32 = Color32::from_rgb(100, 181, 246);
const TURK_MOR: Color32 = Color32::from_rgb(93, 53, 132);
const TURK_YEŞİL: Color32 = Color32::from_rgb(76, 175, 80);
const TURK_KIRMIZI: Color32 = Color32::from_rgb(211, 47, 47);
const TURK_GRİ: Color32 = Color32::from_rgb(95, 99, 104);
const TURK_AÇIK_GRİ: Color32 = Color32::from_rgb(241, 243, 244);
const TURK_KOYU_GRİ: Color32 = Color32::from_rgb(60, 64, 67);

/// TURK Makine Öğrenmesi GUI Uygulaması
struct TurkGui {
    kod: String,
    program: Option<Program>,
    hata_mesajı: Option<String>,
    çıktı: String,
    tema_koyu: bool,
    dosya_yolu: Option<PathBuf>,
    modeller: Vec<ModelGörselleştirme>,
    aktif_sekme: Sekme,
    örnek_kodlar: Vec<ÖrnekKod>,
    çalıştırma_sonuçları: Option<Promise<Result<String, String>>>,
}

struct ModelGörselleştirme {
    isim: String,
    katmanlar: Vec<KatmanÖzellikleri>,
    optimizer: String,
    kayıp_fonksiyonu: String,
}

struct KatmanÖzellikleri {
    nöron_sayısı: usize,
    aktivasyon: String,
}

struct ÖrnekKod {
    başlık: String,
    açıklama: String,
    kod: String,
}

#[derive(PartialEq)]
enum Sekme {
    Düzenleyici,
    Görselleştirme,
    Ayarlar,
    Hakkında,
}

impl Default for TurkGui {
    fn default() -> Self {
        let örnek_kodlar = vec![
            ÖrnekKod {
                başlık: "Basit Sınıflandırıcı".to_string(),
                açıklama: "XOR kapısı benzeri bir örnek için basit bir sınıflandırıcı".to_string(),
                kod: r#"Model sinir_agi {
    Katman(3, Aktivasyon=Relu)
    Katman(5, Aktivasyon=Relu)
    Katman(2, Aktivasyon=Softmax)
    OgrenmeOrani=0,1
    Kayip="cross_entropy"
    Batch=32
    Epoch=5
}

Veri x_train = [[1,0,0], [0,1,0], [0,0,1]]
Veri y_train = [[1,0], [0,1], [1,0]]

Egit(sinir_agi, x_train, y_train)

Veri test_verisi = [1,0,0]
Tahmin(sinir_agi, test_verisi)"#.to_string(),
            },
            ÖrnekKod {
                başlık: "MNIST Sınıflandırıcı".to_string(),
                açıklama: "El yazısı rakamları tanıyan bir sinir ağı modeli".to_string(),
                kod: r#"Model mnist_model {
    Katman(784, Aktivasyon=Relu)
    Katman(128, Aktivasyon=Relu)
    Katman(64, Aktivasyon=Relu)
    Katman(10, Aktivasyon=Softmax)
    OgrenmeOrani=0,001
    Optimizer="Adam"
    Kayip="categorical_crossentropy"
    Batch=64
    Epoch=10
}

VeriSeti mnist {
    Kaynak="mnist"
    EgitimOran=0.8
    TestOran=0.2
}

Egit(mnist_model, mnist.egitim_x, mnist.egitim_y)
Degerlendir(mnist_model, mnist.test_x, mnist.test_y)

Kaydet(mnist_model, "mnist_model.json")"#.to_string(),
            },
        ];

        Self {
            kod: örnek_kodlar[0].kod.clone(),
            program: None,
            hata_mesajı: None,
            çıktı: String::new(),
            tema_koyu: true,
            dosya_yolu: None,
            modeller: Vec::new(),
            aktif_sekme: Sekme::Düzenleyici,
            örnek_kodlar,
            çalıştırma_sonuçları: None,
        }
    }
}

impl eframe::App for TurkGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.tema_koyu {
            ctx.set_visuals(egui::Visuals::dark());
        } else {
            ctx.set_visuals(egui::Visuals::light());
        }

        self.üst_panel_göster(ctx);
        self.sol_panel_göster(ctx);

        match self.aktif_sekme {
            Sekme::Düzenleyici => self.kod_editörü_göster(ctx),
            Sekme::Görselleştirme => self.görselleştirme_göster(ctx),
            Sekme::Ayarlar => self.ayarlar_göster(ctx),
            Sekme::Hakkında => self.hakkında_göster(ctx),
        }

        if !self.çıktı.is_empty() {
            self.alt_panel_göster(ctx);
        }
        
        // Çalıştırma sonuçlarını işle
        if let Some(promise) = &self.çalıştırma_sonuçları {
            if let Some(result) = promise.ready() {
                match result {
                    Ok(output) => {
                        self.çıktı = output.clone();
                        self.hata_mesajı = None;
                    }
                    Err(err) => {
                        self.hata_mesajı = Some(err.clone());
                    }
                }
                self.çalıştırma_sonuçları = None;
            } else {
                // İşlem hala devam ediyor, bir yükleniyor göstergesi göster
                egui::Window::new("Çalıştırılıyor")
                    .collapsible(false)
                    .resizable(false)
                    .show(ctx, |ui| {
                        ui.label("Program çalıştırılıyor...");
                        ui.spinner();
                    });
            }
        }
    }
}

impl TurkGui {
    fn üst_panel_göster(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("üst_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(RichText::new("TURK").color(TURK_MAVİ).strong());
                ui.heading(" Programlama Dili");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button(if self.tema_koyu { "☀" } else { "🌙" }).clicked() {
                        self.tema_koyu = !self.tema_koyu;
                    }
                });
            });
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("📄 Yeni").clicked() {
                    self.kod = String::new();
                    self.program = None;
                    self.hata_mesajı = None;
                    self.çıktı = String::new();
                    self.dosya_yolu = None;
                }

                if ui.button("📂 Aç").clicked() {
                    if let Some(path) = FileDialog::new()
                        .add_filter("TURK Dosyası", &["turk"])
                        .pick_file()
                    {
                        match fs::read_to_string(&path) {
                            Ok(içerik) => {
                                self.kod = içerik;
                                self.dosya_yolu = Some(path);
                                self.hata_mesajı = None;
                            }
                            Err(e) => {
                                self.hata_mesajı = Some(format!("Dosya açılamadı: {}", e));
                            }
                        }
                    }
                }

                if ui.button("💾 Kaydet").clicked() {
                    if let Some(path) = &self.dosya_yolu {
                        if let Err(e) = fs::write(path, &self.kod) {
                            self.hata_mesajı = Some(format!("Dosya kaydedilemedi: {}", e));
                        }
                    } else {
                        if let Some(path) = FileDialog::new()
                            .add_filter("TURK Dosyası", &["turk"])
                            .save_file()
                        {
                            if let Err(e) = fs::write(&path, &self.kod) {
                                self.hata_mesajı = Some(format!("Dosya kaydedilemedi: {}", e));
                            } else {
                                self.dosya_yolu = Some(path);
                            }
                        }
                    }
                }

                ui.separator();

                if ui.button("🔍 Ayrıştır").clicked() {
                    self.kodu_ayrıştır();
                }

                if ui
                    .add(egui::Button::new(
                        RichText::new("▶ Çalıştır").color(TURK_YEŞİL).strong(),
                    ))
                    .clicked()
                {
                    self.kodu_çalıştır();
                }

                if ui.button("🧹 Temizle").clicked() {
                    self.çıktı = String::new();
                    self.hata_mesajı = None;
                }
            });
        });
    }

    fn sol_panel_göster(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("sol_panel")
            .resizable(true)
            .default_width(150.0)
            .show(ctx, |ui| {
                ui.heading("Menü");
                ui.separator();

                if ui
                    .selectable_label(self.aktif_sekme == Sekme::Düzenleyici, "📝 Kod Editörü")
                    .clicked()
                {
                    self.aktif_sekme = Sekme::Düzenleyici;
                }

                if ui
                    .selectable_label(
                        self.aktif_sekme == Sekme::Görselleştirme,
                        "📊 Görselleştirme",
                    )
                    .clicked()
                {
                    self.aktif_sekme = Sekme::Görselleştirme;
                }

                if ui
                    .selectable_label(self.aktif_sekme == Sekme::Ayarlar, "⚙ Ayarlar")
                    .clicked()
                {
                    self.aktif_sekme = Sekme::Ayarlar;
                }

                if ui
                    .selectable_label(self.aktif_sekme == Sekme::Hakkında, "ℹ Hakkında")
                    .clicked()
                {
                    self.aktif_sekme = Sekme::Hakkında;
                }

                ui.separator();
                ui.heading("Örnek Kodlar");

                for (_i, örnek) in self.örnek_kodlar.iter().enumerate() {
                    if ui.button(&örnek.başlık).clicked() {
                        self.kod = örnek.kod.clone();
                    }
                }
            });
    }

    fn kod_editörü_göster(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Kod Editörü");

            if let Some(hata) = &self.hata_mesajı {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("❌ Hata: {}", hata))
                            .color(TURK_KIRMIZI)
                            .strong(),
                    );
                });
            }

            let mut kod = self.kod.clone();
            let _response = egui::ScrollArea::vertical()
                .id_source("kod_editoru")
                .show(ui, |ui| {
                    ui.add_sized(
                        ui.available_size(),
                        egui::TextEdit::multiline(&mut kod)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace),
                    )
                });

            if kod != self.kod {
                self.kod = kod;
            }
        });
    }

    fn görselleştirme_göster(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Model Görselleştirme");

            if self.modeller.is_empty() {
                ui.label("Görselleştirilecek model bulunamadı. Lütfen önce bir model tanımlayıp ayrıştırın.");
                if ui.button("Mevcut kodu ayrıştır").clicked() {
                    self.kodu_ayrıştır();
                }
            } else {
                for model in &self.modeller {
                    ui.collapsing(&model.isim, |ui| {
                        let toplam_katman = model.katmanlar.len();
                        
                        ui.label(format!("Optimizer: {}", model.optimizer));
                        ui.label(format!("Kayıp Fonksiyonu: {}", model.kayıp_fonksiyonu));
                        ui.separator();
                        
                        ui.label("Model Mimarisi:");
                        let ağaç_yüksekliği = 200.0;
                        let genişlik = ui.available_width();
                        
                        // Ağaç diyagramı
                        egui::Frame::canvas(ui.style()).show(ui, |ui| {
                            let (response, painter) = ui.allocate_painter(
                                Vec2::new(genişlik, ağaç_yüksekliği),
                                egui::Sense::hover(),
                            );
                            
                            let rect = response.rect;
                            let katman_aralığı = rect.width() / (toplam_katman as f32 + 1.0);
                            let x_başlangıç = rect.left() + katman_aralığı;
                            
                            // Giriş katmanı
                            let giriş_x = x_başlangıç;
                            let giriş_y = rect.center().y;
                            let giriş_nöron = model.katmanlar[0].nöron_sayısı;
                            let düğüm_renk = TURK_MAVİ;
                            let düğüm_çap = 30.0;
                            
                            painter.circle_filled(
                                egui::pos2(giriş_x, giriş_y),
                                düğüm_çap,
                                düğüm_renk,
                            );
                            
                            painter.text(
                                egui::pos2(giriş_x, giriş_y),
                                egui::Align2::CENTER_CENTER,
                                format!("{}", giriş_nöron),
                                egui::FontId::proportional(14.0),
                                ui.visuals().text_color(),
                            );
                            
                            let mut önceki_x = giriş_x;
                            let mut önceki_y = giriş_y;
                            
                            // Diğer katmanlar
                            for (i, katman) in model.katmanlar.iter().enumerate().skip(1) {
                                let katman_x = x_başlangıç + katman_aralığı * (i as f32 + 1.0);
                                let katman_y = rect.center().y;
                                
                                // Bağlantı çizgileri
                                painter.line_segment(
                                    [
                                        egui::pos2(önceki_x, önceki_y),
                                        egui::pos2(katman_x, katman_y),
                                    ],
                                    Stroke::new(2.0, TURK_GRİ),
                                );
                                
                                // Katman düğümü
                                painter.circle_filled(
                                    egui::pos2(katman_x, katman_y),
                                    düğüm_çap,
                                    if i == toplam_katman - 1 {
                                        TURK_YEŞİL
                                    } else {
                                        TURK_MOR
                                    },
                                );
                                
                                // Nöron sayısı
                                painter.text(
                                    egui::pos2(katman_x, katman_y),
                                    egui::Align2::CENTER_CENTER,
                                    format!("{}", katman.nöron_sayısı),
                                    egui::FontId::proportional(14.0),
                                    ui.visuals().text_color(),
                                );
                                
                                // Aktivasyon fonksiyonu
                                painter.text(
                                    egui::pos2(katman_x, katman_y + düğüm_çap + 5.0),
                                    egui::Align2::CENTER_TOP,
                                    &katman.aktivasyon,
                                    egui::FontId::proportional(12.0),
                                    ui.visuals().weak_text_color(),
                                );
                                
                                önceki_x = katman_x;
                                önceki_y = katman_y;
                            }
                        });
                        
                        ui.separator();
                        
                        // Katman detayları tablosu
                        egui::Grid::new(format!("katmanlar_{}", model.isim))
                            .striped(true)
                            .spacing([10.0, 5.0])
                            .show(ui, |ui| {
                                ui.label(RichText::new("Katman").strong());
                                ui.label(RichText::new("Nöron Sayısı").strong());
                                ui.label(RichText::new("Aktivasyon").strong());
                                ui.end_row();
                                
                                for (i, katman) in model.katmanlar.iter().enumerate() {
                                    ui.label(format!("Katman {}", i + 1));
                                    ui.label(format!("{}", katman.nöron_sayısı));
                                    ui.label(&katman.aktivasyon);
                                    ui.end_row();
                                }
                            });
                    });
                }
            }
        });
    }

    fn ayarlar_göster(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Ayarlar");
            
            ui.checkbox(&mut self.tema_koyu, "Koyu Tema");
            
            ui.separator();
            ui.heading("Kod Editörü Ayarları");
            // Editör ayarları buraya eklenebilir
            
            ui.separator();
            ui.heading("Çalışma Zamanı Ayarları");
            // Çalışma ayarları buraya eklenebilir
        });
    }

    fn hakkında_göster(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(RichText::new("TURK Programlama Dili").color(TURK_MAVİ).strong());
            
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.label("Versiyon: ");
                ui.label(RichText::new("0.1.0").strong());
            });
            
            ui.separator();
            
            ui.label("TURK, Türkçe makine öğrenmesi ve yapay zeka uygulamaları geliştirmek için tasarlanmış açık kaynaklı bir programlama dilidir.");
            
            ui.separator();
            
            ui.heading("Özellikler");
            ui.label("• Türkçe sözdizimi");
            ui.label("• Makine öğrenmesi model tanımlamaları");
            ui.label("• Veri işleme ve görselleştirme");
            ui.label("• Kod analizi ve hata ayıklama");
            
            ui.separator();
            
            ui.heading("Bağlantılar");
            ui.hyperlink_to("GitHub", "https://github.com/turk-lang/turk");
            ui.hyperlink_to("Dokümantasyon", "https://turk-lang.github.io/turk/docs");
            
            ui.separator();
            
            ui.label("© 2023 TURK Dili Takımı. MIT Lisansı altında dağıtılmaktadır.");
        });
    }

    fn alt_panel_göster(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("alt_panel")
            .resizable(true)
            .default_height(200.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Çıktı");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("🧹 Temizle").clicked() {
                            self.çıktı = String::new();
                        }
                    });
                });
                ui.separator();

                egui::ScrollArea::vertical()
                    .id_source("çıktı_alanı")
                    .show(ui, |ui| {
                        ui.add_sized(
                            ui.available_size(),
                            egui::TextEdit::multiline(&mut self.çıktı.clone())
                                .font(egui::TextStyle::Monospace)
                                .interactive(false),
                        );
                    });
            });
    }

    fn kodu_ayrıştır(&mut self) {
        // TURK kodu ayrıştırma
        
        // Kodu tokenize et
        let mut lexer = Token::lexer(&self.kod);
        let mut tokens = Vec::new();
        
        while let Some(token) = lexer.next() {
            match token {
                Ok(token) => {
                    let slice = lexer.slice().to_string();
                    tokens.push((token, slice));
                }
                Err(_) => {
                    self.hata_mesajı = Some("Sözdizimine ayrıştırma hatası oluştu.".to_string());
                    return;
                }
            }
        }
        
        let mut parser = ModelParser::new(tokens);
        match parser.parse() {
            Ok(program) => {
                self.program = Some(program);
                self.hata_mesajı = None;
                self.çıktı = format!("Program başarıyla ayrıştırıldı.\n{:#?}", self.program.as_ref().unwrap());
                
                // Model görselleştirmeleri oluştur
                self.modeller = Vec::new();
                if let Some(ref program) = self.program {
                    for deyim in &program.deyimler {
                        if let Deyim::ModelTanimi { isim, ozellikler } = deyim {
                            let mut katmanlar = Vec::new();
                            
                            for (_i, katman) in ozellikler.katmanlar.iter().enumerate() {
                                let aktivasyon = match katman.aktivasyon {
                                    AktivasyonFonksiyonu::Relu => "ReLU",
                                    AktivasyonFonksiyonu::Sigmoid => "Sigmoid",
                                    AktivasyonFonksiyonu::Tanh => "Tanh",
                                    AktivasyonFonksiyonu::Softmax => "Softmax",
                                };
                                
                                katmanlar.push(KatmanÖzellikleri {
                                    nöron_sayısı: katman.noron_sayisi as usize,
                                    aktivasyon: aktivasyon.to_string(),
                                });
                            }
                            
                            let optimizer = match &ozellikler.optimizer {
                                OptimizerTipi::SGD(lr) => format!("SGD (lr={})", lr),
                                OptimizerTipi::Adam(lr) => format!("Adam (lr={})", lr),
                                OptimizerTipi::AdamW { lr, weight_decay, beta1: _, beta2: _ } => 
                                    format!("AdamW (lr={}, weight_decay={})", lr, weight_decay),
                            };
                            
                            let kayıp_fonksiyonu = match ozellikler.kayip_fonksiyonu {
                                turk_lang::ast::KayipFonksiyonu::MSE => "MSE",
                                turk_lang::ast::KayipFonksiyonu::CrossEntropy => "Cross Entropy",
                                turk_lang::ast::KayipFonksiyonu::BinaryCrossEntropy => "Binary Cross Entropy",
                            };
                            
                            self.modeller.push(ModelGörselleştirme {
                                isim: isim.clone(),
                                katmanlar,
                                optimizer,
                                kayıp_fonksiyonu: kayıp_fonksiyonu.to_string(),
                            });
                        }
                    }
                }
            }
            Err(err) => {
                self.hata_mesajı = Some(format!("Ayrıştırma hatası: {}", err));
                self.program = None;
            }
        }
    }

    fn kodu_çalıştır(&mut self) {
        let kod = self.kod.clone();
        
        // Asenkron çalıştırma için Promise oluştur
        let (sender, promise) = Promise::new();
        self.çalıştırma_sonuçları = Some(promise);
        
        // Arka planda çalıştır
        std::thread::spawn(move || {
            // TURK kodu ayrıştırma
            
            // Kodu tokenize et
            let mut lexer = Token::lexer(&kod);
            let mut tokens = Vec::new();
            
            while let Some(token) = lexer.next() {
                match token {
                    Ok(token) => {
                        let slice = lexer.slice().to_string();
                        tokens.push((token, slice));
                    }
                    Err(_) => {
                        return sender.send(Err("Sözdizimine ayrıştırma hatası oluştu.".to_string()));
                    }
                }
            }
            
            let mut parser = ModelParser::new(tokens);
            match parser.parse() {
                Ok(program) => {
                    // TODO: Gerçek çalıştırma kodu burada olacak
                    let sonuç = format!("Program başarıyla çalıştırıldı.\nSonuç: {:?}", program);
                    sender.send(Ok(sonuç));
                }
                Err(err) => {
                    sender.send(Err(format!("Ayrıştırma hatası: {}", err)));
                }
            }
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    env_logger::init();
    
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::Vec2::new(1280.0, 720.0)),
        min_window_size: Some(egui::Vec2::new(800.0, 600.0)),
        resizable: true,
        vsync: true,
        ..Default::default()
    };
    
    eframe::run_native(
        "TURK Programlama Dili",
        options,
        Box::new(|_cc| Box::new(TurkGui::default()))
    )
} 
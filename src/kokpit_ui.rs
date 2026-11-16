//! Göktürk Kokpit — çizim, animasyon ve komut yardımcıları

use eframe::egui;
use std::time::{Duration, Instant};

pub const SPLASH_SURESI: Duration = Duration::from_millis(600);
pub const VEKIL_TARAMA_SURE: Duration = Duration::from_secs(5);

pub const CAM_RADIUS: f32 = 8.0;
pub const KENAR_INCE: f32 = 0.5;

pub const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(4, 4, 6);
pub const GLASS_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(12, 12, 16, 235);
pub const MICRO_CYAN: egui::Color32 = egui::Color32::from_rgb(0, 240, 200);
pub const DEEP_MOR: egui::Color32 = egui::Color32::from_rgb(48, 32, 72);
pub const TUNC_BRONZ: egui::Color32 = egui::Color32::from_rgb(140, 108, 68);
pub const ACCENT_GREEN: egui::Color32 = egui::Color32::from_rgb(57, 255, 136);
pub const ACCENT_ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 110, 30);
pub const TERMINAL_DIM: egui::Color32 = egui::Color32::from_rgb(90, 95, 110);
pub const TERMINAL_USER: egui::Color32 = egui::Color32::from_rgb(175, 180, 195);
pub const LOG_YELLOW: egui::Color32 = egui::Color32::from_rgb(255, 220, 60);
pub const LOG_NEON_ZINCIR: egui::Color32 = egui::Color32::from_rgb(255, 235, 80);
pub const LOG_CYAN: egui::Color32 = egui::Color32::from_rgb(0, 230, 255);
pub const NEON_CYAN: egui::Color32 = egui::Color32::from_rgb(0, 255, 220);
pub const NEON_ALERT: egui::Color32 = egui::Color32::from_rgb(255, 0, 55);

pub const TURK_TAMGA: &str = "𐱅𐰇𐰼𐰰";

/// egui LTR painter — ekranda 𐱅𐰇𐰼𐰰 (T→K) soldan sağa görünsün
pub const TURK_TAMGA_GOSTER: &str = "𐰰𐰼𐰇𐱅";

/// Metindeki TÜRK tamgasını ekran sırasına çevirir
pub fn turk_tamga_duzelt(metin: &str) -> String {
    metin.replace(TURK_TAMGA, TURK_TAMGA_GOSTER)
}

/// Pencere simgesi (Göktürk tamga, çoklu çözünürlük)
pub fn pencere_simgesi() -> egui::IconData {
    eframe::icon_data::from_png_bytes(include_bytes!("../assets/icons/turk_tamga_256.png"))
        .expect("turk_tamga ikon PNG")
}

pub const BILINEN_KOMUTLAR: &[&str] = &[
    "sistem durumu",
    "not al",
    "cargo check",
    "proje refaktör",
    "proje refaktor",
    "dosya listele",
    "vekil otonom aç",
    "vekil otonom kapat",
    "evet optimize et",
];

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum LogFiltre {
    #[default]
    Tum,
    Vekil,
    Derle,
    Sohbet,
}

pub fn olcek(ctx: &egui::Context) -> f32 {
    (ctx.pixels_per_point() * 1.18).clamp(1.12, 2.35)
}

pub fn olcekli(ctx: &egui::Context, v: f32) -> f32 {
    v * olcek(ctx)
}

pub fn renk_lerp(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    egui::Color32::from_rgb(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
    )
}

pub fn karakter_runik_mi(c: char) -> bool {
    ('\u{10C00}'..='\u{10C4F}').contains(&c)
}

/// Emoji + rune + Latin parçalarını ayrı renk/ağırlıkla çizer.
pub fn runik_etiket_ciz(ui: &mut egui::Ui, tam: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let mut buf = String::new();
        let mut modu = 0_u8; // 0=latin, 1=runik
        for c in tam.chars().chain(std::iter::once('\0')) {
            let is_r = karakter_runik_mi(c);
            let yeni = if is_r { 1 } else { 0 };
            if c == '\0' || yeni != modu {
                if !buf.is_empty() {
                    let text = egui::RichText::new(&buf).monospace();
                    ui.label(match modu {
                        1 => text
                            .strong()
                            .size(17.0)
                            .color(MICRO_CYAN.linear_multiply(0.98)),
                        _ => text
                            .size(14.0)
                            .strong()
                            .color(egui::Color32::from_rgb(185, 192, 205)),
                    });
                    buf.clear();
                }
                modu = yeni;
            }
            if c != '\0' {
                buf.push(c);
            }
        }
    });
}

/// Sol/sağ/orta panel üst başlıkları — Latin etiket daha parlak.
pub fn panel_baslik_ciz(ui: &mut egui::Ui, tam: &str) {
    runik_etiket_ciz(ui, tam);
}

pub fn fontlari_yukle(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    #[cfg(windows)]
    {
        if let Ok(windir) = std::env::var("WINDIR") {
            let segoe = format!(r"{windir}\Fonts\segoeui.ttf");
            if let Ok(bytes) = std::fs::read(&segoe) {
                fonts
                    .font_data
                    .insert("segoe_latin".to_owned(), egui::FontData::from_owned(bytes));
                for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                    if let Some(list) = fonts.families.get_mut(&family) {
                        list.insert(0, "segoe_latin".to_owned());
                    }
                }
            }
        }
    }

    fonts.font_data.insert(
        "gokturk".to_owned(),
        egui::FontData::from_static(include_bytes!(
            "../assets/fonts/NotoSansOldTurkic-Regular.ttf"
        )),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts
            .families
            .entry(family)
            .or_default()
            .push("gokturk".to_owned());
    }
    ctx.set_fonts(fonts);
}

pub fn tema_uygula(ctx: &egui::Context) {
    fontlari_yukle(ctx);
    let mut vis = egui::Visuals::dark();
    vis.panel_fill = PANEL_BG;
    vis.window_fill = PANEL_BG;
    vis.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(18, 18, 22);
    vis.widgets.inactive.bg_fill = egui::Color32::from_rgb(24, 24, 30);
    vis.widgets.hovered.bg_fill = egui::Color32::from_rgb(32, 32, 40);
    vis.widgets.active.bg_fill = egui::Color32::from_rgb(0, 70, 90);
    vis.selection.bg_fill = MICRO_CYAN.linear_multiply(0.18);
    vis.override_text_color = Some(egui::Color32::from_rgb(205, 208, 218));
    ctx.set_visuals(vis);
    let mut style = (*ctx.style()).clone();
    let s = olcek(ctx);
    style.spacing.item_spacing = egui::vec2(10.0 * s, 8.0 * s);
    style.spacing.window_margin = egui::Margin::same(12.0 * s);
    style.spacing.button_padding = egui::vec2(10.0 * s, 6.0 * s);
    style
        .text_styles
        .insert(egui::TextStyle::Body, egui::FontId::proportional(16.0 * s));
    style.text_styles.insert(
        egui::TextStyle::Monospace,
        egui::FontId::monospace(15.0 * s),
    );
    style
        .text_styles
        .insert(egui::TextStyle::Small, egui::FontId::proportional(14.0 * s));
    style.text_styles.insert(
        egui::TextStyle::Button,
        egui::FontId::proportional(15.0 * s),
    );
    style.text_styles.insert(
        egui::TextStyle::Heading,
        egui::FontId::proportional(20.0 * s),
    );
    ctx.set_style(style);
}

pub fn arka_plan_ciz(ctx: &egui::Context, _anim: f32) {
    let ekran = ctx.screen_rect();
    let painter = ctx.layer_painter(egui::LayerId::background());
    painter.rect_filled(ekran, 0.0, PANEL_BG);
    let cx = ekran.center();
    let r = ekran.width().max(ekran.height()) * 0.72;
    painter.circle_filled(cx, r, egui::Color32::from_rgba_premultiplied(8, 10, 14, 18));
    for (dx, dy) in [(-1.0_f32, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let k = egui::pos2(
            if dx < 0.0 {
                ekran.left()
            } else {
                ekran.right()
            },
            if dy < 0.0 {
                ekran.top()
            } else {
                ekran.bottom()
            },
        );
        painter.circle_filled(
            k,
            r * 0.55,
            egui::Color32::from_rgba_premultiplied(0, 0, 0, 28),
        );
    }
}

pub fn cam_panel_ciz(painter: &egui::Painter, rect: egui::Rect, anim: f32) {
    painter.rect_filled(rect, CAM_RADIUS, GLASS_FILL);
    let t = (anim * 0.45).sin() * 0.5 + 0.5;
    let cyan = renk_lerp(
        MICRO_CYAN.linear_multiply(0.22),
        MICRO_CYAN.linear_multiply(0.42),
        t,
    );
    let tunç = renk_lerp(
        TUNC_BRONZ.linear_multiply(0.35),
        TUNC_BRONZ.linear_multiply(0.55),
        1.0 - t,
    );
    let r = rect.expand(-0.25);
    let rad = CAM_RADIUS;
    painter.line_segment(
        [
            egui::pos2(r.left() + rad, r.top()),
            egui::pos2(r.right() - rad, r.top()),
        ],
        egui::Stroke::new(KENAR_INCE, cyan),
    );
    painter.line_segment(
        [
            egui::pos2(r.left(), r.top() + rad),
            egui::pos2(r.left(), r.bottom() - rad),
        ],
        egui::Stroke::new(KENAR_INCE, cyan.linear_multiply(0.75)),
    );
    painter.line_segment(
        [
            egui::pos2(r.left() + rad, r.bottom()),
            egui::pos2(r.right() - rad, r.bottom()),
        ],
        egui::Stroke::new(KENAR_INCE, tunç),
    );
    painter.line_segment(
        [
            egui::pos2(r.right(), r.top() + rad),
            egui::pos2(r.right(), r.bottom() - rad),
        ],
        egui::Stroke::new(KENAR_INCE, renk_lerp(cyan, tunç, t)),
    );
}

pub fn gokturk_tamga_filigrani(painter: &egui::Painter, rect: egui::Rect, anim: f32, otonom: bool) {
    let base = if otonom { 0.04 } else { 0.035 };
    let pulse = if otonom {
        base + (anim * 1.2).sin().abs() * 0.045
    } else {
        base + (anim * 0.35).sin().abs() * 0.012
    };
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        TURK_TAMGA_GOSTER,
        egui::FontId::proportional(148.0),
        MICRO_CYAN.linear_multiply(pulse),
    );
}

pub fn telemetri_rengi(oran: f32, kritik: bool) -> egui::Color32 {
    let o = oran.clamp(0.0, 1.0);
    if kritik || o >= 0.90 {
        return NEON_ALERT.linear_multiply(0.82);
    }
    if o >= 0.72 {
        return renk_lerp(ACCENT_GREEN, LOG_YELLOW, (o - 0.72) / 0.18);
    }
    ACCENT_GREEN.linear_multiply(0.72)
}

pub fn minimal_telemetry_ciz(
    ui: &mut egui::Ui,
    etiket: &str,
    oran: f32,
    yuzde: &str,
    kritik: bool,
    anim: f32,
) {
    let o = oran.clamp(0.0, 1.0);
    ui.horizontal(|ui| {
        runik_etiket_ciz(ui, etiket);
        let bar_gen = (ui.available_width() - 52.0).max(60.0);
        let (rect, _) = ui.allocate_exact_size(egui::vec2(bar_gen, 14.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        let cy = rect.center().y;
        let iz_renk = egui::Color32::from_rgba_premultiplied(35, 35, 42, 160);
        painter.line_segment(
            [egui::pos2(rect.left(), cy), egui::pos2(rect.right(), cy)],
            egui::Stroke::new(0.5, iz_renk),
        );
        let dolu_x = rect.left() + rect.width() * o;
        let renk = telemetri_rengi(o, kritik);
        if o > 0.001 {
            painter.line_segment(
                [egui::pos2(rect.left(), cy), egui::pos2(dolu_x, cy)],
                egui::Stroke::new(0.5, renk),
            );
            let glow_r = if kritik || o >= 0.9 {
                2.5 + (anim * 6.0).sin().abs() * 2.0
            } else {
                2.0
            };
            painter.circle_filled(egui::pos2(dolu_x, cy), glow_r, renk.linear_multiply(0.18));
            painter.circle_filled(egui::pos2(dolu_x, cy), 1.5, renk.linear_multiply(0.9));
        }
        ui.label(
            egui::RichText::new(yuzde)
                .monospace()
                .size(13.0)
                .color(renk.linear_multiply(0.95)),
        );
    });
}

pub fn vu_metre_ciz(ui: &mut egui::Ui, seviye: f32, anim: f32) {
    let w = ui.available_width().min(120.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(w, 10.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let cy = rect.center().y;
    painter.line_segment(
        [egui::pos2(rect.left(), cy), egui::pos2(rect.right(), cy)],
        egui::Stroke::new(0.5, TERMINAL_DIM.linear_multiply(0.4)),
    );
    let s = (seviye + (anim * 8.0).sin().abs() * 0.08).clamp(0.05, 1.0);
    let x = rect.left() + rect.width() * s;
    painter.line_segment(
        [egui::pos2(rect.left(), cy), egui::pos2(x, cy)],
        egui::Stroke::new(1.0, ACCENT_GREEN.linear_multiply(0.85)),
    );
    painter.circle_filled(egui::pos2(x, cy), 2.0, ACCENT_GREEN.linear_multiply(0.35));
}

pub fn log_satir_rengi(satir: &str) -> egui::Color32 {
    if satir.contains("[ZİNCİR]") || satir.starts_with("[Plan:") {
        LOG_NEON_ZINCIR
    } else if satir.contains("operasyon zinciri başarıyla")
        || (satir.contains("[ZİNCİR]") && satir.contains('✓'))
    {
        ACCENT_GREEN
    } else if satir.contains('✓')
        && (satir.contains("[KOD]")
            || satir.contains("[DERLE]")
            || satir.contains("[REFAKTÖR]")
            || (satir.contains("[TÜRK Vekil]") && satir.contains("optimize edildi")))
    {
        ACCENT_GREEN
    } else if satir.contains('▸')
        && (satir.contains("[KOD]")
            || satir.contains("[DERLE]")
            || satir.contains("[REFAKTÖR]")
            || satir.contains("[TÜRK Vekil]"))
    {
        LOG_YELLOW
    } else if satir.contains('🚨') {
        NEON_ALERT
    } else if satir.contains('→') || satir.contains('←') {
        LOG_CYAN
    } else if satir.contains("HATA") || satir.contains('✗') {
        egui::Color32::from_rgb(255, 90, 110)
    } else {
        TERMINAL_DIM
    }
}

pub fn log_filtre_uygun(satir: &str, filtre: LogFiltre) -> bool {
    match filtre {
        LogFiltre::Tum => true,
        LogFiltre::Vekil => {
            satir.contains("[TÜRK Vekil]") || satir.contains("Vekil") || satir.contains('🚨')
        }
        LogFiltre::Derle => {
            satir.contains("[DERLE]")
                || satir.contains("[KOD]")
                || satir.contains("[REFAKTÖR]")
                || satir.contains("[ZİNCİR]")
                || satir.starts_with("[Plan:")
                || satir.contains("Eğitim")
                || satir.contains("Epoch")
        }
        LogFiltre::Sohbet => {
            satir.contains('→')
                || satir.contains('←')
                || satir.contains('🎙')
                || satir.contains("Kuyruk")
                || satir.contains("operasyon zinciri")
        }
    }
}

pub fn tab_oneri(metin: &str) -> Option<String> {
    let t = metin.trim();
    if t.is_empty() {
        return None;
    }
    let lower = t.to_lowercase();
    BILINEN_KOMUTLAR
        .iter()
        .find(|k| k.starts_with(&lower) && k.len() > lower.len())
        .map(|k| k.to_string())
}

pub fn komut_satir_sayisi(metin: &str) -> usize {
    metin.lines().count().max(1).min(4)
}

pub fn splash_ciz(ctx: &egui::Context, baslangic: Instant, anim: f32) {
    let gecen = baslangic.elapsed().as_secs_f32();
    let sure = SPLASH_SURESI.as_secs_f32();
    if gecen >= sure {
        return;
    }
    let alpha = if gecen > sure * 0.55 {
        1.0 - ((gecen - sure * 0.55) / (sure * 0.45)).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let ekran = ctx.screen_rect();
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("turk_splash"),
    ));
    painter.rect_filled(
        ekran,
        0.0,
        egui::Color32::from_rgba_premultiplied(4, 4, 6, (220.0 * alpha) as u8),
    );
    let pulse = 0.55 + (anim * 2.0).sin() * 0.15;
    painter.text(
        ekran.center(),
        egui::Align2::CENTER_CENTER,
        TURK_TAMGA_GOSTER,
        egui::FontId::proportional(112.0),
        MICRO_CYAN.linear_multiply(pulse * alpha),
    );
    painter.text(
        ekran.center() + egui::vec2(0.0, 80.0),
        egui::Align2::CENTER_CENTER,
        "KADİM SİBER KOKPİT",
        egui::FontId::monospace(16.0),
        TERMINAL_DIM.linear_multiply(alpha),
    );
}

#[allow(dead_code)]
pub fn kritik_banner_ciz(ctx: &egui::Context, aktif: bool, anim: f32, ust_ofset: f32) {
    if !aktif {
        return;
    }
    let ekran = ctx.screen_rect();
    let h = 20.0;
    let alpha = 0.55 + (anim * 4.0).sin().abs() * 0.25;
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Middle,
        egui::Id::new("turk_kritik_banner"),
    ));
    let rect = egui::Rect::from_min_size(
        egui::pos2(ekran.left(), ekran.top() + ust_ofset),
        egui::vec2(ekran.width(), h),
    );
    painter.rect_filled(rect, 0.0, NEON_ALERT.linear_multiply(0.20 * alpha));
    painter.line_segment(
        [rect.left_top(), rect.right_top()],
        egui::Stroke::new(0.5, NEON_ALERT.linear_multiply(0.55 * alpha)),
    );
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "⚠ 𐰴 KRİTİK YÜK — Disk / sistem kaynakları sınırda",
        egui::FontId::monospace(10.5),
        NEON_ALERT.linear_multiply(0.92),
    );
}

pub fn otuken_pulse_ciz(ui: &mut egui::Ui, pulse: f32) {
    if pulse <= 0.0 {
        return;
    }
    let rect = ui.max_rect();
    let painter = ui.painter();
    let a = (pulse * std::f32::consts::PI).sin().max(0.0) * 0.35;
    painter.rect_stroke(
        rect.shrink(2.0),
        4.0,
        egui::Stroke::new(1.5, ACCENT_GREEN.linear_multiply(a)),
    );
}

use env_logger::Builder;
use log::{info, warn, LevelFilter};
use ndarray::arr2;
use turk::error::Result;
/// DSL Örnek — main.turk dosyasını okuyup, XOR verisiyle gerçek eğitim yapar.
///
/// Çalıştırmak için:
///   cargo run --example dsl_ornek
///
/// Kök dizinde main.turk olmak zorunda.
use turk::{
    aktivasyon::Activation,
    initializers::{GlorotUniform, Zeros},
    layer::BatchNormalization,
    model::{EarlyStopping, Sequential},
    optimizer::Adam,
};

fn main() -> Result<()> {
    // ── Loglama ────────────────────────────────────────────────────────────
    Builder::new().filter(None, LevelFilter::Info).init();

    info!("======================================================");
    info!(" TÜRK DSL Örneği — main.turk + XOR + BatchNorm + ES  ");
    info!("======================================================");

    // ── main.turk dosyasını oku ────────────────────────────────────────────
    let turk_dosyasi = "main.turk";
    let _icerik = std::fs::read_to_string(turk_dosyasi).unwrap_or_else(|e| {
        warn!("{turk_dosyasi} okunamadı ({e}), yerleşik DSL kullanılıyor.");
        String::new()
    });

    if !_icerik.is_empty() {
        info!("--- main.turk içeriği ---");
        for satir in _icerik.lines() {
            let s = satir.trim();
            if !s.is_empty() {
                info!("  {s}");
            }
        }
        info!("--- içerik sonu ---");
    }

    // ── XOR veri seti ──────────────────────────────────────────────────────
    //   Girdi: 2 özellik   Çıktı: 1 sınıf
    let x_train = arr2(&[[0.0_f64, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let y_train = arr2(&[[0.0_f64], [1.0], [1.0], [0.0]]);

    // Doğrulama için aynı veri (küçük örnekte anlamlı)
    let x_val = x_train.clone();
    let y_val = y_train.clone();

    info!(
        "XOR eğitim verisi: {} örnek, {} özellik",
        x_train.nrows(),
        x_train.ncols()
    );

    // ── Model mimarisi (main.turk ile uyumlu) ─────────────────────────────
    //   Dense(2→8, Relu) → BatchNorm(8) → Dense(8→4, Relu) → Dense(4→1, Sigmoid)
    let mut model = Sequential::new();

    let make_dense = |i: usize, o: usize, act: Activation| {
        turk::layer::Dense::new(
            i,
            o,
            act,
            Box::new(GlorotUniform::new()),
            Box::new(Zeros::new()),
            true,
        )
        .unwrap()
    };

    model
        .add(make_dense(2, 8, Activation::Relu))
        .add(BatchNormalization::new(8))
        .add(make_dense(8, 4, Activation::Relu))
        .add(make_dense(4, 1, Activation::Sigmoid))
        .compile(Adam::new(0.05, 0.9, 0.999, 1e-8));

    info!("Model oluşturuldu: Dense(2→8) → BN(8) → Dense(8→4) → Dense(4→1)");
    info!("Optimizer: Adam(lr=0.05)");

    // ── Erken Durdurma (main.turk parametreleri) ──────────────────────────
    //   sabir: 10, min_delta: 0.0001
    let mut es = EarlyStopping::new(10, 0.0001);
    info!("EarlyStopping: patience=10, min_delta=0.0001");

    // ── Eğitim (100 epoch, batch=4, doğrulama verili) ─────────────────────
    info!("--- Eğitim Başlıyor (epoch: 100, batch_boyutu: 4) ---");

    let history = model.fit_with_early_stopping(
        &x_train.view(),
        &y_train.view(),
        100,
        4,
        Some((&x_val.view(), &y_val.view())),
        Some(&mut es),
    )?;

    // ── Sonuçlar ───────────────────────────────────────────────────────────
    info!("--- Eğitim Tamamlandı ---");
    info!(
        "Çalışan epoch: {}/100 (erken durdurma: {})",
        history.epochs_run(),
        if history.epochs_run() < 100 {
            "Evet"
        } else {
            "Hayır"
        }
    );

    if let Some(ilk) = history.train_losses.first() {
        info!("İlk eğitim kaybı  : {ilk:.6}");
    }
    if let Some(son) = history.last_train_loss() {
        info!("Son eğitim kaybı   : {son:.6}");
    }
    if let Some(val) = history.last_val_loss() {
        info!("Son doğrulama kaybı: {val:.6}");
    }

    // ── Tahmin ─────────────────────────────────────────────────────────────
    info!("--- XOR Tahminleri ---");
    let tahminler = model.predict(&x_train.view())?;
    let girdiler = [
        ([0.0_f64, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];
    let mut dogru = 0usize;
    for (idx, (girdi, beklenen)) in girdiler.iter().enumerate() {
        let tahmin: f64 = tahminler[[idx, 0]];
        let sinif: f64 = if tahmin > 0.5 { 1.0 } else { 0.0 };
        let esit = (sinif - *beklenen).abs() < 1e-6_f64;
        if esit {
            dogru += 1;
        }
        let isaret = if esit { "✓" } else { "✗" };
        info!(
            "  [{:.0}, {:.0}] → beklenen={:.0}  tahmin={:.4}  sınıf={:.0}  {}",
            girdi[0], girdi[1], beklenen, tahmin, sinif, isaret
        );
    }
    info!("Doğruluk: {}/{}", dogru, girdiler.len());

    // ── Eğitim geçmişini JSON'a kaydet ────────────────────────────────────
    let gecmis_dosyasi = "egitim_gecmisi.json";
    history.dosyaya_kaydet(gecmis_dosyasi)?;
    info!("╔══════════════════════════════════════════════════════════╗");
    info!(
        "║  ▸ Eğitim istatistikleri '{}' dosyasına  ║",
        gecmis_dosyasi
    );
    info!(
        "║    başarıyla yazıldı!  [ {} epoch │ MSE {:.6} ]          ║",
        history.epochs_run(),
        history.last_train_loss().unwrap_or(0.0)
    );
    info!("╚══════════════════════════════════════════════════════════╝");

    info!("======================================================");
    info!(" Örnek tamamlandı.");
    info!("======================================================");

    Ok(())
}

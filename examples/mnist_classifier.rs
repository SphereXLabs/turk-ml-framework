use colored::*;
use env_logger::Builder;
use indicatif::{ProgressBar, ProgressStyle};
use log::LevelFilter;
use log::{debug, error, info, warn};
use mnist::{Mnist, MnistBuilder};
use ndarray::{Array2, ArrayView2, Axis};
use std::env;
use std::time::Instant;
use turk::{
    Accuracy, Adam, CrossEntropy, Dense, Dropout, F1Score, GlorotUniform, HeNormal, Loss, Metric,
    MinMaxScaler, Optimizer, Precision, RMSProp, Recall, Result, Sequential, StandardScaler, SGD,
};

fn main() -> Result<()> {
    // Loglama sistemini kur
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    info!("MNIST Sınıflandırıcı Örneği Başlatılıyor...");

    // MNIST verisini indir ve yükle
    info!("MNIST veri seti yükleniyor...");
    let mnist = load_mnist_data()?;

    // Eğitim ve test verilerini hazırla
    let (x_train, y_train, x_test, y_test) = prepare_mnist_data(&mnist)?;

    // Modeli oluştur
    info!("Model oluşturuluyor...");
    let mut model = create_model(x_train.shape()[1])?;

    // Model bilgisini yazdır
    info!("Model Yapısı:\n{}", model.summary());

    // Modeli eğit
    info!("Model eğitiliyor...");
    let başlangıç_zamanı = Instant::now();
    let epochs = 5;
    let batch_size = 128;

    let mut progress_bar = ProgressBar::new((epochs * (x_train.shape()[0] / batch_size)) as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    model.fit_with_callback(
        &x_train.view(),
        &y_train.view(),
        epochs,
        batch_size,
        Some(Box::new(
            move |epoch, batch, total_batches, loss, accuracy| {
                progress_bar.inc(1);
                if batch % 100 == 0 {
                    info!(
                        "Epoch {}/{}, Batch {}/{}, Loss: {:.4}, Accuracy: {:.2}%",
                        epoch + 1,
                        epochs,
                        batch,
                        total_batches,
                        loss,
                        accuracy * 100.0
                    );
                }
            },
        )),
    )?;

    let eğitim_süresi = başlangıç_zamanı.elapsed();
    info!("Eğitim tamamlandı! Süre: {:.2?}", eğitim_süresi);

    // Modeli değerlendir
    info!("Model test verisi üzerinde değerlendiriliyor...");
    let değerlendirme_başlangıcı = Instant::now();
    let test_loss = model.evaluate(&x_test.view(), &y_test.view())?;

    // Metrikleri hesapla
    let tahminler = model.predict(&x_test.view())?;
    let accuracy = Accuracy::new();
    let precision = Precision::new();
    let recall = Recall::new();
    let f1_score = F1Score::new();

    let tahmin_sınıfları = tahminler.map_axis(Axis(1), |örnek| {
        örnek
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as f64)
            .unwrap_or(0.0)
    });

    let gerçek_sınıflar = y_test.map_axis(Axis(1), |örnek| {
        örnek
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as f64)
            .unwrap_or(0.0)
    });

    let accuracy_score = accuracy.calculate(&gerçek_sınıflar.view(), &tahmin_sınıfları.view())?;
    let precision_score = precision.calculate(&gerçek_sınıflar.view(), &tahmin_sınıfları.view())?;
    let recall_score = recall.calculate(&gerçek_sınıflar.view(), &tahmin_sınıfları.view())?;
    let f1_score_val = f1_score.calculate(&gerçek_sınıflar.view(), &tahmin_sınıfları.view())?;

    let değerlendirme_süresi = değerlendirme_başlangıcı.elapsed();

    // Sonuçları yazdır
    print_separator();
    println!(
        "{}",
        "  MNIST SINIFLANDIRICI SONUÇLARI  "
            .bold()
            .bright_white()
            .on_blue()
    );
    print_separator();
    println!(
        "{:<15} {}",
        "Test Kaybı:".yellow(),
        format!("{:.4}", test_loss).cyan()
    );
    println!(
        "{:<15} {}",
        "Doğruluk:".yellow(),
        format!("{:.2}%", accuracy_score * 100.0).green()
    );
    println!(
        "{:<15} {}",
        "Kesinlik:".yellow(),
        format!("{:.4}", precision_score).cyan()
    );
    println!(
        "{:<15} {}",
        "Duyarlılık:".yellow(),
        format!("{:.4}", recall_score).cyan()
    );
    println!(
        "{:<15} {}",
        "F1 Skoru:".yellow(),
        format!("{:.4}", f1_score_val).cyan()
    );
    println!(
        "{:<15} {}",
        "Eğitim Süresi:".yellow(),
        format!("{:.2?}", eğitim_süresi).cyan()
    );
    println!(
        "{:<15} {}",
        "Test Süresi:".yellow(),
        format!("{:.2?}", değerlendirme_süresi).cyan()
    );
    print_separator();

    // Modeli kaydet
    info!("Model kaydediliyor...");
    model.save("examples/mnist_model.json")?;
    info!("Model başarıyla kaydedildi: examples/mnist_model.json");

    Ok(())
}

fn print_separator() {
    println!("{}", "=".repeat(50).bright_blue());
}

/// MNIST verisini yükle
fn load_mnist_data() -> Result<Mnist> {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    Ok(mnist)
}

/// MNIST verilerini Turk kütüphanesi için hazırla
fn prepare_mnist_data(
    mnist: &Mnist,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>)> {
    // Eğitim verilerini hazırla
    let train_size = mnist.trn_img.len() / 784;
    let mut x_train = Array2::zeros((train_size, 784));

    for i in 0..train_size {
        for j in 0..784 {
            let pixel = mnist.trn_img[i * 784 + j] as f64 / 255.0;
            x_train[[i, j]] = pixel;
        }
    }

    // Eğitim etiketlerini one-hot encoding'e dönüştür
    let mut y_train = Array2::zeros((train_size, 10));
    for i in 0..train_size {
        let label = mnist.trn_lbl[i] as usize;
        y_train[[i, label]] = 1.0;
    }

    // Test verilerini hazırla
    let test_size = mnist.tst_img.len() / 784;
    let mut x_test = Array2::zeros((test_size, 784));

    for i in 0..test_size {
        for j in 0..784 {
            let pixel = mnist.tst_img[i * 784 + j] as f64 / 255.0;
            x_test[[i, j]] = pixel;
        }
    }

    // Test etiketlerini one-hot encoding'e dönüştür
    let mut y_test = Array2::zeros((test_size, 10));
    for i in 0..test_size {
        let label = mnist.tst_lbl[i] as usize;
        y_test[[i, label]] = 1.0;
    }

    info!(
        "Veri hazırlama tamamlandı. Eğitim: {} örnek, Test: {} örnek",
        train_size, test_size
    );
    Ok((x_train, y_train, x_test, y_test))
}

/// MNIST için model oluştur
fn create_model(input_size: usize) -> Result<Sequential> {
    let mut model = Sequential::new();

    // Giriş katmanı
    let glorot_init = GlorotUniform::new();
    model.add(Dense::new_with_initializer(
        input_size,
        128,
        "relu",
        Box::new(glorot_init),
    ));
    model.add(Dropout::new(0.3));

    // Gizli katman
    let he_init = HeNormal::new();
    model.add(Dense::new_with_initializer(
        128,
        64,
        "relu",
        Box::new(he_init),
    ));
    model.add(Dropout::new(0.2));

    // Çıkış katmanı
    model.add(Dense::new(64, 10, "softmax"));

    // Modeli derle
    let adam = Adam::new(0.001, 0.9, 0.999, 1e-8);
    model.compile(adam, CrossEntropy::new());

    Ok(model)
}

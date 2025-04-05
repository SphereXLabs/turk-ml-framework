use turk_lang::{
    Sequential, Dense, Activation, Dropout, BatchNorm,
    Adam, SGD, AdamW,
    loss::{Loss, CrossEntropy},
    metrics::{Metric, Accuracy, Precision, Recall, F1Score},
    utils::{one_hot_encode, normalize, confusion_matrix},
    TurkError, Result
};
use mnist::{MnistBuilder, Mnist};
use ndarray::{Array2, Axis, s};
use rayon::prelude::*;
use std::time::Instant;
use indicatif::{ProgressBar, ProgressStyle};
use colored::*;

// Veri yükleme ve önişleme fonksiyonu
fn load_mnist_data() -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)> {
    let start = Instant::now();
    println!("{}", "MNIST veriseti yükleniyor...".blue());
    
    // MNIST verisini indir ve yükle
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();
    
    let train_size = trn_img.len() / 784;
    let test_size = tst_img.len() / 784;
    
    // Verileri ndarray formatına dönüştür
    // Paralel işleme ile performansı artır
    let x_train = Array2::from_shape_vec((train_size, 784), trn_img)
        .map_err(|e| TurkError::NdarrayError(e))?;
    
    let x_test = Array2::from_shape_vec((test_size, 784), tst_img)
        .map_err(|e| TurkError::NdarrayError(e))?;

    // Etiketleri one-hot encode yap
    let y_train = one_hot_encode(&trn_lbl, 10)
        .map_err(|e| TurkError::InvalidInput(format!("Etiket dönüşümü hatası: {}", e)))?;
    
    let y_test = one_hot_encode(&tst_lbl, 10)
        .map_err(|e| TurkError::InvalidInput(format!("Etiket dönüşümü hatası: {}", e)))?;
    
    // Verileri normalize et (0-1 aralığına)
    let x_train = x_train.mapv(|x| x as f32 / 255.0);
    let x_test = x_test.mapv(|x| x as f32 / 255.0);
    
    println!("{} ({:?})", "MNIST veriseti yüklendi".green(), start.elapsed());
    println!("  Eğitim seti: {} örnek", train_size);
    println!("  Test seti: {} örnek", test_size);
    
    Ok((x_train, y_train, x_test, y_test))
}

// Model oluşturma fonksiyonu
fn create_model(model_type: &str) -> Result<Sequential> {
    let mut model = Sequential::new();
    
    match model_type {
        "simple" => {
            model.add(Dense::new(784, 128, Activation::Relu))
                 .add(Dense::new(128, 10, Activation::Softmax))
                 .compile(SGD::new(0.01));
        },
        "medium" => {
            model.add(Dense::new(784, 256, Activation::Relu))
                 .add(Dropout::new(0.2))
                 .add(Dense::new(256, 128, Activation::Relu))
                 .add(Dropout::new(0.2))
                 .add(Dense::new(128, 10, Activation::Softmax))
                 .compile(Adam::new(0.001));
        },
        "advanced" => {
            model.add(Dense::new(784, 512, Activation::Relu))
                 .add(BatchNorm::new(512))
                 .add(Dropout::new(0.25))
                 .add(Dense::new(512, 256, Activation::Relu))
                 .add(BatchNorm::new(256))
                 .add(Dropout::new(0.25))
                 .add(Dense::new(256, 128, Activation::Relu))
                 .add(Dropout::new(0.2))
                 .add(Dense::new(128, 10, Activation::Softmax))
                 .compile(AdamW::new(0.001, 0.0001));
        },
        _ => return Err(TurkError::InvalidInput(format!("Bilinmeyen model tipi: {}", model_type))),
    }
    
    Ok(model)
}

fn main() -> Result<()> {
    // Log seviyesini ayarla
    env_logger::init();
    
    println!("\n{}", "TURK DİLİ MNIST SINIFLANDIRICI".bold().magenta());
    println!("{}", "=================================".magenta());
    
    // Verileri yükle
    let (x_train, y_train, x_test, y_test) = load_mnist_data()?;
    
    // Model oluştur
    let model_type = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "medium".to_string());
    
    println!("\n{} '{}' modeli oluşturuluyor...", ">>".blue(), model_type);
    let mut model = create_model(&model_type)?;
    
    // Model özetini göster
    println!("\n{}", model.summary());
    
    // Eğitim parametreleri
    let epochs = 10;
    let batch_size = 128;
    
    // Eğitim veri setinin bir kısmını doğrulama için ayır (%20)
    let train_size = x_train.shape()[0];
    let valid_size = train_size / 5;
    let train_samples = train_size - valid_size;
    
    let x_valid = x_train.slice(s![train_samples.., ..]).to_owned();
    let y_valid = y_train.slice(s![train_samples.., ..]).to_owned();
    
    let x_train = x_train.slice(s![0..train_samples, ..]).to_owned();
    let y_train = y_train.slice(s![0..train_samples, ..]).to_owned();
    
    println!("\n{}", "Eğitim başlıyor...".green().bold());
    println!("  Epochs: {}", epochs);
    println!("  Batch size: {}", batch_size);
    println!("  Eğitim örnekleri: {}", train_samples);
    println!("  Doğrulama örnekleri: {}", valid_size);
    
    // İlerleme çubuğu oluştur
    let progress_style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:50.cyan/blue} {pos:>7}/{len:7} Epoch {msg}")
        .unwrap()
        .progress_chars("##-");
    
    let progress_bar = ProgressBar::new(epochs as u64)
        .with_style(progress_style);
    
    // Modeli eğit
    let start_time = Instant::now();
    let history = model.fit(
        &x_train, 
        &y_train, 
        epochs, 
        batch_size, 
        Some((&x_valid, &y_valid))
    )?;
    
    // Her epoch sonuçlarını göster
    for (epoch, (train_loss, val_loss)) in history.iter().enumerate() {
        progress_bar.set_position((epoch + 1) as u64);
        
        if let Some(val) = val_loss {
            progress_bar.set_message(format!("- loss: {:.4} - val_loss: {:.4}", train_loss, val));
        } else {
            progress_bar.set_message(format!("- loss: {:.4}", train_loss));
        }
        
        // Her epoch'ta dosyaya kaydet
        if (epoch + 1) % 5 == 0 {
            let checkpoint_path = format!("model_checkpoint_epoch_{}.json", epoch + 1);
            model.save(&checkpoint_path)?;
            println!("\nModel kaydedildi: {}", checkpoint_path);
        }
    }
    
    progress_bar.finish_with_message(format!("Eğitim tamamlandı! Süre: {:?}", start_time.elapsed()));
    
    // Modeli kaydet
    let model_path = "mnist_model.json";
    model.save(model_path)?;
    println!("\nModel kaydedildi: {}", model_path);
    
    // Modeli test et
    println!("\n{}", "Model test ediliyor...".blue());
    let start_eval = Instant::now();
    
    // Metrikleri ayarla
    let accuracy = Accuracy::new(0.5);
    let precision = Precision::new(0.5);
    let recall = Recall::new(0.5);
    let f1 = F1Score::new(0.5);
    
    let metrics = model.evaluate(&x_test, &y_test, &[&accuracy, &precision, &recall, &f1])?;
    
    println!("{} ({:?})", "Test tamamlandı".green(), start_eval.elapsed());
    
    // Sonuçları göster
    println!("\n{}", "METRİKLER".bold().yellow());
    println!("{}", "=========".yellow());
    
    for (name, value) in metrics {
        println!("  {}: {:.6}", name, value);
    }
    
    // Confusion matrix
    println!("\n{}", "KARIŞIKLIK MATRİSİ".bold().yellow());
    println!("{}", "================".yellow());
    
    let y_pred = model.predict(&x_test)?;
    let pred_labels: Vec<usize> = y_pred
        .axis_iter(Axis(0))
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect();
    
    let true_labels: Vec<usize> = y_test
        .axis_iter(Axis(0))
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect();
    
    let cm = confusion_matrix(&pred_labels, &true_labels, 10)?;
    
    // Karışıklık matrisini güzel göster
    println!("      0     1     2     3     4     5     6     7     8     9");
    println!("    --------------------------------------------------");
    
    for (i, row) in cm.outer_iter().enumerate() {
        print!("{}  |", i);
        for val in row.iter() {
            let val_str = format!("{:5}", val);
            if *val > 0 {
                print!("{}", val_str.green());
            } else {
                print!("{}", val_str);
            }
        }
        println!();
    }
    
    println!("\n{}", "İŞLEM TAMAMLANDI".bold().green());
    Ok(())
} 
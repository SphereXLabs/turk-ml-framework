use anyhow::Result;
use mnist::{MnistBuilder, Mnist};
use ndarray::{Array2, s};
use std::time::Instant;
use colored::*;
use turk_lang::{
    Sequential, Dense, Dropout, BatchNorm, Activation, 
    Adam,
    metrics::Accuracy,
};
use env_logger;
use log::info;

fn main() -> Result<()> {
    // Log seviyesini ayarla
    env_logger::init();
    
    println!("{}", "MNIST El Yazısı Rakam Sınıflandırıcı".green().bold());
    println!("{}", "==========================================".green());
    
    // MNIST veri setini yükle
    println!("MNIST veri seti yükleniyor...");
    let start = Instant::now();
    
    let Mnist {
        trn_img, trn_lbl,
        val_img, val_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50000)
        .validation_set_length(10000)
        .test_set_length(10000)
        .finalize();
    
    // Veri setini yeniden şekillendir
    let train_data = ndarray::Array3::from_shape_vec((50000, 28, 28), trn_img)?
        .map(|x| *x as f32 / 255.0);
    let train_labels = Array1::from_vec(trn_lbl);
    
    let val_data = ndarray::Array3::from_shape_vec((10000, 28, 28), val_img)?
        .map(|x| *x as f32 / 255.0);
    let val_labels = Array1::from_vec(val_lbl);
    
    println!("Veri yükleme tamamlandı! ({:.2}s)", start.elapsed().as_secs_f32());
    
    // Verileri düzleştir
    println!("Veriler ön işleniyor...");
    let start = Instant::now();
    
    // Paralel işleme kullanarak verileri düzleştir
    let train_data = flatten_images(&train_data);
    let val_data = flatten_images(&val_data);
    
    // Etiketleri one-hot encode et
    let train_labels = one_hot_encode(&train_labels, 10)?;
    let val_labels = one_hot_encode(&val_labels, 10)?;
    
    println!("Ön işleme tamamlandı! ({:.2}s)", start.elapsed().as_secs_f32());
    println!("Eğitim veri şekli: {:?}", train_data.shape());
    println!("Doğrulama veri şekli: {:?}", val_data.shape());
    
    // Model oluştur
    println!("\n{}", "Model oluşturuluyor...".green());
    let mut model = Sequential::new();
    
    // Model mimarisi
    model.add(Dense::new(784, 128, Activation::Relu));
    model.add(Dropout::new(0.2));
    model.add(BatchNorm::new(128));
    model.add(Dense::new(128, 64, Activation::Relu));
    model.add(Dropout::new(0.2));
    model.add(Dense::new(64, 10, Activation::Softmax));
    
    // Optimizer seçimi
    model.compile(Adam::new(0.001));
    
    // Model özetini göster
    println!("{}", model.summary());
    
    // Hiperparametreler
    let epochs = 5;
    let batch_size = 32;
    
    // Modeli eğit
    info!("Model eğitimi başlatılıyor...");
    let history = model.fit(
        &train_data, 
        &train_labels,
        epochs, 
        batch_size,
        Some((&val_data, &val_labels))
    ).unwrap();
    
    info!("Eğitim tamamlandı!");
    
    println!("Eğitim süresi: {:.2}s", start.elapsed().as_secs_f32());
    
    // Eğitim geçmişini göster
    println!("\n{}", "Eğitim Geçmişi".green());
    println!("{}", "=============".green());
    for (epoch, (train_loss, val_loss)) in history.iter().enumerate() {
        let val_loss_str = match val_loss {
            Some(val) => format!("{:.4}", val),
            None => "N/A".to_string(),
        };
        println!("Epoch {}: Train Loss = {:.4}, Val Loss = {}", epoch + 1, train_loss, val_loss_str);
    }
    
    // Modeli değerlendir
    println!("\n{}", "Model değerlendiriliyor...".green());
    let start = Instant::now();
    
    // Accuracy metriki oluştur
    let accuracy = Accuracy::new();
    
    // Değerlendirme
    let metrics = model.evaluate(&val_data, &val_labels, &[&accuracy])?;
    
    println!("Değerlendirme tamamlandı! ({:.2}s)", start.elapsed().as_secs_f32());
    
    // Metrikleri göster
    println!("\n{}", "Değerlendirme Sonuçları".green());
    println!("{}", "=====================".green());
    for (name, value) in metrics {
        println!("{}: {:.4}", name, value);
    }
    
    // Modeli kaydet
    println!("\n{}", "Model kaydediliyor...".green());
    model.save("mnist_model.json")?;
    println!("Model başarıyla kaydedildi: mnist_model.json");
    
    // Birkaç örnek tahmin yap
    println!("\n{}", "Örnek Tahminler".green());
    println!("{}", "===============".green());
    
    // Rastgele 5 örnek seç
    let indices = [42, 123, 456, 789, 890];
    
    for &idx in &indices {
        let sample = val_data.slice(s![idx..idx+1, ..]).to_owned();
        let true_label = val_labels.slice(s![idx..idx+1, ..]).to_owned();
        
        // Modeli çalıştır
        let prediction = model.predict(&sample)?;
        
        // En yüksek olasılıklı sınıfı bul
        let pred_class = get_argmax(&prediction);
        let true_class = get_argmax(&true_label);
        
        // Sonucu göster
        let result_text = if pred_class == true_class {
            format!("Doğru: {} ✓", pred_class).green()
        } else {
            format!("Yanlış: {} (gerçek: {})", pred_class, true_class).red()
        };
        
        println!("Örnek {}: {}", idx, result_text);
    }
    
    Ok(())
}

// Görüntü verilerini düzleştir (paralel)
fn flatten_images(images: &ndarray::Array3<f32>) -> Array2<f32> {
    let n_samples = images.shape()[0];
    let mut flattened = Array2::<f32>::zeros((n_samples, 784));
    
    // Tek thread kullanarak güvenli bir şekilde işlem yapalım
    for i in 0..n_samples {
        let img_flat = images.slice(s![i, .., ..]).to_owned().into_shape((784,)).unwrap();
        flattened.slice_mut(s![i, ..]).assign(&img_flat);
    }
    
    flattened
}

// Etiketleri one-hot encode et
fn one_hot_encode(labels: &ndarray::Array1<u8>, n_classes: usize) -> Result<Array2<f32>> {
    let n_samples = labels.len();
    let mut one_hot = Array2::<f32>::zeros((n_samples, n_classes));
    
    for (i, &label) in labels.iter().enumerate() {
        // Güvenlik kontrolü: out-of-bounds erişimi engelle
        if label as usize >= n_classes {
            return Err(anyhow::anyhow!("Etiket sayısı sınıf sayısından büyük: {} >= {}", label, n_classes));
        }
        one_hot[[i, label as usize]] = 1.0;
    }
    
    Ok(one_hot)
}

// Argmax fonksiyonu
fn get_argmax(array: &Array2<f32>) -> usize {
    let mut max_idx = 0;
    let mut max_val = f32::MIN;
    
    for (i, &val) in array.slice(s![0, ..]).iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    
    max_idx
}

// Yardımcı tiplemeler
type Array1<T> = ndarray::Array1<T>;
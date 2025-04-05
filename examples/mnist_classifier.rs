use turk_lang::{Sequential, Dense, Activation, SGD, CrossEntropy, one_hot_encode, normalize, train_test_split, Loss};
use ndarray::Array2;
use mnist::{MnistBuilder, Dataset}; // MNIST veri seti için

fn main() {
    // Daha gerçekçi model mimarisi
    let mut model = Sequential::new();
    model.add(Dense::new(784, 128, Activation::Relu));
    model.add(Dense::new(128, 64, Activation::Relu));
    model.add(Dense::new(64, 10, Activation::Softmax));

    // Daha gerçekçi öğrenme oranı
    let optimizer = SGD::new(0.01);
    let loss = CrossEntropy::new();
    
    // Optimizer'ı modele ata
    model.compile(optimizer);

    // Gerçek MNIST veri setini yükle
    println!("MNIST veri seti yükleniyor...");
    let Dataset { trn_img, trn_lbl, tst_img, tst_lbl } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(10_000) // İlk 10k örneği kullan (hızlı olması için)
        .test_set_length(1_000)     // İlk 1k test örneği
        .finalize();

    // MNIST verilerini ndarray formatına dönüştür
    let train_size = trn_img.len() / 784;
    let mut train_features = Array2::zeros((train_size, 784));
    for i in 0..train_size {
        for j in 0..784 {
            train_features[[i, j]] = trn_img[i * 784 + j] as f32 / 255.0;
        }
    }
    
    let test_size = tst_img.len() / 784;
    let mut test_features = Array2::zeros((test_size, 784));
    for i in 0..test_size {
        for j in 0..784 {
            test_features[[i, j]] = tst_img[i * 784 + j] as f32 / 255.0;
        }
    }
    
    // Etiketleri alıp one-hot encoding uygula
    let train_labels: Vec<usize> = trn_lbl.iter().map(|&x| x as usize).collect();
    let test_labels: Vec<usize> = tst_lbl.iter().map(|&x| x as usize).collect();
    
    let one_hot_train = one_hot_encode(&train_labels, 10);
    let one_hot_test = one_hot_encode(&test_labels, 10);

    println!("Eğitim başlıyor...");
    println!("Eğitim örnekleri: {}, Test örnekleri: {}", 
             train_features.nrows(), test_features.nrows());

    // Modeli eğit
    let epochs = 10;
    let batch_size = 32;
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;

        // Mini-batch eğitimi
        for i in (0..train_features.nrows()).step_by(batch_size) {
            let end = (i + batch_size).min(train_features.nrows());
            let mut batch_correct = 0;
            
            // Her örneği ayrı ayrı işle
            for j in i..end {
                let example = train_features.slice(ndarray::s![j..j+1, ..]).to_owned();
                let label = one_hot_train.slice(ndarray::s![j..j+1, ..]).to_owned();
                
                // Veriyi model için doğru şekle dönüştür (transpoz)
                let example_transposed = example.t().to_owned();
                
                // İleri yayılım
                let prediction = model.forward(&example_transposed).unwrap();
                
                // Tahminleri orijinal şekle geri dönüştür
                let prediction_original = prediction.t().to_owned();
                
                // Kayıp hesapla
                let example_loss = loss.compute(&prediction_original, &label).unwrap();
                total_loss += example_loss;

                // Doğruluk hesapla
                let pred_class = prediction_original.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap().0;
                
                let true_class = label.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap().0;
                
                if pred_class == true_class {
                    batch_correct += 1;
                }
                
                // Geri yayılım için gradyanlar
                let grad = loss.gradient(&prediction_original, &label).unwrap();
                let grad_transposed = grad.t().to_owned();
                
                // Geri yayılım
                model.backward(&grad_transposed).unwrap();
                
                // Her örnek sonrası ağırlıkları güncelle
                model.update().unwrap();
            }
            
            // Batch sonuçlarını topla
            correct += batch_correct;
            total += end - i;
            
            // Her 100 batch'te ilerleme raporu
            if (i / batch_size) % 100 == 0 && i > 0 {
                let batch_acc = batch_correct as f32 / (end - i) as f32;
                println!("  Batch {}/{}: Accuracy: {:.4}", 
                         i/batch_size, train_features.nrows()/batch_size, batch_acc);
            }
        }

        let avg_loss = total_loss / train_features.nrows() as f32;
        let train_acc = correct as f32 / total as f32;

        println!("Epoch {}/{} - Loss: {:.4} - Accuracy: {:.4}", 
                epoch + 1, epochs, avg_loss, train_acc);
                
        // Her epoch'ta test et
        test_model(&mut model, &test_features, &one_hot_test, &loss, &test_labels);
    }

    // Son test değerlendirmesi
    println!("\nFinal test değerlendirmesi:");
    test_model(&mut model, &test_features, &one_hot_test, &loss, &test_labels);
    
    // Modeli kaydet
    match model.save("mnist_model.json") {
        Ok(_) => println!("Model başarıyla kaydedildi: mnist_model.json"),
        Err(e) => println!("Model kaydedilemedi: {}", e),
    }
}

// Test fonksiyonu - confusion matrix ekledim
fn test_model(model: &mut Sequential, test_features: &Array2<f32>, 
              test_labels: &Array2<f32>, loss: &CrossEntropy, 
              raw_labels: &[usize]) {
    let mut test_correct = 0;
    let mut test_loss = 0.0;
    
    // Confusion matrix için tahminleri topla
    let mut predictions = Vec::with_capacity(test_features.nrows());
    
    // Her test örneğini ayrı ayrı değerlendir
    for i in 0..test_features.nrows() {
        let example = test_features.slice(ndarray::s![i..i+1, ..]).to_owned();
        let label = test_labels.slice(ndarray::s![i..i+1, ..]).to_owned();
        
        // Veriyi model için doğru şekle dönüştür
        let example_transposed = example.t().to_owned();
        
        // İleri yayılım
        let prediction = model.forward(&example_transposed).unwrap();
        
        // Tahminleri orijinal şekle dönüştür
        let prediction_original = prediction.t().to_owned();
        
        // Test kaybını hesapla
        let test_example_loss = loss.compute(&prediction_original, &label).unwrap();
        test_loss += test_example_loss;
        
        // Doğruluk hesapla
        let pred_class = prediction_original.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
        
        predictions.push(pred_class);
        
        let true_class = label.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
        
        if pred_class == true_class {
            test_correct += 1;
        }
    }
    
    let test_avg_loss = test_loss / test_features.nrows() as f32;
    let test_acc = test_correct as f32 / test_features.nrows() as f32;
    println!("--- TEST - Loss: {:.4} - Accuracy: {:.4} ---", test_avg_loss, test_acc);
    
    // Basit confusion matrix
    let mut confusion = vec![vec![0; 10]; 10];
    for (pred, &true_label) in predictions.iter().zip(raw_labels.iter()) {
        confusion[true_label][*pred] += 1;
    }
    
    // Özet confusion matrix yazdır
    println!("Confusion Matrix (satır: gerçek, sütun: tahmin):");
    println!("      0    1    2    3    4    5    6    7    8    9");
    println!("    ---------------------------------------------");
    for (i, row) in confusion.iter().enumerate() {
        print!("{} | ", i);
        for &val in row {
            print!("{:4} ", val);
        }
        println!();
    }
} 
//! Çeşitli yardımcı işlevler
use crate::error::{Result, TurkError};
use itertools::Itertools;
use ndarray::{s, Array2, ArrayView2, Axis};
use rand::seq::SliceRandom;
#[cfg(feature = "csv")]
use std::fs::File;
#[cfg(feature = "csv")]
use std::io::{BufReader, BufWriter};
#[cfg(feature = "csv")]
use std::path::Path;

/// Tensor matematiksel işlemleri için yardımcı fonksiyonlar
pub mod tensor_ops {
    use crate::error::{Result, TurkError};
    use ndarray::{Array2, ArrayView2, Axis, Zip};

    /// İki tensor arasında nokta çarpımı yapar (matris çarpımı)
    /// Rayon ile paralel hesaplama kullanır
    pub fn dot(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if a.ncols() != b.nrows() {
            return Err(TurkError::shape_mismatch(&format!(
                "Nokta çarpımı için uyumsuz boyutlar: {}x{} ve {}x{}",
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols()
            )));
        }

        // ndarray-rayon özelliği etkinleştirildiğinde, dot işlemi otomatik olarak paralelleştirilir
        Ok(a.dot(b))
    }

    /// L2 normalizasyon uygular
    /// Rayon ile paralel hesaplama kullanır
    pub fn l2_normalize(x: &ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        let squared = x.mapv(|v| v * v);
        let sum_squared = squared.sum_axis(Axis(axis));

        let mut normalized = x.to_owned();

        // Normalize işlemini paralelleştirme (wasm32'de sıralı)
        match axis {
            0 => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    Zip::from(normalized.columns_mut())
                        .and(sum_squared.view())
                        .par_for_each(|mut col, &norm_val| {
                            let norm = norm_val.sqrt();
                            if norm > 1e-10 {
                                col.mapv_inplace(|v| v / norm);
                            }
                        });
                }
                #[cfg(target_arch = "wasm32")]
                {
                    Zip::from(normalized.columns_mut())
                        .and(sum_squared.view())
                        .for_each(|mut col, &norm_val| {
                            let norm = norm_val.sqrt();
                            if norm > 1e-10 {
                                col.mapv_inplace(|v| v / norm);
                            }
                        });
                }
            }
            1 => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    Zip::from(normalized.rows_mut())
                        .and(sum_squared.view())
                        .par_for_each(|mut row, &norm_val| {
                            let norm = norm_val.sqrt();
                            if norm > 1e-10 {
                                row.mapv_inplace(|v| v / norm);
                            }
                        });
                }
                #[cfg(target_arch = "wasm32")]
                {
                    Zip::from(normalized.rows_mut())
                        .and(sum_squared.view())
                        .for_each(|mut row, &norm_val| {
                            let norm = norm_val.sqrt();
                            if norm > 1e-10 {
                                row.mapv_inplace(|v| v / norm);
                            }
                        });
                }
            }
            _ => return Err(TurkError::InvalidValue(format!("Geçersiz eksen: {}", axis))),
        }

        Ok(normalized)
    }

    /// Elementwise çarpım yapar
    /// Rayon ile paralel hesaplama kullanır
    pub fn elementwise_mul(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if a.shape() != b.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "Elementwise çarpım için uyumsuz boyutlar: {:?} ve {:?}",
                a.shape(),
                b.shape()
            )));
        }

        let mut result = Array2::zeros(a.dim());
        #[cfg(not(target_arch = "wasm32"))]
        {
            Zip::from(&mut result)
                .and(a)
                .and(b)
                .par_for_each(|c, &a, &b| {
                    *c = a * b;
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(&mut result).and(a).and(b).for_each(|c, &a, &b| {
                *c = a * b;
            });
        }

        Ok(result)
    }

    /// Tensor üzerinde ortalama hesaplar
    pub fn mean(x: &ArrayView2<f64>, axis: Option<usize>) -> Result<f64> {
        match axis {
            Some(ax) => {
                if ax >= x.ndim() {
                    return Err(TurkError::InvalidValue(format!(
                        "Geçersiz eksen: {}, tensor boyutu: {}",
                        ax,
                        x.ndim()
                    )));
                }

                let mean = x.mean_axis(Axis(ax)).expect("Mean hesaplaması başarısız");
                Ok(mean.sum() / mean.len() as f64)
            }
            None => Ok(x.mean().unwrap_or(0.0)),
        }
    }

    /// İki tensor arasındaki ortalama karesel hatayı hesaplar (MSE)
    /// Rayon ile paralel hesaplama kullanır
    pub fn mse(y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<f64> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "MSE hesaplaması için uyumsuz boyutlar: {:?} ve {:?}",
                y_true.shape(),
                y_pred.shape()
            )));
        }

        let n_samples = y_true.nrows();
        let n_outputs = y_true.ncols();

        #[cfg(not(target_arch = "wasm32"))]
        let squared_error_sum: f64 = Zip::from(y_true)
            .and(y_pred)
            .par_map_collect(|&y_t, &y_p| {
                let diff = y_t - y_p;
                diff * diff
            })
            .sum();
        #[cfg(target_arch = "wasm32")]
        let squared_error_sum: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| {
                let diff = y_t - y_p;
                diff * diff
            })
            .sum();

        Ok(squared_error_sum / (n_samples * n_outputs) as f64)
    }

    /// MSE kaybının gradyanını hesaplar
    /// Rayon ile paralel hesaplama kullanır
    pub fn mse_gradient(y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if y_true.shape() != y_pred.shape() {
            return Err(TurkError::shape_mismatch(&format!(
                "MSE gradyan hesaplaması için uyumsuz boyutlar: {:?} ve {:?}",
                y_true.shape(),
                y_pred.shape()
            )));
        }

        let n_samples = y_true.nrows();
        let n_outputs = y_true.ncols();
        let scale = 2.0 / (n_samples * n_outputs) as f64;

        let mut gradient = Array2::zeros(y_true.dim());
        #[cfg(not(target_arch = "wasm32"))]
        {
            Zip::from(&mut gradient)
                .and(y_pred)
                .and(y_true)
                .par_for_each(|g, &y_p, &y_t| {
                    *g = (y_p - y_t) * scale;
                });
        }
        #[cfg(target_arch = "wasm32")]
        {
            Zip::from(&mut gradient)
                .and(y_pred)
                .and(y_true)
                .for_each(|g, &y_p, &y_t| {
                    *g = (y_p - y_t) * scale;
                });
        }

        Ok(gradient)
    }
}

/// Tensor şekillerinin uyumluluğunu kontrol eder
///
/// İki tensor'un şekil uyumluluğunu kontrol eder ve uyumsuzluk durumunda hata döndürür.
/// Bu fonksiyon, veri doğrulama işlemlerinde kullanılmak üzere yazılmıştır.
///
/// # Parametreler
/// - `x`: Birinci tensor
/// - `y`: İkinci tensor
/// - `check_rows`: Satır sayılarının eşit olması gerekip gerekmediği
/// - `check_cols`: Sütun sayılarının eşit olması gerekip gerekmediği
/// - `error_context`: Hata mesajına eklenecek bağlam bilgisi
///
/// # Dönüş
/// - `Result<()>`: Başarılı ise Ok(()), hata durumunda ilgili hata
pub fn validate_tensor_shapes(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    check_rows: bool,
    check_cols: bool,
    error_context: &str,
) -> Result<()> {
    if check_rows && x.nrows() != y.nrows() {
        return Err(TurkError::shape_mismatch(&format!(
            "{}: satır sayıları uyuşmuyor: {} vs {}",
            error_context,
            x.nrows(),
            y.nrows()
        )));
    }

    if check_cols && x.ncols() != y.ncols() {
        return Err(TurkError::shape_mismatch(&format!(
            "{}: sütun sayıları uyuşmuyor: {} vs {}",
            error_context,
            x.ncols(),
            y.ncols()
        )));
    }

    Ok(())
}

/// Veriyi karıştır
/// Rayon ile paralel hesaplama kullanır
pub fn shuffle_data(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
) -> Result<(Array2<f64>, Array2<f64>)> {
    validate_tensor_shapes(x, y, true, false, "Karıştırılacak veriler")?;

    let n_samples = x.nrows();

    // Indeksleri oluştur ve karıştır
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    // Karıştırılmış dizileri oluştur
    let mut x_shuffled = Array2::zeros((n_samples, x.ncols()));
    let mut y_shuffled = Array2::zeros((n_samples, y.ncols()));

    // Paralelleştirme yerine basit döngü kullanma
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..x.ncols() {
            x_shuffled[[i, j]] = x[[idx, j]];
        }

        for j in 0..y.ncols() {
            y_shuffled[[i, j]] = y[[idx, j]];
        }
    }

    Ok((x_shuffled, y_shuffled))
}

/// Veri kümesini train/test olarak böl
/// Rayon ile paralel hesaplama kullanır
pub fn train_test_split(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    test_size: f64,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>)> {
    validate_tensor_shapes(x, y, true, false, "Train/test ayrımı")?;

    let (x_shuffled, y_shuffled) = shuffle_data(x, y)?;

    let n_samples = x.nrows();
    let n_test = (n_samples as f64 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    // Slice işlemi zaten hızlıdır, ama kopyalama daha paralel olabilir
    let mut x_train = Array2::zeros((n_train, x.ncols()));
    let mut x_test = Array2::zeros((n_test, x.ncols()));
    let mut y_train = Array2::zeros((n_train, y.ncols()));
    let mut y_test = Array2::zeros((n_test, y.ncols()));

    // Paralelleştirme yerine normal döngüler kullan
    // Training verilerini kopyala
    for i in 0..n_train {
        for j in 0..x.ncols() {
            x_train[[i, j]] = x_shuffled[[i, j]];
        }
    }

    for i in 0..n_train {
        for j in 0..y.ncols() {
            y_train[[i, j]] = y_shuffled[[i, j]];
        }
    }

    // Test verilerini kopyala
    for i in 0..n_test {
        for j in 0..x.ncols() {
            x_test[[i, j]] = x_shuffled[[i + n_train, j]];
        }
    }

    for i in 0..n_test {
        for j in 0..y.ncols() {
            y_test[[i, j]] = y_shuffled[[i + n_train, j]];
        }
    }

    Ok((x_train, x_test, y_train, y_test))
}

/// Bootstrap veri örneklemesi yap
/// Rayon ile paralel hesaplama kullanır
pub fn bootstrap_sample(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    test_n: usize,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>)> {
    validate_tensor_shapes(x, y, true, false, "Bootstrap örnekleme")?;

    let n_samples = x.nrows();

    if test_n >= n_samples {
        return Err(TurkError::InvalidValue(format!(
            "Test örnek sayısı toplam örnek sayısından az olmalıdır: {} >= {}",
            test_n, n_samples
        )));
    }

    // Karıştırılmış indeksleri oluştur
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    // Test ve train indekslerini ayır
    let train_indices = &indices[0..n_samples - test_n];
    let test_indices = &indices[n_samples - test_n..];

    // Train ve test veri kümelerini oluştur
    let mut x_train = Array2::zeros((train_indices.len(), x.ncols()));
    let mut y_train = Array2::zeros((train_indices.len(), y.ncols()));
    let mut x_test = Array2::zeros((test_indices.len(), x.ncols()));
    let mut y_test = Array2::zeros((test_indices.len(), y.ncols()));

    // Paralel yerine sıralı işlem - eğitim verileri
    for (i, &idx) in train_indices.iter().enumerate() {
        for j in 0..x.ncols() {
            x_train[[i, j]] = x[[idx, j]];
        }

        for j in 0..y.ncols() {
            y_train[[i, j]] = y[[idx, j]];
        }
    }

    // Paralel yerine sıralı işlem - test verileri
    for (i, &idx) in test_indices.iter().enumerate() {
        for j in 0..x.ncols() {
            x_test[[i, j]] = x[[idx, j]];
        }

        for j in 0..y.ncols() {
            y_test[[i, j]] = y[[idx, j]];
        }
    }

    Ok((x_train, y_train, x_test, y_test))
}

/// Batch'ler oluştur
/// Paralel olarak batch'leri hazırlayabilir
pub fn batch_iterator<'a>(
    x: &'a ArrayView2<'a, f64>,
    y: &'a ArrayView2<'a, f64>,
    batch_size: usize,
) -> Result<Vec<(ArrayView2<'a, f64>, ArrayView2<'a, f64>)>> {
    validate_tensor_shapes(x, y, true, false, "Batch oluşturma")?;

    let n_samples = x.nrows();

    if batch_size == 0 {
        return Err(TurkError::InvalidValue(
            "Batch boyutu sıfır olamaz".to_string(),
        ));
    }

    let n_batches = n_samples.div_ceil(batch_size);
    let mut batches = Vec::with_capacity(n_batches);

    // Batch'leri oluştur
    for i in 0..n_batches {
        let start_idx = i * batch_size;
        let end_idx = std::cmp::min(start_idx + batch_size, n_samples);

        if start_idx >= n_samples {
            break;
        }

        let x_batch = x.slice(s![start_idx..end_idx, ..]);
        let y_batch = y.slice(s![start_idx..end_idx, ..]);

        batches.push((x_batch, y_batch));
    }

    Ok(batches)
}

/// One-hot encoding
pub fn one_hot_encode(labels: &[usize], n_classes: usize) -> Result<Array2<f64>> {
    let n_samples = labels.len();
    let mut encoded = Array2::zeros((n_samples, n_classes));

    for (i, &label) in labels.iter().enumerate() {
        if label >= n_classes {
            return Err(TurkError::InvalidValue(format!(
                "Sınıf etiketi {} geçersiz. Maksimum sınıf sayısı: {}",
                label, n_classes
            )));
        }
        encoded[[i, label]] = 1.0;
    }

    Ok(encoded)
}

/// Argmax: her satırdaki en büyük değerin indeksini döndürür
pub fn argmax(array: &ArrayView2<f64>, axis: usize) -> Vec<usize> {
    array
        .map_axis(Axis(axis), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .to_vec()
}

/// CSV dosyasından veri yükle
#[cfg(feature = "csv")]
pub fn load_csv<P: AsRef<Path>>(path: P, has_header: bool, delimiter: char) -> Result<Array2<f64>> {
    let file = File::open(path.as_ref())
        .map_err(|e| TurkError::IOError(format!("CSV dosyası açılamadı: {}", e)))?;

    let reader = BufReader::new(file);
    let mut csv_reader = csv::ReaderBuilder::new()
        .delimiter(delimiter as u8)
        .has_headers(has_header)
        .from_reader(reader);

    let mut rows = Vec::new();
    let mut n_cols = 0;

    for (row_idx, result) in csv_reader.records().enumerate() {
        let record = result.map_err(|e| {
            TurkError::IOError(format!("CSV satırı {} okunamadı: {}", row_idx + 1, e))
        })?;

        if row_idx == 0 {
            n_cols = record.len();
        } else if record.len() != n_cols {
            return Err(TurkError::DataError(format!(
                "Tutarsız kolon sayısı: satır {} için {} kolon, beklenen {}",
                row_idx + 1,
                record.len(),
                n_cols
            )));
        }

        let row: Result<Vec<f64>> = record
            .iter()
            .map(|s| {
                s.parse::<f64>().map_err(|e| {
                    TurkError::DataError(format!("Sayısal değer dönüştürülemedi '{}': {}", s, e))
                })
            })
            .collect();

        rows.push(row?);
    }

    if rows.is_empty() {
        return Err(TurkError::DataError("CSV dosyası boş".into()));
    }

    let mut data = Array2::zeros((rows.len(), n_cols));
    for (i, row) in rows.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            data[[i, j]] = *val;
        }
    }

    Ok(data)
}

/// CSV dosyasına veri yaz
#[cfg(feature = "csv")]
pub fn save_csv<P: AsRef<Path>>(
    path: P,
    data: &Array2<f64>,
    headers: Option<Vec<String>>,
    delimiter: char,
) -> Result<()> {
    let file = File::create(path.as_ref())
        .map_err(|e| TurkError::IOError(format!("CSV dosyası oluşturulamadı: {}", e)))?;

    let writer = BufWriter::new(file);
    let mut csv_writer = csv::WriterBuilder::new()
        .delimiter(delimiter as u8)
        .from_writer(writer);

    // Başlıklar varsa yaz
    if let Some(headers) = headers {
        if headers.len() != data.ncols() {
            return Err(TurkError::DimensionMismatch(format!(
                "Başlık sayısı ({}) ile veri sütun sayısı ({}) eşleşmiyor",
                headers.len(),
                data.ncols()
            )));
        }

        csv_writer
            .write_record(&headers)
            .map_err(|e| TurkError::IOError(format!("Başlıklar yazılamadı: {}", e)))?;
    }

    // Verileri yaz
    for i in 0..data.nrows() {
        let row: Vec<String> = data.row(i).iter().map(|&x| x.to_string()).collect();

        csv_writer
            .write_record(&row)
            .map_err(|e| TurkError::IOError(format!("Satır {} yazılamadı: {}", i, e)))?;
    }

    csv_writer
        .flush()
        .map_err(|e| TurkError::IOError(format!("CSV dosyası kapatılamadı: {}", e)))?;

    Ok(())
}

/// Confusion matrix hesapla
pub fn confusion_matrix(
    y_true: &[usize],
    y_pred: &[usize],
    n_classes: usize,
) -> Result<Array2<usize>> {
    if y_true.len() != y_pred.len() {
        return Err(TurkError::DimensionMismatch(format!(
            "Gerçek ve tahmin edilen etiketlerin boyutu uyuşmuyor: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    let mut conf_matrix = Array2::zeros((n_classes, n_classes));

    for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
        if true_label >= n_classes || pred_label >= n_classes {
            return Err(TurkError::InvalidValue(format!(
                "Sınıf etiketi geçersiz. Gerçek: {}, Tahmin: {}, Maksimum: {}",
                true_label,
                pred_label,
                n_classes - 1
            )));
        }
        conf_matrix[[true_label, pred_label]] += 1;
    }

    Ok(conf_matrix)
}

/// Sınıflandırma metrikleri hesapla
pub fn classification_report(
    y_true: &[usize],
    y_pred: &[usize],
    labels: Option<&[usize]>,
    target_names: Option<&[String]>,
) -> Result<String> {
    if y_true.len() != y_pred.len() {
        return Err(TurkError::DimensionMismatch(format!(
            "Gerçek ve tahmin edilen etiketlerin boyutu uyuşmuyor: {} vs {}",
            y_true.len(),
            y_pred.len()
        )));
    }

    // Kullanılacak etiketleri belirle
    let actual_labels = if let Some(lbls) = labels {
        lbls.to_vec()
    } else {
        // Veri içinde geçen benzersiz etiketleri bul
        let mut unique_labels: Vec<usize> = y_true
            .iter()
            .chain(y_pred.iter())
            .copied()
            .unique()
            .collect();
        unique_labels.sort();
        unique_labels
    };

    let n_classes = actual_labels.len();

    // Target names kontrolü
    if let Some(names) = target_names {
        if names.len() != n_classes {
            return Err(TurkError::DimensionMismatch(format!(
                "Hedef isimlerinin sayısı ({}) ile sınıf sayısı ({}) uyuşmuyor",
                names.len(),
                n_classes
            )));
        }
    }

    // Confusion matrix hesapla
    let conf_matrix = confusion_matrix(
        y_true,
        y_pred,
        *actual_labels.iter().max().unwrap_or(&0) + 1,
    )?;

    // Metrik hesaplamaları
    let mut precision_vals = Vec::with_capacity(n_classes);
    let mut recall_vals = Vec::with_capacity(n_classes);
    let mut f1_vals = Vec::with_capacity(n_classes);
    let mut support_vals = Vec::with_capacity(n_classes);

    let mut avg_precision = 0.0;
    let mut avg_recall = 0.0;
    let mut avg_f1 = 0.0;
    let mut total_support = 0;

    // Header
    let mut result = String::new();
    result.push_str(&format!(
        "{:12} {:9} {:9} {:9} {:9}\n",
        "", "precision", "recall", "f1-score", "support"
    ));

    for (i, &label) in actual_labels.iter().enumerate() {
        let row_idx = label;
        let col_idx = label;

        let tp = conf_matrix[[row_idx, col_idx]];
        let fp = conf_matrix.column(col_idx).sum() - tp;
        let r#fn = conf_matrix.row(row_idx).sum() - tp;
        // tn değişkeni hiç kullanılmıyor, hesaplamıyorum

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + r#fn > 0 {
            tp as f64 / (tp + r#fn) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let support = tp + r#fn;

        precision_vals.push(precision);
        recall_vals.push(recall);
        f1_vals.push(f1);
        support_vals.push(support);

        avg_precision += precision * support as f64;
        avg_recall += recall * support as f64;
        avg_f1 += f1 * support as f64;
        total_support += support;

        // Sınıf ismini belirle
        let class_name = if let Some(names) = target_names {
            names[i].clone()
        } else {
            format!("class {}", label)
        };

        result.push_str(&format!(
            "{:12} {:9.2} {:9.2} {:9.2} {:9}\n",
            class_name, precision, recall, f1, support
        ));
    }

    // Ortalama değerler
    avg_precision /= total_support as f64;
    avg_recall /= total_support as f64;
    avg_f1 /= total_support as f64;

    result.push_str(&format!(
        "\n{:12} {:9.2} {:9.2} {:9.2} {:9.0}\n",
        "avg / total", avg_precision, avg_recall, avg_f1, total_support
    ));

    Ok(result)
}

/// Belirtilen indekslerdeki satırları yeni bir tensor'da birleştirir
///
/// # Parametreler
/// * `array` - Kaynak tensor
/// * `indices` - Seçilecek satır indeksleri
///
/// # Dönüş
/// * `Result<Array2<f64>>` - Seçilen satırlardan oluşan yeni tensor
pub fn stack_selected_rows(array: &ArrayView2<'_, f64>, indices: &[usize]) -> Result<Array2<f64>> {
    if indices.is_empty() {
        return Err(TurkError::InvalidValue("Boş indeks listesi".to_string()));
    }

    // Sonuç tensorunun boyutları
    let n_rows = indices.len();
    let n_cols = array.ncols();

    // Sonuç tensoru
    let mut result = Array2::zeros((n_rows, n_cols));

    // Seçilen satırları kopyala
    for (dest_idx, &src_idx) in indices.iter().enumerate() {
        if src_idx >= array.nrows() {
            return Err(TurkError::InvalidValue(format!(
                "Geçersiz indeks: {} (toplam satır sayısı: {})",
                src_idx,
                array.nrows()
            )));
        }

        for col in 0..n_cols {
            result[[dest_idx, col]] = array[[src_idx, col]];
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_data() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();

        let y = Array2::from_shape_vec((5, 1), vec![0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();

        let (x_shuffled, y_shuffled) = shuffle_data(&x.view(), &y.view()).unwrap();

        // Boyutlar korunmalı
        assert_eq!(x_shuffled.shape(), x.shape());
        assert_eq!(y_shuffled.shape(), y.shape());

        // Toplam değerler korunmalı
        assert!((x_shuffled.sum() - x.sum()).abs() < 1e-6);
        assert!((y_shuffled.sum() - y.sum()).abs() < 1e-6);
    }

    #[test]
    fn test_train_test_split() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
        )
        .unwrap();

        let y = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )
        .unwrap();

        let (x_train, x_test, y_train, y_test) =
            train_test_split(&x.view(), &y.view(), 0.3).unwrap();

        // Boyut kontrolleri
        assert_eq!(x_train.nrows(), 7);
        assert_eq!(x_test.nrows(), 3);
        assert_eq!(y_train.nrows(), 7);
        assert_eq!(y_test.nrows(), 3);

        // Kolon sayıları korunmalı
        assert_eq!(x_train.ncols(), 2);
        assert_eq!(x_test.ncols(), 2);
        assert_eq!(y_train.ncols(), 1);
        assert_eq!(y_test.ncols(), 1);

        // Toplam değerler korunmalı
        assert!((x_train.sum() + x_test.sum() - x.sum()).abs() < 1e-6);
        assert!((y_train.sum() + y_test.sum() - y.sum()).abs() < 1e-6);
    }

    #[test]
    fn test_one_hot_encode() {
        let labels = vec![0, 2, 1, 0, 3];
        let encoded = one_hot_encode(&labels, 4).unwrap();

        assert_eq!(encoded.shape(), &[5, 4]);

        // Kontroller
        assert_eq!(encoded[[0, 0]], 1.0);
        assert_eq!(encoded[[1, 2]], 1.0);
        assert_eq!(encoded[[2, 1]], 1.0);
        assert_eq!(encoded[[3, 0]], 1.0);
        assert_eq!(encoded[[4, 3]], 1.0);

        // Her satırda toplam 1 olmalı
        for i in 0..5 {
            assert_eq!(encoded.row(i).sum(), 1.0);
        }
    }

    #[test]
    fn test_argmax() {
        let array = Array2::from_shape_vec(
            (3, 4),
            vec![1.0, 5.0, 3.0, 2.0, 7.0, 2.0, 9.0, 1.0, 4.0, 6.0, 3.0, 8.0],
        )
        .unwrap();

        let result = argmax(&array.view(), 1);

        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_confusion_matrix() {
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 2];

        let conf_matrix = confusion_matrix(&y_true, &y_pred, 3).unwrap();

        assert_eq!(conf_matrix.shape(), &[3, 3]);

        // Doğrulama
        assert_eq!(conf_matrix[[0, 0]], 2); // TP for class 0
        assert_eq!(conf_matrix[[1, 2]], 1); // FN for class 1, predicted as 2
        assert_eq!(conf_matrix[[2, 1]], 1); // FN for class 2, predicted as 1
        assert_eq!(conf_matrix[[1, 0]], 1); // FN for class 1, predicted as 0
    }
}

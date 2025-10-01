use ndarray::Array2;

pub fn one_hot_encode(labels: &[usize], num_classes: usize) -> Array2<f32> {
    let n_samples = labels.len();
    let mut encoded = Array2::zeros((n_samples, num_classes));
    
    for (i, &label) in labels.iter().enumerate() {
        encoded[[i, label]] = 1.0;
    }
    
    encoded
}

pub fn normalize(x: &Array2<f32>) -> Array2<f32> {
    use ndarray::Axis;

    let mean = x.mean_axis(Axis(0)).unwrap();
    let std = x.std_axis(Axis(0), 0.0);
    let safe_std = std.mapv(|value| if value.abs() < f32::EPSILON { 1.0 } else { value });

    (x - &mean) / &safe_std
}

pub fn train_test_split(x: &Array2<f32>, y: &Array2<f32>, test_size: f32) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let n_samples = x.nrows();
    let n_test = (n_samples as f32 * test_size) as usize;
    
    let mut indices: Vec<usize> = (0..n_samples).collect();
    rand::seq::SliceRandom::shuffle(&mut indices[..], &mut rand::thread_rng());
    
    let test_indices = &indices[..n_test];
    let train_indices = &indices[n_test..];
    
    let x_train = x.select(ndarray::Axis(0), train_indices).to_owned();
    let x_test = x.select(ndarray::Axis(0), test_indices).to_owned();
    let y_train = y.select(ndarray::Axis(0), train_indices).to_owned();
    let y_test = y.select(ndarray::Axis(0), test_indices).to_owned();
    
    (x_train, x_test, y_train, y_test)
}

pub fn accuracy(predictions: &Array2<f32>, targets: &Array2<f32>) -> f32 {
    let mut correct = 0;
    let n_samples = predictions.nrows();
    
    for i in 0..n_samples {
        let pred_row = predictions.row(i);
        let target_row = targets.row(i);
        
        let pred_class = pred_row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
            
        let target_class = target_row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
            
        if pred_class == target_class {
            correct += 1;
        }
    }
    
    correct as f32 / n_samples as f32
}

pub fn confusion_matrix(predictions: &Array2<f32>, targets: &Array2<f32>) -> Array2<f32> {
    let n_classes = predictions.ncols();
    let mut matrix = Array2::zeros((n_classes, n_classes));
    
    for i in 0..predictions.nrows() {
        let pred_row = predictions.row(i);
        let target_row = targets.row(i);
        
        let pred_class = pred_row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
            
        let target_class = target_row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
            
        matrix[[target_class, pred_class]] += 1.0;
    }
    
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_one_hot_encode() {
        let labels = vec![0, 1, 2];
        let encoded = one_hot_encode(&labels, 3);
        
        assert_eq!(encoded.shape(), [3, 3]);
        assert_eq!(encoded[[0, 0]], 1.0);
        assert_eq!(encoded[[1, 1]], 1.0);
        assert_eq!(encoded[[2, 2]], 1.0);
        assert_eq!(encoded[[0, 1]], 0.0);
        assert_eq!(encoded[[1, 0]], 0.0);
    }

    #[test]
    fn test_normalize() {
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 3.0],
            [7.0, 8.0, 3.0],
            [7.0, 5.0, 3.0], // üçüncü sütun için sıfır varyans
        ];

        let normalized = normalize(&data);

        let col_means = normalized.mean_axis(ndarray::Axis(0)).unwrap();
        let col_stds = normalized.std_axis(ndarray::Axis(0), 0.0);

        for mean in col_means.iter() {
            assert!(mean.abs() < 1e-6);
        }

        for (std, original_std) in col_stds.iter().zip(data.std_axis(ndarray::Axis(0), 0.0).iter()) {
            if original_std.abs() < f32::EPSILON {
                assert!((*std - 0.0).abs() < 1e-6);
            } else {
                assert!((*std - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_train_test_split() {
        let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f32).collect()).unwrap();
        let labels = Array2::from_shape_vec((10, 1), (0..10).map(|x| x as f32).collect()).unwrap();
        
        let (x_train, x_test, y_train, y_test) = train_test_split(&data, &labels, 0.2);
        
        assert_eq!(x_train.nrows(), 8);
        assert_eq!(x_test.nrows(), 2);
        assert_eq!(y_train.nrows(), 8);
        assert_eq!(y_test.nrows(), 2);
    }

    #[test]
    fn test_accuracy() {
        let predictions = array![[1.0, 0.0],
                               [0.0, 1.0],
                               [1.0, 0.0]];
        let targets = array![[1.0, 0.0],
                           [0.0, 1.0],
                           [0.0, 1.0]];
        
        let acc = accuracy(&predictions, &targets);
        assert_eq!(acc, 2.0 / 3.0);
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = array![[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]];
        let targets = array![[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]];
        
        let matrix = confusion_matrix(&predictions, &targets);
        
        assert_eq!(matrix.shape(), [3, 3]);
        assert_eq!(matrix[[0, 0]], 1.0);
        assert_eq!(matrix[[1, 1]], 1.0);
        assert_eq!(matrix[[2, 2]], 1.0);
        assert_eq!(matrix[[0, 1]], 0.0);
        assert_eq!(matrix[[1, 0]], 0.0);
    }

    #[test]
    fn test_crate_import() {
        use crate::Zeros;
        let _z = Zeros;
        assert!(true);
    }
} 
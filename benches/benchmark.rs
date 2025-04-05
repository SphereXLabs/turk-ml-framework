use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, Axis};
use turk_lang::{
    Dense, Dropout, BatchNorm, Activation, SGD, Adam, AdamW, RMSProp, 
    Layer, Optimizer, Loss, MSE, CrossEntropy, 
    initializers::{Initializer, Xavier, HeNormal, RandomNormal}
};

// Dense katmanı benchmark'ı
fn bench_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dense Layer");
    
    let batch_sizes = [1, 8, 32, 128, 512];
    
    for batch_size in batch_sizes {
        // 784->128->10 boyutlarında bir ağ (MNIST için tipik)
        let input_size = 784;
        let hidden_size = 128;
        
        // Giriş verisi oluştur
        let input = Array2::random((batch_size, input_size), ndarray_rand::rand_distr::Uniform::new(0., 1.));
        
        // ReLU aktivasyonlu dense katmanı
        let mut dense_layer = Dense::new(input_size, hidden_size, Activation::Relu);
        
        group.bench_with_input(
            BenchmarkId::new("Forward", batch_size), 
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(dense_layer.forward(black_box(&input))).unwrap()
                })
            }
        );
        
        // Gradient benchmark'ı için hazırlık yapılır
        let output = dense_layer.forward(&input).unwrap();
        let grad = Array2::random((batch_size, hidden_size), ndarray_rand::rand_distr::Uniform::new(0., 1.));
        
        group.bench_with_input(
            BenchmarkId::new("Backward", batch_size), 
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(dense_layer.backward(black_box(&grad))).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

// Dropout katmanı benchmark'ı
fn bench_dropout(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dropout Layer");
    
    let batch_sizes = [32, 128, 512];
    let feature_sizes = [128, 512, 1024];
    
    for batch_size in batch_sizes {
        for feature_size in feature_sizes {
            // Giriş verisi oluştur
            let input = Array2::random((batch_size, feature_size), ndarray_rand::rand_distr::Uniform::new(0., 1.));
            
            // Dropout katmanı (0.5 oranında)
            let mut dropout_layer = Dropout::new(0.5);
            
            // Eğitim modu
            dropout_layer.set_training(true);
            
            group.bench_with_input(
                BenchmarkId::new(format!("Train-Forward-{}x{}", batch_size, feature_size), batch_size),
                &batch_size,
                |b, _| {
                    b.iter(|| {
                        black_box(dropout_layer.forward(black_box(&input))).unwrap()
                    })
                }
            );
            
            // Çıkarım modu
            dropout_layer.set_training(false);
            
            group.bench_with_input(
                BenchmarkId::new(format!("Inference-Forward-{}x{}", batch_size, feature_size), batch_size),
                &batch_size,
                |b, _| {
                    b.iter(|| {
                        black_box(dropout_layer.forward(black_box(&input))).unwrap()
                    })
                }
            );
        }
    }
    
    group.finish();
}

// Batch Normalization katmanı benchmark'ı
fn bench_batch_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("BatchNorm Layer");
    
    let batch_sizes = [32, 128, 512];
    let feature_sizes = [64, 256, 1024];
    
    for batch_size in batch_sizes {
        for feature_size in feature_sizes {
            // Giriş verisi oluştur
            let input = Array2::random((batch_size, feature_size), ndarray_rand::rand_distr::Uniform::new(-1., 1.));
            
            // BatchNorm katmanı
            let mut batchnorm_layer = BatchNorm::new(feature_size);
            
            // Eğitim modu
            batchnorm_layer.set_training(true);
            
            group.bench_with_input(
                BenchmarkId::new(format!("Train-Forward-{}x{}", batch_size, feature_size), batch_size),
                &batch_size,
                |b, _| {
                    b.iter(|| {
                        black_box(batchnorm_layer.forward(black_box(&input))).unwrap()
                    })
                }
            );
            
            // Önce koşturarak hareketli ortalama biriktir
            for _ in 0..10 {
                batchnorm_layer.forward(&input).unwrap();
            }
            
            // Çıkarım modu
            batchnorm_layer.set_training(false);
            
            group.bench_with_input(
                BenchmarkId::new(format!("Inference-Forward-{}x{}", batch_size, feature_size), batch_size),
                &batch_size,
                |b, _| {
                    b.iter(|| {
                        black_box(batchnorm_layer.forward(black_box(&input))).unwrap()
                    })
                }
            );
        }
    }
    
    group.finish();
}

// Optimizer benchmark'ları
fn bench_optimizers(c: &mut Criterion) {
    let mut group = c.benchmark_group("Optimizers");
    
    // Parametre matrisi ve gradient oluştur
    let param_shapes = [(100, 100), (1000, 100), (1000, 1000)];
    
    for (rows, cols) in param_shapes {
        let mut params = Array2::random((rows, cols), ndarray_rand::rand_distr::Normal::new(0.0, 0.1).unwrap());
        let grads = Array2::random((rows, cols), ndarray_rand::rand_distr::Normal::new(0.0, 0.01).unwrap());
        
        // SGD
        let sgd = SGD::new(0.01);
        group.bench_with_input(
            BenchmarkId::new(format!("SGD-{}x{}", rows, cols), rows),
            &rows,
            |b, _| {
                b.iter(|| {
                    let mut p = params.clone();
                    black_box(sgd.update(&mut p, black_box(&grads))).unwrap()
                })
            }
        );
        
        // SGD with momentum
        let sgd_momentum = SGD::with_momentum(0.01, 0.9);
        group.bench_with_input(
            BenchmarkId::new(format!("SGD-Momentum-{}x{}", rows, cols), rows),
            &rows,
            |b, _| {
                b.iter(|| {
                    let mut p = params.clone();
                    let mut opt = sgd_momentum.clone();
                    black_box(opt.update_with_state(&mut p, black_box(&grads))).unwrap()
                })
            }
        );
        
        // Adam
        let adam = Adam::new(0.001);
        group.bench_with_input(
            BenchmarkId::new(format!("Adam-{}x{}", rows, cols), rows),
            &rows,
            |b, _| {
                b.iter(|| {
                    let mut p = params.clone();
                    let mut opt = adam.clone();
                    black_box(opt.update_with_state(&mut p, black_box(&grads))).unwrap()
                })
            }
        );
        
        // AdamW
        let adamw = AdamW::new(0.001, 0.01);
        group.bench_with_input(
            BenchmarkId::new(format!("AdamW-{}x{}", rows, cols), rows),
            &rows,
            |b, _| {
                b.iter(|| {
                    let mut p = params.clone();
                    let mut opt = adamw.clone();
                    black_box(opt.update_with_state(&mut p, black_box(&grads))).unwrap()
                })
            }
        );
        
        // RMSProp
        let rmsprop = RMSProp::new(0.001);
        group.bench_with_input(
            BenchmarkId::new(format!("RMSProp-{}x{}", rows, cols), rows),
            &rows,
            |b, _| {
                b.iter(|| {
                    let mut p = params.clone();
                    let mut opt = rmsprop.clone();
                    black_box(opt.update_with_state(&mut p, black_box(&grads))).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

// Kayıp fonksiyonları benchmark'ı
fn bench_loss_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Loss Functions");
    
    let batch_sizes = [32, 128, 512];
    let output_sizes = [10, 100, 1000];
    
    for batch_size in batch_sizes {
        for output_size in output_sizes {
            // Tahminler ve gerçek değerler
            let predictions = Array2::random((batch_size, output_size), ndarray_rand::rand_distr::Uniform::new(0., 1.));
            let targets = Array2::random((batch_size, output_size), ndarray_rand::rand_distr::Uniform::new(0., 1.));
            
            // MSE (Ortalama Kare Hata)
            let mse = MSE::new();
            group.bench_with_input(
                BenchmarkId::new(format!("MSE-Forward-{}x{}", batch_size, output_size), batch_size),
                &batch_size,
                |b, _| {
                    b.iter(|| {
                        black_box(mse.forward(black_box(&predictions), black_box(&targets))).unwrap()
                    })
                }
            );
            
            group.bench_with_input(
                BenchmarkId::new(format!("MSE-Backward-{}x{}", batch_size, output_size), batch_size),
                &batch_size,
                |b, _| {
                    b.iter(|| {
                        black_box(mse.backward(black_box(&predictions), black_box(&targets))).unwrap()
                    })
                }
            );
            
            // Cross Entropy
            let cross_entropy = CrossEntropy::new();
            
            // Olasılık dağılımlarına dönüştür
            let probs = predictions.map_axis(Axis(1), |x| {
                let sum = x.sum();
                x / sum
            });
            
            group.bench_with_input(
                BenchmarkId::new(format!("CrossEntropy-Forward-{}x{}", batch_size, output_size), batch_size),
                &batch_size,
                |b, _| {
                    b.iter(|| {
                        black_box(cross_entropy.forward(black_box(&probs), black_box(&targets))).unwrap()
                    })
                }
            );
            
            group.bench_with_input(
                BenchmarkId::new(format!("CrossEntropy-Backward-{}x{}", batch_size, output_size), batch_size),
                &batch_size,
                |b, _| {
                    b.iter(|| {
                        black_box(cross_entropy.backward(black_box(&probs), black_box(&targets))).unwrap()
                    })
                }
            );
        }
    }
    
    group.finish();
}

// Ağırlık başlatıcıları benchmark'ı
fn bench_initializers(c: &mut Criterion) {
    let mut group = c.benchmark_group("Initializers");
    
    let shapes = [(100, 100), (1000, 100), (1000, 1000)];
    
    for (rows, cols) in shapes {
        // Xavier (Glorot) başlatıcı
        let xavier = Xavier::new();
        group.bench_with_input(
            BenchmarkId::new(format!("Xavier-{}x{}", rows, cols), rows),
            &rows,
            |b, _| {
                b.iter(|| {
                    black_box(xavier.initialize(black_box(rows), black_box(cols))).unwrap()
                })
            }
        );
        
        // He Normal başlatıcı
        let he_normal = HeNormal::new();
        group.bench_with_input(
            BenchmarkId::new(format!("HeNormal-{}x{}", rows, cols), rows),
            &rows,
            |b, _| {
                b.iter(|| {
                    black_box(he_normal.initialize(black_box(rows), black_box(cols))).unwrap()
                })
            }
        );
        
        // Random Normal başlatıcı
        let random_normal = RandomNormal::new(0.0, 0.1);
        group.bench_with_input(
            BenchmarkId::new(format!("RandomNormal-{}x{}", rows, cols), rows),
            &rows,
            |b, _| {
                b.iter(|| {
                    black_box(random_normal.initialize(black_box(rows), black_box(cols))).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_dense,
    bench_dropout,
    bench_batch_norm,
    bench_optimizers,
    bench_loss_functions,
    bench_initializers
);
criterion_main!(benches); 
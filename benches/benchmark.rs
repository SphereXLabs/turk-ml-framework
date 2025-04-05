use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::{Array2, arr2};
use turk_lang::{
    Layer, Dense, Activation, Dropout, BatchNorm,
    Optimizer, SGD, Adam, AdamW, RMSProp,
    loss::{Loss, MSE}, 
    Sequential, Model
};

fn bench_dense_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_forward");
    group.sample_size(50);
    
    let input_sizes = [10, 100, 500, 1000];
    let output_size = 10;
    
    for &input_size in &input_sizes {
        group.bench_with_input(
            BenchmarkId::new("relu", input_size),
            &input_size,
            |b, &size| {
                let mut layer = Dense::new(size, output_size, Activation::Relu);
                let input = Array2::ones((1, size));
                
                b.iter(|| {
                    black_box(layer.forward(black_box(&input)).unwrap());
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("sigmoid", input_size),
            &input_size,
            |b, &size| {
                let mut layer = Dense::new(size, output_size, Activation::Sigmoid);
                let input = Array2::ones((1, size));
                
                b.iter(|| {
                    black_box(layer.forward(black_box(&input)).unwrap());
                });
            }
        );
    }
    
    group.finish();
}

fn bench_dense_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_backward");
    group.sample_size(50);
    
    let input_sizes = [10, 100, 500, 1000];
    let output_size = 10;
    
    for &input_size in &input_sizes {
        group.bench_with_input(
            BenchmarkId::new("relu", input_size),
            &input_size,
            |b, &size| {
                let mut layer = Dense::new(size, output_size, Activation::Relu);
                let input = Array2::ones((1, size));
                layer.forward(&input).unwrap();
                let grad = Array2::ones((1, output_size));
                
                b.iter(|| {
                    black_box(layer.backward(black_box(&grad)).unwrap());
                });
            }
        );
    }
    
    group.finish();
}

fn bench_dropout(c: &mut Criterion) {
    let mut group = c.benchmark_group("dropout");
    group.sample_size(50);
    
    let sizes = [10, 100, 1000, 5000];
    let dropout_rates = [0.2, 0.5, 0.8];
    
    for &size in &sizes {
        for &rate in &dropout_rates {
            group.bench_with_input(
                BenchmarkId::new(format!("rate_{}", rate), size),
                &size,
                |b, &s| {
                    let mut layer = Dropout::new(rate);
                    let input = Array2::ones((1, s));
                    
                    b.iter(|| {
                        black_box(layer.forward(black_box(&input)).unwrap());
                    });
                }
            );
        }
    }
    
    group.finish();
}

fn bench_batch_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_norm");
    group.sample_size(50);
    
    let batch_sizes = [1, 8, 32, 128];
    let feature_size = 100;
    
    for &batch_size in &batch_sizes {
        group.bench_with_input(
            BenchmarkId::new("forward", batch_size),
            &batch_size,
            |b, &size| {
                let mut layer = BatchNorm::new(feature_size);
                let input = Array2::ones((size, feature_size));
                
                b.iter(|| {
                    black_box(layer.forward(black_box(&input)).unwrap());
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("backward", batch_size),
            &batch_size,
            |b, &size| {
                let mut layer = BatchNorm::new(feature_size);
                let input = Array2::ones((size, feature_size));
                layer.forward(&input).unwrap();
                let grad = Array2::ones((size, feature_size));
                
                b.iter(|| {
                    black_box(layer.backward(black_box(&grad)).unwrap());
                });
            }
        );
    }
    
    group.finish();
}

fn bench_optimizers(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizers");
    group.sample_size(50);
    
    let param_sizes = [(10, 10), (100, 100), (1000, 10)];
    
    for &(rows, cols) in &param_sizes {
        // SGD
        group.bench_with_input(
            BenchmarkId::new("sgd", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(r, c)| {
                let mut optimizer = SGD::new(0.01);
                let mut params = Array2::ones((r, c));
                let grads = Array2::ones((r, c)) * 0.1;
                
                b.iter(|| {
                    black_box(optimizer.step(black_box(&mut params), black_box(&grads)).unwrap());
                });
            }
        );
        
        // Adam
        group.bench_with_input(
            BenchmarkId::new("adam", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(r, c)| {
                let mut optimizer = Adam::new(0.01);
                let mut params = Array2::ones((r, c));
                let grads = Array2::ones((r, c)) * 0.1;
                
                b.iter(|| {
                    black_box(optimizer.step(black_box(&mut params), black_box(&grads)).unwrap());
                });
            }
        );
        
        // AdamW
        group.bench_with_input(
            BenchmarkId::new("adamw", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(r, c)| {
                let mut optimizer = AdamW::new(0.01, 0.001);
                let mut params = Array2::ones((r, c));
                let grads = Array2::ones((r, c)) * 0.1;
                
                b.iter(|| {
                    black_box(optimizer.step(black_box(&mut params), black_box(&grads)).unwrap());
                });
            }
        );
        
        // RMSProp
        group.bench_with_input(
            BenchmarkId::new("rmsprop", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(r, c)| {
                let mut optimizer = RMSProp::new(0.01);
                let mut params = Array2::ones((r, c));
                let grads = Array2::ones((r, c)) * 0.1;
                
                b.iter(|| {
                    black_box(optimizer.step(black_box(&mut params), black_box(&grads)).unwrap());
                });
            }
        );
    }
    
    group.finish();
}

fn bench_sequential_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_model");
    group.sample_size(30);
    
    // XOR veri seti
    let x = arr2(&[
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]);
    
    let y = arr2(&[
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ]);
    
    group.bench_function("training_epoch", |b| {
        b.iter(|| {
            let mut model = Sequential::new();
            model
                .add(Dense::new(2, 4, Activation::Relu))
                .add(Dense::new(4, 1, Activation::Sigmoid))
                .compile(SGD::new(0.1));
                
            black_box(model.fit(black_box(&x), black_box(&y), 5, 1, None).unwrap());
        });
    });
    
    let architectures = [
        ("small", vec![4]),
        ("medium", vec![8, 4]),
        ("large", vec![16, 8, 4]),
    ];
    
    for (name, hidden_layers) in architectures {
        group.bench_function(&format!("forward_{}", name), |b| {
            let mut model = Sequential::new();
            
            // Giriş katmanı
            model.add(Dense::new(2, hidden_layers[0], Activation::Relu));
            
            // Gizli katmanlar
            for i in 0..hidden_layers.len()-1 {
                model.add(Dense::new(hidden_layers[i], hidden_layers[i+1], Activation::Relu));
            }
            
            // Çıkış katmanı
            model.add(Dense::new(
                *hidden_layers.last().unwrap(), 
                1, 
                Activation::Sigmoid
            ));
            
            model.compile(SGD::new(0.1));
            
            b.iter(|| {
                black_box(model.forward(black_box(&x)).unwrap());
            });
        });
    }
    
    group.finish();
}

fn bench_loss_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_functions");
    group.sample_size(50);
    
    let sizes = [10, 100, 1000, 5000];
    
    for &size in &sizes {
        group.bench_with_input(
            BenchmarkId::new("mse", size),
            &size,
            |b, &s| {
                let loss = MSE::new();
                let y_pred = Array2::ones((1, s));
                let y_true = Array2::zeros((1, s));
                
                b.iter(|| {
                    black_box(loss.forward(black_box(&y_pred), black_box(&y_true)).unwrap());
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("mse_backward", size),
            &size,
            |b, &s| {
                let loss = MSE::new();
                let y_pred = Array2::ones((1, s));
                let y_true = Array2::zeros((1, s));
                
                b.iter(|| {
                    black_box(loss.backward(black_box(&y_pred), black_box(&y_true)).unwrap());
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_dense_forward,
    bench_dense_backward,
    bench_dropout,
    bench_batch_norm,
    bench_optimizers,
    bench_sequential_model,
    bench_loss_functions
);
criterion_main!(benches); 
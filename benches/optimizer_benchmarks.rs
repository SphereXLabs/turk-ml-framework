use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use pprof::criterion::{Output, PProfProfiler};
use ndarray::{Array, Array2, Axis};
use turk::{
    Optimizer, SGD, Adam, AdamW, RMSprop,
    Result
};

// Parametre güncelleme performansını ölçen yardımcı fonksiyon
fn benchmark_parameter_update(c: &mut Criterion, optimizer_name: &str, mut optimizer: Box<dyn Optimizer>) {
    let mut group = c.benchmark_group(format!("{}_optimizer", optimizer_name));
    
    // Farklı parametre boyutları için test
    let param_sizes = [1000, 10_000, 100_000, 1_000_000];
    
    for &size in &param_sizes {
        // Rastgele parametreler ve gradyanlar oluştur
        let params = Array::random((size,), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
        let grads = Array::random((size,), ndarray_rand::rand_distr::Uniform::new(-0.1, 0.1));
        
        group.bench_with_input(
            BenchmarkId::new("parameter_update", size), 
            &size,
            |b, _| {
                b.iter(|| {
                    let mut params_clone = params.clone();
                    let updated_params = optimizer.update(&mut params_clone, &grads);
                    black_box(updated_params)
                });
            },
        );
    }
    
    group.finish();
}

// Farklı öğrenme oranları için SGD performansını karşılaştır
fn bench_sgd_learning_rates(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgd_learning_rates");
    
    let learning_rates = [0.001, 0.01, 0.1, 0.5];
    let momentum_values = [0.0, 0.9, 0.99];
    let param_size = 10_000;
    
    // Rastgele parametreler ve gradyanlar oluştur
    let params = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
    let grads = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-0.1, 0.1));
    
    for &lr in &learning_rates {
        for &momentum in &momentum_values {
            let config_str = format!("lr{}_m{}", lr, momentum);
            let mut optimizer = SGD::new(lr);
            optimizer.set_momentum(momentum);
            
            group.bench_with_input(
                BenchmarkId::new("update", &config_str), 
                &lr,
                |b, _| {
                    b.iter(|| {
                        let mut params_clone = params.clone();
                        let updated_params = optimizer.update(&mut params_clone, &grads);
                        black_box(updated_params)
                    });
                },
            );
        }
    }
    
    group.finish();
}

// Farklı beta değerleri için Adam performansını karşılaştır
fn bench_adam_betas(c: &mut Criterion) {
    let mut group = c.benchmark_group("adam_betas");
    
    let lr = 0.001;
    let beta1_values = [0.9, 0.95, 0.99];
    let beta2_values = [0.999, 0.9999];
    let param_size = 10_000;
    
    // Rastgele parametreler ve gradyanlar oluştur
    let params = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
    let grads = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-0.1, 0.1));
    
    for &beta1 in &beta1_values {
        for &beta2 in &beta2_values {
            let config_str = format!("b1{}_b2{}", beta1, beta2);
            let mut optimizer = Adam::new(lr);
            optimizer.set_betas(beta1, beta2);
            
            group.bench_with_input(
                BenchmarkId::new("update", &config_str), 
                &beta1,
                |b, _| {
                    b.iter(|| {
                        let mut params_clone = params.clone();
                        let updated_params = optimizer.update(&mut params_clone, &grads);
                        black_box(updated_params)
                    });
                },
            );
        }
    }
    
    group.finish();
}

// Farklı ağırlık azaltma (weight decay) değerleri için AdamW performansını karşılaştır
fn bench_adamw_weight_decay(c: &mut Criterion) {
    let mut group = c.benchmark_group("adamw_weight_decay");
    
    let lr = 0.001;
    let weight_decay_values = [0.0, 0.001, 0.01, 0.1];
    let param_size = 10_000;
    
    // Rastgele parametreler ve gradyanlar oluştur
    let params = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
    let grads = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-0.1, 0.1));
    
    for &weight_decay in &weight_decay_values {
        let mut optimizer = AdamW::new(lr);
        optimizer.set_weight_decay(weight_decay);
        
        group.bench_with_input(
            BenchmarkId::new("update", weight_decay), 
            &weight_decay,
            |b, _| {
                b.iter(|| {
                    let mut params_clone = params.clone();
                    let updated_params = optimizer.update(&mut params_clone, &grads);
                    black_box(updated_params)
                });
            },
        );
    }
    
    group.finish();
}

// Farklı decay değerleri için RMSprop performansını karşılaştır
fn bench_rmsprop_decay(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsprop_decay");
    
    let lr = 0.001;
    let decay_values = [0.9, 0.95, 0.99, 0.999];
    let param_size = 10_000;
    
    // Rastgele parametreler ve gradyanlar oluştur
    let params = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
    let grads = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-0.1, 0.1));
    
    for &decay in &decay_values {
        let mut optimizer = RMSprop::new(lr);
        optimizer.set_decay(decay);
        
        group.bench_with_input(
            BenchmarkId::new("update", decay), 
            &decay,
            |b, _| {
                b.iter(|| {
                    let mut params_clone = params.clone();
                    let updated_params = optimizer.update(&mut params_clone, &grads);
                    black_box(updated_params)
                });
            },
        );
    }
    
    group.finish();
}

// Epsilon değerlerinin optimizer performansına etkisi
fn bench_epsilon_effect(c: &mut Criterion) {
    let mut group = c.benchmark_group("epsilon_effect");
    
    let lr = 0.001;
    let epsilon_values = [1e-8, 1e-7, 1e-6, 1e-5];
    let param_size = 10_000;
    
    // Rastgele parametreler ve gradyanlar oluştur
    let params = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
    let grads = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-0.1, 0.1));
    
    for &epsilon in &epsilon_values {
        // Adam
        let mut adam = Adam::new(lr);
        adam.set_epsilon(epsilon);
        
        group.bench_with_input(
            BenchmarkId::new("adam", epsilon), 
            &epsilon,
            |b, _| {
                b.iter(|| {
                    let mut params_clone = params.clone();
                    let updated_params = adam.update(&mut params_clone, &grads);
                    black_box(updated_params)
                });
            },
        );
        
        // RMSprop
        let mut rmsprop = RMSprop::new(lr);
        rmsprop.set_epsilon(epsilon);
        
        group.bench_with_input(
            BenchmarkId::new("rmsprop", epsilon), 
            &epsilon,
            |b, _| {
                b.iter(|| {
                    let mut params_clone = params.clone();
                    let updated_params = rmsprop.update(&mut params_clone, &grads);
                    black_box(updated_params)
                });
            },
        );
    }
    
    group.finish();
}

// Optimizerların iterasyon sayısı karşılaştırması
fn bench_iteration_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_iteration_count");
    
    let param_size = 10_000;
    let iteration_counts = [1, 10, 100];
    
    // Rastgele parametreler ve gradyanlar oluştur
    let params = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0));
    let grads = Array::random((param_size,), ndarray_rand::rand_distr::Uniform::new(-0.1, 0.1));
    
    for &iterations in &iteration_counts {
        // SGD
        let sgd = SGD::new(0.01);
        
        group.bench_with_input(
            BenchmarkId::new("sgd", iterations), 
            &iterations,
            |b, _| {
                b.iter(|| {
                    let mut params_clone = params.clone();
                    for _ in 0..iterations {
                        sgd.update(&mut params_clone, &grads);
                    }
                    black_box(params_clone)
                });
            },
        );
        
        // Adam
        let adam = Adam::new(0.001);
        
        group.bench_with_input(
            BenchmarkId::new("adam", iterations), 
            &iterations,
            |b, _| {
                b.iter(|| {
                    let mut params_clone = params.clone();
                    for _ in 0..iterations {
                        adam.update(&mut params_clone, &grads);
                    }
                    black_box(params_clone)
                });
            },
        );
        
        // RMSprop
        let rmsprop = RMSprop::new(0.001);
        
        group.bench_with_input(
            BenchmarkId::new("rmsprop", iterations), 
            &iterations,
            |b, _| {
                b.iter(|| {
                    let mut params_clone = params.clone();
                    for _ in 0..iterations {
                        rmsprop.update(&mut params_clone, &grads);
                    }
                    black_box(params_clone)
                });
            },
        );
    }
    
    group.finish();
}

// Optimizerların genel karşılaştırması
fn compare_optimizers(c: &mut Criterion) {
    // SGD
    benchmark_parameter_update(c, "sgd", Box::new(SGD::new(0.01)));
    
    // Adam
    benchmark_parameter_update(c, "adam", Box::new(Adam::new(0.001)));
    
    // AdamW
    let mut adamw = AdamW::new(0.001);
    adamw.set_weight_decay(0.01);
    benchmark_parameter_update(c, "adamw", Box::new(adamw));
    
    // RMSprop
    benchmark_parameter_update(c, "rmsprop", Box::new(RMSprop::new(0.001)));
}

// Tüm optimizer benchmark'larını çalıştır
fn optimizer_benchmarks(c: &mut Criterion) {
    compare_optimizers(c);
    bench_sgd_learning_rates(c);
    bench_adam_betas(c);
    bench_adamw_weight_decay(c);
    bench_rmsprop_decay(c);
    bench_epsilon_effect(c);
    bench_iteration_count(c);
}
criterion_group! {
    name = optimizer_benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = optimizer_benchmarks
}
criterion_main!(optimizer_benches); 

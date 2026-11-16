//! Tensor operasyonları için performans karşılaştırması
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, ArrayView2, Zip};
use rayon::prelude::*;
use turk::utils::tensor_ops;

/// Sekansiyel elementwise çarpım
fn sequential_mul(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros(a.dim());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            result[[i, j]] = a[[i, j]] * b[[i, j]];
        }
    }
    result
}

/// Rayon ile paralel elementwise çarpım
fn parallel_mul(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros(a.dim());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .par_for_each(|c, &a, &b| {
            *c = a * b;
        });
    result
}

/// Sekansiyel MSE hesaplama
fn sequential_mse(y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> f64 {
    let n_samples = y_true.nrows();
    let n_outputs = y_true.ncols();
    let mut mse = 0.0;

    for i in 0..n_samples {
        for j in 0..n_outputs {
            let diff = y_true[[i, j]] - y_pred[[i, j]];
            mse += diff * diff;
        }
    }

    mse /= (n_samples * n_outputs) as f64;
    mse
}

/// Rayon ile paralel MSE hesaplama
fn parallel_mse(y_true: &ArrayView2<f64>, y_pred: &ArrayView2<f64>) -> f64 {
    let n_samples = y_true.nrows();
    let n_outputs = y_true.ncols();

    let squared_error_sum: f64 = Zip::from(y_true)
        .and(y_pred)
        .par_map_collect(|&y_t, &y_p| {
            let diff = y_t - y_p;
            diff * diff
        })
        .sum();

    squared_error_sum / (n_samples * n_outputs) as f64
}

fn tensor_op_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tensor Operasyonları");

    // Tensor boyutları için parametreler
    let sizes = vec![100, 500, 1000, 5000];

    for size in sizes {
        // Random matrisler oluştur
        let a = Array2::from_shape_fn((size, size), |_| rand::random::<f64>());
        let b = Array2::from_shape_fn((size, size), |_| rand::random::<f64>());

        // ElementWise Çarpım Benchmarkları
        group.bench_with_input(
            BenchmarkId::new("Elementwise_Sequential", size),
            &size,
            |bench, _| {
                bench.iter(|| sequential_mul(&a.view(), &b.view()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Elementwise_Parallel", size),
            &size,
            |bench, _| {
                bench.iter(|| parallel_mul(&a.view(), &b.view()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Elementwise_Library", size),
            &size,
            |bench, _| {
                bench.iter(|| tensor_ops::elementwise_mul(&a.view(), &b.view()).unwrap());
            },
        );

        // MSE Benchmarkları
        group.bench_with_input(
            BenchmarkId::new("MSE_Sequential", size),
            &size,
            |bench, _| {
                bench.iter(|| sequential_mse(&a.view(), &b.view()));
            },
        );

        group.bench_with_input(BenchmarkId::new("MSE_Parallel", size), &size, |bench, _| {
            bench.iter(|| parallel_mse(&a.view(), &b.view()));
        });

        group.bench_with_input(BenchmarkId::new("MSE_Library", size), &size, |bench, _| {
            bench.iter(|| tensor_ops::mse(&a.view(), &b.view()).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, tensor_op_benchmark);
criterion_main!(benches);

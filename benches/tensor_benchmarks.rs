//! Tensor operasyonları için performans karşılaştırması
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array2, ArrayView2, Zip};

fn sequential_mul(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros(a.dim());
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            result[[i, j]] = a[[i, j]] * b[[i, j]];
        }
    }
    result
}

fn parallel_mul(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> Array2<f64> {
    let mut result = Array2::zeros(a.dim());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .par_for_each(|c, &av, &bv| {
            *c = av * bv;
        });
    result
}

fn bench_tensor_mul(c: &mut Criterion) {
    let a = Array2::from_elem((256, 256), 1.1);
    let b = Array2::from_elem((256, 256), 0.9);
    c.bench_function("sequential_mul_256", |bencher| {
        bencher.iter(|| black_box(sequential_mul(&a.view(), &b.view())));
    });
    c.bench_function("parallel_mul_256", |bencher| {
        bencher.iter(|| black_box(parallel_mul(&a.view(), &b.view())));
    });
}

criterion_group!(benches, bench_tensor_mul);
criterion_main!(benches);

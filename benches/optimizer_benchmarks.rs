use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use turk::{Adam, AdamW, Optimizer, RMSProp, SGD};

fn random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| ((i + j) % 7) as f64 * 0.01 - 0.03)
}

fn bench_optimizers(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_update");
    let w = random_matrix(128, 64);
    let b = random_matrix(1, 64);
    let gw = random_matrix(128, 64);
    let gb = random_matrix(1, 64);

    group.bench_function("sgd", |bencher| {
        let opt = SGD::new(0.01, Some(0.9), false);
        bencher.iter(|| {
            let mut weights = w.clone();
            let mut biases = b.clone();
            opt.update(&mut weights, &mut biases, &gw.view(), &gb.view())
                .unwrap();
            black_box((weights, biases));
        });
    });

    group.bench_function("adam", |bencher| {
        let opt = Adam::new(0.001, 0.9, 0.999, 1e-8);
        bencher.iter(|| {
            let mut weights = w.clone();
            let mut biases = b.clone();
            opt.update(&mut weights, &mut biases, &gw.view(), &gb.view())
                .unwrap();
            black_box((weights, biases));
        });
    });

    group.bench_function("adamw", |bencher| {
        let opt = AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01);
        bencher.iter(|| {
            let mut weights = w.clone();
            let mut biases = b.clone();
            opt.update(&mut weights, &mut biases, &gw.view(), &gb.view())
                .unwrap();
            black_box((weights, biases));
        });
    });

    group.bench_function("rmsprop", |bencher| {
        let opt = RMSProp::new(0.001, 0.9, 1e-8);
        bencher.iter(|| {
            let mut weights = w.clone();
            let mut biases = b.clone();
            opt.update(&mut weights, &mut biases, &gw.view(), &gb.view())
                .unwrap();
            black_box((weights, biases));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_optimizers);
criterion_main!(benches);

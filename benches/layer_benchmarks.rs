use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use turk::aktivasyon::Activation;
use turk::initializers::{GlorotUniform, Zeros};
use turk::layer::{Dense, Layer};

fn bench_dense_forward(c: &mut Criterion) {
    let mut dense = Dense::new(
        256,
        128,
        Activation::Relu,
        Box::new(GlorotUniform::new()),
        Box::new(Zeros::new()),
        true,
    )
    .unwrap();
    let input = Array2::from_elem((64, 256), 0.05);
    c.bench_function("dense_forward_64x256", |b| {
        b.iter(|| {
            let out = dense.forward(&input.view()).unwrap();
            black_box(out);
        });
    });
}

criterion_group!(benches, bench_dense_forward);
criterion_main!(benches);

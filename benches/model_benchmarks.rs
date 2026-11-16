use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use turk::aktivasyon::Activation;
use turk::initializers::{GlorotUniform, Zeros};
use turk::layer::Dense;
use turk::{Sequential, SGD};

fn make_model() -> Sequential {
    let mut model = Sequential::new();
    model
        .add(
            Dense::new(
                64,
                32,
                Activation::Relu,
                Box::new(GlorotUniform::new()),
                Box::new(Zeros::new()),
                true,
            )
            .unwrap(),
        )
        .add(
            Dense::new(
                32,
                10,
                Activation::Softmax,
                Box::new(GlorotUniform::new()),
                Box::new(Zeros::new()),
                true,
            )
            .unwrap(),
        )
        .compile(SGD::new(0.01, None, false));
    model
}

fn bench_model_forward(c: &mut Criterion) {
    let mut model = make_model();
    let input = Array2::from_elem((32, 64), 0.1);
    c.bench_function("sequential_forward_32x64", |b| {
        b.iter(|| {
            let out = model.forward(&input.view()).unwrap();
            black_box(out);
        });
    });
}

criterion_group!(benches, bench_model_forward);
criterion_main!(benches);

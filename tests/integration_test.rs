use ndarray::Array2;
use turk::{
    aktivasyon::Activation,
    initializers::{GlorotUniform, Zeros},
    layer::{Dense, Dropout, Layer},
    metrics::{Accuracy, Metric},
    model::Sequential,
    optimizer::{Adam, AdamW, Optimizer, RMSProp, SGD},
    utils::{one_hot_encode, tensor_ops, train_test_split},
    TurkError,
};

fn test_dense(i: usize, o: usize, act: Activation) -> Dense {
    Dense::new(
        i,
        o,
        act,
        Box::new(GlorotUniform::new()),
        Box::new(Zeros::new()),
        true,
    )
    .unwrap()
}

#[test]
fn test_model_dsl_pipeline() {
    // TurkDil üzerinden DSL tabanlı pipeline testi
    let mut dil = turk::TurkDil::new();
    let result = dil.calistir("model_olustur()");
    assert!(result.is_ok());
}

#[test]
fn test_optimizer_implementasyonlari() {
    let grad_w = Array2::ones((2, 2));
    let grad_b = Array2::from_elem((1, 2), 0.05);

    let mut w = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let mut b = Array2::from_elem((1, 2), 0.5);
    SGD::new(0.01, None, false)
        .update(&mut w, &mut b, &grad_w.view(), &grad_b.view())
        .unwrap();
    assert!(w[[0, 0]] < 1.0);

    let mut w = Array2::zeros((2, 2));
    let mut b = Array2::zeros((1, 2));
    Adam::new(0.001, 0.9, 0.999, 1e-8)
        .update(&mut w, &mut b, &grad_w.view(), &grad_b.view())
        .unwrap();
    assert!(w[[0, 0]].abs() > 1e-6);

    let mut w = Array2::zeros((2, 2));
    let mut b = Array2::zeros((1, 2));
    AdamW::new(0.001, 0.9, 0.999, 1e-8, 0.01)
        .update(&mut w, &mut b, &grad_w.view(), &grad_b.view())
        .unwrap();
    assert!(w[[0, 0]].abs() > 1e-6);

    let mut w = Array2::zeros((2, 2));
    let mut b = Array2::zeros((1, 2));
    RMSProp::new(0.001, 0.9, 1e-8)
        .update(&mut w, &mut b, &grad_w.view(), &grad_b.view())
        .unwrap();
    assert!(w[[0, 0]].abs() > 1e-6);
}

#[test]
fn test_katman_implementasyonlari() {
    let mut d = test_dense(2, 3, Activation::Relu);
    let input = Array2::ones((1, 2));
    let out = d.forward(&input.view()).unwrap();
    assert_eq!(out.shape(), &[1, 3]);

    let mut dropout = Dropout::new(0.5);
    let out = dropout.forward(&input.view()).unwrap();
    assert_eq!(out.shape(), input.shape());
}

#[test]
fn test_veri_isleme() {
    let data =
        Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let labels = vec![0usize, 1, 2, 0];
    let oh = one_hot_encode(&labels, 3).unwrap();
    assert_eq!(oh.shape(), &[4, 3]);
    let (tr, te, _y_tr, _y_te) = train_test_split(&data.view(), &oh.view(), 0.25).unwrap();
    assert_eq!(tr.nrows() + te.nrows(), 4);
}

#[test]
fn test_metrikler() {
    let y_true = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
    let y_pred = Array2::from_shape_vec((3, 2), vec![0.9, 0.1, 0.2, 0.8, 0.85, 0.15]).unwrap();
    let acc = Accuracy::new().compute(&y_true.view(), &y_pred.view());
    assert!((0.0..=1.0).contains(&acc));
    let mse = tensor_ops::mse(&y_true.view(), &y_pred.view()).unwrap();
    assert!(mse >= 0.0);
}

#[test]
fn test_hata_yonetimi() {
    let mut model = Sequential::new();
    model
        .add(test_dense(2, 3, Activation::Relu))
        .compile(SGD::new(0.01, None, false));
    let bad = Array2::zeros((1, 4));
    assert!(matches!(
        model.forward(&bad.view()),
        Err(TurkError::ShapeMismatch(_))
    ));
    assert!(model.load("olmayan.json").is_err());
}

#[test]
fn test_model_serialization_roundtrip() {
    let path = std::env::temp_dir().join("turk_integ_test.json");
    let path_str = path.to_str().unwrap();
    let mut m = Sequential::new();
    m.add(test_dense(2, 3, Activation::Relu))
        .add(test_dense(3, 1, Activation::Sigmoid))
        .compile(SGD::new(0.1, None, false));
    m.save(path_str).unwrap();

    let mut loaded = Sequential::new();
    loaded
        .add(test_dense(2, 3, Activation::Relu))
        .add(test_dense(3, 1, Activation::Sigmoid))
        .compile(SGD::new(0.1, None, false));
    loaded.load(path_str).unwrap();

    let inp = Array2::from_elem((1, 2), 0.5);
    let o1 = m.forward(&inp.view()).unwrap();
    let o2 = loaded.forward(&inp.view()).unwrap();
    assert_eq!(o1.shape(), o2.shape());
    let _ = std::fs::remove_file(path);
}

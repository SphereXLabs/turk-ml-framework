//! Embedding, Conv1D gradyan akışı ve DSL ayrıştırma testleri.

use ndarray::Array2;
use turk::aktivasyon::Activation;
use turk::cnn::Conv1D;
use turk::dsl_parser::{komut_ayristir, DslKomut};
use turk::katman_kayit::KatmanKayit;
use turk::layer::{Embedding, Layer};
use turk::model::Sequential;
use turk::TurkDil;

#[test]
fn test_embedding_forward_backward() {
    let mut emb = Embedding::new(10, 4, 3).unwrap();
    assert_eq!(emb.input_size(), 3);
    assert_eq!(emb.output_size(), 12);

    let x = Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 9.0]).unwrap();
    let y = emb.forward(&x.view()).unwrap();
    assert_eq!(y.shape(), &[2, 12]);

    let grad = Array2::ones(y.dim());
    let (g_in, g_w, g_b) = emb.backward(&x.view(), &grad.view()).unwrap();
    assert_eq!(g_in.shape(), x.shape());
    assert_eq!(g_w.shape(), &[10, 4]);
    assert_eq!(g_b.len(), 0);
    // Kullanılan token satırlarında gradyan birikmeli
    assert!(g_w.row(0).iter().all(|&v| (v - 1.0).abs() < 1e-9));
    assert!(g_w.row(1).iter().all(|&v| (v - 1.0).abs() < 1e-9));
    assert!(g_w.iter().any(|&v| v.abs() > 0.0));

    let before = emb.get_weights().unwrap()[[0, 0]];
    emb.update(&g_w.view(), &g_b.view(), 0.1).unwrap();
    let after = emb.get_weights().unwrap()[[0, 0]];
    assert!((after - (before - 0.1)).abs() < 1e-9);
}

#[test]
fn test_conv1d_forward_backward() {
    let mut conv = Conv1D::new(2, 3, 3, 1, 8, Activation::Linear).unwrap();
    assert_eq!(conv.input_size(), 16);
    assert_eq!(conv.out_len, 6);
    assert_eq!(conv.output_size(), 18);

    let x = Array2::from_elem((2, 16), 0.5);
    let y = conv.forward(&x.view()).unwrap();
    assert_eq!(y.shape(), &[2, 18]);

    let grad = Array2::ones(y.dim());
    let (g_in, g_w, g_b) = conv.backward(&x.view(), &grad.view()).unwrap();
    assert_eq!(g_in.shape(), x.shape());
    assert_eq!(g_w.shape(), &[6, 3]);
    assert_eq!(g_b.shape(), &[1, 3]);
    assert!(g_w.iter().any(|&v| v.abs() > 1e-12));
    assert!(g_b.iter().any(|&v| v.abs() > 1e-12));
}

#[test]
fn test_dsl_gommeli_ve_evrisim1d() {
    let k = KatmanKayit::varsayilan();
    let emb = k
        .dsl_olustur_tamam("gommeli(kelime=1000, boyut=32)")
        .unwrap();
    assert_eq!(emb.layer_type(), "Embedding");
    assert_eq!(emb.input_size(), 1);
    assert_eq!(emb.output_size(), 32);

    let emb2 = k
        .dsl_olustur_tamam("gommeli(kelime=100, boyut=8, sekans=5)")
        .unwrap();
    assert_eq!(emb2.input_size(), 5);
    assert_eq!(emb2.output_size(), 40);

    let c1 = k
        .dsl_olustur_tamam("evrisim1d(giris=1, cikis=8, cekirdek=3)")
        .unwrap();
    assert_eq!(c1.layer_type(), "Conv1D");
    assert_eq!(c1.input_size(), 16); // varsayılan uzunluk=16
    assert_eq!(c1.output_size(), 14 * 8);

    let c2 = k
        .dsl_olustur_tamam("conv1d(giris=8, cikis=4, cekirdek=3, adim=1, uzunluk=10)")
        .unwrap();
    assert_eq!(c2.input_size(), 80);
    assert_eq!(c2.output_size(), 8 * 4);
}

#[test]
fn test_dsl_parser_ikili_komutlar() {
    let k = komut_ayristir("modeli_ikili_kaydet(\"m.bin\")", 1).unwrap();
    assert!(matches!(k, DslKomut::ModeliIkiliKaydet { .. }));
    let k = komut_ayristir("ikili_yukle(\"m.bin\")", 2).unwrap();
    assert!(matches!(k, DslKomut::ModeliIkiliYukle { .. }));
}

#[test]
fn test_embedding_conv1d_zincir_ve_bincode() {
    let mut dil = TurkDil::new();
    dil.calistir("model_olustur()").unwrap();
    dil.calistir("katman_ekle(gommeli(kelime=20, boyut=4, sekans=6))")
        .unwrap();
    dil.calistir("katman_ekle(evrisim1d(giris=4, cikis=2, cekirdek=3, uzunluk=6))")
        .unwrap();
    dil.calistir("katman_ekle(yogun(8, 1, sigmoid))").unwrap();
    dil.calistir("derle(adam, loss=mse, ogrenme_orani=0.05)")
        .unwrap();

    let model = dil.model.as_ref().expect("model");
    let ozet = model.summary();
    assert!(ozet.contains("Embedding"));
    assert!(ozet.contains("Conv1D"));

    let bytes = model.to_bincode_bytes().unwrap();
    let json = model.to_json_string().unwrap();
    assert!(bytes.len() * 3 < json.len() || bytes.len() < json.len());

    let mut loaded = Sequential::new();
    // JSON yükleme için önce topoloji gerekir — bincode apply kendi katmanlarını kurar
    loaded.load_from_bincode_bytes(&bytes).unwrap();
    assert_eq!(loaded.layers.len(), 3);
    assert_eq!(loaded.layers[0].layer_type(), "Embedding");
    assert_eq!(loaded.layers[1].layer_type(), "Conv1D");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("alper_test.bin");
    let path_str = path.to_str().unwrap();
    model.modeli_ikili_kaydet(path_str).unwrap();
    let mut from_disk = Sequential::new();
    from_disk.modeli_ikili_yukle(path_str).unwrap();
    assert_eq!(from_disk.layers.len(), 3);
}

#[test]
fn test_turk_dil_katman_ekle_named() {
    let mut dil = TurkDil::new();
    dil.calistir("model_olustur()").unwrap();
    let r = dil.calistir("katman_ekle(gommeli(kelime=50, boyut=8, sekans=5))");
    assert!(r.is_ok(), "{r:?}");
    // uzunluk < cekirdek → katman üretim hatası
    let r = dil.calistir("katman_ekle(evrisim1d(giris=8, cikis=4, cekirdek=3, uzunluk=1))");
    assert!(r.is_err());
    // Embedding çıkışı 5*8=40 → Conv1D giris=8, uzunluk=5
    let r = dil.calistir("katman_ekle(evrisim1d(giris=8, cikis=4, cekirdek=3, uzunluk=5))");
    assert!(r.is_ok(), "{r:?}");
}

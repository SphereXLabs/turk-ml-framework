//! MNIST veri setini CSV formatina aktarir (784 piksel + label).
//!
//! Kullanim:
//!   cargo run --example mnist_csv_export -- mnist_train.csv
//!   cargo run --example mnist_csv_export -- mnist_train.csv 5000

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let cikti = args.get(1).map(String::as_str).unwrap_or("mnist_train.csv");
    let limit: u32 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000)
        .min(60_000);

    let mnist = mnist::MnistBuilder::new()
        .label_format_digit()
        .training_set_length(limit)
        .test_set_length(0)
        .finalize();

    let n = mnist.trn_img.len() / 784;
    eprintln!("MNIST CSV yaziliyor: {cikti} ({n} satir, 784 ozellik + label)");

    let file = File::create(cikti)?;
    let mut w = BufWriter::new(file);

    // Baslik: pixel_0 .. pixel_783, label
    for i in 0..784 {
        write!(w, "pixel_{i},")?;
    }
    writeln!(w, "label")?;

    for i in 0..n {
        for j in 0..784 {
            let p = mnist.trn_img[i * 784 + j] as f64 / 255.0;
            write!(w, "{p},")?;
        }
        writeln!(w, "{}", mnist.trn_lbl[i])?;
    }

    w.flush()?;
    eprintln!("Tamamlandi: {cikti}");
    Ok(())
}

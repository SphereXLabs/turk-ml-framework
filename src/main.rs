//! TÜRK Programlama Dili — Ana CLI Giriş Noktası
//!
//! Kullanım:
//! - `turk run <dosya.turk>` — .turk dosyasını çalıştır
//! - `turk check <dosya.turk>` — statik analiz (eğitim başlatmadan)
//! - `turk repl` — etkileşimli mod
//! - `turk --yardim | -y` — yardım ekranı

use turk::dil::Value;
use turk::error::{Result, TurkError};
use turk::hata_satirini_goster;
use turk::model::Sequential;
use turk::onnx::export_onnx;
use turk::TurkDil;
use turk::{cihaz_listesi_metin, model_bilgi_metin, model_indir, zoo_listesi_metin};

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::process;

use env_logger::Builder;
use log::{error, info, LevelFilter};

// ─── Sabitler ──────────────────────────────────────────────────────────────
const VERSIYON: &str = env!("CARGO_PKG_VERSION");

const BANNER: &str = r#"
  ╔════════════════════════════════════════════════════╗
  ║  ████████╗██╗   ██╗██████╗ ██╗  ██╗               ║
  ║     ██╔══╝██║   ██║██╔══██╗██║ ██╔╝               ║
  ║     ██║   ██║   ██║██████╔╝█████╔╝                ║
  ║     ██║   ██║   ██║██╔══██╗██╔═██╗                ║
  ║     ██║   ╚██████╔╝██║  ██║██║  ██╗               ║
  ║     ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝              ║
  ║   Türkçe Yapay Zeka Programlama Dili               ║
  ╚════════════════════════════════════════════════════╝
"#;

const KULLANIM: &str = r#"
  Kullanım:
    turk run   <dosya.turk>    — .turk dosyasını çalıştır
    turk check <dosya.turk>    — katman/veri uyumunu eğitimsiz doğrula
    turk export <model.json> [cikti.onnx.json] — ONNX JSON dışa aktar
    turk cihaz                  — kullanılabilir hesaplama cihazlarını listele
    turk zoo list               — Model Zoo listesi
    turk zoo bilgi <id>         — model detayı
    turk zoo indir <id> [hedef] — model dosyalarını indir/kopyala
    turk zoo calistir <id>      — Model Zoo betiğini çalıştır
    turk repl                  — etkileşimli REPL modu başlat
    turk -y  | --yardim       — bu yardım ekranını göster

  Örnekler:
    cargo run -- run examples/proje.turk
    cargo run -- check examples/proje.turk
    cargo run -- repl
    cargo run -- run egitimler/mnist.turk

  Desteklenen Komutlar (.turk içinde):
    model_olustur()
    katman_ekle(yogun(girdi, cikti, "aktivasyon"))
    katman_ekle(dropout(oran))
    katman_ekle(batch_norm(ozellik_sayisi))
    derle(adam)  |  derle(sgd)  |  derle(adamw, cihaz=paralel)
    egit(epoch, batch_boyutu)
    kaydet("model.json")
    yukle("model.json")
    onnx_disa_aktar("model.onnx.json")
    model_bilgisi()
    cihaz_bilgisi()
    model_zoo()
    model_indir("xor")
    model_calistir("xor")
"#;

// ─── Ana Giriş ─────────────────────────────────────────────────────────────
fn main() {
    Builder::new()
        .filter_module("turk", LevelFilter::Info)
        .filter_level(LevelFilter::Warn)
        .init();

    if let Err(e) = run() {
        eprintln!("\n  \x1b[31m✗ Hata:\x1b[0m {e}\n");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    match args.get(1).map(String::as_str) {
        // ── turk run <dosya> ───────────────────────────────────────────────
        Some("run") => {
            let dosya = args.get(2).ok_or_else(|| {
                eprintln!("{KULLANIM}");
                TurkError::SyntaxError(
                    "Dosya yolu belirtilmedi. Örnek: turk run examples/proje.turk".into(),
                )
            })?;
            dosyayi_calistir(dosya)
        }

        // ── turk check <dosya> ─────────────────────────────────────────────
        Some("check") => {
            let dosya = args.get(2).ok_or_else(|| {
                eprintln!("{KULLANIM}");
                TurkError::SyntaxError(
                    "Dosya yolu belirtilmedi. Örnek: turk check examples/proje.turk".into(),
                )
            })?;
            dosyayi_kontrol_et(dosya)
        }

        // ── turk export <model.json> [out.onnx.json] ───────────────────────
        Some("export") => {
            let json = args.get(2).ok_or_else(|| {
                eprintln!("{KULLANIM}");
                TurkError::SyntaxError(
                    "Model JSON yolu belirtilmedi. Örnek: turk export model.json".into(),
                )
            })?;
            let out = args.get(3).map(String::as_str).unwrap_or("model.onnx.json");
            model_json_onnx_aktar(json, out)
        }

        // ── turk cihaz ─────────────────────────────────────────────────────
        Some("cihaz") => {
            println!("{BANNER}");
            println!("{}", cihaz_listesi_metin());
            Ok(())
        }

        // ── turk zoo list|bilgi|indir|calistir ────────────────────────────
        Some("zoo") => {
            let alt = args.get(2).map(String::as_str).unwrap_or("list");
            match alt {
                "list" | "liste" => {
                    println!("{BANNER}");
                    println!("{}", zoo_listesi_metin());
                    Ok(())
                }
                "bilgi" | "info" => {
                    let id = args.get(3).ok_or_else(|| {
                        TurkError::SyntaxError(
                            "Model id belirtilmedi. Örnek: turk zoo bilgi xor".into(),
                        )
                    })?;
                    println!("{}", model_bilgi_metin(id)?);
                    Ok(())
                }
                "indir" | "download" => {
                    let id = args.get(3).ok_or_else(|| {
                        TurkError::SyntaxError(
                            "Model id belirtilmedi. Örnek: turk zoo indir xor".into(),
                        )
                    })?;
                    let hedef = args.get(4).map(|s| s.as_str());
                    let yol = model_indir(id, hedef.map(std::path::Path::new))?;
                    println!("  \x1b[32m✓\x1b[0m Model indirildi: {}\n", yol.display());
                    Ok(())
                }
                "calistir" | "run" => {
                    let id = args.get(3).ok_or_else(|| {
                        TurkError::SyntaxError(
                            "Model id belirtilmedi. Örnek: turk zoo calistir xor".into(),
                        )
                    })?;
                    zoo_model_calistir(id)
                }
                other => Err(TurkError::SyntaxError(format!(
                    "Bilinmeyen zoo alt komutu: '{other}'. \
                     Geçerli: list, bilgi, indir, calistir"
                ))),
            }
        }

        // ── turk repl ─────────────────────────────────────────────────────
        Some("repl") => {
            println!("{BANNER}");
            println!("  Sürüm: {VERSIYON}  |  Etkileşimli Mod");
            println!("  Çıkmak için: çıkış  ya da  exit\n");
            etkisesimli_mod()
        }

        // ── yardım ────────────────────────────────────────────────────────
        Some("-y") | Some("--yardim") | Some("help") | Some("-h") | Some("--help") => {
            println!("{BANNER}");
            println!("  Sürüm: {VERSIYON}");
            println!("{KULLANIM}");
            Ok(())
        }

        // ── tek argüman → doğrudan dosya yolu kabul et (backwards compat) ─
        Some(arg) if arg.ends_with(".turk") => dosyayi_calistir(arg),

        // ── bilinmeyen / argümansız ────────────────────────────────────────
        _ => {
            println!("{BANNER}");
            eprintln!("{KULLANIM}");
            if args.len() <= 1 {
                // Argüman yoksa REPL'i aç
                println!("  (Argüman verilmedi — etkileşimli mod başlatılıyor)");
                println!("  Çıkmak için: çıkış  ya da  exit\n");
                etkisesimli_mod()
            } else {
                Err(TurkError::SyntaxError(format!(
                    "Bilinmeyen komut: '{}'. `turk --yardim` ile kullanımı görün.",
                    args[1]
                )))
            }
        }
    }
}

// ─── Dosya Çalıştırıcı ─────────────────────────────────────────────────────
fn dosyayi_calistir(dosya_yolu: &str) -> Result<()> {
    let yol = Path::new(dosya_yolu);

    // Uzantı kontrolü (uyarı, hata değil)
    if yol.extension().and_then(|e| e.to_str()) != Some("turk") {
        eprintln!(
            "  \x1b[33m⚠ Uyarı:\x1b[0m '{dosya_yolu}' .turk uzantılı değil, yine de çalıştırılıyor."
        );
    }

    let icerik = fs::read_to_string(yol)
        .map_err(|e| TurkError::IOError(format!("'{dosya_yolu}' okunamadı: {e}")))?;

    println!();
    println!("  \x1b[35m◈\x1b[0m \x1b[1mTÜRK\x1b[0m çalıştırılıyor: \x1b[36m{dosya_yolu}\x1b[0m");
    println!("  \x1b[90m{}\x1b[0m", "─".repeat(54));

    let mut dil = TurkDil::new();
    let mut satir_no = 0usize;

    for satir in icerik.lines() {
        satir_no += 1;
        let satir = satir.trim();

        // Boş satır ve yorum satırlarını atla
        if satir.is_empty() || satir.starts_with('#') || satir.starts_with("//") {
            continue;
        }

        info!("[{satir_no:>3}] {satir}");

        match dil.calistir_satir(satir, satir_no) {
            Ok(sonuc) => {
                // Anlamlı çıktı varsa göster
                match &sonuc {
                    Value::Number(n) => {
                        println!("  \x1b[32m→\x1b[0m {n}");
                    }
                    Value::Text(t) => {
                        println!("  \x1b[32m→\x1b[0m {t}");
                    }
                    Value::None => {} // sessiz
                    other => {
                        println!("  \x1b[32m→\x1b[0m {other:?}");
                    }
                }
            }
            Err(e) => {
                if let TurkError::SyntaxError(msg) = &e {
                    if let Some(col_part) = msg.split("sütun ").nth(1) {
                        if let Some(col_str) = col_part.split(':').next() {
                            if let Ok(col) = col_str.trim().parse::<usize>() {
                                eprintln!(
                                    "{}",
                                    hata_satirini_goster(satir, col, &msg.replace('\n', " "))
                                );
                            }
                        }
                    }
                }
                eprintln!("  \x1b[31m✗\x1b[0m Satır {satir_no}: {e}");
                return Err(e);
            }
        }
    }

    println!("  \x1b[90m{}\x1b[0m", "─".repeat(54));
    println!("  \x1b[32m✓\x1b[0m Tamamlandı — {satir_no} satır işlendi.\n");

    Ok(())
}

// ─── Statik Analiz (check) ─────────────────────────────────────────────────
fn dosyayi_kontrol_et(dosya_yolu: &str) -> Result<()> {
    let yol = Path::new(dosya_yolu);

    let icerik = fs::read_to_string(yol)
        .map_err(|e| TurkError::IOError(format!("'{dosya_yolu}' okunamadı: {e}")))?;

    println!();
    println!(
        "  \x1b[35m◈\x1b[0m \x1b[1mTÜRK CHECK\x1b[0m — statik analiz: \x1b[36m{dosya_yolu}\x1b[0m"
    );
    println!("  \x1b[90m{}\x1b[0m", "─".repeat(54));

    let mut dil = TurkDil::new();
    let mut satir_no = 0usize;
    let mut komut_sayisi = 0usize;

    for satir in icerik.lines() {
        satir_no += 1;
        let satir = satir.trim();
        if satir.is_empty() || satir.starts_with('#') || satir.starts_with("//") {
            continue;
        }

        komut_sayisi += 1;
        match dil.kontrol_et_satir(satir, satir_no) {
            Ok(()) => info!("[{satir_no:>3}] ✓ {satir}"),
            Err(e) => {
                eprintln!("  \x1b[31m✗ STATİK HATA\x1b[0m Satır {satir_no}: {e}");
                return Err(e);
            }
        }
    }

    println!("  \x1b[90m{}\x1b[0m", "─".repeat(54));
    println!("  \x1b[32m✓\x1b[0m Statik analiz geçti — {komut_sayisi} komut, {satir_no} satır.\n");

    Ok(())
}

fn model_json_onnx_aktar(json_yolu: &str, onnx_yolu: &str) -> Result<()> {
    println!();
    println!("  \x1b[35m◈\x1b[0m \x1b[1mTÜRK EXPORT\x1b[0m — {json_yolu} → {onnx_yolu}");
    let mut model = Sequential::new();
    model.load(json_yolu)?;
    export_onnx(&model, onnx_yolu)?;
    println!("  \x1b[32m✓\x1b[0m ONNX dışa aktarıldı: {onnx_yolu}\n");
    Ok(())
}

fn zoo_model_calistir(id: &str) -> Result<()> {
    println!();
    println!(
        "  \x1b[35m◈\x1b[0m \x1b[1mTÜRK ZOO\x1b[0m — model çalıştırılıyor: \x1b[36m{id}\x1b[0m"
    );
    println!("  \x1b[90m{}\x1b[0m", "─".repeat(54));

    let mut dil = TurkDil::new();
    dil.calistir(&format!("model_calistir(\"{id}\")"))?;

    println!("  \x1b[90m{}\x1b[0m", "─".repeat(54));
    println!("  \x1b[32m✓\x1b[0m Model Zoo tamamlandı.\n");
    Ok(())
}

// ─── REPL (Etkileşimli Mod) ────────────────────────────────────────────────
fn etkisesimli_mod() -> Result<()> {
    let mut dil = TurkDil::new();
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut giris = String::new();

    loop {
        print!("\x1b[35m  turk>\x1b[0m ");
        io::stdout().flush().ok();

        giris.clear();
        match reader.read_line(&mut giris) {
            Ok(0) => break, // EOF
            Err(e) => {
                eprintln!("  \x1b[31m✗\x1b[0m Okuma hatası: {e}");
                continue;
            }
            Ok(_) => {}
        }

        let komut = giris.trim();
        if komut.is_empty() || komut.starts_with('#') || komut.starts_with("//") {
            continue;
        }
        if matches!(komut, "çıkış" | "exit" | "quit" | "q") {
            println!("\n  \x1b[35m◈\x1b[0m TÜRK kapanıyor. Görüşürüz!\n");
            break;
        }

        // "check <dosya>" komutunu REPL içinden de destekle
        if let Some(dosya) = komut.strip_prefix("check ") {
            if let Err(e) = dosyayi_kontrol_et(dosya.trim()) {
                error!("{e}");
            }
            continue;
        }

        // "run <dosya>" komutunu REPL içinden de destekle
        if let Some(dosya) = komut.strip_prefix("run ") {
            if let Err(e) = dosyayi_calistir(dosya.trim()) {
                error!("{e}");
            }
            continue;
        }

        match dil.calistir(komut) {
            Ok(Value::None) => {}
            Ok(sonuc) => println!("  \x1b[32m→\x1b[0m {sonuc:?}"),
            Err(e) => eprintln!("  \x1b[31m✗\x1b[0m {e}"),
        }
    }

    Ok(())
}

// ─── Testler ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_olustur_komutu() {
        let mut dil = TurkDil::new();
        assert!(dil.calistir("model_olustur()").is_ok());
        assert!(dil.model.is_some());
    }

    #[test]
    fn test_dosya_bulunamadiysa_hata() {
        let sonuc = dosyayi_calistir("olmayan_dosya.turk");
        assert!(sonuc.is_err());
    }

    #[test]
    fn test_check_gecerli_script() {
        let dosya = "test_check_script.turk";
        std::fs::write(
            dosya,
            "model_olustur()\n\
             katman_ekle(yogun(2, 4, \"relu\"))\n\
             katman_ekle(yogun(4, 1, \"sigmoid\"))\n\
             derle(adam)\n",
        )
        .unwrap();
        assert!(dosyayi_kontrol_et(dosya).is_ok());
        std::fs::remove_file(dosya).ok();
    }

    #[test]
    fn test_check_boyut_uyumsuzlugu() {
        let dosya = "test_check_bad.turk";
        std::fs::write(
            dosya,
            "model_olustur()\n\
             katman_ekle(yogun(2, 4, \"relu\"))\n\
             katman_ekle(yogun(8, 1, \"sigmoid\"))\n",
        )
        .unwrap();
        assert!(dosyayi_kontrol_et(dosya).is_err());
        std::fs::remove_file(dosya).ok();
    }
}

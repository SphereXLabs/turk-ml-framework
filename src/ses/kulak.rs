//! Mikrofon girdisi — cpal ile arka plan kayıt döngüsü.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat};
use log::warn;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn kulak_akis_hatasi(err: cpal::StreamError) {
    warn!("Kulak akış hatası: {err}");
}

fn akisi_baslat(stream: cpal::Stream) -> Result<cpal::Stream, String> {
    stream
        .play()
        .map(|_| stream)
        .map_err(|e| format!("Akış oynatılamadı: {e}"))
}

#[derive(Debug, Clone)]
pub struct KayitSonucu {
    pub ornekler: Vec<f32>,
    pub ornekleme_hz: u32,
}

enum KulakKomut {
    KayitBaslat,
    KayitDurdur(Sender<KayitSonucu>),
}

/// Mikrofon dinleyicisi — kayıt thread'i üzerinde cpal akışı.
#[derive(Clone)]
pub struct Kulak {
    cmd_tx: Sender<KulakKomut>,
    pub ornekleme_hz: u32,
}

impl Kulak {
    /// Varsayılan giriş cihazında kulak thread'ini başlatır.
    pub fn ac() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("Mikrofon bulunamadı — giriş cihazı yok.")?;
        let config = device
            .default_input_config()
            .map_err(|e| format!("Mikrofon yapılandırması okunamadı: {e}"))?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let sample_format = config.sample_format();

        let (cmd_tx, cmd_rx) = mpsc::channel::<KulakKomut>();
        let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
        let kayit_aktif = Arc::new(AtomicBool::new(false));

        thread::Builder::new()
            .name("turk-kulak".into())
            .spawn(move || {
                kulak_dongusu(
                    device,
                    config.into(),
                    sample_format,
                    channels,
                    sample_rate,
                    cmd_rx,
                    buffer,
                    kayit_aktif,
                );
            })
            .map_err(|e| format!("Kulak thread başlatılamadı: {e}"))?;

        thread::sleep(Duration::from_millis(120));

        Ok(Self {
            cmd_tx,
            ornekleme_hz: sample_rate,
        })
    }

    pub fn kayit_baslat(&self) -> Result<(), String> {
        self.cmd_tx
            .send(KulakKomut::KayitBaslat)
            .map_err(|e| format!("Kayıt başlatılamadı: {e}"))
    }

    pub fn kayit_durdur(&self) -> Result<KayitSonucu, String> {
        let (tx, rx) = mpsc::channel();
        self.cmd_tx
            .send(KulakKomut::KayitDurdur(tx))
            .map_err(|e| format!("Kayıt durdurulamadı: {e}"))?;
        rx.recv_timeout(Duration::from_secs(15))
            .map_err(|e| format!("Kayıt yanıtı alınamadı: {e}"))
    }
}

fn kulak_dongusu(
    device: cpal::Device,
    config: cpal::StreamConfig,
    sample_format: SampleFormat,
    channels: usize,
    sample_rate: u32,
    cmd_rx: Receiver<KulakKomut>,
    buffer: Arc<Mutex<Vec<f32>>>,
    kayit_aktif: Arc<AtomicBool>,
) {
    let stream_olustur = |buf: Arc<Mutex<Vec<f32>>>, aktif: Arc<AtomicBool>| -> Result<cpal::Stream, String> {
        match sample_format {
            SampleFormat::F32 => device
                .build_input_stream(
                    &config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if !aktif.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut b = buf.lock().expect("kulak buffer");
                        if channels <= 1 {
                            b.extend_from_slice(data);
                        } else {
                            for frame in data.chunks(channels) {
                                let ort = frame.iter().copied().sum::<f32>() / channels as f32;
                                b.push(ort);
                            }
                        }
                    },
                    kulak_akis_hatasi,
                    None,
                )
                .map_err(|e| format!("F32 akışı açılamadı: {e}"))
                .and_then(akisi_baslat),
            SampleFormat::I16 => device
                .build_input_stream(
                    &config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        if !aktif.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut b = buf.lock().expect("kulak buffer");
                        if channels <= 1 {
                            for &s in data {
                                b.push(Sample::to_sample::<f32>(s));
                            }
                        } else {
                            for frame in data.chunks(channels) {
                                let ort: f32 = frame
                                    .iter()
                                    .map(|&s| Sample::to_sample::<f32>(s))
                                    .sum::<f32>()
                                    / channels as f32;
                                b.push(ort);
                            }
                        }
                    },
                    kulak_akis_hatasi,
                    None,
                )
                .map_err(|e| format!("I16 akışı açılamadı: {e}"))
                .and_then(akisi_baslat),
            other => Err(format!("Desteklenmeyen örnek formatı: {other:?}")),
        }
    };

    let mut stream = match stream_olustur(Arc::clone(&buffer), Arc::clone(&kayit_aktif)) {
        Ok(s) => Some(s),
        Err(e) => {
            warn!("Kulak akışı başlatılamadı: {e}");
            None
        }
    };

    loop {
        match cmd_rx.recv() {
            Ok(KulakKomut::KayitBaslat) => {
                buffer.lock().expect("kulak buffer").clear();
                kayit_aktif.store(true, Ordering::SeqCst);
                if stream.is_none() {
                    match stream_olustur(Arc::clone(&buffer), Arc::clone(&kayit_aktif)) {
                        Ok(s) => stream = Some(s),
                        Err(e) => warn!("Kulak akışı açılamadı: {e}"),
                    }
                }
            }
            Ok(KulakKomut::KayitDurdur(reply)) => {
                kayit_aktif.store(false, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(150));
                let ornekler = buffer.lock().expect("kulak buffer").clone();
                let _ = reply.send(KayitSonucu {
                    ornekler,
                    ornekleme_hz: sample_rate,
                });
            }
            Err(_) => break,
        }
    }
    drop(stream);
}

//! TÜRK duyusal katmanı — mikrofon yakalama ve Whisper STT.

pub mod fisilda;
pub mod kulak;

pub use fisilda::{konusmayi_metne_cevir, whisper_cli_yolu, whisper_hazir_mi, whisper_model_yolu};
pub use kulak::{KayitSonucu, Kulak};

/// GUI'ye ses işleme sonuçları.
#[derive(Debug, Clone)]
pub enum SesSinyal {
    Metin { metin: String },
    Hata(String),
}

//! Canlı dış dünya entegrasyonu — hava durumu ve web sorguları.

use serde::Deserialize;
use std::time::Duration;

const VARSAYILAN_SEHIR: &str = "İzmir";
const WEB_ISTEK_TIMEOUT: Duration = Duration::from_secs(3);

/// İnternet/API hatasında dönecek yerel mod fallback metni.
pub const HAVA_FALLBACK_MESAJ: &str = "\
[TÜRK]: Sinan, şu an dış dünyadaki hava durumu verilerine erişemedim kanka, \
yerel modda kalıp bilgisayarı korumaya devam ediyorum! 🏎️";

/// open-meteo üzerinden güncel hava durumu (senkron sarmalayıcı, 3 sn timeout).
pub fn canli_hava_durumu_cek(sehir: &str) -> Result<String, String> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("Tokio başlatılamadı: {e}"))?;

    rt.block_on(async {
        match tokio::time::timeout(WEB_ISTEK_TIMEOUT, canli_hava_durumu_cek_async(sehir)).await {
            Ok(Ok(yanit)) => Ok(yanit),
            Ok(Err(_)) | Err(_) => Ok(HAVA_FALLBACK_MESAJ.into()),
        }
    })
}

/// Asenkron HTTP ile hava durumu çeker.
pub async fn canli_hava_durumu_cek_async(sehir: &str) -> Result<String, String> {
    let sehir = sehir.trim();
    let sehir = if sehir.is_empty() {
        VARSAYILAN_SEHIR
    } else {
        sehir
    };

    let client = reqwest::Client::builder()
        .timeout(WEB_ISTEK_TIMEOUT)
        .user_agent("TURK-Sistem/2.4.0")
        .build()
        .map_err(|e| format!("HTTP istemcisi oluşturulamadı: {e}"))?;

    let (lat, lon, ad) = geocode(&client, sehir).await?;
    let sicaklik = hava_sicakligi(&client, lat, lon).await?;

    Ok(hava_yaniti_uret(&ad, sicaklik))
}

async fn geocode(client: &reqwest::Client, sehir: &str) -> Result<(f64, f64, String), String> {
    #[derive(Deserialize)]
    struct GeoYanit {
        results: Option<Vec<GeoSonuc>>,
    }
    #[derive(Deserialize)]
    struct GeoSonuc {
        name: String,
        latitude: f64,
        longitude: f64,
    }

    let yanit = client
        .get("https://geocoding-api.open-meteo.com/v1/search")
        .query(&[("name", sehir), ("count", "1"), ("language", "tr")])
        .send()
        .await
        .map_err(|e| format!("Konum sorgusu başarısız: {e}"))?;

    if !yanit.status().is_success() {
        return Err(format!("Konum API hatası: HTTP {}", yanit.status()));
    }

    let veri: GeoYanit = yanit
        .json()
        .await
        .map_err(|e| format!("Konum yanıtı okunamadı: {e}"))?;

    let sonuc = veri
        .results
        .and_then(|mut v| v.pop())
        .ok_or_else(|| format!("'{sehir}' için konum bulunamadı."))?;

    Ok((sonuc.latitude, sonuc.longitude, sonuc.name))
}

async fn hava_sicakligi(client: &reqwest::Client, lat: f64, lon: f64) -> Result<f32, String> {
    #[derive(Deserialize)]
    struct HavaYanit {
        current: Option<HavaAnlik>,
    }
    #[derive(Deserialize)]
    struct HavaAnlik {
        temperature_2m: f32,
    }

    let yanit = client
        .get("https://api.open-meteo.com/v1/forecast")
        .query(&[
            ("latitude", lat.to_string()),
            ("longitude", lon.to_string()),
            ("current", "temperature_2m".into()),
            ("timezone", "auto".into()),
        ])
        .send()
        .await
        .map_err(|e| format!("Hava sorgusu başarısız: {e}"))?;

    if !yanit.status().is_success() {
        return Err(format!("Hava API hatası: HTTP {}", yanit.status()));
    }

    let veri: HavaYanit = yanit
        .json()
        .await
        .map_err(|e| format!("Hava yanıtı okunamadı: {e}"))?;

    veri.current
        .map(|c| c.temperature_2m)
        .ok_or_else(|| "Sıcaklık verisi alınamadı.".into())
}

fn hava_yaniti_uret(sehir: &str, sicaklik: f32) -> String {
    let derece = sicaklik.round() as i32;
    format!(
        "[TÜRK]: {sehir}'de şu an hava güncel olarak {derece} derece kanka. \
         Karşıyaka sahili esiyordur ama Selçuk'ta gökyüzü tam skydiving yapmalık, pürüzsüz! 🏎️🌤️"
    )
}

/// Doğal dil girdisinden şehir adı ayıklar.
pub fn sehir_ayikla(metin: &str) -> String {
    let k = metin.to_lowercase();
    let sehirler = [
        ("izmir", "İzmir"),
        ("istanbul", "İstanbul"),
        ("ankara", "Ankara"),
        ("antalya", "Antalya"),
        ("bursa", "Bursa"),
        ("adana", "Adana"),
        ("konya", "Konya"),
        ("gaziantep", "Gaziantep"),
        ("trabzon", "Trabzon"),
        ("eskişehir", "Eskişehir"),
        ("eskisehir", "Eskişehir"),
    ];
    for (anahtar, ad) in sehirler {
        if k.contains(anahtar) {
            return ad.to_string();
        }
    }
    VARSAYILAN_SEHIR.to_string()
}

/// Girdinin hava durumu sorgusu olup olmadığını kontrol eder.
pub fn hava_sorgusu_mu(metin: &str) -> bool {
    let k = metin.to_lowercase();
    k.contains("hava durumu")
        || k.contains("hava nasıl")
        || k.contains("hava nasil")
        || k.contains("kaç derece")
        || k.contains("kac derece")
        || (k.contains("hava") && (k.contains("kaç") || k.contains("kac")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sehir_ayikla() {
        assert_eq!(sehir_ayikla("izmirde hava durumu kaç"), "İzmir");
        assert_eq!(sehir_ayikla("hava nasıl"), "İzmir");
    }

    #[test]
    fn test_hava_sorgusu() {
        assert!(hava_sorgusu_mu("izmirde hava durumu kaç"));
        assert!(!hava_sorgusu_mu("merhaba"));
    }

    #[test]
    fn test_hava_fallback_sabit() {
        assert!(HAVA_FALLBACK_MESAJ.contains("yerel modda"));
    }
}

//! Lazy CSV veri akışı — yalnızca batch boyutu kadar satır bellekte tutulur.
//!
//! Satır ofset indeksi ile epoch başına karıştırma (shuffle) disk-dostu yapılır.
//! `bolme` parametresi ile doğrulama kümesi ayrılır (val set bellekte tutulur).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use log::warn;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::error::{Result, TurkError};

/// Her veri satırının dosyadaki byte ofseti.
#[derive(Debug, Clone)]
pub struct CsvSatirIndeksi {
    pub satir_ofsetleri: Vec<u64>,
    pub toplam_satir: usize,
}

/// CSV dosyasının şeması (başlık analizi sonrası sabit metadata).
#[derive(Debug, Clone)]
pub struct CsvSemasi {
    pub kaynak: String,
    pub hedef_idx: usize,
    pub n_sutun: usize,
    pub ozellik_sayisi: usize,
    pub baslik_var: bool,
    pub indeks: CsvSatirIndeksi,
    /// Eğitimde kullanılacak satır indeksleri (0..toplam_satir aralığından).
    pub egitim_satirlari: Vec<usize>,
}

/// Diskten batch batch okuyan veri kaynağı.
#[derive(Debug, Clone)]
pub struct CsvVeriAkisi {
    sema: CsvSemasi,
    /// Doğrulama kümesi (bir kez yüklenir, epoch boyunca bellekte kalır).
    val_onbellek: Option<(Array2<f64>, Array2<f64>)>,
    pub bolme_orani: Option<f64>,
    /// Bellek içi eğitim verisi (WASM / sentetik). Set ise disk okunmaz.
    egitim_onbellek: Option<(Array2<f64>, Array2<f64>)>,
}

impl CsvVeriAkisi {
    /// Dosyayı açar, satır indeksini oluşturur; tüm satırlar RAM'e alınmaz.
    pub fn yeni(dosya_yolu: &str, hedef_kolon: Option<&str>) -> Result<Self> {
        Self::yeni_bolme(dosya_yolu, hedef_kolon, None)
    }

    /// `bolme_orani` verilirse (örn. 0.2) verinin %20'si doğrulama kümesine ayrılır.
    pub fn yeni_bolme(
        dosya_yolu: &str,
        hedef_kolon: Option<&str>,
        bolme_orani: Option<f64>,
    ) -> Result<Self> {
        Self::yeni_bolme_with_progress(dosya_yolu, hedef_kolon, bolme_orani, None::<fn(usize)>)
    }

    /// İndeksleme sırasında her N satırda bir `on_satir(satir_sayisi)` çağrılır (GUI ilerlemesi).
    pub fn yeni_bolme_with_progress<F>(
        dosya_yolu: &str,
        hedef_kolon: Option<&str>,
        bolme_orani: Option<f64>,
        mut on_satir: Option<F>,
    ) -> Result<Self>
    where
        F: FnMut(usize),
    {
        if let Some(orani) = bolme_orani {
            if !(0.0..1.0).contains(&orani) {
                return Err(TurkError::ValueError(format!(
                    "bolme oranı 0 ile 1 arasında olmalı, gelen: {orani}"
                )));
            }
        }

        let (mut sema, indeks) = csv_semasi_ve_indeks(dosya_yolu, hedef_kolon, &mut on_satir)?;

        let (egitim_satirlari, val_satirlari) = satir_bol(&indeks, bolme_orani);

        sema.indeks = indeks;
        sema.egitim_satirlari = egitim_satirlari.clone();

        let val_onbellek = if val_satirlari.is_empty() {
            None
        } else {
            if let Some(cb) = &mut on_satir {
                cb(0); // doğrulama kümesi okunuyor sinyali
            }
            Some(satirlari_oku(&sema, &val_satirlari)?)
        };

        Ok(Self {
            sema,
            val_onbellek,
            bolme_orani,
            egitim_onbellek: None,
        })
    }

    /// Bellek içi (disk kullanmayan) veri akışı — WASM / sentetik XOR vb.
    pub fn bellekten(
        kaynak: &str,
        x: Array2<f64>,
        y: Array2<f64>,
        bolme_orani: Option<f64>,
    ) -> Result<Self> {
        if x.nrows() == 0 {
            return Err(TurkError::DataError("Bellek verisi boş olamaz".into()));
        }
        if x.nrows() != y.nrows() {
            return Err(TurkError::DataError(format!(
                "X/Y satır sayısı uyumsuz: {} vs {}",
                x.nrows(),
                y.nrows()
            )));
        }
        if let Some(orani) = bolme_orani {
            if !(0.0..1.0).contains(&orani) {
                return Err(TurkError::ValueError(format!(
                    "bolme oranı 0 ile 1 arasında olmalı, gelen: {orani}"
                )));
            }
        }

        let n = x.nrows();
        let ozellik_sayisi = x.ncols();
        let mut egitim_idx: Vec<usize> = (0..n).collect();
        let val_idx = if let Some(oran) = bolme_orani {
            let val_n = ((n as f64) * oran).round() as usize;
            if val_n == 0 || val_n >= n {
                Vec::new()
            } else {
                egitim_idx.shuffle(&mut thread_rng());
                let val = egitim_idx.split_off(n - val_n);
                val
            }
        } else {
            Vec::new()
        };

        let mut x_train = Array2::<f64>::zeros((egitim_idx.len(), ozellik_sayisi));
        let mut y_train = Array2::<f64>::zeros((egitim_idx.len(), y.ncols()));
        for (i, &src) in egitim_idx.iter().enumerate() {
            x_train.row_mut(i).assign(&x.row(src));
            y_train.row_mut(i).assign(&y.row(src));
        }

        let val_onbellek = if val_idx.is_empty() {
            None
        } else {
            let mut x_val = Array2::<f64>::zeros((val_idx.len(), ozellik_sayisi));
            let mut y_val = Array2::<f64>::zeros((val_idx.len(), y.ncols()));
            for (i, &src) in val_idx.iter().enumerate() {
                x_val.row_mut(i).assign(&x.row(src));
                y_val.row_mut(i).assign(&y.row(src));
            }
            Some((x_val, y_val))
        };

        let sema = CsvSemasi {
            kaynak: kaynak.to_string(),
            hedef_idx: ozellik_sayisi, // sentetik: hedef ayrı matriste
            n_sutun: ozellik_sayisi + y.ncols(),
            ozellik_sayisi,
            baslik_var: false,
            indeks: CsvSatirIndeksi {
                satir_ofsetleri: vec![0; n],
                toplam_satir: n,
            },
            egitim_satirlari: (0..egitim_idx.len()).collect(),
        };

        Ok(Self {
            sema,
            val_onbellek,
            bolme_orani,
            egitim_onbellek: Some((x_train, y_train)),
        })
    }

    /// Klasik XOR veri seti (4 örnek, 2 özellik).
    pub fn sentetik_xor(bolme_orani: Option<f64>) -> Result<Self> {
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .map_err(|e| TurkError::DataError(format!("XOR X oluşturulamadı: {e}")))?;
        let y = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0])
            .map_err(|e| TurkError::DataError(format!("XOR Y oluşturulamadı: {e}")))?;
        Self::bellekten("sentetik:xor", x, y, bolme_orani)
    }

    pub fn kaynak(&self) -> &str {
        &self.sema.kaynak
    }

    pub fn ozellik_sayisi(&self) -> usize {
        self.sema.ozellik_sayisi
    }

    pub fn sema(&self) -> &CsvSemasi {
        &self.sema
    }

    pub fn egitim_satir_sayisi(&self) -> usize {
        self.sema.egitim_satirlari.len()
    }

    pub fn val_satir_sayisi(&self) -> usize {
        self.val_onbellek
            .as_ref()
            .map(|(x, _)| x.nrows())
            .unwrap_or(0)
    }

    /// Doğrulama tensörlerine erişim (varsa).
    pub fn val_veri(&self) -> Option<(&Array2<f64>, &Array2<f64>)> {
        self.val_onbellek.as_ref().map(|(x, y)| (x, y))
    }

    /// Bir epoch için batch iterator'ü döndürür (her çağrıda karıştırılmış okuma).
    pub fn epoch_batchleri(&self, batch_boyutu: usize) -> Result<CsvEpochOkuyucu> {
        let efektif = batch_boyutu.max(1);
        if efektif > self.sema.egitim_satirlari.len() {
            log::debug!(
                "batch_boyutu ({efektif}) egitim satir sayisindan ({}) buyuk — \
                 tek batch olarak okunacak",
                self.sema.egitim_satirlari.len()
            );
        }
        if let Some((x, y)) = &self.egitim_onbellek {
            CsvEpochOkuyucu::bellekten(x, y, efektif, true)
        } else {
            CsvEpochOkuyucu::ac(&self.sema, efektif, true)
        }
    }
}

/// Tek epoch boyunca diskten veya bellekten batch üreten iterator.
pub struct CsvEpochOkuyucu {
    file: Option<File>,
    sema: CsvSemasi,
    batch_boyutu: usize,
    okuma_sirasi: Vec<usize>,
    okuma_i: usize,
    bitti: bool,
    /// Bellek içi eğitim matrisleri (WASM / sentetik)
    bellek: Option<(Array2<f64>, Array2<f64>)>,
}

impl CsvEpochOkuyucu {
    pub fn ac(sema: &CsvSemasi, batch_boyutu: usize, karistir: bool) -> Result<Self> {
        if batch_boyutu == 0 {
            return Err(TurkError::ValueError("Batch boyutu sıfır olamaz.".into()));
        }
        if sema.egitim_satirlari.is_empty() {
            return Err(TurkError::DataError(
                "Eğitim satırı bulunamadı — CSV boş veya bolme=1.0 olabilir.".into(),
            ));
        }

        let mut okuma_sirasi = sema.egitim_satirlari.clone();
        if karistir {
            okuma_sirasi.shuffle(&mut thread_rng());
        }

        let file = File::open(&sema.kaynak)
            .map_err(|e| TurkError::IOError(format!("'{}' açılamadı: {e}", sema.kaynak)))?;

        Ok(Self {
            file: Some(file),
            sema: sema.clone(),
            batch_boyutu,
            okuma_sirasi,
            okuma_i: 0,
            bitti: false,
            bellek: None,
        })
    }

    /// Bellek içi matrislerden batch üretir (disk yok).
    pub fn bellekten(
        x: &Array2<f64>,
        y: &Array2<f64>,
        batch_boyutu: usize,
        karistir: bool,
    ) -> Result<Self> {
        if batch_boyutu == 0 {
            return Err(TurkError::ValueError("Batch boyutu sıfır olamaz.".into()));
        }
        if x.nrows() == 0 {
            return Err(TurkError::DataError("Eğitim satırı bulunamadı.".into()));
        }

        let mut okuma_sirasi: Vec<usize> = (0..x.nrows()).collect();
        if karistir {
            okuma_sirasi.shuffle(&mut thread_rng());
        }

        Ok(Self {
            file: None,
            sema: CsvSemasi {
                kaynak: "bellek".into(),
                hedef_idx: x.ncols(),
                n_sutun: x.ncols() + y.ncols(),
                ozellik_sayisi: x.ncols(),
                baslik_var: false,
                indeks: CsvSatirIndeksi {
                    satir_ofsetleri: vec![],
                    toplam_satir: x.nrows(),
                },
                egitim_satirlari: okuma_sirasi.clone(),
            },
            batch_boyutu,
            okuma_sirasi,
            okuma_i: 0,
            bitti: false,
            bellek: Some((x.clone(), y.clone())),
        })
    }

    fn satir_offsetten_oku(&mut self, satir_idx: usize) -> Result<String> {
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| TurkError::IOError("Disk dosyası yok (bellek modu)".into()))?;
        let offset = self.sema.indeks.satir_ofsetleri[satir_idx];
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| TurkError::IOError(format!("Seek hatası: {e}")))?;
        let mut satir = String::new();
        dosyadan_satir_oku(file, &mut satir)?;
        Ok(satir)
    }

    fn satir_isle(&self, ham: &str) -> Result<(Vec<f64>, f64)> {
        let degerler: Vec<f64> = ham
            .split(',')
            .map(|s| {
                s.trim()
                    .trim_matches('"')
                    .parse::<f64>()
                    .map_err(|_| TurkError::DataError(format!("Sayısal olmayan hücre: '{s}'")))
            })
            .collect::<Result<Vec<_>>>()?;

        if degerler.len() != self.sema.n_sutun {
            return Err(TurkError::DataError(format!(
                "Sütun sayısı uyumsuz: beklenen {}, gelen {}",
                self.sema.n_sutun,
                degerler.len()
            )));
        }

        let hedef = degerler[self.sema.hedef_idx];
        let ozellikler: Vec<f64> = degerler
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i != self.sema.hedef_idx)
            .map(|(_, v)| v)
            .collect();

        Ok((ozellikler, hedef))
    }

    fn matrise_cevir(
        &self,
        x_satirlar: Vec<Vec<f64>>,
        y_degerler: Vec<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let n = x_satirlar.len();
        let d = self.sema.ozellik_sayisi;
        let mut x = Array2::<f64>::zeros((n, d));
        let mut y = Array2::<f64>::zeros((n, 1));

        for (i, (satir, etiket)) in x_satirlar.iter().zip(y_degerler.iter()).enumerate() {
            for (j, &v) in satir.iter().enumerate() {
                x[[i, j]] = v;
            }
            y[[i, 0]] = *etiket;
        }

        Ok((x, y))
    }

    fn sonraki_batch(&mut self) -> Result<Option<(Array2<f64>, Array2<f64>)>> {
        if self.okuma_i >= self.okuma_sirasi.len() {
            return Ok(None);
        }

        let bitis = (self.okuma_i + self.batch_boyutu).min(self.okuma_sirasi.len());
        let chunk: Vec<usize> = self.okuma_sirasi[self.okuma_i..bitis].to_vec();
        self.okuma_i = bitis;

        if let Some((x_all, y_all)) = &self.bellek {
            let n = chunk.len();
            let d = x_all.ncols();
            let c = y_all.ncols();
            let mut x = Array2::<f64>::zeros((n, d));
            let mut y = Array2::<f64>::zeros((n, c));
            for (i, &src) in chunk.iter().enumerate() {
                x.row_mut(i).assign(&x_all.row(src));
                y.row_mut(i).assign(&y_all.row(src));
            }
            return Ok(Some((x, y)));
        }

        let mut x_satirlar = Vec::with_capacity(chunk.len());
        let mut y_degerler = Vec::with_capacity(chunk.len());

        for satir_idx in chunk {
            let ham = self.satir_offsetten_oku(satir_idx)?;
            let satir = ham.trim();
            if satir.is_empty() {
                continue;
            }
            let (ozellikler, hedef) = self.satir_isle(satir)?;
            x_satirlar.push(ozellikler);
            y_degerler.push(hedef);
        }

        if x_satirlar.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.matrise_cevir(x_satirlar, y_degerler)?))
        }
    }
}

impl Iterator for CsvEpochOkuyucu {
    type Item = Result<(Array2<f64>, Array2<f64>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bitti {
            return None;
        }

        match self.sonraki_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => {
                self.bitti = true;
                None
            }
            Err(e) => {
                self.bitti = true;
                Some(Err(e))
            }
        }
    }
}

/// CSV başlığını okuyup şema metadata'sını döndürür.
pub fn csv_semasi_oku(dosya_yolu: &str, hedef_kolon: Option<&str>) -> Result<CsvSemasi> {
    let (sema, _) = csv_semasi_ve_indeks(dosya_yolu, hedef_kolon, &mut None::<fn(usize)>)?;
    Ok(sema)
}

fn csv_semasi_ve_indeks<F>(
    dosya_yolu: &str,
    hedef_kolon: Option<&str>,
    on_satir: &mut Option<F>,
) -> Result<(CsvSemasi, CsvSatirIndeksi)>
where
    F: FnMut(usize),
{
    let mut file = File::open(dosya_yolu)
        .map_err(|e| TurkError::IOError(format!("'{dosya_yolu}' açılamadı: {e}")))?;

    let mut ilk_satir = String::new();
    dosyadan_satir_oku(&mut file, &mut ilk_satir)
        .map_err(|e| TurkError::IOError(format!("Okuma hatası: {e}")))?;

    if ilk_satir.trim().is_empty() {
        return Err(TurkError::DataError(format!("'{dosya_yolu}' boş dosya")));
    }

    let basliklar: Vec<String> = ilk_satir
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect();

    let n_sutun = basliklar.len();
    if n_sutun < 2 {
        return Err(TurkError::DataError(format!(
            "'{dosya_yolu}' en az 2 sütun içermeli (özellik + hedef)."
        )));
    }

    let baslik_var = basliklar[0].parse::<f64>().is_err();
    let hedef_idx = if baslik_var {
        if let Some(ad) = hedef_kolon {
            basliklar
                .iter()
                .position(|b| b.eq_ignore_ascii_case(ad))
                .unwrap_or_else(|| {
                    warn!("Hedef kolon '{ad}' bulunamadı, son sütun kullanılıyor.");
                    n_sutun - 1
                })
        } else {
            n_sutun - 1
        }
    } else {
        // Başlık yok — ilk satır veri; ofset listesine geri ekle
        n_sutun - 1
    };

    let mut satir_ofsetleri = Vec::new();

    if !baslik_var {
        // İlk satır veri satırı — ofset 0
        satir_ofsetleri.push(0);
    }

    loop {
        let offset = file
            .stream_position()
            .map_err(|e| TurkError::IOError(format!("Ofset okunamadı: {e}")))?;
        let mut satir = String::new();
        let n = dosyadan_satir_oku(&mut file, &mut satir)
            .map_err(|e| TurkError::IOError(format!("Satır okunamadı: {e}")))?;
        if n == 0 {
            break;
        }
        if satir.trim().is_empty() {
            continue;
        }
        satir_ofsetleri.push(offset);
        if let Some(cb) = on_satir {
            let n = satir_ofsetleri.len();
            if n == 1 || n % 250 == 0 {
                cb(n);
            }
        }
    }

    if let Some(cb) = on_satir {
        cb(satir_ofsetleri.len());
    }

    if satir_ofsetleri.is_empty() {
        return Err(TurkError::DataError(format!(
            "'{dosya_yolu}' veri satırı içermiyor."
        )));
    }

    let indeks = CsvSatirIndeksi {
        toplam_satir: satir_ofsetleri.len(),
        satir_ofsetleri,
    };

    let sema = CsvSemasi {
        kaynak: dosya_yolu.to_string(),
        hedef_idx,
        n_sutun,
        ozellik_sayisi: n_sutun - 1,
        baslik_var,
        indeks: indeks.clone(),
        egitim_satirlari: (0..indeks.toplam_satir).collect(),
    };

    Ok((sema, indeks))
}

fn satir_bol(indeks: &CsvSatirIndeksi, bolme_orani: Option<f64>) -> (Vec<usize>, Vec<usize>) {
    let n = indeks.toplam_satir;
    let mut sirali: Vec<usize> = (0..n).collect();
    sirali.shuffle(&mut thread_rng());

    let bolme = bolme_orani.unwrap_or(0.0);
    if bolme <= 0.0 || n < 2 {
        return (sirali, Vec::new());
    }

    let val_sayisi = ((n as f64) * bolme).round() as usize;
    let val_sayisi = val_sayisi.clamp(1, n - 1);
    let egitim_sayisi = n - val_sayisi;

    let egitim = sirali[..egitim_sayisi].to_vec();
    let val = sirali[egitim_sayisi..].to_vec();
    (egitim, val)
}

fn satirlari_oku(
    sema: &CsvSemasi,
    satir_indeksleri: &[usize],
) -> Result<(Array2<f64>, Array2<f64>)> {
    let mut file = File::open(&sema.kaynak)
        .map_err(|e| TurkError::IOError(format!("'{}' açılamadı: {e}", sema.kaynak)))?;

    let mut x_satirlar = Vec::with_capacity(satir_indeksleri.len());
    let mut y_degerler = Vec::with_capacity(satir_indeksleri.len());

    for &idx in satir_indeksleri {
        let offset = sema.indeks.satir_ofsetleri[idx];
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| TurkError::IOError(format!("Seek hatası: {e}")))?;
        let mut ham = String::new();
        dosyadan_satir_oku(&mut file, &mut ham)
            .map_err(|e| TurkError::IOError(format!("Satır okunamadı: {e}")))?;

        let degerler: Vec<f64> = ham
            .trim()
            .split(',')
            .map(|s| {
                s.trim()
                    .trim_matches('"')
                    .parse::<f64>()
                    .map_err(|_| TurkError::DataError(format!("Sayısal olmayan hücre: '{s}'")))
            })
            .collect::<Result<Vec<_>>>()?;

        let hedef = degerler[sema.hedef_idx];
        let ozellikler: Vec<f64> = degerler
            .into_iter()
            .enumerate()
            .filter(|(i, _)| *i != sema.hedef_idx)
            .map(|(_, v)| v)
            .collect();

        x_satirlar.push(ozellikler);
        y_degerler.push(hedef);
    }

    let n = x_satirlar.len();
    let d = sema.ozellik_sayisi;
    let mut x = Array2::<f64>::zeros((n, d));
    let mut y = Array2::<f64>::zeros((n, 1));
    for (i, (satir, etiket)) in x_satirlar.iter().zip(y_degerler.iter()).enumerate() {
        for (j, &v) in satir.iter().enumerate() {
            x[[i, j]] = v;
        }
        y[[i, 0]] = *etiket;
    }

    Ok((x, y))
}

/// Dosyadan tek satır okur (newline hariç).
fn dosyadan_satir_oku(file: &mut File, buf: &mut String) -> Result<usize> {
    buf.clear();
    let mut byte = [0u8; 1];
    let mut count = 0usize;
    loop {
        match file.read(&mut byte) {
            Ok(0) => break,
            Ok(_) => {
                count += 1;
                if byte[0] == b'\n' {
                    break;
                }
                buf.push(byte[0] as char);
            }
            Err(e) => {
                return Err(TurkError::IOError(format!("Okuma hatası: {e}")));
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_semasi_oku() {
        let dosya = "test_csv_sema.csv";
        std::fs::write(dosya, "x1,x2,label\n0,0,0\n0,1,1\n").unwrap();

        let sema = csv_semasi_oku(dosya, Some("label")).unwrap();
        assert_eq!(sema.ozellik_sayisi, 2);
        assert!(sema.baslik_var);
        assert_eq!(sema.indeks.toplam_satir, 2);

        std::fs::remove_file(dosya).ok();
    }

    #[test]
    fn test_epoch_batchleri_lazy() {
        let dosya = "test_csv_lazy.csv";
        std::fs::write(dosya, "x1,x2,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n").unwrap();

        let akis = CsvVeriAkisi::yeni(dosya, Some("y")).unwrap();
        let batches: Vec<_> = akis
            .epoch_batchleri(2)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].0.nrows(), 2);
        assert_eq!(batches[1].0.nrows(), 2);

        std::fs::remove_file(dosya).ok();
    }

    #[test]
    fn test_bolme_ve_val() {
        let dosya = "test_csv_bolme.csv";
        std::fs::write(dosya, "x1,x2,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n").unwrap();

        let akis = CsvVeriAkisi::yeni_bolme(dosya, Some("y"), Some(0.25)).unwrap();
        assert_eq!(akis.egitim_satir_sayisi(), 3);
        assert_eq!(akis.val_satir_sayisi(), 1);
        assert!(akis.val_veri().is_some());

        std::fs::remove_file(dosya).ok();
    }

    #[test]
    fn test_batch_buyuk_egitim_kucuk() {
        let dosya = "test_csv_batch_buyuk.csv";
        std::fs::write(dosya, "x1,x2,y\n0,0,0\n0,1,1\n1,0,1\n").unwrap();

        let akis = CsvVeriAkisi::yeni_bolme(dosya, Some("y"), Some(0.0)).unwrap();
        assert_eq!(akis.egitim_satir_sayisi(), 3);

        // batch=4 > 3 satir: tek batch, ikinci epoch'ta yine calismali
        for _ in 0..2 {
            let batches: Vec<_> = akis
                .epoch_batchleri(4)
                .unwrap()
                .collect::<Result<Vec<_>>>()
                .unwrap();
            assert_eq!(batches.len(), 1);
            assert_eq!(batches[0].0.nrows(), 3);
        }

        std::fs::remove_file(dosya).ok();
    }

    #[test]
    fn test_shuffle_farkli_epoch() {
        let dosya = "test_csv_shuffle.csv";
        std::fs::write(dosya, "x1,x2,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n5,5,0\n6,6,1\n").unwrap();

        let akis = CsvVeriAkisi::yeni(dosya, Some("y")).unwrap();
        let b1: Vec<_> = akis
            .epoch_batchleri(1)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        let b2: Vec<_> = akis
            .epoch_batchleri(1)
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(b1.len(), 6);
        assert_eq!(b2.len(), 6);
        // Karıştırma ile ilk batch sırası büyük olasılıkla farklı olur (6! olasılık)
        let farkli = b1
            .iter()
            .zip(b2.iter())
            .any(|((x1, _), (x2, _))| (x1[[0, 0]] - x2[[0, 0]]).abs() > 1e-9);
        assert!(farkli, "Epoch'lar arası shuffle çalışmıyor olabilir");

        std::fs::remove_file(dosya).ok();
    }
}

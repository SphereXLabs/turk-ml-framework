#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Operator {
    Toplama,
    Cikarma,
    Carpma,
    Bolme,
    ModOperator,
    MatrisCarpim,
    NoktaCarpim,
    Transpoze,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum KarsilastirmaOperator {
    EsitMi,
    EsitDegilMi,
    BuyukMu,
    KucukMu,
    BuyukEsitMi,
    KucukEsitMi,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum MantiksalOperator {
    Ve,
    Veya,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum IstatistikselIslemTipi {
    Ortalama,
    Medyan,
    Mod,
    StandartSapma,
    MinMax,
}

#[derive(Debug, Clone)]
pub enum AktivasyonFonksiyonu {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
}

#[derive(Debug, Clone)]
pub enum KayipFonksiyonu {
    MSE,
    CrossEntropy,
    BinaryCrossEntropy,
}

#[derive(Debug, Clone)]
pub enum OptimizerTipi {
    SGD(f64),  // Öğrenme oranı
    Adam(f64), // Öğrenme oranı
    AdamW {
        lr: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
    },
}

#[derive(Debug, Clone)]
pub struct KatmanOzellikleri {
    pub noron_sayisi: i64,
    pub aktivasyon: AktivasyonFonksiyonu,
    pub kernel_initializer: Option<String>,
    pub bias_initializer: Option<String>,
    pub dropout_rate: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ModelOzellikleri {
    pub katmanlar: Vec<KatmanOzellikleri>,
    pub optimizer: OptimizerTipi,
    pub kayip_fonksiyonu: KayipFonksiyonu,
    pub batch_boyutu: i64,
    pub epoch_sayisi: i64,
}

#[derive(Debug, Clone)]
pub enum Ifade {
    Sayi(i64),
    Metin(String),
    Mantiksal(bool),
    Degisken(String),
    Dizi(Vec<Ifade>),
    Liste(Vec<Ifade>),
    Parantez(Box<Ifade>),
    NoktaNotasyonu {
        nesne: Box<Ifade>,
        ozellik: String,
    },
    FonksiyonCagrisi {
        fonksiyon: Box<Ifade>,
        argumanlar: Vec<Ifade>,
    },
    IkiliIslem {
        sol: Box<Ifade>,
        operator: Operator,
        sag: Box<Ifade>,
    },
    KarsilastirmaIfadesi {
        sol: Box<Ifade>,
        operator: KarsilastirmaOperator,
        sag: Box<Ifade>,
    },
    MantiksalIfade {
        sol: Box<Ifade>,
        operator: MantiksalOperator,
        sag: Box<Ifade>,
    },
    IstatistikselIslem {
        islem_tipi: IstatistikselIslemTipi,
        veri: Box<Ifade>,
    },
    DiziErisim {
        dizi: Box<Ifade>,
        indeks: Box<Ifade>,
    },
    MatrisOlustur { satirlar: Vec<Vec<Ifade>> },
    DiziOlusturucu { baslangic: Box<Ifade>, bitis: Box<Ifade>, artis: Option<Box<Ifade>> },
    ModelOlustur {
        ozellikler: ModelOzellikleri,
    },
    MatrisIslem {
        sol: Box<Ifade>,
        operator: MatrisOperator,
        sag: Box<Ifade>,
    },
}

#[derive(Debug, Clone)]
pub enum Deyim {
    DegiskenTanimi { isim: String, deger: Ifade },
    IfadeDeyimi(Ifade),
    EgerDeyimi { kosul: Ifade, govde: Box<Vec<Deyim>>, degilse_govde: Option<Box<Vec<Deyim>>> },
    DonguDeyimi { kosul: Ifade, govde: Box<Vec<Deyim>> },
    IcinDongusu { degisken: String, dizi: Ifade, govde: Box<Vec<Deyim>> },
    KirDeyimi,
    DevamDeyimi,
    DondurDeyimi(Ifade),
    FonksiyonTanimi { isim: String, parametreler: Vec<String>, govde: Box<Vec<Deyim>> },
    Blok(Vec<Deyim>),
    ModelTanimi { isim: String, ozellikler: ModelOzellikleri },
    ModelEgitimi { model: String, egitim_verisi: Ifade, etiketler: Ifade },
    Tahmin { model: String, girdi: Ifade },
    Atama { isim: String, deger: Ifade },
    Ifade(Ifade),
    VeriTanimi { isim: String, deger: Box<Ifade> },
    Egitim { model: String, x_train: Box<Ifade>, y_train: Box<Ifade> },
}

#[derive(Debug, Clone)]
pub struct Program {
    pub deyimler: Vec<Deyim>,
}

#[derive(Debug, Clone)]
pub enum MatrisOperator {
    Carpim,     // @
    NoktaCarpim, // .
    Transpoze,   // ~
}

#[derive(Debug, Clone)]
pub struct YapayZekaModeli {
    pub ozellikler: ModelOzellikleri,
    pub egitilmis: bool,
}

use crate::{Result, TurkError};
use ndarray::Array2;

#[derive(Debug, Clone)]
pub enum DegerTipi {
    Sayi(f32),
    Metin(String),
    Mantiksal(bool),
    Dizi(Vec<DegerTipi>),
    Model(ModelOzellikleri),
    Birim,
}

impl DegerTipi {
    pub fn to_array(&self) -> Result<Array2<f32>> {
        match self {
            DegerTipi::Dizi(elemanlar) => {
                let mut data = Vec::new();
                for eleman in elemanlar {
                    match eleman {
                        DegerTipi::Sayi(x) => data.push(*x),
                        DegerTipi::Dizi(alt_elemanlar) => {
                            for alt_eleman in alt_elemanlar {
                                if let DegerTipi::Sayi(x) = alt_eleman {
                                    data.push(*x);
                                } else {
                                    return Err(TurkError::InvalidInput(
                                        "Dizi elemanları sayı olmalı".to_string()
                                    ));
                                }
                            }
                        }
                        _ => return Err(TurkError::InvalidInput(
                            "Dizi elemanları sayı olmalı".to_string()
                        )),
                    }
                }
                
                // Diziyi 2D matrise dönüştür
                let n_rows = elemanlar.len();
                let n_cols = if let DegerTipi::Dizi(first) = &elemanlar[0] {
                    first.len()
                } else {
                    1
                };
                
                Ok(Array2::from_shape_vec((n_rows, n_cols), data).map_err(|e| TurkError::NdarrayError(e))?)
            }
            _ => Err(TurkError::InvalidInput(
                "Sadece diziler Array2'ye dönüştürülebilir".to_string()
            )),
        }
    }
} 
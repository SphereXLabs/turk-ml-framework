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
    Ustlu,  // Üs alma operatörü (^)
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
    Varyans,
    Korelasyon,
}

#[derive(Debug, Clone)]
pub enum AktivasyonFonksiyonu {
    Relu,
    LeakyRelu(f64),  // alpha parametresi
    Sigmoid,
    Tanh,
    Softmax,
    Swish,
}

#[derive(Debug, Clone)]
pub enum KayipFonksiyonu {
    MSE,
    CrossEntropy,
    BinaryCrossEntropy,
    HuberLoss(f64),  // delta parametresi
    FocalLoss(f64),  // gamma parametresi
}

#[derive(Debug, Clone)]
pub enum OptimizerTipi {
    SGD {
        lr: f64,
        momentum: Option<f64>,
        nesterov: Option<bool>,
    },
    Adam {
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    },
    AdamW {
        lr: f64,
        weight_decay: f64,
        beta1: f64,
        beta2: f64,
    },
    RMSProp {
        lr: f64,
        rho: f64,
        epsilon: f64,
    },
}

#[derive(Debug, Clone)]
pub enum KernelInitializer {
    Xavier,
    HeNormal,
    Zeros,
    Ones,
    RandomNormal(f64), // standart sapma
}

#[derive(Debug, Clone)]
pub enum KatmanTipi {
    Yogun,
    Evrisimli {
        filtre_sayisi: i64,
        filtre_boyutu: (i64, i64),
        adim: (i64, i64),
        padding: String, // "same" veya "valid"
    },
    Havuzlama {
        boyut: (i64, i64),
        adim: (i64, i64),
        tip: String, // "max" veya "average"
    },
    DropOut(f64),
    BatchNormalization,
    RNN {
        gizli_boyut: i64,
        bidirectional: bool,
    },
    LSTM {
        gizli_boyut: i64,
        bidirectional: bool,
    },
    GRU {
        gizli_boyut: i64,
        bidirectional: bool,
    },
}

#[derive(Debug, Clone)]
pub struct KatmanOzellikleri {
    pub katman_tipi: KatmanTipi,
    pub noron_sayisi: i64,
    pub aktivasyon: AktivasyonFonksiyonu,
    pub kernel_initializer: Option<KernelInitializer>,
    pub bias_initializer: Option<KernelInitializer>,
    pub dropout_rate: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct ModelOzellikleri {
    pub katmanlar: Vec<KatmanOzellikleri>,
    pub optimizer: OptimizerTipi,
    pub kayip_fonksiyonu: KayipFonksiyonu,
    pub batch_boyutu: i64,
    pub epoch_sayisi: i64,
    pub metrikler: Option<Vec<String>>,
    pub erken_durdurma: Option<i64>, // sabır değeri (patience)
}

#[derive(Debug, Clone)]
pub enum Ifade {
    Sayi(i64),
    OndalikSayi(f64),  // Ondalık sayı desteği
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
    VeriYukleme {
        dosya_yolu: String,
        format: Option<String>,   // "csv", "json", "npz" gibi
    },
    VeriBolme {
        veri: Box<Ifade>,
        oran: f64,
        karistir: bool,
    },
    VeriOnIsleme {
        veri: Box<Ifade>,
        islem_tipi: String,  // "normallestir", "standartlastir", "one_hot" gibi
        parametreler: Option<Vec<Ifade>>,
    },
    Lambda {
        parametreler: Vec<String>,
        govde: Box<Ifade>,
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
    ModelKaydetme { model: String, dosya_yolu: String },
    ModelYukleme { isim: String, dosya_yolu: String },
    ModelDegerlendirme { model: String, test_verisi: Ifade, test_etiketleri: Ifade },
    ModelOzeti { model: String },
    VeriGorsellestirme { veri: Ifade, grafik_tipi: String, baslik: Option<String> },
    ImportDeyimi { modul: String, as_isim: Option<String> },
    TryCatch { try_govde: Box<Vec<Deyim>>, catch_degisken: String, catch_govde: Box<Vec<Deyim>> },
    LambdaTanimi { isim: String, lambda: Ifade },
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
    pub egitim_metrikleri: Option<Vec<(String, f64)>>,
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
    VeriKumesi {
        x: Array2<f32>,
        y: Option<Array2<f32>>,
        oznitelik_isimleri: Option<Vec<String>>,
        sinif_isimleri: Option<Vec<String>>,
    },
}

impl DegerTipi {
    pub fn to_array(&self) -> Result<Array2<f32>> {
        match self {
            DegerTipi::Dizi(elemanlar) => {
                // İlk boyutu bulalım (satır sayısı)
                let satir_sayisi = elemanlar.len();
                if satir_sayisi == 0 {
                    return Err(TurkError::TypeError("Boş dizi Array2'ye dönüştürülemez".to_string()));
                }
                
                // Düz bir dizi mi yoksa iki boyutlu bir dizi mi kontrol edelim
                let is_nested = elemanlar.iter().any(|e| matches!(e, DegerTipi::Dizi(_)));
                
                if is_nested {
                    // İç dizilerin boyutunu kontrol edelim (sütun sayısı)
                    let mut sutun_sayisi = 0;
                    let mut data = Vec::new();
                    
                    for eleman in elemanlar {
                        if let DegerTipi::Dizi(ic_dizi) = eleman {
                            if sutun_sayisi == 0 {
                                sutun_sayisi = ic_dizi.len();
                            } else if ic_dizi.len() != sutun_sayisi {
                                return Err(TurkError::ShapeMismatch(
                                    format!("İç dizilerin boyutları farklı: beklenen {}, alınan {}", 
                                            sutun_sayisi, ic_dizi.len())
                                ));
                            }
                            
                            // İç diziyi düzleştirelim
                            for ic_eleman in ic_dizi {
                                if let DegerTipi::Sayi(x) = ic_eleman {
                                    data.push(*x as f32);
                                } else {
                                    return Err(TurkError::TypeError(
                                        "İç dizi elemanları sayı olmalıdır".to_string()
                                    ));
                                }
                            }
                        } else {
                            return Err(TurkError::TypeError(
                                "Dizi içindeki elemanlar dizi olmalıdır".to_string()
                            ));
                        }
                    }
                    
                    // (satır, sütun) şeklinde iki boyutlu matris oluştur
                    let shape = (satir_sayisi, sutun_sayisi);
                    Array2::from_shape_vec(shape, data)
                        .map_err(|e| TurkError::ShapeMismatch(format!("Dizi şekli uyumsuzluğu: {}", e)))
                } else {
                    // Tek boyutlu diziyi ele alalım (vektör)
                    let mut data = Vec::new();
                    
                    for eleman in elemanlar {
                        if let DegerTipi::Sayi(x) = eleman {
                            data.push(*x as f32);
                        } else {
                            return Err(TurkError::TypeError(
                                "Dizi elemanları sayı olmalıdır".to_string()
                            ));
                        }
                    }
                    
                    // Tek örnek olarak şekillendir (1, n) formatında
                    let shape = (1, data.len());
                    Array2::from_shape_vec(shape, data)
                        .map_err(|e| TurkError::ShapeMismatch(format!("Dizi şekli uyumsuzluğu: {}", e)))
                }
            }
            DegerTipi::VeriKumesi { x, .. } => Ok(x.clone()),
            _ => Err(TurkError::TypeError("Bu değer Array2'ye dönüştürülemez".to_string())),
        }
    }
    
    pub fn tip_adi(&self) -> &'static str {
        match self {
            DegerTipi::Sayi(_) => "Sayi",
            DegerTipi::Metin(_) => "Metin",
            DegerTipi::Mantiksal(_) => "Mantiksal",
            DegerTipi::Dizi(_) => "Dizi",
            DegerTipi::Model(_) => "Model",
            DegerTipi::Birim => "Birim",
            DegerTipi::VeriKumesi { .. } => "VeriKumesi",
        }
    }
} 
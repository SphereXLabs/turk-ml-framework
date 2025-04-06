use logos::Logos;

#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
    // Anahtar kelimeler
    #[token("Veri")]
    Veri,
    #[token("Fonksiyon")]
    Fonksiyon,
    #[token("Dondur")]
    Dondur,
    #[token("Yazdir")]
    Yazdir,
    #[token("Eger")]
    Eger,
    #[token("Degilse")]
    Degilse,
    #[token("Dongu")]
    Dongu,
    #[token("Icin")]
    Icin,
    #[token("Icinde")]
    Icinde,
    #[token("Kir")]
    Kir,
    #[token("Devam")]
    Devam,
    #[token("Dogru")]
    Dogru,
    #[token("Yanlis")]
    Yanlis,

    // Yapay zeka anahtar kelimeleri
    #[token("Model")]
    Model,
    #[token("Egit")]
    Egit,
    #[token("Tahmin")]
    Tahmin,
    #[token("Katman")]
    Katman,
    #[token("Aktivasyon")]
    Aktivasyon,
    #[token("OgrenmeOrani")]
    OgrenmeOrani,
    #[token("Epoch")]
    Epoch,
    #[token("Batch")]
    Batch,
    #[token("Kayip")]
    Kayip,
    #[token("Dogruluk")]
    Dogruluk,
    #[token("Veri_Yukle")]
    VeriYukle,
    #[token("Veri_Bol")]
    VeriBol,
    #[token("Veri_Karistir")]
    VeriKaristir,
    #[token("Kaydet")]
    Kaydet,
    #[token("Yukle")]
    Yukle,
    #[token("Optimizer")]
    Optimizer,
    #[token("Dropout")]
    Dropout,
    #[token("BatchNorm")]
    BatchNorm,
    #[token("Degerlendir")]
    Degerlendir,
    #[token("Ozet")]
    Ozet,

    // Optimizer tipleri
    #[token("SGD")]
    SGD,
    #[token("Adam")]
    Adam,
    #[token("AdamW")]
    AdamW,
    #[token("RMSProp")]
    RMSProp,

    // Kayıp fonksiyonları
    #[token("MSE")]
    MSE,
    #[token("CrossEntropy")]
    CrossEntropy,
    #[token("BinaryCrossEntropy")]
    BinaryCrossEntropy,

    // Aktivasyon fonksiyonları
    #[token("Sigmoid")]
    Sigmoid,
    #[token("Tanh")]
    Tanh,
    #[token("Relu")]
    Relu,
    #[token("LeakyRelu")]
    LeakyRelu,
    #[token("Softmax")]
    Softmax,
    #[token("Swish")]
    Swish,

    // Kernel başlatıcılar
    #[token("Xavier")]
    Xavier,
    #[token("HeNormal")]
    HeNormal,
    #[token("RandomNormal")]
    RandomNormal,

    // Operatörler
    #[token("+")]
    Arti,
    #[token("-")]
    Eksi,
    #[token("*")]
    Carpma,
    #[token("/")]
    Bolme,
    #[token("%")]
    Modulo,
    #[token("ve")]
    Ve,
    #[token("veya")]
    Veya,
    #[token("degil")]
    Degil,
    #[token("==")]
    EsitMi,
    #[token("!=")]
    EsitDegilMi,
    #[token(">")]
    BuyukMu,
    #[token("<")]
    KucukMu,
    #[token(">=")]
    BuyukEsitMi,
    #[token("<=")]
    KucukEsitMi,
    #[token("=")]
    Esittir,

    // Özel karakterler
    #[token("@")]
    At,
    #[token(".", priority = 2)]
    Nokta,
    #[token("~")]
    Tilde,

    // Parantezler
    #[token("(")]
    AcParantez,
    #[token(")")]
    KapaParantez,
    #[token("[")]
    AcKoseli,
    #[token("]")]
    KapaKoseli,
    #[token("{")]
    AcKume,
    #[token("}")]
    KapaKume,
    #[token(",")]
    Virgul,
    #[token(";")]
    NoktaliVirgul,
    #[token(":")]
    IkiNokta,

    // Model mimarileri
    #[token("CNN")]
    CNN,
    #[token("RNN")]
    RNN,
    #[token("LSTM")]
    LSTM,
    #[token("GRU")]
    GRU,
    #[token("Transformer")]
    Transformer,
    #[token("Encoder")]
    Encoder, 
    #[token("Decoder")]
    Decoder,

    // İstatistik fonksiyonları
    #[token("Ortalama")]
    Ortalama,
    #[token("Medyan")]
    Medyan,
    #[token("Mod")]
    ModFonksiyon,
    #[token("StandartSapma")]
    StandartSapma,
    #[token("MinMax")]
    MinMax,
    #[token("Varyans")]
    Varyans,
    #[token("Korelasyon")]
    Korelasyon,

    // Veri işleme
    #[token("Normallestir")]
    Normallestir,
    #[token("OneHotKodla")]
    OneHotKodla,
    #[token("CokluSinif")]
    CokluSinif,
    #[token("MetinVektorlestir")]
    MetinVektorlestir,
    #[token("ResimYukle")]
    ResimYukle,
    
    // Değerler
    #[regex("[0-9]+", |lex| lex.slice().parse::<i64>().unwrap())]
    Sayi(i64),
    #[regex(r#""[^"]*""#, |lex| lex.slice()[1..lex.slice().len()-1].to_string())]
    Metin(String),
    #[regex("[a-zA-Z_ğüşıöçĞÜŞİÖÇ][a-zA-Z0-9_ğüşıöçĞÜŞİÖÇ]*", |lex| lex.slice().to_string())]
    Degisken(String),
    #[regex(r"[0-9]+,[0-9]+", |lex| {
        let parts: Vec<&str> = lex.slice().split(',').collect();
        if parts.len() == 2 {
            let tam_kisim = parts[0].parse::<i64>().unwrap_or(0);
            let ondalik_kisim = parts[1].parse::<i64>().unwrap_or(0);
            let ondalik_str = format!("0.{}", ondalik_kisim);
            let ondalik_deger = ondalik_str.parse::<f64>().unwrap_or(0.0);
            (tam_kisim as f64 + ondalik_deger) as i64
        } else {
            0
        }
    })]
    OndalikSayi(i64),

    // Hatalı/Bilinmeyen token
    Illegal,

    // Boşluk ve yorum satırları
    #[regex(r"[ \t\n\f]+", logos::skip)]
    #[regex(r"//[^\n]*", logos::skip)]
    Bosluk,
} 
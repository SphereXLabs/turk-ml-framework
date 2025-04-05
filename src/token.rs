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

    // Aktivasyon fonksiyonları
    #[token("Sigmoid")]
    Sigmoid,
    #[token("Tanh")]
    Tanh,
    #[token("Relu")]
    Relu,
    #[token("Softmax")]
    Softmax,

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

    // Değerler
    #[regex("[0-9]+", |lex| lex.slice().parse::<i64>().unwrap())]
    Sayi(i64),
    #[regex(r#""[^"]*""#, |lex| lex.slice()[1..lex.slice().len()-1].to_string())]
    Metin(String),
    #[regex("[a-zA-Z_ğüşıöçĞÜŞİÖÇ][a-zA-Z0-9_ğüşıöçĞÜŞİÖÇ]*", |lex| lex.slice().to_string())]
    Degisken(String),

    // Hatalı/Bilinmeyen token
    Illegal,

    // Boşluk ve yorum satırları
    #[regex(r"[ \t\n\f]+", logos::skip)]
    #[regex(r"//[^\n]*", logos::skip)]
    Bosluk,
} 
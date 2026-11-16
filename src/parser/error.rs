use crate::token::Token;
use crate::source_position::SourcePosition;
use thiserror::Error;
use std::fmt;

/// Parser hataları
#[derive(Error, Debug)]
pub enum ParseError {
    /// Sözdizimi hatası
    #[error("Sözdizimi hatası: {0}")]
    SyntaxError(String),
    
    /// Tip hatası
    #[error("Tip hatası: {0}")]
    TypeError(String),
    
    /// Beklenmeyen token
    #[error("Beklenmeyen token: {0}, beklenen: {1}")]
    UnexpectedToken(Token, String),
    
    /// Beklenmeyen dosya sonu
    #[error("Beklenmeyen dosya sonu")]
    UnexpectedEOF,
    
    /// Geçersiz değişken
    #[error("Geçersiz değişken: {0}")]
    InvalidVariable(String),
    
    /// Geçersiz fonksiyon
    #[error("Geçersiz fonksiyon: {0}")]
    InvalidFunction(String),
    
    /// Geçersiz parametre
    #[error("Geçersiz parametre: {0}")]
    InvalidParameter(String),
    
    /// Geçersiz model
    #[error("Geçersiz model: {0}")]
    InvalidModel(String),
    
    /// Geçersiz katman
    #[error("Geçersiz katman: {0}")]
    InvalidLayer(String),
    
    /// Geçersiz optimizer
    #[error("Geçersiz optimizer: {0}")]
    InvalidOptimizer(String),
    
    /// Geçersiz kayıp fonksiyonu
    #[error("Geçersiz kayıp fonksiyonu: {0}")]
    InvalidLossFunction(String),
    
    /// Geçersiz aktivasyon
    #[error("Geçersiz aktivasyon: {0}")]
    InvalidActivation(String),
    
    /// Geçersiz boyut
    #[error("Geçersiz boyut: {0}")]
    InvalidDimension(String),
    
    /// Geçersiz değer
    #[error("Geçersiz değer: {0}")]
    InvalidValue(String),
    
    /// Geçersiz işlem
    #[error("Geçersiz işlem: {0}")]
    InvalidOperation(String),
    
    /// Geçersiz dosya
    #[error("Geçersiz dosya: {0}")]
    InvalidFile(String),
    
    /// Geçersiz format
    #[error("Geçersiz format: {0}")]
    InvalidFormat(String),
    
    /// Geçersiz kodlama
    #[error("Geçersiz kodlama: {0}")]
    InvalidEncoding(String),
    
    /// Geçersiz sürüm
    #[error("Geçersiz sürüm: {0}")]
    InvalidVersion(String),
    
    /// Geçersiz konfigürasyon
    #[error("Geçersiz konfigürasyon: {0}")]
    InvalidConfiguration(String),
    
    /// Geçersiz durum
    #[error("Geçersiz durum: {0}")]
    InvalidState(String),
    
    /// Geçersiz argüman
    #[error("Geçersiz argüman: {0}")]
    InvalidArgument(String),
    
    /// Geçersiz sonuç
    #[error("Geçersiz sonuç: {0}")]
    InvalidResult(String),
    
    /// Geçersiz dönüş
    #[error("Geçersiz dönüş: {0}")]
    InvalidReturn(String),
    
    /// Geçersiz döngü
    #[error("Geçersiz döngü: {0}")]
    InvalidLoop(String),
    
    /// Geçersiz koşul
    #[error("Geçersiz koşul: {0}")]
    InvalidCondition(String),
    
    /// Geçersiz atama
    #[error("Geçersiz atama: {0}")]
    InvalidAssignment(String),
    
    /// Geçersiz çağrı
    #[error("Geçersiz çağrı: {0}")]
    InvalidCall(String),
    
    /// Geçersiz tanım
    #[error("Geçersiz tanım: {0}")]
    InvalidDefinition(String),
    
    /// Geçersiz kullanım
    #[error("Geçersiz kullanım: {0}")]
    InvalidUsage(String),
    
    /// Geçersiz erişim
    #[error("Geçersiz erişim: {0}")]
    InvalidAccess(String),
    
    /// Geçersiz izin
    #[error("Geçersiz izin: {0}")]
    InvalidPermission(String),
    
    /// Geçersiz yetki
    #[error("Geçersiz yetki: {0}")]
    InvalidAuthority(String),
    
    /// Geçersiz güvenlik
    #[error("Geçersiz güvenlik: {0}")]
    InvalidSecurity(String),
    
    /// Geçersiz doğrulama
    #[error("Geçersiz doğrulama: {0}")]
    InvalidValidation(String),
    
    /// Geçersiz kontrol
    #[error("Geçersiz kontrol: {0}")]
    InvalidCheck(String),
    
    /// Geçersiz test
    #[error("Geçersiz test: {0}")]
    InvalidTest(String),
    
    /// Geçersiz hata
    #[error("Geçersiz hata: {0}")]
    InvalidError(String),
    
    /// Geçersiz uyarı
    #[error("Geçersiz uyarı: {0}")]
    InvalidWarning(String),
    
    /// Geçersiz bilgi
    #[error("Geçersiz bilgi: {0}")]
    InvalidInfo(String),
    
    /// Geçersiz debug
    #[error("Geçersiz debug: {0}")]
    InvalidDebug(String),
    
    /// Geçersiz trace
    #[error("Geçersiz trace: {0}")]
    InvalidTrace(String),
    
    /// Geçersiz log
    #[error("Geçersiz log: {0}")]
    InvalidLog(String),
    
    /// Geçersiz rapor
    #[error("Geçersiz rapor: {0}")]
    InvalidReport(String),
    
    /// Geçersiz istatistik
    #[error("Geçersiz istatistik: {0}")]
    InvalidStatistic(String),
    
    /// Geçersiz metrik
    #[error("Geçersiz metrik: {0}")]
    InvalidMetric(String),
    
    /// Geçersiz performans
    #[error("Geçersiz performans: {0}")]
    InvalidPerformance(String),
    
    /// Geçersiz optimizasyon
    #[error("Geçersiz optimizasyon: {0}")]
    InvalidOptimization(String),
    
    /// Geçersiz paralelleştirme
    #[error("Geçersiz paralelleştirme: {0}")]
    InvalidParallelization(String),
    
    /// Geçersiz dağıtım
    #[error("Geçersiz dağıtım: {0}")]
    InvalidDistribution(String),
    
    /// Geçersiz ölçekleme
    #[error("Geçersiz ölçekleme: {0}")]
    InvalidScaling(String),
    
    /// Geçersiz yükleme
    #[error("Geçersiz yükleme: {0}")]
    InvalidLoading(String),
    
    /// Geçersiz kaydetme
    #[error("Geçersiz kaydetme: {0}")]
    InvalidSaving(String),
    
    /// Geçersiz aktarım
    #[error("Geçersiz aktarım: {0}")]
    InvalidTransfer(String),
    
    /// Geçersiz dönüşüm
    #[error("Geçersiz dönüşüm: {0}")]
    InvalidConversion(String),
    
    /// Geçersiz işleme
    #[error("Geçersiz işleme: {0}")]
    InvalidProcessing(String),
    
    /// Geçersiz analiz
    #[error("Geçersiz analiz: {0}")]
    InvalidAnalysis(String),
    
    /// Geçersiz sentez
    #[error("Geçersiz sentez: {0}")]
    InvalidSynthesis(String),
    
    /// Geçersiz simülasyon
    #[error("Geçersiz simülasyon: {0}")]
    InvalidSimulation(String),
    
    /// Geçersiz modelleme
    #[error("Geçersiz modelleme: {0}")]
    InvalidModeling(String),
    
    /// Geçersiz tahmin
    #[error("Geçersiz tahmin: {0}")]
    InvalidPrediction(String),
    
    /// Geçersiz sınıflandırma
    #[error("Geçersiz sınıflandırma: {0}")]
    InvalidClassification(String),
    
    /// Geçersiz regresyon
    #[error("Geçersiz regresyon: {0}")]
    InvalidRegression(String),
    
    /// Geçersiz kümeleme
    #[error("Geçersiz kümeleme: {0}")]
    InvalidClustering(String),
    
    /// Geçersiz boyut indirgeme
    #[error("Geçersiz boyut indirgeme: {0}")]
    InvalidDimensionalityReduction(String),
    
    /// Geçersiz özellik çıkarma
    #[error("Geçersiz özellik çıkarma: {0}")]
    InvalidFeatureExtraction(String),
    
    /// Geçersiz özellik seçimi
    #[error("Geçersiz özellik seçimi: {0}")]
    InvalidFeatureSelection(String),
    
    /// Geçersiz özellik mühendisliği
    #[error("Geçersiz özellik mühendisliği: {0}")]
    InvalidFeatureEngineering(String),
}

/// Parser sonuç tipi
pub type Result<T> = std::result::Result<T, ParseError>;
/// Token tipi
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// Sayı
    Sayi(String),
    /// Metin
    Metin(String),
    /// Tanımlayıcı
    Tanimlayici(String),
    /// Artı
    Artı,
    /// Eksi
    Eksi,
    /// Çarp
    Carp,
    /// Böl
    Bol,
    /// Aç parantez
    AcParantez,
    /// Kapa parantez
    KapaParantez,
    /// Aç küme
    AcKume,
    /// Kapa küme
    KapaKume,
    /// Virgül
    Virgul,
    /// Nokta
    Nokta,
    /// Noktalı virgül
    NoktaliVirgul,
    /// Eşittir
    Esittir,
    /// Model
    Model,
    /// Katman
    Katman,
    /// Dense
    Dense,
    /// Conv2D
    Conv2D,
    /// MaxPooling2D
    MaxPooling2D,
    /// Flatten
    Flatten,
    /// SGD
    SGD,
    /// Adam
    Adam,
    /// AdamW
    AdamW,
    /// RMSProp
    RMSProp,
    /// MSE
    MSE,
    /// MAE
    MAE,
    /// CrossEntropy
    CrossEntropy,
    /// BinaryCrossEntropy
    BinaryCrossEntropy,
    /// HuberLoss
    HuberLoss,
    /// Eğit
    Egit,
    /// Veri
    Veri,
    /// Tahmin
    Tahmin,
    /// Eğer
    Eger,
    /// Döngü
    Dongu,
    /// İçin
    Icin,
    /// Döndür
    Dondur,
    /// Fonksiyon
    Fonksiyon,
    /// Değişken
    Degisken(String),
    /// EOF
    EOF,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Sayi(num) => write!(f, "{}", num),
            Token::Metin(text) => write!(f, "\"{}\"", text),
            Token::Tanimlayici(name) => write!(f, "{}", name),
            Token::Artı => write!(f, "+"),
            Token::Eksi => write!(f, "-"),
            Token::Carp => write!(f, "*"),
            Token::Bol => write!(f, "/"),
            Token::AcParantez => write!(f, "("),
            Token::KapaParantez => write!(f, ")"),
            Token::AcKume => write!(f, "{{"),
            Token::KapaKume => write!(f, "}}"),
            Token::Virgul => write!(f, ","),
            Token::Nokta => write!(f, "."),
            Token::NoktaliVirgul => write!(f, ";"),
            Token::Esittir => write!(f, "="),
            Token::Model => write!(f, "Model"),
            Token::Katman => write!(f, "Katman"),
            Token::Dense => write!(f, "Dense"),
            Token::Conv2D => write!(f, "Conv2D"),
            Token::MaxPooling2D => write!(f, "MaxPooling2D"),
            Token::Flatten => write!(f, "Flatten"),
            Token::SGD => write!(f, "SGD"),
            Token::Adam => write!(f, "Adam"),
            Token::AdamW => write!(f, "AdamW"),
            Token::RMSProp => write!(f, "RMSProp"),
            Token::MSE => write!(f, "MSE"),
            Token::MAE => write!(f, "MAE"),
            Token::CrossEntropy => write!(f, "CrossEntropy"),
            Token::BinaryCrossEntropy => write!(f, "BinaryCrossEntropy"),
            Token::HuberLoss => write!(f, "HuberLoss"),
            Token::Egit => write!(f, "Egit"),
            Token::Veri => write!(f, "Veri"),
            Token::Tahmin => write!(f, "Tahmin"),
            Token::Eger => write!(f, "Eger"),
            Token::Dongu => write!(f, "Dongu"),
            Token::Icin => write!(f, "Icin"),
            Token::Dondur => write!(f, "Dondur"),
            Token::Fonksiyon => write!(f, "Fonksiyon"),
            Token::Degisken(name) => write!(f, "{}", name),
            Token::EOF => write!(f, "EOF"),
        }
    }
}

impl ParseError {
    /// Beklenmeyen token hatası oluştur
    pub fn unexpected_token(token: Token, expected: &str) -> Self {
        ParseError::UnexpectedToken(token, expected.to_string())
    }
} 

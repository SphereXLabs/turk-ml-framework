pub mod model_parser;
pub mod expr_parser;
pub mod error;

pub use error::{ParseError, Result};

use crate::ast::Program;
use crate::token::Token;

/// TURK dili parser trait'i
///
/// Bu trait, TURK dili sözdizimini çözümleyen temel arayüzü tanımlar.
/// Farklı parser implementasyonları (örn: model parser, ifade parser) 
/// bu trait'i implemente ederek ortak bir arayüz sağlar.
///
/// # Örnek kullanım
/// ```
/// use turk_lang::parser::{Parser, ModelParser};
/// use turk_lang::ast::Program;
///
/// let code = "Model cifar_model {
///     Katman(3, 64, ReLU),
///     Katman(64, 10, Softmax)
/// }";
///
/// let mut parser = ModelParser::new(code);
/// let program: Program = parser.parse().expect("Ayrıştırma hatası");
/// ```
pub trait Parser: Send + Sync {
    /// Program AST'sini oluşturur
    ///
    /// Verilen kaynak koddan AST (Abstract Syntax Tree) yapısını
    /// oluşturur ve döndürür.
    ///
    /// # Hatalar
    /// Eğer kod geçerli bir TURK programı değilse `ParseError` döndürür.
    fn parse(&mut self) -> Result<Program>;
    
    /// Geçerli token'i ilerletir ve eski token'i döndürür
    ///
    /// # Dönüş
    /// - `Some(Token)`: İlerletilen token
    /// - `None`: Tokenler tükenmişse
    fn advance(&mut self) -> Option<Token>;
    
    /// Geçerli token'i kontrol eder
    ///
    /// # Dönüş
    /// - `Some(&Token)`: Geçerli token referansı
    /// - `None`: Token kalmamışsa
    fn current(&self) -> Option<&Token>;
    
    /// Peek - Sonraki token'i tüketmeden kontrol eder
    ///
    /// # Dönüş
    /// - `Some(&Token)`: Sonraki token referansı
    /// - `None`: Sonraki token yoksa
    fn peek(&self) -> Option<&Token>;
    
    /// Belirli bir token türünü bekler
    ///
    /// Geçerli token beklenen tipe eşitse token'i döndürür ve
    /// parser'ı ilerletir, aksi halde hata döndürür.
    ///
    /// # Parametreler
    /// - `expected`: Beklenen token türü
    ///
    /// # Hatalar
    /// Eğer token beklenen türde değilse `ParseError` döndürür.
    fn expect(&mut self, expected: Token) -> Result<Token>;
    
    /// Belirli token türlerinden birini bekler
    ///
    /// Geçerli token, beklenen tiplerden birine eşitse token'i döndürür
    /// ve parser'ı ilerletir, aksi takdirde hata döndürür.
    ///
    /// # Parametreler
    /// - `expected_tokens`: Beklenen token türleri
    ///
    /// # Hatalar
    /// Eğer token beklenen türlerden biri değilse `ParseError` döndürür.
    fn expect_one_of(&mut self, expected_tokens: &[Token]) -> Result<Token>;
    
    /// Geçerli token belirli bir türde mi kontrol eder
    ///
    /// # Parametreler
    /// - `token_type`: Kontrol edilecek token türü
    ///
    /// # Dönüş
    /// Token belirtilen türde ise `true`, değilse veya token kalmamışsa `false`
    fn check(&self, token_type: &Token) -> bool;
    
    /// Hata raporlama
    ///
    /// Belirlenen hata mesajıyla bir `ParseError` oluşturur.
    ///
    /// # Parametreler
    /// - `message`: Hata mesajı
    fn error(&self, message: &str) -> ParseError;
    
    /// Parser durumunu resetler
    ///
    /// Parser'ı başlangıç durumuna döndürür, tüm
    /// geçici durum bilgilerini temizler.
    fn reset(&mut self);
    
    /// Ayrıştırma konumunu alır
    ///
    /// Şu anki ayrıştırma konumunu (satır, sütun) döndürür.
    ///
    /// # Dönüş
    /// - `(usize, usize)`: (satır, sütun) şeklinde konum bilgisi
    fn get_position(&self) -> (usize, usize);
    
    /// Hata kurtarma
    ///
    /// Ayrıştırma sırasında bir hata oluştuğunda, parser'ı
    /// en yakın güvenli noktaya getirir.
    ///
    /// # Parametreler
    /// - `sync_tokens`: Senkronizasyon tokenları
    fn recover(&mut self, sync_tokens: &[Token]);
} 
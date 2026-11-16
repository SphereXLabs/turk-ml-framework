pub mod model_parser;
pub mod expr_parser;
pub mod error;

pub use error::{ParseError, Result};

use crate::ast::{Deyim, Ifade, Program, KatmanOzellikleri, ModelOzellikleri, OptimizerTipi, KayipFonksiyonu};
use crate::token::Token;
use crate::lexer::Lexer;
use log::debug;
use std::collections::HashMap;
use super::expr_parser::ExprParser;
use super::model_parser::ModelParser;
use crate::ast::*;
use crate::error::TurkError;
use logos::Logos;

/// Türk programlama dilini ayrıştırmak için parser
///
/// # Örnek
///
/// ```
/// use turk_lang::parser::Parser;
/// 
/// // Model tanımlama
/// let kod = r#"
///     model SiniflandiriciModel {
///         katman Dense(784, 128, relu)
///         katman Dense(128, 10, softmax)
///         optimizer Adam(0.001)
///     }
/// "#;
/// 
/// let mut parser = Parser::new(kod);
/// let program = parser.parse().expect("Program ayrıştırılamadı");
/// ```
pub trait Parser {
    /// Programı ayrıştırır
    fn parse(&mut self) -> Result<Program>;
    
    /// Bir sonraki ifadeyi ayrıştırır
    fn deyim_parse(&mut self) -> Result<Option<Deyim>>;
    
    /// Geçerli token'ı tüketmeden döndürür
    fn current(&self) -> Option<&(Token, String)>;
    
    /// Geçerli token'ı tüketmeden döndürür
    fn current_token(&self) -> Option<Token>;
    
    /// Geçerli token text'i döndürür
    fn current_text(&self) -> Option<String>;
    
    /// Token işaretçisini ilerletir ve eski token'ı döndürür
    fn advance(&mut self) -> Option<(Token, String)>;
    
    /// İlerlet ve token'ı döndür
    fn advance_token(&mut self) -> Option<Token>;
    
    /// Belirli bir token tipini bekler
    fn check(&self, expected: &Token) -> bool;
    
    /// Hata oluştur
    fn error(&self, message: &str) -> ParseError;
    
    /// Parser durumunu sıfırla
    fn reset(&mut self);
    
    /// Geçerli pozisyonu al
    fn get_position(&self) -> (usize, usize);
    
    /// Hata kurtarma
    fn recover(&mut self, sync_tokens: &[Token]);
}

/// Parser
pub struct Parser {
    lexer: logos::Lexer<Token>,
    current_token: Option<Token>,
    next_token: Option<Token>,
    variables: HashMap<String, Value>,
}

impl Parser {
    /// Yeni parser oluştur
    pub fn new(input: &str) -> Self {
        let mut lexer = Token::lexer(input);
        let current_token = lexer.next();
        let next_token = lexer.next();
        
        Self {
            lexer,
            current_token,
            next_token,
            variables: HashMap::new(),
        }
    }
    
    /// Token ilerlet
    fn token_ilerlet(&mut self) {
        self.current_token = self.next_token.take();
        self.next_token = self.lexer.next();
    }
    
    /// Token kontrol et
    fn token_kontrol(&mut self, expected: Token) -> Result<(), TurkError> {
        if self.current_token == Some(expected) {
            self.token_ilerlet();
            Ok(())
        } else {
            Err(TurkError::SyntaxError(format!(
                "Beklenen token: {:?}, Bulunan token: {:?}",
                expected,
                self.current_token
            )))
        }
    }
    
    /// Model tanımı parse et
    pub fn model_tanimi_parse(&mut self) -> Result<Model, TurkError> {
        self.token_kontrol(Token::Model)?;
        
        let mut katmanlar = Vec::new();
        let mut ozellikler = ModelOzellikleri::default();
        
        self.token_kontrol(Token::AcKume)?;
        
        while self.current_token != Some(Token::KapaKume) {
            match self.current_token {
                Some(Token::Katman) => {
                    let katman = self.katman_parse()?;
                    katmanlar.push(katman);
                }
                Some(Token::Tanimlayici(name)) => {
                    let ozellik = self.ozellik_parse(&name)?;
                    ozellikler.set(&ozellik.name, ozellik.value)?;
                }
                _ => {
                    return Err(TurkError::SyntaxError(format!(
                        "Beklenmeyen token: {:?}",
                        self.current_token
                    )))
                }
            }
        }
        
        self.token_kontrol(Token::KapaKume)?;
        
        Ok(Model {
            katmanlar,
            ozellikler,
        })
    }
    
    /// Katman parse et
    fn katman_parse(&mut self) -> Result<Katman, TurkError> {
        self.token_kontrol(Token::Katman)?;
        
        let tip = match self.current_token {
            Some(Token::Dense) => KatmanTipi::Dense,
            Some(Token::Conv2D) => KatmanTipi::Conv2D,
            Some(Token::MaxPooling2D) => KatmanTipi::MaxPooling2D,
            Some(Token::Flatten) => KatmanTipi::Flatten,
            _ => {
                return Err(TurkError::SyntaxError(format!(
                    "Beklenmeyen katman tipi: {:?}",
                    self.current_token
                )))
            }
        };
        
        self.token_ilerlet();
        self.token_kontrol(Token::AcKume)?;
        
        let mut ozellikler = KatmanOzellikleri::default();
        
        while self.current_token != Some(Token::KapaKume) {
            if let Some(Token::Tanimlayici(name)) = &self.current_token {
                let ozellik = self.ozellik_parse(name)?;
                ozellikler.set(&ozellik.name, ozellik.value)?;
            } else {
                return Err(TurkError::SyntaxError(format!(
                    "Beklenmeyen token: {:?}",
                    self.current_token
                )));
            }
        }
        
        self.token_kontrol(Token::KapaKume)?;
        
        Ok(Katman { tip, ozellikler })
    }
    
    /// Özellik parse et
    fn ozellik_parse(&mut self, name: &str) -> Result<Ozellik, TurkError> {
        self.token_ilerlet();
        self.token_kontrol(Token::Esittir)?;
        
        let value = self.deger_parse()?;
        
        Ok(Ozellik {
            name: name.to_string(),
            value,
        })
    }
    
    /// Değer parse et
    fn deger_parse(&mut self) -> Result<Value, TurkError> {
        match self.current_token {
            Some(Token::Sayi(num)) => {
                self.token_ilerlet();
                Ok(Value::Sayi(num.parse().unwrap()))
            }
            Some(Token::Metin(text)) => {
                self.token_ilerlet();
                Ok(Value::Metin(text))
            }
            Some(Token::AcKume) => {
                self.token_ilerlet();
                let mut values = Vec::new();
                
                while self.current_token != Some(Token::KapaKume) {
                    values.push(self.deger_parse()?);
                    
                    if self.current_token == Some(Token::Virgul) {
                        self.token_ilerlet();
                    }
                }
                
                self.token_kontrol(Token::KapaKume)?;
                Ok(Value::Dizi(values))
            }
            Some(Token::Degisken(name)) => {
                self.token_ilerlet();
                if let Some(value) = self.variables.get(&name) {
                    Ok(value.clone())
                } else {
                    Err(TurkError::VariableNotFound(name))
                }
            }
            _ => Err(TurkError::SyntaxError(format!(
                "Beklenmeyen değer: {:?}",
                self.current_token
            ))),
        }
    }
}

/// Program
#[derive(Debug)]
pub struct Program {
    /// Deyimler
    pub deyimler: Vec<Deyim>,
} 
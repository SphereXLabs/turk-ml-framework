pub mod model_parser;
pub mod expr_parser;
pub mod error;

pub use model_parser::ModelParser;
pub use expr_parser::ExprParser;
pub use error::{ParseError, Result};

use crate::ast::Program;
use crate::token::Token;

/// TURK dili parser trait'i
pub trait Parser {
    /// Program AST'sini oluşturur
    fn parse(&mut self) -> Result<Program>;
    
    /// Geçerli token'i ilerletir
    fn advance(&mut self) -> Option<Token>;
    
    /// Geçerli token'i kontrol eder
    fn current(&self) -> Option<&Token>;
    
    /// Belirli bir token türünü bekler
    fn expect(&mut self, expected: Token) -> Result<Token>;
    
    /// Hata raporlama
    fn error(&self, message: &str) -> ParseError;
    
    /// Parser durumunu resetler
    fn reset(&mut self);
} 
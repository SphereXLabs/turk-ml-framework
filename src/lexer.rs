use crate::error::{Result, TurkError};
use crate::token::Token;
use log::debug;
use logos::Logos;

/// TÜRK dili için sözcük çözümleyici (lexer) yapısı
pub struct Lexer<'a> {
    /// İşlenecek kaynak kod
    source: &'a str,
    /// Logos lexer
    tokens: logos::Lexer<'a, Token>,
    /// Geçerli token
    current_token: Option<Token>,
    /// Geçerli token pozisyonu
    position: usize,
    /// Satır numarası
    line: usize,
    /// Satırdaki konum
    column: usize,
}

impl<'a> Lexer<'a> {
    /// Yeni bir lexer oluşturur
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Self {
            source,
            tokens: Token::lexer(source),
            current_token: None,
            position: 0,
            line: 1,
            column: 1,
        };

        // İlk token'ı al
        lexer.advance();
        lexer
    }

    /// Bir sonraki token'a ilerle
    pub fn advance(&mut self) -> Option<Token> {
        // Bir sonraki token'ı al
        if let Some(token) = self.tokens.next() {
            // Token'ın konumunu güncelle
            let token_span = self.tokens.span();
            self.update_position(token_span.start);

            self.current_token = Some(token.clone());
            debug!(
                "Token: {:?}, satır: {}, sütun: {}",
                token, self.line, self.column
            );
            Some(token)
        } else {
            self.current_token = None;
            None
        }
    }

    /// Geçerli token'ı döndürür
    pub fn current(&self) -> Option<Token> {
        self.current_token.clone()
    }

    /// Geçerli konumu günceller
    fn update_position(&mut self, new_pos: usize) {
        // Pozisyon değişirse satır ve sütun bilgisini güncelle
        if new_pos > self.position {
            let text_slice = &self.source[self.position..new_pos];
            for c in text_slice.chars() {
                if c == '\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
            }
            self.position = new_pos;
        }
    }

    /// Geçerli token eğer beklenen token ise ilerler, değilse hata döndürür
    pub fn expect(&mut self, expected: Token) -> Result<Token> {
        if let Some(token) = self.current() {
            if token == expected {
                let token = token.clone();
                self.advance();
                Ok(token)
            } else {
                Err(TurkError::SyntaxError(format!(
                    "Satır {}, sütun {}: Beklenen token: {:?}, alınan: {:?}",
                    self.line, self.column, expected, token
                )))
            }
        } else {
            Err(TurkError::SyntaxError(format!(
                "Satır {}, sütun {}: Beklenen token: {:?}, dosya sonu",
                self.line, self.column, expected
            )))
        }
    }

    /// Eğer geçerli token beklenen token ise ilerler ve true döndürür, değilse ilerlemez ve false döndürür
    pub fn match_token(&mut self, expected: Token) -> bool {
        if let Some(token) = self.current() {
            if token == expected {
                self.advance();
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Geçerli satır numarasını döndürür
    pub fn line(&self) -> usize {
        self.line
    }

    /// Geçerli sütun numarasını döndürür
    pub fn column(&self) -> usize {
        self.column
    }
}

/// Tüm token'ları bir vektöre toplar
pub fn tokenize(source: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut lexer = Lexer::new(source);

    while let Some(token) = lexer.current() {
        tokens.push(token);
        lexer.advance();
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_model_olustur() {
        let source = "model_olustur()";
        let tokens = tokenize(source);

        // Test içerisindeki token isimleri güncellenmeli
        assert_eq!(tokens.len(), 3);
        // Token sıralaması kontrol edilebilir
    }

    #[test]
    fn test_tokenize_katman_ekle() {
        let source = "katman_ekle(yogun(10, 5, \"relu\"))";
        let tokens = tokenize(source);

        // Test içerisindeki token isimleri güncellenmeli
        assert_eq!(tokens.len(), 11);
        // Token sıralaması kontrol edilebilir
    }

    #[test]
    fn test_tokenize_derle() {
        let source = "derle(adam)";
        let tokens = tokenize(source);

        // Test içerisindeki token isimleri güncellenmeli
        assert_eq!(tokens.len(), 4);
        // Token sıralaması kontrol edilebilir
    }

    #[test]
    fn test_tokenize_egit() {
        let source = "egit(10, 32)";
        let tokens = tokenize(source);

        // Test içerisindeki token isimleri güncellenmeli
        assert_eq!(tokens.len(), 6);
        // Token sıralaması kontrol edilebilir
    }
}

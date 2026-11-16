use serde::{Deserialize, Serialize};
use std::fmt;

/// Kaynak kodunun konumunu belirtir
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourcePosition {
    /// Satır numarası (1'den başlar)
    pub line: usize,
    /// Sütun numarası (1'den başlar)
    pub column: usize,
    /// Dosyadaki karakter konumu (0'dan başlar)
    pub offset: usize,
}

impl SourcePosition {
    /// Yeni bir konum oluşturur
    pub fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }

    /// Başlangıç konumu
    pub fn start() -> Self {
        Self::new(1, 1, 0)
    }

    /// Geçersiz bir konum döndürür
    pub fn invalid() -> Self {
        Self::new(0, 0, 0)
    }

    /// Konumun geçerli olup olmadığını kontrol eder
    pub fn is_valid(&self) -> bool {
        self.line > 0 && self.column > 0
    }

    /// Bir sonraki konum
    pub fn next(&self, ch: char) -> Self {
        if ch == '\n' {
            Self::new(self.line + 1, 1, self.offset + 1)
        } else {
            Self::new(self.line, self.column + 1, self.offset + 1)
        }
    }

    /// Verilen karaktere göre pozisyonu günceller
    pub fn advance(&mut self, c: char) {
        self.offset += 1;
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
    }

    /// Bir dizeye göre pozisyonu günceller
    pub fn advance_str(&mut self, s: &str) {
        for c in s.chars() {
            self.advance(c);
        }
    }
}

impl fmt::Display for SourcePosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_position() {
        let pos = SourcePosition::new(1, 1, 0);
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 1);
        assert_eq!(pos.offset, 0);

        let next_pos = pos.next('a');
        assert_eq!(next_pos.line, 1);
        assert_eq!(next_pos.column, 2);
        assert_eq!(next_pos.offset, 1);

        let newline_pos = pos.next('\n');
        assert_eq!(newline_pos.line, 2);
        assert_eq!(newline_pos.column, 1);
        assert_eq!(newline_pos.offset, 1);

        let invalid = SourcePosition::invalid();
        assert!(!invalid.is_valid());

        let start = SourcePosition::start();
        assert_eq!(start.line, 1);
        assert_eq!(start.column, 1);
        assert_eq!(start.offset, 0);
        assert!(start.is_valid());
    }

    #[test]
    fn test_new_position() {
        let pos = SourcePosition::new(2, 5, 10);
        assert_eq!(pos.line, 2);
        assert_eq!(pos.column, 5);
        assert_eq!(pos.offset, 10);
    }

    #[test]
    fn test_start_position() {
        let pos = SourcePosition::start();
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 1);
        assert_eq!(pos.offset, 0);
    }

    #[test]
    fn test_advance() {
        let mut pos = SourcePosition::start();

        pos.advance('a');
        assert_eq!(pos.line, 1);
        assert_eq!(pos.column, 2);
        assert_eq!(pos.offset, 1);

        pos.advance('\n');
        assert_eq!(pos.line, 2);
        assert_eq!(pos.column, 1);
        assert_eq!(pos.offset, 2);

        pos.advance('b');
        assert_eq!(pos.line, 2);
        assert_eq!(pos.column, 2);
        assert_eq!(pos.offset, 3);
    }

    #[test]
    fn test_advance_str() {
        let mut pos = SourcePosition::start();

        pos.advance_str("hello\nworld");
        assert_eq!(pos.line, 2);
        assert_eq!(pos.column, 6); // "world" = 5 karakter + 1 (1-based)
        assert_eq!(pos.offset, 11); // "hello\nworld" = 11 karakter
    }

    #[test]
    fn test_display() {
        let pos = SourcePosition::new(5, 10, 42);
        assert_eq!(format!("{}", pos), "5:10");
    }
}

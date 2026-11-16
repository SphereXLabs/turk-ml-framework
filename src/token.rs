use logos::Logos;
use std::fmt;

/// Token enum'u
#[derive(Logos, Debug, PartialEq, Clone)]
pub enum Token {
    // Anahtar Kelimeler
    #[token("model")]
    Model,

    #[token("katman")]
    Layer,

    #[token("aktivasyon")]
    Activation,

    #[token("optimizer")]
    Optimizer,

    #[token("egit")]
    Train,

    #[token("tahmin")]
    Predict,

    #[token("kaydet")]
    Save,

    #[token("yukle")]
    Load,

    // Operatörler
    #[token("+")]
    Add,

    #[token("-")]
    Sub,

    #[token("*")]
    Mul,

    #[token("/")]
    Div,

    #[token("=")]
    Assign,

    #[token("==")]
    Eq,

    #[token("!=")]
    Neq,

    #[token("<")]
    Lt,

    #[token(">")]
    Gt,

    #[token("<=")]
    Le,

    #[token(">=")]
    Ge,

    // Parantezler ve ayraçlar
    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,

    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token(",")]
    Comma,

    #[token(":")]
    Colon,

    #[token(";")]
    Semicolon,

    // İstatistiksel Fonksiyonlar
    #[token("ortalama")]
    Mean,

    #[token("medyan")]
    Median,

    #[token("mod")]
    Mode,

    #[token("varyans")]
    Variance,

    #[token("standart_sapma")]
    StdDev,

    #[token("korelasyon")]
    Correlation,

    #[token("kovaryans")]
    Covariance,

    // Aktivasyon Fonksiyonları
    #[token("relu")]
    Relu,

    #[token("leakyRelu")]
    LeakyRelu,

    #[token("sigmoid")]
    Sigmoid,

    #[token("tanh")]
    Tanh,

    #[token("softmax")]
    Softmax,

    #[token("swish")]
    Swish,

    // Kayıp Fonksiyonları
    #[token("mse")]
    MSE,

    #[token("mae")]
    MAE,

    #[token("crossEntropy")]
    CrossEntropy,

    #[token("binaryCrossEntropy")]
    BinaryCrossEntropy,

    #[token("huberLoss")]
    HuberLoss,

    // Optimizer Tipleri
    #[token("sgd")]
    SGD,

    #[token("adam")]
    Adam,

    #[token("adamw")]
    AdamW,

    #[token("rmsprop")]
    RMSProp,

    // Sayılar ve metinler
    #[regex(r"[0-9]+(\.[0-9]+)?", |lex| lex.slice().parse())]
    Number(f64),

    #[regex(r#""[^"]*""#, |lex| lex.slice()[1..lex.slice().len()-1].to_string())]
    String(String),

    // Tanımlayıcılar (değişken, fonksiyon isimleri)
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Identifier(String),

    // Katman tipleri
    #[token("dense")]
    Dense,

    #[token("conv2d")]
    Conv2D,

    #[token("maxpooling2d")]
    MaxPooling2D,

    #[token("flatten")]
    Flatten,

    // Yorum satırları ve boşluklar
    #[regex(r"[ \t\n\f]+", logos::skip)]
    #[regex(r"//[^\n]*", logos::skip)]
    #[regex(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", logos::skip)]
    Whitespace,

    // Dosya sonu
    EOF,

    // Hata durumu
    #[error]
    Error,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Model => write!(f, "model"),
            Token::Layer => write!(f, "katman"),
            Token::Activation => write!(f, "aktivasyon"),
            Token::Optimizer => write!(f, "optimizer"),
            Token::Train => write!(f, "egit"),
            Token::Predict => write!(f, "tahmin"),
            Token::Save => write!(f, "kaydet"),
            Token::Load => write!(f, "yukle"),
            Token::Add => write!(f, "+"),
            Token::Sub => write!(f, "-"),
            Token::Mul => write!(f, "*"),
            Token::Div => write!(f, "/"),
            Token::Assign => write!(f, "="),
            Token::Eq => write!(f, "=="),
            Token::Neq => write!(f, "!="),
            Token::Lt => write!(f, "<"),
            Token::Gt => write!(f, ">"),
            Token::Le => write!(f, "<="),
            Token::Ge => write!(f, ">="),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::Semicolon => write!(f, ";"),
            Token::Mean => write!(f, "ortalama"),
            Token::Median => write!(f, "medyan"),
            Token::Mode => write!(f, "mod"),
            Token::Variance => write!(f, "varyans"),
            Token::StdDev => write!(f, "standart_sapma"),
            Token::Correlation => write!(f, "korelasyon"),
            Token::Covariance => write!(f, "kovaryans"),
            Token::Relu => write!(f, "relu"),
            Token::LeakyRelu => write!(f, "leakyRelu"),
            Token::Sigmoid => write!(f, "sigmoid"),
            Token::Tanh => write!(f, "tanh"),
            Token::Softmax => write!(f, "softmax"),
            Token::Swish => write!(f, "swish"),
            Token::MSE => write!(f, "mse"),
            Token::MAE => write!(f, "mae"),
            Token::CrossEntropy => write!(f, "crossEntropy"),
            Token::BinaryCrossEntropy => write!(f, "binaryCrossEntropy"),
            Token::HuberLoss => write!(f, "huberLoss"),
            Token::SGD => write!(f, "sgd"),
            Token::Adam => write!(f, "adam"),
            Token::AdamW => write!(f, "adamw"),
            Token::RMSProp => write!(f, "rmsprop"),
            Token::Number(n) => write!(f, "{}", n),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Identifier(s) => write!(f, "{}", s),
            Token::Dense => write!(f, "dense"),
            Token::Conv2D => write!(f, "conv2d"),
            Token::MaxPooling2D => write!(f, "maxpooling2d"),
            Token::Flatten => write!(f, "flatten"),
            Token::Whitespace => write!(f, " "),
            Token::EOF => write!(f, "<EOF>"),
            Token::Error => write!(f, "HATA"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use logos::Logos;

    #[test]
    fn test_basic_tokens() {
        let mut lexer = Token::lexer("model katman aktivasyon optimizer");

        assert_eq!(lexer.next(), Some(Token::Model));
        assert_eq!(lexer.next(), Some(Token::Layer));
        assert_eq!(lexer.next(), Some(Token::Activation));
        assert_eq!(lexer.next(), Some(Token::Optimizer));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_operator_tokens() {
        let mut lexer = Token::lexer("+ - * / = == != < > <= >=");

        assert_eq!(lexer.next(), Some(Token::Add));
        assert_eq!(lexer.next(), Some(Token::Sub));
        assert_eq!(lexer.next(), Some(Token::Mul));
        assert_eq!(lexer.next(), Some(Token::Div));
        assert_eq!(lexer.next(), Some(Token::Assign));
        assert_eq!(lexer.next(), Some(Token::Eq));
        assert_eq!(lexer.next(), Some(Token::Neq));
        assert_eq!(lexer.next(), Some(Token::Lt));
        assert_eq!(lexer.next(), Some(Token::Gt));
        assert_eq!(lexer.next(), Some(Token::Le));
        assert_eq!(lexer.next(), Some(Token::Ge));
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_literal_tokens() {
        let mut lexer = Token::lexer("123 2.5 \"merhaba\"");
        assert_eq!(lexer.next(), Some(Token::Number(123.0)));
        assert_eq!(lexer.next(), Some(Token::Number(2.5)));
        match lexer.next() {
            Some(Token::String(s)) => assert_eq!(s, "merhaba"),
            other => panic!("String bekleniyor, gelen: {other:?}"),
        }
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_identifier_token() {
        let mut lexer = Token::lexer("degisken_ismi Model123");

        match lexer.next() {
            Some(Token::Identifier(s)) => assert_eq!(s, "degisken_ismi"),
            _ => panic!("Değişken tokeni bekleniyor"),
        }

        match lexer.next() {
            Some(Token::Identifier(s)) => assert_eq!(s, "Model123"),
            _ => panic!("Değişken tokeni bekleniyor"),
        }

        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_comment_skip() {
        let mut lexer = Token::lexer("model // Bu bir yorum\nkatman");

        assert_eq!(lexer.next(), Some(Token::Model));
        assert_eq!(lexer.next(), Some(Token::Layer));
        assert_eq!(lexer.next(), None);

        let mut lexer = Token::lexer("model /* Bu çok\nsatırlı\nbir yorum */ katman");

        assert_eq!(lexer.next(), Some(Token::Model));
        assert_eq!(lexer.next(), Some(Token::Layer));
        assert_eq!(lexer.next(), None);
    }
}

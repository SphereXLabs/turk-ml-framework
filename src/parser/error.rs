use crate::token::Token;
use thiserror::Error;
use std::fmt;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Sözdizimi hatası: {0}")]
    SyntaxError(String),
    
    #[error("Beklenmeyen token: {found:?}, beklenen: {expected}")]
    UnexpectedToken {
        found: Token,
        expected: String,
    },
    
    #[error("Geçersiz deyim: {0:?}")]
    GecersizDeyim(Option<(Token, String)>),
    
    #[error("Eksik token: {0}")]
    MissingToken(String),
    
    #[error("Değişken tanımlı değil: {0}")]
    UndefinedVariable(String),
    
    #[error("Tür uyuşmazlığı: {0}")]
    TypeError(String),
    
    #[error("Model yapılandırma hatası: {0}")]
    ModelConfigError(String),
    
    #[error("Katman yapılandırma hatası: {0}")]
    LayerConfigError(String),
    
    #[error("Optimizer yapılandırma hatası: {0}")]
    OptimizerConfigError(String),
    
    #[error("Kayıp fonksiyonu yapılandırma hatası: {0}")]
    LossFunctionConfigError(String),
    
    #[error("Veri yapılandırma hatası: {0}")]
    DataConfigError(String),
    
    #[error("Geçersiz argümanlar: {0}")]
    InvalidArguments(String),
    
    #[error("İç içe geçmiş ifade hatası: {0}")]
    NestedExpressionError(String),
    
    #[error("Operatör öncelik hatası: {0}")]
    OperatorPrecedenceError(String),
    
    #[error("Beklenmeyen dosya sonu")]
    UnexpectedEOF,
    
    #[error("Dosya bulunamadı: {0}")]
    FileNotFound(String),
    
    #[error("İç hata: {0}")]
    InternalError(String),
}

impl ParseError {
    pub fn syntax_error(message: impl Into<String>) -> Self {
        ParseError::SyntaxError(message.into())
    }
    
    pub fn unexpected_token(found: Token, expected: impl Into<String>) -> Self {
        ParseError::UnexpectedToken {
            found,
            expected: expected.into(),
        }
    }
    
    pub fn undefined_variable(name: impl Into<String>) -> Self {
        ParseError::UndefinedVariable(name.into())
    }
    
    pub fn type_error(message: impl Into<String>) -> Self {
        ParseError::TypeError(message.into())
    }
    
    pub fn model_config_error(message: impl Into<String>) -> Self {
        ParseError::ModelConfigError(message.into())
    }
    
    pub fn layer_config_error(message: impl Into<String>) -> Self {
        ParseError::LayerConfigError(message.into())
    }
    
    pub fn optimizer_config_error(message: impl Into<String>) -> Self {
        ParseError::OptimizerConfigError(message.into())
    }
    
    pub fn loss_function_config_error(message: impl Into<String>) -> Self {
        ParseError::LossFunctionConfigError(message.into())
    }
    
    pub fn data_config_error(message: impl Into<String>) -> Self {
        ParseError::DataConfigError(message.into())
    }
    
    pub fn invalid_arguments(message: impl Into<String>) -> Self {
        ParseError::InvalidArguments(message.into())
    }
    
    pub fn nested_expression_error(message: impl Into<String>) -> Self {
        ParseError::NestedExpressionError(message.into())
    }
    
    pub fn operator_precedence_error(message: impl Into<String>) -> Self {
        ParseError::OperatorPrecedenceError(message.into())
    }
    
    pub fn file_not_found(path: impl Into<String>) -> Self {
        ParseError::FileNotFound(path.into())
    }
    
    pub fn internal_error(message: impl Into<String>) -> Self {
        ParseError::InternalError(message.into())
    }
}

pub type Result<T> = std::result::Result<T, ParseError>;

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::SyntaxError(msg) => write!(f, "Sözdizimi hatası: {}", msg),
            ParseError::UnexpectedToken { found, expected } => write!(f, "Beklenmeyen token: {:?}, beklenen: {}", found, expected),
            ParseError::GecersizDeyim(token) => write!(f, "Geçersiz deyim: {:?}", token),
            ParseError::MissingToken(expected) => write!(f, "Eksik token: {}", expected),
            ParseError::UndefinedVariable(var) => write!(f, "Değişken tanımlı değil: {}", var),
            ParseError::TypeError(msg) => write!(f, "Tür uyuşmazlığı: {}", msg),
            ParseError::ModelConfigError(msg) => write!(f, "Model yapılandırma hatası: {}", msg),
            ParseError::LayerConfigError(msg) => write!(f, "Katman yapılandırma hatası: {}", msg),
            ParseError::OptimizerConfigError(msg) => write!(f, "Optimizer yapılandırma hatası: {}", msg),
            ParseError::LossFunctionConfigError(msg) => write!(f, "Kayıp fonksiyonu yapılandırma hatası: {}", msg),
            ParseError::DataConfigError(msg) => write!(f, "Veri yapılandırma hatası: {}", msg),
            ParseError::InvalidArguments(msg) => write!(f, "Geçersiz argümanlar: {}", msg),
            ParseError::NestedExpressionError(msg) => write!(f, "İç içe geçmiş ifade hatası: {}", msg),
            ParseError::OperatorPrecedenceError(msg) => write!(f, "Operatör öncelik hatası: {}", msg),
            ParseError::UnexpectedEOF => write!(f, "Beklenmeyen dosya sonu"),
            ParseError::FileNotFound(path) => write!(f, "Dosya bulunamadı: {}", path),
            ParseError::InternalError(msg) => write!(f, "İç hata: {}", msg),
        }
    }
} 
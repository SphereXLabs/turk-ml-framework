use serde::{Serialize, Deserialize};
use std::fmt;
use std::collections::HashMap;
use crate::token::Token;
use crate::error::TurkError;
use ndarray::{Array2, ArrayView2};

/// Operatör
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operator {
    /// Toplama
    Topla,
    /// Çıkarma
    Cikar,
    /// Çarpma
    Carp,
    /// Bölme
    Bol,
    /// Eşitlik
    Esit,
    /// Eşit değil
    EsitDegil,
    /// Büyük
    Buyuk,
    /// Küçük
    Kucuk,
    /// Büyük eşit
    BuyukEsit,
    /// Küçük eşit
    KucukEsit,
    /// Ve
    Ve,
    /// Veya
    Veya,
    /// Değil
    Degil,
    /// Artı
    Artı,
    /// Eksi
    Eksi,
}

/// İfade
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Ifade {
    /// Sayı
    Sayi(f64),
    /// Metin
    Metin(String),
    /// Değişken
    Degisken(String),
    /// İkili işlem
    Binary {
        /// Operatör
        op: Operator,
        /// Sol ifade
        left: Box<Ifade>,
        /// Sağ ifade
        right: Box<Ifade>,
    },
    /// Tekli işlem
    Unary {
        /// Operatör
        op: Operator,
        /// İfade
        expr: Box<Ifade>,
    },
    /// Fonksiyon çağrısı
    FonksiyonCagir {
        /// Fonksiyon ismi
        isim: String,
        /// Argümanlar
        argumanlar: Vec<Ifade>,
    },
    /// Dizi
    Dizi(Vec<Ifade>),
    /// Sözlük
    Sozluk(Vec<(String, Ifade)>),
    /// Koşul
    Kosul {
        /// Koşul
        kosul: Box<Ifade>,
        /// Doğru durum
        dogru: Box<Ifade>,
        /// Yanlış durum
        yanlis: Box<Ifade>,
    },
    /// Döngü
    Dongu {
        /// Başlangıç
        baslangic: Box<Ifade>,
        /// Koşul
        kosul: Box<Ifade>,
        /// Adım
        adim: Box<Ifade>,
        /// Gövde
        gövde: Box<Ifade>,
    },
    /// İçin döngüsü
    IcinDongu {
        /// Değişken
        degisken: String,
        /// Dizi
        dizi: Box<Ifade>,
        /// Gövde
        gövde: Box<Ifade>,
    },
    /// Döndür
    Dondur(Box<Ifade>),
    /// Kes
    Kes,
    /// Devam
    Devam,
    /// Boş
    Bos,
}

/// Deyim
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Deyim {
    /// İfade
    Ifade(Ifade),
    /// Atama
    Atama {
        /// İsim
        isim: String,
        /// Değer
        deger: Ifade,
    },
    /// Veri tanımı
    VeriTanimi {
        /// İsim
        isim: String,
        /// Değer
        deger: Box<Ifade>,
    },
    /// Model tanımı
    ModelTanimi {
        /// İsim
        isim: String,
        /// Özellikler
        ozellikler: ModelOzellikleri,
    },
    /// Eğit
    Egit {
        /// Model
        model: String,
        /// Veri
        veri: String,
        /// Hedef
        hedef: String,
        /// Batch boyutu
        batch_boyutu: i64,
        /// Epoch sayısı
        epoch_sayisi: i64,
        /// Metrikler
        metrikler: Option<Vec<String>>,
        /// Erken durdurma
        erken_durdurma: Option<i64>,
    },
    /// Tahmin
    Tahmin {
        /// Model
        model: String,
        /// Veri
        veri: String,
    },
    /// Eğer
    Eger {
        /// Koşul
        kosul: Ifade,
        /// Doğru durum
        dogru: Vec<Deyim>,
        /// Yanlış durum
        yanlis: Option<Vec<Deyim>>,
    },
    /// Döngü
    Dongu {
        /// Koşul
        kosul: Ifade,
        /// Gövde
        gövde: Vec<Deyim>,
    },
    /// İçin döngüsü
    IcinDongu {
        /// Değişken
        degisken: String,
        /// Dizi
        dizi: Ifade,
        /// Gövde
        gövde: Vec<Deyim>,
    },
    /// Döndür
    Dondur(Ifade),
    /// Fonksiyon tanımı
    FonksiyonTanimi {
        /// İsim
        isim: String,
        /// Parametreler
        parametreler: Vec<String>,
        /// Gövde
        gövde: Vec<Deyim>,
    },
}

/// Katman tipi
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KatmanTipi {
    /// Yoğun katman
    Dense,
    /// Evrişimli katman
    Conv2D,
    /// Maksimum havuzlama katmanı
    MaxPool2D,
    /// Düzleştirme katmanı
    Flatten,
}

/// Katman özellikleri
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KatmanOzellikleri {
    /// Katman tipi
    pub tip: KatmanTipi,
    /// Birim sayısı
    pub birimler: Option<i64>,
    /// Aktivasyon fonksiyonu
    pub aktivasyon: Option<String>,
    /// Kernel boyutu
    pub kernel_boyutu: Option<i64>,
    /// Stride
    pub strides: Option<i64>,
    /// Padding
    pub padding: Option<String>,
    /// Dropout oranı
    pub dropout: Option<f64>,
}

/// Aktivasyon
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Aktivasyon {
    /// ReLU
    ReLU,
    /// Sigmoid
    Sigmoid,
    /// Tanh
    Tanh,
    /// Softmax
    Softmax,
    /// LeakyReLU
    LeakyReLU(f64),
    /// Swish
    Swish,
    /// Hiçbiri
    Yok,
}

/// Optimimizer tipi
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizerTipi {
    /// SGD
    SGD {
        /// Öğrenme oranı
        lr: f64,
        /// Momentum
        momentum: Option<f64>,
        /// Nesterov
        nesterov: Option<bool>,
    },
    /// Adam
    Adam {
        /// Öğrenme oranı
        lr: f64,
        /// Beta1
        beta1: Option<f64>,
        /// Beta2
        beta2: Option<f64>,
        /// Epsilon
        epsilon: Option<f64>,
    },
    /// AdamW
    AdamW {
        /// Öğrenme oranı
        lr: f64,
        /// Beta1
        beta1: Option<f64>,
        /// Beta2
        beta2: Option<f64>,
        /// Epsilon
        epsilon: Option<f64>,
        /// Weight decay
        weight_decay: Option<f64>,
    },
    /// RMSProp
    RMSProp {
        /// Öğrenme oranı
        lr: f64,
        /// Rho
        rho: Option<f64>,
        /// Epsilon
        epsilon: Option<f64>,
    },
}

/// Kayıp fonksiyonu
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KayipFonksiyonu {
    /// MSE
    MSE,
    /// MAE
    MAE,
    /// Categorical Cross Entropy
    CrossEntropy,
    /// Binary Cross Entropy
    BinaryCrossEntropy,
    /// Huber Loss
    HuberLoss(f64),
}

/// Model özellikleri
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelOzellikleri {
    /// Katmanlar
    pub katmanlar: Vec<Katman>,
    /// Optimizer
    pub optimizer: OptimizerTipi,
    /// Kayıp fonksiyonu
    pub kayip_fonksiyonu: KayipFonksiyonu,
    /// Batch boyutu
    pub batch_boyutu: usize,
    /// Epoch sayısı
    pub epoch_sayisi: usize,
    /// Erken durdurma
    pub erken_durdurma: Option<bool>,
    /// Metrikler
    pub metrikler: Option<Vec<Metrik>>,
}

impl ModelOzellikleri {
    /// Yeni bir model özellikleri
    pub fn new() -> Self {
        Self {
            katmanlar: Vec::new(),
            kayip_fonksiyonu: KayipFonksiyonu::MSE,
            optimizer: OptimizerTipi::SGD {
                lr: 0.01,
                momentum: None,
                nesterov: None,
            },
            batch_boyutu: 32,
            epoch_sayisi: 10,
            erken_durdurma: None,
            metrikler: Some(vec![Metrik::Accuracy]),
        }
    }
}

/// Program
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    /// Deyimler
    pub deyimler: Vec<Deyim>,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Operator::Topla => write!(f, "+"),
            Operator::Cikar => write!(f, "-"),
            Operator::Carp => write!(f, "*"),
            Operator::Bol => write!(f, "/"),
            Operator::Esit => write!(f, "=="),
            Operator::EsitDegil => write!(f, "!="),
            Operator::Buyuk => write!(f, ">"),
            Operator::Kucuk => write!(f, "<"),
            Operator::BuyukEsit => write!(f, ">="),
            Operator::KucukEsit => write!(f, "<="),
            Operator::Ve => write!(f, "&&"),
            Operator::Veya => write!(f, "||"),
            Operator::Degil => write!(f, "!"),
            Operator::Artı => write!(f, "+"),
            Operator::Eksi => write!(f, "-"),
        }
    }
}

impl fmt::Display for Ifade {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ifade::Sayi(n) => write!(f, "{}", n),
            Ifade::Metin(s) => write!(f, "{}", s),
            Ifade::Degisken(s) => write!(f, "{}", s),
            Ifade::Binary { op, left, right } => write!(f, "({} {} {})", left, op, right),
            Ifade::Unary { op, expr } => write!(f, "({} {})", op, expr),
            Ifade::FonksiyonCagir { isim, argumanlar } => {
                write!(f, "{}(", isim)?;
                for (i, arg) in argumanlar.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Ifade::Dizi(elemanlar) => {
                write!(f, "[")?;
                for (i, eleman) in elemanlar.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", eleman)?;
                }
                write!(f, "]")
            }
            Ifade::Sozluk(elemanlar) => {
                write!(f, "{{")?;
                for (i, (key, value)) in elemanlar.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", key, value)?;
                }
                write!(f, "}}")
            }
            Ifade::Kosul { kosul, dogru: _, yanlis: _ } => write!(f, "if ({}) {{ ... }} else {{ ... }}", kosul),
            Ifade::Dongu { baslangic, kosul, adim, gövde: _ } => write!(f, "for ({}; {}; {}) {{ ... }}", baslangic, kosul, adim),
            Ifade::IcinDongu { degisken, dizi, gövde } => write!(f, "for ({}; {}; {}) {{ ... }}", degisken, dizi, gövde),
            Ifade::Dondur(ifade) => write!(f, "return {}", ifade),
            Ifade::Kes => write!(f, "break"),
            Ifade::Devam => write!(f, "continue"),
            Ifade::Bos => write!(f, "()"),
        }
    }
}

impl fmt::Display for KatmanTipi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KatmanTipi::Dense => write!(f, "Dense"),
            KatmanTipi::Conv2D => write!(f, "Conv2D"),
            KatmanTipi::MaxPool2D => write!(f, "MaxPool2D"),
            KatmanTipi::Flatten => write!(f, "Flatten"),
        }
    }
}

impl fmt::Display for KatmanOzellikleri {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {{", self.tip)?;
        if let Some(birimler) = self.birimler {
            write!(f, " birimler: {}", birimler)?;
        }
        if let Some(aktivasyon) = &self.aktivasyon {
            write!(f, ", aktivasyon: {}", aktivasyon)?;
        }
        if let Some(kernel) = self.kernel_boyutu {
            write!(f, ", kernel_boyutu: {}", kernel)?;
        }
        if let Some(strides) = self.strides {
            write!(f, ", strides: {}", strides)?;
        }
        if let Some(padding) = &self.padding {
            write!(f, ", padding: {}", padding)?;
        }
        if let Some(dropout) = self.dropout {
            write!(f, ", dropout: {}", dropout)?;
        }
        write!(f, " }}")
    }
}

impl fmt::Display for OptimizerTipi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OptimizerTipi::SGD { lr, momentum, nesterov } => 
                write!(f, "SGD(lr={}, momentum={:?}, nesterov={:?})", lr, momentum, nesterov),
            OptimizerTipi::Adam { lr, beta1, beta2, epsilon } => 
                write!(f, "Adam(lr={}, beta1={:?}, beta2={:?}, epsilon={:?})", lr, beta1, beta2, epsilon),
            OptimizerTipi::AdamW { lr, weight_decay, beta1, beta2, epsilon } => 
                write!(f, "AdamW(lr={}, weight_decay={:?}, beta1={:?}, beta2={:?}, epsilon={:?})", 
                       lr, weight_decay, beta1, beta2, epsilon),
            OptimizerTipi::RMSProp { lr, rho, epsilon } => 
                write!(f, "RMSProp(lr={}, rho={:?}, epsilon={:?})", lr, rho, epsilon),
        }
    }
}

impl fmt::Display for KayipFonksiyonu {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KayipFonksiyonu::MSE => write!(f, "mse"),
            KayipFonksiyonu::MAE => write!(f, "mae"),
            KayipFonksiyonu::CrossEntropy => write!(f, "cross_entropy"),
            KayipFonksiyonu::BinaryCrossEntropy => write!(f, "binary_cross_entropy"),
            KayipFonksiyonu::HuberLoss(delta) => write!(f, "huber_loss({})", delta),
        }
    }
}

/// Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub ozellikler: ModelOzellikleri,
}

impl Model {
    /// Yeni bir model
    pub fn new(ozellikler: ModelOzellikleri) -> Self {
        Self {
            ozellikler,
        }
    }
    
    /// Katman ekle
    pub fn katman_ekle(&mut self, katman: Katman) {
        self.ozellikler.katmanlar.push(katman);
    }
    
    /// Modeli eğitir
    pub fn egit(
        &self,
        _x: &ArrayView2<f64>,
        _y: &ArrayView2<f64>,
        _epochs: usize,
        _batch_size: usize,
        _opt: Option<&str>
    ) -> Result<Vec<(f64, Option<f64>)>, TurkError> {
        // Eğitim implementasyonu
        Ok(Vec::new())
    }
    
    /// Tahmin
    pub fn tahmin(&self, _x: &ArrayView2<f64>) -> Result<Array2<f64>, TurkError> {
        // Tahmin implementasyonu
        Ok(Array2::zeros((1, 1)))
    }
    
    /// Değerlendirme
    pub fn degerlendir(
        &self,
        _x: &ArrayView2<f64>,
        _y: &ArrayView2<f64>,
    ) -> Result<f64, TurkError> {
        // Değerlendirme implementasyonu
        Ok(0.0)
    }
}

/// Katman
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Katman {
    /// Yoğun katman
    Dense {
        /// Birimler
        birimler: usize,
        /// Aktivasyon
        aktivasyon: Aktivasyon,
        /// Kernel initializer
        kernel_initializer: Option<String>,
        /// Bias initializer
        bias_initializer: Option<String>,
    },
    /// Evrişimli katman
    Conv2D {
        /// Filtreler
        filtreler: usize,
        /// Kernel boyutu
        kernel_boyutu: (usize, usize),
        /// Adımlar
        strides: (usize, usize),
        /// Padding
        padding: String,
        /// Aktivasyon
        aktivasyon: Aktivasyon,
    },
    /// Maksimum havuzlama katmanı
    MaxPool2D {
        /// Havuzlama boyutu
        havuzlama_boyutu: (usize, usize),
        /// Adımlar
        strides: Option<(usize, usize)>,
        /// Padding
        padding: String,
    },
    /// Düzleştirme katmanı
    Flatten,
    /// Dropout katmanı
    Dropout {
        /// Oranı
        orani: f64,
    },
    /// Batch normalizasyon katmanı
    BatchNormalization {
        /// Momentum
        momentum: f64,
        /// Epsilon
        epsilon: f64,
    },
}

/// Metrik
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Metrik {
    /// Doğruluk
    Accuracy,
    /// Precision
    Precision,
    /// Recall
    Recall,
    /// F1 Score
    F1Score,
    /// AUC
    AUC,
}

/// Model Sayı
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSayi {
    /// Optimizer
    pub optimizer: OptimizerTipi,
    /// Kayıp fonksiyonu
    pub kayip_fonksiyonu: KayipFonksiyonu,
    /// Metrikler
    #[serde(default)]
    pub metrikler: Option<Vec<Metrik>>,
}

impl ModelSayi {
    /// Yeni bir model sayısı
    pub fn new() -> Self {
        Self {
            kayip_fonksiyonu: KayipFonksiyonu::MSE,
            optimizer: OptimizerTipi::SGD {
                lr: 0.01,
                momentum: None,
                nesterov: None,
            },
            metrikler: Some(vec![Metrik::Accuracy]),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Statement {
    Model(ModelStatement),
    Function(FunctionStatement),
    If(IfStatement),
    While(WhileStatement),
    Return(ReturnStatement),
    Train(TrainStatement),
    Predict(PredictStatement),
    Expression(ExpressionStatement),
    Block(BlockStatement),
}

#[derive(Debug, Clone)]
pub struct ModelStatement {
    pub name: String,
    pub layers: Vec<LayerStatement>,
}

#[derive(Debug, Clone)]
pub struct LayerStatement {
    pub layer_type: String,
    pub properties: HashMap<String, Expression>,
}

#[derive(Debug, Clone)]
pub struct FunctionStatement {
    pub name: String,
    pub parameters: Vec<Token>,
    pub body: BlockStatement,
}

#[derive(Debug, Clone)]
pub struct IfStatement {
    pub condition: Expression,
    pub then_branch: Box<Statement>,
    pub else_branch: Option<Box<Statement>>,
}

#[derive(Debug, Clone)]
pub struct WhileStatement {
    pub condition: Expression,
    pub body: Box<Statement>,
}

#[derive(Debug, Clone)]
pub struct ReturnStatement {
    pub keyword: Token,
    pub value: Expression,
}

#[derive(Debug, Clone)]
pub struct TrainStatement {
    pub model_name: String,
    pub properties: HashMap<String, Expression>,
}

#[derive(Debug, Clone)]
pub struct PredictStatement {
    pub model_name: String,
    pub input: Expression,
}

#[derive(Debug, Clone)]
pub struct ExpressionStatement {
    pub expression: Expression,
}

#[derive(Debug, Clone)]
pub struct BlockStatement {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub enum Expression {
    Assign(AssignExpression),
    Binary(BinaryExpression),
    Unary(UnaryExpression),
    Logical(LogicalExpression),
    Grouping(GroupingExpression),
    Literal(Literal),
    Variable(String),
}

#[derive(Debug, Clone)]
pub struct AssignExpression {
    pub name: String,
    pub value: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct BinaryExpression {
    pub left: Box<Expression>,
    pub operator: Token,
    pub right: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct UnaryExpression {
    pub operator: Token,
    pub right: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct LogicalExpression {
    pub left: Box<Expression>,
    pub operator: Token,
    pub right: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct GroupingExpression {
    pub expression: Box<Expression>,
}

#[derive(Debug, Clone)]
pub enum Literal {
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
}

impl Expression {
    pub fn accept<T>(&self, visitor: &mut dyn ExpressionVisitor<T>) -> T {
        match self {
            Expression::Assign(expr) => visitor.visit_assign(expr),
            Expression::Binary(expr) => visitor.visit_binary(expr),
            Expression::Unary(expr) => visitor.visit_unary(expr),
            Expression::Logical(expr) => visitor.visit_logical(expr),
            Expression::Grouping(expr) => visitor.visit_grouping(expr),
            Expression::Literal(lit) => visitor.visit_literal(lit),
            Expression::Variable(name) => visitor.visit_variable(name),
        }
    }
}

pub trait ExpressionVisitor<T> {
    fn visit_assign(&mut self, expr: &AssignExpression) -> T;
    fn visit_binary(&mut self, expr: &BinaryExpression) -> T;
    fn visit_unary(&mut self, expr: &UnaryExpression) -> T;
    fn visit_logical(&mut self, expr: &LogicalExpression) -> T;
    fn visit_grouping(&mut self, expr: &GroupingExpression) -> T;
    fn visit_literal(&mut self, lit: &Literal) -> T;
    fn visit_variable(&mut self, name: &String) -> T;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_expression_visitor() {
        struct TestVisitor;
        
        impl ExpressionVisitor<String> for TestVisitor {
            fn visit_assign(&mut self, expr: &AssignExpression) -> String {
                format!("{} = {}", expr.name, expr.value.accept(self))
            }
            
            fn visit_binary(&mut self, expr: &BinaryExpression) -> String {
                format!(
                    "({} {} {})",
                    expr.left.accept(self),
                    expr.operator,
                    expr.right.accept(self)
                )
            }
            
            fn visit_unary(&mut self, expr: &UnaryExpression) -> String {
                format!("({}{})", expr.operator, expr.right.accept(self))
            }
            
            fn visit_logical(&mut self, expr: &LogicalExpression) -> String {
                format!(
                    "({} {} {})",
                    expr.left.accept(self),
                    expr.operator,
                    expr.right.accept(self)
                )
            }
            
            fn visit_grouping(&mut self, expr: &GroupingExpression) -> String {
                format!("({})", expr.expression.accept(self))
            }
            
            fn visit_literal(&mut self, lit: &Literal) -> String {
                match lit {
                    Literal::Number(n) => n.to_string(),
                    Literal::String(s) => s.clone(),
                    Literal::Boolean(b) => b.to_string(),
                    Literal::Nil => "nil".to_string(),
                }
            }
            
            fn visit_variable(&mut self, name: &String) -> String {
                name.clone()
            }
        }
        
        let expr = Expression::Binary(BinaryExpression {
            left: Box::new(Expression::Literal(Literal::Number(1.0))),
            operator: Token::Add,
            right: Box::new(Expression::Literal(Literal::Number(2.0))),
        });
        
        let mut visitor = TestVisitor;
        assert_eq!(expr.accept(&mut visitor), "(1 + 2)");
    }
} 
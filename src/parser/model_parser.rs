use crate::ast::{Deyim, ModelOzellikleri, KatmanOzellikleri, Program, AktivasyonFonksiyonu, OptimizerTipi, KayipFonksiyonu};
use crate::token::Token;
use super::error::{ParseError, Result};
use super::expr_parser::ExprParser;

pub struct ModelParser {
    tokens: Vec<(Token, String)>,
    current: usize,
    expr_parser: ExprParser,
}

impl ModelParser {
    pub fn new(tokens: Vec<(Token, String)>) -> Self {
        let expr_parser = ExprParser::new(tokens.clone());
        Self { 
            tokens, 
            current: 0,
            expr_parser,
        }
    }
    
    pub fn simdiki(&self) -> Option<&(Token, String)> {
        self.tokens.get(self.current)
    }
    
    pub fn ilerle(&mut self) -> Option<&(Token, String)> {
        if self.current < self.tokens.len() {
            self.current += 1;
        }
        self.tokens.get(self.current - 1)
    }
    
    pub fn parse(&mut self) -> Result<Program> {
        let mut deyimler = Vec::new();
        
        while self.simdiki().is_some() {
            if let Some(deyim) = self.deyim_parse() {
                deyimler.push(deyim);
            } else {
                return Err(ParseError::SyntaxError(
                    "Geçersiz deyim yapısı.".to_string()
                ));
            }
        }
        
        Ok(Program { deyimler })
    }
    
    pub fn deyim_parse(&mut self) -> Option<Deyim> {
        if let Some((token, _)) = self.simdiki() {
            match token {
                Token::Model => self.model_parse(),
                Token::Veri => self.veri_parse(),
                Token::Egit => self.egit_parse(),
                Token::Tahmin => self.tahmin_parse(),
                Token::Ortalama | Token::Medyan | Token::ModFonksiyon |
                Token::StandartSapma | Token::Varyans => self.expr_parser.istatistiksel_islem_parse(),
                _ => None,
            }
        } else {
            None
        }
    }
    
    pub fn model_parse(&mut self) -> Option<Deyim> {
        self.ilerle(); // "Model" token'ı tüket
        
        let isim = if let Some((Token::Degisken(isim), _)) = self.simdiki() {
            self.ilerle();
            isim
        } else {
            return None;
        };
        
        if self.bekle(Token::AcKume).is_err() {
            return None;
        }
        
        let mut katmanlar = Vec::new();
        let mut optimizer = None;
        let mut kayip_fonksiyonu = None;
        let mut batch_boyutu = None;
        let mut epoch_sayisi = None;
        
        loop {
            if let Some((token, _)) = self.simdiki() {
                match token {
                    Token::KapaKume => {
                        self.ilerle();
                        break;
                    }
                    Token::Katman => {
                        if self.bekle(Token::Katman).is_err() {
                            return None;
                        }
                        
                        if self.bekle(Token::AcParantez).is_err() {
                            return None;
                        }
                        
                        // Nöron sayısını al
                        if let Some((Token::Sayi(noron_sayisi), _)) = self.simdiki() {
                            self.ilerle();
                            
                            // İsteğe bağlı aktivasyon fonksiyonu
                            let mut aktivasyon = AktivasyonFonksiyonu::Relu; // Varsayılan
                            
                            // Virgülden sonra aktivasyon fonksiyonu varsa
                            if let Ok(()) = self.bekle(Token::Virgul) {
                                // Aktivasyon tipini al
                                if let Some((token, _)) = self.simdiki() {
                                    match token {
                                        Token::Sigmoid => aktivasyon = AktivasyonFonksiyonu::Sigmoid,
                                        Token::Relu => aktivasyon = AktivasyonFonksiyonu::Relu,
                                        Token::Tanh => aktivasyon = AktivasyonFonksiyonu::Tanh,
                                        Token::Softmax => aktivasyon = AktivasyonFonksiyonu::Softmax,
                                        _ => return None,
                                    }
                                    self.ilerle();
                                }
                            }
                            
                            if self.bekle(Token::KapaParantez).is_err() {
                                return None;
                            }
                            
                            katmanlar.push(KatmanOzellikleri {
                                noron_sayisi,
                                aktivasyon,
                                kernel_initializer: None,
                                bias_initializer: None,
                                dropout_rate: None,
                            });
                        } else {
                            return None;
                        }
                    }
                    Token::OgrenmeOrani => {
                        if self.bekle(Token::OgrenmeOrani).is_err() {
                            return None;
                        }
                        
                        if self.bekle(Token::Esittir).is_err() {
                            return None;
                        }
                        
                        if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                            self.ilerle();
                            optimizer = Some(OptimizerTipi::SGD(oran as f64));
                        } else {
                            return None;
                        }
                    }
                    Token::Kayip => {
                        if self.bekle(Token::Kayip).is_err() {
                            return None;
                        }
                        
                        if self.bekle(Token::Esittir).is_err() {
                            return None;
                        }
                        
                        if let Some((token, _)) = self.simdiki() {
                            match token {
                                Token::Degisken(tip) if tip == "CrossEntropy" => {
                                    self.ilerle();
                                    kayip_fonksiyonu = Some(KayipFonksiyonu::CrossEntropy);
                                }
                                Token::Degisken(tip) if tip == "MSE" => {
                                    self.ilerle();
                                    kayip_fonksiyonu = Some(KayipFonksiyonu::MSE);
                                }
                                _ => return None,
                            }
                        } else {
                            return None;
                        }
                    }
                    Token::Batch => {
                        if self.bekle(Token::Batch).is_err() {
                            return None;
                        }
                        
                        if self.bekle(Token::Esittir).is_err() {
                            return None;
                        }
                        
                        if let Some((Token::Sayi(boyut), _)) = self.simdiki() {
                            self.ilerle();
                            batch_boyutu = Some(boyut);
                        } else {
                            return None;
                        }
                    }
                    Token::Epoch => {
                        if self.bekle(Token::Epoch).is_err() {
                            return None;
                        }
                        
                        if self.bekle(Token::Esittir).is_err() {
                            return None;
                        }
                        
                        if let Some((Token::Sayi(sayi), _)) = self.simdiki() {
                            self.ilerle();
                            epoch_sayisi = Some(sayi);
                        } else {
                            return None;
                        }
                    }
                    _ => {
                        self.ilerle(); // Bilinmeyen token'ı atla
                    }
                }
            } else {
                break;
            }
        }
        
        // Varsayılan değerler
        let optimizer = optimizer.unwrap_or(OptimizerTipi::SGD(0.01));
        let kayip_fonksiyonu = kayip_fonksiyonu.unwrap_or(KayipFonksiyonu::MSE);
        let batch_boyutu = batch_boyutu.unwrap_or(32);
        let epoch_sayisi = epoch_sayisi.unwrap_or(10);
        
        if katmanlar.is_empty() {
            return None;
        }
        
        Some(Deyim::ModelTanimi {
            isim,
            ozellikler: ModelOzellikleri {
                katmanlar,
                optimizer,
                kayip_fonksiyonu,
                batch_boyutu,
                epoch_sayisi,
            },
        })
    }
    
    fn veri_parse(&mut self) -> Option<Deyim> {
        self.ilerle(); // "Veri" token'ı tüket
        
        let isim = if let Some((Token::Degisken(isim), _)) = self.simdiki() {
            self.ilerle();
            isim
        } else {
            return None;
        };
        
        if self.bekle(Token::Esittir).is_err() {
            return None;
        }
        
        let deger = self.expr_parser.ifade_parse()?;
        
        Some(Deyim::VeriTanimi {
            isim,
            deger: Box::new(deger),
        })
    }
    
    fn egit_parse(&mut self) -> Option<Deyim> {
        self.ilerle(); // "Egit" token'ı tüket
        
        if self.bekle(Token::AcParantez).is_err() {
            return None;
        }
        
        let model = if let Some((Token::Degisken(model), _)) = self.simdiki() {
            self.ilerle();
            model
        } else {
            return None;
        };
        
        let mut x_train = None;
        let mut y_train = None;
        
        while self.simdiki().is_some() && self.simdiki().unwrap().0 != Token::KapaParantez {
            if let Some((Token::Virgul, _)) = self.simdiki() {
                self.ilerle();
            }
            
            if let Some((Token::Degisken(param_isim), _)) = self.simdiki() {
                self.ilerle();
                
                if self.bekle(Token::Esittir).is_err() {
                    return None;
                }
                
                let deger = self.expr_parser.ifade_parse()?;
                
                match param_isim.as_str() {
                    "x" => x_train = Some(deger),
                    "y" => y_train = Some(deger),
                    _ => {}
                }
            } else {
                break;
            }
        }
        
        if self.bekle(Token::KapaParantez).is_err() {
            return None;
        }
        
        let x_train = x_train?;
        let y_train = y_train?;
        
        Some(Deyim::Egitim {
            model,
            x_train: Box::new(x_train),
            y_train: Box::new(y_train),
        })
    }
    
    fn tahmin_parse(&mut self) -> Option<Deyim> {
        self.ilerle(); // "Tahmin" token'ı tüket
        
        if self.bekle(Token::AcParantez).is_err() {
            return None;
        }
        
        let model = if let Some((Token::Degisken(model), _)) = self.simdiki() {
            self.ilerle();
            model
        } else {
            return None;
        };
        
        let mut x_test = None;
        
        while self.simdiki().is_some() && self.simdiki().unwrap().0 != Token::KapaParantez {
            if let Some((Token::Virgul, _)) = self.simdiki() {
                self.ilerle();
            }
            
            if let Some((Token::Degisken(param_isim), _)) = self.simdiki() {
                self.ilerle();
                
                if self.bekle(Token::Esittir).is_err() {
                    return None;
                }
                
                let deger = self.expr_parser.ifade_parse()?;
                
                if param_isim == "x" {
                    x_test = Some(deger);
                }
            } else {
                break;
            }
        }
        
        if self.bekle(Token::KapaParantez).is_err() {
            return None;
        }
        
        let x_test = x_test?;
        
        Some(Deyim::Tahmin {
            model,
            x_test: Box::new(x_test),
        })
    }
    
    fn bekle(&mut self, beklenen: Token) -> Result<()> {
        if let Some((token, _)) = self.simdiki() {
            if token == &beklenen {
                self.ilerle();
                return Ok(());
            }
        }
        Err(ParseError::MissingToken(format!("{:?} bekleniyor", beklenen)))
    }
}

impl super::Parser for ModelParser {
    fn parse(&mut self) -> Result<Program> {
        self.parse()
    }
} 
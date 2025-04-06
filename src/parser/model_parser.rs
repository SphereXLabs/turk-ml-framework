use crate::ast::{Deyim, ModelOzellikleri, KatmanOzellikleri, Program, AktivasyonFonksiyonu, OptimizerTipi, KayipFonksiyonu, KatmanTipi};
use crate::token::Token;
use super::error::{ParseError, Result};
use super::expr_parser::ExprParser;
use super::Parser;

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
            match self.deyim_parse() {
                Ok(Some(deyim)) => deyimler.push(deyim),
                Ok(None) => break, // Deyim ayrıştırılamadıysa veya beklenmeyen token ise
                Err(e) => return Err(e), // Hata durumunda hemen çık
            }
        }
        
        Ok(Program { deyimler })
    }
    
    pub fn deyim_parse(&mut self) -> Result<Option<Deyim>> {
        if let Some((token, _)) = self.simdiki() {
            match token {
                Token::Model => self.model_parse(),
                Token::Veri => self.veri_parse(),
                Token::Egit => self.egit_parse(),
                Token::Tahmin => self.tahmin_parse(),
                Token::Ortalama | Token::Medyan | Token::ModFonksiyon |
                Token::StandartSapma | Token::Varyans => {
                    match self.expr_parser.istatistiksel_islem_parse() {
                        Ok(Some(deyim)) => Ok(Some(deyim)),
                        Ok(None) => Err(ParseError::syntax_error("İstatistiksel işlem ayrıştırılamadı (beklenmedik son).".to_string())),
                        Err(e) => Err(e),
                    }
                }
                // TODO: Diğer deyim türleri (if, for, etc.) buraya eklenebilir
                _ => Err(ParseError::unexpected_token(
                    token.clone(), 
                    "beklenen bir deyim başlangıcı".to_string()
                )),
            }
        } else {
            Ok(None) // Token kalmadıysa None döndür
        }
    }
    
    pub fn model_parse(&mut self) -> Result<Option<Deyim>> {
        self.ilerle(); // "Model" token'ı tüket
        
        let isim = if let Some((Token::Degisken(isim), _)) = self.simdiki() {
            let isim_str = isim.clone(); // Klonla ve ilerle
            self.ilerle();
            isim_str
        } else {
            return Err(ParseError::unexpected_token(
                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                 "model ismi".to_string()
            ));
        };
        
        self.expect(Token::AcKume)?;
        
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
                        self.ilerle(); // "Katman" tüket
                        self.expect(Token::AcParantez)?;
                        
                        // Nöron sayısını al (* ile dereference)
                        let noron_sayisi = if let Some((Token::Sayi(n), _)) = self.simdiki() {
                            let n_val = *n; // Dereference
                            self.ilerle();
                            n_val
                        } else {
                            return Err(ParseError::unexpected_token(
                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                "nöron sayısı (i64)".to_string()
                            ));
                        };
                            
                        // İsteğe bağlı aktivasyon fonksiyonu
                        let mut aktivasyon = AktivasyonFonksiyonu::Relu; // Varsayılan
                        
                        // Virgülden sonra aktivasyon fonksiyonu varsa
                        if self.check(&Token::Virgul) {
                            self.ilerle(); // Virgülü tüket
                            
                            // "Aktivasyon = Relu" formatını kontrol et
                            if self.check(&Token::Aktivasyon) {
                                self.ilerle(); // "Aktivasyon" token'ını tüket
                                self.expect(Token::Esittir)?; // "=" işaretini tüket
                                
                                // Aktivasyon tipini al
                                if let Some((token, _)) = self.simdiki() {
                                    match token {
                                        Token::Sigmoid => aktivasyon = AktivasyonFonksiyonu::Sigmoid,
                                        Token::Relu => aktivasyon = AktivasyonFonksiyonu::Relu,
                                        Token::Tanh => aktivasyon = AktivasyonFonksiyonu::Tanh,
                                        Token::Softmax => aktivasyon = AktivasyonFonksiyonu::Softmax,
                                        _ => return Err(ParseError::unexpected_token(
                                            token.clone(), 
                                            "geçerli bir aktivasyon fonksiyonu (Relu, Sigmoid vb.)".to_string()
                                        )),
                                    }
                                    self.ilerle(); // Aktivasyon token'ını tüket
                                } else {
                                    return Err(ParseError::UnexpectedEOF);
                                }
                            } else {
                                // Doğrudan aktivasyon fonksiyonu geldiyse (eski format)
                                if let Some((token, _)) = self.simdiki() {
                                    match token {
                                        Token::Sigmoid => aktivasyon = AktivasyonFonksiyonu::Sigmoid,
                                        Token::Relu => aktivasyon = AktivasyonFonksiyonu::Relu,
                                        Token::Tanh => aktivasyon = AktivasyonFonksiyonu::Tanh,
                                        Token::Softmax => aktivasyon = AktivasyonFonksiyonu::Softmax,
                                        _ => return Err(ParseError::unexpected_token(
                                            token.clone(), 
                                            "geçerli bir aktivasyon fonksiyonu (Relu, Sigmoid vb.)".to_string()
                                        )),
                                    }
                                    self.ilerle(); // Aktivasyon token'ını tüket
                                } else {
                                    return Err(ParseError::UnexpectedEOF);
                                }
                            }
                        }
                        
                        self.expect(Token::KapaParantez)?;
                        
                        katmanlar.push(KatmanOzellikleri {
                            katman_tipi: KatmanTipi::Yogun,
                            noron_sayisi, // Dereference edilmiş değer
                            aktivasyon,
                            kernel_initializer: None,
                            bias_initializer: None,
                            dropout_rate: None,
                        });
                    }
                    Token::OgrenmeOrani => {
                        self.ilerle(); // "OgrenmeOrani" tüket
                        self.expect(Token::Esittir)?;
                        
                        // Virgüllerle ondalık sayı desteği
                        if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                            let oran_val = *oran as f64; // Dereference ve cast
                            self.ilerle();
                            
                            // Virgül ve ikinci sayı varsa (0,1 formatını desteklemek için)
                            if self.check(&Token::Virgul) {
                                self.ilerle(); // Virgülü tüket
                                
                                if let Some((Token::Sayi(ondalik), _)) = self.simdiki() {
                                    let ondalik_str = format!("0.{}", ondalik);
                                    let ondalik_val = ondalik_str.parse::<f64>().unwrap_or(0.0);
                                    self.ilerle();
                                    
                                    optimizer = Some(OptimizerTipi::SGD {
                                        lr: oran_val + ondalik_val,
                                        momentum: None,
                                        nesterov: None,
                                    });
                                } else {
                                    return Err(ParseError::unexpected_token(
                                        self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                        "virgülden sonra ondalık kısım".to_string()
                                    ));
                                }
                            } else {
                                // Tam sayı öğrenme oranı
                                optimizer = Some(OptimizerTipi::SGD {
                                    lr: oran_val,
                                    momentum: None,
                                    nesterov: None,
                                });
                            }
                        } else if let Some((Token::OndalikSayi(oran), _)) = self.simdiki() {
                            // OndalikSayi token tipi ile doğrudan ondalık sayı alma
                            let oran_val = *oran as f64 / 100.0;
                            self.ilerle();
                            
                            optimizer = Some(OptimizerTipi::SGD {
                                lr: oran_val,
                                momentum: None,
                                nesterov: None,
                            });
                        } else {
                            return Err(ParseError::unexpected_token(
                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                "öğrenme oranı".to_string()
                            ));
                        }
                    }
                    Token::Optimizer => {
                        self.ilerle(); // "Optimizer" tüket
                        self.expect(Token::Esittir)?;

                        // Optimizer tipini kontrol et
                        if let Some((token, _)) = self.simdiki() {
                            match token {
                                Token::SGD => {
                                    self.ilerle();
                                    // Parametreleri kontrol et
                                    let mut lr = 0.01; // Varsayılan değer
                                    let mut momentum = None;
                                    let mut nesterov = None;
                                    
                                    if self.check(&Token::AcParantez) {
                                        self.ilerle();
                                        
                                        // Parametreleri ayrıştır
                                        while !self.check(&Token::KapaParantez) {
                                            if let Some((Token::Degisken(param_name), _)) = self.simdiki() {
                                                let param = param_name.clone();
                                                self.ilerle();
                                                
                                                self.expect(Token::Esittir)?;
                                                
                                                match param.as_str() {
                                                    "ogrenme_orani" | "lr" => {
                                                        if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                                                            lr = *oran as f64;
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(oran), _)) = self.simdiki() {
                                                            lr = *oran as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "öğrenme oranı değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    "momentum" => {
                                                        if let Some((Token::Sayi(mom), _)) = self.simdiki() {
                                                            momentum = Some(*mom as f64 / 10.0); // 9 -> 0.9
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(mom), _)) = self.simdiki() {
                                                            momentum = Some(*mom as f64 / 100.0);
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "momentum değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    "nesterov" => {
                                                        if let Some((token, _)) = self.simdiki() {
                                                            match token {
                                                                Token::Dogru => nesterov = Some(true),
                                                                Token::Yanlis => nesterov = Some(false),
                                                                _ => return Err(ParseError::unexpected_token(
                                                                    token.clone(),
                                                                    "Dogru veya Yanlis".to_string()
                                                                )),
                                                            }
                                                            self.ilerle();
                                                        }
                                                    },
                                                    _ => return Err(ParseError::unexpected_token(
                                                        Token::Degisken(param),
                                                        "geçerli bir SGD parametresi".to_string()
                                                    )),
                                                }
                                                
                                                // Virgül kontrolü
                                                if self.check(&Token::Virgul) {
                                                    self.ilerle();
                                                }
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "parametre adı".to_string()
                                                ));
                                            }
                                        }
                                        
                                        self.ilerle(); // KapaParantez token'ını tüket
                                    }
                                    
                                    optimizer = Some(OptimizerTipi::SGD {
                                        lr,
                                        momentum,
                                        nesterov,
                                    });
                                },
                                Token::Adam => {
                                    self.ilerle();
                                    // Adam parametrelerini kontrol et
                                    let mut lr = 0.001; // Varsayılan değer
                                    let mut beta1 = 0.9; // Varsayılan
                                    let mut beta2 = 0.999; // Varsayılan
                                    let mut epsilon = 1e-8; // Varsayılan
                                    
                                    if self.check(&Token::AcParantez) {
                                        self.ilerle();
                                        
                                        // Parametreleri ayrıştır
                                        while !self.check(&Token::KapaParantez) {
                                            if let Some((Token::Degisken(param_name), _)) = self.simdiki() {
                                                let param = param_name.clone();
                                                self.ilerle();
                                                
                                                self.expect(Token::Esittir)?;
                                                
                                                match param.as_str() {
                                                    "ogrenme_orani" | "lr" => {
                                                        if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                                                            lr = *oran as f64;
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(oran), _)) = self.simdiki() {
                                                            lr = *oran as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "öğrenme oranı değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    "beta1" => {
                                                        if let Some((Token::Sayi(b), _)) = self.simdiki() {
                                                            beta1 = *b as f64 / 10.0; // 9 -> 0.9
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(b), _)) = self.simdiki() {
                                                            beta1 = *b as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "beta1 değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    "beta2" => {
                                                        if let Some((Token::Sayi(b), _)) = self.simdiki() {
                                                            beta2 = *b as f64 / 10.0; // 9 -> 0.9
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(b), _)) = self.simdiki() {
                                                            beta2 = *b as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "beta2 değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    "epsilon" => {
                                                        if let Some((Token::Sayi(e), _)) = self.simdiki() {
                                                            epsilon = 1.0 / (10.0_f64.powi(*e as i32)); // 8 -> 1e-8
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(e), _)) = self.simdiki() {
                                                            epsilon = *e as f64 / 1000.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "epsilon değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    _ => return Err(ParseError::unexpected_token(
                                                        Token::Degisken(param),
                                                        "geçerli bir Adam parametresi".to_string()
                                                    )),
                                                }
                                                
                                                // Virgül kontrolü
                                                if self.check(&Token::Virgul) {
                                                    self.ilerle();
                                                }
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "parametre adı".to_string()
                                                ));
                                            }
                                        }
                                        
                                        self.ilerle(); // KapaParantez token'ını tüket
                                    }
                                    
                                    optimizer = Some(OptimizerTipi::Adam {
                                        lr,
                                        beta1,
                                        beta2,
                                        epsilon,
                                    });
                                },
                                Token::AdamW => {
                                    self.ilerle();
                                    // AdamW parametrelerini kontrol et
                                    let mut lr = 0.001; // Varsayılan değer
                                    let mut weight_decay = 0.01; // Varsayılan
                                    let mut beta1 = 0.9; // Varsayılan
                                    let mut beta2 = 0.999; // Varsayılan
                                    
                                    if self.check(&Token::AcParantez) {
                                        self.ilerle();
                                        
                                        // Parametreleri ayrıştır
                                        while !self.check(&Token::KapaParantez) {
                                            if let Some((Token::Degisken(param_name), _)) = self.simdiki() {
                                                let param = param_name.clone();
                                                self.ilerle();
                                                
                                                self.expect(Token::Esittir)?;
                                                
                                                match param.as_str() {
                                                    "ogrenme_orani" | "lr" => {
                                                        if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                                                            lr = *oran as f64;
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(oran), _)) = self.simdiki() {
                                                            lr = *oran as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "öğrenme oranı değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    "weight_decay" | "agirlik_azaltma" => {
                                                        if let Some((Token::Sayi(wd), _)) = self.simdiki() {
                                                            weight_decay = *wd as f64;
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(wd), _)) = self.simdiki() {
                                                            weight_decay = *wd as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "weight_decay değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    "beta1" => {
                                                        if let Some((Token::Sayi(b), _)) = self.simdiki() {
                                                            beta1 = *b as f64 / 10.0; // 9 -> 0.9
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(b), _)) = self.simdiki() {
                                                            beta1 = *b as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "beta1 değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    "beta2" => {
                                                        if let Some((Token::Sayi(b), _)) = self.simdiki() {
                                                            beta2 = *b as f64 / 10.0; // 9 -> 0.9
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(b), _)) = self.simdiki() {
                                                            beta2 = *b as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "beta2 değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    _ => return Err(ParseError::unexpected_token(
                                                        Token::Degisken(param),
                                                        "geçerli bir AdamW parametresi".to_string()
                                                    )),
                                                }
                                                
                                                // Virgül kontrolü
                                                if self.check(&Token::Virgul) {
                                                    self.ilerle();
                                                }
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "parametre adı".to_string()
                                                ));
                                            }
                                        }
                                        
                                        self.ilerle(); // KapaParantez token'ını tüket
                                    }
                                    
                                    optimizer = Some(OptimizerTipi::AdamW {
                                        lr,
                                        weight_decay,
                                        beta1,
                                        beta2,
                                    });
                                },
                                Token::RMSProp => {
                                    self.ilerle();
                                    // RMSProp parametrelerini kontrol et
                                    let mut lr = 0.001; // Varsayılan değer
                                    let mut rho = 0.9; // Varsayılan
                                    let mut epsilon = 1e-8; // Varsayılan
                                    
                                    if self.check(&Token::AcParantez) {
                                        self.ilerle();
                                        
                                        // Parametreleri ayrıştır
                                        while !self.check(&Token::KapaParantez) {
                                            if let Some((Token::Degisken(param_name), _)) = self.simdiki() {
                                                let param = param_name.clone();
                                                self.ilerle();
                                                
                                                self.expect(Token::Esittir)?;
                                                
                                                match param.as_str() {
                                                    "ogrenme_orani" | "lr" => {
                                                        if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                                                            lr = *oran as f64;
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(oran), _)) = self.simdiki() {
                                                            lr = *oran as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "öğrenme oranı değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    "rho" | "bozulma_orani" => {
                                                        if let Some((Token::Sayi(r), _)) = self.simdiki() {
                                                            rho = *r as f64 / 10.0; // 9 -> 0.9
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(r), _)) = self.simdiki() {
                                                            rho = *r as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "rho değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    "epsilon" => {
                                                        if let Some((Token::Sayi(e), _)) = self.simdiki() {
                                                            epsilon = 1.0 / (10.0_f64.powi(*e as i32)); // 8 -> 1e-8
                                                            self.ilerle();
                                                        } else if let Some((Token::OndalikSayi(e), _)) = self.simdiki() {
                                                            epsilon = *e as f64 / 100.0;
                                                            self.ilerle();
                                                        } else {
                                                            return Err(ParseError::unexpected_token(
                                                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                                "epsilon değeri".to_string()
                                                            ));
                                                        }
                                                    },
                                                    _ => return Err(ParseError::unexpected_token(
                                                        Token::Degisken(param),
                                                        "geçerli bir RMSProp parametresi".to_string()
                                                    )),
                                                }
                                                
                                                // Virgül kontrolü
                                                if self.check(&Token::Virgul) {
                                                    self.ilerle();
                                                }
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "parametre adı".to_string()
                                                ));
                                            }
                                        }
                                        
                                        self.ilerle(); // KapaParantez token'ını tüket
                                    }
                                    
                                    optimizer = Some(OptimizerTipi::RMSProp {
                                        lr,
                                        rho,
                                        epsilon,
                                    });
                                },
                                Token::Metin(optimizer_name) => {
                                    let optimizer_name_clone = optimizer_name.clone();
                                    self.ilerle();
                                    // Metin olarak verilen optimizer adını analiz et (geriye uyumluluk için)
                                    match optimizer_name_clone.as_str() {
                                        "sgd" => {
                                            optimizer = Some(OptimizerTipi::SGD {
                                                lr: 0.01,
                                                momentum: None,
                                                nesterov: None,
                                            });
                                        },
                                        "adam" => {
                                            optimizer = Some(OptimizerTipi::Adam {
                                                lr: 0.001,
                                                beta1: 0.9,
                                                beta2: 0.999,
                                                epsilon: 1e-8,
                                            });
                                        },
                                        "adamw" => {
                                            optimizer = Some(OptimizerTipi::AdamW {
                                                lr: 0.001,
                                                weight_decay: 0.01,
                                                beta1: 0.9,
                                                beta2: 0.999,
                                            });
                                        },
                                        "rmsprop" => {
                                            optimizer = Some(OptimizerTipi::RMSProp {
                                                lr: 0.001,
                                                rho: 0.9,
                                                epsilon: 1e-8,
                                            });
                                        },
                                        _ => return Err(ParseError::syntax_error(
                                            format!("Bilinmeyen optimizer tipi: {}", optimizer_name_clone)
                                        )),
                                    }
                                },
                                _ => return Err(ParseError::unexpected_token(
                                    token.clone(),
                                    "geçerli bir optimizer (SGD, Adam, AdamW, RMSProp) veya metin değeri".to_string()
                                )),
                            }
                        } else {
                            return Err(ParseError::UnexpectedEOF);
                        }
                    }
                    Token::Kayip => {
                        self.ilerle(); // "Kayip" tüket
                        self.expect(Token::Esittir)?;
                        
                        if let Some((token, _)) = self.simdiki() {
                            match token {
                                Token::Degisken(tip) => {
                                    let tip_str = tip.clone();
                                    self.ilerle();
                                    match tip_str.as_str() {
                                        "CrossEntropy" => kayip_fonksiyonu = Some(KayipFonksiyonu::CrossEntropy),
                                        "MSE" => kayip_fonksiyonu = Some(KayipFonksiyonu::MSE),
                                        "BinaryCrossEntropy" => kayip_fonksiyonu = Some(KayipFonksiyonu::BinaryCrossEntropy),
                                        _ => return Err(ParseError::syntax_error(
                                            format!("Geçersiz kayıp fonksiyonu: {}", tip_str)
                                        )),
                                    }
                                },
                                Token::Metin(tip) => {
                                    let tip_str = tip.clone();
                                    self.ilerle();
                                    match tip_str.as_str() {
                                        "cross_entropy" => kayip_fonksiyonu = Some(KayipFonksiyonu::CrossEntropy),
                                        "mse" => kayip_fonksiyonu = Some(KayipFonksiyonu::MSE),
                                        "binary_cross_entropy" => kayip_fonksiyonu = Some(KayipFonksiyonu::BinaryCrossEntropy),
                                        _ => return Err(ParseError::syntax_error(
                                            format!("Geçersiz kayıp fonksiyonu: {}", tip_str)
                                        )),
                                    }
                                },
                                _ => return Err(ParseError::unexpected_token(
                                    token.clone(),
                                    "kayıp fonksiyonu ismi (CrossEntropy, MSE vb.) veya \"cross_entropy\" gibi bir metin".to_string()
                                )),
                            }
                        } else {
                            return Err(ParseError::unexpected_token(
                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                "kayıp fonksiyonu ismi (CrossEntropy, MSE vb.)".to_string()
                            ));
                        }
                    }
                    Token::Batch => {
                        self.ilerle(); // "Batch" tüket
                        self.expect(Token::Esittir)?;
                        
                        if let Some((Token::Sayi(boyut), _)) = self.simdiki() {
                            let boyut_val = *boyut; // Dereference
                            self.ilerle();
                            batch_boyutu = Some(boyut_val);
                        } else {
                            return Err(ParseError::unexpected_token(
                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                "batch boyutu (i64)".to_string()
                            ));
                        }
                    }
                    Token::Epoch => {
                        self.ilerle(); // "Epoch" tüket
                        self.expect(Token::Esittir)?;
                        
                        if let Some((Token::Sayi(sayi), _)) = self.simdiki() {
                            let sayi_val = *sayi; // Dereference
                            self.ilerle();
                            epoch_sayisi = Some(sayi_val);
                        } else {
                            return Err(ParseError::unexpected_token(
                                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                "epoch sayısı (i64)".to_string()
                            ));
                        }
                    }
                     _ => return Err(ParseError::unexpected_token(
                        token.clone(), 
                        "beklenen model özelliği (Katman, OgrenmeOrani vb.) veya '}'".to_string()
                    )),
                }
            } else {
                 return Err(ParseError::UnexpectedEOF); // Token kalmadıysa hata
            }
        }
        
        // Varsayılan değerler
        let optimizer = optimizer.unwrap_or(OptimizerTipi::SGD {
            lr: 0.01,
            momentum: None,
            nesterov: None,
        });
        let kayip_fonksiyonu = kayip_fonksiyonu.unwrap_or(KayipFonksiyonu::MSE);
         // batch_boyutu ve epoch_sayisi için referans yerine değer kullanıldı
        let batch_boyutu = batch_boyutu.unwrap_or(32);
        let epoch_sayisi = epoch_sayisi.unwrap_or(10);
        
        if katmanlar.is_empty() {
             return Err(ParseError::model_config_error("Model en az bir katman içermelidir.".to_string()));
        }
        
        Ok(Some(Deyim::ModelTanimi {
            isim,
            ozellikler: ModelOzellikleri {
                katmanlar,
                optimizer,
                kayip_fonksiyonu,
                batch_boyutu,
                epoch_sayisi,
                metrikler: None,
                erken_durdurma: None,
            },
        }))
    }
    
    fn veri_parse(&mut self) -> Result<Option<Deyim>> {
        self.ilerle(); // "Veri" token'ı tüket
        
        let isim = if let Some((Token::Degisken(isim), _)) = self.simdiki() {
            let isim_str = isim.clone();
            self.ilerle();
            isim_str
        } else {
            return Err(ParseError::unexpected_token(
                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                "veri ismi".to_string()
            ));
        };
        
        self.expect(Token::Esittir)?;
        
        // ExprParser'ı doğru konuma getirelim ve manuel olarak dizi ifadesini parse edelim
        self.expr_parser = ExprParser::new(self.tokens.clone());
        self.expr_parser.set_current(self.current);
        
        // Dizi ifadesini parse et
        let deger = match self.expr_parser.ifade_parse() {
            Some(ifade) => {
                // ExprParser'ın ilerlediği konumu ModelParser'a aktar
                self.current = self.expr_parser.get_current();
                ifade
            },
            None => return Err(ParseError::syntax_error("Veri tanımı için geçersiz ifade.".to_string())),
        };
        
        Ok(Some(Deyim::VeriTanimi {
            isim,
            deger: Box::new(deger),
        }))
    }
    
    fn egit_parse(&mut self) -> Result<Option<Deyim>> {
        self.ilerle(); // "Egit" token'ı tüket
        
        self.expect(Token::AcParantez)?;
        
        // Model adını al
        let model = if let Some((Token::Degisken(model), _)) = self.simdiki() {
            let model_str = model.clone();
            self.ilerle();
            model_str
        } else {
            return Err(ParseError::unexpected_token(
                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                "eğitilecek model ismi".to_string()
            ));
        };
        
        // Virgül olmalı
        self.expect(Token::Virgul)?;
        
        // Eğitim verisini al
        let x_train_var = if let Some((Token::Degisken(var_name), _)) = self.simdiki() {
            let var_name_str = var_name.clone();
            self.ilerle();
            self.expr_parser.degisken_ifade(var_name_str)
        } else {
            return Err(ParseError::unexpected_token(
                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                "eğitim verisi değişkeni".to_string()
            ));
        };
        
        // Virgül olmalı
        self.expect(Token::Virgul)?;
        
        // Etiket verisini al
        let y_train_var = if let Some((Token::Degisken(var_name), _)) = self.simdiki() {
            let var_name_str = var_name.clone();
            self.ilerle();
            self.expr_parser.degisken_ifade(var_name_str)
        } else {
            return Err(ParseError::unexpected_token(
                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                "etiket verisi değişkeni".to_string()
            ));
        };
        
        // Kapanan parantez
        self.expect(Token::KapaParantez)?;
        
        Ok(Some(Deyim::Egitim {
            model,
            x_train: Box::new(x_train_var),
            y_train: Box::new(y_train_var),
        }))
    }
    
    fn tahmin_parse(&mut self) -> Result<Option<Deyim>> {
        self.ilerle(); // "Tahmin" token'ı tüket
        
        self.expect(Token::AcParantez)?;
        
        // Model adını al
        let model = if let Some((Token::Degisken(model), _)) = self.simdiki() {
            let model_str = model.clone();
            self.ilerle();
            model_str
        } else {
            return Err(ParseError::unexpected_token(
                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                "tahmin yapılacak model ismi".to_string()
            ));
        };
        
        // Virgül olmalı
        self.expect(Token::Virgul)?;
        
        // Test verisini al
        let test_var = if let Some((Token::Degisken(var_name), _)) = self.simdiki() {
            let var_name_str = var_name.clone();
            self.ilerle();
            self.expr_parser.degisken_ifade(var_name_str)
        } else {
            return Err(ParseError::unexpected_token(
                self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                "test verisi değişkeni".to_string()
            ));
        };
        
        // Kapanan parantez
        self.expect(Token::KapaParantez)?;
        
        Ok(Some(Deyim::Tahmin {
            model,
            girdi: test_var,
        }))
    }

    pub fn parse_optimizer(&mut self) -> Result<OptimizerTipi> {
        // Optimizer değerini al
        let mut optimizer_val = OptimizerTipi::SGD {
            lr: 0.01,
            momentum: None,
            nesterov: None,
        };

        if self.check(&Token::Optimizer) {
            self.ilerle(); // "Optimizer" token'ını tüket
            self.expect(Token::Esittir)?;
            
            if let Some((token, _)) = self.simdiki() {
                match token {
                    Token::Adam => {
                        self.ilerle();
                        // Adam parametrelerini kontrol et
                        let mut lr = 0.001; // Varsayılan değer
                        let mut beta1 = 0.9; // Varsayılan
                        let mut beta2 = 0.999; // Varsayılan
                        let mut epsilon = 1e-8; // Varsayılan
                        
                        if self.check(&Token::AcParantez) {
                            self.ilerle();
                            
                            // Parametreleri ayrıştır
                            while !self.check(&Token::KapaParantez) {
                                if let Some((Token::Degisken(param_name), _)) = self.simdiki() {
                                    let param = param_name.clone();
                                    self.ilerle();
                                    
                                    self.expect(Token::Esittir)?;
                                    
                                    match param.as_str() {
                                        "ogrenme_orani" | "lr" => {
                                            if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                                                lr = *oran as f64;
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(oran), _)) = self.simdiki() {
                                                lr = *oran as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "öğrenme oranı değeri".to_string()
                                                ));
                                            }
                                        },
                                        "beta1" => {
                                            if let Some((Token::Sayi(b), _)) = self.simdiki() {
                                                beta1 = *b as f64 / 10.0; // 9 -> 0.9
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(b), _)) = self.simdiki() {
                                                beta1 = *b as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "beta1 değeri".to_string()
                                                ));
                                            }
                                        },
                                        "beta2" => {
                                            if let Some((Token::Sayi(b), _)) = self.simdiki() {
                                                beta2 = *b as f64 / 10.0; // 9 -> 0.9
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(b), _)) = self.simdiki() {
                                                beta2 = *b as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "beta2 değeri".to_string()
                                                ));
                                            }
                                        },
                                        "epsilon" => {
                                            if let Some((Token::Sayi(e), _)) = self.simdiki() {
                                                epsilon = 1.0 / (10.0_f64.powi(*e as i32)); // 8 -> 1e-8
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(e), _)) = self.simdiki() {
                                                epsilon = *e as f64 / 1000.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "epsilon değeri".to_string()
                                                ));
                                            }
                                        },
                                        _ => return Err(ParseError::unexpected_token(
                                            Token::Degisken(param),
                                            "geçerli bir Adam parametresi".to_string()
                                        )),
                                    }
                                    
                                    // Virgül kontrolü
                                    if self.check(&Token::Virgul) {
                                        self.ilerle();
                                    }
                                } else {
                                    return Err(ParseError::unexpected_token(
                                        self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                        "parametre adı".to_string()
                                    ));
                                }
                            }
                            
                            self.ilerle(); // KapaParantez token'ını tüket
                        }
                        
                        return Ok(OptimizerTipi::Adam {
                            lr,
                            beta1,
                            beta2,
                            epsilon,
                        });
                    },
                    Token::SGD => {
                        self.ilerle();
                        // SGD parametrelerini kontrol et
                        let mut lr = 0.01; // Varsayılan değer
                        let mut momentum = None;
                        let mut nesterov = None;
                        
                        if self.check(&Token::AcParantez) {
                            self.ilerle();
                            
                            // Parametreleri ayrıştır
                            while !self.check(&Token::KapaParantez) {
                                if let Some((Token::Degisken(param_name), _)) = self.simdiki() {
                                    let param = param_name.clone();
                                    self.ilerle();
                                    
                                    self.expect(Token::Esittir)?;
                                    
                                    match param.as_str() {
                                        "ogrenme_orani" | "lr" => {
                                            if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                                                lr = *oran as f64;
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(oran), _)) = self.simdiki() {
                                                lr = *oran as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "öğrenme oranı değeri".to_string()
                                                ));
                                            }
                                        },
                                        "momentum" => {
                                            if let Some((Token::Sayi(m), _)) = self.simdiki() {
                                                momentum = Some(*m as f64 / 10.0); // 9 -> 0.9
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(m), _)) = self.simdiki() {
                                                momentum = Some(*m as f64 / 100.0);
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "momentum değeri".to_string()
                                                ));
                                            }
                                        },
                                        "nesterov" => {
                                            if let Some((Token::Dogru, _)) = self.simdiki() {
                                                nesterov = Some(true);
                                                self.ilerle();
                                            } else if let Some((Token::Yanlis, _)) = self.simdiki() {
                                                nesterov = Some(false);
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "nesterov değeri (true veya false)".to_string()
                                                ));
                                            }
                                        },
                                        _ => return Err(ParseError::unexpected_token(
                                            Token::Degisken(param),
                                            "geçerli bir SGD parametresi".to_string()
                                        )),
                                    }
                                    
                                    // Virgül kontrolü
                                    if self.check(&Token::Virgul) {
                                        self.ilerle();
                                    }
                                } else {
                                    return Err(ParseError::unexpected_token(
                                        self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                        "parametre adı".to_string()
                                    ));
                                }
                            }
                            
                            self.ilerle(); // KapaParantez token'ını tüket
                        }
                        
                        return Ok(OptimizerTipi::SGD {
                            lr,
                            momentum,
                            nesterov,
                        });
                    },
                    Token::AdamW => {
                        self.ilerle();
                        // AdamW parametrelerini kontrol et
                        let mut lr = 0.001; // Varsayılan değer
                        let mut weight_decay = 0.01; // Varsayılan
                        let mut beta1 = 0.9; // Varsayılan
                        let mut beta2 = 0.999; // Varsayılan
                        
                        if self.check(&Token::AcParantez) {
                            self.ilerle();
                            
                            // Parametreleri ayrıştır
                            while !self.check(&Token::KapaParantez) {
                                if let Some((Token::Degisken(param_name), _)) = self.simdiki() {
                                    let param = param_name.clone();
                                    self.ilerle();
                                    
                                    self.expect(Token::Esittir)?;
                                    
                                    match param.as_str() {
                                        "ogrenme_orani" | "lr" => {
                                            if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                                                lr = *oran as f64;
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(oran), _)) = self.simdiki() {
                                                lr = *oran as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "öğrenme oranı değeri".to_string()
                                                ));
                                            }
                                        },
                                        "weight_decay" | "agirlik_azaltma" => {
                                            if let Some((Token::Sayi(wd), _)) = self.simdiki() {
                                                weight_decay = *wd as f64;
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(wd), _)) = self.simdiki() {
                                                weight_decay = *wd as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "weight_decay değeri".to_string()
                                                ));
                                            }
                                        },
                                        "beta1" => {
                                            if let Some((Token::Sayi(b), _)) = self.simdiki() {
                                                beta1 = *b as f64 / 10.0; // 9 -> 0.9
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(b), _)) = self.simdiki() {
                                                beta1 = *b as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "beta1 değeri".to_string()
                                                ));
                                            }
                                        },
                                        "beta2" => {
                                            if let Some((Token::Sayi(b), _)) = self.simdiki() {
                                                beta2 = *b as f64 / 10.0; // 9 -> 0.9
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(b), _)) = self.simdiki() {
                                                beta2 = *b as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "beta2 değeri".to_string()
                                                ));
                                            }
                                        },
                                        _ => return Err(ParseError::unexpected_token(
                                            Token::Degisken(param),
                                            "geçerli bir AdamW parametresi".to_string()
                                        )),
                                    }
                                    
                                    // Virgül kontrolü
                                    if self.check(&Token::Virgul) {
                                        self.ilerle();
                                    }
                                } else {
                                    return Err(ParseError::unexpected_token(
                                        self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                        "parametre adı".to_string()
                                    ));
                                }
                            }
                            
                            self.ilerle(); // KapaParantez token'ını tüket
                        }
                        
                        return Ok(OptimizerTipi::AdamW {
                            lr,
                            weight_decay,
                            beta1,
                            beta2,
                        });
                    },
                    Token::RMSProp => {
                        self.ilerle();
                        // RMSProp parametrelerini kontrol et
                        let mut lr = 0.001; // Varsayılan değer
                        let mut rho = 0.9; // Varsayılan
                        let mut epsilon = 1e-8; // Varsayılan
                        
                        if self.check(&Token::AcParantez) {
                            self.ilerle();
                            
                            // Parametreleri ayrıştır
                            while !self.check(&Token::KapaParantez) {
                                if let Some((Token::Degisken(param_name), _)) = self.simdiki() {
                                    let param = param_name.clone();
                                    self.ilerle();
                                    
                                    self.expect(Token::Esittir)?;
                                    
                                    match param.as_str() {
                                        "ogrenme_orani" | "lr" => {
                                            if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                                                lr = *oran as f64;
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(oran), _)) = self.simdiki() {
                                                lr = *oran as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "öğrenme oranı değeri".to_string()
                                                ));
                                            }
                                        },
                                        "rho" | "bozulma_orani" => {
                                            if let Some((Token::Sayi(r), _)) = self.simdiki() {
                                                rho = *r as f64 / 10.0; // 9 -> 0.9
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(r), _)) = self.simdiki() {
                                                rho = *r as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "rho değeri".to_string()
                                                ));
                                            }
                                        },
                                        "epsilon" => {
                                            if let Some((Token::Sayi(e), _)) = self.simdiki() {
                                                epsilon = 1.0 / (10.0_f64.powi(*e as i32)); // 8 -> 1e-8
                                                self.ilerle();
                                            } else if let Some((Token::OndalikSayi(e), _)) = self.simdiki() {
                                                epsilon = *e as f64 / 100.0;
                                                self.ilerle();
                                            } else {
                                                return Err(ParseError::unexpected_token(
                                                    self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                                    "epsilon değeri".to_string()
                                                ));
                                            }
                                        },
                                        _ => return Err(ParseError::unexpected_token(
                                            Token::Degisken(param),
                                            "geçerli bir RMSProp parametresi".to_string()
                                        )),
                                    }
                                    
                                    // Virgül kontrolü
                                    if self.check(&Token::Virgul) {
                                        self.ilerle();
                                    }
                                } else {
                                    return Err(ParseError::unexpected_token(
                                        self.simdiki().map(|(t,_)| t.clone()).unwrap_or(Token::Illegal),
                                        "parametre adı".to_string()
                                    ));
                                }
                            }
                            
                            self.ilerle(); // KapaParantez token'ını tüket
                        }
                        
                        return Ok(OptimizerTipi::RMSProp {
                            lr,
                            rho,
                            epsilon,
                        });
                    },
                    Token::Metin(optimizer_name) => {
                        let optimizer_name_clone = optimizer_name.clone();
                        self.ilerle();
                        // Metin olarak verilen optimizer adını analiz et (geriye uyumluluk için)
                        match optimizer_name_clone.as_str() {
                            "sgd" => {
                                optimizer_val = OptimizerTipi::SGD {
                                    lr: 0.01,
                                    momentum: None,
                                    nesterov: None,
                                };
                            },
                            "adam" => {
                                optimizer_val = OptimizerTipi::Adam {
                                    lr: 0.001,
                                    beta1: 0.9,
                                    beta2: 0.999,
                                    epsilon: 1e-8,
                                };
                            },
                            "adamw" => {
                                optimizer_val = OptimizerTipi::AdamW {
                                    lr: 0.001,
                                    weight_decay: 0.01,
                                    beta1: 0.9,
                                    beta2: 0.999,
                                };
                            },
                            "rmsprop" => {
                                optimizer_val = OptimizerTipi::RMSProp {
                                    lr: 0.001,
                                    rho: 0.9,
                                    epsilon: 1e-8,
                                };
                            },
                            _ => return Err(ParseError::syntax_error(
                                format!("Bilinmeyen optimizer tipi: {}", optimizer_name_clone)
                            )),
                        }
                    },
                    _ => return Err(ParseError::unexpected_token(
                        token.clone(),
                        "geçerli bir optimizer (SGD, Adam, AdamW, RMSProp) veya metin değeri".to_string()
                    )),
                }
            } else {
                return Err(ParseError::UnexpectedEOF);
            }
        }
        
        // Varsayılan değer
        Ok(optimizer_val)
    }
}

impl super::Parser for ModelParser {
    fn parse(&mut self) -> Result<Program> {
        self.parse()
    }
    
    fn advance(&mut self) -> Option<Token> {
        if let Some((token, _)) = self.ilerle() {
            Some(token.clone())
        } else {
            None
        }
    }
    
    fn current(&self) -> Option<&Token> {
        if let Some((token, _)) = self.simdiki() {
            Some(token)
        } else {
            None
        }
    }
    
    fn peek(&self) -> Option<&Token> {
        if self.current + 1 < self.tokens.len() {
            let (token, _) = &self.tokens[self.current + 1];
            Some(token)
        } else {
            None
        }
    }
    
    fn expect(&mut self, expected: Token) -> Result<Token> {
        if let Some(current) = self.current() {
            if current == &expected {
                self.advance().ok_or(ParseError::UnexpectedEOF)
            } else {
                Err(ParseError::unexpected_token(
                    current.clone(), 
                    format!("{:?}", expected)
                ))
            }
        } else {
            Err(ParseError::UnexpectedEOF)
        }
    }
    
    fn expect_one_of(&mut self, expected_tokens: &[Token]) -> Result<Token> {
        if let Some(current) = self.current() {
            if expected_tokens.contains(current) {
                self.advance().ok_or(ParseError::UnexpectedEOF)
            } else {
                let expected_str = expected_tokens.iter()
                    .map(|t| format!("{:?}", t))
                    .collect::<Vec<_>>()
                    .join(" veya ");
                 Err(ParseError::unexpected_token(
                    current.clone(), 
                    expected_str
                ))
            }
        } else {
            Err(ParseError::UnexpectedEOF)
        }
    }
    
    fn check(&self, token_type: &Token) -> bool {
        if let Some(current) = self.current() {
            current == token_type
        } else {
            false
        }
    }
    
    fn error(&self, message: &str) -> ParseError {
        ParseError::SyntaxError(message.to_string())
    }
    
    fn reset(&mut self) {
        self.current = 0;
    }
    
    fn get_position(&self) -> (usize, usize) {
        // ModelParser şu anda satır ve sütun bilgisini tutmuyor,
        // bu yüzden varsayılan değer dönüyoruz
        (0, self.current)
    }
    
    fn recover(&mut self, sync_tokens: &[Token]) {
        while let Some(current) = self.current() {
            if sync_tokens.contains(current) {
                break;
            }
            self.advance();
        }
    }
} 
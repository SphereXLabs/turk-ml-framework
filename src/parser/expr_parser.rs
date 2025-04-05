use crate::ast::{Ifade, Deyim, IstatistikselIslemTipi, KarsilastirmaOperator, Operator};
use crate::token::Token;
use super::error::{ParseError, Result};

pub struct ExprParser {
    tokens: Vec<(Token, String)>,
    current: usize,
}

impl ExprParser {
    pub fn new(tokens: Vec<(Token, String)>) -> Self {
        Self { tokens, current: 0 }
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
    
    pub fn ifade_parse(&mut self) -> Option<Ifade> {
        self.toplama_parse()
    }
    
    fn toplama_parse(&mut self) -> Option<Ifade> {
        let mut sol = self.carpma_parse()?;
        
        while let Some((token, _)) = self.simdiki() {
            match token {
                Token::Arti | Token::Eksi => {
                    let op_token = self.ilerle().unwrap().0.clone();
                    let sag = self.carpma_parse()?;
                    let operator = match op_token {
                        Token::Arti => Operator::Toplama,
                        Token::Eksi => Operator::Cikarma,
                        _ => unreachable!(),
                    };
                    sol = Ifade::IkiliIslem {
                        sol: Box::new(sol),
                        operator,
                        sag: Box::new(sag),
                    };
                }
                _ => break,
            }
        }
        
        Some(sol)
    }
    
    fn carpma_parse(&mut self) -> Option<Ifade> {
        let mut sol = self.birli_parse()?;
        
        while let Some((token, _)) = self.simdiki() {
            match token {
                Token::Carpma | Token::Bolme | Token::Modulo => {
                    let op_token = self.ilerle().unwrap().0.clone();
                    let sag = self.birli_parse()?;
                    let operator = match op_token {
                        Token::Carpma => Operator::Carpma,
                        Token::Bolme => Operator::Bolme,
                        Token::Modulo => Operator::ModOperator,
                        _ => unreachable!(),
                    };
                    sol = Ifade::IkiliIslem {
                        sol: Box::new(sol),
                        operator,
                        sag: Box::new(sag),
                    };
                }
                _ => break,
            }
        }
        
        Some(sol)
    }
    
    fn birli_parse(&mut self) -> Option<Ifade> {
        if let Some((token, _)) = self.simdiki() {
            match token {
                Token::Eksi => {
                    self.ilerle();
                    let ifade = self.birli_parse()?;
                    return Some(Ifade::IkiliIslem {
                        sol: Box::new(Ifade::Sayi(0)),
                        operator: Operator::Cikarma,
                        sag: Box::new(ifade),
                    });
                }
                _ => {}
            }
        }
        
        self.birim_parse()
    }
    
    fn birim_parse(&mut self) -> Option<Ifade> {
        let (token, _literal) = self.simdiki()?.clone();
        
        match token {
            Token::Sayi(n) => {
                self.ilerle();
                Some(Ifade::Sayi(n))
            }
            Token::Metin(s) => {
                self.ilerle();
                Some(Ifade::Metin(s))
            }
            Token::Dogru => {
                self.ilerle();
                Some(Ifade::Mantiksal(true))
            }
            Token::Yanlis => {
                self.ilerle();
                Some(Ifade::Mantiksal(false))
            }
            Token::Degisken(isim) => {
                self.ilerle();
                if self.simdiki().map_or(false, |(t, _)| t == &Token::AcParantez) {
                    self.ilerle();
                    let mut argumanlar = Vec::new();
                    
                    if !self.simdiki().map_or(false, |(t, _)| t == &Token::KapaParantez) {
                        loop {
                            let arg = self.ifade_parse()?;
                            argumanlar.push(arg);
                            
                            if self.simdiki().map_or(false, |(t, _)| t == &Token::Virgul) {
                                self.ilerle();
                            } else {
                                break;
                            }
                        }
                    }
                        
                    if self.simdiki().map_or(false, |(t, _)| t == &Token::KapaParantez) {
                         self.ilerle();
                    } else {
                        return None;
                    }
                    
                    Some(Ifade::FonksiyonCagrisi {
                         fonksiyon: Box::new(Ifade::Degisken(isim)),
                         argumanlar,
                    })
                } else {
                    Some(Ifade::Degisken(isim))
                }
            }
            Token::AcParantez => {
                self.ilerle();
                let ifade = self.ifade_parse()?;
                
                if self.simdiki().map_or(false, |(t, _)| t == &Token::KapaParantez) {
                    self.ilerle();
                    Some(Ifade::Parantez(Box::new(ifade)))
                } else {
                    None
                }
            }
            Token::AcKoseli => {
                 self.ilerle();
                 let mut elemanlar = Vec::new();
                 if !self.simdiki().map_or(false, |(t, _)| t == &Token::KapaKoseli) {
                     loop {
                         elemanlar.push(self.ifade_parse()?);
                         if self.simdiki().map_or(false, |(t, _)| t == &Token::Virgul) {
                             self.ilerle();
                         } else {
                             break;
                         }
                     }
                 }
                 if self.simdiki().map_or(false, |(t, _)| t == &Token::KapaKoseli) {
                     self.ilerle();
                     Some(Ifade::Dizi(elemanlar))
                 } else {
                     None
                 }
             }
            _ => None,
        }
    }
    
    pub fn karsilastirma_parse(&mut self) -> Option<Ifade> {
        let sol = self.ifade_parse()?;
        
        if let Some((token, _)) = self.simdiki() {
            match token {
                Token::EsitMi | Token::EsitDegilMi | Token::BuyukMu |
                Token::KucukMu | Token::BuyukEsitMi | Token::KucukEsitMi => {
                    let op_token = self.ilerle()?.0.clone();
                    let sag = self.ifade_parse()?;
                    
                    let operator = match op_token {
                        Token::EsitMi => KarsilastirmaOperator::EsitMi,
                        Token::EsitDegilMi => KarsilastirmaOperator::EsitDegilMi,
                        Token::BuyukMu => KarsilastirmaOperator::BuyukMu,
                        Token::KucukMu => KarsilastirmaOperator::KucukMu,
                        Token::BuyukEsitMi => KarsilastirmaOperator::BuyukEsitMi,
                        Token::KucukEsitMi => KarsilastirmaOperator::KucukEsitMi,
                        _ => unreachable!(),
                    };
                    
                    return Some(Ifade::KarsilastirmaIfadesi {
                        sol: Box::new(sol),
                        operator,
                        sag: Box::new(sag),
                    });
                }
                _ => {}
            }
        }
        
        Some(sol)
    }
    
    pub fn istatistiksel_islem_parse(&mut self) -> Result<Option<Deyim>> {
        let (token, _) = self.simdiki().ok_or(ParseError::UnexpectedEOF)?.clone();
        let islem_tipi = match token {
            Token::Ortalama => IstatistikselIslemTipi::Ortalama,
            Token::Medyan => IstatistikselIslemTipi::Medyan,
            Token::ModFonksiyon => IstatistikselIslemTipi::Mod,
            Token::StandartSapma => IstatistikselIslemTipi::StandartSapma,
            Token::MinMax => IstatistikselIslemTipi::MinMax,
            _ => return Err(ParseError::unexpected_token(token, "istatistiksel işlem token'ı".to_string())),
        };
        
        self.ilerle();
        self.tuket(Token::AcParantez, "'(' bekleniyor")?;
        let ifade = self.ifade_parse().ok_or_else(|| ParseError::syntax_error("İstatistiksel işlem için geçersiz ifade.".to_string()))?;
        self.tuket(Token::KapaParantez, "')' bekleniyor")?;
        
        Ok(Some(Deyim::IfadeDeyimi(Ifade::IstatistikselIslem {
            islem_tipi,
            veri: Box::new(ifade),
        })))
    }
    
    fn tuket(&mut self, beklenen: Token, hata_mesaji: &str) -> Result<()> {
        if let Some((token, _)) = self.simdiki() {
            if token == &beklenen {
                self.ilerle();
                return Ok(());
            }
        }
        Err(ParseError::MissingToken(hata_mesaji.to_string()))
    }
} 
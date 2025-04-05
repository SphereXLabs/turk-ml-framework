use crate::ast::{Ifade, Deyim, IstatistikselIslemTipi, KarsilastirmaOperator};
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
                    let op = self.ilerle().unwrap().0.clone();
                    let sag = self.carpma_parse()?;
                    sol = Ifade::IkiliIslem {
                        sol: Box::new(sol),
                        islem: match op {
                            Token::Arti => "+".to_string(),
                            Token::Eksi => "-".to_string(),
                            _ => unreachable!(),
                        },
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
                    let op = self.ilerle().unwrap().0.clone();
                    let sag = self.birli_parse()?;
                    sol = Ifade::IkiliIslem {
                        sol: Box::new(sol),
                        islem: match op {
                            Token::Carpma => "*".to_string(),
                            Token::Bolme => "/".to_string(),
                            Token::Modulo => "%".to_string(),
                            _ => unreachable!(),
                        },
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
                    return Some(Ifade::TekliIslem {
                        islem: "-".to_string(),
                        ifade: Box::new(ifade),
                    });
                }
                _ => {}
            }
        }
        
        self.birim_parse()
    }
    
    fn birim_parse(&mut self) -> Option<Ifade> {
        let token = self.simdiki()?.0.clone();
        
        match token {
            Token::Sayi(n) => {
                self.ilerle();
                Some(Ifade::Sayi(n))
            }
            Token::Degisken(isim) => {
                self.ilerle();
                if let Some((Token::AcParantez, _)) = self.simdiki() {
                    self.ilerle();
                    let mut parametreler = Vec::new();
                    
                    if let Some((Token::KapaParantez, _)) = self.simdiki() {
                        self.ilerle();
                    } else {
                        loop {
                            let param = self.ifade_parse()?;
                            parametreler.push(param);
                            
                            if let Some((Token::Virgul, _)) = self.simdiki() {
                                self.ilerle();
                            } else {
                                break;
                            }
                        }
                        
                        if let Some((Token::KapaParantez, _)) = self.simdiki() {
                            self.ilerle();
                        } else {
                            return None;
                        }
                    }
                    
                    Some(Ifade::FonksiyonCagrisi {
                        isim,
                        parametreler,
                    })
                } else {
                    Some(Ifade::Degisken(isim))
                }
            }
            Token::AcParantez => {
                self.ilerle();
                let ifade = self.ifade_parse()?;
                
                if let Some((Token::KapaParantez, _)) = self.simdiki() {
                    self.ilerle();
                    Some(ifade)
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
                    let op = self.ilerle()?.0.clone();
                    let sag = self.ifade_parse()?;
                    
                    let operator = match op {
                        Token::EsitMi => KarsilastirmaOperator::EsitMi,
                        Token::EsitDegilMi => KarsilastirmaOperator::EsitDegilMi,
                        Token::BuyukMu => KarsilastirmaOperator::BuyukMu,
                        Token::KucukMu => KarsilastirmaOperator::KucukMu,
                        Token::BuyukEsitMi => KarsilastirmaOperator::BuyukEsitMi,
                        Token::KucukEsitMi => KarsilastirmaOperator::KucukEsitMi,
                        _ => unreachable!(),
                    };
                    
                    return Some(Ifade::Karsilastirma {
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
    
    pub fn istatistiksel_islem_parse(&mut self) -> Option<Deyim> {
        let islem_tipi = match self.simdiki()?.0 {
            Token::Ortalama => IstatistikselIslemTipi::Ortalama,
            Token::Medyan => IstatistikselIslemTipi::Medyan,
            Token::ModFonksiyon => IstatistikselIslemTipi::Mod,
            Token::StandartSapma => IstatistikselIslemTipi::StandartSapma,
            Token::Varyans => IstatistikselIslemTipi::Varyans,
            _ => return None,
        };
        
        self.ilerle();
        self.tuket(Token::AcParantez, "'(' bekleniyor")?;
        let ifade = self.ifade_parse()?;
        self.tuket(Token::KapaParantez, "')' bekleniyor")?;
        
        Some(Deyim::IstatistikselIslem {
            islem_tipi,
            ifade: Box::new(ifade),
        })
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
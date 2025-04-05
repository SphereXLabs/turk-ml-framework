use crate::ast::{Deyim, Ifade, KarsilastirmaOperator, Program, MantiksalOperator, 
    IstatistikselIslemTipi, KayipFonksiyonu, AktivasyonFonksiyonu, KatmanOzellikleri, ModelOzellikleri};
use crate::token::Token;

#[derive(Debug)]
#[allow(dead_code)]
pub enum ParseError {
    GecersizDeyim(Option<(Token, String)>),
    TanimlanmamisDegisken(String),
    GecersizFonksiyonCagrisi(String),
    GecersizParametreSayisi { fonksiyon: String, beklenen: usize, bulunan: usize },
    SifiraBolmeHatasi,
    DiziBoyutuHatasi { dizi: String, indeks: i64, boyut: usize },
    SozdizimiHatasi(String),
    BeklenenToken(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::GecersizDeyim(token) => write!(f, "Geçersiz deyim: {:?}", token),
            ParseError::TanimlanmamisDegisken(isim) => write!(f, "Tanımlanmış değişken: {}", isim),
            ParseError::GecersizFonksiyonCagrisi(isim) => write!(f, "Geçersiz fonksiyon çağrısı: {}", isim),
            ParseError::GecersizParametreSayisi { fonksiyon, beklenen, bulunan } => 
                write!(f, "{} fonksiyonu {} parametre bekliyor, {} parametre verilmiş", fonksiyon, beklenen, bulunan),
            ParseError::SifiraBolmeHatasi => write!(f, "Sıfıra bölme hatası"),
            ParseError::DiziBoyutuHatasi { dizi, indeks, boyut } => 
                write!(f, "{} dizisinin {} indeksi geçersiz. Dizi boyutu: {}", dizi, indeks, boyut),
            ParseError::SozdizimiHatasi(mesaj) => write!(f, "Sözdizimi hatası: {}", mesaj),
            ParseError::BeklenenToken(token) => write!(f, "Beklenen token: {}", token),
        }
    }
}

impl std::error::Error for ParseError {}

pub struct Parser {
    tokens: Vec<(Token, String)>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<(Token, String)>) -> Self {
        Self {
            tokens,
            current: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Program, ParseError> {
        let mut deyimler = Vec::new();
        
        while !self.bitti() {
            eprintln!("Şu anki token: {:?}", self.simdiki());
            if let Some(deyim) = self.deyim_parse() {
                eprintln!("Deyim başarıyla ayrıştırıldı: {:#?}", deyim);
                deyimler.push(deyim);
            } else {
                let hata = format!("Deyim ayrıştırma hatası, son token: {:?}", self.simdiki());
                eprintln!("{}", hata);
                return Err(ParseError::GecersizDeyim(self.simdiki().cloned()));
            }
        }
        
        Ok(Program { deyimler })
    }

    fn deyim_parse(&mut self) -> Option<Deyim> {
        let token = self.simdiki()?.0.clone();
        eprintln!("Şu anki token: {:?}", token);
        match token {
            Token::Veri => {
                eprintln!("Veri token'ı tespit edildi");
                self.veri_ayristir()
            }
            Token::Egit => {
                eprintln!("Egit token'ı tespit edildi");
                self.egit_parse()
            }
            Token::Model => {
                eprintln!("Model tanımı başlatılıyor");
                self.model_parse()
            }
            Token::Tahmin => self.tahmin_parse(),
            Token::Degisken(isim) => {
                self.ilerle();
                if let Some((Token::Esittir, _)) = self.simdiki() {
                    self.ilerle();
                    if let Some(ifade) = self.ifade_parse() {
                        Some(Deyim::Atama {
                            isim: isim.to_string(),
                            deger: ifade,
                        })
                    } else {
                        eprintln!("Uyarı: Atama değeri bekleniyor");
                        None
                    }
                } else if let Some((Token::AcParantez, _)) = self.simdiki() {
                    // Fonksiyon çağrısı durumu
                    self.ilerle();
                    let mut argumanlar = Vec::new();
                    
                    while let Some((token, _)) = self.simdiki() {
                        let token = token.clone();
                        match token {
                            Token::KapaParantez => {
                                self.ilerle();
                                return Some(Deyim::IfadeDeyimi(Ifade::FonksiyonCagrisi {
                                    fonksiyon: Box::new(Ifade::Degisken(isim.to_string())),
                                    argumanlar,
                                }));
                            }
                            Token::Virgul => {
                                self.ilerle();
                            }
                            _ => {
                                if let Some(arguman) = self.ifade_parse() {
                                    argumanlar.push(arguman);
                                }
                            }
                        }
                    }
                    eprintln!("Uyarı: Fonksiyon çağrısı kapanış parantezi eksik");
                    None
                } else {
                    Some(Deyim::IfadeDeyimi(Ifade::Degisken(isim.to_string())))
                }
            }
            Token::Virgul => {
                self.ilerle();
                self.deyim_parse()
            }
            Token::AcParantez => {
                self.ilerle();
                if let Some(ifade) = self.ifade_parse() {
                    if self.esle(Token::KapaParantez) {
                        Some(Deyim::IfadeDeyimi(ifade))
                    } else {
                        eprintln!("Uyarı: Kapanış parantezi bekleniyor");
                        None
                    }
                } else {
                    eprintln!("Uyarı: Parantez içinde ifade bekleniyor");
                    None
                }
            }
            Token::AcKume => {
                eprintln!("Blok başlatılıyor");
                self.ilerle();
                let mut deyimler = Vec::new();
                while let Some((token, _)) = self.simdiki() {
                    let token = token.clone();
                    if token == Token::KapaKume {
                        self.ilerle();
                        break;
                    }
                    if let Some(deyim) = self.deyim_parse() {
                        deyimler.push(deyim);
                    } else {
                        self.ilerle();
                    }
                }
                Some(Deyim::Blok(deyimler))
            }
            Token::KapaKume => {
                eprintln!("Blok kapanışı");
                self.ilerle();
                None
            }
            _ => {
                eprintln!("Uyarı: Geçersiz token: {:?}", token);
                None
            }
        }
    }

    fn ifade_parse(&mut self) -> Option<Ifade> {
        let mut ifade = self.terim_parse()?;
        
        // Fonksiyon çağrılarını işle
        while self.esle(Token::AcParantez) {
            let argumanlar = self.arguman_listesi_parse();
            if !self.esle(Token::KapaParantez) {
                eprintln!("Uyarı: Kapanış parantezi bekleniyor");
                return None;
            }
            ifade = Ifade::FonksiyonCagrisi {
                fonksiyon: Box::new(ifade),
                argumanlar,
            };
        }
        Some(ifade)
    }

    fn arguman_listesi_parse(&mut self) -> Vec<Ifade> {
        let mut args = Vec::new();
        while !self.siradaki_token_kontrol(Token::KapaParantez) {
            if let Some(arg) = self.ifade_parse() {
                args.push(arg);
            }
            if !self.esle(Token::Virgul) { break; }
        }
        args
    }

    fn terim_parse(&mut self) -> Option<Ifade> {
        let token = self.simdiki()?.0.clone();
        let mut terim = match token {
            Token::Sayi(n) => {
                self.ilerle();
                Ifade::Sayi(n)
            }
            Token::Metin(s) => {
                self.ilerle();
                Ifade::Metin(s)
            }
            Token::Dogru => {
                self.ilerle();
                Ifade::Mantiksal(true)
            }
            Token::Yanlis => {
                self.ilerle();
                Ifade::Mantiksal(false)
            }
            Token::Degisken(isim) => {
                self.ilerle();
                Ifade::Degisken(isim.to_string())
            }
            Token::AcKoseli => {
                self.ilerle();
                let mut elemanlar = Vec::new();
                while let Some((token, _)) = self.simdiki() {
                    let token = token.clone();
                    match token {
                        Token::KapaKoseli => {
                            self.ilerle();
                            break;
                        }
                        Token::Virgul => {
                            self.ilerle();
                        }
                        _ => {
                            if let Some(eleman) = self.ifade_parse() {
                                elemanlar.push(eleman);
                            }
                        }
                    }
                }
                Ifade::Liste(elemanlar)
            }
            Token::AcParantez => {
                self.ilerle();
                if let Some(ifade) = self.ifade_parse() {
                    if self.esle(Token::KapaParantez) {
                        Ifade::Parantez(Box::new(ifade))
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        while let Some((token, _)) = self.simdiki() {
            let token = token.clone();
            match token {
                Token::AcParantez => {
                    self.ilerle();
                    let mut argumanlar = Vec::new();
                    while let Some((token, _)) = self.simdiki() {
                        let token = token.clone();
                        match token {
                            Token::KapaParantez => {
                                self.ilerle();
                                break;
                            }
                            Token::Virgul => {
                                self.ilerle();
                            }
                            _ => {
                                if let Some(arguman) = self.ifade_parse() {
                                    argumanlar.push(arguman);
                                }
                            }
                        }
                    }
                    terim = Ifade::FonksiyonCagrisi {
                        fonksiyon: Box::new(terim),
                        argumanlar,
                    };
                }
                _ => break,
            }
        }
        Some(terim)
    }

    fn simdiki(&self) -> Option<&(Token, String)> {
        self.tokens.get(self.current)
    }

    fn ilerle(&mut self) {
        if self.current < self.tokens.len() {
            self.current += 1;
        }
    }

    fn bitti(&self) -> bool {
        self.current >= self.tokens.len()
    }

    fn esle(&mut self, beklenen: Token) -> bool {
        if let Some((token, _)) = self.simdiki() {
            let token = token.clone();
            if token == beklenen {
                self.ilerle();
                return true;
            }
        }
        false
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.current).map(|(token, _)| token)
    }

    #[allow(dead_code)]
    fn tuket(&mut self, beklenen: Token, hata_mesaji: &str) {
        if !self.esle(beklenen) {
            panic!("{}", hata_mesaji);
        }
    }

    #[allow(dead_code)]
    fn karsilastirma(&mut self) -> Ifade {
        self.karsilastirma_parse().unwrap()
    }

    #[allow(dead_code)]
    fn blok(&mut self) -> Vec<Deyim> {
        let mut deyimler = Vec::new();
        while let Some((token, _)) = self.simdiki() {
            match token {
                Token::KapaKume => {
                    self.ilerle();
                    break;
                }
                _ => if let Some(deyim) = self.deyim_parse() {
                    deyimler.push(deyim);
                }
            }
        }
        deyimler
    }

    #[allow(dead_code)]
    fn ifade(&mut self) -> Ifade {
        if self.esle(Token::AcKoseli) {
            return self.dizi_ifadesi();
        }

        if self.esle(Token::Dogru) {
            return Ifade::Mantiksal(true);
        }

        if self.esle(Token::Yanlis) {
            return Ifade::Mantiksal(false);
        }

        let mut ifade = self.karsilastirma();

        while let Some((token, _)) = self.simdiki() {
            let token = token.clone();
            match token {
                Token::Ve | Token::Veya => {
                    self.ilerle();
                    let operator = match token {
                        Token::Ve => MantiksalOperator::Ve,
                        Token::Veya => MantiksalOperator::Veya,
                        _ => unreachable!(),
                    };
                    let sag = self.karsilastirma();
                    ifade = Ifade::MantiksalIfade {
                        sol: Box::new(ifade),
                        operator,
                        sag: Box::new(sag),
                    };
                }
                _ => break,
            }
        }

        ifade
    }

    #[allow(dead_code)]
    fn dizi_ifadesi(&mut self) -> Ifade {
        let mut elemanlar = Vec::new();
        
        if !self.esle(Token::KapaKoseli) {
            loop {
                elemanlar.push(self.ifade());
                if !self.esle(Token::Virgul) {
                    break;
                }
            }
            self.tuket(Token::KapaKoseli, "'']' bekleniyor");
        }

        Ifade::Dizi(elemanlar)
    }

    #[allow(dead_code)]
    fn istatistiksel_ifade(&mut self) -> Ifade {
        let token = self.peek().unwrap().clone();
        self.ilerle();
        let islem_tipi = match token {
            Token::Ortalama => IstatistikselIslemTipi::Ortalama,
            Token::Medyan => IstatistikselIslemTipi::Medyan,
            Token::ModFonksiyon => IstatistikselIslemTipi::Mod,
            Token::StandartSapma => IstatistikselIslemTipi::StandartSapma,
            Token::MinMax => IstatistikselIslemTipi::MinMax,
            _ => unreachable!(),
        };

        self.tuket(Token::AcParantez, "'(' bekleniyor");
        let veri = Box::new(self.ifade());
        self.tuket(Token::KapaParantez, "')' bekleniyor");

        Ifade::IstatistikselIslem {
            islem_tipi,
            veri,
        }
    }

    fn siradaki_token_kontrol(&self, token: Token) -> bool {
        self.peek().map_or(false, |t| *t == token)
    }

    #[allow(dead_code)]
    fn bekle(&mut self, beklenen: Token) -> Result<(), ParseError> {
        if let Some((token, _)) = self.simdiki() {
            let token = token.clone();
            if token == beklenen {
                self.ilerle();
                Ok(())
            } else {
                Err(ParseError::BeklenenToken(format!("{:?}", beklenen)))
            }
        } else {
            Err(ParseError::BeklenenToken(format!("{:?}", beklenen)))
        }
    }

    fn model_parse(&mut self) -> Option<Deyim> {
        eprintln!("Model tanımı ayrıştırılıyor");
        self.ilerle(); // Model tokenını atla
        
        let isim = if let Some((Token::Degisken(isim), _)) = self.simdiki() {
            eprintln!("Model ismi tespit edildi: {}", isim);
            let isim = isim.to_string();
            self.ilerle();
            isim
        } else {
            eprintln!("Uyarı: Model ismi bekleniyor");
            return None;
        };
        
        if self.bekle(Token::AcKume).is_err() {
            eprintln!("Uyarı: Açılış süslü parantezi bekleniyor");
            return None;
        }
        
        let mut katmanlar = Vec::new();
        let mut ogrenme_orani = 0.01;
        let mut kayip_fonksiyonu = KayipFonksiyonu::MeanSquaredError;
        let mut batch_boyutu = 32_i64;
        let mut epoch_sayisi = 5_i64;
        
        while let Some((token, _)) = self.simdiki() {
            match token {
                Token::Katman => {
                    self.ilerle();
                    if self.bekle(Token::AcParantez).is_err() {
                        eprintln!("Uyarı: Açılış parantezi bekleniyor");
                        return None;
                    }
                    
                    if let Some((Token::Sayi(noron_sayisi), _)) = self.simdiki() {
                        let noron_sayisi = *noron_sayisi;
                        if noron_sayisi <= 0 {
                            eprintln!("Uyarı: Geçersiz nöron sayısı");
                            return None;
                        }
                        self.ilerle();
                        
                        let mut aktivasyon = AktivasyonFonksiyonu::Relu;
                        if self.esle(Token::Virgul) {
                            if self.bekle(Token::Aktivasyon).is_err() {
                                eprintln!("Uyarı: Aktivasyon fonksiyonu bekleniyor");
                                return None;
                            }
                            if self.bekle(Token::Esittir).is_err() {
                                eprintln!("Uyarı: Eşittir işareti bekleniyor");
                                return None;
                            }
                            
                            if let Some((token, _)) = self.simdiki() {
                                aktivasyon = match token {
                                    Token::Relu => AktivasyonFonksiyonu::Relu,
                                    Token::Sigmoid => AktivasyonFonksiyonu::Sigmoid,
                                    Token::Tanh => AktivasyonFonksiyonu::Tanh,
                                    Token::Softmax => AktivasyonFonksiyonu::Softmax,
                                    _ => {
                                        eprintln!("Uyarı: Geçersiz aktivasyon fonksiyonu");
                                        return None;
                                    }
                                };
                                self.ilerle();
                            }
                        }
                        
                        if self.bekle(Token::KapaParantez).is_err() {
                            eprintln!("Uyarı: Kapanış parantezi bekleniyor");
                            return None;
                        }
                        
                        katmanlar.push(KatmanOzellikleri {
                            noron_sayisi: noron_sayisi as i64,
                            aktivasyon,
                        });
                    }
                }
                Token::OgrenmeOrani => {
                    self.ilerle();
                    if self.bekle(Token::Esittir).is_err() {
                        eprintln!("Uyarı: Eşittir işareti bekleniyor");
                        return None;
                    }
                    if let Some((Token::Sayi(oran), _)) = self.simdiki() {
                        let oran = *oran;
                        if oran <= 0 || oran > 100 {
                            eprintln!("Uyarı: Geçersiz öğrenme oranı");
                            return None;
                        }
                        ogrenme_orani = oran as f64 / 100.0;
                        self.ilerle();
                    }
                }
                Token::Kayip => {
                    self.ilerle();
                    if self.bekle(Token::Esittir).is_err() {
                        eprintln!("Uyarı: Eşittir işareti bekleniyor");
                        return None;
                    }
                    if let Some((Token::Metin(kayip), _)) = self.simdiki() {
                        kayip_fonksiyonu = match kayip.as_str() {
                            "mse" => KayipFonksiyonu::MeanSquaredError,
                            "cross_entropy" => KayipFonksiyonu::CrossEntropy,
                            "binary_cross_entropy" => KayipFonksiyonu::BinaryCrossEntropy,
                            _ => {
                                eprintln!("Uyarı: Geçersiz kayıp fonksiyonu");
                                return None;
                            }
                        };
                        self.ilerle();
                    }
                }
                Token::Batch => {
                    self.ilerle();
                    if self.bekle(Token::Esittir).is_err() {
                        eprintln!("Uyarı: Eşittir işareti bekleniyor");
                        return None;
                    }
                    if let Some((Token::Sayi(boyut), _)) = self.simdiki() {
                        let boyut = *boyut;
                        if boyut <= 0 {
                            eprintln!("Uyarı: Geçersiz batch boyutu");
                            return None;
                        }
                        batch_boyutu = boyut as i64;
                        self.ilerle();
                    }
                }
                Token::Epoch => {
                    self.ilerle();
                    if self.bekle(Token::Esittir).is_err() {
                        eprintln!("Uyarı: Eşittir işareti bekleniyor");
                        return None;
                    }
                    if let Some((Token::Sayi(sayi), _)) = self.simdiki() {
                        let sayi = *sayi;
                        if sayi <= 0 {
                            eprintln!("Uyarı: Geçersiz epoch sayısı");
                            return None;
                        }
                        epoch_sayisi = sayi as i64;
                        self.ilerle();
                    }
                }
                Token::KapaKume => {
                    if katmanlar.is_empty() {
                        eprintln!("Uyarı: En az bir katman gerekli");
                        return None;
                    }
                    self.ilerle();
                    eprintln!("Model tanımı tamamlandı: {} ({} katman)", isim, katmanlar.len());
                    return Some(Deyim::ModelTanimi {
                        isim,
                        ozellikler: ModelOzellikleri {
                            katmanlar,
                            ogrenme_orani,
                            kayip_fonksiyonu,
                            batch_boyutu,
                            epoch_sayisi,
                        },
                    });
                }
                _ => self.ilerle(),
            }
        }
        None
    }

    fn egit_parse(&mut self) -> Option<Deyim> {
        eprintln!("Eğitim deyimi ayrıştırılıyor");
        self.ilerle(); // Egit tokenını atla
        
        if self.bekle(Token::AcParantez).is_err() {
            eprintln!("Uyarı: Açılış parantezi bekleniyor");
            return None;
        }
        
        let model = if let Some((Token::Degisken(model), _)) = self.simdiki() {
            eprintln!("Model ismi tespit edildi: {}", model);
            let model = model.to_string();
            self.ilerle();
            model
        } else {
            eprintln!("Uyarı: Model ismi bekleniyor");
            return None;
        };
        
        if self.bekle(Token::Virgul).is_err() {
            eprintln!("Uyarı: Virgül bekleniyor");
            return None;
        }
        
        let egitim_verisi = if let Some(ifade) = self.ifade_parse() {
            eprintln!("Eğitim verisi tespit edildi");
            ifade
        } else {
            eprintln!("Uyarı: Eğitim verisi bekleniyor");
            return None;
        };
        
        if self.bekle(Token::Virgul).is_err() {
            eprintln!("Uyarı: Virgül bekleniyor");
            return None;
        }
        
        let etiketler = if let Some(ifade) = self.ifade_parse() {
            eprintln!("Etiketler tespit edildi");
            ifade
        } else {
            eprintln!("Uyarı: Etiketler bekleniyor");
            return None;
        };
        
        if self.bekle(Token::KapaParantez).is_err() {
            eprintln!("Uyarı: Kapanış parantezi bekleniyor");
            return None;
        }
        
        eprintln!("Eğitim deyimi tamamlandı: {}", model);
        Some(Deyim::ModelEgitimi {
            model,
            egitim_verisi,
            etiketler,
        })
    }

    fn tahmin_parse(&mut self) -> Option<Deyim> {
        eprintln!("Tahmin deyimi ayrıştırılıyor");
        self.ilerle(); // Tahmin tokenını atla
        
        if self.bekle(Token::AcParantez).is_err() {
            eprintln!("Uyarı: Açılış parantezi bekleniyor");
            return None;
        }
        
        let model = if let Some((Token::Degisken(model), _)) = self.simdiki() {
            eprintln!("Model ismi tespit edildi: {}", model);
            let model = model.to_string();
            self.ilerle();
            model
        } else {
            eprintln!("Uyarı: Model ismi bekleniyor");
            return None;
        };
        
        if self.bekle(Token::Virgul).is_err() {
            eprintln!("Uyarı: Virgül bekleniyor");
            return None;
        }
        
        let girdi = if let Some(ifade) = self.ifade_parse() {
            eprintln!("Girdi verisi tespit edildi");
            ifade
        } else {
            eprintln!("Uyarı: Girdi verisi bekleniyor");
            return None;
        };
        
        if self.bekle(Token::KapaParantez).is_err() {
            eprintln!("Uyarı: Kapanış parantezi bekleniyor");
            return None;
        }
        
        eprintln!("Tahmin deyimi tamamlandı: {}", model);
        Some(Deyim::Tahmin { model, girdi })
    }

    #[allow(dead_code)]
    fn eger_parse(&mut self) -> Option<Deyim> {
        eprintln!("Eğer deyimi ayrıştırılıyor");
        self.ilerle(); // Eger tokenını atla
        
        let kosul = if let Some(ifade) = self.ifade_parse() {
            eprintln!("Koşul tespit edildi");
            ifade
        } else {
            eprintln!("Uyarı: Koşul bekleniyor");
            return None;
        };
        
        if self.bekle(Token::AcKume).is_err() {
            eprintln!("Uyarı: Açılış süslü parantezi bekleniyor");
            return None;
        }
        
        let mut govde = Vec::new();
        while let Some((token, _)) = self.simdiki() {
            if token == &Token::KapaKume {
                self.ilerle();
                break;
            }
            if let Some(deyim) = self.deyim_parse() {
                govde.push(deyim);
            } else {
                self.ilerle();
            }
        }
        
        let mut degilse_govde = None;
        if let Some((Token::Degilse, _)) = self.simdiki() {
            eprintln!("Değilse bloğu tespit edildi");
            self.ilerle();
            
            if self.bekle(Token::AcKume).is_err() {
                eprintln!("Uyarı: Açılış süslü parantezi bekleniyor");
                return None;
            }
            
            let mut degilse_deyimleri = Vec::new();
            while let Some((token, _)) = self.simdiki() {
                if token == &Token::KapaKume {
                    self.ilerle();
                    break;
                }
                if let Some(deyim) = self.deyim_parse() {
                    degilse_deyimleri.push(deyim);
                } else {
                    self.ilerle();
                }
            }
            degilse_govde = Some(Box::new(degilse_deyimleri));
        }
        
        eprintln!("Eğer deyimi tamamlandı");
        Some(Deyim::EgerDeyimi {
            kosul,
            govde: Box::new(govde),
            degilse_govde,
        })
    }

    #[allow(dead_code)]
    fn dongu_parse(&mut self) -> Option<Deyim> {
        eprintln!("Döngü deyimi ayrıştırılıyor");
        self.ilerle(); // Dongu tokenını atla
        
        let kosul = if let Some(ifade) = self.ifade_parse() {
            eprintln!("Koşul tespit edildi");
            ifade
        } else {
            eprintln!("Uyarı: Koşul bekleniyor");
            return None;
        };
        
        if self.bekle(Token::AcKume).is_err() {
            eprintln!("Uyarı: Açılış süslü parantezi bekleniyor");
            return None;
        }
        
        let mut govde = Vec::new();
        while let Some((token, _)) = self.simdiki() {
            if token == &Token::KapaKume {
                self.ilerle();
                break;
            }
            if let Some(deyim) = self.deyim_parse() {
                govde.push(deyim);
            } else {
                self.ilerle();
            }
        }
        
        eprintln!("Döngü deyimi tamamlandı");
        Some(Deyim::DonguDeyimi {
            kosul,
            govde: Box::new(govde),
        })
    }

    #[allow(dead_code)]
    fn icin_dongusu_parse(&mut self) -> Option<Deyim> {
        eprintln!("İçin döngüsü ayrıştırılıyor");
        self.ilerle(); // Icin tokenını atla
        
        let degisken = if let Some((Token::Degisken(degisken), _)) = self.simdiki() {
            eprintln!("Değişken tespit edildi: {}", degisken);
            let degisken = degisken.to_string();
            self.ilerle();
            degisken
        } else {
            eprintln!("Uyarı: Değişken bekleniyor");
            return None;
        };
        
        if self.bekle(Token::Icinde).is_err() {
            eprintln!("Uyarı: 'içinde' anahtar kelimesi bekleniyor");
            return None;
        }
        
        let dizi = if let Some(ifade) = self.ifade_parse() {
            eprintln!("Dizi tespit edildi");
            ifade
        } else {
            eprintln!("Uyarı: Dizi bekleniyor");
            return None;
        };
        
        if self.bekle(Token::AcKume).is_err() {
            eprintln!("Uyarı: Açılış süslü parantezi bekleniyor");
            return None;
        }
        
        let mut govde = Vec::new();
        while let Some((token, _)) = self.simdiki() {
            if token == &Token::KapaKume {
                self.ilerle();
                break;
            }
            if let Some(deyim) = self.deyim_parse() {
                govde.push(deyim);
            } else {
                self.ilerle();
            }
        }
        
        eprintln!("İçin döngüsü tamamlandı: {}", degisken);
        Some(Deyim::IcinDongusu {
            degisken,
            dizi,
            govde: Box::new(govde),
        })
    }

    fn karsilastirma_parse(&mut self) -> Option<Ifade> {
        let mut ifade = self.terim_parse()?;
        
        while let Some((token, _)) = self.simdiki() {
            let token = token.clone();
            match token {
                Token::EsitMi | Token::EsitDegilMi | Token::BuyukMu | 
                Token::KucukMu | Token::BuyukEsitMi | Token::KucukEsitMi => {
                    let operator = match token {
                        Token::EsitMi => KarsilastirmaOperator::EsitMi,
                        Token::EsitDegilMi => KarsilastirmaOperator::EsitDegilMi,
                        Token::BuyukMu => KarsilastirmaOperator::BuyukMu,
                        Token::KucukMu => KarsilastirmaOperator::KucukMu,
                        Token::BuyukEsitMi => KarsilastirmaOperator::BuyukEsitMi,
                        Token::KucukEsitMi => KarsilastirmaOperator::KucukEsitMi,
                        _ => unreachable!(),
                    };
                    self.ilerle();
                    if let Some(sag) = self.terim_parse() {
                        ifade = Ifade::KarsilastirmaIfadesi {
                            sol: Box::new(ifade),
                            operator,
                            sag: Box::new(sag),
                        };
                    }
                }
                _ => break,
            }
        }
        Some(ifade)
    }

    fn veri_ayristir(&mut self) -> Option<Deyim> {
        eprintln!("Veri ayrıştırma başladı");
        self.ilerle(); // "Veri" tokenını tüket
        
        let isim = if let Some((Token::Degisken(isim), _)) = self.simdiki() {
            eprintln!("Değişken ismi tespit edildi: {}", isim);
            let isim = isim.to_string();
            self.ilerle();
            isim
        } else {
            eprintln!("Uyarı: Değişken ismi bekleniyor");
            return None;
        };
        
        if self.bekle(Token::Esittir).is_err() {
            eprintln!("Uyarı: Eşittir işareti bekleniyor");
            return None;
        }
        
        let deger = if let Some(ifade) = self.ifade_parse() {
            eprintln!("Değer tespit edildi: {:?}", ifade);
            ifade
        } else {
            eprintln!("Uyarı: Değer bekleniyor");
            return None;
        };
        
        eprintln!("Veri ayrıştırma tamamlandı: {} = {:?}", isim, deger);
        Some(Deyim::DegiskenTanimi { isim, deger })
    }

    #[allow(dead_code)]
    fn dondur_parse(&mut self) -> Option<Deyim> {
        eprintln!("Döndürme deyimi ayrıştırılıyor");
        self.ilerle(); // Dondur tokenını atla
        
        if let Some(ifade) = self.ifade_parse() {
            eprintln!("Döndürme değeri tespit edildi: {:?}", ifade);
            Some(Deyim::DondurDeyimi(ifade))
        } else {
            eprintln!("Uyarı: Döndürme değeri bekleniyor");
            None
        }
    }

    #[allow(dead_code)]
    fn fonksiyon_tanimi_parse(&mut self) -> Option<Deyim> {
        eprintln!("Fonksiyon tanımı ayrıştırılıyor");
        self.ilerle(); // Fonksiyon tokenını atla
        
        let isim = if let Some((Token::Degisken(isim), _)) = self.simdiki() {
            eprintln!("Fonksiyon ismi tespit edildi: {}", isim);
            let isim = isim.to_string();
            self.ilerle();
            isim
        } else {
            eprintln!("Uyarı: Fonksiyon ismi bekleniyor");
            return None;
        };
        
        if self.bekle(Token::AcParantez).is_err() {
            eprintln!("Uyarı: Açılış parantezi bekleniyor");
            return None;
        }
        
        let mut parametreler = Vec::new();
        while let Some((token, _)) = self.simdiki() {
            match token {
                Token::KapaParantez => {
                    self.ilerle();
                    break;
                }
                Token::Degisken(param) => {
                    eprintln!("Parametre tespit edildi: {}", param);
                    parametreler.push(param.to_string());
                    self.ilerle();
                }
                Token::Virgul => {
                    self.ilerle();
                }
                _ => {
                    eprintln!("Uyarı: Geçersiz parametre");
                    return None;
                }
            }
        }
        
        if self.bekle(Token::AcKume).is_err() {
            eprintln!("Uyarı: Açılış süslü parantezi bekleniyor");
            return None;
        }
        
        let mut govde = Vec::new();
        while let Some((token, _)) = self.simdiki() {
            if token == &Token::KapaKume {
                self.ilerle();
                break;
            }
            if let Some(deyim) = self.deyim_parse() {
                govde.push(deyim);
            } else {
                self.ilerle();
            }
        }
        
        eprintln!("Fonksiyon tanımı tamamlandı: {} ({} parametre)", isim, parametreler.len());
        Some(Deyim::FonksiyonTanimi {
            isim,
            parametreler,
            govde: Box::new(govde),
        })
    }

    #[allow(dead_code)]
    fn degisken_ismi_parse(&mut self) -> Result<String, ParseError> {
        let isim = if let Some((Token::Degisken(isim), _)) = self.simdiki() {
            isim.clone()
        } else {
            return Err(ParseError::BeklenenToken("değişken ismi".to_string()));
        };
        self.ilerle();
        Ok(isim.to_string())
    }

    #[allow(dead_code)]
    fn token_ilerlet(&mut self) {
        self.ilerle();
    }

    #[allow(dead_code)]
    fn mantiksal_parse(&mut self) -> Option<Ifade> {
        let mut ifade = self.karsilastirma_parse()?;

        while let Some((token, _)) = self.simdiki() {
            let token = token.clone();
            match token {
                Token::Ve | Token::Veya => {
                    let operator = match token {
                        Token::Ve => MantiksalOperator::Ve,
                        Token::Veya => MantiksalOperator::Veya,
                        _ => unreachable!(),
                    };
                    self.ilerle();
                    if let Some(sag) = self.karsilastirma_parse() {
                        ifade = Ifade::MantiksalIfade {
                            sol: Box::new(ifade),
                            operator,
                            sag: Box::new(sag),
                        };
                    }
                }
                _ => break,
            }
        }
        Some(ifade)
    }
} 
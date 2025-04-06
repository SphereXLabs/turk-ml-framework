use logos::Logos;
use std::collections::HashMap;
use turk_lang::ast::{Deyim, Ifade, Program, DegerTipi, AktivasyonFonksiyonu, OptimizerTipi};
use turk_lang::token::Token;
use turk_lang::model::Sequential;
use turk_lang::layer::{Activation, Dense};
use turk_lang::optimizer::SGD;
use turk_lang::Result;
use turk_lang::parser::model_parser::ModelParser;
use turk_lang::TurkError;
use log::info;
use env_logger;

pub struct Yorumlayici {
    degiskenler: HashMap<String, DegerTipi>,
    modeller: HashMap<String, Sequential>,
}

impl Yorumlayici {
    pub fn new() -> Self {
        Self {
            degiskenler: HashMap::new(),
            modeller: HashMap::new(),
        }
    }

    pub fn yorumla(&mut self, program: &Program) -> Result<()> {
        for deyim in &program.deyimler {
            self.deyim_yorumla(deyim)?;
        }
        Ok(())
    }

    fn deyim_yorumla(&mut self, deyim: &Deyim) -> Result<()> {
        match deyim {
            Deyim::DegiskenTanimi { isim, deger } => {
                let deger = self.ifade_yorumla(deger.clone())?;
                let deger_klon = deger.clone();
                self.degiskenler.insert(isim.clone(), deger);
                eprintln!("Değişken tanımlandı: {} = {:?}", isim, deger_klon);
                Ok(())
            }
            Deyim::VeriTanimi { isim, deger } => {
                let deger = self.ifade_yorumla(*deger.clone())?;
                let deger_klon = deger.clone();
                self.degiskenler.insert(isim.clone(), deger);
                eprintln!("Veri tanımlandı: {} = {:?}", isim, deger_klon);
                Ok(())
            }
            Deyim::Egitim { model, x_train, y_train } => {
                eprintln!("Model eğitimi başlatılıyor: {}", model);
                
                // Önce verileri hazırlayalım
                let x_train_data = self.ifade_yorumla(*x_train.clone())?.to_array()?;
                let y_train_data = self.ifade_yorumla(*y_train.clone())?.to_array()?;
                
                // Sonra modeli alalım
                let model_ref = self.modeller.get_mut(model)
                    .ok_or_else(|| TurkError::InvalidInput(
                        format!("Model bulunamadı: {}", model)
                    ))?;
                
                // Model derlenmediyse, varsayılan optimizer ile derle
                if !model_ref.is_compiled() {
                    model_ref.compile_boxed(Box::new(SGD::new(0.01)));
                }
                
                // Sonra eğit
                model_ref.fit(&x_train_data, &y_train_data, 10, 32, None)?;
                eprintln!("Model eğitimi tamamlandı: {}", model);
                Ok(())
            }
            Deyim::ModelTanimi { isim, ozellikler } => {
                eprintln!("Model tanımlanıyor: {}", isim);
                eprintln!("Model özellikleri: {:?}", ozellikler);
                
                let mut model = Sequential::new();
                
                // Katmanları oluştur
                for (i, katman) in ozellikler.katmanlar.iter().enumerate() {
                    let input_size = if i == 0 {
                        katman.noron_sayisi as usize // İlk katmanın nöron sayısı giriş boyutu olur
                    } else {
                        ozellikler.katmanlar[i-1].noron_sayisi as usize
                    };
                    
                    let output_size = katman.noron_sayisi as usize;
                    
                    let aktivasyon = match katman.aktivasyon {
                        AktivasyonFonksiyonu::Relu => Activation::Relu,
                        AktivasyonFonksiyonu::Sigmoid => Activation::Sigmoid,
                        AktivasyonFonksiyonu::Tanh => Activation::Tanh,
                        AktivasyonFonksiyonu::Softmax => Activation::Softmax,
                        AktivasyonFonksiyonu::LeakyRelu(alpha) => Activation::LeakyRelu(alpha as f32),
                        AktivasyonFonksiyonu::Swish => Activation::Swish,
                    };
                    
                    model.add(Dense::new(input_size, output_size, aktivasyon));
                }
                
                // Varsayılan yapılandırma değerlerini ayarla
                let optimizer: Box<dyn turk_lang::optimizer::Optimizer> = match &ozellikler.optimizer {
                    OptimizerTipi::SGD { lr, momentum, nesterov: _ } => {
                        if let Some(momentum_val) = momentum {
                            Box::new(turk_lang::optimizer::SGD::with_momentum(*lr as f32, *momentum_val as f32))
                        } else {
                            Box::new(turk_lang::optimizer::SGD::new(*lr as f32))
                        }
                    },
                    OptimizerTipi::Adam { lr, beta1, beta2, epsilon } => {
                        Box::new(turk_lang::optimizer::Adam::with_betas(*lr as f32, *beta1 as f32, *beta2 as f32, *epsilon as f32))
                    },
                    OptimizerTipi::AdamW { lr, weight_decay, beta1, beta2 } => {
                        Box::new(turk_lang::optimizer::AdamW::with_betas(*lr as f32, *weight_decay as f32, *beta1 as f32, *beta2 as f32, 1e-8))
                    },
                    OptimizerTipi::RMSProp { lr, rho, epsilon } => {
                        Box::new(turk_lang::optimizer::RMSProp::with_decay(
                            *lr as f32, *rho as f32, *epsilon as f32
                        ))
                    },
                };
                
                // Modeli derle
                model.compile_boxed(optimizer);
                
                self.modeller.insert(isim.clone(), model);
                eprintln!("Model başarıyla tanımlandı: {}", isim);
                Ok(())
            }
            Deyim::ModelEgitimi { model, egitim_verisi, etiketler } => {
                eprintln!("Model eğitimi başlatılıyor: {}", model);
                
                // Önce verileri hazırlayalım
                let x_train = self.ifade_yorumla(egitim_verisi.clone())?.to_array()?;
                let y_train = self.ifade_yorumla(etiketler.clone())?.to_array()?;
                
                // Sonra modeli alalım
                let model_ref = self.modeller.get_mut(model)
                    .ok_or_else(|| TurkError::InvalidInput(
                        format!("Model bulunamadı: {}", model)
                    ))?;
                
                // Model derlenmediyse, varsayılan optimizer ile derle
                if !model_ref.is_compiled() {
                    model_ref.compile_boxed(Box::new(SGD::new(0.01)));
                }
                
                // Sonra eğit
                model_ref.fit(&x_train, &y_train, 10, 32, None)?;
                eprintln!("Model eğitimi tamamlandı: {}", model);
                Ok(())
            }
            Deyim::Tahmin { model, girdi } => {
                eprintln!("Tahmin yapılıyor: {}", model);
                
                let mut x = self.ifade_yorumla(girdi.clone())?.to_array()?;
                
                // Eğer giriş tek boyutlu ise (n,) şeklinde, onu (1,n) şekline dönüştür
                if x.shape()[0] == 1 && x.shape()[1] != 1 {
                    // Zaten doğru şekilde (1,n)
                } else if x.ndim() == 2 && x.shape()[1] == 1 {
                    // (n,1) şeklinde, transpoze et
                    x = x.t().to_owned();
                } else {
                    // Tek örnek olarak yeniden şekillendir
                    let n_cols = x.len();
                    x = x.into_shape((1, n_cols))
                        .map_err(|e| TurkError::NdarrayError(e.into()))?;
                }
                
                let model_ref = self.modeller.get_mut(model)
                    .ok_or_else(|| TurkError::InvalidInput(
                        format!("Model bulunamadı: {}", model)
                    ))?;
                
                let tahmin = model_ref.forward(&x)?;
                
                eprintln!("Tahmin sonucu: {:?}", tahmin);
                Ok(())
            }
            Deyim::Atama { isim, deger } => {
                let deger = self.ifade_yorumla(deger.clone())?;
                self.degiskenler.insert(isim.clone(), deger);
                Ok(())
            }
            Deyim::IfadeDeyimi(ifade) => {
                let _ = self.ifade_yorumla(ifade.clone())?;
                Ok(())
            },
            Deyim::EgerDeyimi { kosul, govde, degilse_govde } => {
                let kosul_degeri = self.ifade_yorumla(kosul.clone())?;
                match kosul_degeri {
                    DegerTipi::Mantiksal(true) => {
                        for deyim in govde.iter() {
                            self.deyim_yorumla(deyim)?;
                        }
                        Ok(())
                    }
                    DegerTipi::Mantiksal(false) => {
                        if let Some(degilse) = degilse_govde {
                            for deyim in degilse.iter() {
                                self.deyim_yorumla(deyim)?;
                            }
                        }
                        Ok(())
                    }
                    _ => Err(TurkError::InvalidInput("Koşul ifadesi mantıksal bir değer döndürmeli".to_string()))
                }
            }
            Deyim::DonguDeyimi { kosul, govde } => {
                while let DegerTipi::Mantiksal(true) = self.ifade_yorumla(kosul.clone())? {
                    for deyim in govde.iter() {
                        self.deyim_yorumla(deyim)?;
                    }
                }
                Ok(())
            }
            Deyim::IcinDongusu { degisken, dizi, govde } => {
                let dizi_degeri = self.ifade_yorumla(dizi.clone())?;
                if let DegerTipi::Dizi(elemanlar) = dizi_degeri {
                    for eleman in elemanlar {
                        self.degiskenler.insert(degisken.clone(), eleman);
                        for deyim in govde.iter() {
                            self.deyim_yorumla(deyim)?;
                        }
                    }
                    Ok(())
                } else {
                    Err(TurkError::InvalidInput("Döngü için dizi bekleniyor".to_string()))
                }
            }
            Deyim::KirDeyimi => Ok(()),
            Deyim::DevamDeyimi => {
                println!("DevamDeyimi çalıştırıldı (şimdilik döngü dışı)");
                Ok(())
            }
            Deyim::Blok(deyimler) => {
                for deyim in deyimler {
                    self.deyim_yorumla(deyim)?;
                }
                Ok(())
            }
            Deyim::FonksiyonTanimi { isim, parametreler: _, govde: _ } => {
                println!("Fonksiyon tanımı: {}", isim);
                Ok(())
            }
            Deyim::DondurDeyimi(deger) => {
                let _ = self.ifade_yorumla(deger.clone())?;
                Ok(())
            }
            Deyim::Ifade(ifade) => {
                println!("Ifade çalıştırıldı: {:?}", ifade);
                let _ = self.ifade_yorumla(ifade.clone())?;
                Ok(())
            }
            Deyim::ModelKaydetme { model, dosya_yolu } => {
                eprintln!("Model kaydediliyor: {} -> {}", model, dosya_yolu);
                
                let model_ref = self.modeller.get(model)
                    .ok_or_else(|| TurkError::InvalidInput(
                        format!("Model bulunamadı: {}", model)
                    ))?;
                
                model_ref.save(dosya_yolu)?;
                eprintln!("Model başarıyla kaydedildi: {}", dosya_yolu);
                Ok(())
            }
            Deyim::ModelYukleme { isim, dosya_yolu } => {
                eprintln!("Model yükleniyor: {} <- {}", isim, dosya_yolu);
                
                let mut model = Sequential::new();
                model.load(dosya_yolu)?;
                
                self.modeller.insert(isim.clone(), model);
                eprintln!("Model başarıyla yüklendi: {}", isim);
                Ok(())
            }
            Deyim::ModelDegerlendirme { model, test_verisi, test_etiketleri } => {
                eprintln!("Model değerlendiriliyor: {}", model);
                
                let x_test = self.ifade_yorumla(test_verisi.clone())?.to_array()?;
                let _y_test = self.ifade_yorumla(test_etiketleri.clone())?.to_array()?;
                
                let model_ref = self.modeller.get_mut(model)
                    .ok_or_else(|| TurkError::InvalidInput(
                        format!("Model bulunamadı: {}", model)
                    ))?;
                
                // Basit bir doğruluk hesabı
                let tahminler = model_ref.forward(&x_test)?;
                // TODO: Gerçek metrik hesaplamaları
                
                eprintln!("Model değerlendirme sonuçları: {:?}", tahminler);
                Ok(())
            }
            Deyim::ModelOzeti { model } => {
                eprintln!("Model özeti gösteriliyor: {}", model);
                
                let model_ref = self.modeller.get(model)
                    .ok_or_else(|| TurkError::InvalidInput(
                        format!("Model bulunamadı: {}", model)
                    ))?;
                
                eprintln!("{}", model_ref.summary());
                Ok(())
            }
            Deyim::VeriGorsellestirme { .. } => {
                eprintln!("Veri görselleştirme: Şu an desteklenmiyor");
                Ok(())
            }
            Deyim::ImportDeyimi { .. } => {
                eprintln!("Import deyimi: Şu an desteklenmiyor");
                Ok(())
            }
            Deyim::TryCatch { .. } => {
                eprintln!("Try-Catch deyimi: Şu an desteklenmiyor");
                Ok(())
            }
            Deyim::LambdaTanimi { .. } => {
                eprintln!("Lambda tanımı: Şu an desteklenmiyor");
                Ok(())
            }
        }
    }

    fn ifade_yorumla(&mut self, ifade: Ifade) -> Result<DegerTipi> {
        match ifade {
            Ifade::Sayi(n) => Ok(DegerTipi::Sayi(n as f32)),
            Ifade::Metin(s) => Ok(DegerTipi::Metin(s)),
            Ifade::Mantiksal(b) => Ok(DegerTipi::Mantiksal(b)),
            Ifade::Degisken(isim) => {
                self.degiskenler.get(&isim)
                    .cloned()
                    .ok_or_else(|| TurkError::InvalidInput(format!("Tanımlanmamış değişken: {}", isim)))
            }
            Ifade::Dizi(elemanlar) => {
                let mut deger_listesi = Vec::new();
                for eleman in elemanlar {
                    deger_listesi.push(self.ifade_yorumla(eleman)?);
                }
                Ok(DegerTipi::Dizi(deger_listesi))
            }
            Ifade::Liste(elemanlar) => {
                let mut deger_listesi = Vec::new();
                for eleman in elemanlar {
                    deger_listesi.push(self.ifade_yorumla(eleman)?);
                }
                Ok(DegerTipi::Dizi(deger_listesi))
            }
            Ifade::FonksiyonCagrisi { fonksiyon, argumanlar } => {
                let fonksiyon_ismi = match *fonksiyon {
                    Ifade::Degisken(ref isim) => isim.clone(),
                    _ => return Err(TurkError::InvalidInput("Fonksiyon ismi bir değişken olmalı".to_string())),
                };
                
                match fonksiyon_ismi.as_str() {
                    "Egit" => {
                        if argumanlar.len() != 3 {
                            return Err(TurkError::InvalidInput("Egit fonksiyonu 3 argüman bekliyor".to_string()));
                        }
                        
                        let _model = self.ifade_yorumla(argumanlar[0].clone())?;
                        let _egitim_verisi = self.ifade_yorumla(argumanlar[1].clone())?;
                        let _etiketler = self.ifade_yorumla(argumanlar[2].clone())?;
                        
                        Ok(DegerTipi::Birim)
                    }
                    _ => Err(TurkError::InvalidInput(format!("Bilinmeyen fonksiyon: {}", fonksiyon_ismi))),
                }
            }
            _ => Err(TurkError::InvalidInput("Desteklenmeyen ifade türü".to_string())),
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    info!("TURK Yorumlayıcı Başlatıldı");
    let kaynak = r#"
        Model sinir_agi {
            Katman(3, Aktivasyon = Relu)
            Katman(5, Aktivasyon = Relu)
            Katman(2, Aktivasyon = Softmax)
            OgrenmeOrani = 0,1
            Kayip = "cross_entropy"
            Batch = 32
            Epoch = 5
        }
        Veri x_train = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        Veri y_train = [[1, 0], [0, 1], [1, 0]]
        Egit(sinir_agi, x_train, y_train)
        Veri test_verisi = [1, 0, 0]
        Tahmin(sinir_agi, test_verisi)
    "#;

    let mut lexer = Token::lexer(kaynak);
    let mut tokens = Vec::new();
    println!("Token'lar üretiliyor:");
    while let Some(token) = lexer.next() {
        match token {
            Ok(token) => {
                println!("Token: {:?}", token);
                tokens.push((token, lexer.slice().to_string()));
            }
            Err(e) => {
                eprintln!("Lexer hatası: {:?}", e);
                return Err(TurkError::LexerError(format!("{:?}", e)));
            }
        }
    }
    println!("Token'lar üretildi, toplam: {}", tokens.len());

    let mut parser = ModelParser::new(tokens);
    let program = parser.parse()?;

    println!("Program başarıyla ayrıştırıldı:");
    for deyim in &program.deyimler {
        println!("{:#?}", deyim);
    }

    let mut yorumlayici = Yorumlayici::new();
    match yorumlayici.yorumla(&program) {
        Ok(_) => println!("Program başarıyla çalıştırıldı."),
        Err(e) => eprintln!("Çalıştırma hatası: {}", e),
    }

    Ok(())
} 
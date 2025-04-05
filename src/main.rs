mod ast;
mod parser;
mod token;
mod model;
mod layer;
mod optimizer;
mod loss;

use logos::Logos;
use std::collections::HashMap;
use crate::ast::{Deyim, Ifade, Program, YapayZekaModeli, DegerTipi};
use crate::token::Token;
use crate::model::{Sequential, Dense};
use crate::layer::Activation;
use crate::optimizer::Adam;
use crate::loss::CrossEntropy;
use ndarray::Array2;
use crate::Result;

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
                let deger = self.ifade_yorumla(deger)?;
                self.degiskenler.insert(isim.clone(), deger);
                eprintln!("Değişken tanımlandı: {} = {:?}", isim, deger);
                Ok(())
            }
            Deyim::ModelTanimi { isim, ozellikler } => {
                eprintln!("Model tanımlanıyor: {}", isim);
                eprintln!("Model özellikleri: {:?}", ozellikler);
                
                let mut model = Sequential::new();
                
                // Katmanları oluştur
                for (i, katman) in ozellikler.katmanlar.iter().enumerate() {
                    let input_size = if i == 0 {
                        ozellikler.giris_boyutu
                    } else {
                        ozellikler.katmanlar[i-1].noron_sayisi
                    };
                    
                    let output_size = if i == ozellikler.katmanlar.len() - 1 {
                        ozellikler.cikis_boyutu
                    } else {
                        katman.noron_sayisi
                    };
                    
                    let aktivasyon = match katman.aktivasyon.as_str() {
                        "relu" => Activation::Relu,
                        "sigmoid" => Activation::Sigmoid,
                        "tanh" => Activation::Tanh,
                        "softmax" => Activation::Softmax,
                        _ => return Err(crate::TinyMLError::InvalidInput(
                            format!("Geçersiz aktivasyon fonksiyonu: {}", katman.aktivasyon)
                        )),
                    };
                    
                    model.add(Dense::new(input_size, output_size, aktivasyon));
                }
                
                // Modeli derle
                model.compile(
                    Adam::new(ozellikler.ogrenme_orani),
                    CrossEntropy::default()
                );
                
                self.modeller.insert(isim.clone(), model);
                eprintln!("Model başarıyla tanımlandı: {}", isim);
                Ok(())
            }
            Deyim::ModelEgitimi { model, egitim_verisi, etiketler } => {
                eprintln!("Model eğitimi başlatılıyor: {}", model);
                
                let model = self.modeller.get_mut(model)
                    .ok_or_else(|| crate::TinyMLError::InvalidInput(
                        format!("Model bulunamadı: {}", model)
                    ))?;
                
                let x_train = self.ifade_yorumla(egitim_verisi)?.to_array()?;
                let y_train = self.ifade_yorumla(etiketler)?.to_array()?;
                
                model.fit(&x_train, &y_train, 10, 32)?;
                eprintln!("Model eğitimi tamamlandı: {}", model);
                Ok(())
            }
            Deyim::Tahmin { model, girdi } => {
                eprintln!("Tahmin yapılıyor: {}", model);
                
                let model = self.modeller.get(model)
                    .ok_or_else(|| crate::TinyMLError::InvalidInput(
                        format!("Model bulunamadı: {}", model)
                    ))?;
                
                let x = self.ifade_yorumla(girdi)?.to_array()?;
                let tahmin = model.forward(&x)?;
                
                eprintln!("Tahmin sonucu: {:?}", tahmin);
                Ok(())
            }
            Deyim::Atama { isim, deger } => {
                let deger = self.ifade_yorumla(deger)?;
                self.degiskenler.insert(isim, deger.clone());
                Ok(())
            }
            Deyim::IfadeDeyimi(ifade) => self.ifade_yorumla(ifade),
            Deyim::EgerDeyimi { kosul, govde, degilse_govde } => {
                let kosul_degeri = self.ifade_yorumla(kosul)?;
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
                    _ => Err("Koşul ifadesi mantıksal bir değer döndürmeli".to_string())
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
                let dizi_degeri = self.ifade_yorumla(dizi)?;
                if let DegerTipi::Dizi(elemanlar) = dizi_degeri {
                    for eleman in elemanlar {
                        self.degiskenler.insert(degisken.clone(), eleman);
                        for deyim in govde.iter() {
                            self.deyim_yorumla(deyim)?;
                        }
                    }
                    Ok(())
                } else {
                    Err("Döngü için dizi bekleniyor".to_string())
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
                self.ifade_yorumla(deger)
            }
            Deyim::Ifade(ifade) => {
                println!("Ifade çalıştırıldı: {:?}", ifade);
                self.ifade_yorumla(ifade)
            }
        }
    }

    fn ifade_yorumla(&mut self, ifade: Ifade) -> Result<DegerTipi> {
        match ifade {
            Ifade::Sayi(n) => Ok(DegerTipi::Sayi(n)),
            Ifade::Metin(s) => Ok(DegerTipi::Metin(s)),
            Ifade::Mantiksal(b) => Ok(DegerTipi::Mantiksal(b)),
            Ifade::Degisken(isim) => {
                self.degiskenler.get(&isim)
                    .cloned()
                    .ok_or_else(|| format!("Değişken bulunamadı: {}", isim))
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
                    _ => return Err("Fonksiyon ismi bir değişken olmalı".to_string()),
                };
                
                match fonksiyon_ismi.as_str() {
                    "Egit" => {
                        if argumanlar.len() != 3 {
                            return Err("Egit fonksiyonu 3 argüman bekliyor".to_string());
                        }
                        
                        let model = self.ifade_yorumla(argumanlar[0].clone())?;
                        let egitim_verisi = self.ifade_yorumla(argumanlar[1].clone())?;
                        let etiketler = self.ifade_yorumla(argumanlar[2].clone())?;
                        
                        self.model_egit(model, egitim_verisi, etiketler)
                    }
                    _ => Err(format!("Bilinmeyen fonksiyon: {}", fonksiyon_ismi)),
                }
            }
            _ => Err("Desteklenmeyen ifade türü".to_string()),
        }
    }

    fn model_egit(&mut self, model: DegerTipi, egitim_verisi: DegerTipi, etiketler: DegerTipi) -> Result<DegerTipi> {
        if let DegerTipi::Model(model_ozellikleri) = model {
            // Model eğitimi simülasyonu
            eprintln!("Model eğitiliyor...");
            eprintln!("Eğitim verisi: {:?}", egitim_verisi);
            eprintln!("Etiketler: {:?}", etiketler);
            Ok(DegerTipi::Model(model_ozellikleri))
        } else {
            Err("Geçersiz model tipi".to_string())
        }
    }

    fn model_tahmin(&self, model: &DegerTipi, girdi: DegerTipi) -> Result<DegerTipi> {
        if let DegerTipi::Model(model_ozellikleri) = model {
            match girdi {
                DegerTipi::Dizi(veriler) => {
                    println!("Tahmin yapılıyor...");
                    println!("Girdi verisi: {:?}", veriler);
                    Ok(DegerTipi::Dizi(Vec::new()))
                }
                _ => Err("Geçersiz girdi verisi".to_string())
            }
        } else {
            Err("Geçersiz model tipi".to_string())
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kaynak = r#"
        Model sinir_agi {
            Katman(784, Aktivasyon = Relu)
            Katman(128, Aktivasyon = Relu)
            Katman(10, Aktivasyon = Softmax)
            OgrenmeOrani = 1
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
                return Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("Lexer hatası: {:?}", e))));
            }
        }
    }
    println!("Token'lar üretildi, toplam: {}", tokens.len());

    let mut parser = parser::Parser::new(tokens);
    let program = parser.parse().map_err(|e| {
        eprintln!("Parser hatası: {}", e);
        std::io::Error::new(std::io::ErrorKind::Other, format!("Parser hatası: {}", e))
    })?;

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
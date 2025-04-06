# TURK Dili Referansı

Bu belge, TURK programlama dilinin yapısını, sözdizimini ve temel bileşenlerini açıklamaktadır.

## İçindekiler

- [Giriş](#giriş)
- [Temel Veri Tipleri](#temel-veri-tipleri)
- [Değişkenler ve Atamalar](#değişkenler-ve-atamalar)
- [Operatörler](#operatörler)
- [Kontrol Yapıları](#kontrol-yapıları)
- [Fonksiyonlar](#fonksiyonlar)
- [Veri İşleme](#veri-işleme)
- [Model Tanımlama](#model-tanımlama)
- [Katmanlar](#katmanlar)
- [Optimizerlar](#optimizerlar)
- [Kayıp Fonksiyonları](#kayıp-fonksiyonları)
- [Metrikler](#metrikler)
- [Eğitim ve Değerlendirme](#eğitim-ve-değerlendirme)
- [Kaydetme ve Yükleme](#kaydetme-ve-yükleme)
- [Örnek Uygulamalar](#örnek-uygulamalar)

## Giriş

TURK, Türkçe yapay zeka programlama dilidir. Makine öğrenmesi ve derin öğrenme uygulamalarını Türkçe sözdizimi kullanarak geliştirmenizi sağlar. Python veya diğer dillere benzer bir yapıda, ancak tamamen Türkçe anahtar kelimeler ve doğal bir dil yapısı kullanır.

Örnek bir TURK programı:

```turk
// TURK ile basit bir sinir ağı
Veri egitim = Veri_Yukle("veriler.csv")

Model SinirAgi {
    Katman(10, 5, aktivasyon=Relu)
    Katman(5, 1, aktivasyon=Sigmoid)
    Optimizer = SGD(ogrenme_orani=0.01)
    Kayip = MSE
}

Egit(SinirAgi, egitim.x, egitim.y, epoch=100)
sonuclar = Degerlendir(SinirAgi, egitim.x, egitim.y)
Yazdir("Doğruluk:", sonuclar.dogruluk)
```

## Temel Veri Tipleri

TURK dili aşağıdaki temel veri tiplerini destekler:

| Tip | Açıklama | Örnek |
|-----|----------|-------|
| Sayi | Tam sayılar | `5`, `42`, `1000` |
| OndalikSayi | Ondalık sayılar | `3.14`, `0.01`, `2,5` |
| Metin | Metin verileri | `"Merhaba"`, `"TURK Dili"` |
| Mantiksal | Doğru/Yanlış değerleri | `Dogru`, `Yanlis` |
| Dizi | Değer dizileri | `[1, 2, 3, 4]`, `["a", "b", "c"]` |
| Matriks | 2B veri yapıları | `[[1, 2], [3, 4]]` |
| VeriKumesi | Eğitim verileri | `Veri_Yukle("veriler.csv")` |
| Model | Yapay zeka modelleri | `Model { ... }` |

## Değişkenler ve Atamalar

Değişken adları Türkçe karakterler de dahil olmak üzere alfanümerik karakterler ve alt çizgi (`_`) içerebilir. Değişkenler tanımlanırken tür belirtmeye gerek yoktur, değer üzerinden otomatik olarak tip çıkarımı yapılır.

```turk
// Değişken tanımlama
sayi = 42
pi = a
metin = "Merhaba Dünya"
liste = [1, 2, 3, 4, 5]
matris = [[1, 2], [3, 4]]
durumlar = [Dogru, Yanlis, Dogru]

// Özel veri tipi tanımlama
Veri egitim = Veri_Yukle("veri.csv")
```

## Operatörler

TURK dili aşağıdaki operatörleri destekler:

### Aritmetik Operatörler

| Operatör | Açıklama | Örnek |
|----------|----------|-------|
| `+` | Toplama | `a + b` |
| `-` | Çıkarma | `a - b` |
| `*` | Çarpma | `a * b` |
| `/` | Bölme | `a / b` |
| `%` | Modül (kalan) | `a % b` |
| `^` | Üs alma | `a ^ b` |

### Karşılaştırma Operatörleri

| Operatör | Açıklama | Örnek |
|----------|----------|-------|
| `==` | Eşitlik | `a == b` |
| `!=` | Eşit değil | `a != b` |
| `>` | Büyüktür | `a > b` |
| `<` | Küçüktür | `a < b` |
| `>=` | Büyük eşit | `a >= b` |
| `<=` | Küçük eşit | `a <= b` |

### Mantıksal Operatörler

| Operatör | Açıklama | Örnek |
|----------|----------|-------|
| `ve` | Mantıksal VE | `a > 5 ve b < 10` |
| `veya` | Mantıksal VEYA | `a > 5 veya b < 10` |
| `degil` | Mantıksal değil | `degil (a == b)` |

### Matris Operatörleri

| Operatör | Açıklama | Örnek |
|----------|----------|-------|
| `@` | Matris çarpımı | `A @ B` |
| `.` | Nokta (eleman bazlı) çarpım | `A . B` |
| `~` | Transpoze | `~A` |

## Kontrol Yapıları

### Koşul Yapıları

```turk
// Basit if-else
Eger kosul {
    // Kod bloğu
} Degilse {
    // Kod bloğu
}

// İç içe if-else
Eger kosul1 {
    // Kod bloğu
} Degilse Eger kosul2 {
    // Kod bloğu
} Degilse {
    // Kod bloğu
}

// Tek satır if-else
sonuc = Eger kosul Eger deger1 Degilse deger2
```

### Döngüler

```turk
// While döngüsü
Dongu kosul {
    // Kod bloğu
}

// For döngüsü (diziler üzerinde)
Icin eleman Icinde dizi {
    // Kod bloğu
}

// İndeks ve değer ile for döngüsü
Icin i, deger Icinde dizi {
    Yazdir("İndeks:", i, "Değer:", deger)
}

// Sayısal aralık ile for döngüsü
Icin i Icinde 1:10 {
    Yazdir(i)
}

// Kırma ve devam etme
Icin i Icinde 1:10 {
    Eger i == 5 {
        Devam
    }
    Eger i == 8 {
        Kir
    }
    Yazdir(i)
}
```

## Fonksiyonlar

```turk
// Basit fonksiyon tanımlama
Fonksiyon topla(a, b) {
    Dondur a + b
}

// Fonksiyon çağırma
sonuc = topla(3, 5)

// Varsayılan parametreler
Fonksiyon guc(sayi, us=2) {
    Dondur sayi ^ us
}

// Lambda fonksiyonlar
kare = (x) => x * x
liste_kareleri = liste.haritala(kare)
```

## Veri İşleme

TURK dili, veri yükleme ve ön işleme için aşağıdaki işlevleri destekler:

```turk
// Veri yükleme
Veri veri = Veri_Yukle("veri.csv", format="csv")
Veri goruntular = ResimYukle("resimler/", boyut=(28, 28))

// Veri bölme
Veri egitim, test = Veri_Bol(veri, oran=0.8, karistir=Dogru)

// Normalizasyon
egitim.x = Normallestir(egitim.x)
test.x = Normallestir(test.x)

// One-hot kodlama
egitim.y = OneHotKodla(egitim.y, sinif_sayisi=10)
test.y = OneHotKodla(test.y, sinif_sayisi=10)

// Metin vektörleştirme
metin_vektorleri = MetinVektorlestir(metinler, yontem="tfidf", max_ozellik=5000)

// İstatistiksel işlemler
ortalama = Ortalama(veriler)
medyan = Medyan(veriler)
standart_sapma = StandartSapma(veriler)
min_max = MinMax(veriler)
korelasyon = Korelasyon(veri1, veri2)
```

## Model Tanımlama

TURK dilinde bir yapay zeka modeli `Model` anahtar kelimesi ile tanımlanır:

```turk
Model ModelAdi {
    // Katmanlar
    Katman(...)
    Katman(...)
    
    // Optimizer
    Optimizer = ...
    
    // Kayıp fonksiyonu
    Kayip = ...
    
    // Eğitim parametreleri
    Batch = 32
    Epoch = 10
}
```

## Katmanlar

TURK dili çeşitli katman tiplerini destekler:

### Tam Bağlantılı (Dense) Katman

```turk
// Basit kullanım
Katman(giris_boyutu, cikis_boyutu, aktivasyon=Relu)

// Detaylı kullanım
Katman(
    giris_boyutu, 
    cikis_boyutu, 
    aktivasyon=Relu,
    kernel_initializer=HeNormal,
    bias_initializer=Zeros,
    regularizasyon=0.01
)
```

### Dropout Katmanı

```turk
Dropout(0.5)  // %50 dropout oranı
```

### Batch Normalization Katmanı

```turk
BatchNorm()
```

### Evrişimli (Konvolüsyon) Katman

```turk
Katman(
    tip = "Evrisimli",
    filtre_sayisi = 32,
    filtre_boyutu = (3, 3),
    aktivasyon = Relu,
    padding = "same"
)
```

### Havuzlama (Pooling) Katmanı

```turk
Katman(
    tip = "Havuzlama",
    boyut = (2, 2),
    tip = "max"  // veya "average"
)
```

### RNN/LSTM/GRU Katmanları

```turk
// RNN
Katman(
    tip = "RNN",
    gizli_boyut = 64,
    bidirectional = Dogru
)

// LSTM
LSTM {
    gizli_boyut = 64,
    bidirectional = Dogru
}

// GRU
GRU {
    gizli_boyut = 32,
    bidirectional = Yanlis
}
```

## Aktivasyon Fonksiyonları

TURK dili aşağıdaki aktivasyon fonksiyonlarını destekler:

| Fonksiyon | Açıklama |
|-----------|----------|
| `Relu` | Rectified Linear Unit |
| `LeakyRelu` | Sızıntılı ReLU |
| `Sigmoid` | Sigmoid fonksiyonu |
| `Tanh` | Hiperbolik tanjant |
| `Softmax` | Softmax fonksiyonu |
| `Swish` | Swish fonksiyonu |

## Optimizerlar

TURK dili aşağıdaki optimizasyon algoritmalarını destekler:

### SGD (Stokastik Gradyan İnişi)

```turk
Optimizer = SGD(
    ogrenme_orani=0.01,
    momentum=0.9,
    nesterov=Dogru
)
```

### Adam

```turk
Optimizer = Adam(
    ogrenme_orani=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)
```

### AdamW

```turk
Optimizer = AdamW(
    ogrenme_orani=0.001,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.999
)
```

### RMSProp

```turk
Optimizer = RMSProp(
    ogrenme_orani=0.001,
    rho=0.9,
    epsilon=1e-8
)
```

## Kayıp Fonksiyonları

TURK dili çeşitli kayıp fonksiyonlarını destekler:

| Fonksiyon | Açıklama | Kullanım |
|-----------|----------|----------|
| `MSE` | Ortalama Kare Hata | `Kayip = MSE` |
| `CrossEntropy` | Çapraz Entropi | `Kayip = CrossEntropy` |
| `BinaryCrossEntropy` | İkili Çapraz Entropi | `Kayip = BinaryCrossEntropy` |
| `HuberLoss` | Huber Kaybı | `Kayip = HuberLoss(delta=1.0)` |
| `FocalLoss` | Odaksal Kayıp | `Kayip = FocalLoss(gamma=2.0)` |

## Metrikler

Eğitim ve değerlendirme sırasında kullanılabilen metrikler:

| Metrik | Açıklama | Kullanım |
|--------|----------|----------|
| `dogruluk` | Doğruluk (Accuracy) | `metrikler = ["dogruluk"]` |
| `hassasiyet` | Hassasiyet (Precision) | `metrikler = ["hassasiyet"]` |
| `duyarlilik` | Duyarlılık (Recall) | `metrikler = ["duyarlilik"]` |
| `f1` | F1 Skoru | `metrikler = ["f1"]` |
| `kayip` | Kayıp değeri | `metrikler = ["kayip"]` |
| `karisiklik_matrisi` | Karışıklık matrisi | `metrik = "karisiklik_matrisi"` |

## Eğitim ve Değerlendirme

### Model Eğitimi

```turk
// Basit eğitim
Egit(
    model = ModelAdi,
    veri = egitim.x,
    etiketler = egitim.y
)

// Detaylı eğitim
Egit(
    model = ModelAdi,
    veri = egitim.x,
    etiketler = egitim.y,
    dogrulama_verisi = test.x,
    dogrulama_etiketleri = test.y,
    batch = 32,
    epoch = 10,
    erken_durdurma = 3,
    metrikler = ["dogruluk", "f1"],
    grafik = Dogru,
    kaydet_en_iyi = Dogru
)
```

### Model Değerlendirmesi

```turk
// Temel değerlendirme
sonuclar = Degerlendir(
    model = ModelAdi,
    veri = test.x,
    etiketler = test.y
)

// Sonuçlara erişim
Yazdir("Doğruluk:", sonuclar.dogruluk)
Yazdir("Kayıp:", sonuclar.kayip)
Yazdir("F1 skoru:", sonuclar.f1)

// Karışıklık matrisi oluşturma
karisiklik_matrisi = Degerlendir(
    model = ModelAdi,
    veri = test.x,
    etiketler = test.y,
    metrik = "karisiklik_matrisi"
)
```

### Tahmin Yapma

```turk
// Tekil tahmin
tahmin = Tahmin(ModelAdi, veri)

// Çoklu tahmin
tahminler = Tahmin(ModelAdi, veri_seti)

// Tahmin sonuçlarını işleme
Icin veri, tahmin Icinde veri_seti, tahminler {
    sinif = Ortalama(tahmin)  // En yüksek olasılıklı sınıfı al
    Yazdir("Tahmin:", sinif)
}
```

## Kaydetme ve Yükleme

### Model Kaydetme

```turk
// Modeli JSON formatında kaydet
Kaydet(ModelAdi, "model.turk")

// Belirli bir formatta kaydet
Kaydet(ModelAdi, "model.h5", format="h5")
```

### Model Yükleme

```turk
// Kaydedilmiş modeli yükle
Yukle(ModelAdi, "model.turk")

// Yüklenen model üzerinde işlem yapma
sonuclar = Degerlendir(ModelAdi, test.x, test.y)
```

## Veri Görselleştirme

```turk
// Çizgi grafiği
VeriGorsellestir(
    veri = egitim_kaybi,
    grafik_tipi = "çizgi",
    baslik = "Eğitim Kaybı",
    x_etiketi = "Epoch",
    y_etiketi = "Kayıp"
)

// Isı haritası
VeriGorsellestir(
    veri = karisiklik_matrisi,
    grafik_tipi = "isı_haritası",
    baslik = "Karışıklık Matrisi",
    x_etiketleri = sinif_isimleri,
    y_etiketleri = sinif_isimleri
)

// Histogram
VeriGorsellestir(
    veri = veri_dagilimi,
    grafik_tipi = "histogram",
    baslik = "Veri Dağılımı",
    x_etiketi = "Değer",
    y_etiketi = "Frekans"
)
```

## Örnek Uygulamalar

### Örnek 1: Basit Regresyon

```turk
// Basit doğrusal regresyon örneği
Veri veri = Veri_Yukle("ev_fiyatlari.csv")
Veri egitim, test = Veri_Bol(veri, oran=0.8)

Model EvFiyatTahmini {
    Katman(5, 10, aktivasyon=Relu)  // 5 özellik (oda sayısı, metrekare, vs.)
    Katman(10, 1, aktivasyon=Dogrusal)  // Fiyat tahmini
    
    Optimizer = Adam(ogrenme_orani=0.001)
    Kayip = MSE
    Batch = 32
    Epoch = 100
}

Egit(EvFiyatTahmini, egitim.x, egitim.y)
sonuclar = Degerlendir(EvFiyatTahmini, test.x, test.y)
Yazdir("Ortalama Kare Hata:", sonuclar.kayip)

// Yeni bir ev için tahmin yap
yeni_ev = [3, 120, 2, 15, 1]  // Özellikler
tahmin_fiyat = Tahmin(EvFiyatTahmini, yeni_ev)
Yazdir("Tahmini Fiyat:", tahmin_fiyat[0])
```

### Örnek 2: Metin Sınıflandırma

```turk
// Haber kategorisi sınıflandırma
Veri haberler = Veri_Yukle("haberler.csv")
Veri egitim, test = Veri_Bol(haberler, oran=0.8)

// Metinleri vektörlere dönüştür
egitim.x = MetinVektorlestir(egitim.x, yontem="tfidf", max_ozellik=5000)
test.x = MetinVektorlestir(test.x, yontem="tfidf", max_ozellik=5000)

// One-hot kodlama (5 kategori: spor, ekonomi, siyaset, teknoloji, kültür)
egitim.y = OneHotKodla(egitim.y, sinif_sayisi=5)
test.y = OneHotKodla(test.y, sinif_sayisi=5)

Model HaberSiniflandirici {
    Katman(5000, 100, aktivasyon=Relu)
    Dropout(0.3)
    Katman(100, 50, aktivasyon=Relu)
    Dropout(0.3)
    Katman(50, 5, aktivasyon=Softmax)
    
    Optimizer = Adam(ogrenme_orani=0.001)
    Kayip = CrossEntropy
    Batch = 64
    Epoch = 10
}

Egit(HaberSiniflandirici, egitim.x, egitim.y)
sonuclar = Degerlendir(HaberSiniflandirici, test.x, test.y)
Yazdir("Doğruluk:", sonuclar.dogruluk)

// Yeni bir haber metni için tahmin
yeni_haber = "Merkez Bankası bugün faiz kararını açıkladı."
yeni_vektor = MetinVektorlestir(yeni_haber, yontem="tfidf", max_ozellik=5000)
tahmin = Tahmin(HaberSiniflandirici, yeni_vektor)

kategori_isimleri = ["Spor", "Ekonomi", "Siyaset", "Teknoloji", "Kültür"]
tahmin_kategori = kategori_isimleri[Ortalama(tahmin)]
Yazdir("Tahmin edilen kategori:", tahmin_kategori)
``` 
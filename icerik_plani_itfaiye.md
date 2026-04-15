# İstanbul'da İtfaiye İstasyonu Yer Optimizasyonu — Proposal İçerik Planı

## Proje Başlığı Önerisi
"Optimizing Fire Station Placement in Istanbul to Minimize Emergency Response Times"

## Öğrenci Numaraları: 150220321, 150210337, 150230910
(İsim ve e-mail bilgileri eklenecek)

---

## SECTION 1: Project Description
3-4 cümle ile projeyi tanıt. Şunları vurgula:
- İstanbul dünyanın en kalabalık şehirlerinden biri, deprem riski çok yüksek
- Yangın ve acil durum müdahale süresi hayat kurtarıcı — her dakika kritik
- Mevcut itfaiye istasyonları yeterli mi? Nüfus yoğunluğu ve şehrin büyümesiyle uyumlu mu?
- Bu projede, İBB'nin gerçek verilerini kullanarak yeni itfaiye istasyonlarının optimal yerleşimini belirleyeceğiz
- Amaç: ortalama müdahale süresini minimize etmek ve kapsama alanını maximize etmek

---

## SECTION 2: Problem Definition
Bu bölüm en kritik — formal matematiksel formülasyon olacak.

### Problem Türü
**p-median Facility Location Problem** veya **Maximal Covering Location Problem (MCLP)**

### Notasyon (Notation)
- I = {1, 2, ..., m} : talep noktaları kümesi (ilçeler veya mahalleler)
- J = {1, 2, ..., n} : aday istasyon konumları kümesi
- J_existing ⊂ J : mevcut itfaiye istasyonlarının konumları
- p : açılacak yeni istasyon sayısı (bütçe parametresi)
- d_i : i noktasındaki talep ağırlığı (nüfus yoğunluğu veya yangın olay sayısı)
- t_ij : i talep noktasından j aday konumuna tahmini müdahale süresi (mesafe/hız)
- T_max : kabul edilebilir maksimum müdahale süresi eşiği (örn. 5-8 dakika)

### Karar Değişkenleri (Decision Variables)
- x_j ∈ {0, 1} : j konumuna yeni istasyon kurulup kurulmayacağı
- y_ij ∈ {0, 1} : i talep noktasının j istasyonuna atanıp atanmadığı

### Amaç Fonksiyonu (Objective Function)

**Versiyon A — p-median (ağırlıklı mesafe minimizasyonu):**
  minimize  Σ_i Σ_j  d_i · t_ij · y_ij

**Versiyon B — MCLP (kapsama maksimizasyonu):**
  maximize  Σ_i  d_i · z_i
  burada z_i = 1 eğer i noktası T_max süresi içinde en az bir istasyon tarafından kapsanıyorsa

(İkisinden birini seçin veya ikisini de multi-objective olarak kullanın)

### Kısıtlar (Constraints)
1. Σ_j x_j = p  (tam olarak p yeni istasyon açılacak)
2. Σ_j y_ij = 1  ∀i ∈ I  (her talep noktası tam bir istasyona atanır)
3. y_ij ≤ x_j  ∀i,j  (atama ancak istasyon varsa yapılır)
4. x_j = 1  ∀j ∈ J_existing  (mevcut istasyonlar korunur)
5. x_j ∈ {0, 1}, y_ij ∈ {0, 1}

### Opsiyonel Ek Kısıtlar (proposal'da bahsedebilirsiniz)
- İstasyonlar arası minimum mesafe kısıtı
- Deprem risk bölgelerine yakınlık tercih kısıtı
- Acil ulaşım yollarına erişim kısıtı (İBB'nin 1. derece acil ulaşım yolları verisi var)

### Hedef Değişken (Target Variable)
Toplam ağırlıklı müdahale süresi (minimize) veya T_max içinde kapsanan nüfus yüzdesi (maximize)

### Özellikler / Parametreler
- Nüfus yoğunluğu (talep ağırlığı olarak)
- Yangın olay sayıları (ilçe bazlı — alternatif talep ağırlığı)
- Ortalama varış süreleri (mevcut performans verisi)
- Mesafe / yol ağı bilgileri
- Deprem risk skorları (opsiyonel ek parametre)

---

## SECTION 3: Dataset
### Birincil Kaynak: İBB Açık Veri Platformu (data.ibb.gov.tr)

| Veri Seti | İçerik | Format |
|-----------|--------|--------|
| İtfaiye İstasyonları Konum Bilgileri | Mevcut istasyonların koordinatları | XLSX, HTML |
| İtfaiye Olaylar Ortalama Varış Süresi | Olaylara dakika cinsinden varış süreleri | XLSX |
| İtfaiye Olaylar | Yangın türleri, doğal afet, can kurtarma müdahaleleri | XLSX |
| İtfaiye Olaylar Ambulans Çıkış Nedenleri | Ambulans çıkış nedenleri | XLSX |
| Yangın Sayısı | Yapısal ve yapısal olmayan yangın sayıları | XLSX |
| İtfaiye İstasyon Sayısı | İstasyon sayıları | XLSX |
| İtfaiye Araç Sayısı | Araç kapasitesi | XLSX |
| İlçe Nüfus Bilgileri | Yaş ve cinsiyet kırılımlı nüfus | XLSX |
| 1. Derece Acil Ulaşım Yolları | Yol kapanma risk analizi, bina yıkılma riskleri | GeoJSON |
| İstanbul Sağlık Kurum ve Kuruluşları | Hastane ve sağlık kurumu konumları | XLSX |

### Veri İşleme Süreci (kısaca anlat)
- İstanbul haritası grid sisteme bölünecek (örn. 1km × 1km)
- Her grid hücresi bir talep noktası olarak tanımlanacak
- Nüfus verileri grid hücrelerine dağıtılacak
- Mevcut istasyon konumları haritaya yerleştirilecek
- Mesafe/süre matrisi hesaplanacak (Haversine veya yol ağı mesafesi)
- Yangın olay verileri talep ağırlığı olarak kullanılacak

---

## SECTION 4: Methodology
### 3 yöntemi karşılaştırmalı kullanacağız:

### 1. Mixed-Integer Linear Programming (MILP)
- **Neden uygun:** p-median / MCLP problemleri doğası gereği tamsayılı programlama problemidir. Küçük-orta ölçekte optimal çözümü garanti eder.
- **Araç:** Python PuLP veya scipy.optimize
- **Avantaj:** Global optimum, kesin çözüm
- **Dezavantaj:** Problem boyutu büyüdükçe hesaplama süresi artar

### 2. Genetic Algorithm (GA)
- **Neden uygun:** Kombinatoryal yapı GA'nın binary encoding'ine doğrudan uyar. Her aday konum bir gen (0 veya 1). Büyük arama uzaylarında etkili.
- **Operatörler:** Tournament selection, uniform crossover, bit-flip mutation
- **Avantaj:** Büyük ölçekli problemlerde çalışabilir, multi-objective versiyonu (NSGA-II) da uygulanabilir
- **Dezavantaj:** Optimal çözüm garantisi yok, parametre ayarı gerekli

### 3. Simulated Annealing (SA)
- **Neden uygun:** Yerel minimumlardan kaçış yeteneği var, facility location problemlerinde literürde yaygın.
- **Komşuluk tanımı:** Bir istasyonu kapat, başka bir aday konuma aç (swap move)
- **Avantaj:** Basit implementasyon, iyi sonuçlar
- **Dezavantaj:** Soğuma şeması ve başlangıç sıcaklığı ayarı gerekir

### Karşılaştırma Kriterleri
- Çözüm kalitesi (objective function değeri)
- Hesaplama süresi
- Kapsama oranı (nüfusun yüzde kaçı T_max içinde)
- Farklı p değerleri için sonuçlar (p = 3, 5, 10 gibi)

---

## REFERENCES (öneriler)
- Hakimi, S.L. (1964). "Optimum locations of switching centers and the absolute centers and medians of a graph." Operations Research.
- Church, R., ReVelle, C. (1974). "The maximal covering location problem." Papers in Regional Science.
- Toregas, C., et al. (1971). "The location of emergency service facilities." Operations Research.
- Aktaş, E., et al. (2013). "Optimizing fire station locations for the Istanbul Metropolitan Municipality." Interfaces.
- Yang, L., et al. (2007). "Fire station location problem using GA." Fire Safety Journal.
- İBB Açık Veri Platformu: https://data.ibb.gov.tr

---

## ÖNEMLİ HATIRLATMALAR
- Toplam 2 sayfa (referanslar hariç) — kısa ve öz yaz
- LLM çıktısı gibi görünmesin — kendi cümlelerinle, kendi tarzınla yaz
- Matematiksel formülasyonu LaTeX'te düzgün yaz (\sum, \min, \forall vb.)
- GitHub repo linki eklemeyi unutma
- Deadline: 17 Nisan 2026, 23:59

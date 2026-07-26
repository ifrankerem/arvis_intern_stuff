# Yüz Kalitesi Uygulaması

Uygulama; canlı kamera önizlemesi, kamera seçimi, yüz kalitesi/hizalama
durumları, FFT, DCT/blok, wavelet ve high-pass residual ön kontrol sonuçları ile
kayıt düğmeleri içeren basit bir masaüstü arayüzüne sahiptir.

## Ön kontrol mimarisi

Kamera açılmadan önce iki ayrı moddan biri seçilir:

- **Model-free Pre-Control:** FFT ile Moiré/periyodik desen denetimleri çalışır.
  MediaPipe import edilmez ve hiçbir model dosyası yüklenmez. Yüz bölgesi olarak
  ekrandaki sabit kılavuz kullanılır; kişinin yüzünü bu alanı dolduracak şekilde
  konumlandırması gerekir. Moiré denetimi yeni FFT hesaplamaz; aynı karenin
  mevcut, kaydırılmış sayısal güç ve log spektrumlarını kullanır. DCT / Block
  Analysis aynı standardize hizalanmış yüz crop'unun penceresiz gri temsilinde
  8×8 blok istatistiklerini inceler. Wavelet modülü aynı penceresiz crop'ta
  model-free çok-ölçekli doku analizi yapar. High-Pass Residual Analysis ise
  aynı float32 luminance crop'unda ince ölçekli artık yapısını ölçer.
- **Model Analizi:** MediaPipe ile yüz kalitesi ve hizalama çalışır. FFT
  pre-control çalışmaz.

FFT sınıfının `analyze_face_box(frame, face_box)` girişi modelden bağımsızdır;
ileride sabit UI kılavuzu yerine başka bir model-free ROI yöntemi verilebilir.

### Ortak matematiksel analiz context'i

Her geçerli model-free kare için `ModelFreePreControlContext` bir kez üretilir.
Context; ham kamera karesi, aynalanmış analiz karesi, yüksek çözünürlüklü ROI,
gri crop, 256×256 penceresiz standardize hizalanmış crop, FFT'ye özel Hann
pencereli analiz crop'u, kalite ölçümleri ve ortak FFT ara değerlerini taşır.
FFT yalnızca context builder içinde hesaplanır; Global FFT, Moiré ve
Radial/Angular modülleri aynı kompleks FFT, kaydırılmış FFT, magnitude, power ve
analitik log-power dizilerini paylaşır.

Model-free modda geometrik yüz hizalama veya pose modeli çalışmadığı için
`aligned_face_crop` aynı guide ROI'yi gösterir, `alignment_applied=False` ve
`pose_alignment_valid=None` kalır. Bu alanlar ileride model-free bir hizalama
yöntemi eklendiğinde veri akışını değiştirmeden doldurulabilir.

Bütün matematiksel modüller `ModelFreeAnalysisResult` döndürür. Ham özellikler,
ham skor, stabilize skor, confidence, evidence, warning, debug verisi ve
kalibrasyon durumu ayrı alanlardır. Kalite veya spektrum geçersizse sonuç
`available=False`, durum `Analysis unavailable`/`Unavailable` ve skorlar `None`
olur; eksik veri normal sıfır skoru olarak gösterilmez. Modüller bağımsız hata sınırlarında
çalıştığı için bir modül hatası diğer analizleri ve kamera akışını durdurmaz.

Planlanan altı matematiksel modülün enable/disable bayrakları, analiz boyutu,
FFT penceresi, kalite/frekans/zamansal eşikler, debug modu ve calibration path
`config.py` içindeki tek model-free yapılandırma bölümündedir. Şu anda Global
FFT, Moiré, Radial/Angular, DCT / Block Analysis, Wavelet ve High-Pass Residual
Analysis etkin durumdadır. Tüm karar eşikleri deneysel olarak işaretlenmiştir.

### Modül 1: Global FFT

Global FFT analizi `global_fft_pre_control.py`, Moiré analizi
`moire_pre_control.py`, Radial/Angular analizi `radial_angular_pre_control.py`
içinde tutulur. `pre_control.py`, eski importları bozmamak için bu sınıfları
yeniden dışarı açan küçük bir uyumluluk katmanıdır.
Yeni modüller kendi ayrı dosyalarında aynı ortak context/result API'siyle
eklenmelidir.

Global FFT modülü ortak context'teki mevcut power spectrum'u kullanır; yeni FFT
veya resize yapmaz. DC merkezinin dışındaki yapılandırılabilir low, middle ve
high radial bantlardan şu feature'ları çıkarır: üç bandın enerji oranları,
spectral centroid, normalize spectral entropy, log-log radial spectral slope,
total spectral energy ve high-to-low energy ratio.

Henüz gerçek bir baseline calibration dosyası olmadığı için skor modu
`experimental`, `calibrated=False` durumundadır. Provisional feature aralıkları
özellikle konservatiftir ve bilimsel/evrensel yüz eşikleri değildir. Ham skor ile
rolling-median stabilize skor ayrı tutulur; tek kare final durumu değiştirmez.
Global FFT sonucu yalnızca frekans dağılımı sapmasını açıklar, authenticity veya
saldırı türü sınıflandırmaz.

### Modül 3: Radial/Angular Spectrum

Module 3 ortak context'teki merkezlenmiş `power_spectrum` dizisini yeniden FFT
hesaplamadan configurable radial ve `[0, 180°)` angular bin'lere dönüştürür.
Radial profilde mean, median, normalize energy ve log-power; angular profilde
yön başına normalize energy saklanır. Slope/fit error, radial entropy, dominant
radial frequency, narrow-band concentration, axial angular mean/variance,
angular entropy, anisotropy ve horizontal/vertical/diagonal concentration
feature'ları üretilir.

Dominant frequency yönü ile görüntüdeki çizgi yönü ayrı raporlanır; ikincisi
birincisine diktir. Calibration dosyasında uyumlu radial ve angular bona-fide
profilleri varsa karşılaştırılır. Dosya yoksa deneysel aralıklar kullanılır ve
`calibrated=False` döner. Raw radial, angular ve combined skorlar ile rolling
median stabilize skor birbirinden ayrıdır.

Debug kaydı radial/angular CSV profillerini, iki profil grafiğini, dominant
frequency ve image-line yönlerinin işaretlendiği spektrumu ve sayısal JSON
raporunu da aynı timestamp ile kaydeder.

### Modül 4: DCT / Block Analysis

Module 4, standardize hizalanmış gri yüz crop'unu eksik kenar bloklarını
atacak biçimde 8×8 bloklara ayırır ve her blokta floating-point 2B DCT
hesaplar. DC istatistikleri; low/middle/high AC enerjileri; AC/DC oranı;
sıfıra-yakın katsayı oranı; katsayı entropy/kurtosis değerleri ve komşu blok
değişimleri ham feature olarak korunur. Her 8 piksel sınırındaki süreksizlik,
yakındaki sınır-dışı süreksizliklerle karşılaştırılır. İç-yüz patch'lerindeki
robust DCT sapması ayrıca yerel tutarlılık sinyali üretir.

Modül `dct_band_anomaly_score`, `coefficient_sparsity_score`,
`blockiness_score`, `local_dct_inconsistency_score` ve `final_dct_score`
değerlerini üretir. Uyumlu `dct_block_analysis.feature_profiles` kalibrasyonu
yoksa puan deneysel ve kalibre edilmemiş olarak işaretlenir. Bulanık, küçük,
düşük çözünürlüklü veya aşırı yumuşatılmış görüntüler belirsiz ya da
kullanılamaz döner.

Uygulama DroidCam/video kaynağından çözülmüş kare aldığı için orijinal JPEG
baytları ve quantization tabloları mevcut değildir. Bu modül özgün JPEG
quantization tablosu, kesin JPEG kalite faktörü veya definitive double-JPEG
geçmişi tespit ettiğini iddia etmez. Normal kamera/video sıkıştırması, resize,
keskinleştirme ve blur da benzer 8×8 izleri üretebilir; sonuç sahtecilik kararı
değil, matematiksel blok-frekans anomali sinyalidir.

**Tüm Debug Çıktılarını Kaydet (1 Kare)** düğmesi DCT band-energy map, 8×8 blok
sınırı görselleştirmesi, blockiness heatmap ve katsayı istatistik raporunu
tek timestamp'li model-free debug klasörüne ekler.

### Modül 5: Wavelet Analysis

Module 5 varsayılan olarak `db2`, `periodization` sınır modu ve iki ayrıştırma
seviyesi kullanır. 256×256 standardize hizalanmış crop'tan sırasıyla 128×128 ve
64×64 LL/LH/HL/HH bantları üretilir. Wavelet ve seviye sayısı `config.py`
içinden değiştirilebilir. İç-yüz elips maskesi katsayıları değiştirmeden yalnızca
istatistik örnekleme alanını sınırlar; böylece saç, kıyafet ve arka plan etkisi
azaltılır.

Her seviyenin horizontal, vertical ve diagonal detail bandında enerji,
normalize enerji oranı, mean absolute coefficient, variance, median absolute
deviation, entropy, kurtosis ve sparsity tutulur. Patch analizi izole yüksek
frekans, aşırı düzgün bölgeler, yönsel doku sapmaları ve komşu patch geçişlerini
birlikte değerlendirir. Tek bir yüksek enerjili patch kesin sahtecilik olarak
yorumlanmaz.

Modül `wavelet_energy_score`, `directional_wavelet_score`,
`local_wavelet_inconsistency_score` ve `final_wavelet_score` üretir. Uyumlu
`wavelet_analysis.feature_profiles` kalibrasyonu yoksa `calibrated=False`
döner. Ham ve rolling-median stabilize skor ayrıdır; geçersiz kareler geçmişe
eklenmez. Küçük yüz, ağır blur, clipping veya geçersiz ayrıştırma boyutu belirsiz
ya da kullanılamaz sonuç üretir.

PyWavelets runtime bağımlılığıdır. Import edilemiyorsa uygulama paket yüklemeye
çalışmaz; Wavelet sonucu `Analysis unavailable` olur ve diğer modüller çalışmayı
sürdürür. Bağımlılık kullanıcı tarafından `PyWavelets` paketi kurularak veya
projenin `requirements.txt` dosyası uygulanarak sağlanabilir.

Debug export her seviyenin LL/LH/HL/HH ham float subband'lerini `.npy`, normalize
görsellerini `.png`, açıklayıcı patch anomaly heatmap'ini ve JSON feature
raporunu üretir. Birleşik kayıt düğmesi normalize görselleri ve heatmap'i tek
timestamp'li debug klasörüne ekler. Heatmap bir neural-network attention map
değildir.

### Modül 6: High-Pass Residual Analysis

Module 6, ortak 256×256 standardize hizalanmış penceresiz gri/luminance crop'u
tek bir 0–255 float32 sayısal ölçekte kullanır. `original - GaussianBlur`
residual'ı varsayılan olarak 5×5 çekirdek ve sigma 1.2 ile; Laplacian ve Sobel
cevapları 3×3 çekirdeklerle hesaplanır. Analiz residual'ları signed float32
olarak korunur. Ayrı uint8 normalizasyon yalnızca debug görselleri içindir.
Parametrelerin tamamı `config.py` içinden değiştirilebilir.

Sabit, model-free iç-yüz ağırlık maskesi crop sınırını, saç/alın ve alt
kıyafet bölgesini azaltır; göz/kaş bandına daha düşük ağırlık verir. Gaussian
residual varyansı, mean absolute deviation, RMS enerji, entropy, kurtosis,
pozitif/negatif denge; Laplacian varyansı; gradient enerji ve edge density ham
feature olarak saklanır. 32×32 patch'lerin residual enerjisi, robust yerel
mesafesi, komşu farkı ve enerji varyasyonu ayrıca yerel tutarlılık üretir.

Modül `gaussian_residual_score`, `laplacian_score`, `gradient_score`,
`local_residual_inconsistency_score` ve `final_residual_score` değerlerini
üretir. Uyumlu `high_pass_residual_analysis.feature_profiles` bona-fide
kalibrasyonu varsa iki taraflı sapma bu baseline'a göre hesaplanır. Kalibrasyon
yoksa konservatif geliştirme aralıkları kullanılır, `calibrated=False` döner ve
GUI bunu deneysel/kalibre edilmemiş olarak gösterir. Çok yüksek enerji kadar
anormal düşük enerji de sapma olabilir; hiçbiri tek başına fraud kararı değildir.

Ham ve rolling-median stabilize skorlar ayrı tutulur; unavailable kareler
zamansal geçmişe eklenmez. Şiddetli blur, düşük ışık/ISO noise, clipping, küçük
ve büyütülmüş yüz crop'u, kamera sharpening'i ve DroidCam/video compression
confidence değerini düşürür veya sonucu belirsiz/kullanılamaz yapar. Modül
Noiseprint, neural PRNU, CNN, F3-Net ya da başka bir pretrained network içermez.

Debug export Gaussian residual görselini, Laplacian cevabını, gradient
magnitude'ı, patch residual-energy map'ini ve JSON feature raporunu
tek timestamp'li model-free debug klasörüne ekler.

### Final: İki aşamalı matematiksel füzyon

`mathematical_fusion.py`, ortak FFT spektrumunu kullanan Global FFT, Moiré ve
Radial/Angular skorlarını önce `fft_family_score` içinde birleştirir. DCT/Block,
Wavelet ve Residual skorları ayrı `local_transform_score` grubunu oluşturur.
Final hesapta yalnızca bu iki grup `combined_mathematical_risk_score` üretir;
dolayısıyla korelasyonlu FFT sinyalleri üç bağımsız oy olarak sayılmaz.

Hem modül hem grup aşamasında hesap, `score × weight × confidence` toplamının
`weight × confidence` toplamına bölünmesidir. Unavailable, skoru bulunmayan,
kalitesi belirsiz veya etkili confidence değeri yetersiz modül paya ve paydaya
girmez. En az dört geçerli modül ve her aileden en az bir geçerli modül yoksa
sonuç `Inconclusive` olur; eksik analiz hiçbir zaman sıfır risk veya normal
olarak kabul edilmez. Bütün deneysel ağırlık ve eşikler `config.py` içindeki
tek `MATHEMATICAL_FUSION_CONFIG` bölümündedir.

Füzyon güncel kare skorunu, rolling median'ı, sunum kanıtı temporal
percentile'ını, final temporal kararını ve kullanıcıya gösterilen skoru ayrı
alanlarda tutar. JSON, TXT, CLI ve GUI aynı `score_summary` yapısını okur;
bulunmayan değerler başka bir skorla doldurulmaz. Yalnızca geçerli kaliteli
kareler geçmişe girer. Uyarının açılması ve kapanması ayrı ardışık-kare
kapılarına bağlıdır; tek karelik skor sıçraması warning flicker üretmez.
Analiz veya füzyon şema sürümü değiştiğinde temporal geçmiş temizlenir. Final
durumlar yalnızca
`Normal mathematical evidence`, `Weak anomaly evidence`,
`Suspicious mathematical evidence`, `High mathematical risk`, `Inconclusive`
ve `Uncalibrated` olabilir. Kesin authenticity sınıflandırması yapılmaz.

GUI, teknik skorların üstünde sade bir Türkçe **Genel Sonuç** gösterir. Normal
durumda yalnızca o karede belirgin ekran/baskı izi görülmediğini ve bunun tek
başına canlılık kanıtı olmadığını; zayıf durumda olağandışı izler bulunduğunu;
şüpheli veya yüksek durumda ekran gösterimi ya da basılı fotoğraf saldırısı
olabileceğini ve bunun kesin karar olmadığını açıkça belirtir.
Sonuçlandırılamayan karelerde kullanıcıya ışık, netlik ve yüz konumu için
uygulanabilir tekrar-deneme yönlendirmesi verilir.

Wavelet ve Residual, şiddetli kırpılmış parlaklıklarda doku ölçümünü güvenilmez
sayıp fusion dışında kalır. Bu sırada ölçülmüş clipping oranı artık kaybolmaz:
final fusion bu izi yerel DCT tutarsızlığı ve FFT geniş-bant dağılımı ile
birlikte tek bir çapraz-yöntem sunum artefaktı olarak değerlendirir. Clipping
tek başına saldırı skoru üretmez; sert ışık ve hatalı pozlama da aynı izi
oluşturabilir. Bu sunum kanalı iki-aile ağırlıklı güncel kare skorunun yerine
geçmez; yalnızca altı ROI modülünden türetilen ayrı temporal sunum kararına
girer.

Clipping üretmeyen ekranlar için ikinci yol, FFT orta/yüksek bant enerjisini
DCT katsayı yoğunluğu ve Residual/Wavelet ince-doku ölçümleriyle doğrular. Bu
yolun etkinleşmesi için FFT ailesiyle birlikte en az iki bağımsız transform
sinyalinin aynı karede eşiği geçmesi gerekir; tek başına yüksek keskinlik veya
entropy ekran saldırısı kabul edilmez.

Autofocus nedeniyle transform desteği tam eşiğin hemen altına düştüğünde skor
artık doğrudan sıfırlanmaz. FFT ile en az bir transformun orta düzey desteği
`49/100` altında sınırlandırılmış partial evidence üretir; bu sinyal tek başına
şüpheli uyarısı açamaz. Güçlü sunum kanıtının son sekiz kare içinde üç kez
görülmesi uyarıyı açabilir ve kısa odak kayıplarında uyarı altı ardışık recovery
karesine kadar korunur. Böylece tek kare sıçramaları filtrelenirken ekran
dokusunun autofocus ile aralıklı kaybolması normal kabul edilmez.

Tam kamera karesi yalnızca kamera görüntüsü, yüz kılavuzu/crop üretimi,
görselleştirme ve debug kaydı için tutulur. Telefon, tablet veya monitör
çerçevesi; bezel, dikdörtgen, paralel arka plan çizgileri ve diğer tam-kare
geometri öğeleri fraud skoruna, kanıta veya temporal geçmişe dahil edilmez.
Final karar yalnızca yüz ROI'sindeki altı matematiksel modülün iki aileli
füzyonu ve bu modüllerden türetilen frekans/doku kanıtlarını kullanır.

Kalibrasyon yolu proje kökündeki `model_free_calibration.json` dosyasıdır.
Uygulama dosyayı oluşturmaz. Dosya yoksa bütün heuristic/fusion değerleri
deneysel, `calibrated=false` olarak çalışır. Uyumlu final-fusion bölümü aşağıdaki
şemayı kullanır; `module_weights`, `group_weights` ve `status_thresholds`
opsiyoneldir, fakat kalibre final mapping için baseline ve score mapping
zorunludur:

```json
{
  "mathematical_fusion": {
    "bona_fide_baseline": {
      "combined_mathematical_risk_score": {
        "mean": 0.0,
        "standard_deviation": 1.0
      }
    },
    "score_mapping": {
      "z_score_start": 1.0,
      "z_score_full": 4.0
    },
    "module_weights": {
      "fft": 1.0,
      "moire": 1.0,
      "radial_angular": 1.0,
      "dct_block": 1.0,
      "wavelet": 1.0,
      "residual": 1.0
    },
    "group_weights": {
      "fft_family": 1.0,
      "local_transform": 1.0
    },
    "status_thresholds": {
      "weak_anomaly_score": 25.0,
      "suspicious_score": 50.0,
      "high_risk_score": 75.0,
      "recovery_score": 42.0
    }
  }
}
```

Bu değerler yalnızca şema örneğidir; bona-fide veri ölçülmeden dosya
oluşturulmamalı veya örnek sayılar gerçek kalibrasyon gibi kullanılmamalıdır.

Yeni ön kontrol sonuçları `latest_pre_control_results` koleksiyonuna eklendiğinde
arayüz bunları aynı **Ön kontrol analizleri** kartında gösterebilir.
Her sonuçta `display_name` ve `status`; isteğe bağlı olarak `score`, `warning`
ve `attack_type` alanları kullanılabilir.

Moiré denetimi DC merkezini maskeleyip orta/yüksek frekans bandındaki belirgin
tepeleri, merkez simetrisini ve yön yoğunlaşmasını ölçer. Tek kare uyarı üretmez;
kısa bir skor geçmişi ve açma/kapama gecikmesi kullanır. Tüm başlangıç eşikleri
`config.py` içindeki `EXPERIMENTAL_MOIRE_*` bölümünde toplanmıştır ve evrensel
değildir; kullanılan kamera, ekran, baskı, ışık ve mesafe örnekleriyle kalibre
edilmelidir.

Sonuç kesin bir canlılık veya sahte yüz sınıflandırması değildir. Saç/sakal,
çizgili kıyafet, desenli arka plan, panjur, JPEG sıkıştırması, kamera
keskinleştirmesi ve yeniden boyutlandırma da periyodik frekans tepeleri
üretebilir. Altı yöntemin yüz ROI'sindeki analizinde kullanılan Hann penceresi
ve konservatif zamansal kapı bu yanlış pozitifleri azaltır. Görünür ekran
kenarları veya arka plandaki paralel çizgiler karar girdisi değildir.

Model-free pre-control çalışırken **Tüm Debug Çıktılarını Kaydet (1 Kare)**
düğmesi veya `s` tuşu, `model_free_debug/analysis_<timestamp>/` altında tek bir
klasör oluşturur. Bu klasörde yüz girişleri, altı modülün görselleri, bütün ham
feature değerleri, raw/stabilized modül skorları, iki grup skoru, combined skor,
kalite değerleri, configuration threshold'ları ve final evidence listesi yer
alır. Ham tam kamera karesi, aynalanmış analiz karesi ve kılavuz bounding-box
koordinatları yalnızca yeniden üretilebilirlik, görsel inceleme ve crop
doğrulaması için saklanır; fraud skoruna girmez. Aynı içerik machine-readable
JSON ve human-readable TXT raporuyla kaydedilir. Sürekli açık kalan ayrı OpenCV
pencereleri oluşturulmaz; `q` uygulamayı kapatır.

## Çalıştırma

```bash
cd pad_mini
python3 -m pip install -r requirements.txt
python3 main.py
```

Kamera listesinin yanındaki yenile düğmesi, sonradan bağlanan USB veya sanal
kameraları tekrar tarar. Açılır kutu düzenlenebilir; OpenCV'nin okuyabildiği bir
ağ yayını adresi (ör. telefon kamera uygulamasının verdiği URL) buraya doğrudan
yazılabilir.

### DroidCam ile doğrudan bağlantı

Bilgisayara ayrı bir DroidCam istemcisi veya sanal kamera sürücüsü kurmak
gerekmez. Telefon ve bilgisayar aynı Wi-Fi ağındayken telefondaki DroidCam
uygulamasında görünen IP adresini arayüzdeki **DroidCam • Wi-Fi** alanına girip
**Telefon Kamerasına Bağlan** düğmesine basın. Port yazılmazsa DroidCam'in
varsayılan `4747` portu kullanılır. Başarıyla bağlanan son adres kullanıcı
ayarlarına kaydedilir ve uygulamanın sonraki açılışında otomatik doldurulur.

## Telefon kamerası

- Android cihaz USB webcam modunu destekliyorsa telefonu Type-C ile bağlayın,
  USB seçeneklerinden **Webcam** modunu seçin ve uygulamada kamera listesini
  yenileyin.
- Telefon bu modu sunmuyorsa DroidCam uygulamasının Wi-Fi IP adresi yukarıdaki
  doğrudan bağlantı alanında kullanılabilir.
- iPhone, Linux'ta yalnızca USB kablosu takılarak standart webcam olmaz. Linux
  uyumlu bir telefon kamera uygulaması/sanal kamera ya da uygulamanın sunduğu ağ
  video adresi gerekir.
- macOS'ta desteklenen iPhone ve Mac modellerinde Apple'ın Süreklilik Kamerası
  özelliği kullanılabilir; kamera işletim sisteminde göründüğünde bu uygulamanın
  listesinde de seçilebilir.

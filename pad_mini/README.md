# Yüz Kalitesi Uygulaması

Uygulama; canlı kamera önizlemesi, kamera seçimi, yüz kalitesi/hizalama
durumları, FFT ön kontrol sonucu ve kayıt düğmeleri içeren basit bir masaüstü
arayüzüne sahiptir.

## Ön kontrol mimarisi

Kamera açılmadan önce iki ayrı moddan biri seçilir:

- **Model-free Pre-Control:** FFT ile Moiré/periyodik desen denetimleri çalışır.
  MediaPipe import edilmez ve hiçbir model dosyası yüklenmez. Yüz bölgesi olarak
  ekrandaki sabit kılavuz kullanılır; kişinin yüzünü bu alanı dolduracak şekilde
  konumlandırması gerekir. Moiré denetimi yeni FFT hesaplamaz; aynı karenin
  mevcut, kaydırılmış sayısal güç ve log spektrumlarını kullanır.
- **Model Analizi:** MediaPipe ile yüz kalitesi ve hizalama çalışır. FFT
  pre-control çalışmaz.

FFT sınıfının `analyze_face_box(frame, face_box)` girişi modelden bağımsızdır;
ileride sabit UI kılavuzu yerine başka bir model-free ROI yöntemi verilebilir.

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
üretebilir. Analizin yalnızca kılavuz ROI'sine uygulanması, crop kenarlarındaki
Hann penceresi ve konservatif zamansal kapı bu yanlış pozitifleri azaltır.

Model-free pre-control çalışırken **FFT Örneğini Kaydet (1 Kare)** düğmesi veya
`s` tuşu, o anki geçerli yüz ROI'sini ve mevcut FFT görselleştirmesini bir kez
`fft_samples/` klasörüne kaydeder. Sürekli açık kalan ayrı OpenCV pencereleri
oluşturulmaz; `q` uygulamayı kapatır.

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

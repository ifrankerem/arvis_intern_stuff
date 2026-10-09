<div align="center">

# 🔬 arvis_intern_stuff

**ARVIS staj çalışmaları — yüz kalitesi ve moiré ön kontrolü.**

![Python](https://img.shields.io/badge/language-Python-3776AB?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy)
![Status](https://img.shields.io/badge/status-research-FAB095?style=flat-square)
![Stars](https://img.shields.io/github/stars/ifrankerem/arvis_intern_stuff?style=flat-square)

</div>

---

## 📋 İçindekiler

- [Hakkında](#hakkında)
- [İçerik](#içerik)
- [Kurulum](#kurulum)
- [Çalıştırma](#çalıştırma)
- [Güvenlik Modeli](#güvenlik-modeli)
- [Belgeler](#belgeler)
- [Yazar](#yazar)

---

## 📖 Hakkında

Depo iki ayrı çalışma barındırır:

- **`pad_mini/`** — kamera görüntüsünden yüz bölgesi (ROI) tespiti, hizalama
  kalitesi ölçümü ve **moiré / periyodik desen** sinyal analizi yapan masaüstü
  uygulaması. Canlı önizleme ile FFT, DCT/blok, wavelet, high-pass residual ve
  autocorrelation/cepstrum temelli ön kontrol sonuçlarını arayüzde gösterir.
- **`yolo/`** — deneme ve prototip betikleri.

Amaç, yüz doğrulama sistemlerini liveness sinyali üretmeden önce durdurduğu
kalite kapısını kurmak: düşük kaliteli veya desteklenmeyen bir kareden "canlı"
demek yerine, kanıtın yetersiz olduğunu açıkça söylemek.

---

## 📦 İçerik

| Dizin | Açıklama |
|---|---|
| `pad_mini/main.py` | uygulama giriş noktası |
| `pad_mini/application_gui.py` | OpenCV tabanlı arayüz |
| `pad_mini/camera_stream.py`, `camera_sources.py` | kamera yakalama ve kaynak seçimi |
| `pad_mini/precontrol.py` | iki aşamalı füzyon ve karar mekanizması |
| `pad_mini/moire_pre_control.py`, `global_fft_pre_control.py` | global FFT ve moiré kolları |
| `pad_mini/radial_angular_pre_control.py` | merkezkaç / açısal spektral analiz |
| `pad_mini/periodicity_pre_control.py` | periyodiklik analizi |
| `pad_mini/dct_block_pre_control.py` | 8×8 blok istatistikleri |
| `pad_mini/wavelet_pre_control.py` | model-free çok ölçekli doku analizi |
| `pad_mini/high_pass_residual_pre_control.py` | ince ölçekli artık sinyal yapısı |
| `pad_mini/face_landmarker.py`, `face_quality_application.py` | MediaPipe tabanlı model analizi kolu |
| `pad_mini/requirements.txt` | bağımlılıklar |

---

## ⚙️ Kurulum

```sh
cd pad_mini
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Çalıştırma

```sh
python main.py                # uygulama
python application_gui.py     # arayüz
python pre_control.py         # model-free ön kontrol kolu
python face_quality_application.py   # model analizi kolu
```

---

## 🛡️ Güvenlik Modeli

- Kalibrasyon verisi yokken deneysel skor **üretilmez**; sonuç
  `INSUFFICIENT_EVIDENCE` olur.
- Kalitesiz veya desteklenmeyen karede skor uydurulmaz.
- Model-free modda MediaPipe yalnızca yüz kutusu bulma ve takip altyapısıdır;
  çıktısı liveness veya saldırı skoruna katılmaz.
- Yüz bulunamazsa sabit merkez bölgeden skor üretilmez.
- ROI kadraj kenarına değdiğinde analizler kapanmaz; reliability değerleri
  düşürülür ve arayüz uyarı gösterir.

---

## 📚 Belgeler

`pad_mini/` içinde araştırma ve doğrulama zincirinin belgeleri duruyor:
`repository_audit.md`, `deterministic_method_catalog.md`, `literature_review.md`,
`source_traceability.md`, `precontrol_architecture.md`, `implementation_priority.md`,
`calibration_protocol.md`, `benchmark_protocol.md`, `ablation_plan.md`,
`api_output_example.json`, `limitations.md`.

---

## 👤 Yazar

**İrfan Kerem Arslan** — [@ifrankerem](https://github.com/ifrankerem)

---

## 📄 Lisans

Staj araştırma çalışması; öğrenme ve portfolyo amacıyla paylaşılmıştır.

---

## 🙏 Teşekkür

- [awesome-readme](https://github.com/matiassingers/awesome-readme) — bu README'nın yapısı
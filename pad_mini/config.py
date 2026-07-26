from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parent

CAMERA_INDEX = 0
MIRROR_CAMERA_IMAGE = True
MAXIMUM_FACE_COUNT = 2

MODEL_PATH = PROJECT_DIRECTORY / "models" / "face_landmarker.task"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"
FFT_SAMPLE_DIRECTORY = PROJECT_DIRECTORY / "fft_samples"
DCT_DEBUG_DIRECTORY = PROJECT_DIRECTORY / "dct_samples"
WAVELET_DEBUG_DIRECTORY = PROJECT_DIRECTORY / "wavelet_samples"
RESIDUAL_DEBUG_DIRECTORY = PROJECT_DIRECTORY / "residual_samples"
MODEL_FREE_DEBUG_DIRECTORY = PROJECT_DIRECTORY / "model_free_debug"

# Goruntu kalitesi esikleri
MINIMUM_FACE_AREA_RATIO = 0.08
MINIMUM_BLUR_SCORE = 80.0
MINIMUM_BRIGHTNESS = 60.0
MAXIMUM_BRIGHTNESS = 200.0

# Kafa ve agiz hizalama esikleri
MAXIMUM_YAW_DEGREES = 15.0
MAXIMUM_PITCH_DEGREES = 12.0
MAXIMUM_ROLL_DEGREES = 10.0
MAXIMUM_MOUTH_ANGLE_DEGREES = 10.0
MAXIMUM_MOUTH_LATERAL_DIFFERENCE = 0.25
MAXIMUM_JAW_OPEN_SCORE = 0.35

# Model-free matematiksel PreControl ortak yapilandirmasi.
# Bu bolumdeki tum karar esikleri deneyseldir ve evrensel degildir. Gercek
# kamera, ekran, baski, mesafe ve isik verileriyle kalibre edilmelidir.
MODEL_FREE_ANALYSIS_IMAGE_SIZE = 256
MODEL_FREE_FFT_WINDOW_TYPE = "hann"
MODEL_FREE_FFT_TUKEY_ALPHA = 0.25
MODEL_FREE_DEBUG_MODE = False
MODEL_FREE_ANALYSIS_SCHEMA_VERSION = 3
MODEL_FREE_FUSION_CONFIGURATION_VERSION = 3
MODEL_FREE_RUNTIME_MODE = "BALANCED"
# Canli GUI, pahali arastirma modullerini her karede calistirmak yerine dusuk
# gecikmeli cekirdegi kullanir. Offline/debug akislar varsayilan BALANCED modu
# kullanmaya devam eder.
MODEL_FREE_GUI_RUNTIME_MODE = "FAST"
MODEL_FREE_FACE_DETECTION_ENABLED = True
# Eski detector'suz birim/debug cagrilarinin API uyumlulugu icindir. GUI bu
# kutuyu cizmez veya detector etkin akista analiz ROI'si olarak kullanmaz.
MODEL_FREE_GUIDE_DIAMETER_RATIO = 0.42
MODEL_FREE_GUIDE_CENTER_Y_RATIO = 0.48
MODEL_FREE_FACE_DETECTION_CONFIDENCE = 0.35
MODEL_FREE_FACE_PRESENCE_CONFIDENCE = 0.40
MODEL_FREE_FACE_TRACKING_CONFIDENCE = 0.35
MODEL_FREE_FACE_BOX_SMOOTHING_ALPHA = 0.32
MODEL_FREE_FACE_TRACK_HOLD_SECONDS = 0.80
MODEL_FREE_FACE_BOX_HORIZONTAL_EXPANSION = 0.10
MODEL_FREE_FACE_BOX_VERTICAL_EXPANSION = 0.12
MODEL_FREE_FACE_BOX_JUMP_IOU_THRESHOLD = 0.12
MODEL_FREE_FACE_EDGE_RELIABILITY_FACTOR = 0.75
MODEL_FREE_MODE_MODULES = {
    "FAST": (
        "global_fft",
        "moire",
        "dct_block_compression",
        "high_pass_residual",
    ),
    "BALANCED": (
        "global_fft",
        "moire",
        "radial_angular_spectrum",
        "periodicity",
        "dct_block_compression",
        "wavelet",
        "high_pass_residual",
    ),
    "RESEARCH": (
        "global_fft",
        "moire",
        "radial_angular_spectrum",
        "periodicity",
        "dct_block_compression",
        "wavelet",
        "high_pass_residual",
    ),
}
# Screen/phone/monitor borders and other full-frame geometry are never used
# as presentation-attack evidence. Kept as an explicit compatibility flag so
# exported configuration snapshots document the disabled behavior.
MODEL_FREE_FRAME_STRUCTURE_ENABLED = False
MODEL_FREE_CALIBRATION_FILE_PATH = (
    PROJECT_DIRECTORY / "model_free_calibration.json"
)
MODEL_FREE_MODULE_ENABLED = {
    "global_fft": True,
    "moire": True,
    "radial_angular_spectrum": True,
    "periodicity": True,
    "dct_block_compression": True,
    "wavelet": True,
    "high_pass_residual": True,
}

# Autocorrelation/cepstrum periodicity. These are transparent experimental
# operating points for synthetic testing; deployment values must come from a
# camera/PAI-specific calibration file and never from the final test split.
EXPERIMENTAL_PERIODICITY_MINIMUM_SIDE = 96
EXPERIMENTAL_PERIODICITY_MAXIMUM_SIDE = 384
EXPERIMENTAL_PERIODICITY_RESIDUAL_SIGMA = 2.0
EXPERIMENTAL_PERIODICITY_MINIMUM_RESIDUAL_STD = 0.20
EXPERIMENTAL_PERIODICITY_UNSUPPORTED_CLIPPING = 0.35
EXPERIMENTAL_PERIODICITY_MINIMUM_LAG_PIXELS = 4.0
EXPERIMENTAL_PERIODICITY_MAXIMUM_LAG_RATIO = 0.35
EXPERIMENTAL_PERIODICITY_PERIOD_TOLERANCE = 0.35
EXPERIMENTAL_PERIODICITY_PATCH_SCALE_RATIOS = (0.25, 0.50)
EXPERIMENTAL_PERIODICITY_TARGET_PATCH_COUNT = 20
EXPERIMENTAL_PERIODICITY_AUTOCORRELATION_START = 0.12
EXPERIMENTAL_PERIODICITY_AUTOCORRELATION_FULL = 0.48
EXPERIMENTAL_PERIODICITY_CEPSTRUM_Z_START = 5.0
EXPERIMENTAL_PERIODICITY_CEPSTRUM_Z_FULL = 18.0
EXPERIMENTAL_PERIODICITY_PATCH_VOTE_START = 0.15
EXPERIMENTAL_PERIODICITY_PATCH_VOTE_FULL = 0.65
EXPERIMENTAL_PERIODICITY_PATCH_AC_MINIMUM = 0.16
EXPERIMENTAL_PERIODICITY_PATCH_CEPSTRUM_Z_MINIMUM = 5.0
EXPERIMENTAL_PERIODICITY_AUTOCORRELATION_WEIGHT = 0.32
EXPERIMENTAL_PERIODICITY_CEPSTRUM_WEIGHT = 0.28
EXPERIMENTAL_PERIODICITY_PATCH_VOTE_WEIGHT = 0.30
EXPERIMENTAL_PERIODICITY_AGREEMENT_WEIGHT = 0.10
EXPERIMENTAL_PERIODICITY_TRIGGER_SCORE = 70.0
EXPERIMENTAL_PERIODICITY_SUPPORTING_SCORE = 45.0
EXPERIMENTAL_PERIODICITY_REQUIRED_PATCH_VOTE = 0.25
EXPERIMENTAL_PERIODICITY_REQUIRED_DOMAIN_AGREEMENT = 0.30
EXPERIMENTAL_PERIODICITY_MAXIMUM_RELIABILITY = 0.60
EXPERIMENTAL_PERIODICITY_HISTORY_SIZE = 10
EXPERIMENTAL_PERIODICITY_INVALID_RESET_FRAMES = 4

# Deneysel ortak yuz/ROI kalite kapisi.
EXPERIMENTAL_MODEL_FREE_MINIMUM_FACE_SIDE = 96
EXPERIMENTAL_MODEL_FREE_MINIMUM_FACE_AREA_RATIO = 0.035
EXPERIMENTAL_MODEL_FREE_MINIMUM_BLUR_SCORE = 90.0
EXPERIMENTAL_MODEL_FREE_MINIMUM_BRIGHTNESS = 45.0
EXPERIMENTAL_MODEL_FREE_MAXIMUM_BRIGHTNESS = 215.0
EXPERIMENTAL_MODEL_FREE_FRAME_EDGE_MARGIN_RATIO = 0.01

# Deneysel Global FFT frekans bantlari.
# Radius, FFT merkezinden goruntunun yarim kisa kenarina gore normalize edilir.
EXPERIMENTAL_FFT_DC_EXCLUSION_RADIUS = 0.02
EXPERIMENTAL_FFT_LOW_INNER_RADIUS = 0.02
EXPERIMENTAL_FFT_LOW_OUTER_RADIUS = 0.16
EXPERIMENTAL_FFT_MID_INNER_RADIUS = 0.16
EXPERIMENTAL_FFT_MID_OUTER_RADIUS = 0.45
EXPERIMENTAL_FFT_HIGH_INNER_RADIUS = 0.45
EXPERIMENTAL_FFT_ANALYSIS_OUTER_RADIUS = 0.92
EXPERIMENTAL_FFT_HIGH_OUTER_RADIUS = 0.92
EXPERIMENTAL_FFT_RADIAL_SLOPE_BIN_COUNT = 48
EXPERIMENTAL_FFT_MINIMUM_SLOPE_BIN_COUNT = 8

# Experimental mode: bunlar bilimsel veya evrensel yuz referanslari degildir.
# Her feature icin provisional [minimum, maximum], bu araligin disinda 100 puan
# sapmaya ulasilacak mesafe ve skor agirligi tanimlanir. Gercek calibration
# verisi geldiginde bu profil calibration dosyasindan uretilmelidir.
MODEL_FREE_GLOBAL_FFT_SCORING_MODE = "experimental"
EXPERIMENTAL_FFT_FEATURE_PROFILES = {
    "low_frequency_energy_ratio": {
        "minimum": 0.05,
        "maximum": 0.95,
        "deviation_scale": 0.20,
        "weight": 0.20,
    },
    "middle_frequency_energy_ratio": {
        "minimum": 0.03,
        "maximum": 0.80,
        "deviation_scale": 0.20,
        "weight": 0.10,
    },
    "high_frequency_energy_ratio": {
        "minimum": 0.005,
        "maximum": 0.65,
        "deviation_scale": 0.25,
        "weight": 0.20,
    },
    "spectral_centroid": {
        "minimum": 0.05,
        "maximum": 0.70,
        "deviation_scale": 0.20,
        "weight": 0.15,
    },
    "spectral_entropy": {
        "minimum": 0.10,
        "maximum": 0.99,
        "deviation_scale": 0.20,
        "weight": 0.15,
    },
    "spectral_slope": {
        "minimum": -8.0,
        "maximum": 0.25,
        "deviation_scale": 3.0,
        "weight": 0.10,
    },
    "high_to_low_energy_ratio": {
        "minimum": 0.005,
        "maximum": 5.0,
        "deviation_scale": 3.0,
        "weight": 0.10,
    },
}
EXPERIMENTAL_FFT_MAXIMUM_CONFIDENCE = 0.60
EXPERIMENTAL_FFT_EVIDENCE_DEVIATION = 0.50

# Deneysel Global FFT zamansal stabilizasyonu.
EXPERIMENTAL_FFT_HISTORY_SIZE = 12
EXPERIMENTAL_FFT_MINIMUM_VALID_FRAMES = 6
EXPERIMENTAL_FFT_SUSPICIOUS_SCORE = 60.0
EXPERIMENTAL_FFT_HIGH_ANOMALY_SCORE = 85.0
EXPERIMENTAL_FFT_INVALID_RESET_FRAMES = 3

# Deneysel Moire / periyodik desen frekans ve skor esikleri.
EXPERIMENTAL_MOIRE_INNER_RADIUS = 0.14
EXPERIMENTAL_MOIRE_OUTER_RADIUS = 0.82
EXPERIMENTAL_MOIRE_BACKGROUND_SIGMA = 4.0
EXPERIMENTAL_MOIRE_LOCAL_MAXIMUM_SIZE = 5
EXPERIMENTAL_MOIRE_PEAK_MINIMUM_DISTANCE = 5
EXPERIMENTAL_MOIRE_LOCAL_CONTRAST_SIZE = 9
EXPERIMENTAL_MOIRE_MINIMUM_PEAK_Z_SCORE = 6.5
EXPERIMENTAL_MOIRE_MINIMUM_CONTRAST_Z_SCORE = 4.0
EXPERIMENTAL_MOIRE_MINIMUM_PEAK_ENERGY_SHARE = 0.0002
EXPERIMENTAL_MOIRE_MAXIMUM_PEAK_COUNT = 24
EXPERIMENTAL_MOIRE_PEAK_PATCH_RADIUS = 2
EXPERIMENTAL_MOIRE_STRONG_PEAK_Z_SCORE = 11.0
EXPERIMENTAL_MOIRE_TARGET_PEAK_COUNT = 8
EXPERIMENTAL_MOIRE_MINIMUM_PERIODIC_PEAK_COUNT = 2
EXPERIMENTAL_MOIRE_PROMINENCE_WEIGHT = 0.55
EXPERIMENTAL_MOIRE_PEAK_COUNT_WEIGHT = 0.20
EXPERIMENTAL_MOIRE_PEAK_ENERGY_WEIGHT = 0.25
EXPERIMENTAL_MOIRE_ENERGY_CONCENTRATION_START = 0.002
EXPERIMENTAL_MOIRE_ENERGY_CONCENTRATION_FULL = 0.04
EXPERIMENTAL_MOIRE_SYMMETRY_TOLERANCE_PIXELS = 3.0
EXPERIMENTAL_MOIRE_MINIMUM_SYMMETRY_AMPLITUDE_RATIO = 0.35
EXPERIMENTAL_MOIRE_SYMMETRY_AMPLITUDE_WEIGHT = 0.70
EXPERIMENTAL_MOIRE_SYMMETRY_DISTANCE_WEIGHT = 0.30
EXPERIMENTAL_MOIRE_DIRECTION_BIN_COUNT = 12
EXPERIMENTAL_MOIRE_DIRECTION_SHARE_START = 0.35
EXPERIMENTAL_MOIRE_DIRECTION_SHARE_FULL = 0.75
EXPERIMENTAL_MOIRE_SCREEN_REPLAY_SYMMETRY_SCORE = 0.45
EXPERIMENTAL_MOIRE_SCREEN_REPLAY_DIRECTION_SCORE = 0.35
EXPERIMENTAL_MOIRE_PERIODIC_EVIDENCE_SCORE = 0.55
EXPERIMENTAL_MOIRE_SUPPORTING_PERIODIC_SCORE = 0.45
EXPERIMENTAL_MOIRE_SYMMETRY_EVIDENCE_SCORE = 0.55
EXPERIMENTAL_MOIRE_DIRECTION_EVIDENCE_SCORE = 0.50

EXPERIMENTAL_MOIRE_PERIODIC_WEIGHT = 0.60
EXPERIMENTAL_MOIRE_SYMMETRY_WEIGHT = 0.22
EXPERIMENTAL_MOIRE_DIRECTION_WEIGHT = 0.18
EXPERIMENTAL_MOIRE_SUSPICIOUS_SCORE = 72.0
EXPERIMENTAL_MOIRE_SCREEN_REPLAY_SCORE = 86.0
EXPERIMENTAL_MOIRE_RELEASE_SCORE = 52.0
EXPERIMENTAL_MOIRE_HISTORY_SIZE = 10
EXPERIMENTAL_MOIRE_MINIMUM_HISTORY = 6
EXPERIMENTAL_MOIRE_REQUIRED_SUSPICIOUS_FRAMES = 4
EXPERIMENTAL_MOIRE_REQUIRED_RELEASE_FRAMES = 4
EXPERIMENTAL_MOIRE_INVALID_RESET_FRAMES = 4
EXPERIMENTAL_MOIRE_REGION_IOU_RESET_THRESHOLD = 0.45
EXPERIMENTAL_MOIRE_LOCAL_PATCH_SIZE_RATIO = 0.15
EXPERIMENTAL_MOIRE_LOCAL_PATCH_MINIMUM_SIZE = 40
EXPERIMENTAL_MOIRE_LOCAL_PATCH_MAXIMUM_SIZE = 64
EXPERIMENTAL_MOIRE_LOCAL_PATCH_INNER_MARGIN_RATIO = 0.08
EXPERIMENTAL_MOIRE_LOCAL_PATCH_MAXIMUM_X_RATIO = 0.82
EXPERIMENTAL_MOIRE_LOCAL_PATCH_MAXIMUM_Y_RATIO = 0.78
EXPERIMENTAL_MOIRE_LOCAL_PATCH_MINIMUM_STD = 5.0
EXPERIMENTAL_MOIRE_LOCAL_PATCH_STRONG_SCORE = 50.0
EXPERIMENTAL_MOIRE_LOCAL_PATCH_MINIMUM_VOTES = 2
EXPERIMENTAL_MOIRE_LOCAL_PATCH_TOP_COUNT = 3
EXPERIMENTAL_MOIRE_LOCAL_PATCH_PEAK_WEIGHT = 0.70
EXPERIMENTAL_MOIRE_LOCAL_PATCH_CONSISTENCY_WEIGHT = 0.30
EXPERIMENTAL_MOIRE_LOCAL_SUPPORTING_SCORE = 45.0

# Module 3: Deneysel radial ve angular spectrum ayarlari.
# Bu profil araliklari bilimsel/evrensel esikler degildir; calibration verisi
# olmadan yalnizca konservatif gelistirme baslangic degerleridir.
RADIAL_ANGULAR_SCORING_MODE = "auto"
EXPERIMENTAL_RADIAL_ANGULAR_INNER_RADIUS = 0.02
EXPERIMENTAL_RADIAL_ANGULAR_OUTER_RADIUS = 0.92
EXPERIMENTAL_RADIAL_BIN_COUNT = 48
EXPERIMENTAL_ANGULAR_BIN_COUNT = 36
EXPERIMENTAL_ANGULAR_SECTOR_HALF_WIDTH_DEGREES = 12.5
EXPERIMENTAL_RADIAL_MINIMUM_FIT_BIN_COUNT = 8
EXPERIMENTAL_RADIAL_NARROW_BAND_NEIGHBOR_COUNT = 2

EXPERIMENTAL_RADIAL_FEATURE_PROFILES = {
    "radial_spectral_slope": {
        "minimum": -8.0,
        "maximum": 0.50,
        "deviation_scale": 3.0,
        "weight": 0.25,
    },
    "slope_fit_error": {
        "minimum": 0.0,
        "maximum": 0.65,
        "deviation_scale": 0.50,
        "weight": 0.20,
    },
    "radial_entropy": {
        "minimum": 0.15,
        "maximum": 1.0,
        "deviation_scale": 0.20,
        "weight": 0.15,
    },
    "dominant_radial_energy_ratio": {
        "minimum": 0.0,
        "maximum": 0.20,
        "deviation_scale": 0.30,
        "weight": 0.20,
    },
    "narrow_band_energy_concentration": {
        "minimum": 1.0,
        "maximum": 8.0,
        "deviation_scale": 8.0,
        "weight": 0.20,
    },
}

EXPERIMENTAL_ANGULAR_FEATURE_PROFILES = {
    "maximum_angular_energy": {
        "minimum": 0.0,
        "maximum": 0.20,
        "deviation_scale": 0.30,
        "weight": 0.25,
    },
    "angular_entropy": {
        "minimum": 0.25,
        "maximum": 1.0,
        "deviation_scale": 0.25,
        "weight": 0.20,
    },
    "directional_anisotropy": {
        "minimum": 0.0,
        "maximum": 0.75,
        "deviation_scale": 0.25,
        "weight": 0.25,
    },
    "horizontal_concentration": {
        "minimum": 0.0,
        "maximum": 0.65,
        "deviation_scale": 0.35,
        "weight": 0.10,
    },
    "vertical_concentration": {
        "minimum": 0.0,
        "maximum": 0.65,
        "deviation_scale": 0.35,
        "weight": 0.10,
    },
    "diagonal_concentration": {
        "minimum": 0.0,
        "maximum": 0.75,
        "deviation_scale": 0.25,
        "weight": 0.10,
    },
}

EXPERIMENTAL_RADIAL_SCORE_WEIGHT = 0.55
EXPERIMENTAL_ANGULAR_SCORE_WEIGHT = 0.45
EXPERIMENTAL_CALIBRATED_PROFILE_WEIGHT = 0.50
EXPERIMENTAL_CALIBRATED_PROFILE_DEVIATION_FULL = 0.35
EXPERIMENTAL_RADIAL_ANGULAR_HISTORY_SIZE = 10
EXPERIMENTAL_RADIAL_ANGULAR_MINIMUM_HISTORY = 6
EXPERIMENTAL_RADIAL_ANGULAR_SUSPICIOUS_SCORE = 60.0
EXPERIMENTAL_RADIAL_DIRECTIONAL_STATUS_SCORE = 60.0
EXPERIMENTAL_RADIAL_NARROW_BAND_STATUS_SCORE = 60.0
EXPERIMENTAL_RADIAL_ANGULAR_MAXIMUM_CONFIDENCE = 0.60
EXPERIMENTAL_RADIAL_ANGULAR_EVIDENCE_DEVIATION = 0.50
EXPERIMENTAL_RADIAL_ANGULAR_INVALID_RESET_FRAMES = 4
EXPERIMENTAL_RADIAL_ANGULAR_REGION_IOU_RESET_THRESHOLD = 0.45

# Module 4: DCT / Block Analysis.
#
# Kamera kareleri decode edilmis piksel matrisleri oldugu icin bu modul JPEG
# quantization tablosu veya double-JPEG gecmisi tespit ettigini iddia etmez.
# Ayarlar yalnizca mevcut piksel goruntusundeki 8x8 DCT dagilimlarini, blok
# siniri sureksizliklerini ve yerel tutarsizliklari puanlar.
DCT_BLOCK_SIZE = 8
DCT_SCORING_MODE = "auto"
DCT_NEAR_ZERO_COEFFICIENT_THRESHOLD = 1.0
DCT_INNER_FACE_MARGIN_RATIO = 0.14
DCT_LOCAL_PATCH_BLOCK_SIZE = 4
DCT_MINIMUM_BLOCK_COUNT = 64
DCT_MINIMUM_SOURCE_SIDE = 96
DCT_UNCERTAIN_SOURCE_SIDE = 160
DCT_UNCERTAIN_UPSCALE_RATIO = 1.60
DCT_UNCERTAIN_BLUR_SCORE = 80.0
DCT_EXTREME_SPARSITY_RATIO = 0.96

# Frekans bantlari, 8x8 DCT icindeki u+v indis toplamiyla tanimlanir. DC (0,0)
# dislanir; low=1..3, middle=4..7 ve high=8..14 olur.
DCT_LOW_FREQUENCY_MAX_INDEX_SUM = 3
DCT_MIDDLE_FREQUENCY_MAX_INDEX_SUM = 7

# Kalibrasyon yokken kullanilan gelistirme profilleri. Bunlar bilimsel yuz
# normlari degildir. Uyumlu calibration dosyasindaki dct_block_analysis /
# feature_profiles bolumu mevcutsa otomatik olarak ek referans kabul edilir.
EXPERIMENTAL_DCT_BAND_FEATURE_PROFILES = {
    "low_frequency_ac_energy_ratio": {
        "minimum": 0.30,
        "maximum": 0.995,
        "deviation_scale": 0.30,
        "weight": 0.30,
    },
    "middle_frequency_ac_energy_ratio": {
        "minimum": 0.003,
        "maximum": 0.55,
        "deviation_scale": 0.30,
        "weight": 0.25,
    },
    "high_frequency_ac_energy_ratio": {
        "minimum": 0.0001,
        "maximum": 0.30,
        "deviation_scale": 0.25,
        "weight": 0.25,
    },
    "ac_to_dc_ratio_mean": {
        "minimum": 0.00005,
        "maximum": 0.75,
        "deviation_scale": 0.50,
        "weight": 0.20,
    },
}

EXPERIMENTAL_DCT_SPARSITY_FEATURE_PROFILES = {
    "near_zero_ac_coefficient_ratio": {
        "minimum": 0.0,
        "maximum": 0.88,
        "deviation_scale": 0.12,
        "weight": 0.55,
    },
    "coefficient_entropy_mean": {
        "minimum": 0.08,
        "maximum": 0.98,
        "deviation_scale": 0.20,
        "weight": 0.25,
    },
    "coefficient_kurtosis_global": {
        "minimum": 1.0,
        "maximum": 80.0,
        "deviation_scale": 80.0,
        "weight": 0.20,
    },
}

EXPERIMENTAL_DCT_COMPONENT_WEIGHTS = {
    "dct_band_anomaly_score": 0.30,
    "coefficient_sparsity_score": 0.20,
    "blockiness_score": 0.30,
    "local_dct_inconsistency_score": 0.20,
}
EXPERIMENTAL_DCT_CALIBRATION_BLEND_WEIGHT = 0.55
EXPERIMENTAL_DCT_HISTORY_SIZE = 10
EXPERIMENTAL_DCT_MINIMUM_HISTORY = 5
EXPERIMENTAL_DCT_BLOCK_STRUCTURE_SCORE = 58.0
EXPERIMENTAL_DCT_LOCAL_INCONSISTENCY_SCORE = 62.0
EXPERIMENTAL_DCT_SUSPICIOUS_SCORE = 68.0
EXPERIMENTAL_DCT_MAXIMUM_CONFIDENCE = 0.55
EXPERIMENTAL_DCT_EVIDENCE_SCORE = 55.0
EXPERIMENTAL_DCT_INVALID_RESET_FRAMES = 4
EXPERIMENTAL_DCT_REGION_IOU_RESET_THRESHOLD = 0.45

# Blok siniri / yakin sinir-disi fark oraninin skora donusumu. 1.0 esit
# sureksizlik demektir; sadece sinirda belirgin fazlalik olmasi puan uretir.
EXPERIMENTAL_DCT_BLOCKINESS_RATIO_START = 1.08
EXPERIMENTAL_DCT_BLOCKINESS_RATIO_FULL = 2.20

# Yerel patch robust mesafelerinin deneysel skor donusumu.
EXPERIMENTAL_DCT_LOCAL_DISTANCE_START = 2.0
EXPERIMENTAL_DCT_LOCAL_DISTANCE_FULL = 7.0
EXPERIMENTAL_DCT_LOCAL_OUTLIER_DISTANCE = 3.5
EXPERIMENTAL_DCT_LOCAL_OUTLIER_RATIO_FULL = 0.35

# Module 5: Wavelet Analysis.
# PyWavelets runtime'da opsiyonel olarak import edilir. Kutuphane yoksa bu
# modul unavailable doner; diger model-free moduller calismaya devam eder.
WAVELET_NAME = "db2"
WAVELET_DECOMPOSITION_LEVELS = 2
WAVELET_BOUNDARY_MODE = "periodization"
WAVELET_SCORING_MODE = "auto"
WAVELET_USE_INNER_FACE_MASK = True
WAVELET_INNER_MASK_HORIZONTAL_RADIUS_RATIO = 0.40
WAVELET_INNER_MASK_VERTICAL_RADIUS_RATIO = 0.45
WAVELET_DETAIL_NEAR_ZERO_THRESHOLD = 1.0
WAVELET_LOCAL_PATCH_FACE_SIZE = 32
WAVELET_LOCAL_MINIMUM_MASK_COVERAGE = 0.35
WAVELET_MINIMUM_SOURCE_SIDE = 96
WAVELET_UNCERTAIN_SOURCE_SIDE = 160
WAVELET_UNCERTAIN_UPSCALE_RATIO = 1.60
WAVELET_UNCERTAIN_BLUR_SCORE = 80.0
WAVELET_CLIPPED_PIXEL_LOW = 2.0
WAVELET_CLIPPED_PIXEL_HIGH = 253.0
WAVELET_UNCERTAIN_CLIPPING_RATIO = 0.08
WAVELET_UNAVAILABLE_CLIPPING_RATIO = 0.25

# Kalibrasyon yokken kullanilan gelistirme araliklari. Bunlar bilimsel veya
# evrensel yuz-dokusu normlari degildir.
EXPERIMENTAL_WAVELET_ENERGY_FEATURE_PROFILES = {
    "level_1_detail_to_approximation_energy_ratio": {
        "minimum": 0.00001,
        "maximum": 0.45,
        "deviation_scale": 0.40,
        "weight": 0.30,
    },
    "level_2_detail_to_approximation_energy_ratio": {
        "minimum": 0.00001,
        "maximum": 0.55,
        "deviation_scale": 0.45,
        "weight": 0.25,
    },
    "global_detail_entropy_mean": {
        "minimum": 0.08,
        "maximum": 0.99,
        "deviation_scale": 0.20,
        "weight": 0.20,
    },
    "global_detail_sparsity_mean": {
        "minimum": 0.0,
        "maximum": 0.92,
        "deviation_scale": 0.08,
        "weight": 0.25,
    },
}

EXPERIMENTAL_WAVELET_DIRECTION_DOMINANCE_START = 0.55
EXPERIMENTAL_WAVELET_DIRECTION_DOMINANCE_FULL = 0.90
EXPERIMENTAL_WAVELET_ANISOTROPY_START = 0.35
EXPERIMENTAL_WAVELET_ANISOTROPY_FULL = 0.80
EXPERIMENTAL_WAVELET_LOCAL_DISTANCE_START = 2.0
EXPERIMENTAL_WAVELET_LOCAL_DISTANCE_FULL = 6.0
EXPERIMENTAL_WAVELET_LOCAL_OUTLIER_DISTANCE = 3.0
EXPERIMENTAL_WAVELET_LOCAL_OUTLIER_RATIO_FULL = 0.35
EXPERIMENTAL_WAVELET_NEIGHBOR_DIFFERENCE_START = 1.5
EXPERIMENTAL_WAVELET_NEIGHBOR_DIFFERENCE_FULL = 5.0

EXPERIMENTAL_WAVELET_COMPONENT_WEIGHTS = {
    "wavelet_energy_score": 0.35,
    "directional_wavelet_score": 0.25,
    "local_wavelet_inconsistency_score": 0.40,
}
EXPERIMENTAL_WAVELET_CALIBRATION_BLEND_WEIGHT = 0.55
EXPERIMENTAL_WAVELET_HISTORY_SIZE = 10
EXPERIMENTAL_WAVELET_MINIMUM_HISTORY = 5
EXPERIMENTAL_WAVELET_LOCAL_STATUS_SCORE = 50.0
EXPERIMENTAL_WAVELET_DIRECTIONAL_STATUS_SCORE = 60.0
EXPERIMENTAL_WAVELET_SUSPICIOUS_SCORE = 68.0
EXPERIMENTAL_WAVELET_MAXIMUM_CONFIDENCE = 0.55
EXPERIMENTAL_WAVELET_EVIDENCE_SCORE = 55.0
EXPERIMENTAL_WAVELET_INVALID_RESET_FRAMES = 4
EXPERIMENTAL_WAVELET_REGION_IOU_RESET_THRESHOLD = 0.45

# Module 6: High-Pass Residual Analysis.
# Tum filtreler ayni penceresiz float32 luminance crop'una uygulanir. Sayisal
# residual'lar signed kalir; uint8 normalizasyon yalnizca debug gorselleridir.
RESIDUAL_GAUSSIAN_KERNEL_SIZE = 5
RESIDUAL_GAUSSIAN_SIGMA = 1.2
RESIDUAL_LAPLACIAN_KERNEL_SIZE = 3
RESIDUAL_SOBEL_KERNEL_SIZE = 3
RESIDUAL_EDGE_MAGNITUDE_THRESHOLD = 24.0
RESIDUAL_SCORING_MODE = "auto"

# Model-free sabit mekansal agirlik maskesi. Ellipse crop disini, dikey ramp
# sac/kiyafet bolgelerini ve eye band katsayisi kas/goz etkisini azaltir.
RESIDUAL_MASK_HORIZONTAL_RADIUS_RATIO = 0.40
RESIDUAL_MASK_VERTICAL_RADIUS_RATIO = 0.45
RESIDUAL_MASK_TOP_RAMP_END_RATIO = 0.25
RESIDUAL_MASK_BOTTOM_RAMP_START_RATIO = 0.82
RESIDUAL_MASK_EYE_BAND_CENTER_RATIO = 0.38
RESIDUAL_MASK_EYE_BAND_HALF_HEIGHT_RATIO = 0.075
RESIDUAL_MASK_EYE_BAND_WEIGHT = 0.45
RESIDUAL_MINIMUM_MASK_WEIGHT = 0.08

RESIDUAL_LOCAL_PATCH_SIZE = 32
RESIDUAL_LOCAL_MINIMUM_MASK_COVERAGE = 0.30
RESIDUAL_MINIMUM_SOURCE_SIDE = 96
RESIDUAL_UNCERTAIN_SOURCE_SIDE = 160
RESIDUAL_UNCERTAIN_UPSCALE_RATIO = 1.60
RESIDUAL_UNCERTAIN_BLUR_SCORE = 90.0
RESIDUAL_UNCERTAIN_MINIMUM_BRIGHTNESS = 70.0
RESIDUAL_UNCERTAIN_MAXIMUM_BRIGHTNESS = 205.0
RESIDUAL_CLIPPED_PIXEL_LOW = 2.0
RESIDUAL_CLIPPED_PIXEL_HIGH = 253.0
RESIDUAL_UNCERTAIN_CLIPPING_RATIO = 0.08
RESIDUAL_UNAVAILABLE_CLIPPING_RATIO = 0.25
RESIDUAL_LOW_LIGHT_NOISE_RMS = 12.0

# Kalibrasyon yokken yalnizca konservatif gelistirme araliklari kullanilir.
# Range disindaki dusuk ve yuksek degerler iki tarafli sapma uretir.
EXPERIMENTAL_GAUSSIAN_RESIDUAL_FEATURE_PROFILES = {
    "gaussian_residual_variance": {
        "minimum": 6.25,
        "maximum": 625.0,
        "deviation_scale": 6.25,
        "weight": 0.24,
    },
    "gaussian_residual_mean_absolute_deviation": {
        "minimum": 1.50,
        "maximum": 20.0,
        "deviation_scale": 1.50,
        "weight": 0.20,
    },
    "gaussian_residual_rms_energy": {
        "minimum": 2.50,
        "maximum": 25.0,
        "deviation_scale": 2.50,
        "weight": 0.24,
    },
    "gaussian_residual_entropy": {
        "minimum": 0.10,
        "maximum": 0.99,
        "deviation_scale": 0.20,
        "weight": 0.12,
    },
    "gaussian_residual_kurtosis": {
        "minimum": 1.0,
        "maximum": 40.0,
        "deviation_scale": 40.0,
        "weight": 0.10,
    },
    "gaussian_residual_positive_negative_balance": {
        "minimum": -0.25,
        "maximum": 0.25,
        "deviation_scale": 0.50,
        "weight": 0.10,
    },
}

EXPERIMENTAL_LAPLACIAN_FEATURE_PROFILES = {
    "laplacian_variance": {
        "minimum": 100.0,
        "maximum": 8000.0,
        "deviation_scale": 100.0,
        "weight": 0.55,
    },
    "laplacian_rms_energy": {
        "minimum": 10.0,
        "maximum": 90.0,
        "deviation_scale": 10.0,
        "weight": 0.30,
    },
    "laplacian_kurtosis": {
        "minimum": 1.0,
        "maximum": 50.0,
        "deviation_scale": 50.0,
        "weight": 0.15,
    },
}

EXPERIMENTAL_GRADIENT_FEATURE_PROFILES = {
    "gradient_energy": {
        "minimum": 100.0,
        "maximum": 25000.0,
        "deviation_scale": 100.0,
        "weight": 0.50,
    },
    "gradient_mean_magnitude": {
        "minimum": 5.0,
        "maximum": 100.0,
        "deviation_scale": 5.0,
        "weight": 0.25,
    },
    "high_frequency_edge_density": {
        "minimum": 0.01,
        "maximum": 0.85,
        "deviation_scale": 0.01,
        "weight": 0.25,
    },
}

EXPERIMENTAL_RESIDUAL_LOCAL_DISTANCE_START = 2.0
EXPERIMENTAL_RESIDUAL_LOCAL_DISTANCE_FULL = 6.0
EXPERIMENTAL_RESIDUAL_LOCAL_OUTLIER_DISTANCE = 3.0
EXPERIMENTAL_RESIDUAL_LOCAL_OUTLIER_RATIO_FULL = 0.35
EXPERIMENTAL_RESIDUAL_NEIGHBOR_DIFFERENCE_START = 1.5
EXPERIMENTAL_RESIDUAL_NEIGHBOR_DIFFERENCE_FULL = 5.0

EXPERIMENTAL_RESIDUAL_COMPONENT_WEIGHTS = {
    "gaussian_residual_score": 0.30,
    "laplacian_score": 0.20,
    "gradient_score": 0.20,
    "local_residual_inconsistency_score": 0.30,
}
EXPERIMENTAL_RESIDUAL_CALIBRATION_BLEND_WEIGHT = 0.60
EXPERIMENTAL_RESIDUAL_HISTORY_SIZE = 10
EXPERIMENTAL_RESIDUAL_MINIMUM_HISTORY = 5
EXPERIMENTAL_RESIDUAL_ENERGY_STATUS_SCORE = 55.0
EXPERIMENTAL_RESIDUAL_LOCAL_STATUS_SCORE = 50.0
EXPERIMENTAL_RESIDUAL_SUSPICIOUS_SCORE = 68.0
EXPERIMENTAL_RESIDUAL_MAXIMUM_CONFIDENCE = 0.55
EXPERIMENTAL_RESIDUAL_EVIDENCE_SCORE = 55.0
EXPERIMENTAL_RESIDUAL_INVALID_RESET_FRAMES = 4
EXPERIMENTAL_RESIDUAL_REGION_IOU_RESET_THRESHOLD = 0.45

# Cross-method presentation-artifact safeguard.
#
# Wavelet ve residual modulleri siddetli clipping durumunda doku olcumunu
# guvenilmez sayip sonucu fusion'dan cikarir. Ekran tekrarlarinda gorulen bu
# clipping bilgisini tamamen kaybetmemek icin final fusion, clipping'i DCT yerel
# tutarsizligi ve FFT genis-bant dagilimi ile birlikte bir kez degerlendirir.
# Tek basina clipping saldiri skoru uretmez; sert isik/pozlama da ayni izi
# olusturabilir.
EXPERIMENTAL_PRESENTATION_CLIPPING_START = 0.10
EXPERIMENTAL_PRESENTATION_CLIPPING_FULL = 0.24
EXPERIMENTAL_PRESENTATION_DCT_LOCAL_START = 25.0
EXPERIMENTAL_PRESENTATION_DCT_LOCAL_FULL = 60.0
EXPERIMENTAL_PRESENTATION_FFT_MIDDLE_START = 0.055
EXPERIMENTAL_PRESENTATION_FFT_MIDDLE_FULL = 0.14
EXPERIMENTAL_PRESENTATION_FFT_ENTROPY_START = 0.54
EXPERIMENTAL_PRESENTATION_FFT_ENTROPY_FULL = 0.72
EXPERIMENTAL_PRESENTATION_DCT_SUPPORT_WEIGHT = 0.60
EXPERIMENTAL_PRESENTATION_FFT_SUPPORT_WEIGHT = 0.40
EXPERIMENTAL_PRESENTATION_MINIMUM_SUPPORT = 0.20
EXPERIMENTAL_PRESENTATION_CLIPPING_BASE_WEIGHT = 0.55
EXPERIMENTAL_PRESENTATION_SUPPORT_WEIGHT = 0.45

# Clipping bulunmayan ekranlarda ikinci yol: ayni karede FFT genis-bant enerji,
# DCT katsayi yogunlugu ve residual/wavelet ince doku sinyallerinin birlikte
# yukselmesi gerekir. Tek bir keskinlik veya entropy olcumu yeterli degildir.
EXPERIMENTAL_PRESENTATION_FFT_HIGH_TO_LOW_START = 0.015
EXPERIMENTAL_PRESENTATION_FFT_HIGH_TO_LOW_FULL = 0.080
EXPERIMENTAL_PRESENTATION_DCT_MIDDLE_START = 0.075
EXPERIMENTAL_PRESENTATION_DCT_MIDDLE_FULL = 0.170
EXPERIMENTAL_PRESENTATION_DCT_HIGH_START = 0.008
EXPERIMENTAL_PRESENTATION_DCT_HIGH_FULL = 0.045
EXPERIMENTAL_PRESENTATION_DCT_DENSE_COEFFICIENT_START = 0.55
EXPERIMENTAL_PRESENTATION_DCT_DENSE_COEFFICIENT_FULL = 0.35
EXPERIMENTAL_PRESENTATION_RESIDUAL_RMS_START = 4.0
EXPERIMENTAL_PRESENTATION_RESIDUAL_RMS_FULL = 10.0
EXPERIMENTAL_PRESENTATION_LAPLACIAN_VARIANCE_START = 1500.0
EXPERIMENTAL_PRESENTATION_LAPLACIAN_VARIANCE_FULL = 8500.0
EXPERIMENTAL_PRESENTATION_EDGE_DENSITY_START = 0.48
EXPERIMENTAL_PRESENTATION_EDGE_DENSITY_FULL = 0.75
EXPERIMENTAL_PRESENTATION_WAVELET_DENSE_DETAIL_START = 0.36
EXPERIMENTAL_PRESENTATION_WAVELET_DENSE_DETAIL_FULL = 0.20
EXPERIMENTAL_PRESENTATION_TRANSFORM_DCT_WEIGHT = 0.35
EXPERIMENTAL_PRESENTATION_TRANSFORM_RESIDUAL_WEIGHT = 0.40
EXPERIMENTAL_PRESENTATION_TRANSFORM_WAVELET_WEIGHT = 0.25
EXPERIMENTAL_PRESENTATION_BROADBAND_FFT_MINIMUM = 0.65
EXPERIMENTAL_PRESENTATION_BROADBAND_TRANSFORM_MINIMUM = 0.40
EXPERIMENTAL_PRESENTATION_BROADBAND_MINIMUM_SUPPORT_COUNT = 2
EXPERIMENTAL_PRESENTATION_BROADBAND_SUPPORT_START = 0.55
EXPERIMENTAL_PRESENTATION_BROADBAND_SUPPORT_FULL = 0.90
EXPERIMENTAL_PRESENTATION_BROADBAND_MINIMUM_SCORE = 25.0
EXPERIMENTAL_PRESENTATION_BROADBAND_FFT_WEIGHT = 0.55
EXPERIMENTAL_PRESENTATION_BROADBAND_TRANSFORM_WEIGHT = 0.45
EXPERIMENTAL_PRESENTATION_BROADBAND_COMPONENT_MINIMUM = 0.40

# Tam capraz-yontem kapisini gecmeyen fakat bir FFT + transform birlikteligi
# tasiyan kareler sifira dusurulmez. Bu kismi skor supheli esigin altinda
# sinirlanir; yalnizca Weak anomaly olabilir veya zamansal uyarinin korunmasina
# yardim eder.
EXPERIMENTAL_PRESENTATION_PARTIAL_FFT_MINIMUM = 0.35
EXPERIMENTAL_PRESENTATION_PARTIAL_TRANSFORM_MINIMUM = 0.15
EXPERIMENTAL_PRESENTATION_PARTIAL_COMPONENT_MINIMUM = 0.25
EXPERIMENTAL_PRESENTATION_PARTIAL_MINIMUM_SUPPORT_COUNT = 1
EXPERIMENTAL_PRESENTATION_PARTIAL_SUPPORT_START = 0.30
EXPERIMENTAL_PRESENTATION_PARTIAL_SUPPORT_FULL = 0.65
EXPERIMENTAL_PRESENTATION_PARTIAL_MINIMUM_SCORE = 25.0
EXPERIMENTAL_PRESENTATION_PARTIAL_MAXIMUM_SCORE = 49.0

# Ekran dokusu kamera autofocus'u nedeniyle kareler arasinda kaybolabilir.
# Uc guclu kanitin kisa bir pencereye dagilmasina izin verilir; aktif uyarinin
# kapanmasi icin daha uzun, kesintisiz recovery gerekir.
EXPERIMENTAL_PRESENTATION_TEMPORAL_HISTORY_SIZE = 12
EXPERIMENTAL_PRESENTATION_TEMPORAL_PERCENTILE = 70.0
EXPERIMENTAL_PRESENTATION_ACTIVATION_WINDOW = 8
EXPERIMENTAL_PRESENTATION_REQUIRED_SUSPICIOUS_HITS = 3
EXPERIMENTAL_PRESENTATION_REQUIRED_WEAK_HITS = 2
EXPERIMENTAL_PRESENTATION_REQUIRED_RECOVERY_FRAMES = 6

# Final two-stage mathematical fusion. The three shared-spectrum FFT modules
# are fused inside one family before that family enters the final calculation,
# so their correlated evidence cannot receive three full independent votes.
# All values below are experimental until model_free_calibration.json supplies
# a compatible mathematical_fusion section.
MATHEMATICAL_FUSION_CONFIG = {
    "module_groups": {
        "fft_family": (
            "fft",
            "moire",
            "radial_angular",
            "periodicity",
        ),
        "local_transform": ("dct_block", "wavelet", "residual"),
    },
    "module_weights": {
        "fft": 0.25,
        "moire": 0.30,
        "radial_angular": 0.20,
        "periodicity": 0.25,
        "dct_block": 0.34,
        "wavelet": 0.33,
        "residual": 0.33,
    },
    "group_weights": {
        "fft_family": 0.45,
        "local_transform": 0.55,
    },
    "minimum_valid_modules": 4,
    "minimum_valid_modules_per_group": {
        "fft_family": 1,
        "local_transform": 1,
    },
    "minimum_effective_confidence": 0.01,
    "uncalibrated_confidence_cap": 0.60,
    "history_size": 12,
    "minimum_history": 5,
    "weak_anomaly_score": 25.0,
    "suspicious_score": 50.0,
    "high_risk_score": 75.0,
    "recovery_score": 42.0,
    "required_suspicious_frames": 3,
    "required_recovery_frames": 3,
    "invalid_reset_frames": 4,
    "region_iou_reset_threshold": 0.45,
    "module_evidence_score": 50.0,
}

# Stage-B attack fusion. Family weights express physical relevance, not
# learned coefficients. They remain experimental until a calibration protocol
# records a compatible deployment section.
PRECONTROL_ATTACK_FUSION_CONFIG = {
    "method_weights": {
        "fft": 0.25,
        "moire": 0.30,
        "radial_angular": 0.20,
        "periodicity": 0.25,
        "dct_block": 1.0,
        "wavelet": 0.50,
        "residual": 0.50,
    },
    "attack_family_weights": {
        "replay_screen_score": {
            "frequency": 0.55,
            "compression_recapture": 0.20,
            "spatial_texture": 0.25,
        },
        "print_attack_score": {
            "frequency": 0.15,
            "compression_recapture": 0.45,
            "spatial_texture": 0.40,
        },
        "recapture_score": {
            "frequency": 0.40,
            "compression_recapture": 0.35,
            "spatial_texture": 0.25,
        },
        # These remain explicit unsupported outputs until their evidence
        # families are implemented and calibrated.
        "planar_surface_score": {},
        "physiological_absence_score": {},
        "sensor_inconsistency_score": {},
    },
    "expected_family_methods": {
        "frequency": 4,
        "compression_recapture": 1,
        "spatial_texture": 2,
    },
    "suspicious_score": 50.0,
    "high_risk_score": 75.0,
    "minimum_decision_reliability": 0.35,
    "minimum_live_reliability": 0.65,
    "minimum_supported_families": 2,
}

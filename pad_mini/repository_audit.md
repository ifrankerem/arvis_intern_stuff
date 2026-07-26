# Deterministic PreControl Repository Audit

Audit date: 2026-07-26  
Scope: every Python source file, configuration entry, test, UI integration point, and generated-debug path in this repository.  
Audit boundary: this document describes the repository before the Phase 13 P0 implementation. Existing uncommitted work was treated as user-owned and was not reverted.

Post-audit implementation note (2026-07-26): the GUI PreControl path now reuses MediaPipe strictly as ROI localization/tracking infrastructure, with EMA box smoothing and a bounded short-loss hold. The fixed center guide is no longer displayed or used by the detector-enabled GUI path; a missing detected face forces unsupported/insufficient-quality output. The audit table below intentionally remains the pre-change baseline.

## Executive findings

The repository has two distinct frame-processing paths. `FaceQualityApplication` uses MediaPipe landmarks for face quality/alignment infrastructure, while `ModelFreePreControlApplication` deliberately loads no face model and analyzes a fixed square guide. The model-free path therefore does not currently receive a detected or aligned face, even though some field names say “face” and “aligned”.

All six deterministic modules consume the same 256×256 grayscale guide crop. Global FFT, moiré, and radial/angular analysis share a mean/std-normalized, 2-D Hann-windowed FFT. DCT, wavelet, and residual methods consume the unwindowed resized grayscale crop. This is computationally efficient, but it removes sensor provenance, hides original compression-grid alignment, and creates correlated evidence.

The camera thread preserves only the newest decoded BGR frame and a counter. It does not preserve acquisition timestamps, source/container timestamps, nominal FPS, inter-frame intervals, dropped-frame estimates, encoded bytes, codec/JPEG provenance, or resize/transcode history. Temporal flicker, rPPG, and motion methods are consequently unsupported in the audited baseline.

The current fusion correctly excludes unavailable methods instead of treating them as zero and first combines the three FFT-derived methods as one family. It nevertheless has only two broad families, no attack-specific risks, no common reason-code contract, no per-method runtime, and no final abstention taxonomy. Its built-in thresholds are explicitly experimental and are not deployment calibration.

## Audit table

| Component | File and function | Current input | Current output | Mathematical operation | Existing threshold | Weakness | Recommended change | Backward-compatibility risk |
|---|---|---|---|---|---|---|---|---|
| Program entry | `main.py:main` | Process invocation | Tk application | Instantiates GUI only | None | No headless or benchmark entry | Keep GUI entry; add separate benchmark CLI | Low |
| Camera discovery | `camera_sources.py:discover_camera_sources`, `parse_camera_source`, `build_droidcam_url` | V4L2 devices, integer index, URL | OpenCV-compatible source | Device enumeration and URL parsing | Probe indices 0–5; DroidCam default port 4747 | No source/protocol metadata object | Add immutable capture-source descriptor while retaining accepted scalar sources | Low |
| Latest-frame capture | `camera_stream.py:LatestFrameCamera` | OpenCV `VideoCapture` decoded frames | `(new, frame, frame_number)` | Background read; drops queued/old frames | 10 consecutive errors; buffer size 1 | No monotonic acquisition time, source PTS, FPS, cadence, codec, drop count, or provenance | Add metadata-returning API and retain the three-value API | Low |
| GUI frame loop | `application_gui.py:FaceQualityGui._update_frame` | Latest decoded BGR frame | Preview and app result | Polls every 5 ms | Tk delay 5 ms | Discards capture timing at app boundary | Pass capture metadata when the selected application accepts it | Medium |
| Landmark infrastructure | `face_landmarker.py:FaceLandmarker` | Mirrored BGR frame, monotonic-ms timestamp | Face box, landmarks, blendshapes, pose matrix | MediaPipe face landmark inference and geometric extraction | Maximum two faces | Valid infrastructure, but isolated from model-free path; must never make liveness decision | Optionally pass detected ROI/landmarks as provenance-only infrastructure | Medium |
| Model-based quality path | `face_quality_application.py:FaceQualityApplication.process_frame` | Raw camera frame | `FrameProcessingResult` and JSON-ready quality/alignment | BGR mirror, landmark detection, Laplacian blur, brightness, pose/mouth geometry | Area 0.08; blur 80; brightness 60–200; pose/mouth limits in `config.py` | Reported `processing_time_ms` is elapsed session time, not frame latency; not connected to PreControl | Correct timing separately; keep model path independent | Medium |
| Model-free guide ROI | `model_free_pre_control_application.py:create_guide_box`, `prepare_frame` | Decoded camera frame | Mirrored analysis frame and fixed square | Square side `0.58×min(W,H)`, centered at `x=0.5W`, `y=0.48H` | Fixed geometry | It is not a detected face and may contain hair, clothing, background, or no face | Represent it honestly as `guide_roi`; add raw/expanded/subregion/background/full-frame ROI set | Medium |
| Shared context/quality gate | `model_free_analysis.py:ModelFreePreControlContextBuilder.build`, `_measure_quality` | Mirrored frame and guide box | Shared crop, quality values, FFT tensors | Clamp/crop; grayscale; brightness mean; Laplacian variance; resize to 256 | Side ≥96; area ≥0.035; blur ≥90; brightness 45–215; edge margin 0.01 | “Aligned” crop is an alias when none supplied; one hard gate suppresses all methods; no clipping fraction or transform provenance | Add ROI/provenance/capture metadata; use method-specific support/reliability gates | Medium |
| Shared FFT preprocessing | `model_free_analysis.py:_standardize_aligned_crop`, `_prepare_fft_crop`, `_create_fft_window` | Guide crop | Float grayscale, windowed normalized crop, complex FFT, magnitude/power/log views | AREA/CUBIC resize; zero mean/unit std; separable Hann; `fft2`/`fftshift` | Size 256; window `hann` | Only Hann/none; resized grayscale only; no Y/chroma/RGB/gradient/residual planes | Support Hann/Hamming/Tukey comparison and plane provenance; share transforms by exact input signature | Medium |
| Global FFT | `global_fft_pre_control.py:GlobalFFTPreController.analyze` | Shared power spectrum | Ratios, centroid, entropy, slope, score/history/debug | Annular sums; normalized Shannon entropy; energy centroid; OLS log radial slope | DC 0.02; low 0.02–0.16; mid 0.16–0.45; high 0.45–0.92; suspicious 60; high 85 | Not merely a high-frequency ratio, but still lacks flatness, roll-off, kurtosis, non-DC/symmetric peaks, robust slope, channel and local/global comparisons; overlaps radial/moiré | Add raw candidate features and robust fits; keep family-level correlation control | Medium |
| Moiré/periodic peaks | `moire_pre_control.py:MoirePeriodicPatternPreController.analyze` | Shared global log/power spectrum | Peak list, prominence/energy/symmetry/direction scores and temporal status | Gaussian spectral baseline; median/MAD z; local maxima/NMS; inversion symmetry; axial histogram | Annulus 0.14–0.82; peak z 6.5; contrast z 4; max 24; suspicious 72; replay 86 | Local spectral-neighborhood comparison exists, but no local patch FFT, multiscale spatial voting, harmonic/lattice score, or patch heatmap; natural periodic textures can trigger it | Add multiscale patch votes and require cross-measure consistency; compare face/expanded/background | Medium |
| Radial/angular spectrum | `radial_angular_pre_control.py:RadialAngularSpectrumPreController.analyze` | Shared power spectrum | Radial profiles/slope/entropy/peak and angular entropy/anisotropy/circular statistics | 48 annular means/energies; OLS log-log fit; 36 axial bins; doubled-angle circular mean/variance | Radius 0.02–0.92; 48 radial/36 angular bins | Circular treatment is correct, but regression is not robust and evidence is one ROI/scale; dominant bin is unstable | Add robust slope, peak confidence, cross-ROI and multiscale orientation stability | Medium |
| DCT/JPEG/block | `dct_block_pre_control.py:DCTBlockAnalysisPreController.analyze` | Unwindowed resized grayscale crop | 8×8 coefficient and boundary metrics, local map, score/history | Per-block `cv2.dct`; band energies; zero ratio; entropy/kurtosis; phase boundary differences | Block 8; near-zero 1; quality and risk thresholds in `config.py` | Resize can destroy/forge 8×8 alignment; decoded video has no JPEG quantization tables; fixed grid is not encoded-file double-JPEG evidence | Analyze unresized raw ROI with all 8 grid phases; record origin; mark quantization/double-JPEG unsupported without encoded data | Medium |
| Wavelet | `wavelet_pre_control.py:WaveletAnalysisPreController.analyze` | Unwindowed resized grayscale crop | Per-level LL/LH/HL/HH statistics and local map | PyWavelets `dwt2`, `db2`, two levels, periodization; energy/entropy/kurtosis/sparsity/directional balance | Default `db2`, level 2; quality/risk thresholds in `config.py` | No family ablation; no calibrated cross-scale statistic; resampling and residual/DCT correlation | Add Haar/db/sym/bior ablation tool; retain one deployment wavelet only after held-out validation | Medium |
| High-pass/residual | `high_pass_residual_pre_control.py:HighPassResidualPreController.analyze` | Unwindowed resized grayscale crop | Gaussian/Laplacian/gradient and local residual metrics/maps | Gaussian subtraction; Laplacian; Sobel; weighted region stats, entropy/kurtosis and local robust inconsistency | Kernel 5, σ1.2 and thresholds in `config.py` | No median/bilateral/DoG/local-mean variants, residual autocorrelation/spectrum/channel consistency; cannot distinguish sensor noise after resize | Add residual-bank ablation and periodicity metrics on unresampled raw ROI; do not label PRNU | Medium |
| Two-stage fusion | `mathematical_fusion.py:MathematicalFusionController.analyze` | Six legacy module results plus context | Combined risk, two group scores, history/status/evidence | Reliability-weighted arithmetic mean within `fft_family` and `local_transform`, then weighted group mean; temporal median/70th percentile | ≥4 valid; ≥1/group; group weights 0.45/0.55; suspicious 50; high 75 | Correct missing-data behavior and some correlation control; only two families; confidence is not explicit uncertainty coverage; no attack-specific outputs/final states/reason codes | Introduce common method/family contract, eight-family-capable fusion, attack score/coverage matrix, abstention-first state mapping | Medium–High |
| Calibration loading | `mathematical_fusion.py:_load_calibration`, `_map_calibrated_score` | Optional `model_free_calibration.json` | Active weights/thresholds and linear score mapping | Mean/std-based mapping and config compatibility check | Schema/fusion versions 2 | No robust median/MAD feature calibration; defaults remain broad experimental score heuristics | Add human-readable per-feature direction, median/MAD/percentile/winsor mapping and split provenance | Medium |
| Shared legacy result | `model_free_analysis.py:ModelFreeAnalysisResult` | Analyzer values | Module result consumed by UI/fusion/export/tests | Dataclass plus convenience properties/factories | Score convention 0–100; confidence 0–1 | Missing evidence family, attack targets, supported alias, normalized 0–1 score, reason codes, explanation, visualization paths, runtime | Extend with backward-compatible default fields and canonical serialization | Low–Medium |
| UI | `application_gui.py`, `model_free_pre_control_application.py:draw_*` | Frame, module/fusion results | Live preview, scores, live spectrum, warnings | Tk rendering and OpenCV overlays | GUI-specific display cutoffs | No attack-specific state/reliability/unsupported reason view; spectrum is descriptive only | Bind UI to canonical decision while retaining current labels | Medium |
| Debug export | `model_free_pre_control_application.py:save_debug_sample`, `_save_unified_visualizations` | Latest context/results | Schema-v2 JSON/text and PNG maps | Serializes computed maps and metrics | Debug save on demand | Good computed-measurement discipline; lacks runtime, ROI provenance, capture timing, method contract, autocorrelation/cepstrum | Upgrade schema additively; never fabricate unavailable visualization | Low |
| Tests | `tests/test_dct_block_pre_control.py`, `test_wavelet_pre_control.py`, `test_high_pass_residual_pre_control.py`, `test_mathematical_fusion.py`, `test_application_gui.py` | Synthetic images/results | Unit assertions | Deterministic signal generation and fusion checks | 50 tests in audited baseline | No direct FFT/moiré/radial tests; no timestamp/flicker/motion/color/rPPG/JPEG-byte tests | Add mathematical monotonicity, rotation, support/reliability, capture metadata and output-contract suites | Low |
| Visualization fixtures | `model_free_debug/`, `fft_samples/`, `dct_samples/`, `wavelet_samples/`, `residual_samples/` | Saved runs | Images/reports | Debug artifacts | None | Generated files can be confused with benchmark evidence | Keep out of metrics; record configuration and provenance in every run | Low |

## Current method inputs and transformations

| Method | Actual audited ROI | Resize | Color representation | Window | Temporal buffer |
|---|---|---:|---|---|---|
| Global FFT | Fixed guide square | 256×256 | BGR→gray, mean/std | 2-D Hann | Module score deque only |
| Moiré | Fixed guide square | 256×256 | Same shared gray FFT | 2-D Hann | Score/state deque only |
| Radial/angular | Fixed guide square | 256×256 | Same shared gray FFT | 2-D Hann | Score deque only |
| DCT/block | Fixed guide square | 256×256 | BGR→gray | None | Score deque only |
| Wavelet | Fixed guide square | 256×256 | BGR→gray | None | Score deque only |
| High-pass residual | Fixed guide square | 256×256 | BGR→gray | None | Score deque only |

There is no image-frame ring buffer. Each module stores only scalar/history state needed for stabilization. Original frames are discarded by the latest-frame camera after replacement. The `time.time()` supplied to model-free analysis is a processing-time wall-clock timestamp rather than capture PTS.

## Existing formulas and duplicated evidence

Let `P(u,v)=|FFT2(w·(I-μ)/σ)|²`, with a separable Hann window `w`. Global FFT integrates `P` in annuli; radial/angular analysis rebins the same `P`; moiré detects local maxima in a robustly background-normalized version of `log(1+P)`. These are three views of one transform and one spatial ROI, so they are correlated, not three independent sensors. The current first-stage `fft_family` grouping is therefore directionally correct.

DCT, wavelet, and high-pass residual all measure local high-frequency structure on the same resampled luminance crop. Their bases differ, but blur, sharpening, interpolation, and compression can move all three scores together. Treating them as one “local transform” family reduces but does not quantify this correlation.

## Threshold and calibration status

Every built-in decision profile is labelled experimental in `config.py`. No compatible `model_free_calibration.json` is shipped, no device-specific calibration is selected at runtime, and no held-out benchmark result exists in the repository. Therefore:

- current scores are engineering indicators, not validated PAD probabilities;
- an apparently low risk must not be emitted as a confident `LIVE` decision;
- no APCER, BPCER, ACER, ROC, EER, or operating point can truthfully be reported from this repository yet;
- calibration, development, and final test captures must be identity/device/PAI-disjoint as specified in `calibration_protocol.md` and `benchmark_protocol.md`.

## Public API compatibility requirements

The following interfaces are already consumed internally or by tests and should be preserved:

- `LatestFrameCamera.read_latest(previous_frame_number) -> (bool, frame, number)`;
- `ModelFreePreControlApplication.process_frame(camera_frame)`;
- the six keys in `latest_pre_control_results`;
- legacy `ModelFreeAnalysisResult` fields and convenience properties;
- `MathematicalFusionController.analyze(results, context)`;
- `FrameProcessingResult(display_frame, face_image, analysis_result)`;
- schema-v2 fields in saved debug reports.

New metadata, result-contract, ROI, and decision fields should therefore be additive. A schema version increase is appropriate for exports, but existing fields must remain available.

## Audit conclusion

Major changes may proceed only behind additive interfaces. The immediate P0 work is: capture/provenance metadata, explicit ROI topology, a canonical method-result adapter with runtimes/reason codes, robust configurable normalization, autocorrelation/cepstrum as corroborating frequency evidence, family/attack-specific uncertainty-aware decision output, synthetic tests, and benchmark tooling. Temporal flicker, geometry, rPPG, color/material, and sensor-forensic decisions remain unsupported until their required data and calibration exist.

# Implementation Priority and Approval Record

The user request authorizes implementation after audit. This file records the approved P0 boundary used in this change set; it avoids silently treating every researched idea as production-ready.

## P0 — implemented now

| Change | Why | Integration | Cost | Validation |
|---|---|---|---|---|
| Capture timing/provenance | Temporal methods cannot be interpreted without cadence/source facts | Additive `read_latest_with_metadata`; context/debug report | Negligible | API compatibility and skipped-frame tests |
| ROI topology/provenance | One transformed crop is unsafe for every method | `ModelFreePreControlContext.rois` | Memory copies; bounded | ROI/mask/provenance unit test |
| Canonical method contract | Needed for support, reliability, reasons, runtime and API | Additive fields + `to_method_result` adapter | Negligible | Contract serialization test |
| Robust normalization primitives | Raw metrics must have explicit direction/mapping | `deterministic_normalization.py` | Negligible | high/low/outside monotonic tests |
| Extended Global FFT raw features | Current FFT lacked requested shape/peak descriptors | Existing `GlobalFFTPreController` | `O(N)` after shared FFT | finite/bounded feature tests |
| Hann/Hamming/Tukey support | Required leakage ablation | Context builder, Hann remains default | None at runtime | four-window unit test |
| Autocorrelation + real cepstrum | Valuable single-frame corroboration of weak periodic structure | New frequency-family module | ≈17 local inverse FFT analyses/frame in BALANCED | amplitude, rotation, unsupported, debug tests |
| Correlation-aware family integration | New periodicity must not gain an independent full vote | Existing Stage-A `fft_family` | Negligible | fusion regression suite |
| Attack-specific uncertainty output | A low-confidence low score must not become LIVE | `PreControlDecisionBuilder` and debug JSON | Negligible | no-LIVE-without-calibration and quality abstention tests |
| FAST/BALANCED/RESEARCH profiles | Explicit CPU/deployment trade-off | `config.py`; BALANCED default | Mode-dependent | full regression suite |
| Benchmark and synthetic tooling | Required reproducible validation path | `benchmarks/`, `benchmark_metrics.py`, fixtures | Offline | synthetic runner and metrics tests |

## P1 — after core capture/calibration data exists

1. Patch FFT/harmonic/lattice upgrade for moiré with face/expanded/background voting.
2. Original-grid DCT phase scan and encoded JPEG/video provenance ingestion.
3. Wavelet-family ablation; retain only a demonstrably independent family/default.
4. Residual-bank ablation with residual autocorrelation and channel consistency.
5. Temporal flicker/PWM/row-phase detector using `temporal_sampling.py` support gates.
6. Classical LK/Farnebäck + RANSAC homography/affine/parallax support evidence.
7. POS/CHROM with skin-mask ROIs, SQI, PSD and cross-ROI coherence; positive evidence only.
8. Deterministic chromaticity and conservative passive specularity.
9. LBP/LPQ/Gabor/HOG statistics only after bona-fide range calibration.
10. High-resolution print halftone and ESF/LSF edge-profile measurements.
11. Blind residual/CFA/provenance consistency on unresampled raw inputs.

## P2 — experimental or hardware-dependent

- trusted-camera PRNU enrollment/matching;
- autofocus sweep/depth from defocus;
- exposure/flash/torch challenge response;
- multi-camera parallax;
- polarization/NIR/depth hardware.

These must be separate optional capture capabilities, never silently mixed into passive RGB scores.

## Rejected

- CNN, ViT/TimeSformer, pretrained PAD/deepfake models and learned feature extractors;
- SVM, Random Forest, XGBoost, k-NN, logistic regression or learned fusion;
- F3-Net/learned frequency models, CMA’s learned adapter, learned edge dictionaries, Noiseprint and neural rPPG/flow;
- blink-as-liveness and “little motion equals attack” rules;
- double-JPEG or PRNU source claims when encoded/reference evidence is absent.

## Completion gates for P1

A P1 method becomes decision-enabled only when its support rules, raw features, deterministic normalization, debug measurement, latency, calibration split, per-PAI APCER/BPCER effect, correlation matrix, and leave-one-family-out ablation are recorded. Runtime-only redundancy is removed or down-weighted.


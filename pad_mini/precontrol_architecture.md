# Deterministic PreControl Architecture

## Outcome

The architecture preserves the existing GUI and six module APIs while adding provenance, explicit support/reliability, a seventh autocorrelation/cepstrum method, attack-specific risks and abstention. It remains passive and RGB-only; the PAD decision is deterministic and model-free. MediaPipe now supplies ROI infrastructure but cannot vote on liveness.

```text
decoded frame + capture metadata
              |
              v
 MediaPipe ROI localization/tracking (infrastructure only)
              |
              v
 raw/mirrored frame and ROI topology
 (raw, aligned*, expanded, subregions, background ring, full frame)
              |
              v
 shared computations keyed by exact input/provenance
 (gray, window, FFT, gradients/residuals where safe)
              |
              +-----------------------------+
              | deterministic methods       |
              | score + reliability + raw   |
              +-----------------------------+
              |
              v
 Stage A: within-family reliability-weighted fusion
              |
              v
 Stage B: attack-specific family fusion
              |
              v
 overall score + uncertainty interval + abstention-aware state
```

`aligned*` currently means an identity alias of the detected raw face ROI unless external alignment is explicitly supplied. No fixed center guide is displayed. When no face is detected an empty ROI placeholder is marked unsupported and cannot produce a PAD score. Provenance records these distinctions so sensor/DCT methods cannot mistake an alias for a geometrically aligned or untransformed source.

## Capture packet

`LatestFrameCamera.read_latest_with_metadata` adds acquisition monotonic/wall time, interarrival interval, nominal FPS, source position when exposed by OpenCV, source kind, timestamp basis/reliability, decoded-byte status and consumer-skipped frames. The legacy `read_latest` triple remains unchanged.

Decoder-arrival time is not source exposure time. It is adequate for cadence diagnostics, not automatically adequate for flicker phase or rPPG. `temporal_sampling.assess_temporal_sampling` returns unsupported on insufficient, non-monotonic or very irregular timestamps.

## ROI strategy

`ModelFreePreControlContext.rois` contains `ModelFreeROI` values with frame box, optional mask, coordinate space, transform history, semantic basis, dimensions and validity. Current subregions are deterministic rectangles relative to a verified landmark face box; they are approximate regions rather than pixel-accurate anatomical/skin masks.

- FFT/radial/global residual: detected raw face ROI; future comparisons use expanded/background.
- Moiré: global FFT plus single-scale local patch FFT/voting in the detected face ROI; P1 adds multiscale and expanded/background comparisons.
- DCT/grid: unresized raw ROI is the intended P1 source; never an aligned/resized crop for encoded-grid claims.
- Wavelet/microtexture: aligned ROI where stable comparison matters.
- rPPG/chromaticity: forehead/left cheek/right cheek with skin validity.
- Specularity: forehead/cheeks/nose and eye/glasses exclusion.
- Flow/homography: tracked raw face, relative facial subregions and background.
- Sensor/PRNU/CFA: raw unresampled ROI/full frame only.
- Flicker/rolling shutter: stable luminance/chroma ROIs and full-frame rows with capture timing.

## Method contract

New integrations consume:

```python
MethodResult(
    method_name: str,
    evidence_family: str,
    attack_targets: list[str],
    supported: bool,
    raw_metrics: dict[str, object],
    normalized_score: float,       # [0, 1], not a probability
    reliability: float,            # [0, 1]
    triggered: bool,
    reason_codes: list[str],
    human_explanation: str,
    visualization_paths: dict[str, str],
    runtime_ms: float,
    warnings: list[str],
)
```

Legacy `ModelFreeAnalysisResult` retains its 0–100 raw/stabilized fields and exposes additive family/target/reason/runtime fields plus `to_method_result()`. Unsupported inputs have no raw score and zero reliability.

## Normalization

`deterministic_normalization.py` implements robust z, winsorization, high/low/outside suspicion direction, and monotonic cubic smoothstep. For a calibrated median `m` and MAD `d`, signed robust standardization is `z=(x-m)/(1.4826d+ε_floor)`. The floor is feature-specific configuration, not a hidden universal constant.

Each feature configuration must declare:

- units and ROI/transform;
- high/low/outside suspicion direction;
- median/MAD or valid interval from calibration captures;
- winsor percentiles and smoothstep start/full points;
- device/profile/version and split provenance.

The P0 periodicity module uses transparent experimental smoothstep bounds. Existing six method scores remain experimental until migrated to per-feature robust calibration. This is why uncalibrated low scores cannot yield `LIVE`.

## Two-stage fusion

Stage A uses a reliability-weighted arithmetic mean:

`F_k = Σ_i(w_i r_i s_i) / Σ_i(w_i r_i)`

within an evidence family. Missing/unsupported methods are excluded. Family reliability is mean method reliability multiplied by coverage. Global FFT, moiré, radial/angular and autocorrelation/cepstrum share the `frequency` family, preventing four full votes for one periodic artifact.

Stage B uses the same transparent weighted mean across families for each attack. Current experimental family relevance is:

| Attack output | Frequency | Compression/recapture | Spatial texture |
|---|---:|---:|---:|
| replay_screen_score | 0.55 | 0.20 | 0.25 |
| print_attack_score | 0.15 | 0.45 | 0.40 |
| recapture_score | 0.40 | 0.35 | 0.25 |

`planar_surface_score`, `physiological_absence_score`, and `sensor_inconsistency_score` are explicit unsupported outputs until their families exist. Noisy-OR was rejected for current Stage A because correlated moderate methods accumulate aggressively. It remains an ablation candidate across demonstrably independent families only.

Overall risk is the highest supported attack-specific risk, accompanied by reliability and interval:

- lower bound `S·r`;
- upper bound `S+(100-S)(1-r)`;
- uncertainty `100(1-r)`.

This is an uncertainty display interval, not a statistical confidence interval.

## State machine

- invalid frame quality → `INSUFFICIENT_QUALITY`, no invented score;
- no supported attack output → `UNSUPPORTED_CAPTURE`;
- inadequate family count/reliability → `INSUFFICIENT_EVIDENCE`;
- uncalibrated score ≥ experimental suspicious threshold → `SUSPICIOUS`;
- uncalibrated low score → `INSUFFICIENT_EVIDENCE`, never `LIVE`;
- calibrated high/suspicious operating points → `HIGH_RISK`/`SUSPICIOUS`;
- calibrated low risk plus reliability ≥0.65 → `LIVE`.

## Runtime modes

- **FAST:** global FFT, moiré, DCT/block and residual. Four-module minimum remains satisfied.
- **BALANCED (default):** all six prior methods plus autocorrelation/cepstrum.
- **RESEARCH:** currently BALANCED plus all debug outputs; future P1 modules belong here first.

Production debug image generation remains off/on-demand. Periodicity performs two patch scales and is the main added BALANCED CPU cost. Deployment profiling must determine an analysis stride; resolution cannot be reduced until periodic artifacts are checked for loss.

## Debug/explainability

Debug schema 3 adds capture metadata, ROI provenance, per-method contract fields/runtime, the canonical decision, autocorrelation, cepstrum and patch heatmap. Existing FFT, peaks, radial/angular, DCT, wavelet and residual images remain. Every image is directly derived from a reported measurement; unavailable methods receive labelled placeholders in saved debug bundles.

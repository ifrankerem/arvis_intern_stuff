# Calibration Protocol

## Non-negotiable split

Create three immutable partitions before choosing thresholds:

1. **Calibration/development:** estimate bona-fide distributions, feature direction and operating points.
2. **Validation:** select method/family weights, wavelet/filter variants and runtime mode.
3. **Final test:** one locked evaluation; never change parameters from its labels.

Split by identity, physical camera, display/printer PAI and capture session where possible. Frames from one video must never span partitions. Record a cryptographic manifest hash and configuration version.

## Capture matrix

Calibration bona fide must cover people/skin appearances, glasses/facial hair, indoor/outdoor and low/high illumination, distance, pose, stillness/motion and each supported camera/codec path. Attack development must cover LCD/OLED phone, tablet, laptop/monitor, brightness/refresh/angle/distance/fullscreen/windowed/compressed playback, matte/glossy/office/photo print, resolution, flat/curved/cutout/partial presentation.

Each row records identity pseudonym, session, camera make/model/device ID, driver/API, nominal FPS, measured timestamps, resolution, exposure/focus if available, codec/transcode path, PAI make/model/material, display refresh/brightness when known, print process/paper, distance/angle/lighting, attack species and split.

## Feature calibration

For each raw feature and exact `(device profile, runtime mode, ROI, transform, resolution)`:

1. Reject unsupported/invalid samples using the method’s support rules.
2. Inspect bona-fide and attack distributions by condition; never pool until heterogeneity is understood.
3. Choose suspicion direction: high, low or outside a valid interval.
4. Estimate bona-fide median and MAD; also retain 0.5/1/5/50/95/99/99.5 percentiles and sample count.
5. Winsorize only with declared calibration percentiles.
6. Map robust z or interval distance through a monotonic piecewise/smoothstep function.
7. Set reliability from measurable quality/cadence/coverage, not from the risk score itself.
8. Save parameters to a human-readable versioned file such as `precontrol_calibration.example.json`.

A near-zero MAD requires investigation. The feature may be quantized/degenerate; select a documented measurement-resolution floor or mark it unsupported. Do not let `ε` create extreme artificial z-scores.

## Threshold selection

Select deployment thresholds from validation ROC/DET curves at a documented BPCER/APCER policy, separately by attack species and device profile. Record the exact objective, e.g. maximum BPCER 1% subject to minimum replay detection. Do not optimize overall accuracy.

`LIVE` additionally requires calibrated low risk, minimum family coverage and reliability. Abstention is not folded into BPCER/APCER; report it separately. Unsupported captures must trigger retry/fallback policy.

## Family weight selection without learned fusion

Weights remain manually selected, but evidence must be quantitative:

- pairwise Spearman/Pearson score and feature correlations;
- attack-condition error overlap;
- leave-one-family-out APCER/BPCER change;
- latency and memory contribution;
- robustness across cameras/PAIs.

Reduce weight or remove methods with high correlation, no independent error correction or unacceptable cost. Weight changes are configuration review decisions, not classifier training.

## Device-specific and default profiles

- `experimental_default`: transparent development mappings; may emit `SUSPICIOUS`, never `LIVE`.
- `device_calibration`: exact camera/app/codec profile with feature robust statistics.
- `deployment_thresholds`: approved final operating points, policy version and validity dates.

If device matching fails, fall back to experimental/unsupported rather than choosing the nearest device silently.

## Recalibration triggers

Camera/driver/ISP update, resolution/FPS change, new transport/transcode, ROI/alignment change, FFT/window/filter change, score schema change, new PAI class, or statistically significant drift invalidates affected profile sections. Schema version 3 deliberately invalidates old temporal history.


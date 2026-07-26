# Controlled Capture and Benchmark Protocol

## Objective

Measure a locked deterministic PreControl by PAI species, camera and capture condition. The protocol follows ISO/IEC 30107-3 reporting principles and uses NISTIR 8491 as evidence that aggregate results hide strong PAI variation. No result from an external paper is claimed for this application.

## Capture design

Use multiple independent identities, sessions and physical cameras. Randomize presentation order. Preserve original video, encoded metadata, arrival timestamps and a manifest; do not benchmark from UI screenshots or regenerated social-media copies unless that is an explicit network-transcode condition.

### Bona fide

- indoor daylight/tungsten/LED and outdoor shade/sun;
- low/nominal/high brightness and exposure transition;
- near/nominal/far distance;
- frontal and allowed pose envelope;
- stillness and ordinary head/expression motion;
- glasses/no glasses, facial hair and varied skin appearance;
- each camera, resolution, FPS and network path.

### Replay

- LCD and OLED phones, tablet, laptop and monitor;
- several brightness levels and configurable refresh rates;
- distance, camera/display angle and focus;
- fullscreen/windowed, visible/hidden device boundary;
- original/high-compression/transcoded video;
- static photo replay and genuine/deepfake video replay as separate species.

### Print

- matte/glossy, office/photo printer, low/high resolution;
- flat, curved, eye cutout and partial print;
- distance/angle and illumination;
- paper edge visible/hidden as separate conditions.

## Manifest and execution

Use `benchmarks/manifest_example.csv` with absolute path, `bona_fide|attack`, attack species, unique capture ID and split. Run:

```bash
python3 benchmarks/evaluate_manifest.py manifest.csv \
  --split test --threshold <locked-validation-threshold> \
  --output benchmark_results/test_report.json
```

The threshold option is offline evaluation only. It does not bypass production calibration/abstention. Reset temporal/module state between captures. Never average frames from different captures as if they were independent trials.

## Required metrics

- APCER by PAI species and aggregate;
- BPCER by bona-fide condition and aggregate;
- ACER only as `(APCER+BPCER)/2`, never alone;
- ROC and EER where appropriate;
- TPR at selected FPR values;
- abstention rate and unsupported rate;
- confusion by attack species;
- method/family/total latency, p50/p95;
- peak RSS and, for edge targets, platform-specific memory/energy;
- decision reliability/coverage distributions.

APCER denominator is supported attack presentations for the score operating point; all excluded/non-response counts remain separately reported. System-level policy should also report how retries/fallbacks resolve abstentions.

## Quality control

Before scoring, verify identity/session/PAI split, file hashes, frame count, timestamps monotonicity, actual decoded dimensions/FPS, label audit and no calibration/test leakage. Flag captures with auto-exposure oscillation, dropped frames, frame duplication, decode failure or face/guide miss; do not delete them without a declared rule.

## External datasets

Replay-Attack, OULU-NPU and UTKPAD can test comparability and domain shift subject to their licenses/protocols. They do not replace controlled device captures, especially for flicker, sensor provenance, raw timing and printer material variables not retained in released media.

## Acceptance gates

Deployment requires predeclared APCER/BPCER policy, bounded unsupported rate, stable per-condition latency, no final-test tuning, and written limitations for unseen PAIs. A good overall ACER cannot compensate for catastrophic failure on one important attack species.


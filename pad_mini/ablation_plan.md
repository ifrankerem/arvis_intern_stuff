# Ablation and Correlation Plan

## Locked experiment matrix

Run every row on the same validation and final-test manifests/configuration except for the declared ablation:

1. Original six methods and original two-group fusion.
2. Original six plus autocorrelation/cepstrum.
3. Original six plus each future P1 method individually.
4. Each evidence family alone.
5. All implemented families.
6. Full system minus each family in turn.
7. FAST, BALANCED and RESEARCH runtime profiles.
8. Hann vs Hamming vs Tukey vs no window.
9. Wavelet Haar/db2/db4/sym4/bior variants and one/two/three levels.
10. Residual filters individually and as the selected bank.
11. Raw ROI vs resized/aligned ROI only where mathematically permissible.

The final test is run once after selection; it is not the ablation-selection set.

## Measurements

For each raw feature, method score, family score and attack score compute:

- Pearson correlation for approximately linear behavior;
- Spearman rank correlation for monotonic dependence;
- conditional correlation by camera, lighting and PAI species;
- mutual trigger overlap and Jaccard index;
- false-positive and false-negative capture-ID overlap;
- APCER/BPCER/ACER, ROC/EER, TPR at selected FPR;
- abstention/unsupported rate;
- mean/p50/p95 runtime and peak memory.

Do not correlate every video frame as an independent sample. Aggregate to capture-level scores first or use a hierarchical analysis that respects identity/session/capture.

## Independence decision

For method `m`, record:

`marginal_value = metric(full) - metric(full_without_m)`

for each attack species and operating point, alongside added latency. A method is reduced/removed when it has high same-family correlation, near-identical errors, negligible leave-one-out benefit and material runtime. A method with unique value on one important PAI can remain even if aggregate ACER changes little.

## Required plots/tables

- feature and score correlation heatmaps ordered by family;
- attack-species APCER table per ablation;
- bona-fide-condition BPCER table;
- error-overlap matrix using capture IDs;
- latency contribution waterfall;
- score/reliability/coverage distributions;
- ROC/DET per attack family and runtime mode.

## Specific hypotheses

- Autocorrelation/cepstrum should improve weak periodic-pattern stability but remain correlated with FFT/moiré; its full independent Stage-B weight should fail the design review.
- DCT, wavelet and residual will correlate under blur/resize/compression; only independent error correction justifies retaining all three.
- No-window FFT should show more crop-boundary leakage; Hann/Hamming/Tukey trade main-lobe width and leakage.
- Device/PAI-specific gains must not be inferred from synthetic gratings alone.
- rPPG, when added, should improve positive physiological evidence in high-SQI windows but must increase abstention rather than false attack claims in poor windows.

## Removal rule

Remove or reduce a method if, across the locked validation captures, it provides no material per-PAI operating-point benefit, does not correct unique errors, or degrades latency/unsupported rate beyond the deployment budget. Record the removal; do not hide unsuccessful ablations.


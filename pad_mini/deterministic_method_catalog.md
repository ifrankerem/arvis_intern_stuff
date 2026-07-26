# Deterministic Method Catalog

This catalog separates a physical observation from a score. Every score is a bounded engineering mapping of raw measurements; it is not a probability of attack. All methods are model-free and use no trained decision component.

## Implemented core methods

### 1. Global FFT — frequency family

**Purpose and attacks.** Describe broadband spectral shape and concentrated periodic energy associated with display/print recapture. It supports replay, print, and general recapture evidence but is not attack-specific alone.

**Input.** Current implementation uses a 256×256 grayscale detected-face crop, zero mean/unit standard deviation, and a separable 2-D Hann window. Hann is the default; Hamming, Tukey (`α=0.25`), and no window are supported for ablation. DC and near-DC are excluded at normalized radius `r<0.02`.

**Measurements.** For `P(u,v)=|FFT2(wI)|²` and annulus `A`, energy ratio is `Σ_A P / Σ_analysis P`. It reports low/middle/high ratios; centroid `Σ rP/ΣP`; normalized Shannon entropy; flatness `exp(mean(log(P+ε)))/(mean(P)+ε)`; 85% roll-off radius; Pearson kurtosis; high/low energy; a Theil–Sen-style median pairwise log-log radial slope and fit MAD; dominant non-DC peak ratio; and dominant conjugate-pair energy/amplitude symmetry.

**Assumptions/failures.** Window choice, resize interpolation, focus, sharpening, compression, makeup, hair, and natural texture alter the result. Conjugate symmetry is guaranteed for real images, so symmetry alone is not attack evidence; only concentrated pair energy is informative. Current color/gradient/residual-plane comparisons remain P1.

**Correlation/complexity.** Highly correlated with moiré, radial/angular, autocorrelation, and cepstrum. `O(N log N)` for the shared FFT; scalar features are `O(N)`.

### 2. Moiré and periodic spectral peaks — frequency family

**Purpose.** Detect narrow periodic peaks consistent with screen pixel/subpixel lattices, sampling aliases, or print halftone structure.

**Formula.** On `L=log(1+P)`, estimate a smooth background `B=Gaussian(L,σ=4)` and residual `R=L-B`. Robust prominence is `(R-median(R))/(1.4826·MAD(R)+ε)`. Candidate peaks must be local maxima inside `0.14≤r≤0.82`, pass prominence, local-contrast, and energy-share gates, and survive non-maximum suppression. Inversion-paired peaks and an axial orientation histogram support, but do not independently prove, periodicity.

**Current strengths.** A peak is compared with its local spectral neighborhood; one energetic frequency does not automatically trigger a frame. The implementation now also applies single-scale local patch FFT inside the detected face ROI. At least two patches must contain strong symmetric peaks, and their dominant orientation must agree before local evidence raises the score. A computed spatial heatmap and every patch measurement are exported.

**Current gap.** Multiscale patch voting, harmonic/lattice consistency, and face-versus-expanded/background comparison remain P1. Striped clothes, hair, blinds, brick, eyeglass edges, and natural patterns are explicit false-positive classes, so local evidence remains temporal and supporting rather than an unconditional replay verdict.

**Complexity.** One shared global FFT plus `K` small patch transforms,
`O(N log N + K·p² log p)`, and linear filtering/maxima operations. The default
detected ROI typically yields about 20–30 non-overlapping patches.

### 3. Radial and angular spectrum — frequency family

**Purpose.** Separate broadband radial decay from directional anisotropy.

**Formula.** Radial bins report mean/median/energy density, normalized profile, entropy, dominant-radius concentration, and a log-log slope. Angular bins cover axial orientation `θ∈[0,π)`. Correct circular statistics double the angle: resultant `R=|Σp_k exp(j2θ_k)|/Σp_k`, axial mean `0.5·arg(Σp_k exp(j2θ_k))`, circular variance `1-R`. Thus the first and last bins are neighbors, not independent endpoints.

**Gap.** Current fit is ordinary least squares, one ROI and one scale. P1 adds robust slope, radial peak confidence, region comparison (guide/forehead/cheeks/expanded/background), and temporal orientation stability.

**Complexity.** `O(N+B²)` after shared FFT, with `B` radial/angular bins.

### 4. DCT, block, and decoded-compression evidence — compression/recapture family

**Purpose.** Measure block-grid discontinuity, coefficient sparsity, and local recompression inconsistency.

**Formula.** Each 8×8 block uses the orthonormal DCT. It records low/mid/high AC energy, AC/DC ratio, exact/near-zero coefficient ratio, magnitude entropy/kurtosis, block-to-block variation, horizontal/vertical boundary differences, phase-periodic blockiness, and robust local descriptor inconsistency.

**Critical provenance rule.** The current input is a decoded and resized frame. It does not contain JPEG quantization tables or encoded coefficients. Therefore the module does **not** claim a JPEG quality factor or double JPEG. Such decisions require encoded bytes and grid/provenance tracking. Ordinary camera/video/network compression is a confounder, not an attack label.

**Gap/cost.** P1 should scan all eight grid phases on the unresized raw ROI and ingest encoded metadata when available. Complexity is `O(N)`.

### 5. Wavelet decomposition — spatial texture family

**Purpose.** Describe scale- and direction-dependent texture loss or excess.

**Formula.** Two-level `db2` DWT with periodization reports, for LL/LH/HL/HH, energy `Σc²`, normalized energy, entropy of normalized coefficient magnitudes, kurtosis, near-zero sparsity, mean absolute coefficient, MAD, directional imbalance, and local robust inconsistency.

**Selection rule.** `db2` is an experimental default, not a scientifically selected winner. Haar, Daubechies, Symlets, and biorthogonal families must be compared on development data by attack species, correlation, and latency. The final test set must not select the wavelet.

**Failures/correlation.** Blur, resampling, compression and sharpening dominate many subbands. Evidence overlaps DCT and residual analysis. Complexity is `O(N)` per level.

### 6. High-pass and residual analysis — spatial texture family

**Purpose.** Quantify local high-frequency residual structure without calling it sensor PRNU.

**Formula.** Current residual bank contains Gaussian subtraction `I-Gσ*I`, Laplacian, and Sobel gradient. Weighted facial-interior statistics include signed variance/MAD/RMS, entropy, kurtosis, positive/negative balance, gradient energy/density/direction balance, and patch-wise robust inconsistency.

**Failure rule.** The analyzed crop is resized; consequently its residual cannot support reference-camera PRNU attribution. Low light, denoising, sharpening, codec ringing, and interpolation all change it.

**Gap.** Median, bilateral, DoG, local-mean, Wiener-like residuals, residual autocorrelation/spectrum, and channel consistency require ablation before deployment. Complexity is `O(KN)` for `K` residual filters.

### 7. Autocorrelation and real cepstrum — frequency family (P0 added)

**Purpose.** Corroborate repeated display/print structure in lag/quefrency domains and spatial patches.

**Input.** Unaligned raw guide crop, grayscale, optional downscale only above 384 px, Gaussian high-pass residual (`σ=2`), local Hann windows. It refuses sides below 96 px, severe clipping, or negligible residual contrast.

**Formula.** Wiener–Khinchin autocorrelation is `R=FFTshift(IFFT2(|FFT2(wI)|²))`, normalized by its center. The real cepstrum map is `C=|FFTshift(IFFT2(log(1+|FFT2(wI)|)))|`. Center lags are masked; off-center local maxima use median/MAD prominence. Global lag/cepstral periods are compared with `exp(-|p_R-p_C|/(τ·max(p_R,p_C)))`. Two patch scales vote spatially. Score components use declared monotonic smoothstep mappings and are fused only inside the frequency family.

**Outputs.** Off-center autocorrelation ratio/z/coordinate/period, cepstral ratio/z/coordinate/period, cross-domain agreement, patch vote, patch period CV, lattice regularity, raw patch measurements, autocorrelation map, cepstrum map, and patch heatmap.

**Failures.** Repeated bona-fide textures can trigger it; weak or out-of-band lattices can disappear after optics/resize; autocorrelation symmetry is inherent. It is supporting evidence only. Global plus patch work is approximately `O(KN log N)`.

## Candidate methods not yet decision-enabled

| Candidate | Principle | Inputs | Mode | Attack value | Cost | Main risk/overlap | Rank |
|---|---|---|---|---|---|---|---|
| Temporal flicker/PWM/rolling shutter | Detrend ROI Y/chroma, Welch PSD, narrowband SNR, row-phase slope | Accurate timestamps, FPS/cadence, stable ROI, ≥1–3 s video | Temporal | Replay screen | Medium | Aliasing; illumination/auto-exposure; unsupported on irregular cadence | P1 |
| Deterministic chromaticity | `r=R/(R+G+B)`, `g=G/(R+G+B)`, YCbCr/Lab/opponent moments, clipping and correlation | Color raw/skin ROIs | Single/temporal | Replay/print | Low | Camera white balance and skin/lighting dependence; learned CMA excluded | P1 |
| Specularity/material | Highlight mask/components, saturation, diffuse/specular ratios, motion; optional flash difference | Color ROI; preferably controlled illumination | Single/temporal | Glossy print/screen/mask | Low–medium | Harsh light, glasses, oily skin; passive single-frame reliability often low | P1 |
| Interpretable microtexture | Uniform LBP entropy/ratio, LPQ phase histogram, Gabor energy ratios, HOG concentration | Stable regional crops | Single/temporal | Print/replay | Medium | Descriptor alone is not liveness; calibrated ranges required; overlaps wavelet/residual | P1 |
| Homography/parallax/non-rigid flow | LK/Farnebäck flow, RANSAC homography residual, affine residual, region-relative parallax | Tracked raw ROIs, timestamps, motion | Temporal | Planar print/screen | Medium–high | Still genuine users; flexible/curved prints; replayed facial motion | P1 |
| POS/CHROM rPPG | Classical color projection, band-pass PSD, ROI coherence/phase/SQI | ≥5–10 s accurate color video and skin ROIs | Temporal | Positive physiological support | Medium | Pulse absence is non-evidence; replay can preserve pulse; motion/illumination | P1 |
| Halftone/edge profile | Dot periodicity/angle; ESF→LSF width/overshoot/MTF proxies | High-resolution raw print edges | Single | Print | Medium | Resolution and content dependence; learned edge dictionary excluded | P1 |
| Blind noise/CFA consistency | Raw residual stationarity, row/column periodicity, CFA/demosaic consistency | Unresampled color raw ROI | Single/temporal | Recapture/provenance | High | ISP destroys cues; blind result is not source identity | P1/P2 |
| Reference PRNU match | Correlate trusted camera fingerprint with frame residual | Many trusted same-camera reference frames | Temporal/calibrated | Injection/source mismatch | High | Impossible without enrolled physical-camera fingerprint | P2 |
| Active focus/exposure/flash | Focus sweep response, defocus, exposure/flash challenge, controlled parallax | Camera-control API/hardware | Active | Planar/material attacks | Medium | Hardware/privacy/UX; separate layer only | P2 |
| Blink or “little motion” | Event/motion count | Video | Weak generic | Low | Replays blink; genuine users remain still | Reject as liveness decision |
| Learned color/texture/frequency adapters | CNN/ViT/SVM/classifier | Images/training data | Any | Potentially broad | Varies | Violates deterministic-stage constraint | Reject |

## ROI routing

| ROI | Intended methods | Restrictions |
|---|---|---|
| Aligned face | Stable color/microtexture comparisons | Never for PRNU, encoded DCT grid, or raw sensor evidence |
| Raw face/guide | FFT, DCT phase scan, residual, autocorrelation/cepstrum | Current guide is not a detected face |
| Expanded face | Moiré, print/screen boundary context, material comparison | Do not use visible device borders as automatic attack proof |
| Forehead/cheeks | Future POS/CHROM, chromaticity, specularity | Current regions are relative to a landmark-located face box but are not pixel-accurate anatomical/skin masks |
| Nose/eyes | Geometry/specularity support | Eyeglasses and highlight confounds |
| Background ring | Face/background spectrum/noise/motion comparison | Ring mask must be honored; zeroing it before FFT creates false edges |
| Full frame | Row-wise flicker and capture provenance | Never substitute full-frame background content for face evidence |

# Primary-Source Literature Review

## Standards and evaluation evidence

[ISO/IEC 30107-3:2023](https://www.iso.org/standard/79520.html) defines principles and methods for PAD testing/reporting; it does not prescribe this implementation or guarantee system security. APCER and BPCER must be reported by presentation-attack instrument/species and operating point, with non-responses/abstentions visible.

[NISTIR 8491, Face Analysis Technology Evaluation Part 10](https://www.nist.gov/publications/face-analysis-technology-evaluation-fate-part-10-performance-passive-software-based) evaluates passive software PAD on conventional 2-D imagery. Its central deployment lesson is heterogeneity: performance varies by algorithm, use case and PAI, and no paper result may be transferred to this application. Its finding that simple score fusion can help does not justify correlated feature over-counting.

Official benchmark resources used in the proposed protocol include the [Idiap Replay-Attack database](https://www.idiap.ch/software/bob/docs/bob/bob.db.replay/v3.0.9/) and the [OULU-NPU protocol/results site](https://sites.google.com/site/oulunpudatabase/results). The 2024 [UTKPAD official dataset page](https://www.idiap.ch/en/scientific-research/data/utkpad/index_html?set_language=en) adds a recent, device-diverse replay resource, but its construction and single-frame derivation make it complementary rather than a replacement for real video captures. [LivDet-Face 2026](https://face2026.livdet.org/) is a relevant current independent competition/protocol source; results available after this audit date must be evaluated before use.

## Frequency, moiré, recapture, and cepstrum

Patel, Han, Jain and Ott, [“Live Face Video vs. Spoof Face Video: Use of Moiré Patterns to Detect Replay Video Attacks,” ICB 2015](https://biometrics.cse.msu.edu/Publications/Face/PatelHanJainOtt_LivevsSpoofFaceVideo_ICB15.pdf), motivates display/camera sampling aliases and examines multiple color/region representations. Its learned classification components are not reproduced; only the physical periodicity observation informs FFT peak measurements.

Thongkamwitoon, Muammar and Dragotti, [“An Image Recapture Detection Algorithm Based on Learning Dictionaries of Edge Profiles,” TIFS 2015](https://doi.org/10.1109/TIFS.2015.2392566), describes recapture artifacts from display grids, CFA sampling, edge blur, clipping and color. The K-SVD/dictionary/SVM stages are forbidden here. Deterministic ESF/LSF widths, overshoot and residual periodicity are only adapted candidate measurements.

Sequeira et al., “Presentation Attack Detection Algorithm for Face and Iris Biometrics,” EUSIPCO 2014, Section 2.1, provides a 2-D cepstrum PAD construction. Its learned BSIF/SVM decision is excluded. The P0 module instead computes a standard real cepstrum and reports transparent off-center peaks, period agreement and patch votes.

Popescu and Farid, [“Exposing Digital Forgeries by Detecting Traces of Re-sampling”](https://farid.berkeley.edu/downloads/publications/sp05.pdf), shows that interpolation creates periodic statistical correlations. This supports investigating raw-ROI resampling traces, but its estimator is not yet in the runtime path and resizing by this application must be recorded as a confounder.

## Texture, color, reflection, and print

Määttä, Hadid and Pietikäinen, [“Face Spoofing Detection From Single Images Using Micro-Texture Analysis,” IJCB 2011](https://doi.org/10.1109/IJCB.2011.6117510), uses multiscale LBP followed by a trained classifier. LBP’s deterministic histogram/uniform-pattern statistics are valid P1 candidates; its classifier and reported accuracy are not imported.

Boulkenafet, Komulainen and Hadid, [“Face Anti-Spoofing Based on Color Texture Analysis,” TIFS 2016](https://arxiv.org/abs/1511.06316), motivates RGB/HSV/YCbCr color distortions from display/print reproduction. The LBP/SVM decision is excluded. Region-wise deterministic chromaticity, clipping, entropy and channel-correlation measures are P1.

Chen et al., [“CMA: A Chromaticity Map Adapter for Robust Detection of Screen-Recapture Document Images,” CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_CMA_A_Chromaticity_Map_Adapter_for_Robust_Detection_of_Screen-Recapture_CVPR_2024_paper.html), provides recent evidence that chromaticity can retain recapture information after degradations. It targets documents and uses a learned transformer adapter. Only the deterministic chromaticity-map motivation transfers; neither architecture nor paper performance does.

Shafer, [“Using Color to Separate Reflection Components,” 1985](https://doi.org/10.1002/col.5080100409), supplies the dichromatic diffuse/specular model. Passive highlight statistics are conservative P1 evidence because illumination and glasses violate simple assumptions. Flash-difference approaches such as Ebihara et al., [SpecDiff](https://arxiv.org/abs/1907.12400), belong to the optional active layer and their trained classifier is excluded.

Thongkamwitoon et al. support edge-profile cues; Hu et al., [TIFS 2022 halftone distortion work](https://doi.org/10.1109/TIFS.2022.3192999), supports periodic cell/displacement measurements. Document/printer results do not transfer automatically to face prints, especially below the optical resolution needed to resolve dots.

## Temporal display and motion

Hajj-Ahmad et al., “Flicker Forensics for Camcorder Piracy,” TIFS 2016, models interaction among display backlight/refresh behavior and rolling-shutter row timing. It supports detrended temporal PSD and row-phase analysis without assuming a fixed 50/60/90/120 Hz. Camera sampling can alias or make flicker appear spatially static; timestamp/FPS reliability is therefore a hard support condition.

Kollreider et al., [“Non-intrusive liveness detection by face images,” Image and Vision Computing 2009](https://doi.org/10.1016/j.imavis.2007.05.004), and classical optical-flow PAD work motivate comparing central and outer facial motion. A RANSAC homography/affine residual is an interpretable P1 adaptation; no “little motion means attack” rule is accepted.

Bharadwaj et al., [“Computationally Efficient Face Spoofing Detection With Motion Magnification,” CVPR Workshops 2013](https://openaccess.thecvf.com/content_cvpr_workshops_2013/W02/html/Bharadwaj_Computationally_Efficient_Face_2013_CVPR_paper.html), supports subtle temporal motion investigation, but any learned decision stage is out of scope. Classical LK/Farnebäck, projective residual and parallax remain supporting evidence.

## Physiological evidence

de Haan and Jeanne, [“Robust Pulse Rate From Chrominance-Based rPPG,” TBME 2013](https://pubmed.ncbi.nlm.nih.gov/23744659/), defines CHROM color projections. Wang et al., [“Algorithmic Principles of Remote-PPG,” TBME 2017](https://doi.org/10.1109/TBME.2016.2609282), derives POS and analyzes projection principles. These are model-free, but reliable use needs timestamped multi-ROI video, detrending/band-pass filtering, PSD quality, motion/illumination checks, and ROI coherence.

Recent sources reinforce quality gating: [“Optimal signal quality index for remote photoplethysmogram sensing,” 2024](https://www.nature.com/articles/s44328-024-00002-1) studies an explicit SQI, and [“The role of face regions in remote photoplethysmography,” npj Digital Medicine 2025](https://www.nature.com/articles/s41746-025-01814-9.pdf) recommends region-aware/SQI analysis. Absence of a pulse remains non-proof, and a replay can preserve pulse content.

## Sensor provenance and compression

Lukáš, Fridrich and Goljan, [“Digital Camera Identification From Sensor Pattern Noise,” TIFS 2006](https://ws.binghamton.edu/fridrich/publications.html), establishes PRNU as a camera fingerprint estimated from multiple images. Reference-camera matching is impossible here without trusted enrollment frames from the same physical sensor. Resized guide crops cannot support that claim.

JPEG’s 8×8 DCT/quantization structure is standardized, while double-compression research relies on encoded quantized-coefficient histograms and grid alignment. Pevný and Fridrich, [“Detection of Double-Compression in JPEG Images,” TIFS 2008](https://doi.org/10.1109/TIFS.2008.922456), includes a trained SVM and operates on JPEG coefficient evidence. The current decoded-video module therefore reports block/DCT measurements but explicitly marks double-JPEG inference unsupported.

## 2024–2026 scope decision

The recent PAD literature is predominantly learned. For example, 2026 region-relationship PAD uses a pretrained TimeSformer ([IEEE TDSC 2026](https://doi.org/10.1109/TDSC.2026.3664399)); it is rejected for PreControl despite its region-comparison motivation. Likewise, current VLM/CNN/transformer PAD papers are useful context but not implementation sources. The recent sources actually adopted here concern evaluation/datasets, deterministic chromaticity motivation, and rPPG signal quality—not learned decision machinery.


# Final Limitations

1. The default GUI path uses MediaPipe landmarks only to localize and stabilize the face ROI. This is learned infrastructure, not liveness evidence. The ROI is not geometrically aligned, and named subregions remain approximate box-relative regions rather than verified skin/anatomical masks.
2. Built-in thresholds and attack-family weights are experimental. No repository dataset supports a calibrated `LIVE` claim; the implementation therefore withholds `LIVE` without compatible deployment calibration.
3. Spatial frequency evidence can be triggered by bona-fide repeated texture, clothing, hair, blinds, brick, glasses, sensor/codec artifacts, sharpening and resizing. Frequency methods are correlated and fused as one family.
4. The application sees decoded frames. It cannot infer JPEG quantization tables, reliable double JPEG, original codec history, camera exposure time or raw sensor samples unless upstream capture provides them.
5. DCT/block, wavelet and residual methods all use a resized grayscale crop and remain correlated under blur/compression/interpolation. Sensor/PRNU claims are prohibited on this representation.
6. Autocorrelation/cepstrum improves periodicity corroboration but does not distinguish display pixels from natural lattices by itself. Its thresholds are synthetic-development values.
7. Temporal flicker/PWM/rolling-shutter, homography/parallax, POS/CHROM rPPG, chromaticity/material, halftone/edge and sensor provenance are researched P1/P2 items, not current decision evidence.
8. Decoder-arrival timestamps have at most medium declared reliability and can contain buffering jitter. Accurate exposure/row timing may require camera APIs beyond OpenCV.
9. rPPG absence, stillness and blinking cannot be definitive attack/live evidence. Replays may preserve pulse and natural behavior.
10. A flat/curved/high-quality print or high-resolution/high-refresh display can suppress the targeted artifacts; masks, 3-D heads, virtual-camera injection and direct digital/deepfake injection are not comprehensively covered.
11. Debug heatmaps are computed measurements, not localization ground truth. Their presence does not establish causality.
12. Synthetic tests verify mathematical direction and support behavior, not field accuracy. APCER/BPCER/ACER/ROC/EER remain unreported until controlled captures are collected under the locked protocol.
13. Runtime was exercised on the development machine only. Mobile/edge latency, memory, thermal behavior and artifact loss after downsampling remain to be measured.
14. Overall uncertainty interval is an explainability bound derived from reliability, not a frequentist confidence interval or calibrated probability.
15. PAD is one security layer. It must be combined with trusted capture, injection defenses, identity verification, retry/rate policy and monitoring; this PreControl alone cannot prove authenticity.

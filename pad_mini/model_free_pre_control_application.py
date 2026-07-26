"""Karar katmani model-free olan deterministik PreControl uygulamasi."""

from datetime import datetime
import csv
import json
import time

import cv2
import numpy as np

import config
from data_models import FaceBox, FrameProcessingResult
from model_free_analysis import (
    ModelFreeAnalysisResult,
    ModelFreePreControlContextBuilder,
)
from global_fft_pre_control import GlobalFFTPreController
from moire_pre_control import MoirePeriodicPatternPreController
from radial_angular_pre_control import (
    RadialAngularSpectrumPreController,
)
from dct_block_pre_control import DCTBlockAnalysisPreController
from wavelet_pre_control import WaveletAnalysisPreController
from high_pass_residual_pre_control import HighPassResidualPreController
from periodicity_pre_control import PeriodicityPreController
from mathematical_fusion import MathematicalFusionController
from precontrol_decision import PreControlDecisionBuilder
from face_roi_tracker import StableFaceROITracker


class ModelFreePreControlApplication:
    """Tespit edilen ROI'de model-free matematiksel analizleri calistirir.

    MediaPipe etkinse yalnizca ROI bulma/takip altyapisi olarak kullanilir;
    liveness veya presentation-attack kararina hicbir model cikisi verilmez.
    """

    def __init__(
        self,
        face_detector=None,
        enable_face_detection=None,
        runtime_mode=None,
    ):
        if enable_face_detection is None:
            enable_face_detection = config.MODEL_FREE_FACE_DETECTION_ENABLED
        self.face_detection_enabled = bool(enable_face_detection)
        self.runtime_mode = (
            str(runtime_mode).upper()
            if runtime_mode is not None
            else None
        )
        if (
            self.runtime_mode is not None
            and self.runtime_mode not in config.MODEL_FREE_MODE_MODULES
        ):
            raise ValueError(
                "Unsupported model-free runtime mode: " + self.runtime_mode
            )
        self._owns_face_detector = False
        if self.face_detection_enabled and face_detector is None:
            from face_landmarker import FaceLandmarker

            face_detector = FaceLandmarker(
                config.MODEL_PATH,
                config.MAXIMUM_FACE_COUNT,
                config.MIRROR_CAMERA_IMAGE,
                minimum_detection_confidence=(
                    config.MODEL_FREE_FACE_DETECTION_CONFIDENCE
                ),
                minimum_presence_confidence=(
                    config.MODEL_FREE_FACE_PRESENCE_CONFIDENCE
                ),
                minimum_tracking_confidence=(
                    config.MODEL_FREE_FACE_TRACKING_CONFIDENCE
                ),
                roi_only=True,
            )
            self._owns_face_detector = True
        self.face_detector = face_detector
        self.face_roi_tracker = StableFaceROITracker(
            face_detector if self.face_detection_enabled else None,
            smoothing_alpha=config.MODEL_FREE_FACE_BOX_SMOOTHING_ALPHA,
            hold_seconds=config.MODEL_FREE_FACE_TRACK_HOLD_SECONDS,
            horizontal_expansion_ratio=(
                config.MODEL_FREE_FACE_BOX_HORIZONTAL_EXPANSION
            ),
            vertical_expansion_ratio=(
                config.MODEL_FREE_FACE_BOX_VERTICAL_EXPANSION
            ),
            jump_iou_threshold=(
                config.MODEL_FREE_FACE_BOX_JUMP_IOU_THRESHOLD
            ),
            minimum_frame_margin_ratio=(
                config.EXPERIMENTAL_MODEL_FREE_FRAME_EDGE_MARGIN_RATIO
            ),
        )
        self.context_builder = ModelFreePreControlContextBuilder()
        self.fft_pre_controller = GlobalFFTPreController()
        self.moire_pre_controller = MoirePeriodicPatternPreController()
        self.radial_angular_pre_controller = (
            RadialAngularSpectrumPreController()
        )
        self.dct_block_pre_controller = DCTBlockAnalysisPreController()
        self.wavelet_pre_controller = WaveletAnalysisPreController()
        self.residual_pre_controller = HighPassResidualPreController()
        self.periodicity_pre_controller = PeriodicityPreController()
        self.fusion_controller = MathematicalFusionController()
        self.decision_builder = PreControlDecisionBuilder()
        # Every application session starts with a clean fusion history. The
        # controller also resets itself if the schema/configuration signature
        # changes while the process is running.
        self.fusion_controller.reset()
        self.analysis_modules = {
            "fft": ("global_fft", self.fft_pre_controller),
            "moire": ("moire", self.moire_pre_controller),
            "radial_angular": (
                "radial_angular_spectrum",
                self.radial_angular_pre_controller,
            ),
            "periodicity": (
                "periodicity",
                self.periodicity_pre_controller,
            ),
            "dct_block": (
                "dct_block_compression",
                self.dct_block_pre_controller,
            ),
            "wavelet": (
                "wavelet",
                self.wavelet_pre_controller,
            ),
            "residual": (
                "high_pass_residual",
                self.residual_pre_controller,
            ),
        }
        self.latest_pre_control_results = {}
        self.latest_combined_result = None
        self.latest_precontrol_decision = None
        self.latest_context = None
        self.latest_face_image = None
        self.latest_fft_visualization = None
        self.latest_analysis_result = None
        self.latest_face_tracking_result = None
        self.is_closed = False

    def process_frame(self, camera_frame, frame_metadata=None):
        frame_started = time.perf_counter()
        analysis_frame = self.prepare_frame(camera_frame)
        guide_box = (
            FaceBox(0, 0, 0, 0)
            if self.face_detection_enabled
            else self.create_guide_box(analysis_frame)
        )
        self.latest_face_image = None
        self.latest_fft_visualization = None

        capture_metadata = dict(frame_metadata or {})
        capture_metadata["model_free_runtime_mode"] = (
            self.runtime_mode
            or str(config.MODEL_FREE_RUNTIME_MODE).upper()
        )
        frame_timestamp = float(
            capture_metadata.get(
                "acquisition_monotonic_s",
                time.monotonic(),
            )
        )
        tracking_result = None
        analysis_box = guide_box
        quality_override_reason = None
        roi_semantic_basis = "fixed_guide"
        face_roi_edge_contact = False
        if self.face_detection_enabled:
            tracking_result = self.face_roi_tracker.update(
                analysis_frame,
                frame_timestamp,
            )
            if tracking_result.supported:
                analysis_box = tracking_result.box
                roi_semantic_basis = "detected_face_landmarks"
                face_roi_edge_contact = self._box_touches_frame_edge(
                    analysis_box,
                    analysis_frame,
                )
            else:
                quality_override_reason = tracking_result.reason
                roi_semantic_basis = "no_detected_face"
            capture_metadata.update(
                {
                    "face_roi_source": roi_semantic_basis,
                    "face_detection_status": tracking_result.status,
                    "face_detection_count": tracking_result.detection_count,
                    "face_detection_fresh": (
                        tracking_result.fresh_detection
                    ),
                    "face_tracking_reliability": (
                        tracking_result.reliability
                    ),
                    "face_roi_edge_contact": face_roi_edge_contact,
                    "face_detector_used_for_roi_only": True,
                    "face_detector_confidence_used_as_pad_evidence": False,
                }
            )
        else:
            capture_metadata.update(
                {
                    "face_roi_source": roi_semantic_basis,
                    "face_detection_status": "DISABLED",
                    "face_roi_edge_contact": False,
                    "face_detector_used_for_roi_only": False,
                    "face_detector_confidence_used_as_pad_evidence": False,
                }
            )
        self.latest_face_tracking_result = tracking_result
        context = self.context_builder.build(
            frame_timestamp,
            camera_frame,
            analysis_frame,
            analysis_box,
            capture_metadata=capture_metadata,
            quality_override_reason=quality_override_reason,
            roi_semantic_basis=roi_semantic_basis,
            allow_frame_edge_contact=(
                tracking_result is not None
                and tracking_result.supported
            ),
        )
        self.latest_context = context

        self.latest_pre_control_results = {
            result_key: self._run_analysis_module(
                module_key,
                analyzer,
                context,
            )
            for result_key, (module_key, analyzer)
            in self.analysis_modules.items()
        }
        fusion_started = time.perf_counter()
        self.latest_combined_result = self.fusion_controller.analyze(
            self.latest_pre_control_results,
            context,
        )
        self.latest_combined_result.runtime_ms = (
            time.perf_counter() - fusion_started
        ) * 1000.0
        self.latest_combined_result.evidence_family = "fusion"
        self.latest_combined_result.attack_targets = [
            "replay_screen",
            "print_attack",
            "recapture",
        ]
        self.latest_precontrol_decision = self.decision_builder.build(
            self.latest_pre_control_results,
            context,
            self.latest_combined_result,
            runtime_ms=(time.perf_counter() - frame_started) * 1000.0,
        )
        fft_result = self.latest_pre_control_results["fft"]
        moire_result = self.latest_pre_control_results["moire"]

        if context.has_valid_fft:
            self.latest_face_image = (
                context.original_high_resolution_face_crop.copy()
            )
            self.latest_fft_visualization = self.create_fft_visualization(
                context.log_magnitude_visualization
            )

        display_frame = analysis_frame.copy()
        self.draw_guide(
            display_frame,
            guide_box,
            fft_result,
            moire_result,
            tracking_result=tracking_result,
            analysis_box=analysis_box,
        )

        return FrameProcessingResult(
            display_frame,
            self.latest_face_image,
            None,
        )

    def _run_analysis_module(self, module_key, analyzer, context):
        started = time.perf_counter()
        active_mode = self.runtime_mode or str(
            config.MODEL_FREE_RUNTIME_MODE
        ).upper()
        mode_modules = config.MODEL_FREE_MODE_MODULES.get(active_mode, ())
        if (
            not config.MODEL_FREE_MODULE_ENABLED.get(module_key, False)
            or module_key not in mode_modules
        ):
            result = ModelFreeAnalysisResult.unavailable(
                analyzer.MODULE_NAME,
                "module disabled by configuration or runtime mode",
                debug_data={"possible_attack": "none"},
                calibrated=False,
            )
            return self._apply_method_contract(
                module_key,
                result,
                (time.perf_counter() - started) * 1000.0,
            )

        try:
            result = analyzer.analyze(context)
        except Exception as error:
            # Bir opsiyonel matematik modulu diger modulleri veya kamera
            # dongusunu durdurmamalidir.
            print(
                "%s module failed: %s"
                % (analyzer.MODULE_NAME, str(error))
            )
            result = ModelFreeAnalysisResult.unavailable(
                analyzer.MODULE_NAME,
                "module execution failed",
                debug_data={
                    "possible_attack": "none",
                    "exception": repr(error),
                },
                calibrated=False,
            )
        if (
            result.available
            and context.capture_metadata.get("face_detection_status")
            == "HELD"
        ):
            tracking_reliability = float(
                np.clip(
                    context.capture_metadata.get(
                        "face_tracking_reliability",
                        0.0,
                    ),
                    0.0,
                    1.0,
                )
            )
            result.confidence = float(result.confidence or 0.0) * (
                tracking_reliability
            )
            result.warnings.append(
                "Method reliability reduced because the face ROI is "
                "temporarily held rather than freshly detected"
            )
        if (
            result.available
            and context.capture_metadata.get("face_roi_edge_contact", False)
        ):
            result.confidence = float(result.confidence or 0.0) * float(
                config.MODEL_FREE_FACE_EDGE_RELIABILITY_FACTOR
            )
            result.warnings.append(
                "Method reliability reduced because the detected face ROI "
                "touches the frame margin"
            )
        return self._apply_method_contract(
            module_key,
            result,
            (time.perf_counter() - started) * 1000.0,
        )

    def _apply_method_contract(self, module_key, result, runtime_ms):
        contract = {
            "global_fft": (
                "frequency",
                ["replay_screen", "print_attack", "recapture"],
            ),
            "moire": (
                "frequency",
                ["replay_screen", "recapture"],
            ),
            "radial_angular_spectrum": (
                "frequency",
                ["replay_screen", "print_attack", "recapture"],
            ),
            "periodicity": (
                "frequency",
                ["replay_screen", "print_attack", "recapture"],
            ),
            "dct_block_compression": (
                "compression_recapture",
                ["replay_screen", "print_attack", "recapture"],
            ),
            "wavelet": (
                "spatial_texture",
                ["replay_screen", "print_attack", "recapture"],
            ),
            "high_pass_residual": (
                "spatial_texture",
                ["replay_screen", "print_attack", "recapture"],
            ),
        }
        family, targets = contract.get(module_key, ("unassigned", []))
        if result.evidence_family == "unassigned":
            result.evidence_family = family
        if not result.attack_targets:
            result.attack_targets = list(targets)
        result.runtime_ms = max(0.0, float(runtime_ms))
        if not result.human_explanation:
            result.human_explanation = "; ".join(result.evidence) or result.status
        if result.available and result.score is not None and not result.triggered:
            result.triggered = bool(
                result.score
                >= config.MATHEMATICAL_FUSION_CONFIG["module_evidence_score"]
                and result.reliability > 0.0
                and result.status not in ("Analysis Uncertain", "Uncalibrated")
            )
        if result.triggered and not result.reason_codes:
            result.reason_codes = [module_key.upper() + "_RISK_ELEVATED"]
        if not result.available and not result.reason_codes:
            result.reason_codes = [module_key.upper() + "_UNSUPPORTED"]
        return result

    def create_fft_visualization(self, log_magnitude_visualization):
        """Context'teki 0-255 log-magnitude gorselinin ekran kopyasi."""
        visualization = cv2.cvtColor(
            log_magnitude_visualization.copy(),
            cv2.COLOR_GRAY2BGR,
        )
        self.draw_fft_annotations(visualization)
        return visualization

    def draw_fft_annotations(self, visualization):
        height, width = visualization.shape[:2]
        center_x = width // 2
        center_y = height // 2

        cv2.drawMarker(
            visualization,
            (center_x, center_y),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            12,
            1,
        )
        self.draw_label(
            visualization,
            "Low Frequency",
            (center_x + 8, max(14, center_y - 8)),
        )
        self.draw_label(
            visualization,
            "High Frequency",
            (7, height - 9),
        )

    def draw_label(self, image, text, position):
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
        )

    def save_debug_sample(self):
        context = self.latest_context
        if (
            context is None
            or context.original_high_resolution_face_crop is None
            or not self.latest_pre_control_results
            or self.latest_combined_result is None
        ):
            print("Gecerli model-free analiz sonucu yok; kaydedilmedi.")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_directory = (
            config.MODEL_FREE_DEBUG_DIRECTORY
            / ("analysis_" + timestamp)
        ).resolve()
        output_directory.mkdir(parents=True, exist_ok=False)

        visualization_manifest = self._save_unified_visualizations(
            output_directory,
            context,
        )
        for item in visualization_manifest:
            result = self.latest_pre_control_results.get(item["module"])
            if result is not None and item["saved"]:
                result.visualization_paths[item["file"]] = str(
                    output_directory / item["file"]
                )
        score_summary = self.latest_combined_result.score_summary
        report = {
            "schema_version": config.MODEL_FREE_ANALYSIS_SCHEMA_VERSION,
            "fusion_configuration_version": (
                config.MODEL_FREE_FUSION_CONFIGURATION_VERSION
            ),
            "export_timestamp": timestamp,
            "frame_timestamp": context.frame_timestamp,
            "capture_metadata": context.capture_metadata,
            "roi_provenance": {
                name: roi.provenance()
                for name, roi in context.rois.items()
            },
            "visualizations": visualization_manifest,
            "calibration": self.fusion_controller.get_calibration_summary(),
            "quality": {
                "valid": context.face_quality_valid,
                "reason": context.quality_reason,
                "blur": context.blur_value,
                "brightness": context.brightness_value,
                "exposure_valid": context.exposure_valid,
                "pose_alignment_valid": context.pose_alignment_valid,
                "alignment_applied": context.alignment_applied,
                "frame_dimensions": context.frame_dimensions,
                "face_dimensions": context.face_dimensions,
                "analysis_dimensions": context.analysis_dimensions,
                "face_bounding_box": {
                    "x": context.face_bounding_box.x,
                    "y": context.face_bounding_box.y,
                    "width": context.face_bounding_box.width,
                    "height": context.face_bounding_box.height,
                },
            },
            "modules": {
                name: self._result_report_entry(result)
                for name, result in self.latest_pre_control_results.items()
            },
            "fusion": self._result_report_entry(
                self.latest_combined_result
            ),
            "precontrol_decision": (
                self.latest_precontrol_decision.to_dict()
                if self.latest_precontrol_decision is not None
                else None
            ),
            "score_summary": score_summary,
            "group_scores": {
                "fft_family_score": (
                    self.latest_combined_result.raw_features.get(
                        "fft_family_score"
                    )
                ),
                "fft_family_confidence": (
                    self.latest_combined_result.raw_features.get(
                        "fft_family_confidence"
                    )
                ),
                "local_transform_score": (
                    self.latest_combined_result.raw_features.get(
                        "local_transform_score"
                    )
                ),
                "local_transform_confidence": (
                    self.latest_combined_result.raw_features.get(
                        "local_transform_confidence"
                    )
                ),
            },
            "combined_score": {
                **score_summary,
                "confidence": self.latest_combined_result.confidence,
                "status": self.latest_combined_result.status,
                "calibrated": self.latest_combined_result.calibrated,
            },
            "final_evidence": self.latest_combined_result.evidence,
            "final_warnings": self.latest_combined_result.warnings,
            "configuration_thresholds": (
                self._configuration_threshold_snapshot()
            ),
            "limitations": [
                "The output is supporting evidence, not an authenticity verdict.",
                "The correlated FFT modules are fused as one spectral family.",
                "Unavailable modules are excluded rather than assigned a zero score.",
                "Camera, lighting, blur, compression and resizing can affect "
                "every module.",
            ],
        }
        report = self._json_safe(report)
        json_path = output_directory / "model_free_analysis_report.json"
        text_path = output_directory / "model_free_analysis_report.txt"
        try:
            with json_path.open("w", encoding="utf-8") as report_file:
                json.dump(
                    report,
                    report_file,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            text_path.write_text(
                self._human_readable_report(report),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as error:
            print("Birlesik model-free rapor kaydedilemedi: " + str(error))
            return False

        print("Model-free debug klasoru kaydedildi: " + str(output_directory))
        for line in self._score_summary_report_lines(
            report["score_summary"]
        ):
            print(line)
        return all(item["saved"] for item in visualization_manifest)

    def _save_unified_visualizations(self, directory, context):
        outputs = []

        def save(name, image, module):
            if image is None:
                image = self._status_placeholder(module)
            path = directory / name
            saved = bool(cv2.imwrite(str(path), image))
            outputs.append(
                {
                    "module": module,
                    "file": name,
                    "saved": saved,
                }
            )

        save(
            "input_frame_original.png",
            context.original_frame,
            "input",
        )
        save(
            "input_frame_analysis.png",
            context.analysis_frame,
            "input",
        )
        save(
            "input_face_original.png",
            context.original_high_resolution_face_crop,
            "input",
        )
        save("input_face_aligned.png", context.aligned_face_crop, "input")
        standardized = context.standardized_aligned_face_crop
        standardized_visualization = (
            np.clip(standardized, 0.0, 255.0).astype(np.uint8)
            if standardized is not None
            else None
        )
        save(
            "input_face_standardized_luminance.png",
            standardized_visualization,
            "input",
        )
        fft_input = context.standardized_analysis_crop
        fft_input_visualization = (
            cv2.normalize(
                fft_input,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            ).astype(np.uint8)
            if fft_input is not None
            else None
        )
        save(
            "input_fft_windowed.png",
            fft_input_visualization,
            "input",
        )

        fft_result = self.latest_pre_control_results.get("fft")
        fft_available = fft_result is not None and fft_result.available
        fft_bands = None
        if fft_available and context.log_magnitude_visualization is not None:
            fft_bands = self.fft_pre_controller.create_frequency_band_overlay(
                context.log_magnitude_visualization
            )
        save(
            "01_global_fft_spectrum.png",
            self.latest_fft_visualization if fft_available else None,
            "fft",
        )
        save("01_global_fft_bands.png", fft_bands, "fft")
        moire_result = self.latest_pre_control_results.get("moire")
        moire_available = (
            moire_result is not None and moire_result.available
        )
        save(
            "02_moire_periodic_peaks.png",
            (
                self.moire_pre_controller.create_debug_visualization(
                    context.log_magnitude_visualization
                )
                if moire_available
                else None
            ),
            "moire",
        )
        save(
            "02_moire_local_heatmap.png",
            (
                self.moire_pre_controller.get_local_heatmap()
                if moire_available
                else None
            ),
            "moire",
        )

        radial_result = self.latest_pre_control_results.get(
            "radial_angular"
        )
        radial_features = (
            radial_result.raw_features
            if radial_result is not None and radial_result.available
            else None
        )
        if radial_features:
            radial_image = (
                self.radial_angular_pre_controller.create_profile_image(
                    radial_features["radial_normalized_energy_profile"],
                    "Radial normalized energy profile",
                    (0, 200, 255),
                )
            )
            angular_image = (
                self.radial_angular_pre_controller.create_profile_image(
                    radial_features["angular_energy_profile"],
                    "Angular normalized energy profile",
                    (255, 180, 0),
                )
            )
            direction_image = (
                self.radial_angular_pre_controller.create_direction_overlay(
                    context.log_magnitude_visualization,
                    radial_features[
                        "dominant_frequency_angle_degrees"
                    ],
                )
            )
        else:
            radial_image = angular_image = direction_image = None
        save("03_radial_profile.png", radial_image, "radial_angular")
        save("03_angular_profile.png", angular_image, "radial_angular")
        save(
            "03_dominant_directions.png",
            direction_image,
            "radial_angular",
        )

        periodicity_result = self.latest_pre_control_results.get(
            "periodicity"
        )
        periodicity_images = (
            self.periodicity_pre_controller.get_debug_images()
            if periodicity_result is not None and periodicity_result.available
            else {}
        )
        save(
            "03b_autocorrelation_map.png",
            periodicity_images.get("autocorrelation_map"),
            "periodicity",
        )
        save(
            "03b_cepstrum_map.png",
            periodicity_images.get("cepstrum_map"),
            "periodicity",
        )
        save(
            "03b_patch_periodicity_heatmap.png",
            periodicity_images.get("patch_periodicity_heatmap"),
            "periodicity",
        )

        dct_result = self.latest_pre_control_results.get("dct_block")
        dct_images = (
            self.dct_block_pre_controller.get_debug_images() or {}
            if dct_result is not None and dct_result.available
            else {}
        )
        save(
            "04_dct_band_energy_map.png",
            dct_images.get("dct_band_energy_map"),
            "dct_block",
        )
        save(
            "04_dct_block_boundaries.png",
            dct_images.get("block_boundary_visualization"),
            "dct_block",
        )
        save(
            "04_dct_blockiness_heatmap.png",
            dct_images.get("blockiness_heatmap"),
            "dct_block",
        )

        wavelet_result = self.latest_pre_control_results.get("wavelet")
        wavelet_available = (
            wavelet_result is not None and wavelet_result.available
        )
        subbands = (
            self.wavelet_pre_controller.get_debug_subbands() or {}
            if wavelet_available
            else {}
        )
        if subbands:
            for level_number, level_data in subbands.items():
                for band_name, values in level_data.items():
                    visualization = (
                        self.wavelet_pre_controller
                        .create_normalized_subband_visualization(
                            values,
                            is_detail=(band_name != "LL"),
                        )
                    )
                    save(
                        "05_wavelet_level_%d_%s.png"
                        % (level_number, band_name),
                        visualization,
                        "wavelet",
                    )
        else:
            save("05_wavelet_unavailable.png", None, "wavelet")
        save(
            "05_wavelet_anomaly_heatmap.png",
            (
                self.wavelet_pre_controller.get_anomaly_heatmap()
                if wavelet_available
                else None
            ),
            "wavelet",
        )

        residual_result = self.latest_pre_control_results.get("residual")
        residual_images = (
            self.residual_pre_controller.get_debug_images() or {}
            if residual_result is not None and residual_result.available
            else {}
        )
        for key, filename in (
            ("gaussian_residual", "06_gaussian_residual.png"),
            ("laplacian", "06_laplacian.png"),
            ("gradient_magnitude", "06_gradient_magnitude.png"),
            (
                "patch_residual_energy_map",
                "06_patch_residual_energy_map.png",
            ),
        ):
            save(filename, residual_images.get(key), "residual")
        return outputs

    def _status_placeholder(self, module_key):
        result = self.latest_pre_control_results.get(module_key)
        status = result.status if result is not None else "Unavailable"
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.putText(
            image,
            module_key,
            (12, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            status[:31],
            (12, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 190, 255),
            1,
            cv2.LINE_AA,
        )
        return image

    def _result_report_entry(self, result):
        entry = {
            "module_name": result.module_name,
            "available": result.available,
            "status": result.status,
            "calibrated": result.calibrated,
            "raw_score": result.raw_score,
            "stabilized_score": result.stabilized_score,
            "display_score": (
                result.score_summary["display_score"]
                if "score_summary" in result.raw_features
                else result.score
            ),
            "confidence": result.confidence,
            "supported": result.supported,
            "normalized_score": result.normalized_score,
            "reliability": result.reliability,
            "evidence_family": result.evidence_family,
            "attack_targets": result.attack_targets,
            "triggered": result.triggered,
            "reason_codes": result.reason_codes,
            "human_explanation": result.human_explanation,
            "visualization_paths": result.visualization_paths,
            "runtime_ms": result.runtime_ms,
            "raw_features": result.raw_features,
            "evidence": result.evidence,
            "warnings": result.warnings,
            "debug_data": result.debug_data,
        }
        if "score_summary" in result.raw_features:
            entry["score_summary"] = result.score_summary
        return entry

    def _configuration_threshold_snapshot(self):
        prefixes = (
            "MODEL_FREE_",
            "EXPERIMENTAL_",
            "DCT_",
            "WAVELET_",
            "RESIDUAL_",
            "MATHEMATICAL_FUSION_",
        )
        return {
            name: getattr(config, name)
            for name in dir(config)
            if name.startswith(prefixes)
        }

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {
                str(key): self._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return self._json_safe(value.tolist())
        if isinstance(value, np.generic):
            return self._json_safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if hasattr(value, "as_posix"):
            return str(value)
        return value

    def _human_readable_report(self, report):
        lines = [
            "MODEL-FREE PRECONTROL MATHEMATICAL REPORT",
            "=" * 47,
            "Timestamp: " + str(report["export_timestamp"]),
            "Calibration: "
            + (
                "calibrated"
                if report["combined_score"]["calibrated"]
                else "experimental / uncalibrated"
            ),
            "",
            "QUALITY",
            "Blur: %s" % report["quality"]["blur"],
            "Brightness: %s" % report["quality"]["brightness"],
            "Valid: %s" % report["quality"]["valid"],
            "",
            "MODULE SCORES",
        ]
        for name, module in report["modules"].items():
            lines.append(
                "%s: raw=%s stabilized=%s confidence=%s status=%s"
                % (
                    name,
                    module["raw_score"],
                    module["stabilized_score"],
                    module["confidence"],
                    module["status"],
                )
            )
        lines.extend(
            [
                "",
                "GROUP SCORES",
                "FFT family: %s (confidence=%s)"
                % (
                    report["group_scores"]["fft_family_score"],
                    report["group_scores"]["fft_family_confidence"],
                ),
                "Local/transform: %s (confidence=%s)"
                % (
                    report["group_scores"]["local_transform_score"],
                    report["group_scores"]["local_transform_confidence"],
                ),
                "",
                "FINAL",
                *self._score_summary_report_lines(
                    report["score_summary"]
                ),
                "Status: " + report["combined_score"]["status"],
                "Consecutive suspicious frames: %s"
                % report["fusion"]["raw_features"].get(
                    "consecutive_suspicious_frame_count"
                ),
                "Consecutive recovery frames: %s"
                % report["fusion"]["raw_features"].get(
                    "consecutive_recovery_frame_count"
                ),
                "",
                "EVIDENCE",
            ]
        )
        lines.extend("- " + item for item in report["final_evidence"])
        if report["final_warnings"]:
            lines.extend(["", "WARNINGS"])
            lines.extend("- " + item for item in report["final_warnings"])
        lines.extend(
            [
                "",
                "This report is supporting mathematical evidence only; "
                "it is not an authenticity verdict.",
            ]
        )
        return "\n".join(lines) + "\n"

    def _score_summary_report_lines(self, score_summary):
        return [
            "Current frame risk: %s"
            % self._format_report_score(
                score_summary.get("current_frame_score")
            ),
            "Rolling median: %s"
            % self._format_report_score(
                score_summary.get("rolling_median")
            ),
            "Temporal percentile: %s"
            % self._format_report_score(
                score_summary.get("temporal_percentile")
            ),
            "Final temporal decision: %s"
            % self._format_report_score(
                score_summary.get("temporal_decision_score")
            ),
            "Displayed risk: %s"
            % self._format_report_score(
                score_summary.get("display_score")
            ),
        ]

    def _format_report_score(self, value):
        if value is None:
            return "N/A"
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if not np.isfinite(numeric_value):
            return "N/A"
        return "%.2f" % numeric_value

    def save_fft_sample(self):
        """Eski GUI/cagrilar icin tum debug kaydina uyumluluk alias'i."""
        return self.save_debug_sample()

    def _save_radial_angular_debug(self, timestamp, context):
        result = self.latest_pre_control_results.get("radial_angular")
        if result is None or not result.available:
            print("Radial/angular sonucu yok; profil dosyalari kaydedilmedi.")
            return False

        directory = config.FFT_SAMPLE_DIRECTORY
        radial_csv_path = (
            directory / ("radial_profile_" + timestamp + ".csv")
        ).resolve()
        angular_csv_path = (
            directory / ("angular_profile_" + timestamp + ".csv")
        ).resolve()
        radial_image_path = (
            directory / ("radial_profile_" + timestamp + ".png")
        ).resolve()
        angular_image_path = (
            directory / ("angular_profile_" + timestamp + ".png")
        ).resolve()
        directions_path = (
            directory / ("dominant_directions_" + timestamp + ".png")
        ).resolve()
        report_path = (
            directory
            / ("radial_angular_features_" + timestamp + ".json")
        ).resolve()

        features = result.raw_features
        radial_profile_image = (
            self.radial_angular_pre_controller.create_profile_image(
                features["radial_normalized_energy_profile"],
                "Radial normalized energy profile",
                (0, 200, 255),
            )
        )
        angular_profile_image = (
            self.radial_angular_pre_controller.create_profile_image(
                features["angular_energy_profile"],
                "Angular normalized energy profile",
                (255, 180, 0),
            )
        )
        direction_overlay = (
            self.radial_angular_pre_controller.create_direction_overlay(
                context.log_magnitude_visualization,
                features["dominant_frequency_angle_degrees"],
            )
        )
        image_outputs = {
            radial_image_path: radial_profile_image,
            angular_image_path: angular_profile_image,
            directions_path: direction_overlay,
        }
        if not all(
            cv2.imwrite(str(path), image)
            for path, image in image_outputs.items()
        ):
            print("Radial/angular debug gorselleri kaydedilemedi.")
            return False

        try:
            with radial_csv_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as radial_file:
                writer = csv.writer(radial_file)
                writer.writerow(
                    [
                        "radius",
                        "mean_power",
                        "median_power",
                        "normalized_energy",
                        "log_power",
                    ]
                )
                writer.writerows(
                    zip(
                        features["radial_bin_centers"],
                        features["radial_mean_power_profile"],
                        features["radial_median_power_profile"],
                        features["radial_normalized_energy_profile"],
                        features["radial_log_power_profile"],
                    )
                )

            with angular_csv_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as angular_file:
                writer = csv.writer(angular_file)
                writer.writerow(
                    ["frequency_angle_degrees", "normalized_energy"]
                )
                writer.writerows(
                    zip(
                        features["angular_bin_centers_degrees"],
                        features["angular_energy_profile"],
                    )
                )

            report = {
                "frame_timestamp": context.frame_timestamp,
                "module_name": result.module_name,
                "status": result.status,
                "calibrated": result.calibrated,
                "calibration_file": str(
                    config.MODEL_FREE_CALIBRATION_FILE_PATH
                ),
                "raw_score": result.raw_score,
                "stabilized_score": result.stabilized_score,
                "confidence": result.confidence,
                "raw_features": features,
                "evidence": result.evidence,
                "warnings": result.warnings,
                "debug_data": result.debug_data,
                "radial_bin_count": config.EXPERIMENTAL_RADIAL_BIN_COUNT,
                "angular_bin_count": config.EXPERIMENTAL_ANGULAR_BIN_COUNT,
                "analysis_annulus": [
                    config.EXPERIMENTAL_RADIAL_ANGULAR_INNER_RADIUS,
                    config.EXPERIMENTAL_RADIAL_ANGULAR_OUTER_RADIUS,
                ],
            }
            with report_path.open("w", encoding="utf-8") as report_file:
                json.dump(report, report_file, indent=2, ensure_ascii=False)
        except (OSError, TypeError, ValueError) as error:
            print("Radial/angular debug raporu kaydedilemedi: " + str(error))
            return False

        for path in image_outputs:
            print("Radial/angular debug dosyasi kaydedildi: " + str(path))
        print("Radial profil CSV kaydedildi: " + str(radial_csv_path))
        print("Angular profil CSV kaydedildi: " + str(angular_csv_path))
        print("Radial/angular raporu kaydedildi: " + str(report_path))
        return True

    def _save_dct_block_debug(self, timestamp, context):
        result = self.latest_pre_control_results.get("dct_block")
        images = self.dct_block_pre_controller.get_debug_images()
        statistics = (
            self.dct_block_pre_controller
            .get_coefficient_statistics_report()
        )
        if (
            result is None
            or not result.available
            or images is None
            or statistics is None
        ):
            print("DCT/block sonucu yok; debug dosyalari kaydedilmedi.")
            return False

        directory = config.DCT_DEBUG_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        image_paths = {
            (
                directory / ("dct_band_energy_map_" + timestamp + ".png")
            ).resolve(): images["dct_band_energy_map"],
            (
                directory
                / ("block_boundary_visualization_" + timestamp + ".png")
            ).resolve(): images["block_boundary_visualization"],
            (
                directory / ("blockiness_heatmap_" + timestamp + ".png")
            ).resolve(): images["blockiness_heatmap"],
        }
        if not all(
            cv2.imwrite(str(path), image)
            for path, image in image_paths.items()
        ):
            print("DCT/block debug gorselleri kaydedilemedi.")
            return False

        report_path = (
            directory
            / ("coefficient_statistics_report_" + timestamp + ".json")
        ).resolve()
        report = {
            "frame_timestamp": context.frame_timestamp,
            "module_name": result.module_name,
            "status": result.status,
            "calibrated": result.calibrated,
            "scoring_mode": result.debug_data.get("scoring_mode"),
            "calibration_file": str(
                config.MODEL_FREE_CALIBRATION_FILE_PATH
            ),
            "raw_score": result.raw_score,
            "stabilized_score": result.stabilized_score,
            "confidence": result.confidence,
            "coefficient_statistics": statistics,
            "raw_features": result.raw_features,
            "evidence": result.evidence,
            "warnings": result.warnings,
            "debug_data": result.debug_data,
            "quality": {
                "valid": context.face_quality_valid,
                "reason": context.quality_reason,
                "blur": context.blur_value,
                "brightness": context.brightness_value,
                "source_face_dimensions": context.face_dimensions,
            },
            "preprocessing": {
                "input": "standardized aligned grayscale face crop",
                "analysis_dimensions": context.analysis_dimensions,
                "block_size": config.DCT_BLOCK_SIZE,
                "incomplete_edge_blocks": "discarded",
                "floating_point_dct": True,
                "fft_hann_window_applied": False,
            },
            "limitations": [
                "Input is a decoded camera frame, not original JPEG bytes.",
                "Original JPEG quantization tables are not available.",
                "No definitive double-JPEG history is claimed.",
                "Normal video compression, resizing, blur and smoothing may "
                "create similar evidence.",
            ],
        }
        try:
            with report_path.open("w", encoding="utf-8") as report_file:
                json.dump(report, report_file, indent=2, ensure_ascii=False)
        except (OSError, TypeError, ValueError) as error:
            print("DCT/block istatistik raporu kaydedilemedi: " + str(error))
            return False

        for path in image_paths:
            print("DCT/block debug dosyasi kaydedildi: " + str(path))
        print("DCT/block istatistik raporu kaydedildi: " + str(report_path))
        return True

    def _save_wavelet_debug(self, timestamp, context):
        result = self.latest_pre_control_results.get("wavelet")
        subbands = self.wavelet_pre_controller.get_debug_subbands()
        heatmap = self.wavelet_pre_controller.get_anomaly_heatmap()
        feature_report = self.wavelet_pre_controller.get_feature_report()
        if (
            result is None
            or not result.available
            or subbands is None
            or heatmap is None
            or feature_report is None
        ):
            print(
                "Wavelet sonucu kullanilamiyor; debug dosyalari "
                "kaydedilmedi."
            )
            return False

        directory = config.WAVELET_DEBUG_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        raw_paths = []
        visualization_paths = {}
        try:
            for level_number, level_data in subbands.items():
                for band_name, values in level_data.items():
                    stem = "wavelet_level_%d_%s_" % (
                        level_number,
                        band_name,
                    )
                    raw_path = (
                        directory / (stem + "raw_" + timestamp + ".npy")
                    ).resolve()
                    np.save(str(raw_path), values, allow_pickle=False)
                    raw_paths.append(raw_path)

                    visualization = (
                        self.wavelet_pre_controller
                        .create_normalized_subband_visualization(
                            values,
                            is_detail=(band_name != "LL"),
                        )
                    )
                    visualization_path = (
                        directory
                        / (stem + "normalized_" + timestamp + ".png")
                    ).resolve()
                    visualization_paths[visualization_path] = visualization
        except (OSError, TypeError, ValueError) as error:
            print("Wavelet raw subband dosyasi kaydedilemedi: " + str(error))
            return False

        heatmap_path = (
            directory / ("wavelet_anomaly_heatmap_" + timestamp + ".png")
        ).resolve()
        visualization_paths[heatmap_path] = heatmap
        if not all(
            cv2.imwrite(str(path), image)
            for path, image in visualization_paths.items()
        ):
            print("Wavelet debug gorselleri kaydedilemedi.")
            return False

        report_path = (
            directory / ("wavelet_feature_report_" + timestamp + ".json")
        ).resolve()
        report = {
            "frame_timestamp": context.frame_timestamp,
            "module_name": result.module_name,
            "status": result.status,
            "calibrated": result.calibrated,
            "scoring_mode": result.debug_data.get("scoring_mode"),
            "calibration_file": str(
                config.MODEL_FREE_CALIBRATION_FILE_PATH
            ),
            "raw_score": result.raw_score,
            "stabilized_score": result.stabilized_score,
            "confidence": result.confidence,
            "raw_features": feature_report,
            "evidence": result.evidence,
            "warnings": result.warnings,
            "debug_data": result.debug_data,
            "subband_mapping": {
                "LL": "approximation",
                "LH": "PyWavelets horizontal detail",
                "HL": "PyWavelets vertical detail",
                "HH": "diagonal detail",
            },
            "preprocessing": {
                "input": "standardized aligned grayscale face crop",
                "wavelet": config.WAVELET_NAME,
                "levels": config.WAVELET_DECOMPOSITION_LEVELS,
                "boundary_mode": config.WAVELET_BOUNDARY_MODE,
                "inner_face_mask": config.WAVELET_USE_INNER_FACE_MASK,
            },
            "heatmap_note": (
                "Explanatory patch-level wavelet anomaly map; it is not "
                "a neural-network attention map."
            ),
        }
        try:
            with report_path.open("w", encoding="utf-8") as report_file:
                json.dump(report, report_file, indent=2, ensure_ascii=False)
        except (OSError, TypeError, ValueError) as error:
            print("Wavelet feature raporu kaydedilemedi: " + str(error))
            return False

        for path in raw_paths:
            print("Wavelet raw subband kaydedildi: " + str(path))
        for path in visualization_paths:
            print("Wavelet debug gorseli kaydedildi: " + str(path))
        print("Wavelet feature raporu kaydedildi: " + str(report_path))
        return True

    def _save_residual_debug(self, timestamp, context):
        result = self.latest_pre_control_results.get("residual")
        images = self.residual_pre_controller.get_debug_images()
        feature_report = self.residual_pre_controller.get_feature_report()
        if (
            result is None
            or not result.available
            or images is None
            or feature_report is None
        ):
            print(
                "Residual sonucu kullanilamiyor; debug dosyalari "
                "kaydedilmedi."
            )
            return False

        directory = config.RESIDUAL_DEBUG_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        image_paths = {
            (
                directory / ("gaussian_residual_" + timestamp + ".png")
            ).resolve(): images["gaussian_residual"],
            (
                directory / ("laplacian_" + timestamp + ".png")
            ).resolve(): images["laplacian"],
            (
                directory / ("gradient_magnitude_" + timestamp + ".png")
            ).resolve(): images["gradient_magnitude"],
            (
                directory
                / ("patch_residual_energy_map_" + timestamp + ".png")
            ).resolve(): images["patch_residual_energy_map"],
        }
        if not all(
            cv2.imwrite(str(path), image)
            for path, image in image_paths.items()
        ):
            print("Residual debug gorselleri kaydedilemedi.")
            return False

        report_path = (
            directory / ("residual_feature_report_" + timestamp + ".json")
        ).resolve()
        report = {
            "frame_timestamp": context.frame_timestamp,
            "module_name": result.module_name,
            "status": result.status,
            "calibrated": result.calibrated,
            "scoring_mode": result.debug_data.get("scoring_mode"),
            "calibration_file": str(
                config.MODEL_FREE_CALIBRATION_FILE_PATH
            ),
            "raw_score": result.raw_score,
            "stabilized_score": result.stabilized_score,
            "confidence": result.confidence,
            "raw_features": feature_report,
            "evidence": result.evidence,
            "warnings": result.warnings,
            "debug_data": result.debug_data,
            "quality": {
                "valid": context.face_quality_valid,
                "reason": context.quality_reason,
                "blur": context.blur_value,
                "brightness": context.brightness_value,
                "source_face_dimensions": context.face_dimensions,
            },
            "preprocessing": {
                "input": "standardized aligned grayscale/luminance face crop",
                "analysis_dimensions": context.analysis_dimensions,
                "numeric_type": "float32",
                "signed_residuals_retained_for_analysis": True,
                "uint8_normalization": "debug visualization only",
                "inner_face_spatial_weighting": True,
                "gaussian_kernel_size": (
                    config.RESIDUAL_GAUSSIAN_KERNEL_SIZE
                ),
                "gaussian_sigma": config.RESIDUAL_GAUSSIAN_SIGMA,
                "laplacian_kernel_size": (
                    config.RESIDUAL_LAPLACIAN_KERNEL_SIZE
                ),
                "sobel_kernel_size": config.RESIDUAL_SOBEL_KERNEL_SIZE,
                "local_patch_size": config.RESIDUAL_LOCAL_PATCH_SIZE,
            },
            "limitations": [
                "Residual evidence supports analysis but is not a fraud decision.",
                "Blur, low light, ISO noise and camera sharpening can alter residuals.",
                "DroidCam and video compression can alter fine-scale statistics.",
                "No Noiseprint, neural PRNU, CNN, F3-Net or pretrained "
                "network is used.",
            ],
        }
        try:
            with report_path.open("w", encoding="utf-8") as report_file:
                json.dump(report, report_file, indent=2, ensure_ascii=False)
        except (OSError, TypeError, ValueError) as error:
            print("Residual feature raporu kaydedilemedi: " + str(error))
            return False

        for path in image_paths:
            print("Residual debug gorseli kaydedildi: " + str(path))
        print("Residual feature raporu kaydedildi: " + str(report_path))
        return True

    def prepare_frame(self, camera_frame):
        if config.MIRROR_CAMERA_IMAGE:
            return cv2.flip(camera_frame, 1)
        return camera_frame

    def create_guide_box(self, frame):
        frame_height, frame_width = frame.shape[:2]
        side = int(
            min(frame_width, frame_height)
            * config.MODEL_FREE_GUIDE_DIAMETER_RATIO
        )
        center_x = frame_width // 2
        center_y = int(
            frame_height * config.MODEL_FREE_GUIDE_CENTER_Y_RATIO
        )

        return FaceBox(
            center_x - side // 2,
            center_y - side // 2,
            side,
            side,
        ).clamp_to_frame(frame_width, frame_height)

    def _box_touches_frame_edge(self, box, frame):
        frame_height, frame_width = frame.shape[:2]
        margin_ratio = (
            config.EXPERIMENTAL_MODEL_FREE_FRAME_EDGE_MARGIN_RATIO
        )
        margin_x = int(frame_width * margin_ratio)
        margin_y = int(frame_height * margin_ratio)
        return bool(
            box.x <= margin_x
            or box.y <= margin_y
            or box.x + box.width >= frame_width - margin_x
            or box.y + box.height >= frame_height - margin_y
        )

    def draw_guide(
        self,
        frame,
        _guide_box,
        fft_result,
        moire_result,
        tracking_result=None,
        analysis_box=None,
    ):
        # Metot adi eski cagrilarla API uyumlulugu icin korunur. Sabit merkez
        # dairesi artik cizilmez ve detector etkin GUI akışında analiz girdisi
        # degildir; tek ROI referansi asagidaki takip kutusudur.
        guide_color = (0, 255, 0) if fft_result.passed else (0, 165, 255)
        combined_result = self.latest_combined_result
        decision = self.latest_precontrol_decision
        if decision is not None and decision.classification in (
            "SUSPICIOUS",
            "HIGH_RISK",
        ):
            guide_color = (0, 0, 255)
        elif decision is not None and decision.classification in (
            "INSUFFICIENT_QUALITY",
            "INSUFFICIENT_EVIDENCE",
            "UNSUPPORTED_CAPTURE",
        ):
            guide_color = (0, 165, 255)
        elif decision is not None and decision.classification == "LIVE":
            guide_color = (0, 255, 0)
        elif (
            combined_result is not None
            and combined_result.status
            in (
                "Suspicious mathematical evidence",
                "High mathematical risk",
            )
        ):
            guide_color = (0, 0, 255)
        elif (
            combined_result is not None
            and combined_result.status == "Weak anomaly evidence"
        ):
            guide_color = (0, 165, 255)
        elif fft_result.warning:
            guide_color = (0, 0, 255)

        if (
            tracking_result is not None
            and tracking_result.supported
            and analysis_box is not None
        ):
            tracking_color = (
                (0, 210, 255)
                if tracking_result.status == "HELD"
                else (255, 255, 0)
            )
            cv2.rectangle(
                frame,
                analysis_box.get_top_left().to_tuple(),
                analysis_box.get_bottom_right().to_tuple(),
                tracking_color,
                2,
            )
        self.draw_text(
            frame,
            "MODEL-FREE PRE-CONTROL",
            35,
            (255, 255, 255),
            0.75,
            2,
        )
        self.draw_text(
            frame,
            self._tracking_display_text(tracking_result),
            68,
            guide_color,
            0.65,
            2,
        )

        score_text = "Global FFT: " + fft_result.status
        if fft_result.score is not None:
            score_text += " | %d/100" % round(fft_result.score)
        self.draw_text(
            frame,
            score_text,
            101,
            (255, 255, 255),
            0.60,
            2,
        )
        self.draw_text(
            frame,
            "Kalibrasyon: "
            + ("Calibrated" if fft_result.calibrated else "Experimental"),
            132,
            guide_color,
            0.55,
            2,
        )

        moire_color = (0, 255, 0)
        if moire_result.status in ("Analysis Uncertain", "Unavailable"):
            moire_color = (0, 165, 255)
        elif moire_result.status in (
            "Suspicious",
            "Possible Screen Replay",
        ):
            moire_color = (0, 0, 255)

        moire_text = "Moire: " + moire_result.status
        if moire_result.score is not None:
            moire_text += " | %d/100" % round(moire_result.score)
        self.draw_text(
            frame,
            moire_text,
            163,
            moire_color,
            0.55,
            2,
        )

        if decision is not None:
            combined_text = "PreControl: " + decision.classification
            if decision.overall_risk_0_100 is not None:
                combined_text += " | %d/100" % round(
                    decision.overall_risk_0_100
                )
            combined_text += " | r=%.2f" % decision.overall_reliability
            self.draw_text(
                frame,
                combined_text,
                194,
                guide_color,
                0.55,
                2,
            )

        if decision is not None and decision.classification in (
            "SUSPICIOUS",
            "HIGH_RISK",
        ):
            self.draw_combined_warning(frame)
        elif combined_result is not None and combined_result.warning:
            self.draw_combined_warning(frame)
        elif moire_result.warning:
            self.draw_moire_warning(frame)
        elif fft_result.warning:
            self.draw_global_fft_warning(frame)

    def _tracking_display_text(self, tracking_result):
        if not self.face_detection_enabled:
            return "Yuz tespiti kapali - legacy test akisi"
        if tracking_result is None:
            return "Yuz tespiti bekleniyor"
        if tracking_result.status == "DETECTED":
            return "Yuz takip ediliyor - analiz turkuaz ROI kutusunda"
        if tracking_result.status == "HELD":
            return "Yuz kisa sureli takip ile korunuyor"
        if tracking_result.status == "MULTIPLE_FACES":
            return "Kadrajda yalnizca bir yuz olmali"
        if tracking_result.status in (
            "DETECTOR_ERROR",
            "DETECTOR_UNAVAILABLE",
        ):
            return "Yuz tespit altyapisi kullanilamiyor"
        return "Yuz bulunamadi - kameraya bak ve kadrajda kal"

    def draw_combined_warning(self, frame):
        frame_height, frame_width = frame.shape[:2]
        overlay = frame.copy()
        top = max(226, frame_height - 165)
        cv2.rectangle(
            overlay,
            (0, top),
            (frame_width, frame_height),
            (0, 0, 180),
            -1,
        )
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        self.draw_text(
            frame,
            "WARNING: Possible screen or printed-photo presentation",
            top + 48,
            (255, 255, 255),
            0.55,
            2,
        )
        self.draw_text(
            frame,
            "Supporting evidence only; not a definitive verdict",
            top + 82,
            (255, 255, 255),
            0.50,
            1,
        )

    def draw_moire_warning(self, frame):
        frame_height, frame_width = frame.shape[:2]
        overlay = frame.copy()
        top = max(195, frame_height - 165)
        cv2.rectangle(
            overlay,
            (0, top),
            (frame_width, frame_height),
            (0, 0, 180),
            -1,
        )
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        self.draw_text(
            frame,
            "WARNING: Possible screen replay / periodic display pattern",
            top + 48,
            (255, 255, 255),
            0.55,
            2,
        )
        self.draw_text(
            frame,
            "Suspicion signal only - not a definitive authenticity verdict",
            top + 88,
            (255, 255, 255),
            0.48,
            1,
        )

    def draw_global_fft_warning(self, frame):
        frame_height, frame_width = frame.shape[:2]
        overlay = frame.copy()
        top = max(165, frame_height - 150)
        cv2.rectangle(
            overlay,
            (0, top),
            (frame_width, frame_height),
            (0, 0, 180),
            -1,
        )
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        self.draw_text(
            frame,
            "WARNING: Global spectral anomaly",
            top + 48,
            (255, 255, 255),
            0.75,
            2,
        )
        self.draw_text(
            frame,
            "Experimental signal - not an authenticity verdict",
            top + 88,
            (255, 255, 255),
            0.58,
            1,
        )

    def draw_text(self, frame, text, y, color, scale, thickness):
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )

    def shutdown(self):
        if self.is_closed:
            return
        self.fft_pre_controller.reset()
        self.moire_pre_controller.reset()
        self.radial_angular_pre_controller.reset()
        self.dct_block_pre_controller.reset()
        self.wavelet_pre_controller.reset()
        self.residual_pre_controller.reset()
        self.periodicity_pre_controller.reset()
        self.fusion_controller.reset()
        self.face_roi_tracker.reset()
        if self._owns_face_detector and self.face_detector is not None:
            self.face_detector.close()
        self.latest_combined_result = None
        self.latest_precontrol_decision = None
        self.latest_context = None
        self.latest_face_tracking_result = None
        self.is_closed = True

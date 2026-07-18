from __future__ import annotations

import time

from data_models import (
    Challenge,
    FaceLandmarkDetection,
    FaceQuality,
    LivenessResult,
)


class ActiveLivenessAnalyzer:
    """Göz kırpma ve kafa çevirme görevlerini sırayla takip eder."""

    CHALLENGE_SEQUENCE = (
        Challenge.BLINK,
        Challenge.TURN_LEFT,
        Challenge.TURN_RIGHT,
    )

    def __init__(
        self,
        blink_closed_threshold: float = 0.55,
        blink_open_threshold: float = 0.30,
        turn_angle_threshold: float = 15.0,
        stable_frames_required: int = 3,
    ) -> None:
        self.blink_closed_threshold = blink_closed_threshold
        self.blink_open_threshold = blink_open_threshold
        self.turn_angle_threshold = turn_angle_threshold
        self.stable_frames_required = stable_frames_required

        self._started_at = time.perf_counter()
        self._completed_challenges: list[Challenge] = []
        self._blink_started_open = False
        self._blink_closed_seen = False
        self._stable_frame_count = 0

    @property
    def completed_challenges(self) -> tuple[Challenge, ...]:
        return tuple(self._completed_challenges)

    @property
    def is_complete(self) -> bool:
        return len(self._completed_challenges) == len(
            self.CHALLENGE_SEQUENCE
        )

    @property
    def current_challenge(self) -> Challenge | None:
        if self.is_complete:
            return None

        return self.CHALLENGE_SEQUENCE[len(self._completed_challenges)]

    @property
    def current_instruction(self) -> str:
        instructions = {
            Challenge.BLINK: "GOZ KIRP",
            Challenge.TURN_LEFT: "KAFANI SOLA CEVIR",
            Challenge.TURN_RIGHT: "KAFANI SAGA CEVIR",
        }

        if self.current_challenge is None:
            return "CANLILIK TAMAMLANDI"

        return instructions[self.current_challenge]

    def update(
        self,
        detected_face: FaceLandmarkDetection,
        quality: FaceQuality,
    ) -> None:
        if self.is_complete or not quality.is_acceptable:
            self._stable_frame_count = 0
            return

        if self.current_challenge == Challenge.BLINK:
            self._update_blink(detected_face.average_blink_score)
            return

        if self.current_challenge == Challenge.TURN_LEFT:
            self._update_head_turn(
                condition=(
                    detected_face.yaw_degrees
                    <= -self.turn_angle_threshold
                )
            )
            return

        if self.current_challenge == Challenge.TURN_RIGHT:
            self._update_head_turn(
                condition=(
                    detected_face.yaw_degrees
                    >= self.turn_angle_threshold
                )
            )

    def build_result(
        self,
        face_detected: bool,
        quality: FaceQuality | None,
    ) -> LivenessResult:
        quality_is_acceptable = (
            quality is not None and quality.is_acceptable
        )

        if not face_detected:
            quality_status = "NO_FACE"
        elif quality_is_acceptable:
            quality_status = "ACCEPTABLE"
        else:
            quality_status = "UNACCEPTABLE"

        verdict = "BONA_FIDE" if self.is_complete else "IN_PROGRESS"

        if not face_detected:
            risk_score = 100
        else:
            completed_count = len(self._completed_challenges)
            risk_score = max(18, 90 - (24 * completed_count))

            if not quality_is_acceptable:
                risk_score = min(100, risk_score + 10)

        processing_time_ms = int(
            (time.perf_counter() - self._started_at) * 1000
        )

        return LivenessResult(
            face_detected=face_detected,
            quality_status=quality_status,
            liveness_type="ACTIVE",
            challenge_sequence=self.CHALLENGE_SEQUENCE,
            completed_challenges=self.completed_challenges,
            verdict=verdict,
            risk_score=risk_score,
            processing_time_ms=processing_time_ms,
        )

    def _update_blink(self, blink_score: float) -> None:
        if blink_score <= self.blink_open_threshold:
            if self._blink_closed_seen:
                self._complete_current_challenge()
                return

            self._blink_started_open = True

        if (
            self._blink_started_open
            and blink_score >= self.blink_closed_threshold
        ):
            self._blink_closed_seen = True

    def _update_head_turn(self, condition: bool) -> None:
        if condition:
            self._stable_frame_count += 1
        else:
            self._stable_frame_count = 0

        if self._stable_frame_count >= self.stable_frames_required:
            self._complete_current_challenge()

    def _complete_current_challenge(self) -> None:
        if self.current_challenge is not None:
            self._completed_challenges.append(self.current_challenge)

        self._stable_frame_count = 0

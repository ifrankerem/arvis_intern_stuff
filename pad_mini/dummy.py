from typing import Any

from contracts.deterministic_method import (
    DeterministicMethod,
    DeterministicMethodInput,
    DeterministicMethodResult,
)
from contracts.media import ImageArray


class ExampleDeterministicMethod(DeterministicMethod):
    @property
    def name(self) -> str:
        return "example_deterministic_method"

    def run(
        self,
        method_input: DeterministicMethodInput,
    ) -> DeterministicMethodResult:
        image = method_input.media.data

        processed_image = self._preprocess(image)
        raw_value = self._analyze(processed_image)
        score = self._normalize_score(raw_value)
        details = self._build_details(raw_value)

        return DeterministicMethodResult(
            method_name=self.name,
            score=score,
            details=details,
        )

    def _preprocess(
        self,
        image: ImageArray,
    ) -> ImageArray:
        """
        Prepare the image for analysis.

        Examples:
        - Resize the image.
        - Convert the color space.
        - Apply filtering.
        - Select a region of interest.
        """
        return image

    def _analyze(
        self,
        image: ImageArray,
    ) -> float:
        """
        Execute the deterministic analysis and return a raw value.

        Replace this dummy value with the actual algorithm.
        """
        return 0.0

    def _normalize_score(
        self,
        raw_value: float,
    ) -> float:
        """
        Convert the raw algorithm output into a score between 0 and 1.

        0.0 means no suspicious signal.
        1.0 means maximum suspicious signal.
        """
        return min(1.0, max(0.0, raw_value))

    def _build_details(
        self,
        raw_value: float,
    ) -> dict[str, Any]:
        """
        Build optional diagnostic information about the analysis.
        """
        return {
            "raw_value": raw_value,
        }

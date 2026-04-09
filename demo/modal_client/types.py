from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


PromptMode = Literal["visual", "geometric"]


@dataclass(frozen=True, slots=True)
class TextPrompt:
    texts: list[str]
    type: str = "text"

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "texts": [text.strip() for text in self.texts if text.strip()],
        }


@dataclass(frozen=True, slots=True)
class BoxPrompt:
    box: tuple[float, float, float, float]
    mode: PromptMode = "visual"
    label_text: str | None = None
    type: str = "box"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "mode": self.mode,
            "box": [float(value) for value in self.box],
        }
        if self.label_text:
            payload["label_text"] = self.label_text
        return payload


@dataclass(frozen=True, slots=True)
class PointPrompt:
    points: list[tuple[float, float, int]]
    mode: PromptMode = "geometric"
    label_text: str | None = None
    type: str = "point"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "mode": self.mode,
            "points": [
                {"x": float(x), "y": float(y), "label": int(label)}
                for x, y, label in self.points
            ],
        }
        if self.label_text:
            payload["label_text"] = self.label_text
        return payload


@dataclass(frozen=True, slots=True)
class InferRequest:
    prompt: TextPrompt | BoxPrompt | PointPrompt
    score_threshold: float = 0.3
    intrinsics: list[list[float]] | None = None
    use_predicted_intrinsics: bool | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": self.prompt.to_payload(),
            "score_threshold": float(self.score_threshold),
        }
        if self.intrinsics is not None:
            payload["intrinsics"] = self.intrinsics
        if self.use_predicted_intrinsics is not None:
            payload["use_predicted_intrinsics"] = self.use_predicted_intrinsics
        return payload


@dataclass(frozen=True, slots=True)
class ImageSize:
    height: int
    width: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageSize:
        return cls(
            height=int(data["height"]),
            width=int(data["width"]),
        )


@dataclass(frozen=True, slots=True)
class DepthSummary:
    shape: list[int]
    min: float | None
    max: float | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DepthSummary:
        return cls(
            shape=[int(value) for value in data["shape"]],
            min=None if data.get("min") is None else float(data["min"]),
            max=None if data.get("max") is None else float(data["max"]),
        )


@dataclass(frozen=True, slots=True)
class ConfidenceSummary:
    shape: list[int]
    mean: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfidenceSummary:
        return cls(
            shape=[int(value) for value in data["shape"]],
            mean=float(data["mean"]),
        )


@dataclass(frozen=True, slots=True)
class Detection:
    box2d: list[float]
    box3d: list[float]
    score: float
    score_2d: float
    score_3d: float
    class_id: int
    class_name: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Detection:
        return cls(
            box2d=[float(value) for value in data["box2d"]],
            box3d=[float(value) for value in data["box3d"]],
            score=float(data["score"]),
            score_2d=float(data["score_2d"]),
            score_3d=float(data["score_3d"]),
            class_id=int(data["class_id"]),
            class_name=str(data["class_name"]),
        )


@dataclass(frozen=True, slots=True)
class InferResponse:
    detections: list[Detection]
    num_detections: int
    prompt: dict[str, Any]
    image_size: ImageSize
    score_threshold: float
    use_predicted_intrinsics: bool
    provided_intrinsics: list[list[float]] | None
    predicted_intrinsics: list[list[float]] | None
    depth_summary: DepthSummary | None
    confidence_summary: ConfidenceSummary | None
    model_device: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferResponse:
        return cls(
            detections=[
                Detection.from_dict(item) for item in data.get("detections", [])
            ],
            num_detections=int(data["num_detections"]),
            prompt=dict(data["prompt"]),
            image_size=ImageSize.from_dict(data["image_size"]),
            score_threshold=float(data["score_threshold"]),
            use_predicted_intrinsics=bool(data["use_predicted_intrinsics"]),
            provided_intrinsics=data.get("provided_intrinsics"),
            predicted_intrinsics=data.get("predicted_intrinsics"),
            depth_summary=(
                None
                if data.get("depth_summary") is None
                else DepthSummary.from_dict(data["depth_summary"])
            ),
            confidence_summary=(
                None
                if data.get("confidence_summary") is None
                else ConfidenceSummary.from_dict(data["confidence_summary"])
            ),
            model_device=str(data["model_device"]),
        )

    def render_intrinsics(
        self,
        *,
        prefer_predicted: bool = True,
    ) -> list[list[float]] | None:
        if prefer_predicted and self.predicted_intrinsics is not None:
            return self.predicted_intrinsics
        if self.provided_intrinsics is not None:
            return self.provided_intrinsics
        return self.predicted_intrinsics

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [
                {
                    "box2d": detection.box2d,
                    "box3d": detection.box3d,
                    "score": detection.score,
                    "score_2d": detection.score_2d,
                    "score_3d": detection.score_3d,
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                }
                for detection in self.detections
            ],
            "num_detections": self.num_detections,
            "prompt": self.prompt,
            "image_size": {
                "height": self.image_size.height,
                "width": self.image_size.width,
            },
            "score_threshold": self.score_threshold,
            "use_predicted_intrinsics": self.use_predicted_intrinsics,
            "provided_intrinsics": self.provided_intrinsics,
            "predicted_intrinsics": self.predicted_intrinsics,
            "depth_summary": (
                None
                if self.depth_summary is None
                else {
                    "shape": self.depth_summary.shape,
                    "min": self.depth_summary.min,
                    "max": self.depth_summary.max,
                }
            ),
            "confidence_summary": (
                None
                if self.confidence_summary is None
                else {
                    "shape": self.confidence_summary.shape,
                    "mean": self.confidence_summary.mean,
                }
            ),
            "model_device": self.model_device,
        }

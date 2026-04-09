from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import Detection, InferResponse


_FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
]

_EDGES = [
    # Local corner order in `_box3d_to_corners` is:
    # bottom face 0-1-2-3, top face 4-5-6-7.
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

_PALETTE_RGB = [
    (231, 76, 60),
    (59, 130, 246),
    (34, 197, 94),
    (245, 158, 11),
    (168, 85, 247),
    (6, 182, 212),
    (236, 72, 153),
    (249, 115, 22),
    (244, 114, 182),
    (16, 185, 129),
]


def _get_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    return _PALETTE_RGB[class_id % len(_PALETTE_RGB)]


def _rgb_to_bgr(color_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (color_rgb[2], color_rgb[1], color_rgb[0])


def _quaternion_to_rotation_matrix(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
) -> np.ndarray:
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _box3d_to_corners(box3d: list[float]) -> np.ndarray:
    cx, cy, cz, width, length, height, qw, qx, qy, qz = box3d
    hw = width / 2.0
    hl = length / 2.0
    hh = height / 2.0
    local = np.array(
        [
            [-hl, -hh, -hw],
            [hl, -hh, -hw],
            [hl, hh, -hw],
            [-hl, hh, -hw],
            [-hl, -hh, hw],
            [hl, -hh, hw],
            [hl, hh, hw],
            [-hl, hh, hw],
        ],
        dtype=np.float32,
    )
    rotation = _quaternion_to_rotation_matrix(qw, qx, qy, qz)
    center = np.array([cx, cy, cz], dtype=np.float32)
    return (local @ rotation.T) + center


def _project_point(point_3d: np.ndarray, intrinsics: np.ndarray) -> tuple[float, float]:
    x, y, z = point_3d
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    return float(fx * x / z + cx), float(fy * y / z + cy)


def _clip_to_near(
    p1: np.ndarray,
    p2: np.ndarray,
    near: float,
) -> np.ndarray:
    k_up = abs(p1[2] - near)
    k_down = abs(p1[2] - p2[2])
    k = min(k_up / k_down, 1.0) if k_down > 0 else 1.0
    return (1.0 - k) * p1 + k * p2


def _resolve_intrinsics(
    response: InferResponse,
    *,
    prefer_predicted: bool = True,
) -> np.ndarray | None:
    intrinsics = response.render_intrinsics(prefer_predicted=prefer_predicted)
    if intrinsics is None:
        return None
    return np.array(intrinsics, dtype=np.float32)


def _draw_prompt_overlay(image_bgr: np.ndarray, prompt: dict[str, Any]) -> np.ndarray:
    output = image_bgr.copy()
    prompt_type = prompt.get("type")

    if prompt_type == "box":
        x1, y1, x2, y2 = [int(round(value)) for value in prompt["box"]]
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    elif prompt_type == "point":
        for point in prompt.get("points", []):
            if isinstance(point, dict):
                x = int(round(point["x"]))
                y = int(round(point["y"]))
                label = int(point["label"])
            else:
                x = int(round(point[0]))
                y = int(round(point[1]))
                label = int(point[2])
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(output, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.circle(output, (x, y), 6, (255, 255, 255), 2, cv2.LINE_AA)

    return output


def _draw_2d_boxes(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.box2d]
        color_bgr = _rgb_to_bgr(_color_for_class(detection.class_id))
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color_bgr,
            2,
            cv2.LINE_AA,
        )
    return output


def _draw_3d_boxes(
    image_bgr: np.ndarray,
    detections: list[Detection],
    intrinsics: np.ndarray,
    *,
    near_clip: float = 0.15,
) -> np.ndarray:
    output = image_bgr.copy()
    image_h, image_w = output.shape[:2]
    margin = max(image_h, image_w)

    for detection in detections:
        color_bgr = _rgb_to_bgr(_color_for_class(detection.class_id))
        corners = _box3d_to_corners(detection.box3d)
        for idx0, idx1 in _EDGES:
            p1 = corners[idx0]
            p2 = corners[idx1]

            if p1[2] < near_clip and p2[2] < near_clip:
                continue
            if p1[2] < near_clip:
                p1 = _clip_to_near(p1, p2, near_clip)
            elif p2[2] < near_clip:
                p2 = _clip_to_near(p2, p1, near_clip)

            u1, v1 = _project_point(p1, intrinsics)
            u2, v2 = _project_point(p2, intrinsics)
            if (
                abs(u1) > margin * 2
                or abs(v1) > margin * 2
                or abs(u2) > margin * 2
                or abs(v2) > margin * 2
            ):
                continue

            cv2.line(
                output,
                (int(round(u1)), int(round(v1))),
                (int(round(u2)), int(round(v2))),
                color_bgr,
                2,
                cv2.LINE_AA,
            )
    return output


def _build_label(detection: Detection) -> str:
    return (
        f"{detection.class_name} "
        f"2D:{detection.score_2d:.2f} "
        f"3D:{detection.score_3d:.2f}"
    )


def _draw_labels_with_pillow(
    image_bgr: np.ndarray,
    detections: list[Detection],
    *,
    font_size: int = 15,
) -> np.ndarray:
    if not detections:
        return image_bgr

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(font_size)
    image_h, image_w = image_bgr.shape[:2]

    for detection in detections:
        label = _build_label(detection)
        color_rgb = _color_for_class(detection.class_id)
        x1, y1, _, _ = detection.box2d
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w = right - left
        text_h = bottom - top
        pad_x = 6
        pad_y = 4

        rx0 = max(2, int(round(x1)))
        ry0 = max(2, int(round(y1 - text_h - 2 * pad_y - 6)))
        rx1 = min(image_w - 2, rx0 + text_w + 2 * pad_x)
        ry1 = min(image_h - 2, ry0 + text_h + 2 * pad_y)

        if rx1 - rx0 < text_w + 2 * pad_x:
            rx0 = max(2, rx1 - text_w - 2 * pad_x)
        if ry1 - ry0 < text_h + 2 * pad_y:
            ry0 = max(2, ry1 - text_h - 2 * pad_y)

        draw.rounded_rectangle(
            [rx0, ry0, rx1, ry1],
            radius=5,
            fill=tuple(color_rgb) + (215,),
        )
        draw.text(
            (rx0 + pad_x, ry0 + pad_y - top),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    rendered = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR)


def load_image_bgr(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image.copy()

    image_path = Path(image)
    loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if loaded is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return loaded


def render_response(
    image: str | Path | np.ndarray,
    response: InferResponse,
    *,
    draw_2d: bool = True,
    draw_3d: bool = True,
    draw_labels: bool = True,
    draw_prompt: bool = True,
    prefer_predicted_intrinsics: bool = True,
    near_clip: float = 0.15,
) -> np.ndarray:
    rendered = load_image_bgr(image)

    if draw_prompt:
        rendered = _draw_prompt_overlay(rendered, response.prompt)
    if draw_2d:
        rendered = _draw_2d_boxes(rendered, response.detections)

    if draw_3d:
        intrinsics = _resolve_intrinsics(
            response,
            prefer_predicted=prefer_predicted_intrinsics,
        )
        if intrinsics is not None:
            rendered = _draw_3d_boxes(
                rendered,
                response.detections,
                intrinsics,
                near_clip=near_clip,
            )

    if draw_labels:
        rendered = _draw_labels_with_pillow(rendered, response.detections)

    return rendered


def save_rendered_image(path: str | Path, image_bgr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Failed to write image: {path}")

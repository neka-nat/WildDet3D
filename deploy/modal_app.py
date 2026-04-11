import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile


APP_NAME = os.environ.get("WILDDET3D_MODAL_APP_NAME", "wilddet3d-api")
API_LABEL = os.environ.get("WILDDET3D_MODAL_LABEL", "wilddet3d-api")
GPU_TYPE = os.environ.get("WILDDET3D_MODAL_GPU", "L40S")
MIN_CONTAINERS = int(os.environ.get("WILDDET3D_MODAL_MIN_CONTAINERS", "0"))
SCALEDOWN_WINDOW = int(
    os.environ.get("WILDDET3D_MODAL_SCALEDOWN_WINDOW", str(15 * 60))
)
VIS4D_SOURCE_URL = os.environ.get(
    "WILDDET3D_VIS4D_SOURCE_URL",
    "https://github.com/SysCV/vis4d/archive/refs/heads/main.zip",
)
MODEL_VOLUME_NAME = os.environ.get(
    "WILDDET3D_MODEL_VOLUME_NAME", "wilddet3d-models"
)
HF_CACHE_VOLUME_NAME = os.environ.get(
    "WILDDET3D_HF_CACHE_VOLUME_NAME", "wilddet3d-hf-cache"
)
MODEL_VOLUME_PATH = "/models"
HF_CACHE_VOLUME_PATH = "/hf-cache"
REMOTE_REPO_ROOT = "/root/wilddet3d"
DEFAULT_CHECKPOINT_FILE = os.environ.get(
    "WILDDET3D_CHECKPOINT_FILE",
    "wilddet3d_alldata_all_prompt_v1.0.pt",
)
HF_MODEL_REPO = os.environ.get("WILDDET3D_HF_MODEL_REPO", "allenai/WildDet3D")
MAX_IMAGE_BYTES = int(
    os.environ.get("WILDDET3D_MAX_IMAGE_BYTES", str(20 * 1024 * 1024))
)
REQUIRES_PROXY_AUTH = os.environ.get(
    "WILDDET3D_REQUIRES_PROXY_AUTH", "1"
) not in {"0", "false", "False", "no", "NO"}

_THIS_DIR = Path(__file__).resolve().parent


def _discover_repo_root() -> Path:
    candidates = [
        Path(REMOTE_REPO_ROOT),
        _THIS_DIR,
        _THIS_DIR.parent,
    ]
    for candidate in candidates:
        try:
            if (
                (candidate / "wilddet3d").is_dir()
                and (candidate / "third_party").is_dir()
            ):
                return candidate
        except OSError:
            continue
    return _THIS_DIR.parent


REPO_ROOT = _discover_repo_root()
LOCAL_WILDDET3D_DIR = REPO_ROOT / "wilddet3d"
LOCAL_THIRD_PARTY_DIR = REPO_ROOT / "third_party"
LOCAL_SAM3_DIR = LOCAL_THIRD_PARTY_DIR / "sam3" / "sam3"
LOCAL_LINGBOT_DIR = LOCAL_THIRD_PARTY_DIR / "lingbot_depth" / "mdm"


def _validate_local_sources() -> None:
    missing_paths: list[str] = []
    if not LOCAL_WILDDET3D_DIR.is_dir():
        missing_paths.append(str(LOCAL_WILDDET3D_DIR))
    if not LOCAL_SAM3_DIR.is_dir():
        missing_paths.append(str(LOCAL_SAM3_DIR))
    if not LOCAL_LINGBOT_DIR.is_dir():
        missing_paths.append(str(LOCAL_LINGBOT_DIR))

    if missing_paths:
        joined = "\n".join(f"  - {path}" for path in missing_paths)
        raise RuntimeError(
            "WildDet3D Modal deployment requires initialized submodules.\n"
            "Missing expected source paths:\n"
            f"{joined}\n"
            "Run `git submodule update --init --recursive` before deploying."
        )


def _runtime_packages() -> list[str]:
    packages = [
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy",
        "Pillow",
        "einops",
        "timm>=0.6.0",
        "transformers",
        "utils3d",
        "huggingface_hub",
        "opencv-python-headless",
        "pycocotools",
        "pyquaternion",
        "scipy",
        "terminaltables",
        "ml_collections==1.1.0",
        "tqdm",
        f"vis4d @ {VIS4D_SOURCE_URL}",
        "lightning",
        "pydantic<2",
        "jsonargparse[signatures]",
        "cloudpickle",
        "devtools",
        "termcolor",
        "h5py",
        "kornia",
        "ftfy",
        "regex",
        "iopath",
        "open_clip_torch",
        "safetensors",
        "fastapi>=0.115.0,<1.0",
        "python-multipart>=0.0.9",
    ]
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(packages))


_validate_local_sources()

model_volume = modal.Volume.from_name(
    MODEL_VOLUME_NAME,
    create_if_missing=True,
)
hf_cache_volume = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    create_if_missing=True,
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(_runtime_packages())
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": HF_CACHE_VOLUME_PATH,
            "HUGGINGFACE_HUB_CACHE": HF_CACHE_VOLUME_PATH,
            "TRANSFORMERS_CACHE": HF_CACHE_VOLUME_PATH,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_dir(
        LOCAL_WILDDET3D_DIR,
        remote_path=f"{REMOTE_REPO_ROOT}/wilddet3d",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        LOCAL_THIRD_PARTY_DIR / "sam3",
        remote_path=f"{REMOTE_REPO_ROOT}/third_party/sam3",
        ignore=["**/__pycache__", "**/*.pyc", ".git", ".github"],
    )
    .add_local_dir(
        LOCAL_THIRD_PARTY_DIR / "lingbot_depth",
        remote_path=f"{REMOTE_REPO_ROOT}/third_party/lingbot_depth",
        ignore=["**/__pycache__", "**/*.pyc", ".git", ".github"],
    )
)

app = modal.App(APP_NAME)


class RequestValidationError(ValueError):
    """Invalid inference request."""


def _normalize_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise RequestValidationError(f"`{field_name}` must be a boolean.")


def _ensure_list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise RequestValidationError(f"`{field_name}` must be a list.")
    return value


def _normalize_box(box: Any) -> list[float]:
    values = _ensure_list(box, "prompt.box")
    if len(values) != 4:
        raise RequestValidationError("`prompt.box` must have four values.")
    return [float(v) for v in values]


def _normalize_points(points: Any) -> list[tuple[float, float, int]]:
    values = _ensure_list(points, "prompt.points")
    if not values:
        raise RequestValidationError("`prompt.points` must not be empty.")

    normalized: list[tuple[float, float, int]] = []
    for idx, point in enumerate(values):
        if isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")
            label = point.get("label")
        else:
            raw_point = _ensure_list(point, f"prompt.points[{idx}]")
            if len(raw_point) != 3:
                raise RequestValidationError(
                    f"`prompt.points[{idx}]` must have three values."
                )
            x, y, label = raw_point

        label = int(label)
        if label not in (0, 1):
            raise RequestValidationError(
                f"`prompt.points[{idx}].label` must be 0 or 1."
            )
        normalized.append((float(x), float(y), label))
    return normalized


def _normalize_intrinsics(intrinsics: Any) -> list[list[float]] | None:
    if intrinsics is None:
        return None
    rows = _ensure_list(intrinsics, "intrinsics")
    if len(rows) != 3:
        raise RequestValidationError("`intrinsics` must be a 3x3 matrix.")
    normalized: list[list[float]] = []
    for idx, row in enumerate(rows):
        cols = _ensure_list(row, f"intrinsics[{idx}]")
        if len(cols) != 3:
            raise RequestValidationError("`intrinsics` must be a 3x3 matrix.")
        normalized.append([float(value) for value in cols])
    return normalized


def _normalize_prompt(prompt: Any) -> dict[str, Any]:
    if not isinstance(prompt, dict):
        raise RequestValidationError("`prompt` must be an object.")

    prompt_type = prompt.get("type", "text")
    if prompt_type not in {"text", "box", "point"}:
        raise RequestValidationError(
            "`prompt.type` must be one of: text, box, point."
        )

    if prompt_type == "text":
        texts = prompt.get("texts", [])
        if isinstance(texts, str):
            texts = [t.strip() for t in texts.split(".") if t.strip()]
        texts = [str(t).strip() for t in _ensure_list(texts, "prompt.texts")]
        texts = [text for text in texts if text]
        if not texts:
            texts = ["object"]
        return {"type": "text", "texts": texts}

    mode = prompt.get("mode", "geometric")
    if mode not in {"visual", "geometric"}:
        raise RequestValidationError(
            "`prompt.mode` must be one of: visual, geometric."
        )

    label_text = prompt.get("label_text")
    if label_text is not None:
        label_text = str(label_text).strip() or None

    if prompt_type == "box":
        return {
            "type": "box",
            "mode": mode,
            "box": _normalize_box(prompt.get("box")),
            "label_text": label_text,
        }

    return {
        "type": "point",
        "mode": mode,
        "points": _normalize_points(prompt.get("points")),
        "label_text": label_text,
    }


def _normalize_request(
    payload: dict[str, Any],
    *,
    image_bytes: bytes,
) -> dict[str, Any]:
    prompt = _normalize_prompt(payload.get("prompt", {}))
    intrinsics = _normalize_intrinsics(payload.get("intrinsics"))
    score_threshold = float(payload.get("score_threshold", 0.3))
    if not 0.0 <= score_threshold <= 1.0:
        raise RequestValidationError("`score_threshold` must be in [0, 1].")

    use_predicted_intrinsics = payload.get("use_predicted_intrinsics")
    if use_predicted_intrinsics is None:
        use_predicted_intrinsics = intrinsics is None
    else:
        use_predicted_intrinsics = _normalize_bool(
            use_predicted_intrinsics,
            "use_predicted_intrinsics",
        )

    if intrinsics is None and not use_predicted_intrinsics:
        raise RequestValidationError(
            "`intrinsics` can only be omitted when "
            "`use_predicted_intrinsics` is true or omitted."
        )

    return {
        "image_bytes": image_bytes,
        "prompt": prompt,
        "intrinsics": intrinsics,
        "score_threshold": score_threshold,
        "use_predicted_intrinsics": use_predicted_intrinsics,
    }


def _build_prompt_text(mode: str, label_text: str | None) -> str:
    if label_text:
        return f"{mode}: {label_text}"
    return mode


def _transform_coords_to_input_space(
    x: float,
    y: float,
    original_hw: tuple[int, int],
    input_hw: tuple[int, int],
    padding: tuple[int, int, int, int],
) -> tuple[float, float]:
    orig_h, orig_w = original_hw
    pad_left, pad_right, pad_top, pad_bottom = padding

    content_w = input_hw[1] - pad_left - pad_right
    content_h = input_hw[0] - pad_top - pad_bottom

    scale_x = content_w / orig_w
    scale_y = content_h / orig_h

    return x * scale_x + pad_left, y * scale_y + pad_top


def _scale_intrinsics_to_original(
    predicted_k: Any,
    input_hw: tuple[int, int],
    original_hw: tuple[int, int],
) -> list[list[float]] | None:
    if predicted_k is None:
        return None

    import torch

    if isinstance(predicted_k, torch.Tensor):
        predicted_k = predicted_k.detach().cpu().clone()
    else:
        predicted_k = torch.tensor(predicted_k, dtype=torch.float32)

    if predicted_k.ndim == 3:
        predicted_k = predicted_k[0]

    input_h, input_w = input_hw
    orig_h, orig_w = original_hw
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    predicted_k[0, 0] *= scale_x
    predicted_k[1, 1] *= scale_y
    predicted_k[0, 2] *= scale_x
    predicted_k[1, 2] *= scale_y
    return predicted_k.tolist()


def _to_intrinsics_list(intrinsics: Any) -> list[list[float]] | None:
    if intrinsics is None:
        return None

    import torch

    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.detach().cpu().clone()
    else:
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

    if intrinsics.ndim == 3:
        intrinsics = intrinsics[0]

    return intrinsics.tolist()


def _resolve_effective_intrinsics(
    *,
    provided_intrinsics: list[list[float]] | None,
    predicted_intrinsics: list[list[float]] | None,
    placeholder_intrinsics: list[list[float]] | None,
    use_predicted_intrinsics: bool,
) -> tuple[list[list[float]] | None, str]:
    if use_predicted_intrinsics and predicted_intrinsics is not None:
        return predicted_intrinsics, "predicted"
    if provided_intrinsics is not None:
        return provided_intrinsics, "provided"
    if predicted_intrinsics is not None:
        return predicted_intrinsics, "predicted"
    if placeholder_intrinsics is not None:
        return placeholder_intrinsics, "placeholder"
    return None, "none"


def _cross_category_nms(
    boxes2d: Any,
    boxes3d: Any,
    scores: Any,
    scores_2d: Any,
    scores_3d: Any,
    class_ids: Any,
    iou_threshold: float = 0.8,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    import torch

    if len(boxes2d) <= 1:
        return boxes2d, boxes3d, scores, scores_2d, scores_3d, class_ids

    order = scores.argsort(descending=True)
    boxes2d = boxes2d[order]
    boxes3d = boxes3d[order]
    scores = scores[order]
    scores_2d = scores_2d[order]
    scores_3d = scores_3d[order]
    class_ids = class_ids[order]

    x1 = torch.max(boxes2d[:, None, 0], boxes2d[None, :, 0])
    y1 = torch.max(boxes2d[:, None, 1], boxes2d[None, :, 1])
    x2 = torch.min(boxes2d[:, None, 2], boxes2d[None, :, 2])
    y2 = torch.min(boxes2d[:, None, 3], boxes2d[None, :, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area = (
        (boxes2d[:, 2] - boxes2d[:, 0]) * (boxes2d[:, 3] - boxes2d[:, 1])
    )
    union = area[:, None] + area[None, :] - inter
    iou = inter / (union + 1e-6)

    suppressed: set[int] = set()
    keep: list[int] = []
    for i in range(len(boxes2d)):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i + 1, len(boxes2d)):
            if j in suppressed:
                continue
            if iou[i, j] >= iou_threshold:
                suppressed.add(j)

    keep_tensor = torch.tensor(keep, dtype=torch.long, device=boxes2d.device)
    return (
        boxes2d[keep_tensor],
        boxes3d[keep_tensor],
        scores[keep_tensor],
        scores_2d[keep_tensor],
        scores_3d[keep_tensor],
        class_ids[keep_tensor],
    )


@contextmanager
def _temporary_env_var(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={
        MODEL_VOLUME_PATH: model_volume,
        HF_CACHE_VOLUME_PATH: hf_cache_volume,
    },
)
def cache_model_artifacts(
    checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
) -> dict[str, str]:
    """Populate the checkpoint volume before the API is deployed."""
    import shutil

    from huggingface_hub import hf_hub_download

    os.makedirs(MODEL_VOLUME_PATH, exist_ok=True)

    cached_file = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=checkpoint_file,
        cache_dir=HF_CACHE_VOLUME_PATH,
    )
    target_path = Path(MODEL_VOLUME_PATH) / checkpoint_file
    if Path(cached_file).resolve() != target_path.resolve():
        shutil.copy2(cached_file, target_path)

    model_volume.commit()
    hf_cache_volume.commit()
    return {
        "checkpoint_file": checkpoint_file,
        "checkpoint_path": str(target_path),
    }


@app.local_entrypoint()
def prepare(
    checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
) -> None:
    result = cache_model_artifacts.remote(checkpoint_file=checkpoint_file)
    print(json.dumps(result, indent=2))


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    min_containers=MIN_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=60 * 20,
    volumes={
        MODEL_VOLUME_PATH: model_volume,
        HF_CACHE_VOLUME_PATH: hf_cache_volume,
    },
)
class WildDet3DAPI:
    @modal.enter()
    def load(self) -> None:
        import threading

        import torch

        sys_path = os.sys.path
        if REMOTE_REPO_ROOT not in sys_path:
            sys_path.insert(0, REMOTE_REPO_ROOT)

        model_volume.reload()
        hf_cache_volume.reload()

        checkpoint_path = Path(MODEL_VOLUME_PATH) / DEFAULT_CHECKPOINT_FILE
        if not checkpoint_path.exists():
            raise RuntimeError(
                "Checkpoint not found in the model volume. "
                "Run `modal run deploy/modal_app.py` first or copy "
                f"`{DEFAULT_CHECKPOINT_FILE}` into the `{MODEL_VOLUME_NAME}` "
                "volume."
            )

        from wilddet3d import build_model, preprocess

        self._checkpoint_path = str(checkpoint_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._preprocess = preprocess
        self._model = build_model(
            checkpoint=self._checkpoint_path,
            score_threshold=0.0,
            canonical_rotation=True,
            skip_pretrained=True,
            device=self._device,
        )
        self._lock = threading.Lock()

    def _decode_image(self, image_bytes: bytes) -> Any:
        import numpy as np
        from PIL import Image

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise RequestValidationError(
                f"Uploaded image exceeds {MAX_IMAGE_BYTES} bytes."
            )

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:  # pragma: no cover - pillow raises many types
            raise RequestValidationError("Failed to decode image bytes.") from exc
        return np.array(image).astype(np.float32)

    def _run_inference(self, request: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        import torch

        image = self._decode_image(request["image_bytes"])
        intrinsics = request["intrinsics"]
        intrinsics_np = (
            np.array(intrinsics, dtype=np.float32) if intrinsics is not None else None
        )
        data = self._preprocess(image, intrinsics_np)

        prompt = request["prompt"]
        use_predicted_intrinsics = request["use_predicted_intrinsics"]
        score_threshold = request["score_threshold"]

        with self._lock:
            with _temporary_env_var(
                "SAM3_USE_PRED_K",
                "1" if use_predicted_intrinsics else "0",
            ):
                if prompt["type"] == "text":
                    class_names = prompt["texts"]
                    results = self._model(
                        images=data["images"].to(self._device),
                        intrinsics=data["intrinsics"].to(self._device)[None],
                        input_hw=[data["input_hw"]],
                        original_hw=[data["original_hw"]],
                        padding=[data["padding"]],
                        input_texts=class_names,
                        return_predicted_intrinsics=True,
                    )
                elif prompt["type"] == "box":
                    x1_orig, y1_orig, x2_orig, y2_orig = prompt["box"]
                    x1, y1 = _transform_coords_to_input_space(
                        x1_orig,
                        y1_orig,
                        data["original_hw"],
                        data["input_hw"],
                        data["padding"],
                    )
                    x2, y2 = _transform_coords_to_input_space(
                        x2_orig,
                        y2_orig,
                        data["original_hw"],
                        data["input_hw"],
                        data["padding"],
                    )
                    prompt_text = _build_prompt_text(
                        prompt["mode"],
                        prompt["label_text"],
                    )
                    class_names = [prompt_text]
                    results = self._model(
                        images=data["images"].to(self._device),
                        intrinsics=data["intrinsics"].to(self._device)[None],
                        input_hw=[data["input_hw"]],
                        original_hw=[data["original_hw"]],
                        padding=[data["padding"]],
                        input_boxes=[[float(x1), float(y1), float(x2), float(y2)]],
                        prompt_text=prompt_text,
                        return_predicted_intrinsics=True,
                    )
                else:
                    transformed_points: list[tuple[float, float, int]] = []
                    for x_orig, y_orig, label in prompt["points"]:
                        x, y = _transform_coords_to_input_space(
                            x_orig,
                            y_orig,
                            data["original_hw"],
                            data["input_hw"],
                            data["padding"],
                        )
                        transformed_points.append((x, y, label))

                    prompt_text = _build_prompt_text(
                        prompt["mode"],
                        prompt["label_text"],
                    )
                    class_names = [prompt_text]
                    results = self._model(
                        images=data["images"].to(self._device),
                        intrinsics=data["intrinsics"].to(self._device)[None],
                        input_hw=[data["input_hw"]],
                        original_hw=[data["original_hw"]],
                        padding=[data["padding"]],
                        input_points=[transformed_points],
                        prompt_text=prompt_text,
                        return_predicted_intrinsics=True,
                    )

        (
            boxes,
            boxes3d,
            scores,
            scores_2d,
            scores_3d,
            class_ids,
            depth_maps,
            predicted_k,
            confidence_maps,
        ) = results

        if len(boxes[0]) > 1:
            (
                boxes[0],
                boxes3d[0],
                scores[0],
                scores_2d[0],
                scores_3d[0],
                class_ids[0],
            ) = _cross_category_nms(
                boxes[0],
                boxes3d[0],
                scores[0],
                scores_2d[0],
                scores_3d[0],
                class_ids[0],
            )

        is_single_target = prompt["type"] == "point" or (
            prompt["type"] == "box" and prompt["mode"] == "geometric"
        )
        if is_single_target and len(boxes[0]) > 1:
            best = scores[0].argmax()
            boxes[0] = boxes[0][best : best + 1]
            boxes3d[0] = boxes3d[0][best : best + 1]
            scores[0] = scores[0][best : best + 1]
            scores_2d[0] = scores_2d[0][best : best + 1]
            scores_3d[0] = scores_3d[0][best : best + 1]
            class_ids[0] = class_ids[0][best : best + 1]

        detections: list[dict[str, Any]] = []
        score_mask = scores[0] >= score_threshold
        selected_indices = score_mask.nonzero(as_tuple=False).flatten().tolist()
        for idx in selected_indices:
            class_id = int(class_ids[0][idx].item())
            class_name = (
                class_names[class_id]
                if 0 <= class_id < len(class_names)
                else str(class_id)
            )
            detections.append(
                {
                    "box2d": boxes[0][idx].detach().cpu().tolist(),
                    "box3d": boxes3d[0][idx].detach().cpu().tolist(),
                    "score": float(scores[0][idx].item()),
                    "score_2d": float(scores_2d[0][idx].item()),
                    "score_3d": float(scores_3d[0][idx].item()),
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )

        predicted_intrinsics = _scale_intrinsics_to_original(
            predicted_k,
            input_hw=data["input_hw"],
            original_hw=data["original_hw"],
        )
        placeholder_intrinsics = _to_intrinsics_list(
            data["original_intrinsics"]
        )
        (
            effective_intrinsics,
            effective_intrinsics_source,
        ) = _resolve_effective_intrinsics(
            provided_intrinsics=intrinsics,
            predicted_intrinsics=predicted_intrinsics,
            placeholder_intrinsics=placeholder_intrinsics,
            use_predicted_intrinsics=use_predicted_intrinsics,
        )

        depth_summary = None
        if depth_maps is not None and len(depth_maps) > 0:
            depth_map = depth_maps[0].detach().cpu()
            valid_depth = depth_map[depth_map > 0.01]
            depth_summary = {
                "shape": list(depth_map.shape),
                "min": (
                    float(valid_depth.min().item()) if valid_depth.numel() > 0 else None
                ),
                "max": (
                    float(valid_depth.max().item()) if valid_depth.numel() > 0 else None
                ),
            }

        confidence_summary = None
        if confidence_maps is not None and len(confidence_maps) > 0:
            confidence_map = confidence_maps[0].detach().cpu()
            confidence_summary = {
                "shape": list(confidence_map.shape),
                "mean": float(confidence_map.float().mean().item()),
            }

        return {
            "detections": detections,
            "num_detections": len(detections),
            "prompt": prompt,
            "image_size": {
                "height": int(data["original_hw"][0]),
                "width": int(data["original_hw"][1]),
            },
            "score_threshold": score_threshold,
            "use_predicted_intrinsics": use_predicted_intrinsics,
            "provided_intrinsics": intrinsics,
            "predicted_intrinsics": predicted_intrinsics,
            "effective_intrinsics": effective_intrinsics,
            "effective_intrinsics_source": effective_intrinsics_source,
            "depth_summary": depth_summary,
            "confidence_summary": confidence_summary,
            "model_device": self._device,
        }

    @modal.asgi_app(
        label=API_LABEL,
        requires_proxy_auth=REQUIRES_PROXY_AUTH,
    )
    def web(self):
        web_app = FastAPI(
            title="WildDet3D API",
            version="0.1.0",
            description=(
                "Modal deployment for WildDet3D promptable 3D object detection."
            ),
        )

        @web_app.get("/")
        def root() -> dict[str, Any]:
            return {
                "service": "wilddet3d",
                "status": "ok",
                "docs": "/docs",
                "healthz": "/healthz",
                "infer": "/infer",
            }

        @web_app.get("/healthz")
        def healthz() -> dict[str, Any]:
            return {
                "status": "ok",
                "device": self._device,
                "checkpoint_file": Path(self._checkpoint_path).name,
                "checkpoint_path": self._checkpoint_path,
                "model_volume": MODEL_VOLUME_NAME,
                "hf_cache_volume": HF_CACHE_VOLUME_NAME,
            }

        @web_app.post("/infer")
        async def infer(
            image: UploadFile = File(...),
            request_json: str = Form("{}"),
        ) -> dict[str, Any]:
            try:
                image_bytes = await image.read()
                payload = json.loads(request_json)

                normalized_request = _normalize_request(
                    payload,
                    image_bytes=image_bytes,
                )
                return self._run_inference(normalized_request)
            except RequestValidationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="`request_json` must be valid JSON.",
                ) from exc
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        return web_app

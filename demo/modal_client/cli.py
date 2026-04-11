from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv()


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from demo.modal_client.client import WildDet3DModalClient
    from demo.modal_client.render import render_response, save_rendered_image
    from demo.modal_client.types import (
        BoxPrompt,
        InferRequest,
        PointPrompt,
        TextPrompt,
    )
else:
    from .client import WildDet3DModalClient
    from .render import render_response, save_rendered_image
    from .types import BoxPrompt, InferRequest, PointPrompt, TextPrompt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WildDet3D Modal API client",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("WILDDET3D_URL"),
        help="Base URL of the deployed WildDet3D API.",
    )
    parser.add_argument(
        "--image",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--healthz-only",
        action="store_true",
        help="Only call /healthz and print the response.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Call /healthz before inference to trigger cold start separately.",
    )
    parser.add_argument(
        "--output-image",
        help="Path to save the rendered visualization.",
    )
    parser.add_argument(
        "--output-json",
        help="Path to save the raw JSON response.",
    )
    parser.add_argument(
        "--modal-key",
        default=os.environ.get("MODAL_KEY"),
        help="Modal proxy auth key.",
    )
    parser.add_argument(
        "--modal-secret",
        default=os.environ.get("MODAL_SECRET"),
        help="Modal proxy auth secret.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("WILDDET3D_CLIENT_TIMEOUT", "600")),
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--prompt-type",
        choices=["text", "box", "point"],
        default="text",
        help="Prompt type to send.",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        help="Text prompt categories for --prompt-type text.",
    )
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Box prompt coordinates in original image space.",
    )
    parser.add_argument(
        "--point",
        dest="points",
        action="append",
        nargs=3,
        metavar=("X", "Y", "LABEL"),
        help="Point prompt as x y label. Repeat for multiple points.",
    )
    parser.add_argument(
        "--mode",
        choices=["visual", "geometric"],
        help="Prompt mode for box or point prompts.",
    )
    parser.add_argument(
        "--label-text",
        help="Optional label text for box or point prompts.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Score threshold applied on the server.",
    )
    parser.add_argument(
        "--intrinsics",
        nargs=4,
        type=float,
        metavar=("FX", "FY", "CX", "CY"),
        help="Optional camera intrinsics as fx fy cx cy.",
    )
    parser.add_argument(
        "--use-predicted-intrinsics",
        dest="use_predicted_intrinsics",
        action="store_true",
        help="Force the server to use predicted intrinsics.",
    )
    parser.add_argument(
        "--no-use-predicted-intrinsics",
        dest="use_predicted_intrinsics",
        action="store_false",
        help="Force the server to use provided intrinsics (requires --intrinsics).",
    )
    parser.set_defaults(use_predicted_intrinsics=None)
    parser.add_argument(
        "--skip-2d",
        action="store_true",
        help="Do not draw 2D boxes.",
    )
    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="Do not draw 3D boxes.",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Do not draw labels.",
    )
    parser.add_argument(
        "--skip-prompt",
        action="store_true",
        help="Do not draw the input prompt overlay.",
    )
    parser.add_argument(
        "--prefer-provided-intrinsics",
        action="store_true",
        help="Prefer provided intrinsics for 3D rendering when both are present.",
    )
    return parser


def _build_intrinsics(args: argparse.Namespace) -> list[list[float]] | None:
    if args.intrinsics is None:
        return None
    fx, fy, cx, cy = args.intrinsics
    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def _build_prompt(args: argparse.Namespace) -> TextPrompt | BoxPrompt | PointPrompt:
    if args.prompt_type == "text":
        texts = args.texts or ["object"]
        return TextPrompt(texts=list(texts))
    if args.prompt_type == "box":
        if args.box is None:
            raise ValueError("--box is required for --prompt-type box.")
        mode = args.mode or "visual"
        return BoxPrompt(
            box=tuple(float(value) for value in args.box),
            mode=mode,
            label_text=args.label_text,
        )
    if not args.points:
        raise ValueError("--point is required for --prompt-type point.")
    mode = args.mode or "geometric"
    points = [
        (float(x), float(y), int(label))
        for x, y, label in args.points
    ]
    return PointPrompt(
        points=points,
        mode=mode,
        label_text=args.label_text,
    )


def _default_output_image_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}.wilddet3d.jpg")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.url:
        parser.error("--url is required or set WILDDET3D_URL.")
    if not args.healthz_only and not args.image:
        parser.error("--image is required unless --healthz-only is set.")

    client = WildDet3DModalClient(
        args.url,
        modal_key=args.modal_key,
        modal_secret=args.modal_secret,
        timeout=args.timeout,
    )
    try:
        if args.healthz_only:
            print(json.dumps(client.healthz(), indent=2, ensure_ascii=False))
            return 0

        image_path = Path(args.image)
        try:
            prompt = _build_prompt(args)
        except ValueError as exc:
            parser.error(str(exc))
        infer_request = InferRequest(
            prompt=prompt,
            score_threshold=args.score_threshold,
            intrinsics=_build_intrinsics(args),
            use_predicted_intrinsics=args.use_predicted_intrinsics,
        )

        if args.warmup:
            warmup_response = client.healthz()
            print(
                "Warmup complete:",
                json.dumps(warmup_response, ensure_ascii=False),
            )

        response = client.infer(image_path, infer_request)
    except requests.Timeout as exc:
        parser.exit(2, f"{exc}\n")

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(response.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    output_image_path = Path(args.output_image) if args.output_image else _default_output_image_path(image_path)
    rendered = render_response(
        image_path,
        response,
        draw_2d=not args.skip_2d,
        draw_3d=not args.skip_3d,
        draw_labels=not args.skip_labels,
        draw_prompt=not args.skip_prompt,
        prefer_predicted_intrinsics=not args.prefer_provided_intrinsics,
    )
    save_rendered_image(output_image_path, rendered)

    print(f"Detections: {response.num_detections}")
    print(f"Rendered image: {output_image_path}")
    if args.output_json:
        print(f"Response JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

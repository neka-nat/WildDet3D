# Modal Deployment

This repository now includes a Modal deployment entrypoint at `deploy/modal_app.py`.

## What It Provides

- GPU-backed FastAPI endpoint on Modal
- Checkpoint storage in a dedicated Modal Volume
- Separate Hugging Face cache Volume for LingBot and other downloaded artifacts
- A single `/infer` endpoint that accepts multipart uploads
- A `/healthz` endpoint for basic health checks

## Prerequisites

1. Initialize submodules:

```bash
git submodule update --init --recursive
```

2. Install and authenticate Modal locally.

## Prepare Model Weights

Populate the checkpoint Volume before deploying:

```bash
modal run deploy/modal_app.py
```

By default this downloads:

- `allenai/WildDet3D`
- `wilddet3d_alldata_all_prompt_v1.0.pt`

The checkpoint is stored in the `wilddet3d-models` Volume by default.

## Deploy

```bash
modal deploy deploy/modal_app.py
```

The deployed app exposes:

- `GET /`
- `GET /healthz`
- `POST /infer`
- `GET /docs`

Proxy auth is enabled by default. If you want a public endpoint, set:

```bash
export WILDDET3D_REQUIRES_PROXY_AUTH=0
```

before deploying.

## Python Client

A local Python client and renderer are available in:

- `demo/modal_client/`

See [demo/modal_client/README.md](../demo/modal_client/README.md) for CLI and Python examples.

## Request Format

`POST /infer` accepts multipart form data.

- `image`: binary image file
- `request_json`: JSON string with the inference payload

Example:

```bash
curl -X POST "$WILDDET3D_URL/infer" \
  -H "Modal-Key: $MODAL_KEY" \
  -H "Modal-Secret: $MODAL_SECRET" \
  -F 'image=@example.jpg' \
  -F 'request_json={
    "prompt": {
      "type": "box",
      "mode": "visual",
      "box": [100, 180, 320, 420],
      "label_text": "car"
    },
    "score_threshold": 0.25
  }'
```

## Prompt Shapes

### Text prompt

```json
{
  "prompt": {
    "type": "text",
    "texts": ["car", "bicycle"]
  }
}
```

### Box prompt

```json
{
  "prompt": {
    "type": "box",
    "mode": "visual",
    "box": [100, 180, 320, 420],
    "label_text": "car"
  }
}
```

`mode` can be:

- `visual`
- `geometric`

### Point prompt

```json
{
  "prompt": {
    "type": "point",
    "mode": "geometric",
    "points": [
      {"x": 150, "y": 240, "label": 1},
      {"x": 220, "y": 300, "label": 0}
    ]
  }
}
```

## Environment Variables

- `WILDDET3D_MODAL_APP_NAME`
- `WILDDET3D_MODAL_LABEL`
- `WILDDET3D_MODAL_GPU`
- `WILDDET3D_MODAL_MIN_CONTAINERS`
- `WILDDET3D_MODAL_SCALEDOWN_WINDOW`
- `WILDDET3D_MODEL_VOLUME_NAME`
- `WILDDET3D_HF_CACHE_VOLUME_NAME`
- `WILDDET3D_CHECKPOINT_FILE`
- `WILDDET3D_REQUIRES_PROXY_AUTH`

## Notes

- The deployment assumes the repo source and both git submodules are available locally at deploy time.
- The first cold start may still populate the Hugging Face cache Volume for LingBot-related assets.
- The first request after deploy or idle scale-down can take a few minutes. If you want to avoid cold starts, deploy with `WILDDET3D_MODAL_MIN_CONTAINERS=1`. `WILDDET3D_MODAL_SCALEDOWN_WINDOW` defaults to 900 seconds.
- The current API returns structured detections and intrinsics metadata, not visualization images.

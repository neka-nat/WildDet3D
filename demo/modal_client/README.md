# WildDet3D Modal Client

Python client and local visualization utilities for the Modal deployment in [docs/MODAL.md](../../docs/MODAL.md).

## Install

```bash
pip install -r demo/modal_client/requirements.txt
```

## CLI Usage

Set the API URL and optional Modal proxy auth credentials:

```bash
export WILDDET3D_URL="https://your-modal-endpoint.modal.run"
export MODAL_KEY="..."
export MODAL_SECRET="..."
```

### Text prompt

```bash
python -m demo.modal_client.cli \
  --image image.jpg \
  --prompt-type text \
  --texts car person bicycle
```

The first request can take several minutes after deploy or idle scale-down.
The CLI defaults to a 600 second timeout. You can also warm the endpoint first:

```bash
python -m demo.modal_client.cli --healthz-only
python -m demo.modal_client.cli \
  --image image.jpg \
  --prompt-type text \
  --texts car \
  --warmup
```

### Box prompt

```bash
python -m demo.modal_client.cli \
  --image image.jpg \
  --prompt-type box \
  --mode visual \
  --box 100 180 320 420 \
  --label-text car
```

### Point prompt

```bash
python -m demo.modal_client.cli \
  --image image.jpg \
  --prompt-type point \
  --mode geometric \
  --point 150 240 1 \
  --point 220 300 0
```

The CLI saves a rendered image next to the input by default:

- `image.jpg` -> `image.wilddet3d.jpg`

You can also save the raw response:

```bash
python -m demo.modal_client.cli \
  --image image.jpg \
  --prompt-type text \
  --texts chair table \
  --output-json outputs/result.json \
  --output-image outputs/result.jpg
```

## Python API

```python
from demo.modal_client.client import WildDet3DModalClient
from demo.modal_client.render import render_response, save_rendered_image
from demo.modal_client.types import InferRequest, TextPrompt

client = WildDet3DModalClient.from_env()
request = InferRequest(
    prompt=TextPrompt(["car", "person"]),
    score_threshold=0.3,
)
response = client.infer("image.jpg", request)

rendered = render_response("image.jpg", response)
save_rendered_image("image.wilddet3d.jpg", rendered)
```

## Rendering Notes

- 2D boxes are drawn with OpenCV.
- 3D wireframes are projected locally using the intrinsics returned by the API.
- Labels are drawn with Pillow on top of the OpenCV output.
- If no renderable intrinsics are available, the renderer falls back to 2D-only overlays.

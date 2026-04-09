from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

from .types import InferRequest, InferResponse


DEFAULT_TIMEOUT = float(os.environ.get("WILDDET3D_CLIENT_TIMEOUT", "600"))


class WildDet3DModalClient:
    def __init__(
        self,
        base_url: str,
        *,
        modal_key: str | None = None,
        modal_secret: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("Accept", "application/json")
        self.session.headers.setdefault("User-Agent", "wilddet3d-modal-client/0.1")

        if modal_key:
            self.session.headers["Modal-Key"] = modal_key
        if modal_secret:
            self.session.headers["Modal-Secret"] = modal_secret

    @classmethod
    def from_env(
        cls,
        *,
        base_url_env: str = "WILDDET3D_URL",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> WildDet3DModalClient:
        base_url = os.environ.get(base_url_env)
        if not base_url:
            raise ValueError(
                f"Environment variable `{base_url_env}` is not set."
            )
        return cls(
            base_url,
            modal_key=os.environ.get("MODAL_KEY"),
            modal_secret=os.environ.get("MODAL_SECRET"),
            timeout=timeout,
        )

    def _raise_for_response(self, response: requests.Response) -> None:
        if response.ok:
            return

        detail = None
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail")

        if detail:
            raise requests.HTTPError(
                f"{response.status_code} {response.reason}: {detail}",
                response=response,
            )
        response.raise_for_status()

    def healthz(self) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}/healthz",
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise requests.Timeout(
                "Health check timed out. The Modal container may still be cold "
                "starting. Retry with a larger timeout."
            ) from exc
        self._raise_for_response(response)
        return response.json()

    def infer(
        self,
        image_path: str | Path,
        infer_request: InferRequest,
    ) -> InferResponse:
        image_path = Path(image_path)
        with image_path.open("rb") as handle:
            image_bytes = handle.read()
        return self.infer_bytes(
            image_bytes,
            infer_request,
            filename=image_path.name,
        )

    def infer_bytes(
        self,
        image_bytes: bytes,
        infer_request: InferRequest,
        *,
        filename: str = "image.jpg",
    ) -> InferResponse:
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        payload = infer_request.to_payload()
        files = {
            "image": (filename, image_bytes, mime_type),
        }
        data = {
            "request_json": json.dumps(payload),
        }
        try:
            response = self.session.post(
                f"{self.base_url}/infer",
                data=data,
                files=files,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise requests.Timeout(
                "Inference request timed out. The first request after deploy or "
                "idle scale-down can take several minutes while Modal starts a "
                "GPU container and loads WildDet3D. Retry with a larger timeout "
                "or warm the endpoint with /healthz first."
            ) from exc
        self._raise_for_response(response)
        return InferResponse.from_dict(response.json())

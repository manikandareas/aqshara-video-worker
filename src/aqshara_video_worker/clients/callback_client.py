from __future__ import annotations

import hashlib
import json

import httpx

from aqshara_video_worker.config import WorkerSettings
from aqshara_video_worker.schemas import (
    InternalVideoComplete,
    InternalVideoFail,
    InternalVideoProgress,
)


class CallbackClient:
    def __init__(
        self,
        settings: WorkerSettings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=settings.callback_base_url,
            timeout=httpx.Timeout(10.0, connect=5.0),
            headers={
                "x-internal-service-token": settings.internal_service_token,
            },
        )

    async def send_progress(
        self,
        job_id: str,
        payload: InternalVideoProgress,
    ) -> None:
        await self._post(f"/internal/video-jobs/{job_id}/progress", payload)

    async def send_complete(
        self,
        job_id: str,
        payload: InternalVideoComplete,
    ) -> None:
        await self._post(f"/internal/video-jobs/{job_id}/complete", payload)

    async def send_fail(
        self,
        job_id: str,
        payload: InternalVideoFail,
    ) -> None:
        await self._post(f"/internal/video-jobs/{job_id}/fail", payload)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, path: str, payload: object) -> None:
        body = self._dump(payload)
        response = await self._client.post(
            path,
            json=body,
            headers={
                "x-idempotency-key": self._build_idempotency_key(path, body),
            },
        )
        response.raise_for_status()

    @staticmethod
    def _dump(payload: object) -> dict:
        if hasattr(payload, "model_dump"):
            return payload.model_dump(exclude_none=True)  # type: ignore[return-value]
        raise TypeError("Callback payload must be a Pydantic model")

    @staticmethod
    def _build_idempotency_key(path: str, body: dict) -> str:
        canonical = json.dumps(
            {
                "path": path,
                "body": body,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

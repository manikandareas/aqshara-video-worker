from __future__ import annotations

import httpx
import pytest

from aqshara_video_worker.clients.callback_client import CallbackClient
from aqshara_video_worker.config import WorkerSettings
from aqshara_video_worker.schemas import InternalVideoProgress


@pytest.mark.asyncio
async def test_callback_client_preserves_api_v1_base_path() -> None:
    captured_request: httpx.Request | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_request
        captured_request = request
        return httpx.Response(status_code=204)

    transport = httpx.MockTransport(handler)
    client = CallbackClient(
        WorkerSettings(
            VIDEO_WORKER_CALLBACK_BASE_URL="http://127.0.0.1:3000/api/v1/",
            VIDEO_INTERNAL_SERVICE_TOKEN="token_1",
            R2_ENDPOINT="https://example.r2.cloudflarestorage.com",
            R2_ACCESS_KEY_ID="key",
            R2_SECRET_ACCESS_KEY="secret",
            R2_BUCKET="bucket",
        ),
        client=httpx.AsyncClient(
            base_url="http://127.0.0.1:3000/api/v1/",
            headers={"x-internal-service-token": "token_1"},
            transport=transport,
        ),
    )

    try:
        await client.send_progress(
            "vjob_1",
            InternalVideoProgress(
                pipeline_stage="preprocessing",
                progress_pct=10,
            ),
        )
    finally:
        await client.close()

    assert captured_request is not None
    assert (
        str(captured_request.url)
        == "http://127.0.0.1:3000/api/v1/internal/video-jobs/vjob_1/progress"
    )
    assert captured_request.headers["x-idempotency-key"]

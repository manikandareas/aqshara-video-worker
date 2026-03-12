from __future__ import annotations

from typing import Protocol

from aqshara_video_worker.schemas import (
    InternalVideoComplete,
    InternalVideoFail,
    InternalVideoProgress,
)


class VideoEventPublisher(Protocol):
    async def send_progress(
        self,
        job_id: str,
        payload: InternalVideoProgress,
    ) -> None: ...

    async def send_complete(
        self,
        job_id: str,
        payload: InternalVideoComplete,
    ) -> None: ...

    async def send_fail(
        self,
        job_id: str,
        payload: InternalVideoFail,
    ) -> None: ...

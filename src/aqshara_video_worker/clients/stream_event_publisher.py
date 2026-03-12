from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from aqshara_video_worker.schemas import (
    InternalVideoComplete,
    InternalVideoFail,
    InternalVideoProgress,
    VideoGenerateCommand,
)
from aqshara_video_worker.transport.redis_streams import RedisVideoTransport


class RedisStreamEventPublisher:
    def __init__(
        self,
        transport: RedisVideoTransport,
        command: VideoGenerateCommand,
        worker_id: str,
    ) -> None:
        self._transport = transport
        self._command = command
        self._worker_id = worker_id

    async def send_accepted(self) -> None:
        await self._publish("job.accepted")

    async def send_heartbeat(self) -> None:
        await self._publish("job.heartbeat")

    async def send_progress(
        self,
        job_id: str,
        payload: InternalVideoProgress,
    ) -> None:
        event_type = "scene.progress" if payload.scene is not None else "job.progress"
        await self._publish_model(event_type, job_id, payload)

    async def send_complete(
        self,
        job_id: str,
        payload: InternalVideoComplete,
    ) -> None:
        await self._publish_model("job.completed", job_id, payload)

    async def send_fail(
        self,
        job_id: str,
        payload: InternalVideoFail,
    ) -> None:
        await self._publish_model("job.failed", job_id, payload)

    async def _publish_model(
        self,
        event_type: str,
        job_id: str,
        payload: InternalVideoProgress | InternalVideoComplete | InternalVideoFail,
    ) -> None:
        await self._publish(
            event_type,
            job_id=job_id,
            payload=payload.model_dump(exclude_none=True),
        )

    async def _publish(
        self,
        event_type: str,
        *,
        job_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        event = self._build_event(event_type=event_type, job_id=job_id)
        if payload is not None:
            event["payload"] = payload

        await self._transport.publish_event(event)

    def _build_event(
        self,
        *,
        event_type: str,
        job_id: str | None = None,
    ) -> dict[str, object]:
        return {
            "schema_version": self._command.schema_version,
            "event_id": str(uuid4()),
            "event_type": event_type,
            "job_id": job_id or self._command.job_id,
            "attempt": self._command.attempt,
            "worker_id": self._worker_id,
            "occurred_at": datetime.now(UTC).isoformat(),
        }

from __future__ import annotations

import json

from redis.asyncio import Redis

from aqshara_video_worker.config import WorkerSettings
from aqshara_video_worker.schemas import VideoGenerateCommand


class RedisVideoTransport:
    def __init__(
        self,
        settings: WorkerSettings,
        client: Redis | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or Redis.from_url(settings.redis_url, decode_responses=True)

    async def ensure_consumer_group(self) -> None:
        try:
            await self._client.xgroup_create(
                self._settings.video_command_stream_name,
                self._settings.video_worker_consumer_group,
                id="$",
                mkstream=True,
            )
        except Exception as error:
            if not self._is_existing_group_error(error):
                raise

    async def read_commands(self) -> list[tuple[str, VideoGenerateCommand]]:
        response = await self._client.xreadgroup(
            groupname=self._settings.video_worker_consumer_group,
            consumername=self._settings.video_worker_consumer_name,
            streams={self._settings.video_command_stream_name: ">"},
            count=self._settings.video_stream_batch_size,
            block=self._settings.video_stream_block_ms,
        )

        if not response:
            return []

        entries: list[tuple[str, VideoGenerateCommand]] = []
        for _stream_name, stream_entries in response:
            for stream_id, fields in stream_entries:
                payload = fields.get("payload", "{}")
                entries.append(
                    (
                        stream_id,
                        VideoGenerateCommand.model_validate_json(payload),
                    )
                )

        return entries

    async def acknowledge_command(self, stream_id: str) -> None:
        await self._client.xack(
            self._settings.video_command_stream_name,
            self._settings.video_worker_consumer_group,
            stream_id,
        )

    async def publish_event(self, payload: dict) -> str:
        stream_id = await self._client.xadd(
            self._settings.video_event_stream_name,
            {"payload": json.dumps(payload, separators=(",", ":"))},
        )
        if stream_id is None:
            raise RuntimeError("Failed to publish video worker event")
        return stream_id

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _is_existing_group_error(error: Exception) -> bool:
        return "BUSYGROUP" in str(error)

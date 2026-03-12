from __future__ import annotations

import json

import pytest

from aqshara_video_worker.clients.stream_event_publisher import (
    RedisStreamEventPublisher,
)
from aqshara_video_worker.schemas import (
    InternalVideoComplete,
    InternalVideoProgress,
    InternalVideoSceneProgress,
    VideoGenerateCommand,
)


class RecordingTransport:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def publish_event(self, payload: dict) -> str:
        self.events.append(payload)
        return "1-0"


@pytest.mark.asyncio
async def test_stream_event_publisher_emits_job_progress_without_scene() -> None:
    transport = RecordingTransport()
    publisher = RedisStreamEventPublisher(
        transport=transport,  # type: ignore[arg-type]
        command=VideoGenerateCommand(
            schema_version="2026-03-11",
            command_id="cmd_1",
            topic="video.generate.command",
            job_id="vjob_1",
            document_id="doc_1",
            owner_id="user_1",
            attempt=1,
            target_duration_sec=60,
            voice="alloy",
            language="en",
            render_profile="720p",
            ocr_object_key="documents/doc_1/artifacts/ocr/raw.json",
            output_prefix="videos/vjob_1",
            correlation_id="corr_1",
            occurred_at="2026-03-11T00:00:00Z",
        ),
        worker_id="worker-1",
    )

    await publisher.send_progress(
        "vjob_1",
        InternalVideoProgress(
            pipeline_stage="summarizing",
            progress_pct=20,
        ),
    )

    assert transport.events[0]["event_type"] == "job.progress"
    assert transport.events[0]["job_id"] == "vjob_1"
    assert transport.events[0]["payload"]["pipeline_stage"] == "summarizing"


@pytest.mark.asyncio
async def test_stream_event_publisher_emits_scene_progress_with_scene_payload() -> None:
    transport = RecordingTransport()
    publisher = RedisStreamEventPublisher(
        transport=transport,  # type: ignore[arg-type]
        command=VideoGenerateCommand(
            schema_version="2026-03-11",
            command_id="cmd_2",
            topic="video.generate.command",
            job_id="vjob_2",
            document_id="doc_2",
            owner_id="user_2",
            attempt=2,
            target_duration_sec=75,
            voice="nova",
            language="id",
            render_profile="720p",
            ocr_object_key="documents/doc_2/artifacts/ocr/raw.json",
            output_prefix="videos/vjob_2",
            correlation_id="corr_2",
            occurred_at="2026-03-11T00:00:00Z",
        ),
        worker_id="worker-2",
    )

    await publisher.send_progress(
        "vjob_2",
        InternalVideoProgress(
            pipeline_stage="rendering",
            progress_pct=95,
            scene=InternalVideoSceneProgress(
                scene_index=3,
                status="done",
                template_type="bullet",
                video_object_key="videos/vjob_2/artifacts/render/scene-03.mp4",
            ),
        ),
    )

    assert transport.events[0]["event_type"] == "scene.progress"
    assert transport.events[0]["attempt"] == 2
    assert transport.events[0]["payload"]["scene"]["scene_index"] == 3


@pytest.mark.asyncio
async def test_stream_event_publisher_emits_complete_payload() -> None:
    transport = RecordingTransport()
    publisher = RedisStreamEventPublisher(
        transport=transport,  # type: ignore[arg-type]
        command=VideoGenerateCommand(
            schema_version="2026-03-11",
            command_id="cmd_3",
            topic="video.generate.command",
            job_id="vjob_3",
            document_id="doc_3",
            owner_id="user_3",
            attempt=1,
            target_duration_sec=60,
            voice="alloy",
            language="en",
            render_profile="720p",
            ocr_object_key="documents/doc_3/artifacts/ocr/raw.json",
            output_prefix="videos/vjob_3",
            correlation_id="corr_3",
            occurred_at="2026-03-11T00:00:00Z",
        ),
        worker_id="worker-3",
    )

    await publisher.send_complete(
        "vjob_3",
        InternalVideoComplete(
            final_video_object_key="videos/vjob_3/final.mp4",
            duration_sec=58.2,
            resolution="1280x720",
        ),
    )

    encoded = json.dumps(transport.events[0])
    assert '"event_type": "job.completed"' in encoded
    assert transport.events[0]["payload"]["final_video_object_key"] == "videos/vjob_3/final.mp4"

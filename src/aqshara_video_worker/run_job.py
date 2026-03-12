from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass

from aqshara_video_worker.clients.merge_client import MergeClient, create_merge_client
from aqshara_video_worker.clients.render_client import RenderClient, create_render_client
from aqshara_video_worker.clients.stream_event_publisher import RedisStreamEventPublisher
from aqshara_video_worker.clients.storage_client import StorageClient
from aqshara_video_worker.clients.tts_client import OpenAITtsClient
from aqshara_video_worker.config import WorkerSettings
from aqshara_video_worker.pipeline.runner import PipelineRunner
from aqshara_video_worker.schemas import VideoGenerateCommand, VideoGenerateJobPayload
from aqshara_video_worker.transport.redis_streams import RedisVideoTransport


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerDependencies:
    storage_client: StorageClient
    tts_client: OpenAITtsClient
    render_client: RenderClient
    merge_client: MergeClient


async def _run_worker() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = WorkerSettings()
    transport = RedisVideoTransport(settings)
    dependencies = _create_worker_dependencies(settings)

    try:
        await transport.ensure_consumer_group()

        while True:
            commands = await transport.read_commands()
            for stream_id, command in commands:
                await _process_command(
                    settings=settings,
                    transport=transport,
                    dependencies=dependencies,
                    command=command,
                )
                await transport.acknowledge_command(stream_id)
    finally:
        await transport.close()
        await dependencies.tts_client.close()
        await dependencies.render_client.close()


def _create_worker_dependencies(settings: WorkerSettings) -> WorkerDependencies:
    return WorkerDependencies(
        storage_client=StorageClient(settings),
        tts_client=OpenAITtsClient(settings),
        render_client=create_render_client(settings),
        merge_client=create_merge_client(settings),
    )


async def _process_command(
    *,
    settings: WorkerSettings,
    transport: RedisVideoTransport,
    dependencies: WorkerDependencies,
    command: VideoGenerateCommand,
) -> None:
    job = _build_job_payload(command)
    publisher = RedisStreamEventPublisher(
        transport=transport,
        command=command,
        worker_id=settings.video_worker_id,
    )
    runner = PipelineRunner(
        publisher,
        dependencies.storage_client,
        dependencies.tts_client,
        dependencies.render_client,
        dependencies.merge_client,
        render_profile=command.render_profile or settings.video_render_profile,
    )
    heartbeat_task = asyncio.create_task(_heartbeat_loop(publisher, settings))

    try:
        await publisher.send_accepted()
        _log_worker_event(
            "Starting video job",
            video_job_id=job.video_job_id,
            document_id=job.document_id,
            request_id=job.request_id,
            attempt=job.attempt,
            worker_id=settings.video_worker_id,
        )
        await runner.run(job)
        _log_worker_event(
            "Video job completed",
            video_job_id=job.video_job_id,
            worker_id=settings.video_worker_id,
        )
    except Exception as error:
        await runner.report_failure(job, error)
        logger.exception("Video worker failed")
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task


async def _heartbeat_loop(
    publisher: RedisStreamEventPublisher,
    settings: WorkerSettings,
) -> None:
    while True:
        await asyncio.sleep(settings.video_heartbeat_interval_sec)
        await publisher.send_heartbeat()


def _build_job_payload(command: VideoGenerateCommand) -> VideoGenerateJobPayload:
    return VideoGenerateJobPayload(
        video_job_id=command.job_id,
        document_id=command.document_id,
        actor_id=command.owner_id,
        target_duration_sec=command.target_duration_sec,
        voice=command.voice,
        language=command.language,
        request_id=command.request_id,
        attempt=command.attempt,
    )


def _log_worker_event(message: str, **details: object) -> None:
    logger.info(json.dumps({"message": message, **details}))


def main() -> None:
    raise SystemExit(asyncio.run(_run_worker()))


if __name__ == "__main__":
    main()

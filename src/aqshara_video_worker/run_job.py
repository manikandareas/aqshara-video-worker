from __future__ import annotations

import asyncio
import json
import logging
import sys

from aqshara_video_worker.clients.callback_client import CallbackClient
from aqshara_video_worker.clients.merge_client import create_merge_client
from aqshara_video_worker.clients.render_client import create_render_client
from aqshara_video_worker.clients.storage_client import StorageClient
from aqshara_video_worker.clients.tts_client import OpenAITtsClient
from aqshara_video_worker.config import WorkerSettings
from aqshara_video_worker.pipeline.runner import PipelineRunner
from aqshara_video_worker.schemas import VideoGenerateJobPayload


logger = logging.getLogger(__name__)


async def _run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    raw_payload = sys.stdin.read()
    if not raw_payload.strip():
        raise ValueError("Expected a JSON job payload on stdin")

    job = VideoGenerateJobPayload.model_validate_json(raw_payload)
    settings = WorkerSettings()
    callback_client = CallbackClient(settings)
    storage_client = StorageClient(settings)
    tts_client = OpenAITtsClient(settings)
    render_client = create_render_client(settings)
    merge_client = create_merge_client(settings)
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        merge_client,
        render_profile=settings.video_render_profile,
    )

    try:
        logger.info(
            json.dumps(
                {
                    "message": "Starting video job",
                    "video_job_id": job.video_job_id,
                    "document_id": job.document_id,
                    "request_id": job.request_id,
                    "attempt": job.attempt,
                }
            )
        )
        await runner.run(job)
    except Exception as error:
        await runner.report_failure(job, error)
        logger.exception("Video worker failed")
        return 1
    finally:
        await callback_client.close()
        await tts_client.close()
        await render_client.close()

    logger.info(
        json.dumps(
            {
                "message": "Video job completed",
                "video_job_id": job.video_job_id,
            }
        )
    )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

import boto3

from aqshara_video_worker.config import WorkerSettings


class StorageClient:
    def __init__(self, settings: WorkerSettings) -> None:
        self._bucket = settings.r2_bucket
        self._client = boto3.client(
            "s3",
            region_name=settings.r2_region,
            endpoint_url=settings.r2_endpoint,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
        )

    @staticmethod
    def create_video_final_key(video_job_id: str) -> str:
        return f"videos/{video_job_id}/final.mp4"

    @staticmethod
    def create_video_artifact_key(video_job_id: str, filename: str) -> str:
        return f"videos/{video_job_id}/artifacts/{filename}"

    @staticmethod
    def create_video_scene_audio_key(video_job_id: str, scene_index: int) -> str:
        return f"videos/{video_job_id}/artifacts/audio/scene-{scene_index:02d}.wav"

    @staticmethod
    def create_video_scene_code_key(video_job_id: str, scene_index: int) -> str:
        return f"videos/{video_job_id}/artifacts/code/scene-{scene_index:02d}.py"

    @staticmethod
    def create_video_scene_render_key(video_job_id: str, scene_index: int) -> str:
        return f"videos/{video_job_id}/artifacts/render/scene-{scene_index:02d}.mp4"

    @staticmethod
    def create_video_scene_render_log_key(
        video_job_id: str,
        scene_index: int,
    ) -> str:
        return f"videos/{video_job_id}/artifacts/render/scene-{scene_index:02d}.log"

    @staticmethod
    def create_video_merge_log_key(video_job_id: str) -> str:
        return f"videos/{video_job_id}/artifacts/merge.log"

    @staticmethod
    def create_document_ocr_artifact_key(document_id: str) -> str:
        return f"documents/{document_id}/artifacts/ocr/raw.json"

    async def upload_bytes(
        self,
        key: str,
        body: bytes,
        content_type: str,
    ) -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
        )

    async def upload_text(
        self,
        key: str,
        body: str,
        content_type: str,
    ) -> None:
        await self.upload_bytes(key, body.encode("utf-8"), content_type)

    async def download_bytes(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        body = response["Body"].read()
        return body if isinstance(body, bytes) else bytes(body)

    async def download_json(self, key: str) -> object:
        raw = await self.download_bytes(key)
        return json.loads(raw.decode("utf-8"))

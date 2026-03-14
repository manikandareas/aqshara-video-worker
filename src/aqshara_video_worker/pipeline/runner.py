from __future__ import annotations

import asyncio
import ast
import json
import re
from dataclasses import dataclass
from time import monotonic

from pydantic import ValidationError

from aqshara_video_worker.clients.ai_clients import CreativeGenerationClient, TtsClient
from aqshara_video_worker.clients.event_publisher import VideoEventPublisher
from aqshara_video_worker.clients.creative_generation_client import (
    CreativeConfigurationError,
    CreativeGenerationArtifacts,
    CreativeGenerationError,
)
from aqshara_video_worker.clients.merge_client import (
    AudioSyncValidationError,
    MergeClient,
    MergeClientError,
    MergeConfigurationError,
    MergeSceneAsset,
)
from aqshara_video_worker.clients.render_client import (
    EmptyRenderError,
    RenderClient,
    RenderConfigurationError,
    RenderClientError,
    RenderTimeoutError,
)
from aqshara_video_worker.clients.storage_client import StorageClient
from aqshara_video_worker.clients.tts_client import (
    AudioDurationError,
    EmptyAudioError,
    TtsConfigurationError,
    TtsGenerationError,
)
from aqshara_video_worker.pipeline.codegen import (
    CodeValidationError,
    ManimCodegenError,
    build_scene_module,
    validate_generated_code,
)
from aqshara_video_worker.pipeline.storyboard import (
    build_storyboard_artifacts,
    dumps_director_plan,
    dumps_storyboard,
    dumps_summary,
)
from aqshara_video_worker.schemas import (
    InternalVideoComplete,
    InternalVideoFail,
    InternalVideoMetrics,
    InternalVideoProgress,
    InternalVideoQualityGate,
    InternalVideoSceneProgress,
    SceneCodeDraftSpec,
    SceneRenderQASpec,
    SceneSpec,
    VideoGenerateJobPayload,
)


@dataclass(frozen=True)
class SceneCodeArtifact:
    scene: SceneSpec
    class_name: str
    scene_module: str
    object_key: str
    code_source: str = "template"


@dataclass(frozen=True)
class SceneRenderArtifact:
    scene_index: int
    code_artifact: SceneCodeArtifact
    render_result: object
    render_profile: str
    video_key: str
    log_key: str
    artifact_keys: list[str]


class PipelineRunner:
    def __init__(
        self,
        callback_client: VideoEventPublisher,
        storage_client: StorageClient,
        tts_client: TtsClient,
        render_client: RenderClient,
        merge_client: MergeClient,
        creative_client: CreativeGenerationClient | None = None,
        render_profile: str = "720p",
        tts_concurrency: int = 3,
        render_concurrency: int = 2,
        render_qa_enabled: bool = True,
        render_qa_max_revisions: int = 2,
        render_qa_sample_frames: int = 3,
    ) -> None:
        self._callback_client = callback_client
        self._storage_client = storage_client
        self._tts_client = tts_client
        self._render_client = render_client
        self._merge_client = merge_client
        self._creative_client = creative_client
        self._render_profile = render_profile if render_profile in ("480p", "720p", "1080p") else "720p"
        self._tts_concurrency = max(1, tts_concurrency)
        self._render_concurrency = max(1, render_concurrency)
        self._render_qa_enabled = render_qa_enabled
        self._render_qa_max_revisions = max(0, render_qa_max_revisions)
        self._render_qa_sample_frames = max(1, render_qa_sample_frames)

    async def run(self, job: VideoGenerateJobPayload) -> None:
        started_at = monotonic()
        artifact_keys: list[str] = []
        scene_audio_metadata: dict[int, tuple[int, str, bytes]] = {}
        scene_code_metadata: dict[int, SceneCodeArtifact] = {}
        scene_render_bytes: dict[int, bytes] = {}

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="preprocessing",
                progress_pct=5,
                message="Loading OCR artifact",
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        ocr_key = self._storage_client.create_document_ocr_artifact_key(
            job.document_id,
        )
        ocr_result = await self._storage_client.download_json(ocr_key)

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="summarizing",
                progress_pct=20,
                message="Generating structured summary",
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        creative_artifacts: CreativeGenerationArtifacts | None = None
        if self._creative_client is not None:
            creative_artifacts = await self._creative_client.generate_artifacts(
                ocr_result=ocr_result,
                target_duration_sec=job.target_duration_sec,
                language=job.language,
            )

        artifacts = creative_artifacts or build_storyboard_artifacts(
            ocr_result,
            target_duration_sec=job.target_duration_sec,
        )

        summary_key = self._storage_client.create_video_artifact_key(
            job.video_job_id,
            "summary.json",
        )
        await self._storage_client.upload_text(
            summary_key,
            dumps_summary(artifacts.summary),
            "application/json",
        )
        artifact_keys.append(summary_key)
        artifact_keys.extend(
            await self._persist_creative_planning_artifacts(
                job=job,
                creative_artifacts=creative_artifacts,
            )
        )

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="storyboarding",
                progress_pct=40,
                message="Building director plan and storyboard scenes",
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        director_plan_key = self._storage_client.create_video_artifact_key(
            job.video_job_id,
            "director_plan.json",
        )
        await self._storage_client.upload_text(
            director_plan_key,
            dumps_director_plan(artifacts.director_plan),
            "application/json",
        )
        artifact_keys.append(director_plan_key)

        storyboard_key = self._storage_client.create_video_artifact_key(
            job.video_job_id,
            "storyboard.json",
        )
        await self._storage_client.upload_text(
            storyboard_key,
            dumps_storyboard(artifacts.storyboard),
            "application/json",
        )
        artifact_keys.append(storyboard_key)

        scenes_key = self._storage_client.create_video_artifact_key(
            job.video_job_id,
            "scenes.md",
        )
        await self._storage_client.upload_text(
            scenes_key,
            artifacts.scenes_markdown,
            "text/markdown",
        )
        artifact_keys.append(scenes_key)

        for index, scene in enumerate(artifacts.storyboard.scenes, start=1):
            progress_pct = 40 + int((index / len(artifacts.storyboard.scenes)) * 25)
            await self._callback_client.send_progress(
                job.video_job_id,
                InternalVideoProgress(
                    pipeline_stage="storyboarding",
                    progress_pct=progress_pct,
                    quality_gate=InternalVideoQualityGate(
                        storyboard_valid=False,
                        code_valid=False,
                        render_valid=False,
                        audio_sync_valid=False,
                    ),
                    scene=InternalVideoSceneProgress(
                        scene_index=scene.scene_index,
                        template_type=scene.template_type,
                        status="pending",
                        planned_duration_ms=scene.planned_duration_ms,
                    ),
                    metrics=InternalVideoMetrics(
                        elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="storyboard_validating",
                progress_pct=70,
                message="Validating scenes.md and SceneSpec payloads",
                quality_gate=InternalVideoQualityGate(
                    storyboard_valid=True,
                    code_valid=False,
                    render_valid=False,
                    audio_sync_valid=False,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        total_scenes = len(artifacts.storyboard.scenes)
        (
            scene_audio_metadata,
            tts_artifact_keys,
        ) = await self._generate_scene_audio_batch(
            job=job,
            scenes=artifacts.storyboard.scenes,
            total_scenes=total_scenes,
            started_at=started_at,
        )
        artifact_keys.extend(tts_artifact_keys)

        aligned_storyboard = self._build_audio_aligned_storyboard(
            artifacts.storyboard,
            scene_audio_metadata,
        )
        creative_scene_code_drafts: dict[int, SceneCodeDraftSpec] = {}
        if creative_artifacts is not None and self._creative_client is not None:
            creative_scene_code_drafts = await self._creative_client.generate_scene_code_drafts(
                paper_analysis=creative_artifacts.paper_analysis,
                director_plan=creative_artifacts.director_plan,
                storyboard=aligned_storyboard,
                language=job.language,
            )

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="code_validating",
                progress_pct=90,
                message="Generating Manim scene modules",
                quality_gate=InternalVideoQualityGate(
                    storyboard_valid=True,
                    code_valid=False,
                    render_valid=False,
                    audio_sync_valid=False,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        codegen_results = await asyncio.gather(
            *[
                self._generate_single_scene_code(
                    job=job,
                    scene=scene,
                    index=index,
                    total_scenes=total_scenes,
                    scene_audio_metadata=scene_audio_metadata,
                    creative_scene_code_drafts=creative_scene_code_drafts,
                    started_at=started_at,
                )
                for index, scene in enumerate(aligned_storyboard.scenes, start=1)
            ]
        )
        for scene_index, code_artifact, extra_keys in sorted(codegen_results, key=lambda r: r[0]):
            artifact_keys.extend(extra_keys)
            scene_code_metadata[scene_index] = code_artifact

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="rendering",
                progress_pct=95,
                message="Rendering scene videos",
                render_profile=self._render_profile,
                quality_gate=InternalVideoQualityGate(
                    storyboard_valid=True,
                    code_valid=True,
                    render_valid=False,
                    audio_sync_valid=False,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        render_artifacts = await self._render_scene_batch(
            job=job,
            scenes=aligned_storyboard.scenes,
            total_scenes=total_scenes,
            scene_audio_metadata=scene_audio_metadata,
            scene_code_metadata=scene_code_metadata,
            started_at=started_at,
        )
        for render_artifact in render_artifacts:
            scene_code_metadata[render_artifact.scene_index] = render_artifact.code_artifact
            scene_render_bytes[render_artifact.scene_index] = (
                render_artifact.render_result.video_bytes
            )
            artifact_keys.extend(render_artifact.artifact_keys)

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="merging",
                progress_pct=98,
                message="Composing rendered scenes into final video",
                render_profile=self._render_profile,
                quality_gate=InternalVideoQualityGate(
                    storyboard_valid=True,
                    code_valid=True,
                    render_valid=True,
                    audio_sync_valid=False,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        merged_video = await self._merge_client.merge_scenes(
            scenes=[
                MergeSceneAsset(
                    scene_index=scene.scene_index,
                    video_bytes=scene_render_bytes[scene.scene_index],
                    audio_bytes=scene_audio_metadata[scene.scene_index][2],
                    audio_duration_ms=scene_audio_metadata[scene.scene_index][0],
                )
                for scene in aligned_storyboard.scenes
            ],
            render_profile=self._render_profile,
        )
        merge_log_key = self._storage_client.create_video_merge_log_key(job.video_job_id)
        await self._storage_client.upload_text(
            merge_log_key,
            "\n".join(
                part
                for part in [merged_video.stdout.strip(), merged_video.stderr.strip()]
                if part
            ),
            "text/plain",
        )
        artifact_keys.append(merge_log_key)

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="uploading",
                progress_pct=99,
                message="Uploading final video artifact",
                render_profile=self._render_profile,
                quality_gate=InternalVideoQualityGate(
                    storyboard_valid=True,
                    code_valid=True,
                    render_valid=True,
                    audio_sync_valid=True,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        final_video_key = self._storage_client.create_video_final_key(
            job.video_job_id,
        )
        await self._storage_client.upload_bytes(
            final_video_key,
            merged_video.video_bytes,
            "video/mp4",
        )

        await self._callback_client.send_complete(
            job.video_job_id,
            InternalVideoComplete(
                final_video_object_key=final_video_key,
                duration_sec=merged_video.duration_sec,
                resolution=(
                    "1280x720" if self._render_profile == "720p" else "854x480"
                ),
                artifact_keys=artifact_keys,
            ),
        )

    async def report_failure(
        self,
        job: VideoGenerateJobPayload,
        error: Exception,
        debug_artifact_keys: list[str] | None = None,
    ) -> None:
        error_code, pipeline_stage = self._classify_failure(error)
        await self._callback_client.send_fail(
            job.video_job_id,
            InternalVideoFail(
                pipeline_stage=pipeline_stage,
                error_code=error_code,
                error_message=str(error),
                debug_artifact_keys=debug_artifact_keys,
            ),
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> int:
        return int((monotonic() - started_at) * 1000)

    @staticmethod
    def _classify_failure(error: Exception) -> tuple[str, str]:
        if isinstance(error, ValidationError):
            return ("SCENESPEC_INVALID", "storyboard_validating")
        if isinstance(error, CreativeConfigurationError):
            return ("CREATIVE_CONFIGURATION_ERROR", "storyboarding")
        if isinstance(error, CreativeGenerationError):
            return ("CREATIVE_GENERATION_FAILED", "storyboarding")
        if isinstance(error, TtsConfigurationError):
            return ("TTS_CONFIGURATION_ERROR", "tts_generating")
        if isinstance(error, EmptyAudioError):
            return ("TTS_EMPTY_AUDIO", "tts_generating")
        if isinstance(error, AudioDurationError):
            return ("TTS_DURATION_INVALID", "tts_generating")
        if isinstance(error, TtsGenerationError):
            return ("TTS_GENERATION_FAILED", "tts_generating")
        if isinstance(error, ManimCodegenError):
            return ("MANIM_CODEGEN_FAILED", "code_validating")
        if isinstance(error, CodeValidationError):
            return ("CODE_VALIDATION_FAILED", "code_validating")
        if isinstance(error, EmptyRenderError):
            return ("RENDER_EMPTY_OUTPUT", "rendering")
        if isinstance(error, RenderTimeoutError):
            return ("RENDER_TIMEOUT", "rendering")
        if isinstance(error, RenderConfigurationError):
            return ("RENDER_CONFIGURATION_ERROR", "rendering")
        if isinstance(error, RenderClientError):
            return ("RENDER_EXECUTION_FAILED", "rendering")
        if isinstance(error, MergeConfigurationError):
            return ("MERGE_CONFIGURATION_ERROR", "merging")
        if isinstance(error, AudioSyncValidationError):
            return ("AUDIO_SYNC_INVALID", "merging")
        if isinstance(error, MergeClientError):
            return ("MERGE_EXECUTION_FAILED", "merging")
        return ("INTERNAL_UNKNOWN_ERROR", "failed")

    async def _persist_creative_planning_artifacts(
        self,
        *,
        job: VideoGenerateJobPayload,
        creative_artifacts: CreativeGenerationArtifacts | None,
    ) -> list[str]:
        if creative_artifacts is None:
            return []

        uploads = [
            (
                self._storage_client.create_video_artifact_key(
                    job.video_job_id,
                    "paper_analysis.json",
                ),
                creative_artifacts.paper_analysis.model_dump_json(indent=2),
            ),
            (
                self._storage_client.create_video_artifact_key(
                    job.video_job_id,
                    "script_plan.json",
                ),
                creative_artifacts.script_plan.model_dump_json(indent=2),
            ),
        ]
        artifact_keys: list[str] = []
        for key, body in uploads:
            await self._storage_client.upload_text(key, body, "application/json")
            artifact_keys.append(key)
        return artifact_keys

    async def _persist_scene_creative_artifacts(
        self,
        *,
        job: VideoGenerateJobPayload,
        scene_index: int,
        scene_code_draft: SceneCodeDraftSpec | None,
    ) -> list[str]:
        if scene_code_draft is None:
            return []

        critique_key = self._storage_client.create_video_artifact_key(
            job.video_job_id,
            f"creative/scene-{scene_index:02d}-critique.json",
        )
        revised_key = self._storage_client.create_video_artifact_key(
            job.video_job_id,
            f"creative/scene-{scene_index:02d}-revised.py",
        )
        await self._storage_client.upload_text(
            critique_key,
            scene_code_draft.critique.model_dump_json(indent=2),
            "application/json",
        )
        await self._storage_client.upload_text(
            revised_key,
            scene_code_draft.revised_code,
            "text/x-python",
        )
        return [critique_key, revised_key]

    async def _generate_single_scene_code(
        self,
        *,
        job: VideoGenerateJobPayload,
        scene: SceneSpec,
        index: int,
        total_scenes: int,
        scene_audio_metadata: dict[int, tuple[int, str, bytes]],
        creative_scene_code_drafts: dict | None,
        started_at: float,
    ) -> tuple[int, SceneCodeArtifact, list[str]]:
        actual_audio_duration_ms, scene_audio_key, _audio_bytes = scene_audio_metadata[
            scene.scene_index
        ]
        scene_code_key = self._storage_client.create_video_scene_code_key(
            job.video_job_id,
            scene.scene_index,
        )
        extra_keys: list[str] = []
        ai_scene_code_draft = None
        if creative_scene_code_drafts:
            ai_scene_code_draft = creative_scene_code_drafts.get(scene.scene_index)
            extra_keys.extend(
                await self._persist_scene_creative_artifacts(
                    job=job,
                    scene_index=scene.scene_index,
                    scene_code_draft=ai_scene_code_draft,
                )
            )
        code_artifact, _fallback_reason = await self._prepare_scene_code_artifact(
            job=job,
            scene=scene,
            scene_code_key=scene_code_key,
            started_at=started_at,
            ai_scene_code_draft=ai_scene_code_draft,
        )
        await self._storage_client.upload_text(
            scene_code_key,
            code_artifact.scene_module,
            "text/x-python",
        )
        extra_keys.append(scene_code_key)

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="code_validating",
                progress_pct=90 + int((index / total_scenes) * 4),
                message=f"Validated Manim code for scene {scene.scene_index}",
                quality_gate=InternalVideoQualityGate(
                    storyboard_valid=True,
                    code_valid=index == total_scenes,
                    render_valid=False,
                    audio_sync_valid=False,
                ),
                scene=InternalVideoSceneProgress(
                    scene_index=code_artifact.scene.scene_index,
                    template_type=code_artifact.scene.template_type,
                    status="done",
                    planned_duration_ms=code_artifact.scene.planned_duration_ms,
                    actual_audio_duration_ms=actual_audio_duration_ms,
                    audio_object_key=scene_audio_key,
                    manim_code_object_key=scene_code_key,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )
        return (scene.scene_index, code_artifact, extra_keys)

    async def _generate_scene_audio_batch(
        self,
        *,
        job: VideoGenerateJobPayload,
        scenes: list[SceneSpec],
        total_scenes: int,
        started_at: float,
    ) -> tuple[dict[int, tuple[int, str, bytes]], list[str]]:
        semaphore = asyncio.Semaphore(self._tts_concurrency)
        results = await asyncio.gather(
            *[
                self._generate_scene_audio(
                    job=job,
                    scene=scene,
                    index=index,
                    total_scenes=total_scenes,
                    started_at=started_at,
                    semaphore=semaphore,
                )
                for index, scene in enumerate(scenes, start=1)
            ]
        )
        scene_audio_metadata: dict[int, tuple[int, str, bytes]] = {}
        artifact_keys: list[str] = []
        for scene_index, duration_ms, object_key, audio_bytes in sorted(results):
            scene_audio_metadata[scene_index] = (duration_ms, object_key, audio_bytes)
            artifact_keys.append(object_key)
        return scene_audio_metadata, artifact_keys

    async def _generate_scene_audio(
        self,
        *,
        job: VideoGenerateJobPayload,
        scene: SceneSpec,
        index: int,
        total_scenes: int,
        started_at: float,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, int, str, bytes]:
        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="tts_generating",
                progress_pct=70 + int(((index - 1) / total_scenes) * 15),
                message=f"Generating narration audio for scene {scene.scene_index}",
                quality_gate=InternalVideoQualityGate(
                    storyboard_valid=True,
                    code_valid=False,
                    render_valid=False,
                    audio_sync_valid=False,
                ),
                scene=InternalVideoSceneProgress(
                    scene_index=scene.scene_index,
                    template_type=scene.template_type,
                    status="processing",
                    planned_duration_ms=scene.planned_duration_ms,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        async with semaphore:
            audio_bytes = await self._tts_client.generate_speech(
                text=scene.narration_text,
                voice=job.voice,
                language=job.language,
                voice_instruction=scene.narration_cues.voice_instruction,
            )
        actual_audio_duration_ms = self._tts_client.measure_duration_ms(audio_bytes)
        scene_audio_key = self._storage_client.create_video_scene_audio_key(
            job.video_job_id,
            scene.scene_index,
        )
        await self._storage_client.upload_bytes(
            scene_audio_key,
            audio_bytes,
            "audio/wav",
        )
        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="tts_generating",
                progress_pct=70 + int((index / total_scenes) * 20),
                message=f"Generated narration audio for scene {scene.scene_index}",
                quality_gate=InternalVideoQualityGate(
                    storyboard_valid=True,
                    code_valid=False,
                    render_valid=False,
                    audio_sync_valid=False,
                ),
                scene=InternalVideoSceneProgress(
                    scene_index=scene.scene_index,
                    template_type=scene.template_type,
                    status="done",
                    planned_duration_ms=scene.planned_duration_ms,
                    actual_audio_duration_ms=actual_audio_duration_ms,
                    audio_object_key=scene_audio_key,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )
        return (scene.scene_index, actual_audio_duration_ms, scene_audio_key, audio_bytes)

    async def _render_scene_batch(
        self,
        *,
        job: VideoGenerateJobPayload,
        scenes: list[SceneSpec],
        total_scenes: int,
        scene_audio_metadata: dict[int, tuple[int, str, bytes]],
        scene_code_metadata: dict[int, SceneCodeArtifact],
        started_at: float,
    ) -> list[SceneRenderArtifact]:
        semaphore = asyncio.Semaphore(self._render_concurrency)
        start_job = getattr(self._render_client, "start_job", None)
        finish_job = getattr(self._render_client, "finish_job", None)
        if callable(start_job):
            await start_job()
        try:
            render_artifacts = await asyncio.gather(
                *[
                    self._render_single_scene(
                        job=job,
                        scene=scene,
                        index=index,
                        total_scenes=total_scenes,
                        scene_audio_metadata=scene_audio_metadata[scene.scene_index],
                        code_artifact=scene_code_metadata[scene.scene_index],
                        started_at=started_at,
                        semaphore=semaphore,
                    )
                    for index, scene in enumerate(scenes, start=1)
                ]
            )
        finally:
            if callable(finish_job):
                await finish_job()
        return sorted(render_artifacts, key=lambda artifact: artifact.scene_index)

    async def _render_single_scene(
        self,
        *,
        job: VideoGenerateJobPayload,
        scene: SceneSpec,
        index: int,
        total_scenes: int,
        scene_audio_metadata: tuple[int, str, bytes],
        code_artifact: SceneCodeArtifact,
        started_at: float,
        semaphore: asyncio.Semaphore,
    ) -> SceneRenderArtifact:
        actual_audio_duration_ms, scene_audio_key, _audio_bytes = scene_audio_metadata
        current_artifact = code_artifact
        qa_artifact_keys: list[str] = []
        revision_attempt = 0

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="rendering",
                progress_pct=95 + int(((index - 1) / total_scenes) * 4),
                message=f"Rendering scene {scene.scene_index}",
                render_profile=self._render_profile,
                quality_gate=InternalVideoQualityGate(
                    storyboard_valid=True,
                    code_valid=True,
                    render_valid=False,
                    audio_sync_valid=False,
                ),
                scene=InternalVideoSceneProgress(
                    scene_index=scene.scene_index,
                    template_type=current_artifact.scene.template_type,
                    status="processing",
                    planned_duration_ms=current_artifact.scene.planned_duration_ms,
                    actual_audio_duration_ms=actual_audio_duration_ms,
                    audio_object_key=scene_audio_key,
                    manim_code_object_key=current_artifact.object_key,
                    qa_status="pending",
                    revision_attempt=revision_attempt,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

        while self._render_qa_enabled and current_artifact.code_source == "creative":
            async with semaphore:
                draft_artifact, draft_render_result, _draft_profile, _fallback_reason = (
                    await self._render_scene_with_fallback(
                        job=job,
                        code_artifact=current_artifact,
                        scene_audio_key=scene_audio_key,
                        actual_audio_duration_ms=actual_audio_duration_ms,
                        started_at=started_at,
                        active_render_profile="480p",
                    )
                )
            current_artifact = draft_artifact

            await self._callback_client.send_progress(
                job.video_job_id,
                InternalVideoProgress(
                    pipeline_stage="scene_reviewing",
                    progress_pct=95,
                    message=f"Reviewing layout quality for scene {scene.scene_index}",
                    render_profile="480p",
                    scene=InternalVideoSceneProgress(
                        scene_index=scene.scene_index,
                        template_type=current_artifact.scene.template_type,
                        status="processing",
                        planned_duration_ms=current_artifact.scene.planned_duration_ms,
                        actual_audio_duration_ms=actual_audio_duration_ms,
                        audio_object_key=scene_audio_key,
                        manim_code_object_key=current_artifact.object_key,
                        qa_status="pending",
                        revision_attempt=revision_attempt,
                    ),
                    metrics=InternalVideoMetrics(
                        elapsed_ms=self._elapsed_ms(started_at),
                    ),
                ),
            )

            extract_preview_frames = getattr(
                self._render_client,
                "extract_preview_frames",
                None,
            )
            sample_frames: list[bytes] = []
            if callable(extract_preview_frames):
                sample_frames = await extract_preview_frames(
                    video_bytes=draft_render_result.video_bytes,
                    sample_count=self._render_qa_sample_frames,
                )
            review = await self._review_rendered_scene(
                scene=current_artifact.scene,
                code_artifact=current_artifact,
                render_profile="480p",
                sample_frames=sample_frames,
            )
            qa_artifact_keys.extend(
                await self._persist_render_qa_artifacts(
                    job=job,
                    scene_index=scene.scene_index,
                    review=review,
                    sample_frames=sample_frames,
                    revision_attempt=revision_attempt,
                )
            )
            if review.qa_status == "pass" or not review.requires_revision:
                break
            if revision_attempt >= self._render_qa_max_revisions:
                fallback_scene = self._build_premium_fallback_scene(current_artifact.scene)
                if fallback_scene == current_artifact.scene:
                    break
                class_name, scene_module = build_scene_module(fallback_scene)
                validate_generated_code(scene_module, expected_class_name=class_name)
                current_artifact = SceneCodeArtifact(
                    scene=fallback_scene,
                    class_name=class_name,
                    scene_module=scene_module,
                    object_key=current_artifact.object_key,
                )
                await self._emit_fallback_progress(
                    job=job,
                    pipeline_stage="scene_revising",
                    scene=current_artifact.scene,
                    started_at=started_at,
                    reason="render_qa_template_fallback",
                    render_profile="480p",
                    scene_audio_key=scene_audio_key,
                    actual_audio_duration_ms=actual_audio_duration_ms,
                    scene_code_key=current_artifact.object_key,
                )
                break

            revision_attempt += 1
            await self._callback_client.send_progress(
                job.video_job_id,
                InternalVideoProgress(
                    pipeline_stage="scene_revising",
                    progress_pct=95,
                    message=f"Revising scene {scene.scene_index} after layout QA",
                    render_profile="480p",
                    scene=InternalVideoSceneProgress(
                        scene_index=scene.scene_index,
                        template_type=current_artifact.scene.template_type,
                        status="processing",
                        planned_duration_ms=current_artifact.scene.planned_duration_ms,
                        actual_audio_duration_ms=actual_audio_duration_ms,
                        audio_object_key=scene_audio_key,
                        manim_code_object_key=current_artifact.object_key,
                        qa_status="revising",
                        revision_attempt=revision_attempt,
                    ),
                    metrics=InternalVideoMetrics(
                        elapsed_ms=self._elapsed_ms(started_at),
                    ),
                ),
            )
            if self._creative_client is not None:
                revised_code = await self._creative_client.revise_scene_code_for_render_qa(
                    scene_json=current_artifact.scene.model_dump_json(indent=2),
                    scene_code=current_artifact.scene_module,
                    review=review,
                )
                validate_generated_code(revised_code)
                class_name = self._extract_class_name(revised_code)
                if class_name is None:
                    raise CodeValidationError("Revised QA code did not define a class")
                current_artifact = SceneCodeArtifact(
                    scene=current_artifact.scene,
                    class_name=class_name,
                    scene_module=revised_code,
                    object_key=current_artifact.object_key,
                )
                await self._storage_client.upload_text(
                    current_artifact.object_key,
                    revised_code,
                    "text/x-python",
                )
            else:
                fallback_scene = self._build_premium_fallback_scene(current_artifact.scene)
                class_name, scene_module = build_scene_module(fallback_scene)
                validate_generated_code(scene_module, expected_class_name=class_name)
                current_artifact = SceneCodeArtifact(
                    scene=fallback_scene,
                    class_name=class_name,
                    scene_module=scene_module,
                    object_key=current_artifact.object_key,
                )
                await self._storage_client.upload_text(
                    current_artifact.object_key,
                    scene_module,
                    "text/x-python",
                )

        async with semaphore:
            current_artifact, render_result, render_profile, _fallback_reason = (
                await self._render_scene_with_fallback(
                    job=job,
                    code_artifact=current_artifact,
                    scene_audio_key=scene_audio_key,
                    actual_audio_duration_ms=actual_audio_duration_ms,
                    started_at=started_at,
                    active_render_profile=self._render_profile,
                )
            )

        scene_video_key = self._storage_client.create_video_scene_render_key(
            job.video_job_id,
            scene.scene_index,
        )
        scene_log_key = self._storage_client.create_video_scene_render_log_key(
            job.video_job_id,
            scene.scene_index,
        )
        await self._storage_client.upload_bytes(
            scene_video_key,
            render_result.video_bytes,
            "video/mp4",
        )
        await self._storage_client.upload_text(
            scene_log_key,
            "\n".join(
                [
                    f"profile={render_result.render_profile}",
                    f"resolution={render_result.resolution}",
                    render_result.stdout,
                    render_result.stderr,
                ]
            ).strip(),
            "text/plain",
        )
        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="rendering",
                progress_pct=95 + int((index / total_scenes) * 4),
                message=f"Rendered scene {scene.scene_index}",
                render_profile=self._render_profile,
                quality_gate=InternalVideoQualityGate(
                    storyboard_valid=True,
                    code_valid=True,
                    render_valid=index == total_scenes,
                    audio_sync_valid=False,
                ),
                scene=InternalVideoSceneProgress(
                    scene_index=scene.scene_index,
                    template_type=current_artifact.scene.template_type,
                    status="done",
                    planned_duration_ms=current_artifact.scene.planned_duration_ms,
                    actual_audio_duration_ms=actual_audio_duration_ms,
                    audio_object_key=scene_audio_key,
                    manim_code_object_key=current_artifact.object_key,
                    video_object_key=scene_video_key,
                    qa_status="passed",
                    revision_attempt=revision_attempt,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )
        return SceneRenderArtifact(
            scene_index=scene.scene_index,
            code_artifact=current_artifact,
            render_result=render_result,
            render_profile=render_profile,
            video_key=scene_video_key,
            log_key=scene_log_key,
            artifact_keys=[scene_video_key, scene_log_key, *qa_artifact_keys],
        )

    async def _review_rendered_scene(
        self,
        *,
        scene: SceneSpec,
        code_artifact: SceneCodeArtifact,
        render_profile: str,
        sample_frames: list[bytes],
    ) -> SceneRenderQASpec:
        if self._creative_client is not None and sample_frames:
            return await self._creative_client.review_rendered_scene(
                scene_index=scene.scene_index,
                scene_json=scene.model_dump_json(indent=2),
                scene_code=code_artifact.scene_module,
                render_profile=render_profile,
                sample_frames=sample_frames,
            )

        font_sizes = [
            int(match)
            for match in re.findall(r"font_size=(\d+)", code_artifact.scene_module)
        ]
        text_nodes = code_artifact.scene_module.count("Text(")
        issues: list[str] = []
        if font_sizes and min(font_sizes) < 18:
            issues.append("Some text appears too small for reliable readability.")
        if text_nodes > 8:
            issues.append("Scene contains too many text elements and may feel overcrowded.")
        return SceneRenderQASpec(
            scene_index=scene.scene_index,
            strengths=["Fallback QA used static readability heuristics."],
            issues=issues,
            revision_brief=(
                "Reduce text density and increase font sizes."
                if issues
                else "Rendered scene looks readable."
            ),
            requires_revision=bool(issues),
            qa_status="revise" if issues else "pass",
        )

    async def _persist_render_qa_artifacts(
        self,
        *,
        job: VideoGenerateJobPayload,
        scene_index: int,
        review: SceneRenderQASpec,
        sample_frames: list[bytes],
        revision_attempt: int,
    ) -> list[str]:
        review_key = self._storage_client.create_video_artifact_key(
            job.video_job_id,
            f"creative/scene-{scene_index:02d}-render-qa-{revision_attempt:02d}.json",
        )
        await self._storage_client.upload_text(
            review_key,
            review.model_dump_json(indent=2),
            "application/json",
        )
        artifact_keys = [review_key]
        for frame_index, frame_bytes in enumerate(sample_frames, start=1):
            frame_key = self._storage_client.create_video_artifact_key(
                job.video_job_id,
                (
                    "creative/"
                    f"scene-{scene_index:02d}-qa-frame-{revision_attempt:02d}-{frame_index:02d}.png"
                ),
            )
            await self._storage_client.upload_bytes(
                frame_key,
                frame_bytes,
                "image/png",
            )
            artifact_keys.append(frame_key)
        return artifact_keys

    async def _prepare_scene_code_artifact(
        self,
        *,
        job: VideoGenerateJobPayload,
        scene: SceneSpec,
        scene_code_key: str,
        started_at: float,
        ai_scene_code_draft: SceneCodeDraftSpec | None = None,
    ) -> tuple[SceneCodeArtifact, str | None]:
        if ai_scene_code_draft is not None:
            candidate_codes = [
                ai_scene_code_draft.revised_code,
                ai_scene_code_draft.draft_code,
            ]
            for candidate_code in candidate_codes:
                try:
                    validate_generated_code(candidate_code)
                except CodeValidationError:
                    continue
                class_name = self._extract_class_name(candidate_code)
                if class_name is None:
                    continue
                return (
                    SceneCodeArtifact(
                        scene=scene,
                        class_name=class_name,
                        scene_module=candidate_code,
                        object_key=scene_code_key,
                        code_source="creative",
                    ),
                    None,
                )

        try:
            class_name, scene_module = build_scene_module(scene)
            validate_generated_code(scene_module, expected_class_name=class_name)
            return (
                SceneCodeArtifact(
                    scene=scene,
                    class_name=class_name,
                    scene_module=scene_module,
                    object_key=scene_code_key,
                ),
                None,
            )
        except (ManimCodegenError, CodeValidationError):
            fallback_scene = self._build_premium_fallback_scene(scene)
            if fallback_scene is None:
                raise

            class_name, scene_module = build_scene_module(fallback_scene)
            validate_generated_code(scene_module, expected_class_name=class_name)
            await self._emit_fallback_progress(
                job=job,
                pipeline_stage="code_validating",
                scene=fallback_scene,
                started_at=started_at,
                reason="code_validation_template_fallback",
            )
            return (
                SceneCodeArtifact(
                    scene=fallback_scene,
                    class_name=class_name,
                    scene_module=scene_module,
                    object_key=scene_code_key,
                ),
                "code_validation_template_fallback",
            )

    @staticmethod
    def _extract_class_name(code: str) -> str | None:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                return node.name
        return None

    async def _render_scene_with_fallback(
        self,
        *,
        job: VideoGenerateJobPayload,
        code_artifact: SceneCodeArtifact,
        scene_audio_key: str,
        actual_audio_duration_ms: int,
        started_at: float,
        active_render_profile: str,
    ) -> tuple[SceneCodeArtifact, object, str, str | None]:
        fallback_reason: str | None = None
        current_artifact = code_artifact
        current_profile = active_render_profile
        tried_template_fallback = current_artifact.scene.scene_kind == "fallback"

        while True:
            try:
                render_result = await self._render_client.render_scene(
                    scene_index=current_artifact.scene.scene_index,
                    class_name=current_artifact.class_name,
                    scene_code=current_artifact.scene_module,
                    render_profile=current_profile,
                )
                return (
                    current_artifact,
                    render_result,
                    current_profile,
                    fallback_reason,
                )
            except RenderConfigurationError:
                raise
            except (EmptyRenderError, RenderTimeoutError, RenderClientError):
                if current_profile == "720p":
                    current_profile = "480p"
                    fallback_reason = "render_profile_downgrade_480p"
                    await self._emit_fallback_progress(
                        job=job,
                        pipeline_stage="rendering",
                        scene=current_artifact.scene,
                        started_at=started_at,
                        reason=fallback_reason,
                        render_profile=current_profile,
                        scene_audio_key=scene_audio_key,
                        actual_audio_duration_ms=actual_audio_duration_ms,
                        scene_code_key=current_artifact.object_key,
                    )
                    continue

                if not tried_template_fallback:
                    fallback_scene = self._build_premium_fallback_scene(
                        current_artifact.scene,
                    )
                    if fallback_scene is not None:
                        class_name, scene_module = build_scene_module(fallback_scene)
                        validate_generated_code(
                            scene_module,
                            expected_class_name=class_name,
                        )
                        await self._storage_client.upload_text(
                            current_artifact.object_key,
                            scene_module,
                            "text/x-python",
                        )
                        current_artifact = SceneCodeArtifact(
                            scene=fallback_scene,
                            class_name=class_name,
                            scene_module=scene_module,
                            object_key=current_artifact.object_key,
                        )
                        tried_template_fallback = True
                        fallback_reason = "render_template_fallback_premium"
                        await self._emit_fallback_progress(
                            job=job,
                            pipeline_stage="rendering",
                            scene=current_artifact.scene,
                            started_at=started_at,
                            reason=fallback_reason,
                            render_profile=current_profile,
                            scene_audio_key=scene_audio_key,
                            actual_audio_duration_ms=actual_audio_duration_ms,
                            scene_code_key=current_artifact.object_key,
                        )
                        continue

                raise

    async def _emit_fallback_progress(
        self,
        *,
        job: VideoGenerateJobPayload,
        pipeline_stage: str,
        scene: SceneSpec,
        started_at: float,
        reason: str,
        render_profile: str | None = None,
        scene_audio_key: str | None = None,
        actual_audio_duration_ms: int | None = None,
        scene_code_key: str | None = None,
    ) -> None:
        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage=pipeline_stage,
                progress_pct=95 if pipeline_stage == "rendering" else 90,
                message=f"Applying fallback for scene {scene.scene_index}",
                fallback_applied=True,
                fallback_reason=reason,
                render_profile=render_profile,
                scene=InternalVideoSceneProgress(
                    scene_index=scene.scene_index,
                    template_type=scene.template_type,
                    status="processing",
                    planned_duration_ms=scene.planned_duration_ms,
                    actual_audio_duration_ms=actual_audio_duration_ms,
                    audio_object_key=scene_audio_key,
                    manim_code_object_key=scene_code_key,
                ),
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

    @staticmethod
    def _build_audio_aligned_storyboard(
        storyboard,
        scene_audio_metadata: dict[int, tuple[int, str, bytes]],
    ):
        return storyboard.model_copy(
            update={
                "scenes": [
                    scene.model_copy(
                        update={
                            "target_render_duration_ms": scene_audio_metadata[
                                scene.scene_index
                            ][0],
                        }
                    )
                    for scene in storyboard.scenes
                ]
            }
        )

    @staticmethod
    def _build_premium_fallback_scene(scene: SceneSpec) -> SceneSpec | None:
        if scene.scene_kind == "fallback":
            return None

        safe_template = {
            "hook": "hook",
            "problem": "problem",
            "mechanism": "mechanism",
            "evidence": "evidence",
            "takeaway": "takeaway",
        }.get(scene.scene_kind, "bullet")
        return scene.model_copy(
            update={
                "template_type": safe_template,
                "scene_kind": "fallback",
                "purpose": f"{scene.purpose} (premium fallback)",
                "visual_elements": scene.visual_elements[:3] or [scene.narration_text],
                "camera_plan": scene.camera_plan.model_copy(update={"mode": "static"}),
                "transition_strategy": "fade",
                "chart_data": scene.chart_data[:3] if scene.chart_data else None,
            },
        )

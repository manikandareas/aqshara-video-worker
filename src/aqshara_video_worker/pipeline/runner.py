from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from pydantic import ValidationError

from aqshara_video_worker.clients.callback_client import CallbackClient
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
    OpenAITtsClient,
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
    SceneSpec,
    VideoGenerateJobPayload,
)


@dataclass(frozen=True)
class SceneCodeArtifact:
    scene: SceneSpec
    class_name: str
    scene_module: str
    object_key: str


class PipelineRunner:
    def __init__(
        self,
        callback_client: CallbackClient,
        storage_client: StorageClient,
        tts_client: OpenAITtsClient,
        render_client: RenderClient,
        merge_client: MergeClient,
        render_profile: str = "720p",
    ) -> None:
        self._callback_client = callback_client
        self._storage_client = storage_client
        self._tts_client = tts_client
        self._render_client = render_client
        self._merge_client = merge_client
        self._render_profile = "480p" if render_profile == "480p" else "720p"

    async def run(self, job: VideoGenerateJobPayload) -> None:
        started_at = monotonic()
        artifact_keys: list[str] = []
        scene_audio_metadata: dict[int, tuple[int, str, bytes]] = {}
        scene_code_metadata: dict[int, SceneCodeArtifact] = {}
        scene_render_bytes: dict[int, bytes] = {}
        active_render_profile = self._render_profile

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

        artifacts = build_storyboard_artifacts(
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

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="storyboarding",
                progress_pct=40,
                message="Building storyboard scenes",
                metrics=InternalVideoMetrics(
                    elapsed_ms=self._elapsed_ms(started_at),
                ),
            ),
        )

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
        for index, scene in enumerate(artifacts.storyboard.scenes, start=1):
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

            audio_bytes = await self._tts_client.generate_speech(
                text=scene.narration_text,
                voice=job.voice,
                language=job.language,
            )
            actual_audio_duration_ms = self._tts_client.measure_duration_ms(
                audio_bytes,
            )
            scene_audio_key = self._storage_client.create_video_scene_audio_key(
                job.video_job_id,
                scene.scene_index,
            )
            await self._storage_client.upload_bytes(
                scene_audio_key,
                audio_bytes,
                "audio/wav",
            )
            artifact_keys.append(scene_audio_key)
            scene_audio_metadata[scene.scene_index] = (
                actual_audio_duration_ms,
                scene_audio_key,
                audio_bytes,
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

        for index, scene in enumerate(artifacts.storyboard.scenes, start=1):
            actual_audio_duration_ms, scene_audio_key, _audio_bytes = scene_audio_metadata[
                scene.scene_index
            ]
            scene_code_key = self._storage_client.create_video_scene_code_key(
                job.video_job_id,
                scene.scene_index,
            )
            code_artifact, _fallback_reason = await self._prepare_scene_code_artifact(
                job=job,
                scene=scene,
                scene_code_key=scene_code_key,
                started_at=started_at,
            )
            await self._storage_client.upload_text(
                scene_code_key,
                code_artifact.scene_module,
                "text/x-python",
            )
            artifact_keys.append(scene_code_key)
            scene_code_metadata[scene.scene_index] = code_artifact

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

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="rendering",
                progress_pct=95,
                message="Rendering scene videos",
                render_profile=active_render_profile,
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

        for index, scene in enumerate(artifacts.storyboard.scenes, start=1):
            actual_audio_duration_ms, scene_audio_key, _audio_bytes = scene_audio_metadata[
                scene.scene_index
            ]
            code_artifact = scene_code_metadata[scene.scene_index]
            await self._callback_client.send_progress(
                job.video_job_id,
                InternalVideoProgress(
                    pipeline_stage="rendering",
                    progress_pct=95 + int(((index - 1) / total_scenes) * 4),
                    message=f"Rendering scene {scene.scene_index}",
                    render_profile=active_render_profile,
                    quality_gate=InternalVideoQualityGate(
                        storyboard_valid=True,
                        code_valid=True,
                        render_valid=False,
                        audio_sync_valid=False,
                    ),
                    scene=InternalVideoSceneProgress(
                        scene_index=scene.scene_index,
                        template_type=code_artifact.scene.template_type,
                        status="processing",
                        planned_duration_ms=code_artifact.scene.planned_duration_ms,
                        actual_audio_duration_ms=actual_audio_duration_ms,
                        audio_object_key=scene_audio_key,
                        manim_code_object_key=code_artifact.object_key,
                    ),
                    metrics=InternalVideoMetrics(
                        elapsed_ms=self._elapsed_ms(started_at),
                    ),
                ),
            )

            code_artifact, render_result, active_render_profile, _fallback_reason = (
                await self._render_scene_with_fallback(
                    job=job,
                    code_artifact=code_artifact,
                    scene_audio_key=scene_audio_key,
                    actual_audio_duration_ms=actual_audio_duration_ms,
                    started_at=started_at,
                    active_render_profile=active_render_profile,
                )
            )
            scene_code_metadata[scene.scene_index] = code_artifact
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
            artifact_keys.extend([scene_video_key, scene_log_key])
            scene_render_bytes[scene.scene_index] = render_result.video_bytes

            await self._callback_client.send_progress(
                job.video_job_id,
                InternalVideoProgress(
                    pipeline_stage="rendering",
                    progress_pct=95 + int((index / total_scenes) * 4),
                    message=f"Rendered scene {scene.scene_index}",
                    render_profile=active_render_profile,
                    quality_gate=InternalVideoQualityGate(
                        storyboard_valid=True,
                        code_valid=True,
                        render_valid=index == total_scenes,
                        audio_sync_valid=False,
                    ),
                    scene=InternalVideoSceneProgress(
                        scene_index=scene.scene_index,
                        template_type=code_artifact.scene.template_type,
                        status="done",
                        planned_duration_ms=code_artifact.scene.planned_duration_ms,
                        actual_audio_duration_ms=actual_audio_duration_ms,
                        audio_object_key=scene_audio_key,
                        manim_code_object_key=code_artifact.object_key,
                        video_object_key=scene_video_key,
                    ),
                    metrics=InternalVideoMetrics(
                        elapsed_ms=self._elapsed_ms(started_at),
                    ),
                ),
            )

        await self._callback_client.send_progress(
            job.video_job_id,
            InternalVideoProgress(
                pipeline_stage="merging",
                progress_pct=98,
                message="Composing rendered scenes into final video",
                render_profile=active_render_profile,
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
                for scene in artifacts.storyboard.scenes
            ],
            render_profile=active_render_profile,
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
                render_profile=active_render_profile,
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
                    "1280x720" if active_render_profile == "720p" else "854x480"
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

    async def _prepare_scene_code_artifact(
        self,
        *,
        job: VideoGenerateJobPayload,
        scene: SceneSpec,
        scene_code_key: str,
        started_at: float,
    ) -> tuple[SceneCodeArtifact, str | None]:
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
            fallback_scene = self._build_bullet_fallback_scene(scene)
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
        tried_bullet_fallback = current_artifact.scene.template_type == "bullet"

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

                if not tried_bullet_fallback:
                    fallback_scene = self._build_bullet_fallback_scene(
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
                        tried_bullet_fallback = True
                        fallback_reason = "render_template_fallback_bullet"
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
    def _build_bullet_fallback_scene(scene: SceneSpec) -> SceneSpec | None:
        if scene.template_type == "bullet":
            return None

        return scene.model_copy(
            update={
                "template_type": "bullet",
                "purpose": f"{scene.purpose} (fallback)",
                "visual_elements": scene.visual_elements[:4] or [scene.narration_text],
            },
        )

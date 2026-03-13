import pytest
from pydantic import ValidationError

from aqshara_video_worker.clients.creative_generation_client import (
    CreativeConfigurationError,
    CreativeGenerationArtifacts,
    CreativeGenerationError,
)
from aqshara_video_worker.clients.merge_client import (
    AudioSyncValidationError,
    MergeExecutionError,
)
from aqshara_video_worker.clients.render_client import RenderClientError
from aqshara_video_worker.clients.render_client import RenderConfigurationError
from aqshara_video_worker.clients.render_client import RenderTimeoutError
from aqshara_video_worker.pipeline.codegen import CodeValidationError
from aqshara_video_worker.pipeline.runner import PipelineRunner
from aqshara_video_worker.pipeline.storyboard import build_storyboard_artifacts
from aqshara_video_worker.schemas import (
    PaperAnalysisSpec,
    SceneCodeCritiqueSpec,
    SceneCodeDraftSpec,
    SceneScriptSpec,
    SceneSpec,
    ScriptPlanSpec,
    StoryboardSpec,
    VideoGenerateJobPayload,
)


class RecordingCallbackClient:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    async def send_progress(self, job_id: str, payload) -> None:
        self.events.append(("progress", job_id, payload.model_dump(exclude_none=True)))

    async def send_complete(self, job_id: str, payload) -> None:
        self.events.append(("complete", job_id, payload.model_dump(exclude_none=True)))

    async def send_fail(self, job_id: str, payload) -> None:
        self.events.append(("fail", job_id, payload.model_dump(exclude_none=True)))


class RecordingStorageClient:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[str] = []

    @staticmethod
    def create_video_artifact_key(video_job_id: str, filename: str) -> str:
        return f"videos/{video_job_id}/artifacts/{filename}"

    @staticmethod
    def create_video_final_key(video_job_id: str) -> str:
        return f"videos/{video_job_id}/final.mp4"

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
    def create_video_scene_render_log_key(video_job_id: str, scene_index: int) -> str:
        return f"videos/{video_job_id}/artifacts/render/scene-{scene_index:02d}.log"

    @staticmethod
    def create_video_merge_log_key(video_job_id: str) -> str:
        return f"videos/{video_job_id}/artifacts/merge.log"

    @staticmethod
    def create_document_ocr_artifact_key(document_id: str) -> str:
        return f"documents/{document_id}/artifacts/ocr/raw.json"

    async def upload_text(self, key: str, body: str, content_type: str) -> None:
        self.uploads.append((key, content_type))

    async def upload_bytes(self, key: str, body: bytes, content_type: str) -> None:
        self.uploads.append((key, content_type))

    async def download_json(self, key: str) -> object:
        self.downloads.append(key)
        return {
            "pages": [
                {
                    "index": 0,
                    "markdown": (
                        "# Efficient Paper Distillation\n\n"
                        "This paper addresses the problem of slow paper review workflows. "
                        "Researchers need a concise way to understand a paper fast.\n\n"
                        "We present a pipeline that extracts structure, summarizes each part, "
                        "and builds an animated teaching script.\n\n"
                        "Our evaluation shows faster comprehension and better recall for readers.\n\n"
                        "In conclusion, the approach turns dense papers into short learning assets."
                    ),
                }
            ]
        }


class RecordingTtsClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str, str | None]] = []

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str,
        voice_instruction: str | None = None,
    ) -> bytes:
        self.requests.append((text, voice, language, voice_instruction))
        return b"fake wav bytes"

    @staticmethod
    def measure_duration_ms(audio_bytes: bytes) -> int:
        assert audio_bytes == b"fake wav bytes"
        return 12_345


class RecordingRenderClient:
    def __init__(self) -> None:
        self.requests: list[tuple[int, str, str, str]] = []

    async def render_scene(
        self,
        *,
        scene_index: int,
        class_name: str,
        scene_code: str,
        render_profile: str,
    ):
        self.requests.append((scene_index, class_name, scene_code, render_profile))
        return type(
            "RenderResult",
            (),
            {
                "video_bytes": f"rendered-scene-{scene_index}".encode("utf-8"),
                "stdout": f"Rendered scene {scene_index}",
                "stderr": "",
                "resolution": "1280x720",
                "render_profile": render_profile,
            },
        )()


class DowngradingRenderClient:
    def __init__(self) -> None:
        self.requests: list[tuple[int, str]] = []

    async def render_scene(
        self,
        *,
        scene_index: int,
        class_name: str,
        scene_code: str,
        render_profile: str,
    ):
        self.requests.append((scene_index, render_profile))
        if scene_index == 1 and render_profile == "720p":
            raise RenderClientError("simulated high-quality render failure")

        return type(
            "RenderResult",
            (),
            {
                "video_bytes": f"rendered-{render_profile}-scene-{scene_index}".encode(
                    "utf-8"
                ),
                "stdout": f"Rendered scene {scene_index}",
                "stderr": "",
                "resolution": "854x480" if render_profile == "480p" else "1280x720",
                "render_profile": render_profile,
            },
        )()


class ConfigFailingRenderClient:
    def __init__(self) -> None:
        self.requests: list[tuple[int, str]] = []

    async def render_scene(
        self,
        *,
        scene_index: int,
        class_name: str,
        scene_code: str,
        render_profile: str,
    ):
        self.requests.append((scene_index, render_profile))
        raise RenderConfigurationError("missing daytona configuration")


class RecordingMergeClient:
    def __init__(self) -> None:
        self.requests: list[tuple[list[int], str]] = []

    async def merge_scenes(self, *, scenes, render_profile: str):
        self.requests.append(([scene.scene_index for scene in scenes], render_profile))
        return type(
            "MergeResult",
            (),
            {
                "video_bytes": b"final merged video bytes",
                "stdout": "merged scenes",
                "stderr": "",
                "duration_sec": 61.725,
            },
        )()


class FailingMergeClient:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def merge_scenes(self, *, scenes, render_profile: str):
        raise self._error


class RecordingCreativeClient:
    def __init__(self) -> None:
        self.requests: list[tuple[int, str]] = []
        self.scene_draft_requests: list[list[int]] = []
        self._scene_code_drafts: dict[int, SceneCodeDraftSpec] = {}

    async def generate_artifacts(
        self,
        *,
        ocr_result: object,
        target_duration_sec: int,
        language: str,
    ) -> CreativeGenerationArtifacts:
        self.requests.append((target_duration_sec, language))
        baseline = build_storyboard_artifacts(ocr_result, target_duration_sec)
        script_plan = self._build_script_plan(baseline.storyboard.scenes)
        storyboard = self._build_storyboard(baseline.storyboard, script_plan)
        scene_code_drafts = {
            scene.scene_index: SceneCodeDraftSpec(
                scene_index=scene.scene_index,
                draft_code=self._build_scene_code(scene.scene_index, revised=False),
                critique=SceneCodeCritiqueSpec(
                    scene_index=scene.scene_index,
                    strengths=["clear visual"],
                    issues=["tight timing"],
                    revision_brief="Keep the scene simple and valid.",
                    requires_revision=True,
                ),
                revised_code=self._build_scene_code(scene.scene_index, revised=True),
            )
            for scene in storyboard.scenes
        }
        self._scene_code_drafts = scene_code_drafts
        return CreativeGenerationArtifacts(
            summary=baseline.summary,
            paper_analysis=PaperAnalysisSpec(
                topic=baseline.summary.topic,
                problem=baseline.summary.problem,
                method=baseline.summary.method,
                result=baseline.summary.result,
                conclusion=baseline.summary.conclusion,
                key_entities=["AI", "Manim"],
                visual_opportunities=["camera push"],
                misconceptions=["slides are enough"],
            ),
            director_plan=baseline.director_plan,
            script_plan=script_plan,
            storyboard=storyboard,
            scenes_markdown=baseline.scenes_markdown,
            scene_code_drafts=scene_code_drafts,
        )

    @staticmethod
    def _build_script_plan(scenes: list[SceneSpec]) -> ScriptPlanSpec:
        return ScriptPlanSpec(
            hook_line="What if the paper could explain itself visually?",
            tone="clear and energetic",
            scenes=[
                SceneScriptSpec(
                    scene_index=scene.scene_index,
                    narration_text=f"AI script for scene {scene.scene_index}.",
                    voice_instruction="Speak with momentum and clarity.",
                    emphasis_terms=["AI", "visual"],
                )
                for scene in scenes
            ],
        )

    @staticmethod
    def _build_storyboard(
        storyboard: StoryboardSpec,
        script_plan: ScriptPlanSpec,
    ) -> StoryboardSpec:
        return storyboard.model_copy(
            update={
                "hook": script_plan.hook_line,
                "scenes": [
                    scene.model_copy(
                        update={
                            "narration_text": f"AI script for scene {scene.scene_index}.",
                            "narration_cues": scene.narration_cues.model_copy(
                                update={
                                    "voice_instruction": "Speak with momentum and clarity."
                                }
                            ),
                        }
                    )
                    for scene in storyboard.scenes
                ],
            }
        )

    @staticmethod
    def _build_scene_code(scene_index: int, *, revised: bool) -> str:
        render_seed = 2 if revised else 1
        label = "AI revised scene" if revised else "AI scene"
        play_call = (
            "        self.play(Write(label), run_time=0.6)\n"
            if revised
            else "        self.play(Write(label))\n"
        )
        return (
            "from manim import *\n\n"
            'config.background_color = "#0B1020"\n\n'
            f"class AIScene{scene_index:02d}(Scene):\n"
            f"    RENDER_SEED = {render_seed}\n\n"
            "    def construct(self):\n"
            f"        label = Text('{label} {scene_index}')\n"
            f"{play_call}"
            "        self.wait(0.2)\n"
        )

    async def generate_scene_code_drafts(
        self,
        *,
        paper_analysis: PaperAnalysisSpec,
        director_plan,
        storyboard,
        language: str,
    ) -> dict[int, SceneCodeDraftSpec]:
        del paper_analysis, director_plan, language
        self.scene_draft_requests.append(
            [scene.target_render_duration_ms or scene.planned_duration_ms for scene in storyboard.scenes]
        )
        return dict(self._scene_code_drafts)


class FailingCreativeClient:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def generate_artifacts(
        self,
        *,
        ocr_result: object,
        target_duration_sec: int,
        language: str,
    ) -> CreativeGenerationArtifacts:
        raise self._error

    async def generate_scene_code_drafts(
        self,
        *,
        paper_analysis,
        director_plan,
        storyboard,
        language: str,
    ) -> dict[int, SceneCodeDraftSpec]:
        del paper_analysis, director_plan, storyboard, language
        raise self._error

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_runner_emits_progress_and_complete_callbacks() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = RecordingRenderClient()
    merge_client = RecordingMergeClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        merge_client,
    )

    await runner.run(
        VideoGenerateJobPayload(
            video_job_id="vjob_1",
            document_id="doc_1",
            actor_id="user_1",
            target_duration_sec=60,
            voice="alloy",
            language="en",
            request_id="req_1",
            attempt=1,
        )
    )

    assert callback_client.events[0][0] == "progress"
    assert callback_client.events[-1][0] == "complete"
    assert storage_client.downloads == ["documents/doc_1/artifacts/ocr/raw.json"]
    assert storage_client.uploads == [
        ("videos/vjob_1/artifacts/summary.json", "application/json"),
        ("videos/vjob_1/artifacts/director_plan.json", "application/json"),
        ("videos/vjob_1/artifacts/storyboard.json", "application/json"),
        ("videos/vjob_1/artifacts/scenes.md", "text/markdown"),
        ("videos/vjob_1/artifacts/audio/scene-01.wav", "audio/wav"),
        ("videos/vjob_1/artifacts/audio/scene-02.wav", "audio/wav"),
        ("videos/vjob_1/artifacts/audio/scene-03.wav", "audio/wav"),
        ("videos/vjob_1/artifacts/audio/scene-04.wav", "audio/wav"),
        ("videos/vjob_1/artifacts/audio/scene-05.wav", "audio/wav"),
        ("videos/vjob_1/artifacts/code/scene-01.py", "text/x-python"),
        ("videos/vjob_1/artifacts/code/scene-02.py", "text/x-python"),
        ("videos/vjob_1/artifacts/code/scene-03.py", "text/x-python"),
        ("videos/vjob_1/artifacts/code/scene-04.py", "text/x-python"),
        ("videos/vjob_1/artifacts/code/scene-05.py", "text/x-python"),
        ("videos/vjob_1/artifacts/render/scene-01.mp4", "video/mp4"),
        ("videos/vjob_1/artifacts/render/scene-01.log", "text/plain"),
        ("videos/vjob_1/artifacts/render/scene-02.mp4", "video/mp4"),
        ("videos/vjob_1/artifacts/render/scene-02.log", "text/plain"),
        ("videos/vjob_1/artifacts/render/scene-03.mp4", "video/mp4"),
        ("videos/vjob_1/artifacts/render/scene-03.log", "text/plain"),
        ("videos/vjob_1/artifacts/render/scene-04.mp4", "video/mp4"),
        ("videos/vjob_1/artifacts/render/scene-04.log", "text/plain"),
        ("videos/vjob_1/artifacts/render/scene-05.mp4", "video/mp4"),
        ("videos/vjob_1/artifacts/render/scene-05.log", "text/plain"),
        ("videos/vjob_1/artifacts/merge.log", "text/plain"),
        ("videos/vjob_1/final.mp4", "video/mp4"),
    ]

    complete_payload = callback_client.events[-1][2]
    assert complete_payload["artifact_keys"] == [
        "videos/vjob_1/artifacts/summary.json",
        "videos/vjob_1/artifacts/director_plan.json",
        "videos/vjob_1/artifacts/storyboard.json",
        "videos/vjob_1/artifacts/scenes.md",
        "videos/vjob_1/artifacts/audio/scene-01.wav",
        "videos/vjob_1/artifacts/audio/scene-02.wav",
        "videos/vjob_1/artifacts/audio/scene-03.wav",
        "videos/vjob_1/artifacts/audio/scene-04.wav",
        "videos/vjob_1/artifacts/audio/scene-05.wav",
        "videos/vjob_1/artifacts/code/scene-01.py",
        "videos/vjob_1/artifacts/code/scene-02.py",
        "videos/vjob_1/artifacts/code/scene-03.py",
        "videos/vjob_1/artifacts/code/scene-04.py",
        "videos/vjob_1/artifacts/code/scene-05.py",
        "videos/vjob_1/artifacts/render/scene-01.mp4",
        "videos/vjob_1/artifacts/render/scene-01.log",
        "videos/vjob_1/artifacts/render/scene-02.mp4",
        "videos/vjob_1/artifacts/render/scene-02.log",
        "videos/vjob_1/artifacts/render/scene-03.mp4",
        "videos/vjob_1/artifacts/render/scene-03.log",
        "videos/vjob_1/artifacts/render/scene-04.mp4",
        "videos/vjob_1/artifacts/render/scene-04.log",
        "videos/vjob_1/artifacts/render/scene-05.mp4",
        "videos/vjob_1/artifacts/render/scene-05.log",
        "videos/vjob_1/artifacts/merge.log",
    ]
    assert len(tts_client.requests) == 5
    assert tts_client.requests[0][3]
    assert len(render_client.requests) == 5
    assert merge_client.requests == [([1, 2, 3, 4, 5], "720p")]

    stages_seen = [
        event[2]["pipeline_stage"]
        for event in callback_client.events
        if event[0] == "progress"
    ]
    assert "code_validating" in stages_seen
    assert "rendering" in stages_seen
    assert "merging" in stages_seen

    tts_done_events = [
        event
        for event in callback_client.events
        if event[0] == "progress"
        and event[2].get("pipeline_stage") == "tts_generating"
        and event[2].get("scene", {}).get("status") == "done"
    ]
    assert tts_done_events[0][2]["scene"]["audio_object_key"] == (
        "videos/vjob_1/artifacts/audio/scene-01.wav"
    )
    assert tts_done_events[0][2]["scene"]["actual_audio_duration_ms"] == 12_345

    code_done_events = [
        event
        for event in callback_client.events
        if event[0] == "progress"
        and event[2].get("pipeline_stage") == "code_validating"
        and event[2].get("scene", {}).get("status") == "done"
    ]
    assert len(code_done_events) == 5
    assert code_done_events[0][2]["scene"]["manim_code_object_key"] == (
        "videos/vjob_1/artifacts/code/scene-01.py"
    )
    assert code_done_events[-1][2]["quality_gate"]["code_valid"] is True

    render_done_events = [
        event
        for event in callback_client.events
        if event[0] == "progress"
        and event[2].get("pipeline_stage") == "rendering"
        and event[2].get("scene", {}).get("status") == "done"
    ]
    assert len(render_done_events) == 5
    assert render_done_events[0][2]["scene"]["video_object_key"] == (
        "videos/vjob_1/artifacts/render/scene-01.mp4"
    )
    assert render_done_events[-1][2]["quality_gate"]["render_valid"] is True
    assert complete_payload["duration_sec"] == 61.725


@pytest.mark.asyncio
async def test_runner_maps_validation_failures_to_storyboard_invalid() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = RecordingRenderClient()
    merge_client = RecordingMergeClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        merge_client,
    )

    try:
        SceneSpec(
            scene_index=1,
            title="Broken",
            template_type="title",
            purpose="Invalid duration",
            narration_text="Too short",
            planned_duration_ms=100,
            visual_elements=["headline"],
        )
    except ValidationError as error:
        await runner.report_failure(
            VideoGenerateJobPayload(
                video_job_id="vjob_2",
                document_id="doc_2",
                actor_id="user_2",
                target_duration_sec=60,
                voice="alloy",
                language="en",
                attempt=1,
            ),
            error,
        )

    assert callback_client.events == [
        (
            "fail",
            "vjob_2",
            {
                "pipeline_stage": "storyboard_validating",
                "error_code": "SCENESPEC_INVALID",
                "error_message": callback_client.events[0][2]["error_message"],
            },
        ),
    ]


@pytest.mark.asyncio
async def test_runner_maps_code_validation_failures_to_code_validating() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = RecordingRenderClient()
    merge_client = RecordingMergeClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        merge_client,
    )

    await runner.report_failure(
        VideoGenerateJobPayload(
            video_job_id="vjob_3",
            document_id="doc_3",
            actor_id="user_3",
            target_duration_sec=60,
            voice="alloy",
            language="en",
            attempt=1,
        ),
        CodeValidationError("Forbidden call in generated code: open"),
    )

    assert callback_client.events == [
        (
            "fail",
            "vjob_3",
            {
                "pipeline_stage": "code_validating",
                "error_code": "CODE_VALIDATION_FAILED",
                "error_message": "Forbidden call in generated code: open",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_runner_maps_merge_failures_to_merging_stage() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = RecordingRenderClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        FailingMergeClient(MergeExecutionError("ffmpeg concat failed")),
    )

    await runner.report_failure(
        VideoGenerateJobPayload(
            video_job_id="vjob_4",
            document_id="doc_4",
            actor_id="user_4",
            target_duration_sec=60,
            voice="alloy",
            language="en",
            attempt=1,
        ),
        MergeExecutionError("ffmpeg concat failed"),
    )

    assert callback_client.events == [
        (
            "fail",
            "vjob_4",
            {
                "pipeline_stage": "merging",
                "error_code": "MERGE_EXECUTION_FAILED",
                "error_message": "ffmpeg concat failed",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_runner_maps_render_timeout_failures_to_rendering_stage() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = RecordingRenderClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        RecordingMergeClient(),
    )

    await runner.report_failure(
        VideoGenerateJobPayload(
            video_job_id="vjob_timeout",
            document_id="doc_timeout",
            actor_id="user_timeout",
            target_duration_sec=60,
            voice="alloy",
            language="en",
            attempt=1,
        ),
        RenderTimeoutError("Timed out rendering scene 1 in Daytona"),
    )

    assert callback_client.events == [
        (
            "fail",
            "vjob_timeout",
            {
                "pipeline_stage": "rendering",
                "error_code": "RENDER_TIMEOUT",
                "error_message": "Timed out rendering scene 1 in Daytona",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_runner_maps_audio_sync_failures_to_merging_stage() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = RecordingRenderClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        FailingMergeClient(AudioSyncValidationError("Drift exceeded threshold")),
    )

    await runner.report_failure(
        VideoGenerateJobPayload(
            video_job_id="vjob_5",
            document_id="doc_5",
            actor_id="user_5",
            target_duration_sec=60,
            voice="alloy",
            language="en",
            attempt=1,
        ),
        AudioSyncValidationError("Drift exceeded threshold"),
    )

    assert callback_client.events == [
        (
            "fail",
            "vjob_5",
            {
                "pipeline_stage": "merging",
                "error_code": "AUDIO_SYNC_INVALID",
                "error_message": "Drift exceeded threshold",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_runner_downgrades_render_profile_after_render_failure() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = DowngradingRenderClient()
    merge_client = RecordingMergeClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        merge_client,
    )

    await runner.run(
        VideoGenerateJobPayload(
            video_job_id="vjob_fallback",
            document_id="doc_fallback",
            actor_id="user_fallback",
            target_duration_sec=60,
            voice="alloy",
            language="en",
            attempt=1,
        )
    )

    assert render_client.requests[0] == (1, "720p")
    assert render_client.requests[1] == (1, "480p")
    assert merge_client.requests == [([1, 2, 3, 4, 5], "480p")]

    fallback_events = [
        event
        for event in callback_client.events
        if event[0] == "progress"
        and event[2].get("fallback_reason") == "render_profile_downgrade_480p"
    ]
    assert fallback_events
    assert fallback_events[0][2]["render_profile"] == "480p"
    assert callback_client.events[-1][2]["resolution"] == "854x480"


@pytest.mark.asyncio
async def test_runner_does_not_retry_render_configuration_failures() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = ConfigFailingRenderClient()
    merge_client = RecordingMergeClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        merge_client,
    )

    with pytest.raises(RenderConfigurationError):
        await runner.run(
            VideoGenerateJobPayload(
                video_job_id="vjob_config",
                document_id="doc_config",
                actor_id="user_config",
                target_duration_sec=60,
                voice="alloy",
                language="en",
                attempt=1,
            )
        )

    assert render_client.requests == [(1, "720p")]


@pytest.mark.asyncio
async def test_runner_uses_creative_client_artifacts_when_available() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = RecordingRenderClient()
    merge_client = RecordingMergeClient()
    creative_client = RecordingCreativeClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        merge_client,
        creative_client=creative_client,
    )

    await runner.run(
        VideoGenerateJobPayload(
            video_job_id="vjob_ai",
            document_id="doc_ai",
            actor_id="user_ai",
            target_duration_sec=60,
            voice="alloy",
            language="en",
            attempt=1,
        )
    )

    assert creative_client.requests == [(60, "en")]
    assert creative_client.scene_draft_requests == [[12_345, 12_345, 12_345, 12_345, 12_345]]
    assert ("videos/vjob_ai/artifacts/paper_analysis.json", "application/json") in storage_client.uploads
    assert ("videos/vjob_ai/artifacts/script_plan.json", "application/json") in storage_client.uploads
    assert (
        "videos/vjob_ai/artifacts/creative/scene-01-critique.json",
        "application/json",
    ) in storage_client.uploads
    assert (
        "videos/vjob_ai/artifacts/creative/scene-01-revised.py",
        "text/x-python",
    ) in storage_client.uploads
    assert "AI script for scene 1." in tts_client.requests[0][0]
    assert "AIScene01" in render_client.requests[0][1]


@pytest.mark.asyncio
async def test_runner_fails_fast_when_creative_generation_fails() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = RecordingRenderClient()
    merge_client = RecordingMergeClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        merge_client,
        creative_client=FailingCreativeClient(
            CreativeGenerationError("creative provider unavailable")
        ),
    )

    with pytest.raises(CreativeGenerationError, match="creative provider unavailable"):
        await runner.run(
            VideoGenerateJobPayload(
                video_job_id="vjob_creative_fail",
                document_id="doc_creative_fail",
                actor_id="user_creative_fail",
                target_duration_sec=60,
                voice="alloy",
                language="en",
                attempt=1,
            )
        )

    assert tts_client.requests == []
    assert render_client.requests == []
    assert merge_client.requests == []
    assert not any(
        event[0] == "progress" and event[2].get("fallback_reason") == "creative_ai_fallback_deterministic"
        for event in callback_client.events
    )


@pytest.mark.asyncio
async def test_runner_fails_fast_when_creative_config_is_invalid() -> None:
    callback_client = RecordingCallbackClient()
    storage_client = RecordingStorageClient()
    tts_client = RecordingTtsClient()
    render_client = RecordingRenderClient()
    merge_client = RecordingMergeClient()
    runner = PipelineRunner(
        callback_client,
        storage_client,
        tts_client,
        render_client,
        merge_client,
        creative_client=FailingCreativeClient(
            CreativeConfigurationError("VIDEO_CREATIVE_API_KEY is required")
        ),
    )

    with pytest.raises(
        CreativeConfigurationError,
        match="VIDEO_CREATIVE_API_KEY is required",
    ):
        await runner.run(
            VideoGenerateJobPayload(
                video_job_id="vjob_creative_config_fail",
                document_id="doc_creative_config_fail",
                actor_id="user_creative_config_fail",
                target_duration_sec=60,
                voice="alloy",
                language="en",
                attempt=1,
            )
        )

    assert tts_client.requests == []
    assert render_client.requests == []

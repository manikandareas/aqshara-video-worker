from __future__ import annotations

from typing import Protocol

from aqshara_video_worker.clients.creative_generation_client import (
    CreativeGenerationArtifacts,
)
from aqshara_video_worker.schemas import (
    DirectorPlanSpec,
    PaperAnalysisSpec,
    SceneCodeDraftSpec,
    SceneRenderQASpec,
    StoryboardSpec,
    VideoLanguage,
)


class TtsClient(Protocol):
    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: VideoLanguage,
        voice_instruction: str | None = None,
    ) -> bytes: ...

    @staticmethod
    def measure_duration_ms(audio_bytes: bytes) -> int: ...

    async def close(self) -> None: ...


class CreativeGenerationClient(Protocol):
    async def generate_artifacts(
        self,
        *,
        ocr_result: object,
        target_duration_sec: int,
        language: str,
    ) -> CreativeGenerationArtifacts: ...

    async def generate_scene_code_drafts(
        self,
        *,
        paper_analysis: PaperAnalysisSpec,
        director_plan: DirectorPlanSpec,
        storyboard: StoryboardSpec,
        language: str,
    ) -> dict[int, SceneCodeDraftSpec]: ...

    async def review_rendered_scene(
        self,
        *,
        scene_index: int,
        scene_json: str,
        scene_code: str,
        render_profile: str,
        sample_frames: list[bytes],
    ) -> SceneRenderQASpec: ...

    async def revise_scene_code_for_render_qa(
        self,
        *,
        scene_json: str,
        scene_code: str,
        review: SceneRenderQASpec,
    ) -> str: ...

    async def close(self) -> None: ...

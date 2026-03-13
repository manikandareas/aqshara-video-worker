from __future__ import annotations

from typing import Protocol

from aqshara_video_worker.clients.creative_generation_client import (
    CreativeGenerationArtifacts,
)
from aqshara_video_worker.schemas import (
    DirectorPlanSpec,
    PaperAnalysisSpec,
    SceneCodeDraftSpec,
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

    async def close(self) -> None: ...

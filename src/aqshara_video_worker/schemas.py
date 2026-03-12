from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SchemaVersion = Literal["2026-03-11"]
TemplateType = Literal["title", "bullet", "pipeline", "chart", "conclusion"]
SceneStatus = Literal["pending", "processing", "done", "error"]
VideoLanguage = Literal["en", "id"]
PipelineStage = Literal[
    "queued",
    "preprocessing",
    "summarizing",
    "storyboarding",
    "storyboard_validating",
    "tts_generating",
    "code_validating",
    "rendering",
    "merging",
    "uploading",
    "completed",
    "failed",
]


class VideoGenerateJobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_job_id: str
    document_id: str
    actor_id: str
    target_duration_sec: int = Field(ge=30, le=90)
    voice: str = Field(min_length=1)
    language: VideoLanguage
    request_id: str | None = None
    attempt: int = Field(ge=1)


class VideoGenerateCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: SchemaVersion
    command_id: str
    topic: str
    job_id: str
    document_id: str
    owner_id: str
    request_id: str | None = None
    attempt: int = Field(ge=1)
    target_duration_sec: int = Field(ge=30, le=90)
    voice: str = Field(min_length=1)
    language: VideoLanguage
    render_profile: Literal["480p", "720p"] | None = None
    ocr_object_key: str = Field(min_length=1)
    output_prefix: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    trace_id: str | None = None
    occurred_at: str


class SceneConstraintSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_mobjects: int = Field(default=12, ge=1, le=64)
    max_animations: int = Field(default=10, ge=1, le=64)
    allow_camera_movement: bool = False


class SceneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_index: int = Field(ge=1)
    title: str = Field(min_length=1)
    template_type: TemplateType
    purpose: str = Field(min_length=1)
    narration_text: str = Field(min_length=1)
    planned_duration_ms: int = Field(ge=3000, le=30000)
    visual_elements: list[str] = Field(min_length=1)
    transition_in: str | None = None
    transition_out: str | None = None
    constraints: SceneConstraintSpec = Field(default_factory=SceneConstraintSpec)


class StructuredSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    method: str = Field(min_length=1)
    result: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)
    source_excerpt_count: int = Field(ge=0)


class StoryboardSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1)
    hook: str = Field(min_length=1)
    target_audience: str = Field(min_length=1)
    estimated_length_sec: int = Field(ge=30, le=90)
    key_insight: str = Field(min_length=1)
    narrative_arc: list[str] = Field(min_length=4, max_length=4)
    scenes: list[SceneSpec] = Field(min_length=3, max_length=5)
    transitions: list[str] = Field(default_factory=list)
    color_palette: list[str] = Field(min_length=3, max_length=5)
    implementation_order: list[int] = Field(min_length=3, max_length=5)


class InternalVideoQualityGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storyboard_valid: bool
    code_valid: bool
    render_valid: bool
    audio_sync_valid: bool


class InternalVideoSceneProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_index: int = Field(ge=1)
    template_type: TemplateType | None = None
    status: SceneStatus
    planned_duration_ms: int | None = Field(default=None, ge=0)
    actual_audio_duration_ms: int | None = Field(default=None, ge=0)
    audio_object_key: str | None = None
    manim_code_object_key: str | None = None
    video_object_key: str | None = None


class InternalVideoMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elapsed_ms: int | None = Field(default=None, ge=0)


class InternalVideoProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_stage: PipelineStage
    progress_pct: int = Field(ge=0, le=100)
    message: str | None = None
    fallback_applied: bool | None = None
    fallback_reason: str | None = None
    validation_errors: list[str] | None = None
    render_profile: Literal["480p", "720p"] | None = None
    quality_gate: InternalVideoQualityGate | None = None
    scene: InternalVideoSceneProgress | None = None
    metrics: InternalVideoMetrics | None = None


class InternalVideoComplete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_video_object_key: str
    final_thumbnail_object_key: str | None = None
    duration_sec: float = Field(ge=0)
    resolution: str
    artifact_keys: list[str] | None = None


class InternalVideoFail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline_stage: PipelineStage = "failed"
    error_code: str
    error_message: str
    is_retryable: bool | None = None
    failed_scene_index: int | None = Field(default=None, ge=1)
    debug_artifact_keys: list[str] | None = None

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SchemaVersion = Literal["2026-03-11"]
TemplateType = Literal[
    "title",
    "bullet",
    "pipeline",
    "chart",
    "conclusion",
    "hook",
    "problem",
    "mechanism",
    "evidence",
    "takeaway",
]
SceneKind = Literal["hook", "problem", "mechanism", "evidence", "takeaway", "fallback"]
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
    "scene_reviewing",
    "scene_revising",
    "merging",
    "uploading",
    "completed",
    "failed",
]
CameraMode = Literal["static", "focus", "pan", "reveal"]
TransitionStrategy = Literal["fade", "transform", "highlight", "zoom"]
MotionType = Literal["write", "fade", "transform", "highlight", "grow"]


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


class CameraPlanSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: CameraMode = "static"
    target: str | None = None
    scale: float = Field(default=1.0, ge=0.5, le=1.5)


class NarrationCueSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice_instruction: str = Field(default="", max_length=240)
    pause_after_ms: int = Field(default=250, ge=0, le=1500)
    emphasis_terms: list[str] = Field(default_factory=list)


class ChartDatumSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    value: float = Field(ge=0.1, le=100.0)
    emphasis: bool = False


class SceneBeatSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beat_index: int = Field(ge=1)
    visual: str = Field(min_length=1)
    narration: str = Field(min_length=1)
    motion: MotionType = "fade"
    emphasis: str | None = None


class DirectorSceneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_index: int = Field(ge=1)
    scene_kind: SceneKind
    headline: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    visual_goal: str = Field(min_length=1)
    visual_metaphor: str = Field(min_length=1)
    continuity_from: str | None = None
    transition_strategy: TransitionStrategy = "fade"
    camera_plan: CameraPlanSpec = Field(default_factory=CameraPlanSpec)
    beats: list[SceneBeatSpec] = Field(min_length=2, max_length=5)
    emphasis_terms: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    chart_data: list[ChartDatumSpec] | None = None
    narration_cues: NarrationCueSpec = Field(default_factory=NarrationCueSpec)
    planned_duration_ms: int = Field(ge=3000, le=30000)


class SceneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_index: int = Field(ge=1)
    title: str = Field(min_length=1)
    template_type: TemplateType
    scene_kind: SceneKind = "fallback"
    purpose: str = Field(min_length=1)
    narration_text: str = Field(min_length=1)
    planned_duration_ms: int = Field(ge=3000, le=30000)
    target_render_duration_ms: int | None = Field(default=None, ge=3000, le=30000)
    visual_elements: list[str] = Field(min_length=1)
    visual_beats: list[SceneBeatSpec] = Field(default_factory=list)
    emphasis_terms: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    camera_plan: CameraPlanSpec = Field(default_factory=CameraPlanSpec)
    transition_strategy: TransitionStrategy = "fade"
    chart_data: list[ChartDatumSpec] | None = None
    narration_cues: NarrationCueSpec = Field(default_factory=NarrationCueSpec)
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


class PaperAnalysisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    method: str = Field(min_length=1)
    result: str = Field(min_length=1)
    conclusion: str = Field(min_length=1)
    key_entities: list[str] = Field(default_factory=list)
    visual_opportunities: list[str] = Field(default_factory=list)
    misconceptions: list[str] = Field(default_factory=list)


class DirectorPlanSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1)
    style_profile: str = Field(min_length=1)
    visual_language: str = Field(min_length=1)
    narrative_question: str = Field(min_length=1)
    recurring_motifs: list[str] = Field(min_length=2, max_length=5)
    target_story_arc: list[str] = Field(min_length=4, max_length=5)
    quality_score: float = Field(ge=0.0, le=1.0)
    scenes: list[DirectorSceneSpec] = Field(min_length=3, max_length=5)


class SceneScriptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_index: int = Field(ge=1)
    narration_text: str = Field(min_length=1)
    voice_instruction: str = Field(default="", max_length=240)
    emphasis_terms: list[str] = Field(default_factory=list)


class ScriptPlanSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hook_line: str = Field(min_length=1)
    tone: str = Field(min_length=1)
    scenes: list[SceneScriptSpec] = Field(min_length=3, max_length=5)


class SceneCodeCritiqueSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_index: int = Field(ge=1)
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    revision_brief: str = Field(min_length=1)
    requires_revision: bool = True


class SceneRenderQASpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_index: int = Field(ge=1)
    strengths: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    revision_brief: str = Field(min_length=1)
    requires_revision: bool = True
    qa_status: Literal["pass", "revise"] = "revise"


class SceneCodeDraftSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_index: int = Field(ge=1)
    draft_code: str = Field(min_length=1)
    critique: SceneCodeCritiqueSpec
    revised_code: str = Field(min_length=1)


class StoryboardSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(min_length=1)
    hook: str = Field(min_length=1)
    target_audience: str = Field(min_length=1)
    estimated_length_sec: int = Field(ge=30, le=90)
    key_insight: str = Field(min_length=1)
    narrative_arc: list[str] = Field(min_length=4, max_length=5)
    style_profile: str = Field(min_length=1)
    quality_score: float = Field(ge=0.0, le=1.0)
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
    qa_status: Literal["pending", "passed", "revising", "failed"] | None = None
    revision_attempt: int | None = Field(default=None, ge=0)


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

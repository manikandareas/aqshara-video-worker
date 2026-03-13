from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, cast

from openai import APIError
from openai import APIStatusError
from openai import AsyncOpenAI

from aqshara_video_worker.config import WorkerSettings
from aqshara_video_worker.pipeline.storyboard import (
    build_storyboard_artifacts,
    build_storyboard_from_director_plan,
    render_scenes_markdown,
)
from aqshara_video_worker.schemas import (
    DirectorPlanSpec,
    NarrationCueSpec,
    PaperAnalysisSpec,
    SceneCodeCritiqueSpec,
    SceneCodeDraftSpec,
    SceneSpec,
    SceneScriptSpec,
    ScriptPlanSpec,
    StoryboardSpec,
    StructuredSummary,
)


class CreativeConfigurationError(RuntimeError):
    pass


class CreativeGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CreativeGenerationArtifacts:
    summary: StructuredSummary
    paper_analysis: PaperAnalysisSpec
    director_plan: DirectorPlanSpec
    script_plan: ScriptPlanSpec
    storyboard: StoryboardSpec
    scenes_markdown: str
    scene_code_drafts: dict[int, SceneCodeDraftSpec] = field(default_factory=dict)


class OpenAICompatibleCreativeGenerationClient:
    def __init__(
        self,
        settings: WorkerSettings,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._provider = settings.video_creative_provider
        self._api_key = settings.video_creative_api_key
        self._base_url = settings.video_creative_base_url
        self._generation_model = settings.video_creative_generation_model
        self._critique_model = settings.video_creative_critique_model
        self._codegen_model = settings.video_creative_codegen_model
        self._max_revisions = max(0, settings.video_ai_max_revisions)
        self._timeout_sec = settings.video_creative_timeout_sec
        self._enabled = settings.video_ai_creative_enabled
        self._client = client or AsyncOpenAI(
            api_key=settings.video_creative_api_key,
            base_url=settings.video_creative_base_url,
            timeout=settings.video_creative_timeout_sec,
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def close(self) -> None:
        await self._client.close()

    async def generate_artifacts(
        self,
        *,
        ocr_result: object,
        target_duration_sec: int,
        language: str,
    ) -> CreativeGenerationArtifacts:
        self._ensure_generation_ready()

        baseline = build_storyboard_artifacts(ocr_result, target_duration_sec)
        paper_text = _extract_paper_context(ocr_result)
        paper_analysis = await self._generate_paper_analysis(
            paper_text, baseline.summary
        )
        summary = _summary_from_paper_analysis(
            paper_analysis, baseline.summary.source_excerpt_count
        )
        director_plan = await self._generate_director_plan(
            paper_analysis=paper_analysis,
            baseline_director_plan=baseline.director_plan,
            target_duration_sec=target_duration_sec,
        )
        script_plan = await self._generate_script_plan(
            paper_analysis=paper_analysis,
            director_plan=director_plan,
            language=language,
        )
        storyboard = build_storyboard_from_director_plan(
            summary=summary,
            director_plan=director_plan,
            target_duration_sec=target_duration_sec,
        )
        storyboard = _apply_script_plan(storyboard, script_plan)
        return CreativeGenerationArtifacts(
            summary=summary,
            paper_analysis=paper_analysis,
            director_plan=director_plan,
            script_plan=script_plan,
            storyboard=storyboard,
            scenes_markdown=render_scenes_markdown(storyboard, director_plan),
        )

    async def generate_scene_code_drafts(
        self,
        *,
        paper_analysis: PaperAnalysisSpec,
        director_plan: DirectorPlanSpec,
        storyboard: StoryboardSpec,
        language: str,
    ) -> dict[int, SceneCodeDraftSpec]:
        return await self._generate_scene_code_drafts(
            paper_analysis=paper_analysis,
            director_plan=director_plan,
            storyboard=storyboard,
            language=language,
        )

    async def _generate_paper_analysis(
        self,
        paper_text: str,
        baseline_summary: StructuredSummary,
    ) -> PaperAnalysisSpec:
        schema_hint = {
            "topic": "string",
            "problem": "string",
            "method": "string",
            "result": "string",
            "conclusion": "string",
            "key_entities": ["string"],
            "visual_opportunities": ["string"],
            "misconceptions": ["string"],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research-video creative analyst. Extract the strongest educational angle "
                    "from the paper. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Paper OCR markdown:\n{paper_text}\n\n"
                    f"Baseline summary:\n{baseline_summary.model_dump_json(indent=2)}\n\n"
                    f"Return JSON matching this shape:\n{json.dumps(schema_hint, indent=2)}"
                ),
            },
        ]
        payload = await self._chat_json(self._generation_model, messages)
        return PaperAnalysisSpec.model_validate(payload)

    async def _generate_director_plan(
        self,
        *,
        paper_analysis: PaperAnalysisSpec,
        baseline_director_plan: DirectorPlanSpec,
        target_duration_sec: int,
    ) -> DirectorPlanSpec:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a creative director for Manim educational videos. "
                    "Return only valid JSON for DirectorPlanSpec. Keep 5 scenes max, and each scene should be visually specific. "
                    "Use only these enum values: transition_strategy in [fade, transform, highlight, zoom], "
                    "camera_plan.mode in [static, focus, pan, reveal], beat.motion in [write, fade, transform, highlight, grow]. "
                    "If chart_data is present, values must stay within 0.1 to 100."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Paper analysis:\n{paper_analysis.model_dump_json(indent=2)}\n\n"
                    f"Baseline director plan:\n{baseline_director_plan.model_dump_json(indent=2)}\n\n"
                    f"Target duration: {target_duration_sec} seconds.\n"
                    "Make the plan more cinematic and conceptually sharp, keeping scene kinds focused on hook/problem/mechanism/evidence/takeaway. "
                    "Be expressive in visual/beat descriptions, but keep enum fields strictly within the allowed values."
                ),
            },
        ]
        payload = await self._chat_json(self._generation_model, messages)
        payload = _normalize_director_plan_payload(payload, baseline_director_plan)
        return DirectorPlanSpec.model_validate(payload)

    async def _generate_script_plan(
        self,
        *,
        paper_analysis: PaperAnalysisSpec,
        director_plan: DirectorPlanSpec,
        language: str,
    ) -> ScriptPlanSpec:
        schema_hint = {
            "hook_line": "string",
            "tone": "string",
            "scenes": [
                {
                    "scene_index": 1,
                    "narration_text": "string",
                    "voice_instruction": "string",
                    "emphasis_terms": ["string"],
                }
            ],
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You write concise narration for visual explainers. Return only valid JSON. "
                    "Narration must align with the director plan and be easy for TTS."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Language: {language}\n\n"
                    f"Paper analysis:\n{paper_analysis.model_dump_json(indent=2)}\n\n"
                    f"Director plan:\n{director_plan.model_dump_json(indent=2)}\n\n"
                    f"Return JSON matching this shape:\n{json.dumps(schema_hint, indent=2)}"
                ),
            },
        ]
        payload = await self._chat_json(self._generation_model, messages)
        script_plan = ScriptPlanSpec.model_validate(payload)
        if len(script_plan.scenes) != len(director_plan.scenes):
            raise CreativeGenerationError(
                "Script plan scene count did not match director plan"
            )
        return script_plan

    async def _generate_scene_code_drafts(
        self,
        *,
        paper_analysis: PaperAnalysisSpec,
        director_plan: DirectorPlanSpec,
        storyboard: StoryboardSpec,
        language: str,
    ) -> dict[int, SceneCodeDraftSpec]:
        director_scenes_by_index = {
            scene.scene_index: scene for scene in director_plan.scenes
        }
        drafts: dict[int, SceneCodeDraftSpec] = {}
        for scene in storyboard.scenes:
            director_scene = director_scenes_by_index[scene.scene_index]
            draft_code = await self._generate_scene_code(
                paper_analysis=paper_analysis,
                director_scene_json=director_scene.model_dump_json(indent=2),
                scene_json=scene.model_dump_json(indent=2),
                language=language,
            )
            critique = await self._critique_scene_code(
                scene_json=scene.model_dump_json(indent=2),
                scene_code=draft_code,
            )
            revised_code = draft_code
            if critique.requires_revision and self._max_revisions > 0:
                revised_code = await self._revise_scene_code(
                    scene_json=scene.model_dump_json(indent=2),
                    scene_code=draft_code,
                    critique=critique,
                )
            drafts[scene.scene_index] = SceneCodeDraftSpec(
                scene_index=scene.scene_index,
                draft_code=draft_code,
                critique=critique,
                revised_code=revised_code,
            )
        return drafts

    def _ensure_generation_ready(self) -> None:
        if not self.is_enabled:
            raise CreativeConfigurationError("Creative generation is not enabled")

        required_settings = (
            (self._api_key, "VIDEO_CREATIVE_API_KEY"),
            (self._base_url, "VIDEO_CREATIVE_BASE_URL"),
            (self._generation_model, "VIDEO_CREATIVE_GENERATION_MODEL"),
            (self._critique_model, "VIDEO_CREATIVE_CRITIQUE_MODEL"),
            (self._codegen_model, "VIDEO_CREATIVE_CODEGEN_MODEL"),
        )
        for value, env_name in required_settings:
            if value:
                continue
            raise CreativeConfigurationError(
                f"{env_name} is required when creative AI is enabled"
            )

    async def _generate_scene_code(
        self,
        *,
        paper_analysis: PaperAnalysisSpec,
        director_scene_json: str,
        scene_json: str,
        language: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Write a single ManimCE scene class body for the provided structured scene. "
                    "Use only Manim imports, no filesystem/network access, and no markdown fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Language: {language}\n\n"
                    f"Paper analysis:\n{paper_analysis.model_dump_json(indent=2)}\n\n"
                    f"Director scene:\n{director_scene_json}\n\n"
                    f"Scene contract:\n{scene_json}\n\n"
                    "Return complete Python code with `from manim import *`, `config.background_color`, "
                    "and exactly one Scene or MovingCameraScene subclass."
                ),
            },
        ]
        return _strip_code_fences(await self._chat_text(self._codegen_model, messages))

    async def _critique_scene_code(
        self,
        *,
        scene_json: str,
        scene_code: str,
    ) -> SceneCodeCritiqueSpec:
        schema_hint = {
            "scene_index": 1,
            "strengths": ["string"],
            "issues": ["string"],
            "revision_brief": "string",
            "requires_revision": True,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict Manim scene reviewer. Judge visual clarity, animation continuity, "
                    "camera use, and likely syntax/runtime issues. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scene contract:\n{scene_json}\n\n"
                    f"Candidate Manim code:\n{scene_code}\n\n"
                    f"Return JSON matching this shape:\n{json.dumps(schema_hint, indent=2)}"
                ),
            },
        ]
        payload = await self._chat_json(self._critique_model, messages)
        return SceneCodeCritiqueSpec.model_validate(payload)

    async def _revise_scene_code(
        self,
        *,
        scene_json: str,
        scene_code: str,
        critique: SceneCodeCritiqueSpec,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Revise the ManimCE scene code to address the critique. "
                    "Return only Python code with the same class-name intent and no markdown fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Scene contract:\n{scene_json}\n\n"
                    f"Original code:\n{scene_code}\n\n"
                    f"Critique:\n{critique.model_dump_json(indent=2)}"
                ),
            },
        ]
        return _strip_code_fences(await self._chat_text(self._codegen_model, messages))

    async def _chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        raw_content = await self._chat_text(model, messages, json_mode=True)
        payload = _load_json_object(raw_content)
        if not isinstance(payload, dict):
            raise CreativeGenerationError(
                "OpenAI creative response must decode to a JSON object"
            )
        return payload

    async def _chat_text(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=cast(Any, messages),  # pyright: ignore[reportArgumentType]
                response_format=cast(
                    Any,
                    {"type": "json_object"} if json_mode else None,
                ),  # pyright: ignore[reportArgumentType]
                timeout=self._timeout_sec,
            )
        except APIStatusError as error:
            detail = (getattr(error, "body", None) or {}).get("message", "")
            raise CreativeGenerationError(
                detail or "OpenAI creative request failed"
            ) from error
        except APIError as error:
            raise CreativeGenerationError("OpenAI creative request failed") from error

        choice = response.choices[0]
        content = choice.message.content if getattr(choice, "message", None) else None
        if not content:
            raise CreativeGenerationError("OpenAI creative response was empty")
        return content


def create_creative_generation_client(
    settings: WorkerSettings,
) -> OpenAICompatibleCreativeGenerationClient | None:
    if not settings.video_ai_creative_enabled:
        return None
    if settings.video_creative_provider != "openai_compatible":
        raise CreativeConfigurationError(
            f"Unsupported creative provider: {settings.video_creative_provider}"
        )
    return OpenAICompatibleCreativeGenerationClient(settings)


def _extract_paper_context(ocr_result: object) -> str:
    if not isinstance(ocr_result, dict):
        return ""
    pages = ocr_result.get("pages")
    if not isinstance(pages, list):
        return ""
    fragments: list[str] = []
    for page in pages[:5]:
        if not isinstance(page, dict):
            continue
        markdown = page.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            fragments.append(markdown.strip())
    text = "\n\n".join(fragments)
    return text[:12_000]


def _summary_from_paper_analysis(
    paper_analysis: PaperAnalysisSpec,
    source_excerpt_count: int,
) -> StructuredSummary:
    return StructuredSummary(
        topic=paper_analysis.topic,
        problem=paper_analysis.problem,
        method=paper_analysis.method,
        result=paper_analysis.result,
        conclusion=paper_analysis.conclusion,
        source_excerpt_count=source_excerpt_count,
    )


def _apply_script_plan(
    storyboard: StoryboardSpec,
    script_plan: ScriptPlanSpec,
) -> StoryboardSpec:
    script_by_scene = {scene.scene_index: scene for scene in script_plan.scenes}
    updated_scenes: list[SceneSpec] = []
    for scene in storyboard.scenes:
        script_scene = script_by_scene.get(scene.scene_index)
        if script_scene is None:
            updated_scenes.append(scene)
            continue

        emphasis_terms = script_scene.emphasis_terms or scene.emphasis_terms
        narration_cue_emphasis_terms = (
            script_scene.emphasis_terms or scene.narration_cues.emphasis_terms
        )
        updated_scenes.append(
            scene.model_copy(
                update={
                    "narration_text": script_scene.narration_text,
                    "emphasis_terms": emphasis_terms,
                    "narration_cues": scene.narration_cues.model_copy(
                        update={
                            "voice_instruction": script_scene.voice_instruction,
                            "emphasis_terms": narration_cue_emphasis_terms,
                        }
                    ),
                }
            )
        )
    return storyboard.model_copy(
        update={
            "hook": script_plan.hook_line,
            "scenes": updated_scenes,
        }
    )


def _strip_code_fences(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _load_json_object(raw_content: str) -> object:
    cleaned = _strip_code_fences(raw_content).strip().lstrip("\ufeff")
    if not cleaned:
        raise CreativeGenerationError("OpenAI creative response was empty")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidate = _extract_first_json_object(cleaned)
    if candidate is None:
        raise CreativeGenerationError("OpenAI creative response was not valid JSON")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as error:
        raise CreativeGenerationError(
            "OpenAI creative response was not valid JSON"
        ) from error


def _extract_first_json_object(value: str) -> str | None:
    start = value.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(value)):
        char = value[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]

    return None


def _normalize_director_plan_payload(
    payload: dict[str, object],
    baseline_director_plan: DirectorPlanSpec,
) -> dict[str, object]:
    normalized: dict[str, object] = dict(payload)
    normalized["quality_score"] = _clamp_float(
        normalized.get("quality_score"),
        minimum=0.0,
        maximum=1.0,
        fallback=baseline_director_plan.quality_score,
    )

    recurring_motifs = normalized.get("recurring_motifs")
    if not isinstance(recurring_motifs, list) or not 2 <= len(recurring_motifs) <= 5:
        normalized["recurring_motifs"] = baseline_director_plan.recurring_motifs

    target_story_arc = normalized.get("target_story_arc")
    if not isinstance(target_story_arc, list) or not 4 <= len(target_story_arc) <= 5:
        normalized["target_story_arc"] = baseline_director_plan.target_story_arc

    raw_scenes = normalized.get("scenes")
    if not isinstance(raw_scenes, list) or not 3 <= len(raw_scenes) <= 5:
        normalized["scenes"] = baseline_director_plan.model_dump()["scenes"]
        return normalized

    baseline_scenes = {
        scene.scene_index: scene for scene in baseline_director_plan.scenes
    }
    repaired_scenes: list[dict[str, Any]] = []
    for position, raw_scene in enumerate(raw_scenes):
        if not isinstance(raw_scene, dict):
            fallback_scene = baseline_director_plan.scenes[
                min(position, len(baseline_director_plan.scenes) - 1)
            ]
            repaired_scenes.append(fallback_scene.model_dump())
            continue

        scene_index = raw_scene.get("scene_index")
        if isinstance(scene_index, int):
            fallback_scene = baseline_scenes.get(scene_index)
        else:
            fallback_scene = None
        if fallback_scene is None:
            fallback_scene = baseline_director_plan.scenes[
                min(position, len(baseline_director_plan.scenes) - 1)
            ]

        repaired_scene = dict(raw_scene)
        repaired_scene["transition_strategy"] = _coerce_enum(
            repaired_scene.get("transition_strategy"),
            allowed=("fade", "transform", "highlight", "zoom"),
            fallback=fallback_scene.transition_strategy,
        )

        camera_plan = repaired_scene.get("camera_plan")
        if isinstance(camera_plan, dict):
            repaired_camera_plan = dict(camera_plan)
        else:
            repaired_camera_plan = fallback_scene.camera_plan.model_dump()
        repaired_camera_plan["mode"] = _coerce_enum(
            repaired_camera_plan.get("mode"),
            allowed=("static", "focus", "pan", "reveal"),
            fallback=fallback_scene.camera_plan.mode,
        )
        repaired_camera_plan["scale"] = _clamp_float(
            repaired_camera_plan.get("scale"),
            minimum=0.5,
            maximum=1.5,
            fallback=fallback_scene.camera_plan.scale,
        )
        if not isinstance(repaired_camera_plan.get("target"), str):
            repaired_camera_plan["target"] = fallback_scene.camera_plan.target
        repaired_scene["camera_plan"] = repaired_camera_plan

        beats = repaired_scene.get("beats")
        if isinstance(beats, list) and beats:
            repaired_beats: list[dict[str, Any]] = []
            fallback_beats = {beat.beat_index: beat for beat in fallback_scene.beats}
            for beat_position, raw_beat in enumerate(beats):
                if not isinstance(raw_beat, dict):
                    fallback_beat = fallback_scene.beats[
                        min(beat_position, len(fallback_scene.beats) - 1)
                    ]
                    repaired_beats.append(fallback_beat.model_dump())
                    continue
                beat_index = raw_beat.get("beat_index")
                if isinstance(beat_index, int):
                    fallback_beat = fallback_beats.get(beat_index)
                else:
                    fallback_beat = None
                if fallback_beat is None:
                    fallback_beat = fallback_scene.beats[
                        min(beat_position, len(fallback_scene.beats) - 1)
                    ]
                repaired_beat = dict(raw_beat)
                repaired_beat["motion"] = _coerce_enum(
                    repaired_beat.get("motion"),
                    allowed=("write", "fade", "transform", "highlight", "grow"),
                    fallback=fallback_beat.motion,
                )
                repaired_beats.append(repaired_beat)
            repaired_scene["beats"] = repaired_beats

        chart_data = repaired_scene.get("chart_data")
        if isinstance(chart_data, list):
            repaired_scene["chart_data"] = _normalize_chart_data(chart_data)

        repaired_scenes.append(repaired_scene)

    normalized["scenes"] = repaired_scenes
    return normalized


def _coerce_enum(value: object, *, allowed: tuple[str, ...], fallback: str) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in allowed:
            return lowered
        keyword_map = {
            "fade": ("crossfade", "fade", "dissolve", "still", "settle", "soft"),
            "transform": (
                "slide",
                "split",
                "merge",
                "wipe",
                "graft",
                "translate",
                "shift",
            ),
            "highlight": (
                "highlight",
                "pulse",
                "spotlight",
                "light up",
                "glow",
                "ring",
            ),
            "zoom": ("zoom", "push", "focus", "orbit"),
            "write": ("write", "label", "annotate", "text", "caption"),
            "pan": ("pan", "track", "flow", "sweep", "conveyor", "left-right"),
            "reveal": ("reveal", "uncover", "open", "expose"),
            "grow": ("grow", "branch", "expand", "build", "sprout"),
            "static": ("static", "steady", "locked", "stillness"),
        }
        for candidate in allowed:
            if any(token in lowered for token in keyword_map.get(candidate, ())):
                return candidate
    return fallback


def _clamp_float(
    value: object, *, minimum: float, maximum: float, fallback: float
) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return float(min(max(float(value), minimum), maximum))
    return fallback


def _normalize_chart_data(chart_data: list[object]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_value = 0.0
    for index, item in enumerate(chart_data):
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        raw_value = item.get("value")
        if not isinstance(label, str) or not label.strip():
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        value = max(float(raw_value), 0.1)
        max_value = max(max_value, value)
        rows.append(
            {
                "label": label.strip(),
                "value": value,
                "emphasis": bool(item.get("emphasis", index == len(chart_data) - 1)),
            }
        )
    if not rows:
        return []
    if max_value <= 100.0:
        return rows
    scale = 100.0 / max_value
    for row in rows:
        row["value"] = max(round(float(row["value"]) * scale, 2), 0.1)
    return rows

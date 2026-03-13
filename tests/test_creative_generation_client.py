from __future__ import annotations

import json

import pytest

from aqshara_video_worker.clients.creative_generation_client import (
    CreativeConfigurationError,
    CreativeGenerationError,
    OpenAICompatibleCreativeGenerationClient,
    _load_json_object,
    _normalize_director_plan_payload,
)
from aqshara_video_worker.config import WorkerSettings
from aqshara_video_worker.pipeline.storyboard import build_storyboard_artifacts
from aqshara_video_worker.schemas import DirectorPlanSpec


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self._responses.pop(0))


class _FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _FakeChatCompletions(responses)


class _FakeOpenAIClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = _FakeChat(responses)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_creative_generation_client_builds_artifacts_from_openai_responses() -> (
    None
):
    responses = [
        json.dumps(
            {
                "topic": "Adaptive Research Videos",
                "problem": "Dense papers are slow to digest.",
                "method": "Use AI planning before animation.",
                "result": "Viewers understand the paper faster.",
                "conclusion": "AI can direct better explainer videos.",
                "key_entities": ["AI", "Manim"],
                "visual_opportunities": ["camera zoom", "semantic highlights"],
                "misconceptions": ["Slides alone are enough"],
            }
        ),
        json.dumps(
            {
                "topic": "Adaptive Research Videos",
                "style_profile": "3b1b-lite",
                "visual_language": "Progressive disclosure and transform continuity.",
                "narrative_question": "How can AI make paper videos actually feel directed?",
                "recurring_motifs": ["zoom", "highlight", "transformation"],
                "target_story_arc": [
                    "hook",
                    "stakes",
                    "mechanism",
                    "evidence",
                    "takeaway",
                ],
                "quality_score": 0.91,
                "scenes": [
                    {
                        "scene_index": 1,
                        "scene_kind": "hook",
                        "headline": "Adaptive Research Videos",
                        "purpose": "Open with a visual promise.",
                        "visual_goal": "Make the viewer care immediately.",
                        "visual_metaphor": "Title zooms into payoff.",
                        "continuity_from": None,
                        "transition_strategy": "zoom",
                        "camera_plan": {
                            "mode": "focus",
                            "target": "payoff",
                            "scale": 0.9,
                        },
                        "beats": [
                            {
                                "beat_index": 1,
                                "visual": "Paper title",
                                "narration": "Start with the promise.",
                                "motion": "write",
                                "emphasis": "promise",
                            },
                            {
                                "beat_index": 2,
                                "visual": "Payoff",
                                "narration": "Then reveal the outcome.",
                                "motion": "highlight",
                                "emphasis": "outcome",
                            },
                        ],
                        "emphasis_terms": ["promise", "outcome"],
                        "entities": ["AI", "Manim"],
                        "chart_data": None,
                        "narration_cues": {
                            "voice_instruction": "Be curious.",
                            "pause_after_ms": 300,
                            "emphasis_terms": ["promise"],
                        },
                        "planned_duration_ms": 9600,
                    },
                    {
                        "scene_index": 2,
                        "scene_kind": "problem",
                        "headline": "The Problem",
                        "purpose": "Set stakes.",
                        "visual_goal": "Contrast overload and clarity.",
                        "visual_metaphor": "Split screen.",
                        "continuity_from": "payoff",
                        "transition_strategy": "transform",
                        "camera_plan": {"mode": "pan", "target": "gap", "scale": 0.95},
                        "beats": [
                            {
                                "beat_index": 1,
                                "visual": "Dense paper",
                                "narration": "Papers are dense.",
                                "motion": "fade",
                                "emphasis": "dense",
                            },
                            {
                                "beat_index": 2,
                                "visual": "Gap",
                                "narration": "The gap is time.",
                                "motion": "transform",
                                "emphasis": "gap",
                            },
                        ],
                        "emphasis_terms": ["gap"],
                        "entities": ["AI"],
                        "chart_data": None,
                        "narration_cues": {
                            "voice_instruction": "Be grounded.",
                            "pause_after_ms": 250,
                            "emphasis_terms": ["gap"],
                        },
                        "planned_duration_ms": 12000,
                    },
                    {
                        "scene_index": 3,
                        "scene_kind": "mechanism",
                        "headline": "How It Works",
                        "purpose": "Show the pipeline.",
                        "visual_goal": "Move through steps.",
                        "visual_metaphor": "Conveyor steps.",
                        "continuity_from": "gap",
                        "transition_strategy": "highlight",
                        "camera_plan": {"mode": "pan", "target": "step", "scale": 0.92},
                        "beats": [
                            {
                                "beat_index": 1,
                                "visual": "Analyze",
                                "narration": "Analyze the paper.",
                                "motion": "grow",
                                "emphasis": "Analyze",
                            },
                            {
                                "beat_index": 2,
                                "visual": "Plan",
                                "narration": "Plan visuals.",
                                "motion": "transform",
                                "emphasis": "Plan",
                            },
                        ],
                        "emphasis_terms": ["Analyze", "Plan"],
                        "entities": ["AI"],
                        "chart_data": None,
                        "narration_cues": {
                            "voice_instruction": "Be crisp.",
                            "pause_after_ms": 220,
                            "emphasis_terms": ["Plan"],
                        },
                        "planned_duration_ms": 15600,
                    },
                    {
                        "scene_index": 4,
                        "scene_kind": "evidence",
                        "headline": "Evidence",
                        "purpose": "Show the strongest result.",
                        "visual_goal": "Highlight one metric.",
                        "visual_metaphor": "Bars and highlight ring.",
                        "continuity_from": "step",
                        "transition_strategy": "zoom",
                        "camera_plan": {
                            "mode": "focus",
                            "target": "bar",
                            "scale": 0.88,
                        },
                        "beats": [
                            {
                                "beat_index": 1,
                                "visual": "Baseline",
                                "narration": "Compare to baseline.",
                                "motion": "fade",
                                "emphasis": "baseline",
                            },
                            {
                                "beat_index": 2,
                                "visual": "Improvement",
                                "narration": "Then isolate the gain.",
                                "motion": "highlight",
                                "emphasis": "gain",
                            },
                        ],
                        "emphasis_terms": ["gain"],
                        "entities": ["AI"],
                        "chart_data": [
                            {"label": "baseline", "value": 42, "emphasis": False},
                            {"label": "method", "value": 68, "emphasis": True},
                        ],
                        "narration_cues": {
                            "voice_instruction": "Slow on the metric.",
                            "pause_after_ms": 320,
                            "emphasis_terms": ["gain"],
                        },
                        "planned_duration_ms": 13200,
                    },
                    {
                        "scene_index": 5,
                        "scene_kind": "takeaway",
                        "headline": "Takeaway",
                        "purpose": "End with a clear closing.",
                        "visual_goal": "Compress the argument.",
                        "visual_metaphor": "Recap card.",
                        "continuity_from": "bar",
                        "transition_strategy": "fade",
                        "camera_plan": {
                            "mode": "static",
                            "target": "recap",
                            "scale": 1.0,
                        },
                        "beats": [
                            {
                                "beat_index": 1,
                                "visual": "AI can direct",
                                "narration": "AI improves planning.",
                                "motion": "fade",
                                "emphasis": "AI",
                            },
                            {
                                "beat_index": 2,
                                "visual": "Better videos",
                                "narration": "The output feels more directed.",
                                "motion": "highlight",
                                "emphasis": "directed",
                            },
                        ],
                        "emphasis_terms": ["AI", "directed"],
                        "entities": ["AI", "Manim"],
                        "chart_data": None,
                        "narration_cues": {
                            "voice_instruction": "Resolve with confidence.",
                            "pause_after_ms": 280,
                            "emphasis_terms": ["directed"],
                        },
                        "planned_duration_ms": 9600,
                    },
                ],
            }
        ),
        json.dumps(
            {
                "hook_line": "What if the paper could direct itself?",
                "tone": "clear and energetic",
                "scenes": [
                    {
                        "scene_index": 1,
                        "narration_text": "Scene 1 narration.",
                        "voice_instruction": "Excited but precise.",
                        "emphasis_terms": ["paper"],
                    },
                    {
                        "scene_index": 2,
                        "narration_text": "Scene 2 narration.",
                        "voice_instruction": "Grounded.",
                        "emphasis_terms": ["gap"],
                    },
                    {
                        "scene_index": 3,
                        "narration_text": "Scene 3 narration.",
                        "voice_instruction": "Instructional.",
                        "emphasis_terms": ["plan"],
                    },
                    {
                        "scene_index": 4,
                        "narration_text": "Scene 4 narration.",
                        "voice_instruction": "Measured.",
                        "emphasis_terms": ["result"],
                    },
                    {
                        "scene_index": 5,
                        "narration_text": "Scene 5 narration.",
                        "voice_instruction": "Resolving.",
                        "emphasis_terms": ["takeaway"],
                    },
                ],
            }
        ),
        "from manim import *\n\nconfig.background_color = '#0B1020'\n\nclass AIScene01(Scene):\n    RENDER_SEED = 1\n\n    def construct(self):\n        self.play(Write(Text('scene 1')))\n        self.wait(0.2)\n",
        json.dumps(
            {
                "scene_index": 1,
                "strengths": ["clear"],
                "issues": ["timing"],
                "revision_brief": "Tighten pacing.",
                "requires_revision": True,
            }
        ),
        "from manim import *\n\nconfig.background_color = '#0B1020'\n\nclass AIScene01(Scene):\n    RENDER_SEED = 2\n\n    def construct(self):\n        self.play(Write(Text('scene 1 revised')), run_time=0.6)\n        self.wait(0.2)\n",
        "from manim import *\n\nconfig.background_color = '#0B1020'\n\nclass AIScene02(Scene):\n    RENDER_SEED = 1\n\n    def construct(self):\n        self.play(Write(Text('scene 2')))\n        self.wait(0.2)\n",
        json.dumps(
            {
                "scene_index": 2,
                "strengths": ["clear"],
                "issues": [],
                "revision_brief": "Looks good.",
                "requires_revision": False,
            }
        ),
        "from manim import *\n\nconfig.background_color = '#0B1020'\n\nclass AIScene03(Scene):\n    RENDER_SEED = 1\n\n    def construct(self):\n        self.play(Write(Text('scene 3')))\n        self.wait(0.2)\n",
        json.dumps(
            {
                "scene_index": 3,
                "strengths": ["clear"],
                "issues": [],
                "revision_brief": "Looks good.",
                "requires_revision": False,
            }
        ),
        "from manim import *\n\nconfig.background_color = '#0B1020'\n\nclass AIScene04(Scene):\n    RENDER_SEED = 1\n\n    def construct(self):\n        self.play(Write(Text('scene 4')))\n        self.wait(0.2)\n",
        json.dumps(
            {
                "scene_index": 4,
                "strengths": ["clear"],
                "issues": [],
                "revision_brief": "Looks good.",
                "requires_revision": False,
            }
        ),
        "from manim import *\n\nconfig.background_color = '#0B1020'\n\nclass AIScene05(Scene):\n    RENDER_SEED = 1\n\n    def construct(self):\n        self.play(Write(Text('scene 5')))\n        self.wait(0.2)\n",
        json.dumps(
            {
                "scene_index": 5,
                "strengths": ["clear"],
                "issues": [],
                "revision_brief": "Looks good.",
                "requires_revision": False,
            }
        ),
    ]
    fake_client = _FakeOpenAIClient(responses)
    client = OpenAICompatibleCreativeGenerationClient(
        WorkerSettings.model_validate(
            {
                "VIDEO_CREATIVE_API_KEY": "creative-key",
                "VIDEO_CREATIVE_BASE_URL": "https://openrouter.ai/api/v1",
                "VIDEO_CREATIVE_GENERATION_MODEL": "openai/gpt-4.1",
                "VIDEO_CREATIVE_CRITIQUE_MODEL": "openai/gpt-4.1-mini",
                "VIDEO_CREATIVE_CODEGEN_MODEL": "openai/gpt-4.1",
                "REDIS_URL": "redis://localhost:6379/0",
                "R2_ENDPOINT": "https://example.r2.cloudflarestorage.com",
                "R2_ACCESS_KEY_ID": "key",
                "R2_SECRET_ACCESS_KEY": "secret",
                "R2_BUCKET": "bucket",
            }
        ),
        client=fake_client,  # type: ignore[arg-type]
    )

    try:
        artifacts = await client.generate_artifacts(
            ocr_result={
                "pages": [
                    {
                        "index": 0,
                        "markdown": (
                            "# Adaptive Research Videos\n\n"
                            "The problem is that papers are time-consuming to digest.\n\n"
                            "Our method builds a structured summary and scene plan.\n\n"
                            "Results show better retention and faster comprehension.\n\n"
                            "The conclusion is that paper-to-video conversion improves accessibility."
                        ),
                    }
                ]
            },
            target_duration_sec=60,
            language="en",
        )
        scene_code_drafts = await client.generate_scene_code_drafts(
            paper_analysis=artifacts.paper_analysis,
            director_plan=artifacts.director_plan,
            storyboard=artifacts.storyboard.model_copy(
                update={
                    "scenes": [
                        scene.model_copy(
                            update={"target_render_duration_ms": scene.planned_duration_ms}
                        )
                        for scene in artifacts.storyboard.scenes
                    ]
                }
            ),
            language="en",
        )
    finally:
        await client.close()

    assert artifacts.paper_analysis.topic == "Adaptive Research Videos"
    assert artifacts.storyboard.hook == "What if the paper could direct itself?"
    assert artifacts.script_plan.scenes[0].voice_instruction == "Excited but precise."
    assert artifacts.scene_code_drafts == {}
    assert scene_code_drafts[1].critique.requires_revision is True
    assert "AIScene01" in scene_code_drafts[1].revised_code
    assert len(fake_client.chat.completions.calls) == 14


@pytest.mark.asyncio
async def test_creative_generation_client_requires_separate_creative_config() -> None:
    client = OpenAICompatibleCreativeGenerationClient(
        WorkerSettings(
            _env_file=None,
            REDIS_URL="redis://localhost:6379/0",
            R2_ENDPOINT="https://example.r2.cloudflarestorage.com",
            R2_ACCESS_KEY_ID="key",
            R2_SECRET_ACCESS_KEY="secret",
            R2_BUCKET="bucket",
        ),
        client=_FakeOpenAIClient([]),  # type: ignore[arg-type]
    )

    try:
        with pytest.raises(
            CreativeConfigurationError,
            match="VIDEO_CREATIVE_API_KEY is required",
        ):
            await client.generate_artifacts(
                ocr_result={"pages": []},
                target_duration_sec=60,
                language="en",
            )
    finally:
        await client.close()


def test_load_json_object_accepts_wrapped_json_content() -> None:
    payload = _load_json_object(
        'Here is the JSON:\n```json\n{"topic":"A","problem":"B"}\n```'
    )
    assert payload == {"topic": "A", "problem": "B"}


def test_load_json_object_rejects_content_without_json() -> None:
    with pytest.raises(CreativeGenerationError, match="not valid JSON"):
        _load_json_object("No structured output available.")


def test_normalize_director_plan_payload_repairs_invalid_enum_fields_and_chart_scale() -> (
    None
):
    baseline = build_storyboard_artifacts(
        {
            "pages": [
                {
                    "index": 0,
                    "markdown": (
                        "# Graph Reasoning\n\n"
                        "The problem is that long papers hide the core mechanism.\n\n"
                        "Our method organizes the steps into a visual pipeline.\n\n"
                        "Results show faster comprehension and better recall.\n\n"
                        "The conclusion is that structure improves explainability."
                    ),
                }
            ]
        },
        60,
    )
    payload = baseline.director_plan.model_dump()
    payload["quality_score"] = 1.8
    payload["scenes"][0]["transition_strategy"] = (
        "dramatic crossfade and rapid time-lapse grow"
    )
    payload["scenes"][0]["camera_plan"]["mode"] = "push in then orbit"
    payload["scenes"][0]["beats"][0]["motion"] = "subtle tremble, then stillness"
    payload["scenes"][0]["beats"][1]["motion"] = (
        "sweeping spotlight, rapid branch growth"
    )
    payload["scenes"][3]["chart_data"] = [
        {"label": "baseline", "value": 1143, "emphasis": False},
        {"label": "method", "value": 4216, "emphasis": True},
    ]

    normalized = _normalize_director_plan_payload(payload, baseline.director_plan)
    repaired = DirectorPlanSpec.model_validate(normalized)

    assert repaired.quality_score == 1.0
    assert repaired.scenes[0].transition_strategy == "fade"
    assert repaired.scenes[0].camera_plan.mode == "focus"
    assert repaired.scenes[0].beats[0].motion == "fade"
    assert repaired.scenes[0].beats[1].motion in {"highlight", "grow"}
    assert repaired.scenes[3].chart_data is not None
    assert max(datum.value for datum in repaired.scenes[3].chart_data) == 100.0

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from aqshara_video_worker.schemas import (
    CameraPlanSpec,
    ChartDatumSpec,
    DirectorPlanSpec,
    DirectorSceneSpec,
    NarrationCueSpec,
    SceneBeatSpec,
    SceneConstraintSpec,
    SceneSpec,
    StoryboardSpec,
    StructuredSummary,
)


DEFAULT_COLOR_PALETTE = ["slate", "cyan", "amber", "emerald", "rose"]
SCENE_WEIGHTS = [0.16, 0.2, 0.26, 0.22, 0.16]
STYLE_PROFILE = "3b1b-lite"
VISUAL_LANGUAGE = "Progressive disclosure, semantic color, and continuity-driven camera focus."


@dataclass(frozen=True)
class StoryboardArtifacts:
    summary: StructuredSummary
    director_plan: DirectorPlanSpec
    storyboard: StoryboardSpec
    scenes_markdown: str


def build_storyboard_artifacts(
    ocr_result: object,
    target_duration_sec: int,
) -> StoryboardArtifacts:
    pages = _extract_pages(ocr_result)
    paragraphs = _extract_paragraphs(pages)
    if not paragraphs:
        raise ValueError("OCR artifact does not contain usable markdown paragraphs")

    topic = _extract_topic(pages, paragraphs)
    summary = StructuredSummary(
        topic=topic,
        problem=_select_paragraph(
            paragraphs,
            ["problem", "challenge", "motivation", "introduction"],
            fallback_index=0,
        ),
        method=_select_paragraph(
            paragraphs,
            ["method", "approach", "we propose", "we present", "pipeline"],
            fallback_index=min(1, len(paragraphs) - 1),
        ),
        result=_select_paragraph(
            paragraphs,
            ["result", "evaluation", "experiment", "accuracy", "performance"],
            fallback_index=max(0, len(paragraphs) // 2),
        ),
        conclusion=_select_paragraph(
            paragraphs,
            ["conclusion", "future work", "discussion", "summary"],
            fallback_index=len(paragraphs) - 1,
        ),
        source_excerpt_count=len(paragraphs),
    )

    narrative_question = _compose_narrative_question(summary)
    entities = _extract_entities(topic, paragraphs)
    result_claim = _extract_quantitative_claims(summary.result)
    durations = _allocate_durations(target_duration_sec)

    director_plan = DirectorPlanSpec(
        topic=summary.topic,
        style_profile=STYLE_PROFILE,
        visual_language=VISUAL_LANGUAGE,
        narrative_question=narrative_question,
        recurring_motifs=[
            "semantic highlight ring",
            "camera push into the current focal idea",
            "left-to-right research progression",
        ],
        target_story_arc=["hook", "problem", "mechanism", "evidence", "takeaway"],
        quality_score=0.0,
        scenes=[
            _build_hook_director_scene(summary, entities, durations[0]),
            _build_problem_director_scene(summary, entities, durations[1]),
            _build_mechanism_director_scene(summary, entities, durations[2]),
            _build_evidence_director_scene(summary, entities, result_claim, durations[3]),
            _build_takeaway_director_scene(summary, entities, durations[4]),
        ],
    )
    director_plan = director_plan.model_copy(
        update={"quality_score": _score_director_plan(director_plan)},
    )

    storyboard = build_storyboard_from_director_plan(
        summary=summary,
        director_plan=director_plan,
        target_duration_sec=target_duration_sec,
    )

    return StoryboardArtifacts(
        summary=summary,
        director_plan=director_plan,
        storyboard=storyboard,
        scenes_markdown=render_scenes_markdown(storyboard, director_plan),
    )


def dumps_summary(summary: StructuredSummary) -> str:
    return json.dumps(summary.model_dump(), indent=2)


def dumps_director_plan(director_plan: DirectorPlanSpec) -> str:
    return json.dumps(director_plan.model_dump(), indent=2)


def dumps_storyboard(storyboard: StoryboardSpec) -> str:
    return json.dumps(storyboard.model_dump(), indent=2)


def build_storyboard_from_director_plan(
    *,
    summary: StructuredSummary,
    director_plan: DirectorPlanSpec,
    target_duration_sec: int,
) -> StoryboardSpec:
    scenes = [_build_scene_spec(scene) for scene in director_plan.scenes]
    return StoryboardSpec(
        topic=summary.topic,
        hook=director_plan.narrative_question,
        target_audience="Learners who want a visually clear and fast overview of the paper.",
        estimated_length_sec=target_duration_sec,
        key_insight=summary.result,
        narrative_arc=director_plan.target_story_arc,
        style_profile=director_plan.style_profile,
        quality_score=director_plan.quality_score,
        scenes=scenes,
        transitions=[
            f"scene {scene.scene_index}: {scene.transition_strategy}"
            for scene in director_plan.scenes
        ],
        color_palette=DEFAULT_COLOR_PALETTE,
        implementation_order=[scene.scene_index for scene in scenes],
    )


def _build_scene_spec(scene: DirectorSceneSpec) -> SceneSpec:
    narration_text = _compose_narration_text(scene)
    visual_elements = [scene.visual_goal, scene.visual_metaphor, *scene.entities[:2]]
    return SceneSpec(
        scene_index=scene.scene_index,
        title=scene.headline,
        template_type=_scene_template(scene.scene_kind),
        scene_kind=scene.scene_kind,
        purpose=scene.purpose,
        narration_text=narration_text,
        planned_duration_ms=scene.planned_duration_ms,
        visual_elements=[value for value in visual_elements if value],
        visual_beats=scene.beats,
        emphasis_terms=scene.emphasis_terms,
        entities=scene.entities,
        camera_plan=scene.camera_plan,
        transition_strategy=scene.transition_strategy,
        chart_data=scene.chart_data,
        narration_cues=scene.narration_cues,
        transition_in=scene.continuity_from,
        transition_out=scene.visual_goal,
        constraints=SceneConstraintSpec(
            max_mobjects=16 if scene.scene_kind == "mechanism" else 12,
            max_animations=12 if scene.scene_kind in {"mechanism", "evidence"} else 9,
            allow_camera_movement=scene.camera_plan.mode != "static",
        ),
    )


def _scene_template(scene_kind: str) -> str:
    return {
        "hook": "hook",
        "problem": "problem",
        "mechanism": "mechanism",
        "evidence": "evidence",
        "takeaway": "takeaway",
    }.get(scene_kind, "bullet")


def _build_hook_director_scene(
    summary: StructuredSummary,
    entities: list[str],
    duration_ms: int,
) -> DirectorSceneSpec:
    emphasis = _extract_emphasis_terms(summary.topic, summary.result)
    beats = [
        SceneBeatSpec(
            beat_index=1,
            visual=_sentence_excerpt(summary.topic, max_words=7),
            narration=f"Start with the central promise: {summary.topic}",
            motion="write",
            emphasis=emphasis[0] if emphasis else summary.topic,
        ),
        SceneBeatSpec(
            beat_index=2,
            visual=_sentence_excerpt(summary.problem, max_words=10),
            narration=_sentence_excerpt(summary.problem, max_words=16),
            motion="fade",
            emphasis=emphasis[1] if len(emphasis) > 1 else None,
        ),
        SceneBeatSpec(
            beat_index=3,
            visual=_sentence_excerpt(summary.result, max_words=10),
            narration=f"The payoff is immediate: {_sentence_excerpt(summary.result, max_words=16)}",
            motion="highlight",
            emphasis=emphasis[-1] if emphasis else None,
        ),
    ]
    return DirectorSceneSpec(
        scene_index=1,
        scene_kind="hook",
        headline=summary.topic,
        purpose="Open on the paper promise before diving into details.",
        visual_goal="Make the viewer care within the first few seconds.",
        visual_metaphor="A focal card that zooms from the paper title into the key claim.",
        transition_strategy="zoom",
        camera_plan=CameraPlanSpec(mode="focus", target="key claim", scale=0.9),
        beats=beats,
        emphasis_terms=emphasis,
        entities=entities[:3],
        narration_cues=NarrationCueSpec(
            voice_instruction="Sound curious but confident, then pause before the payoff.",
            pause_after_ms=350,
            emphasis_terms=emphasis,
        ),
        planned_duration_ms=duration_ms,
    )


def _build_problem_director_scene(
    summary: StructuredSummary,
    entities: list[str],
    duration_ms: int,
) -> DirectorSceneSpec:
    beats = [
        SceneBeatSpec(
            beat_index=1,
            visual="Dense paper context",
            narration=_sentence_excerpt(summary.problem, max_words=16),
            motion="fade",
            emphasis="problem",
        ),
        SceneBeatSpec(
            beat_index=2,
            visual="Gap between paper depth and viewer time",
            narration="The real tension is not just complexity, but how quickly a learner needs the point.",
            motion="transform",
            emphasis="tension",
        ),
        SceneBeatSpec(
            beat_index=3,
            visual="Clear problem statement",
            narration="So the video reframes the research gap into one concrete question worth following.",
            motion="highlight",
            emphasis="question",
        ),
    ]
    return DirectorSceneSpec(
        scene_index=2,
        scene_kind="problem",
        headline="Why This Paper Matters",
        purpose="Turn the abstract paper setup into a concrete viewer problem.",
        visual_goal="Contrast research density with viewer attention.",
        visual_metaphor="A split-screen of overload on one side and a clean research question on the other.",
        continuity_from=summary.topic,
        transition_strategy="transform",
        camera_plan=CameraPlanSpec(mode="pan", target="research gap", scale=0.95),
        beats=beats,
        emphasis_terms=["research gap", "attention", *entities[:1]],
        entities=entities[:2],
        narration_cues=NarrationCueSpec(
            voice_instruction="Use a grounded explanatory tone and stress the research gap.",
            pause_after_ms=250,
            emphasis_terms=["research gap"],
        ),
        planned_duration_ms=duration_ms,
    )


def _build_mechanism_director_scene(
    summary: StructuredSummary,
    entities: list[str],
    duration_ms: int,
) -> DirectorSceneSpec:
    steps = _pipeline_steps(summary.method)
    beats = [
        SceneBeatSpec(
            beat_index=index + 1,
            visual=step,
            narration=f"Step {index + 1}: {step}.",
            motion="grow" if index == 0 else "transform",
            emphasis=step.split()[0],
        )
        for index, step in enumerate(steps[:4])
    ]
    return DirectorSceneSpec(
        scene_index=3,
        scene_kind="mechanism",
        headline="How The Method Works",
        purpose="Show the method as a progressive transformation, not a static list.",
        visual_goal="Guide the eye through the method flow one decision at a time.",
        visual_metaphor="A conveyor of research steps where each block activates the next one.",
        continuity_from="research gap",
        transition_strategy="highlight",
        camera_plan=CameraPlanSpec(mode="pan", target="active method step", scale=0.92),
        beats=beats,
        emphasis_terms=[step.split()[0] for step in steps[:3]],
        entities=entities[:3],
        narration_cues=NarrationCueSpec(
            voice_instruction="Keep the cadence crisp and instructional, with small pauses between steps.",
            pause_after_ms=220,
            emphasis_terms=[step.split()[0] for step in steps[:3]],
        ),
        planned_duration_ms=duration_ms,
    )


def _build_evidence_director_scene(
    summary: StructuredSummary,
    entities: list[str],
    chart_data: list[ChartDatumSpec],
    duration_ms: int,
) -> DirectorSceneSpec:
    beats = [
        SceneBeatSpec(
            beat_index=1,
            visual="Baseline versus method",
            narration="Now move from mechanism to evidence.",
            motion="fade",
            emphasis="evidence",
        ),
        SceneBeatSpec(
            beat_index=2,
            visual=_sentence_excerpt(summary.result, max_words=10),
            narration=_sentence_excerpt(summary.result, max_words=16),
            motion="grow",
            emphasis=chart_data[-1].label if chart_data else "result",
        ),
        SceneBeatSpec(
            beat_index=3,
            visual="Highlight the strongest claim",
            narration="The visual should land on the strongest metric rather than showing every number at once.",
            motion="highlight",
            emphasis="strongest metric",
        ),
    ]
    return DirectorSceneSpec(
        scene_index=4,
        scene_kind="evidence",
        headline="Evidence That It Works",
        purpose="Turn the headline result into a single visual proof point.",
        visual_goal="Make the strongest result feel earned and easy to compare.",
        visual_metaphor="Bars rise against a quieter baseline, then the winning claim gets isolated.",
        continuity_from="active method step",
        transition_strategy="zoom",
        camera_plan=CameraPlanSpec(mode="focus", target="highlighted bar", scale=0.88),
        beats=beats,
        emphasis_terms=["baseline", "improvement", *entities[:1]],
        entities=entities[:2],
        chart_data=chart_data,
        narration_cues=NarrationCueSpec(
            voice_instruction="Sound assured and slightly slower on the final metric.",
            pause_after_ms=320,
            emphasis_terms=[datum.label for datum in chart_data if datum.emphasis],
        ),
        planned_duration_ms=duration_ms,
    )


def _build_takeaway_director_scene(
    summary: StructuredSummary,
    entities: list[str],
    duration_ms: int,
) -> DirectorSceneSpec:
    takeaways = _bullet_points(summary.conclusion)
    while len(takeaways) < 2:
        takeaways.append(_sentence_excerpt(summary.result, max_words=10))
    beats = [
        SceneBeatSpec(
            beat_index=index + 1,
            visual=point,
            narration=point,
            motion="fade" if index < len(takeaways) - 1 else "highlight",
            emphasis=point.split()[0],
        )
        for index, point in enumerate(takeaways[:3])
    ]
    return DirectorSceneSpec(
        scene_index=5,
        scene_kind="takeaway",
        headline="What To Remember",
        purpose="Close the video on a compact, memorable takeaway.",
        visual_goal="Compress the paper into a small number of durable insights.",
        visual_metaphor="A clean recap card where each point settles into place.",
        continuity_from="highlighted bar",
        transition_strategy="fade",
        camera_plan=CameraPlanSpec(mode="static", target="recap card", scale=1.0),
        beats=beats,
        emphasis_terms=[point.split()[0] for point in takeaways[:2]],
        entities=entities[:2],
        narration_cues=NarrationCueSpec(
            voice_instruction="Finish with clarity and a slight sense of resolution.",
            pause_after_ms=280,
            emphasis_terms=[point.split()[0] for point in takeaways[:2]],
        ),
        planned_duration_ms=duration_ms,
    )


def _score_director_plan(plan: DirectorPlanSpec) -> float:
    beat_score = min(sum(len(scene.beats) for scene in plan.scenes) / 18.0, 1.0)
    camera_variety = len({scene.camera_plan.mode for scene in plan.scenes}) / 4.0
    emphasis_density = min(
        sum(len(scene.emphasis_terms) for scene in plan.scenes) / 10.0,
        1.0,
    )
    chart_bonus = 0.15 if any(scene.chart_data for scene in plan.scenes) else 0.0
    return round(min(0.35 + (beat_score * 0.25) + (camera_variety * 0.15) + (emphasis_density * 0.1) + chart_bonus, 1.0), 2)


def _compose_narration_text(scene: DirectorSceneSpec) -> str:
    pause = " " if scene.narration_cues.pause_after_ms < 250 else " ... "
    return pause.join(beat.narration.rstrip(".") + "." for beat in scene.beats)


def _extract_pages(ocr_result: object) -> list[dict[str, object]]:
    if not isinstance(ocr_result, dict):
        return []

    pages = ocr_result.get("pages")
    if not isinstance(pages, list):
        return []

    normalized_pages: list[dict[str, object]] = []
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        normalized_pages.append(
            {
                "index": page.get("index", index - 1),
                "markdown": page.get("markdown", ""),
            }
        )
    return normalized_pages


def _extract_paragraphs(pages: list[dict[str, object]]) -> list[str]:
    paragraphs: list[str] = []
    for page in pages:
        markdown = page.get("markdown")
        if not isinstance(markdown, str):
            continue
        for chunk in re.split(r"\n\s*\n+", markdown):
            normalized = _clean_text(chunk)
            if normalized:
                paragraphs.append(normalized)
    return paragraphs


def _extract_topic(pages: list[dict[str, object]], paragraphs: list[str]) -> str:
    for page in pages:
        markdown = page.get("markdown")
        if not isinstance(markdown, str):
            continue
        for line in markdown.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return _clean_text(stripped.lstrip("#"))
    return _sentence_excerpt(paragraphs[0], max_words=8)


def _select_paragraph(
    paragraphs: list[str],
    keywords: list[str],
    fallback_index: int,
) -> str:
    for paragraph in paragraphs:
        lower = paragraph.lower()
        if any(keyword in lower for keyword in keywords):
            return _sentence_excerpt(paragraph)
    return _sentence_excerpt(paragraphs[fallback_index])


def _compose_narrative_question(summary: StructuredSummary) -> str:
    return (
        f"How does {summary.topic} turn {summary.problem.lower()} "
        f"into {summary.result.lower()}?"
    )


def _allocate_durations(target_duration_sec: int) -> list[int]:
    total_ms = target_duration_sec * 1000
    durations = [int(total_ms * weight) for weight in SCENE_WEIGHTS]
    durations[-1] += total_ms - sum(durations)
    return durations


def _bullet_points(text: str) -> list[str]:
    sentences = _split_sentences(text)
    points = sentences[:3]
    if not points:
        points = [_sentence_excerpt(text, max_words=10)]
    return points


def _pipeline_steps(text: str) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) >= 3:
        return [sentence.rstrip(".") for sentence in sentences[:4]]

    keywords = re.split(r",|;|->| then | and ", text)
    cleaned = [_clean_text(keyword).rstrip(".") for keyword in keywords]
    steps = [value for value in cleaned if value][:4]
    if len(steps) >= 3:
        return steps

    return [
        "Anchor the research context",
        _sentence_excerpt(text, max_words=8).rstrip("."),
        "Reveal the paper insight",
    ]


def _extract_quantitative_claims(text: str) -> list[ChartDatumSpec]:
    numeric_fragments = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    if not numeric_fragments:
        return [
            ChartDatumSpec(label="baseline", value=42.0),
            ChartDatumSpec(label="paper method", value=68.0, emphasis=True),
            ChartDatumSpec(label="retention", value=61.0),
        ]

    values = [float(fragment.rstrip("%")) for fragment in numeric_fragments[:3]]
    while len(values) < 3:
        values.append(max(values[-1] - 8.0, 10.0))
    labels = ["baseline", "paper method", "best claim"]
    return [
        ChartDatumSpec(label=label, value=value, emphasis=index == 1)
        for index, (label, value) in enumerate(zip(labels, values, strict=False))
    ]


def _extract_entities(topic: str, paragraphs: list[str]) -> list[str]:
    candidates = [topic]
    for paragraph in paragraphs[:6]:
        title_words = re.findall(r"\b[A-Z][A-Za-z0-9\-]{3,}\b", paragraph)
        candidates.extend(title_words[:2])
    normalized: list[str] = []
    for candidate in candidates:
        cleaned = _clean_text(candidate)
        if cleaned and cleaned.lower() not in {item.lower() for item in normalized}:
            normalized.append(cleaned)
    return normalized[:5]


def _extract_emphasis_terms(*values: str) -> list[str]:
    emphasis: list[str] = []
    for value in values:
        for token in re.findall(r"\b[A-Za-z][A-Za-z\-]{4,}\b", value):
            lowered = token.lower()
            if lowered in {"paper", "their", "which", "there", "about"}:
                continue
            if token not in emphasis:
                emphasis.append(token)
    return emphasis[:4]


def render_scenes_markdown(
    storyboard: StoryboardSpec,
    director_plan: DirectorPlanSpec,
) -> str:
    scene_blocks: list[str] = []
    for scene in storyboard.scenes:
        beat_lines = [
            (
                f"  - beat {beat.beat_index}: visual={beat.visual}; "
                f"motion={beat.motion}; narration={beat.narration}"
            )
            for beat in scene.visual_beats
        ]
        chart_lines = [
            f"  - {datum.label}: {datum.value}"
            for datum in (scene.chart_data or [])
        ]
        scene_blocks.append(
            "\n".join(
                [
                    f"## Scene {scene.scene_index}: {scene.title}",
                    f"- duration_target_sec: {scene.planned_duration_ms // 1000}",
                    f"- purpose: {scene.purpose}",
                    f"- scene_kind: {scene.scene_kind}",
                    f"- visual_elements: {', '.join(scene.visual_elements)}",
                    f"- camera_plan: {scene.camera_plan.mode} -> {scene.camera_plan.target or 'frame'}",
                    f"- narration_notes: {scene.narration_text}",
                    (
                        "- technical_notes: template="
                        f"{scene.template_type}, transition={scene.transition_strategy}, "
                        f"max_mobjects={scene.constraints.max_mobjects}, "
                        f"max_animations={scene.constraints.max_animations}"
                    ),
                    "- beats:",
                    *beat_lines,
                    *(
                        ["- chart_data:", *chart_lines]
                        if chart_lines
                        else []
                    ),
                ]
            )
        )

    return "\n\n".join(
        [
            "# Scenes",
            "## Overview",
            f"- topic: {storyboard.topic}",
            f"- hook: {storyboard.hook}",
            f"- target_audience: {storyboard.target_audience}",
            f"- estimated_length: {storyboard.estimated_length_sec}s",
            f"- key_insight: {storyboard.key_insight}",
            f"- style_profile: {director_plan.style_profile}",
            f"- quality_score: {director_plan.quality_score}",
            "",
            "## Narrative Arc",
            *[f"- {step}" for step in director_plan.target_story_arc],
            "",
            "## Visual Language",
            f"- {director_plan.visual_language}",
            "",
            "## Recurring Motifs",
            *[f"- {motif}" for motif in director_plan.recurring_motifs],
            "",
            "## Transitions & Flow",
            *[f"- {transition}" for transition in storyboard.transitions],
            "",
            "## Color Palette",
            f"- {', '.join(storyboard.color_palette)}",
            "",
            "## Implementation Order",
            f"- {', '.join(str(index) for index in storyboard.implementation_order)}",
            "",
            *scene_blocks,
        ]
    )


def _sentence_excerpt(text: str, *, max_words: int = 20) -> str:
    sentences = _split_sentences(text)
    if sentences:
        text = sentences[0]
    words = text.split()
    excerpt = " ".join(words[:max_words]).strip()
    return excerpt.rstrip(".") + "."


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", _clean_text(text))
    return [part for part in parts if part]


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()

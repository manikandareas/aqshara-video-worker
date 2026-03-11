from __future__ import annotations

import json
import re
from dataclasses import dataclass

from aqshara_video_worker.schemas import SceneSpec, StoryboardSpec, StructuredSummary


DEFAULT_COLOR_PALETTE = ["slate", "cyan", "amber", "emerald"]
SCENE_WEIGHTS = [0.15, 0.22, 0.24, 0.21, 0.18]


@dataclass(frozen=True)
class StoryboardArtifacts:
    summary: StructuredSummary
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

    durations = _allocate_durations(target_duration_sec)
    scenes = [
        SceneSpec(
            scene_index=1,
            title=topic,
            template_type="title",
            purpose="Hook the viewer with the paper topic and main promise.",
            narration_text=_compose_hook(summary),
            planned_duration_ms=durations[0],
            visual_elements=[topic, summary.problem, summary.result],
        ),
        SceneSpec(
            scene_index=2,
            title="Problem Framing",
            template_type="bullet",
            purpose="Explain the problem space and why the paper matters.",
            narration_text=summary.problem,
            planned_duration_ms=durations[1],
            visual_elements=_bullet_points(summary.problem),
        ),
        SceneSpec(
            scene_index=3,
            title="Method Pipeline",
            template_type="pipeline",
            purpose="Show the method flow in a simple step-by-step pipeline.",
            narration_text=summary.method,
            planned_duration_ms=durations[2],
            visual_elements=_pipeline_steps(summary.method),
        ),
        SceneSpec(
            scene_index=4,
            title="Key Result",
            template_type="chart",
            purpose="Highlight the main evidence or result from the paper.",
            narration_text=summary.result,
            planned_duration_ms=durations[3],
            visual_elements=_chart_elements(summary.result),
        ),
        SceneSpec(
            scene_index=5,
            title="Takeaways",
            template_type="conclusion",
            purpose="Close with the core conclusion and practical takeaway.",
            narration_text=summary.conclusion,
            planned_duration_ms=durations[4],
            visual_elements=_bullet_points(summary.conclusion),
        ),
    ]

    storyboard = StoryboardSpec(
        topic=summary.topic,
        hook=_compose_hook(summary),
        target_audience="Learners who want a fast overview of the paper.",
        estimated_length_sec=target_duration_sec,
        key_insight=summary.result,
        narrative_arc=[
            "hook",
            "build_up",
            "insight",
            "closure",
        ],
        scenes=scenes,
        transitions=[
            "title -> problem: zoom into the research gap",
            "problem -> method: shift from challenge to solution steps",
            "method -> result: compare outcome against expectation",
            "result -> conclusion: land on the final takeaway",
        ],
        color_palette=DEFAULT_COLOR_PALETTE,
        implementation_order=[scene.scene_index for scene in scenes],
    )

    return StoryboardArtifacts(
        summary=summary,
        storyboard=storyboard,
        scenes_markdown=_render_scenes_markdown(storyboard),
    )


def dumps_summary(summary: StructuredSummary) -> str:
    return json.dumps(summary.model_dump(), indent=2)


def dumps_storyboard(storyboard: StoryboardSpec) -> str:
    return json.dumps(storyboard.model_dump(), indent=2)


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


def _compose_hook(summary: StructuredSummary) -> str:
    return (
        f"{summary.topic}. {summary.problem} "
        f"The key insight is: {summary.result}"
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
        return [sentence for sentence in sentences[:4]]

    keywords = re.split(r",|;|->| then | and ", text)
    cleaned = [_clean_text(keyword) for keyword in keywords]
    steps = [value for value in cleaned if value][:4]
    if len(steps) >= 3:
        return steps

    return [
        "Input paper context",
        _sentence_excerpt(text, max_words=8),
        "Produce the paper insight",
    ]


def _chart_elements(text: str) -> list[str]:
    numeric_fragments = re.findall(r"\b\d+(?:\.\d+)?%?\b", text)
    if numeric_fragments:
        return ["baseline", "improved metric", f"highlight {numeric_fragments[0]}"]
    return ["baseline trend", "improved trend", "key highlight"]


def _render_scenes_markdown(storyboard: StoryboardSpec) -> str:
    scene_blocks: list[str] = []
    for scene in storyboard.scenes:
        scene_blocks.append(
            "\n".join(
                [
                    f"## Scene {scene.scene_index}: {scene.title}",
                    f"- duration_target_sec: {scene.planned_duration_ms // 1000}",
                    f"- purpose: {scene.purpose}",
                    f"- visual_elements: {', '.join(scene.visual_elements)}",
                    f"- narration_notes: {scene.narration_text}",
                    (
                        "- technical_notes: template="
                        f"{scene.template_type}, max_mobjects={scene.constraints.max_mobjects}, "
                        f"max_animations={scene.constraints.max_animations}"
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
            "",
            "## Narrative Arc",
            "- hook -> build_up -> insight -> closure",
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

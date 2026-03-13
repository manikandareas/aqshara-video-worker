from __future__ import annotations

import ast
import hashlib
import textwrap

from aqshara_video_worker.schemas import ChartDatumSpec, SceneBeatSpec, SceneSpec


ALLOWED_IMPORT_MODULES = {"manim"}
FORBIDDEN_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}
FORBIDDEN_ROOT_NAMES = {
    "boto3",
    "httpx",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
FORBIDDEN_ATTRIBUTE_NAMES = {
    "connect",
    "delete",
    "execv",
    "execve",
    "get",
    "open",
    "popen",
    "post",
    "put",
    "read_bytes",
    "read_text",
    "recv",
    "remove",
    "rename",
    "replace",
    "request",
    "rmdir",
    "run",
    "send",
    "spawn",
    "system",
    "touch",
    "unlink",
    "write_bytes",
    "write_text",
}
FORBIDDEN_NODE_TYPES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Global,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.While,
    ast.With,
)


class ManimCodegenError(Exception):
    pass


class CodeValidationError(Exception):
    pass


def build_scene_module(scene: SceneSpec) -> tuple[str, str]:
    class_name = _scene_class_name(scene)
    render_seed = _render_seed(scene)
    base_class = _scene_base_class(scene)
    body = _render_template_body(scene)
    construct_body = textwrap.indent(body.strip(), " " * 8, lambda line: True)
    module = (
        "from manim import *\n\n"
        'config.background_color = "#0B1020"\n\n\n'
        f"class {class_name}({base_class}):\n"
        f"    RENDER_SEED = {render_seed}\n\n"
        "    def construct(self):\n"
        f"{construct_body}\n"
    )
    return class_name, module


def validate_generated_code(code: str, expected_class_name: str | None = None) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise CodeValidationError(f"Generated code is not valid Python: {error.msg}") from error

    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(class_defs) != 1:
        raise CodeValidationError("Generated code must define exactly one Scene subclass")

    scene_class = class_defs[0]
    if expected_class_name and scene_class.name != expected_class_name:
        raise CodeValidationError(
            f"Generated class name {scene_class.name} does not match expected {expected_class_name}"
        )

    if not any(_is_scene_base(base) for base in scene_class.bases):
        raise CodeValidationError("Generated class must inherit from Scene or MovingCameraScene")

    construct_methods = [
        node
        for node in scene_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "construct"
    ]
    if len(construct_methods) != 1:
        raise CodeValidationError("Generated class must define a single construct method")

    allowed_class_nodes = (ast.Assign, ast.FunctionDef)
    for node in scene_class.body:
        if not isinstance(node, allowed_class_nodes):
            raise CodeValidationError(
                "Generated Scene class may only contain assignments and the construct method"
            )

    for node in scene_class.body:
        if isinstance(node, ast.Assign):
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if target_names != ["RENDER_SEED"]:
                raise CodeValidationError(
                    "Generated Scene class may only define the RENDER_SEED assignment"
                )

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            raise CodeValidationError(f"Forbidden syntax in generated code: {type(node).__name__}")

        if isinstance(node, ast.Import):
            raise CodeValidationError("Direct import statements are not allowed")

        if isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORT_MODULES:
                raise CodeValidationError(f"Import from module {node.module!r} is not allowed")

        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in FORBIDDEN_CALL_NAMES:
                raise CodeValidationError(f"Forbidden call in generated code: {call_name}")

            root_name = _root_name(node.func)
            if root_name and root_name in FORBIDDEN_ROOT_NAMES:
                raise CodeValidationError(f"Forbidden root object in generated code: {root_name}")

            attr_name = _attribute_name(node.func)
            if attr_name and attr_name in FORBIDDEN_ATTRIBUTE_NAMES:
                raise CodeValidationError(f"Forbidden attribute call in generated code: {attr_name}")


def _scene_base_class(scene: SceneSpec) -> str:
    if scene.constraints.allow_camera_movement or scene.camera_plan.mode != "static":
        return "MovingCameraScene"
    return "Scene"


def _render_template_body(scene: SceneSpec) -> str:
    match scene.template_type:
        case "hook" | "title":
            return _render_hook_scene(scene)
        case "problem":
            return _render_problem_scene(scene)
        case "mechanism" | "pipeline":
            return _render_mechanism_scene(scene)
        case "evidence" | "chart":
            return _render_evidence_scene(scene)
        case "takeaway" | "conclusion":
            return _render_takeaway_scene(scene)
        case "bullet":
            return _render_bullet_scene(scene)
        case _:
            raise ManimCodegenError(
                f"Unsupported template type for codegen: {scene.template_type}"
            )


def _render_hook_scene(scene: SceneSpec) -> str:
    beats = _normalize_beats(scene.visual_beats, scene.narration_text)
    chips = _normalize_points(scene.entities or scene.emphasis_terms, fallback=scene.title, limit=3)
    focus_text = beats[-1].visual
    timings, final_hold = _timing_slots(scene, 6)
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=42, color=WHITE, weight=BOLD).scale_to_fit_width(11)
        strap = Text({scene.purpose!r}, font_size=24, color=BLUE_B).scale_to_fit_width(10)
        focus = Text({focus_text!r}, font_size=28, color=YELLOW_D).scale_to_fit_width(9.4)
        panel = RoundedRectangle(
            corner_radius=0.24,
            width=11.5,
            height=5.2,
            color=BLUE_D,
            fill_color=BLUE_E,
            fill_opacity=0.18,
        )
        chip_labels = VGroup(
            *[
                Text(label, font_size=18, color=GRAY_A).scale_to_fit_width(2.2)
                for label in {chips!r}
            ]
        )
        chip_cards = VGroup(
            *[
                VGroup(
                    RoundedRectangle(
                        corner_radius=0.14,
                        width=max(label.width + 0.45, 1.9),
                        height=0.55,
                        color=GRAY_C,
                        fill_color="#162338",
                        fill_opacity=0.75,
                    ),
                    label,
                )
                for label in chip_labels
            ]
        ).arrange(RIGHT, buff=0.18)
        for card in chip_cards:
            card[1].move_to(card[0].get_center())
        focus_ring = SurroundingRectangle(focus, color=YELLOW_D, buff=0.18)
        self.play(FadeIn(panel, scale=0.96), run_time={timings[0]}, rate_func=smooth)
        self.play(Write(title), run_time={timings[1]}, rate_func=smooth)
        self.play(FadeIn(strap, shift=UP * 0.12), run_time={timings[2]}, rate_func=smooth)
        self.play(LaggedStart(*[FadeIn(card, shift=UP * 0.08) for card in chip_cards], lag_ratio=0.15), run_time={timings[3]})
        self.play(FadeIn(focus, shift=UP * 0.1), Create(focus_ring), run_time={timings[4]}, rate_func=smooth)
        if hasattr(self.camera, "frame"):
            self.play(self.camera.frame.animate.scale(0.92).move_to(focus), run_time={timings[5]}, rate_func=smooth)
        self.wait({final_hold})
        """
    ).strip()


def _render_problem_scene(scene: SceneSpec) -> str:
    beats = _normalize_beats(scene.visual_beats, scene.narration_text)
    left_text = beats[0].visual
    right_text = beats[-1].visual
    timings, final_hold = _timing_slots(scene, 6)
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=34, color=WHITE, weight=BOLD).to_edge(UP)
        left_card = RoundedRectangle(corner_radius=0.2, width=4.8, height=3.6, color=RED_D, fill_color="#30161A", fill_opacity=0.78)
        right_card = RoundedRectangle(corner_radius=0.2, width=4.8, height=3.6, color=BLUE_D, fill_color="#162338", fill_opacity=0.82)
        cards = VGroup(left_card, right_card).arrange(RIGHT, buff=0.6).next_to(title, DOWN, buff=0.8)
        left_label = Text({left_text!r}, font_size=24, color=WHITE).scale_to_fit_width(3.9).move_to(left_card.get_center())
        right_label = Text({right_text!r}, font_size=24, color=WHITE).scale_to_fit_width(3.9).move_to(right_card.get_center())
        bridge = Arrow(left_card.get_right(), right_card.get_left(), buff=0.15, color=YELLOW_D, stroke_width=5)
        caption = Text({beats[1].visual!r}, font_size=20, color=GRAY_A).scale_to_fit_width(9.8).to_edge(DOWN)
        highlight = SurroundingRectangle(right_card, color=YELLOW_D, buff=0.15)
        self.play(Write(title), run_time={timings[0]}, rate_func=smooth)
        self.play(FadeIn(left_card, shift=LEFT * 0.15), FadeIn(right_card, shift=RIGHT * 0.15), run_time={timings[1]})
        self.play(FadeIn(left_label), FadeIn(right_label), run_time={timings[2]})
        self.play(GrowArrow(bridge), FadeIn(caption, shift=UP * 0.08), run_time={timings[3]})
        self.play(Create(highlight), run_time={timings[4]}, rate_func=smooth)
        if hasattr(self.camera, "frame"):
            self.play(self.camera.frame.animate.scale(0.94).move_to(right_card), run_time={timings[5]}, rate_func=smooth)
        self.wait({final_hold})
        """
    ).strip()


def _render_mechanism_scene(scene: SceneSpec) -> str:
    beats = _normalize_beats(scene.visual_beats, scene.narration_text, min_count=3)
    step_labels = [beat.visual for beat in beats[:4]]
    focus_steps = max(len(step_labels) - 1, 0)
    timings, final_hold = _timing_slots(scene, 5 + (focus_steps * 2))
    transition_timings = timings[5:]
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=34, color=WHITE, weight=BOLD).to_edge(UP)
        boxes = VGroup()
        for label in {step_labels!r}:
            base = RoundedRectangle(
                corner_radius=0.18,
                width=2.25,
                height=1.2,
                color=BLUE_D,
                fill_color="#17314A",
                fill_opacity=0.88,
            )
            caption = Text(label, font_size=20, color=WHITE).scale_to_fit_width(1.85)
            card = VGroup(base, caption)
            caption.move_to(base.get_center())
            boxes.add(card)
        boxes.arrange(RIGHT, buff=0.4).next_to(title, DOWN, buff=1.0)
        connectors = VGroup(
            *[
                Arrow(
                    boxes[index][0].get_right(),
                    boxes[index + 1][0].get_left(),
                    buff=0.08,
                    stroke_width=4,
                    color=YELLOW_D,
                )
                for index in range(len(boxes) - 1)
            ]
        )
        stage_caption = Text({scene.purpose!r}, font_size=20, color=GRAY_A).scale_to_fit_width(10).to_edge(DOWN)
        focus_ring = SurroundingRectangle(boxes[0][0], color=YELLOW_D, buff=0.14)
        transition_timings = {transition_timings!r}
        self.play(Write(title), run_time={timings[0]}, rate_func=smooth)
        self.play(LaggedStart(*[FadeIn(box, shift=UP * 0.12) for box in boxes], lag_ratio=0.18), run_time={timings[1]})
        self.play(Create(connectors), run_time={timings[2]}, rate_func=smooth)
        self.play(FadeIn(stage_caption, shift=UP * 0.08), run_time={timings[3]})
        self.play(Create(focus_ring), run_time={timings[4]})
        transition_index = 0
        for box in boxes[1:]:
            self.play(
                focus_ring.animate.move_to(box[0].get_center()),
                box[0].animate.set_fill(color="#235D7D", opacity=0.95),
                run_time=transition_timings[transition_index],
                rate_func=smooth,
            )
            transition_index += 1
            if hasattr(self.camera, "frame"):
                self.play(
                    self.camera.frame.animate.move_to(box[0]).scale(0.96),
                    run_time=transition_timings[transition_index],
                    rate_func=smooth,
                )
                transition_index += 1
        self.wait({final_hold})
        """
    ).strip()


def _render_evidence_scene(scene: SceneSpec) -> str:
    chart_data = _normalize_chart_data(scene.chart_data)
    labels = [datum.label for datum in chart_data]
    values = [datum.value for datum in chart_data]
    emphasis_index = next(
        (index for index, datum in enumerate(chart_data) if datum.emphasis),
        len(chart_data) - 1,
    )
    timings, final_hold = _timing_slots(scene, 6)
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=34, color=WHITE, weight=BOLD).to_edge(UP)
        axes = Axes(
            x_range=[0, {len(values) + 1}, 1],
            y_range=[0, {max(max(values) + 10, 20):.1f}, 10],
            x_length=7.6,
            y_length=4.2,
            tips=False,
            axis_config={{"color": GRAY_B}},
        ).next_to(title, DOWN, buff=0.85)
        bars = VGroup()
        labels_group = VGroup()
        for index, value in enumerate({values!r}, start=1):
            is_focus = index - 1 == {emphasis_index}
            bar = Rectangle(
                width=0.9,
                height=max((value / {max(max(values), 1):.2f}) * 3.2, 0.6),
                color=YELLOW_D if is_focus else BLUE_D,
                fill_color=YELLOW_D if is_focus else BLUE_E,
                fill_opacity=0.82,
            )
            bar.move_to(axes.c2p(index, bar.height / 2))
            value_label = Text(f"{{value:.0f}}", font_size=20, color=WHITE).next_to(bar, UP, buff=0.14)
            bars.add(VGroup(bar, value_label))
            caption = Text({labels!r}[index - 1], font_size=18, color=GRAY_A).scale_to_fit_width(2.1)
            caption.next_to(axes.c2p(index, 0), DOWN, buff=0.24)
            labels_group.add(caption)
        insight = Text({scene.visual_beats[-1].visual if scene.visual_beats else scene.narration_text!r}, font_size=20, color=WHITE).scale_to_fit_width(10).to_edge(DOWN)
        highlight = SurroundingRectangle(bars[{emphasis_index}][0], color=YELLOW_D, buff=0.15)
        self.play(Write(title), run_time={timings[0]}, rate_func=smooth)
        self.play(Create(axes), run_time={timings[1]}, rate_func=smooth)
        self.play(LaggedStart(*[FadeIn(bar_group, shift=UP * 0.12) for bar_group in bars], lag_ratio=0.18), run_time={timings[2]})
        self.play(FadeIn(labels_group), run_time={timings[3]})
        self.play(Create(highlight), FadeIn(insight, shift=UP * 0.08), run_time={timings[4]})
        if hasattr(self.camera, "frame"):
            self.play(self.camera.frame.animate.scale(0.88).move_to(bars[{emphasis_index}][0]), run_time={timings[5]}, rate_func=smooth)
        self.wait({final_hold})
        """
    ).strip()


def _render_takeaway_scene(scene: SceneSpec) -> str:
    beats = _normalize_beats(scene.visual_beats, scene.narration_text, min_count=2)
    takeaways = [beat.visual for beat in beats[:3]]
    timings, final_hold = _timing_slots(scene, 4)
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=34, color=WHITE, weight=BOLD).to_edge(UP)
        card = RoundedRectangle(
            corner_radius=0.24,
            width=10.8,
            height=4.9,
            color=EMERALD,
            fill_color="#12362C",
            fill_opacity=0.78,
        ).next_to(title, DOWN, buff=0.7)
        bullets = VGroup(
            *[
                Text(f"• {{point}}", font_size=24, color=GRAY_A).scale_to_fit_width(8.9)
                for point in {takeaways!r}
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to(card.get_center())
        closing = Text({scene.purpose!r}, font_size=20, color=YELLOW_D).scale_to_fit_width(9.2).to_edge(DOWN)
        highlight = SurroundingRectangle(bullets[-1], color=YELLOW_D, buff=0.16)
        self.play(Write(title), run_time={timings[0]}, rate_func=smooth)
        self.play(FadeIn(card, scale=0.97), run_time={timings[1]}, rate_func=smooth)
        self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.1) for item in bullets], lag_ratio=0.16), run_time={timings[2]})
        self.play(Create(highlight), FadeIn(closing, shift=UP * 0.08), run_time={timings[3]})
        self.wait({final_hold})
        """
    ).strip()


def _render_bullet_scene(scene: SceneSpec) -> str:
    points = _normalize_points(scene.visual_elements, fallback=scene.narration_text, min_count=2, limit=4)
    timings, final_hold = _timing_slots(scene, len(points) + 2)
    bullet_timings = timings[1 : 1 + len(points)]
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=34, color=WHITE).to_edge(UP)
        bullets = VGroup(
            *[
                Text(point, font_size=24, color=GRAY_A).scale_to_fit_width(9.5)
                for point in {points!r}
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).next_to(title, DOWN, buff=0.6)
        highlight = SurroundingRectangle(bullets[0], color=YELLOW_D, buff=0.18)
        bullet_timings = {bullet_timings!r}
        self.play(Write(title), run_time={timings[0]}, rate_func=smooth)
        bullet_index = 0
        for bullet in bullets:
            self.play(
                FadeIn(bullet, shift=RIGHT * 0.2),
                run_time=bullet_timings[bullet_index],
                rate_func=smooth,
            )
            bullet_index += 1
        self.play(Create(highlight), run_time={timings[-1]}, rate_func=smooth)
        self.wait({final_hold})
        """
    ).strip()


def _scene_class_name(scene: SceneSpec) -> str:
    words = "".join(character if character.isalnum() else " " for character in scene.title)
    compact = "".join(part.capitalize() for part in words.split())
    base = compact or "GeneratedScene"
    return f"Scene{scene.scene_index:02d}{base}"


def _render_seed(scene: SceneSpec) -> int:
    digest = hashlib.sha256(scene.model_dump_json().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _timing_slots(scene: SceneSpec, step_count: int) -> tuple[list[float], float]:
    total_duration_sec = _target_duration_sec(scene)
    if step_count <= 0:
        return ([], round(min(0.25, total_duration_sec), 2))

    final_hold = min(0.5, max(0.15, min(0.25, total_duration_sec * 0.08)))
    available = max(total_duration_sec - final_hold, 0.1 * step_count)
    minimum_step = 0.2
    maximum_step = 1.8
    slots = [available / step_count] * step_count

    for _ in range(4):
        adjusted = False
        overflow = 0.0
        underflow = 0.0
        for index, value in enumerate(slots):
            if value > maximum_step:
                overflow += value - maximum_step
                slots[index] = maximum_step
                adjusted = True
            elif value < minimum_step:
                underflow += minimum_step - value
                slots[index] = minimum_step
                adjusted = True

        if not adjusted:
            break

        free_indexes = [
            index for index, value in enumerate(slots) if minimum_step < value < maximum_step
        ]
        if overflow > 0 and free_indexes:
            share = overflow / len(free_indexes)
            for index in free_indexes:
                slots[index] += share
        if underflow > 0 and free_indexes:
            share = underflow / len(free_indexes)
            for index in free_indexes:
                slots[index] = max(minimum_step, slots[index] - share)

    current_total = sum(slots)
    if current_total > 0:
        scale = available / current_total
        slots = [max(minimum_step, min(maximum_step, value * scale)) for value in slots]

    rounded = [round(value, 2) for value in slots]
    return (rounded, round(total_duration_sec - sum(rounded), 2))


def _target_duration_sec(scene: SceneSpec) -> float:
    duration_ms = scene.target_render_duration_ms or scene.planned_duration_ms
    return max(duration_ms / 1000.0, 0.5)


def _normalize_points(
    values: list[str],
    *,
    fallback: str,
    min_count: int = 1,
    limit: int = 4,
) -> list[str]:
    cleaned = [value.strip() for value in values if value.strip()]
    if len(cleaned) >= min_count:
        return cleaned[:limit]

    fallback_points = [segment.strip() for segment in fallback.split(".") if segment.strip()]
    cleaned.extend(fallback_points)
    while len(cleaned) < min_count:
        cleaned.append(fallback.strip())
    return cleaned[:limit]


def _normalize_beats(
    beats: list[SceneBeatSpec],
    fallback: str,
    *,
    min_count: int = 1,
) -> list[SceneBeatSpec]:
    cleaned = [beat for beat in beats if beat.visual.strip() and beat.narration.strip()]
    if len(cleaned) >= min_count:
        return cleaned

    fallback_points = _normalize_points([], fallback=fallback, min_count=min_count, limit=max(min_count, 3))
    return [
        SceneBeatSpec(
            beat_index=index + 1,
            visual=point,
            narration=point,
            motion="fade",
        )
        for index, point in enumerate(fallback_points)
    ]


def _normalize_chart_data(chart_data: list[ChartDatumSpec] | None) -> list[ChartDatumSpec]:
    if chart_data:
        return chart_data[:3]
    return [
        ChartDatumSpec(label="baseline", value=42.0),
        ChartDatumSpec(label="paper method", value=68.0, emphasis=True),
        ChartDatumSpec(label="retention", value=61.0),
    ]


def _is_scene_base(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id in {"Scene", "MovingCameraScene"}


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _root_name(node: ast.expr) -> str | None:
    current = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _attribute_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    return None

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass

from aqshara_video_worker.schemas import ChartDatumSpec, SceneBeatSpec, SceneSpec


@dataclass(frozen=True)
class DesignSystem:
    bg_primary: str = "#0B1020"
    bg_card: str = "#0F1B31"
    bg_card_accent: str = "#162338"
    accent_primary: str = "YELLOW_D"
    accent_secondary: str = "BLUE_D"
    text_primary: str = "WHITE"
    text_secondary: str = "GRAY_A"
    highlight: str = "YELLOW_D"
    font_title: str = "sans-serif"
    font_body: str = "sans-serif"
    spacing_sm: float = 0.2
    spacing_md: float = 0.45
    spacing_lg: float = 0.8


DEFAULT_DESIGN = DesignSystem()


def _truncate_label(text: str, max_chars: int = 55) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "\u2026"


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
    bg_layer = _render_background_layer()
    body = _render_template_body(scene)
    construct_body = textwrap.indent(
        bg_layer + "\n" + body.strip(), " " * 8, lambda line: True,
    )
    ds = DEFAULT_DESIGN
    module = (
        "from manim import *\n\n"
        f'config.background_color = "{ds.bg_primary}"\n\n\n'
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


def _render_background_layer() -> str:
    return textwrap.dedent(
        """\
        _bg_glow = Circle(radius=4.5, fill_opacity=0.12, stroke_width=0)
        _bg_glow.set_fill(color=["#102847", "#0B1020"])
        _bg_glow.move_to(ORIGIN)
        _bg_vig = Rectangle(width=16, height=10, stroke_width=0)
        _bg_vig.set_fill(color="#000000", opacity=0.0)
        _bg_vig.set_stroke(color="#000000", width=60, opacity=0.35)
        self.add(_bg_glow, _bg_vig)
        """
    ).strip()


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
    variant = _scene_variant(scene, 3)
    if variant == 0:
        return textwrap.dedent(
            f"""
            title = Text({scene.title!r}, font_size=44, color=WHITE, weight=BOLD).scale_to_fit_width(10.6).to_edge(UP, buff=0.6)
            focus = Text({focus_text!r}, font_size=30, color=YELLOW_D, weight=BOLD).scale_to_fit_width(8.8)
            strap = Text({scene.purpose!r}, font_size=24, color=GRAY_A).scale_to_fit_width(9.4)
            halo = Circle(radius=2.2, color=BLUE_D, stroke_opacity=0.4).set_fill("#102847", opacity=0.35)
            panel = RoundedRectangle(corner_radius=0.28, width=11.2, height=5.2, color=BLUE_D, fill_color="#0F1B31", fill_opacity=0.92)
            chip_labels = VGroup(*[Text(label, font_size=18, color=WHITE).scale_to_fit_width(2.2) for label in {chips!r}]).arrange(RIGHT, buff=0.16)
            chip_cards = VGroup(*[SurroundingRectangle(label, buff=0.18, color=GRAY_C, corner_radius=0.14) for label in chip_labels])
            chip_group = VGroup(*[VGroup(card, label.move_to(card.get_center())) for card, label in zip(chip_cards, chip_labels, strict=False)]).arrange(RIGHT, buff=0.18).to_edge(DOWN, buff=0.8)
            hero = VGroup(halo, focus).move_to(ORIGIN)
            strap.next_to(hero, DOWN, buff=0.6)
            focus_ring = SurroundingRectangle(focus, color=YELLOW_D, buff=0.2)
            self.add(panel)
            self.play(FadeIn(title, shift=UP * 0.12), run_time={timings[0]}, rate_func=smooth)
            self.play(FadeIn(halo, scale=0.92), FadeIn(focus, shift=UP * 0.1), run_time={timings[1]}, rate_func=smooth)
            self.play(Create(focus_ring), run_time={timings[2]}, rate_func=smooth)
            self.play(FadeIn(strap, shift=UP * 0.08), run_time={timings[3]}, rate_func=smooth)
            self.play(LaggedStart(*[FadeIn(group, shift=UP * 0.08) for group in chip_group], lag_ratio=0.16), run_time={timings[4]})
            if hasattr(self.camera, "frame"):
                self.play(self.camera.frame.animate.scale(0.92).move_to(hero), run_time={timings[5]}, rate_func=smooth)
            self.wait({final_hold})
            """
        ).strip()
    if variant == 1:
        return textwrap.dedent(
            f"""
            title = Text({scene.title!r}, font_size=42, color=WHITE, weight=BOLD).scale_to_fit_width(10.2).to_edge(UP, buff=0.5)
            question = Text({scene.purpose!r}, font_size=26, color=BLUE_B).scale_to_fit_width(8.8).next_to(title, DOWN, buff=0.55)
            focus = Text({focus_text!r}, font_size=30, color=YELLOW_D, weight=BOLD).scale_to_fit_width(6.5)
            side_panel = RoundedRectangle(corner_radius=0.22, width=4.6, height=4.4, color=BLUE_D, fill_color="#13253E", fill_opacity=0.9)
            side_panel.to_edge(RIGHT, buff=1.0).shift(DOWN * 0.35)
            focus.move_to(side_panel.get_center())
            lead_line = Line(LEFT * 4.4, side_panel.get_left() + RIGHT * 0.2, color=YELLOW_D, stroke_width=5).shift(DOWN * 0.2)
            chip_stack = VGroup(*[Text(label, font_size=20, color=GRAY_A).scale_to_fit_width(4.4) for label in {chips!r}]).arrange(DOWN, aligned_edge=LEFT, buff=0.26).move_to(LEFT * 3 + DOWN * 0.5)
            accent = Dot(color=YELLOW_D).move_to(lead_line.get_start())
            self.play(FadeIn(title, shift=UP * 0.12), run_time={timings[0]}, rate_func=smooth)
            self.play(FadeIn(question, shift=UP * 0.08), run_time={timings[1]}, rate_func=smooth)
            self.play(Create(lead_line), FadeIn(accent, scale=0.8), run_time={timings[2]}, rate_func=smooth)
            self.play(LaggedStart(*[FadeIn(item, shift=RIGHT * 0.12) for item in chip_stack], lag_ratio=0.18), run_time={timings[3]})
            self.play(FadeIn(side_panel, scale=0.96), FadeIn(focus), run_time={timings[4]}, rate_func=smooth)
            if hasattr(self.camera, "frame"):
                self.play(self.camera.frame.animate.scale(0.9).move_to(side_panel), run_time={timings[5]}, rate_func=smooth)
            self.wait({final_hold})
            """
        ).strip()
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=44, color=WHITE, weight=BOLD).scale_to_fit_width(10.6)
        strap = Text({scene.purpose!r}, font_size=24, color=BLUE_B).scale_to_fit_width(9.2)
        focus = Text({focus_text!r}, font_size=30, color=YELLOW_D, weight=BOLD).scale_to_fit_width(8.4)
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
    step_labels = [beat.visual for beat in beats[:3]]
    focus_steps = max(len(step_labels) - 1, 0)
    timings, final_hold = _timing_slots(scene, 5 + (focus_steps * 2))
    transition_timings = timings[5:]
    variant = _scene_variant(scene, 3)
    if variant == 0:
        return textwrap.dedent(
            f"""
            title = Text({scene.title!r}, font_size=36, color=WHITE, weight=BOLD).scale_to_fit_width(10.2).to_edge(UP, buff=0.5)
            stage_caption = Text({scene.purpose!r}, font_size=22, color=GRAY_A).scale_to_fit_width(9.2).to_edge(DOWN, buff=0.5)
            columns = VGroup()
            for label in {step_labels!r}:
                block = RoundedRectangle(corner_radius=0.2, width=2.8, height=2.0, color=BLUE_D, fill_color="#14314A", fill_opacity=0.92)
                inner = RoundedRectangle(corner_radius=0.16, width=2.25, height=1.25, color=YELLOW_D, fill_color="#193C5A", fill_opacity=0.55)
                caption = Text(label, font_size=22, color=WHITE).scale_to_fit_width(2.1)
                card = VGroup(block, inner, caption)
                inner.move_to(block.get_center())
                caption.move_to(inner.get_center())
                columns.add(card)
            columns.arrange(RIGHT, buff=0.55).next_to(title, DOWN, buff=0.9)
            connectors = VGroup(*[Arrow(columns[index][0].get_right(), columns[index + 1][0].get_left(), buff=0.1, stroke_width=5, color=YELLOW_D) for index in range(len(columns) - 1)])
            guide = SurroundingRectangle(columns[0][0], color=YELLOW_D, buff=0.16)
            transition_timings = {transition_timings!r}
            self.play(FadeIn(title, shift=UP * 0.1), run_time={timings[0]}, rate_func=smooth)
            self.play(LaggedStart(*[FadeIn(column, shift=UP * 0.1) for column in columns], lag_ratio=0.18), run_time={timings[1]})
            self.play(Create(connectors), run_time={timings[2]}, rate_func=smooth)
            self.play(FadeIn(stage_caption, shift=UP * 0.08), run_time={timings[3]})
            self.play(Create(guide), run_time={timings[4]})
            transition_index = 0
            for column in columns[1:]:
                self.play(guide.animate.move_to(column[0].get_center()), run_time=transition_timings[transition_index], rate_func=smooth)
                transition_index += 1
                if hasattr(self.camera, "frame"):
                    self.play(self.camera.frame.animate.scale(0.95).move_to(column[0]), run_time=transition_timings[transition_index], rate_func=smooth)
                    transition_index += 1
            self.wait({final_hold})
            """
        ).strip()
    if variant == 1:
        return textwrap.dedent(
            f"""
            title = Text({scene.title!r}, font_size=36, color=WHITE, weight=BOLD).scale_to_fit_width(10.2).to_edge(UP, buff=0.45)
            stage_caption = Text({scene.purpose!r}, font_size=22, color=GRAY_A).scale_to_fit_width(9.4).to_edge(DOWN, buff=0.45)
            nodes = VGroup()
            positions = [LEFT * 4 + UP * 0.5, ORIGIN + DOWN * 0.2, RIGHT * 4 + UP * 0.5]
            colors = [BLUE_D, TEAL_D, YELLOW_D]
            fills = ["#14314A", "#143A3D", "#4A3C14"]
            for label, position, color, fill in zip({step_labels!r}, positions, colors, fills, strict=False):
                orb = Circle(radius=1.0, color=color, fill_color=fill, fill_opacity=0.95)
                caption = Text(label, font_size=22, color=WHITE).scale_to_fit_width(1.8)
                group = VGroup(orb, caption).move_to(position)
                caption.move_to(orb.get_center())
                nodes.add(group)
            links = VGroup(*[CurvedArrow(nodes[index][0].get_right(), nodes[index + 1][0].get_left(), angle=-0.3, color=GRAY_A, stroke_width=4) for index in range(len(nodes) - 1)])
            highlight = SurroundingRectangle(nodes[0][0], color=YELLOW_D, buff=0.15)
            transition_timings = {transition_timings!r}
            self.play(FadeIn(title, shift=UP * 0.1), run_time={timings[0]}, rate_func=smooth)
            self.play(LaggedStart(*[FadeIn(node, scale=0.9) for node in nodes], lag_ratio=0.16), run_time={timings[1]})
            self.play(Create(links), run_time={timings[2]}, rate_func=smooth)
            self.play(FadeIn(stage_caption, shift=UP * 0.08), run_time={timings[3]})
            self.play(Create(highlight), run_time={timings[4]})
            transition_index = 0
            for node in nodes[1:]:
                self.play(highlight.animate.move_to(node[0].get_center()), run_time=transition_timings[transition_index], rate_func=smooth)
                transition_index += 1
                if hasattr(self.camera, "frame"):
                    self.play(self.camera.frame.animate.scale(0.94).move_to(node[0]), run_time=transition_timings[transition_index], rate_func=smooth)
                    transition_index += 1
            self.wait({final_hold})
            """
        ).strip()
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=36, color=WHITE, weight=BOLD).to_edge(UP)
        boxes = VGroup()
        for label in {step_labels!r}:
            base = RoundedRectangle(
                corner_radius=0.18,
                width=2.55,
                height=1.35,
                color=BLUE_D,
                fill_color="#17314A",
                fill_opacity=0.88,
            )
            caption = Text(label, font_size=22, color=WHITE).scale_to_fit_width(2.0)
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
        stage_caption = Text({scene.purpose!r}, font_size=22, color=GRAY_A).scale_to_fit_width(9.4).to_edge(DOWN)
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
    insight_text = scene.visual_beats[-1].visual if scene.visual_beats else scene.narration_text
    variant = _scene_variant(scene, 3)
    if variant == 0:
        return textwrap.dedent(
            f"""
            title = Text({scene.title!r}, font_size=36, color=WHITE, weight=BOLD).scale_to_fit_width(10.4).to_edge(UP, buff=0.45)
            left_panel = RoundedRectangle(corner_radius=0.22, width=6.9, height=4.8, color=BLUE_D, fill_color="#101F34", fill_opacity=0.92).shift(LEFT * 2.45 + DOWN * 0.1)
            right_panel = RoundedRectangle(corner_radius=0.22, width=3.7, height=4.8, color=GRAY_C, fill_color="#0F1522", fill_opacity=0.92).shift(RIGHT * 4.0 + DOWN * 0.1)
            metrics = VGroup()
            for label, value, is_focus in zip({labels!r}, {values!r}, {[datum.emphasis for datum in chart_data]!r}, strict=False):
                name = Text(label, font_size=20, color=GRAY_A).scale_to_fit_width(2.6)
                stat = Text(f"{{value:.0f}}", font_size=30 if is_focus else 26, color=YELLOW_D if is_focus else WHITE, weight=BOLD)
                row = VGroup(name, stat).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
                metrics.add(row)
            metrics.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(right_panel.get_center())
            bars = VGroup()
            max_value = {max(max(values), 1):.2f}
            for index, value in enumerate({values!r}):
                is_focus = index == {emphasis_index}
                track = RoundedRectangle(corner_radius=0.12, width=5.2, height=0.58, color=GRAY_C, fill_color="#1D2A3C", fill_opacity=0.75)
                fill = RoundedRectangle(corner_radius=0.12, width=max((value / max_value) * 5.2, 0.8), height=0.58, color=YELLOW_D if is_focus else BLUE_D, fill_color=YELLOW_D if is_focus else BLUE_E, fill_opacity=0.92)
                fill.align_to(track, LEFT)
                fill.move_to(track.get_left() + RIGHT * (fill.width / 2))
                label = Text({labels!r}[index], font_size=20, color=WHITE).scale_to_fit_width(2.0)
                row = VGroup(label, VGroup(track, fill)).arrange(RIGHT, buff=0.35)
                bars.add(row)
            bars.arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to(left_panel.get_center())
            insight = Text({insight_text!r}, font_size=22, color=WHITE).scale_to_fit_width(9.2).to_edge(DOWN, buff=0.45)
            self.play(FadeIn(title, shift=UP * 0.08), run_time={timings[0]}, rate_func=smooth)
            self.play(FadeIn(left_panel, scale=0.97), FadeIn(right_panel, scale=0.97), run_time={timings[1]}, rate_func=smooth)
            self.play(LaggedStart(*[FadeIn(row, shift=RIGHT * 0.1) for row in bars], lag_ratio=0.16), run_time={timings[2]})
            self.play(LaggedStart(*[FadeIn(metric, shift=UP * 0.08) for metric in metrics], lag_ratio=0.14), run_time={timings[3]})
            self.play(FadeIn(insight, shift=UP * 0.08), run_time={timings[4]})
            if hasattr(self.camera, "frame"):
                self.play(self.camera.frame.animate.scale(0.9).move_to(left_panel), run_time={timings[5]}, rate_func=smooth)
            self.wait({final_hold})
            """
        ).strip()
    if variant == 1:
        return textwrap.dedent(
            f"""
            title = Text({scene.title!r}, font_size=36, color=WHITE, weight=BOLD).scale_to_fit_width(10.4).to_edge(UP, buff=0.45)
            headline_value = Text(f"{{{values[emphasis_index]:.0f}}}", font_size=54, color=YELLOW_D, weight=BOLD)
            headline_value.next_to(title, DOWN, buff=0.55)
            focus_label = Text({labels[emphasis_index]!r}, font_size=22, color=GRAY_A).next_to(headline_value, DOWN, buff=0.12)
            comparison = VGroup()
            for label, value in zip({labels!r}, {values!r}, strict=False):
                stat = Text(f"{{value:.0f}}", font_size=26, color=WHITE, weight=BOLD)
                caption = Text(label, font_size=20, color=GRAY_A).scale_to_fit_width(2.2)
                comparison.add(VGroup(stat, caption).arrange(DOWN, buff=0.08))
            comparison.arrange(RIGHT, buff=0.8).next_to(focus_label, DOWN, buff=0.8)
            divider = Line(LEFT * 4.8, RIGHT * 4.8, color=GRAY_C).next_to(comparison, DOWN, buff=0.55)
            insight = Text({insight_text!r}, font_size=22, color=WHITE).scale_to_fit_width(9.0).next_to(divider, DOWN, buff=0.4)
            halo = SurroundingRectangle(comparison[{emphasis_index}], color=YELLOW_D, buff=0.18)
            self.play(FadeIn(title, shift=UP * 0.08), run_time={timings[0]}, rate_func=smooth)
            self.play(FadeIn(headline_value, scale=0.92), FadeIn(focus_label, shift=UP * 0.08), run_time={timings[1]}, rate_func=smooth)
            self.play(LaggedStart(*[FadeIn(item, shift=UP * 0.1) for item in comparison], lag_ratio=0.14), run_time={timings[2]})
            self.play(Create(halo), Create(divider), run_time={timings[3]}, rate_func=smooth)
            self.play(FadeIn(insight, shift=UP * 0.08), run_time={timings[4]})
            if hasattr(self.camera, "frame"):
                self.play(self.camera.frame.animate.scale(0.92).move_to(comparison[{emphasis_index}]), run_time={timings[5]}, rate_func=smooth)
            self.wait({final_hold})
            """
        ).strip()
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=36, color=WHITE, weight=BOLD).to_edge(UP)
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
            value_label = Text(f"{{value:.0f}}", font_size=22, color=WHITE).next_to(bar, UP, buff=0.14)
            bars.add(VGroup(bar, value_label))
            caption = Text({labels!r}[index - 1], font_size=18, color=GRAY_A).scale_to_fit_width(2.1)
            caption.next_to(axes.c2p(index, 0), DOWN, buff=0.24)
            labels_group.add(caption)
        insight = Text({insight_text!r}, font_size=22, color=WHITE).scale_to_fit_width(9.4).to_edge(DOWN)
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
            color="#2FAE7A",
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


def _scene_variant(scene: SceneSpec, variant_count: int) -> int:
    if variant_count <= 1:
        return 0
    return _render_seed(scene) % variant_count


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
    cleaned = [_truncate_label(value.strip()) for value in values if value.strip()]
    if len(cleaned) >= min_count:
        return cleaned[:limit]

    fallback_points = [_truncate_label(segment.strip()) for segment in fallback.split(".") if segment.strip()]
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

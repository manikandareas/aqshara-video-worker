from __future__ import annotations

import ast
import hashlib
import textwrap

from aqshara_video_worker.schemas import SceneSpec


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
    body = _render_template_body(scene)
    construct_body = textwrap.indent(
        body.strip(),
        " " * 8,
        lambda line: True,
    )
    module = (
        "from manim import *\n\n"
        'config.background_color = "#0B1020"\n\n\n'
        f"class {class_name}(Scene):\n"
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
        raise CodeValidationError("Generated class must inherit from Scene")

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
            target_names = [
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            ]
            if target_names != ["RENDER_SEED"]:
                raise CodeValidationError(
                    "Generated Scene class may only define the RENDER_SEED assignment"
                )

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            raise CodeValidationError(
                f"Forbidden syntax in generated code: {type(node).__name__}"
            )

        if isinstance(node, ast.Import):
            raise CodeValidationError("Direct import statements are not allowed")

        if isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORT_MODULES:
                raise CodeValidationError(
                    f"Import from module {node.module!r} is not allowed"
                )

        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in FORBIDDEN_CALL_NAMES:
                raise CodeValidationError(
                    f"Forbidden call in generated code: {call_name}"
                )

            root_name = _root_name(node.func)
            if root_name and root_name in FORBIDDEN_ROOT_NAMES:
                raise CodeValidationError(
                    f"Forbidden root object in generated code: {root_name}"
                )

            attr_name = _attribute_name(node.func)
            if attr_name and attr_name in FORBIDDEN_ATTRIBUTE_NAMES:
                raise CodeValidationError(
                    f"Forbidden attribute call in generated code: {attr_name}"
                )


def _render_template_body(scene: SceneSpec) -> str:
    match scene.template_type:
        case "title":
            return _render_title_scene(scene)
        case "bullet":
            return _render_bullet_scene(scene)
        case "pipeline":
            return _render_pipeline_scene(scene)
        case "chart":
            return _render_chart_scene(scene)
        case "conclusion":
            return _render_conclusion_scene(scene)
        case _:
            raise ManimCodegenError(
                f"Unsupported template type for codegen: {scene.template_type}"
            )


def _render_title_scene(scene: SceneSpec) -> str:
    subtitle = scene.visual_elements[1] if len(scene.visual_elements) > 1 else scene.purpose
    insight = scene.visual_elements[-1]
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=38, color=WHITE).scale_to_fit_width(11)
        subtitle = Text({subtitle!r}, font_size=24, color=BLUE_B).scale_to_fit_width(10)
        insight = Text({insight!r}, font_size=26, color=YELLOW_D).scale_to_fit_width(9)
        divider = Line(LEFT * 4.8, RIGHT * 4.8, color=BLUE_D)
        group = VGroup(title, subtitle, divider, insight).arrange(DOWN, buff=0.4)
        self.play(Write(title), run_time=1.2, rate_func=smooth)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.9, rate_func=smooth)
        self.play(Create(divider), run_time=0.8, rate_func=smooth)
        self.play(FadeIn(insight, shift=UP * 0.15), run_time=1.0, rate_func=smooth)
        self.wait(0.6)
        """
    ).strip()


def _render_bullet_scene(scene: SceneSpec) -> str:
    points = _normalize_points(scene.visual_elements, fallback=scene.narration_text, limit=4)
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
        self.play(Write(title), run_time=0.9, rate_func=smooth)
        for bullet in bullets:
            self.play(FadeIn(bullet, shift=RIGHT * 0.2), run_time=0.65, rate_func=smooth)
        self.play(Create(highlight), run_time=0.7, rate_func=smooth)
        self.wait(0.5)
        """
    ).strip()


def _render_pipeline_scene(scene: SceneSpec) -> str:
    steps = _normalize_points(scene.visual_elements, fallback=scene.narration_text, min_count=3, limit=4)
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=34, color=WHITE).to_edge(UP)
        step_labels = {steps!r}
        boxes = VGroup()
        for label in step_labels:
            box = RoundedRectangle(
                corner_radius=0.18,
                width=2.2,
                height=1.1,
                color=BLUE_D,
                fill_color=BLUE_E,
                fill_opacity=0.45,
            )
            caption = Text(label, font_size=20, color=WHITE).scale_to_fit_width(1.8)
            boxes.add(VGroup(box, caption))
        boxes.arrange(RIGHT, buff=0.45).next_to(title, DOWN, buff=1.0)
        connectors = VGroup(
            *[
                Arrow(
                    boxes[index].get_right(),
                    boxes[index + 1].get_left(),
                    buff=0.12,
                    stroke_width=4,
                    color=YELLOW_D,
                )
                for index in range(len(boxes) - 1)
            ]
        )
        emphasis = SurroundingRectangle(boxes[-1], color=YELLOW_D, buff=0.12)
        self.play(Write(title), run_time=0.8, rate_func=smooth)
        self.play(
            LaggedStart(*[FadeIn(box, shift=UP * 0.15) for box in boxes], lag_ratio=0.2),
            run_time=1.4,
            rate_func=smooth,
        )
        self.play(Create(connectors), run_time=0.8, rate_func=smooth)
        self.play(Create(emphasis), run_time=0.7, rate_func=smooth)
        self.wait(0.5)
        """
    ).strip()


def _render_chart_scene(scene: SceneSpec) -> str:
    labels = _normalize_points(scene.visual_elements, fallback="trend", min_count=3, limit=3)
    values = _chart_values(labels)
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=34, color=WHITE).to_edge(UP)
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=7.5,
            y_length=4.0,
            tips=False,
            axis_config={{"color": GRAY_B}},
        ).next_to(title, DOWN, buff=0.8)
        bar_labels = {labels!r}
        bar_values = {values!r}
        bars = VGroup()
        captions = VGroup()
        for index, value in enumerate(bar_values, start=1):
            bar = Rectangle(
                width=0.85,
                height=value,
                color=BLUE_D if index < len(bar_values) else YELLOW_D,
                fill_color=BLUE_E if index < len(bar_values) else YELLOW_D,
                fill_opacity=0.7,
            )
            bar.move_to(axes.c2p(index, value / 2))
            label = Text(bar_labels[index - 1], font_size=18, color=GRAY_A).scale_to_fit_width(2.1)
            label.next_to(axes.c2p(index, 0), DOWN, buff=0.25)
            bars.add(bar)
            captions.add(label)
        highlight = Text({scene.narration_text!r}, font_size=20, color=WHITE).scale_to_fit_width(10).to_edge(DOWN)
        self.play(Write(title), run_time=0.8, rate_func=smooth)
        self.play(Create(axes), run_time=0.9, rate_func=smooth)
        self.play(
            LaggedStart(*[FadeIn(bar, shift=UP * 0.2) for bar in bars], lag_ratio=0.18),
            run_time=1.2,
            rate_func=smooth,
        )
        self.play(FadeIn(captions), run_time=0.7, rate_func=smooth)
        self.play(FadeIn(highlight, shift=UP * 0.15), run_time=0.8, rate_func=smooth)
        self.wait(0.5)
        """
    ).strip()


def _render_conclusion_scene(scene: SceneSpec) -> str:
    takeaways = _normalize_points(scene.visual_elements, fallback=scene.narration_text, limit=3)
    return textwrap.dedent(
        f"""
        title = Text({scene.title!r}, font_size=34, color=WHITE).to_edge(UP)
        card = RoundedRectangle(
            corner_radius=0.22,
            width=10.5,
            height=4.6,
            color=EMERALD,
            fill_color=GREEN_E,
            fill_opacity=0.35,
        ).next_to(title, DOWN, buff=0.7)
        takeaways = VGroup(
            *[
                Text(f"• {{point}}", font_size=24, color=GRAY_A).scale_to_fit_width(8.9)
                for point in {takeaways!r}
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to(card.get_center())
        closing = Text("From dense paper to clear takeaway.", font_size=20, color=YELLOW_D).to_edge(DOWN)
        self.play(Write(title), run_time=0.8, rate_func=smooth)
        self.play(FadeIn(card, scale=0.96), run_time=0.8, rate_func=smooth)
        self.play(
            LaggedStart(*[FadeIn(item, shift=UP * 0.12) for item in takeaways], lag_ratio=0.18),
            run_time=1.1,
            rate_func=smooth,
        )
        self.play(FadeIn(closing, shift=UP * 0.1), run_time=0.7, rate_func=smooth)
        self.wait(0.6)
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


def _chart_values(labels: list[str]) -> list[float]:
    values: list[float] = []
    for index, label in enumerate(labels, start=1):
        base = 1.2 + (index * 0.55)
        modifier = (sum(ord(character) for character in label) % 6) * 0.1
        values.append(round(min(base + modifier, 3.6), 2))
    return values


def _is_scene_base(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "Scene"


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

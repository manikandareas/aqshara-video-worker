import pytest

from aqshara_video_worker.pipeline.codegen import (
    CodeValidationError,
    build_scene_module,
    validate_generated_code,
)
from aqshara_video_worker.schemas import SceneSpec


@pytest.mark.parametrize(
    ("template_type", "visual_elements"),
    [
        ("title", ["Topic", "Problem framing", "Key result"]),
        ("bullet", ["Point one", "Point two", "Point three"]),
        ("pipeline", ["Input", "Transform", "Output"]),
        ("chart", ["baseline", "improved metric", "highlight 92%"]),
        ("conclusion", ["Takeaway one", "Takeaway two", "Takeaway three"]),
    ],
)
def test_build_scene_module_outputs_valid_manim_code(
    template_type: str,
    visual_elements: list[str],
) -> None:
    scene = SceneSpec(
        scene_index=1,
        title=f"{template_type.title()} Scene",
        template_type=template_type,  # type: ignore[arg-type]
        purpose="Explain the scene clearly",
        narration_text="Narration for the scene goes here.",
        planned_duration_ms=8000,
        visual_elements=visual_elements,
    )

    class_name, code = build_scene_module(scene)

    assert class_name.startswith("Scene01")
    assert "from manim import *" in code
    assert "class " in code

    validate_generated_code(code, expected_class_name=class_name)

    construct_line_index = code.splitlines().index("    def construct(self):")
    for line in code.splitlines()[construct_line_index + 1 :]:
        if not line.strip():
            continue
        assert line.startswith("        ")


def test_validate_generated_code_rejects_forbidden_imports() -> None:
    with pytest.raises(CodeValidationError, match="not allowed"):
        validate_generated_code(
            """
from manim import *
import os

class UnsafeScene(Scene):
    def construct(self):
        self.wait(1)
"""
        )


def test_validate_generated_code_rejects_forbidden_calls() -> None:
    with pytest.raises(CodeValidationError, match="Forbidden call"):
        validate_generated_code(
            """
from manim import *

class UnsafeScene(Scene):
    def construct(self):
        open("secret.txt")
"""
        )


def test_validate_generated_code_rejects_class_level_statements() -> None:
    with pytest.raises(CodeValidationError, match="may only define the RENDER_SEED assignment"):
        validate_generated_code(
            """
from manim import *

class UnsafeScene(Scene):
    RENDER_SEED = 1
    helper = Text("oops")

    def construct(self):
        self.wait(1)
"""
        )

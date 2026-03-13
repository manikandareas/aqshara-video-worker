import pytest
from pydantic import ValidationError

from aqshara_video_worker.schemas import SceneSpec, VideoGenerateJobPayload


def test_job_payload_requires_positive_attempt() -> None:
    with pytest.raises(ValidationError):
        VideoGenerateJobPayload(
            video_job_id="vjob_1",
            document_id="doc_1",
            actor_id="user_1",
            target_duration_sec=60,
            voice="alloy",
            language="en",
            attempt=0,
        )


def test_scene_spec_enforces_template_and_duration_bounds() -> None:
    scene = SceneSpec(
        scene_index=1,
        title="Hook",
        template_type="hook",
        scene_kind="hook",
        purpose="Introduce the paper",
        narration_text="This paper explores a new method.",
        planned_duration_ms=12000,
        visual_elements=["headline", "subtitle"],
    )

    assert scene.template_type == "hook"
    assert scene.scene_kind == "hook"

    with pytest.raises(ValidationError):
        SceneSpec(
            scene_index=1,
            title="Invalid",
            template_type="orbit",
            purpose="Bad template",
            narration_text="Nope",
            planned_duration_ms=12000,
            visual_elements=["headline"],
        )

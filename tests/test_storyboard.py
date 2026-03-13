from aqshara_video_worker.pipeline.storyboard import build_storyboard_artifacts


def test_storyboard_builder_creates_summary_scenes_and_markdown() -> None:
    artifacts = build_storyboard_artifacts(
        {
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
    )

    assert artifacts.summary.topic == "Adaptive Research Videos"
    assert artifacts.director_plan.style_profile == "3b1b-lite"
    assert len(artifacts.storyboard.scenes) == 5
    assert sum(scene.planned_duration_ms for scene in artifacts.storyboard.scenes) == 60000
    assert "## Scene 1: Adaptive Research Videos" in artifacts.scenes_markdown
    assert "## Recurring Motifs" in artifacts.scenes_markdown

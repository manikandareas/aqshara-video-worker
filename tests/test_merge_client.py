import pytest

from aqshara_video_worker.clients.merge_client import (
    AudioSyncValidationError,
    FfmpegMergeClient,
    MergeConfigurationError,
    MergeSceneAsset,
)
from aqshara_video_worker.config import WorkerSettings


BASE_SETTINGS = {
    "callback_base_url": "http://127.0.0.1:3000/api/v1",
    "internal_service_token": "token",
    "r2_endpoint": "http://localhost:9000",
    "r2_region": "auto",
    "r2_access_key_id": "key",
    "r2_secret_access_key": "secret",
    "r2_bucket": "bucket",
    "video_tts_openai_api_key": None,
    "video_tts_openai_base_url": "https://api.openai.com/v1",
    "video_tts_openai_model": "gpt-4o-mini-tts",
    "video_tts_openai_timeout_sec": 60,
    "video_creative_provider": "openai_compatible",
    "video_creative_api_key": None,
    "video_creative_base_url": None,
    "video_creative_generation_model": None,
    "video_creative_critique_model": None,
    "video_creative_codegen_model": None,
    "video_creative_timeout_sec": 90.0,
    "video_render_backend": "mock",
    "video_render_profile": "720p",
    "video_render_timeout_sec": 180,
    "video_merge_timeout_sec": 30,
    "video_audio_sync_max_drift_pct": 15.0,
    "ffmpeg_binary": "ffmpeg",
    "ffprobe_binary": "ffprobe",
    "daytona_api_url": None,
    "daytona_api_key": None,
    "daytona_target": None,
    "daytona_python_version": "3.12",
    "daytona_render_image": None,
    "daytona_create_timeout_sec": 300,
    "daytona_render_cpu": 2,
    "daytona_render_memory_gb": 4,
    "daytona_render_disk_gb": 8,
}


def build_settings(**overrides: object) -> WorkerSettings:
    return WorkerSettings.model_construct(**(BASE_SETTINGS | overrides))


@pytest.mark.asyncio
async def test_merge_client_merges_scene_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aqshara_video_worker.clients.merge_client.shutil.which",
        lambda binary: f"/usr/bin/{binary}",
    )
    client = FfmpegMergeClient(build_settings())

    async def fake_run_command(command: list[str]):
        output_path = command[-1]
        if output_path.endswith(".mp4"):
            with open(output_path, "wb") as handle:
                handle.write(f"generated:{output_path}".encode("utf-8"))

        return type(
            "CommandResult",
            (),
            {
                "stdout": f"ran {' '.join(command[:3])}",
                "stderr": "",
            },
        )()

    async def fake_probe_duration(output_path):
        path = str(output_path)
        if "clips/scene-01" in path or path.endswith("scene-01.mp4"):
            return 6.0
        if "clips/scene-02" in path or path.endswith("scene-02.mp4"):
            return 6.0
        return 12.0

    monkeypatch.setattr(client, "_run_command", fake_run_command)
    monkeypatch.setattr(client, "_probe_duration", fake_probe_duration)

    result = await client.merge_scenes(
        scenes=[
            MergeSceneAsset(
                scene_index=2,
                video_bytes=b"video-2",
                audio_bytes=b"audio-2",
                audio_duration_ms=6000,
            ),
            MergeSceneAsset(
                scene_index=1,
                video_bytes=b"video-1",
                audio_bytes=b"audio-1",
                audio_duration_ms=6000,
            ),
        ],
        render_profile="720p",
    )

    assert result.video_bytes.startswith(b"generated:")
    assert result.duration_sec == 12.0
    assert "render_profile=720p" in result.stdout
    assert "audio-sync:" in result.stdout


@pytest.mark.asyncio
async def test_merge_client_rejects_audio_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aqshara_video_worker.clients.merge_client.shutil.which",
        lambda binary: f"/usr/bin/{binary}",
    )
    client = FfmpegMergeClient(build_settings())

    async def fake_run_command(command: list[str]):
        output_path = command[-1]
        if output_path.endswith(".mp4"):
            with open(output_path, "wb") as handle:
                handle.write(b"video")

        return type(
            "CommandResult",
            (),
            {
                "stdout": "",
                "stderr": "",
            },
        )()

    async def fake_probe_duration(output_path):
        path = str(output_path)
        if "clips/scene-01" in path:
            return 10.0
        if path.endswith("scene-01.mp4"):
            return 10.0
        return 20.0

    monkeypatch.setattr(client, "_run_command", fake_run_command)
    monkeypatch.setattr(client, "_probe_duration", fake_probe_duration)

    with pytest.raises(AudioSyncValidationError):
        await client.merge_scenes(
            scenes=[
                MergeSceneAsset(
                    scene_index=1,
                    video_bytes=b"video-1",
                    audio_bytes=b"audio-1",
                    audio_duration_ms=10_000,
                ),
            ],
            render_profile="720p",
        )


@pytest.mark.asyncio
async def test_merge_client_rejects_scene_level_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "aqshara_video_worker.clients.merge_client.shutil.which",
        lambda binary: f"/usr/bin/{binary}",
    )
    client = FfmpegMergeClient(build_settings())

    async def fake_run_command(command: list[str]):
        output_path = command[-1]
        if output_path.endswith(".mp4"):
            with open(output_path, "wb") as handle:
                handle.write(b"video")

        return type(
            "CommandResult",
            (),
            {
                "stdout": "",
                "stderr": "",
            },
        )()

    async def fake_probe_duration(output_path):
        path = str(output_path)
        if "clips/scene-01" in path:
            return 11.5
        if path.endswith("scene-01.mp4"):
            return 3.0
        return 11.5

    monkeypatch.setattr(client, "_run_command", fake_run_command)
    monkeypatch.setattr(client, "_probe_duration", fake_probe_duration)

    with pytest.raises(AudioSyncValidationError, match="scene=1"):
        await client.merge_scenes(
            scenes=[
                MergeSceneAsset(
                    scene_index=1,
                    video_bytes=b"video-1",
                    audio_bytes=b"audio-1",
                    audio_duration_ms=10_000,
                ),
            ],
            render_profile="720p",
        )


def test_merge_client_requires_ffmpeg_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aqshara_video_worker.clients.merge_client.shutil.which",
        lambda binary: None if binary == "ffmpeg" else "/usr/bin/ffprobe",
    )

    with pytest.raises(MergeConfigurationError, match="Missing required media binary: ffmpeg"):
        FfmpegMergeClient(build_settings())

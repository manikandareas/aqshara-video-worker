import pytest

from aqshara_video_worker.clients.render_client import (
    MockRenderClient,
    RenderConfigurationError,
    RenderClientError,
    DaytonaRenderClient,
)
from aqshara_video_worker.config import WorkerSettings


def build_settings() -> WorkerSettings:
    return WorkerSettings.model_construct(
        callback_base_url="http://127.0.0.1:8000/api/v1",
        internal_service_token="token",
        r2_endpoint="http://localhost:9000",
        r2_region="auto",
        r2_access_key_id="key",
        r2_secret_access_key="secret",
        r2_bucket="bucket",
        openai_api_key=None,
        openai_base_url="https://api.openai.com/v1",
        openai_tts_model="gpt-4o-mini-tts",
        openai_timeout_sec=60,
        video_render_backend="mock",
        video_render_profile="720p",
        video_render_timeout_sec=30,
        video_merge_timeout_sec=30,
        video_audio_sync_max_drift_pct=15.0,
        ffmpeg_binary="/opt/homebrew/bin/ffmpeg",
        ffprobe_binary="/opt/homebrew/bin/ffprobe",
        daytona_api_url=None,
        daytona_api_key=None,
        daytona_target=None,
        daytona_python_version="3.12",
        daytona_render_image=None,
        daytona_create_timeout_sec=300,
        daytona_render_cpu=2,
        daytona_render_memory_gb=4,
        daytona_render_disk_gb=8,
    )


@pytest.mark.asyncio
async def test_mock_render_client_produces_mp4_bytes() -> None:
    client = MockRenderClient(build_settings())

    result = await client.render_scene(
        scene_index=1,
        class_name="Scene01",
        scene_code="from manim import *",
        render_profile="720p",
    )

    assert len(result.video_bytes) > 0
    assert result.video_bytes[4:8] == b"ftyp"
    assert result.resolution == "1280x720"


@pytest.mark.asyncio
async def test_mock_render_client_requires_ffmpeg_binary() -> None:
    settings = build_settings()
    settings.ffmpeg_binary = "/missing/ffmpeg"
    client = MockRenderClient(settings)

    with pytest.raises(RenderConfigurationError, match="Missing required media binary"):
        await client.render_scene(
            scene_index=1,
            class_name="Scene01",
            scene_code="from manim import *",
            render_profile="720p",
        )


class _FakeExecResult:
    def __init__(self, *, exit_code=0, result="", stdout="", stderr="") -> None:
        self.exit_code = exit_code
        self.result = result
        self.stdout = stdout
        self.stderr = stderr


class _FakeProcess:
    async def exec(self, command: str, timeout=None):
        if command.startswith("mkdir -p "):
            return _FakeExecResult(exit_code=0)
        return _FakeExecResult(
            exit_code=1,
            result="manim traceback",
            stderr="ImportError: cairo backend missing",
        )


class _FakeFs:
    async def upload_file(self, content: bytes, path: str) -> None:
        return None


class _FakeSandbox:
    def __init__(self) -> None:
        self.process = _FakeProcess()
        self.fs = _FakeFs()

    async def delete(self) -> None:
        return None


class _FakeDaytonaClient:
    async def create(self, params, timeout=None):
        return _FakeSandbox()

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_daytona_render_client_surfaces_exec_output() -> None:
    settings = build_settings()
    settings.video_render_backend = "daytona"
    settings.daytona_api_key = "key"
    settings.daytona_target = "eu"
    client = DaytonaRenderClient(settings)
    client._client = _FakeDaytonaClient()

    with pytest.raises(RenderClientError, match="ImportError: cairo backend missing"):
        await client.render_scene(
            scene_index=1,
            class_name="Scene01",
            scene_code="from manim import *",
            render_profile="720p",
        )


class _QuotaFallbackDaytonaClient:
    def __init__(self) -> None:
        self.memory_attempts: list[int] = []

    async def create(self, params, timeout=None):
        self.memory_attempts.append(params.resources.memory)
        if params.resources.memory > 2:
            raise Exception(
                "Failed to create sandbox: Total memory limit exceeded. Maximum allowed: 10GiB."
            )
        return _FakeSandbox()

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_daytona_render_client_retries_with_smaller_resources() -> None:
    settings = build_settings()
    settings.video_render_backend = "daytona"
    settings.daytona_api_key = "key"
    settings.daytona_target = "eu"
    client = DaytonaRenderClient(settings)
    fake_client = _QuotaFallbackDaytonaClient()
    client._client = fake_client

    with pytest.raises(RenderClientError):
        await client.render_scene(
            scene_index=1,
            class_name="Scene01",
            scene_code="from manim import *",
            render_profile="720p",
        )

    assert fake_client.memory_attempts == [4, 2]

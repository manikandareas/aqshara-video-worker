from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Protocol

import httpx

from aqshara_video_worker.config import WorkerSettings

try:
    from daytona import (
        AsyncDaytona,
        CreateSandboxFromImageParams,
        DaytonaConfig,
        Image,
        Resources,
    )
except ImportError:  # pragma: no cover - guarded at runtime
    AsyncDaytona = None
    DaytonaConfig = None
    Image = None
    Resources = None
    CreateSandboxFromImageParams = None


RenderProfile = Literal["480p", "720p", "1080p"]


@dataclass(frozen=True)
class RenderSceneResult:
    video_bytes: bytes
    stdout: str
    stderr: str
    resolution: str
    render_profile: RenderProfile


class RenderClientError(Exception):
    pass


class EmptyRenderError(RenderClientError):
    pass


class RenderTimeoutError(RenderClientError):
    pass


class RenderConfigurationError(RenderClientError):
    pass


@dataclass(frozen=True)
class DaytonaResourceProfile:
    cpu: int
    memory_gb: int
    disk_gb: int


class RenderClient(Protocol):
    async def start_job(self) -> None: ...

    async def finish_job(self) -> None: ...

    async def render_scene(
        self,
        *,
        scene_index: int,
        class_name: str,
        scene_code: str,
        render_profile: RenderProfile,
    ) -> RenderSceneResult: ...

    async def extract_preview_frames(
        self,
        *,
        video_bytes: bytes,
        sample_count: int,
    ) -> list[bytes]: ...

    async def close(self) -> None: ...


class MockRenderClient:
    def __init__(self, settings: WorkerSettings) -> None:
        self._ffmpeg_binary = settings.ffmpeg_binary
        self._timeout_sec = settings.video_render_timeout_sec
        self._ffprobe_binary = settings.ffprobe_binary

    async def start_job(self) -> None:
        return None

    async def finish_job(self) -> None:
        return None

    async def render_scene(
        self,
        *,
        scene_index: int,
        class_name: str,
        scene_code: str,
        render_profile: RenderProfile,
    ) -> RenderSceneResult:
        resolution = "1280x720" if render_profile == "720p" else "854x480"
        width, height = (1280, 720) if render_profile == "720p" else (854, 480)

        try:
            with TemporaryDirectory(prefix="aqshara-mock-render-") as tmp_dir:
                output_path = Path(tmp_dir) / f"scene-{scene_index:02d}.mp4"
                process = await asyncio.create_subprocess_exec(
                    self._ffmpeg_binary,
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=black:s={width}x{height}:d=1:r=30",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(output_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self._timeout_sec,
                    )
                except TimeoutError as error:
                    process.kill()
                    await process.communicate()
                    raise RenderTimeoutError(
                        f"Timed out rendering mock scene {scene_index}",
                    ) from error

                decoded_stdout = stdout.decode("utf-8", errors="replace")
                decoded_stderr = stderr.decode("utf-8", errors="replace")
                if process.returncode != 0:
                    raise RenderClientError(
                        "Mock render command failed with exit code "
                        f"{process.returncode}: {decoded_stderr.strip()}",
                    )

                video_bytes = output_path.read_bytes()
        except FileNotFoundError as error:
            raise RenderConfigurationError(
                f"Missing required media binary: {error.filename}",
            ) from error

        if not video_bytes:
            raise EmptyRenderError("Rendered output was empty")

        return RenderSceneResult(
            video_bytes=video_bytes,
            stdout=(
                f"Rendered scene {scene_index} with mock backend "
                f"using profile {render_profile}"
            ),
            stderr="",
            resolution=resolution,
            render_profile=render_profile,
        )

    async def close(self) -> None:
        return None

    async def extract_preview_frames(
        self,
        *,
        video_bytes: bytes,
        sample_count: int,
    ) -> list[bytes]:
        return await _extract_preview_frames(
            ffmpeg_binary=self._ffmpeg_binary,
            ffprobe_binary=self._ffprobe_binary,
            video_bytes=video_bytes,
            sample_count=sample_count,
            timeout_sec=self._timeout_sec,
        )


class DaytonaRenderClient:
    _default_api_url = "https://app.daytona.io/api"

    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings
        self._client = self._create_daytona_client()
        self._sandbox_queue: asyncio.Queue | None = None
        self._sandboxs: list[object] = []
        self._provision_notes: list[str | None] = []
        self._pool_lock = asyncio.Lock()

    async def render_scene(
        self,
        *,
        scene_index: int,
        class_name: str,
        scene_code: str,
        render_profile: RenderProfile,
    ) -> RenderSceneResult:
        if not self._settings.daytona_api_key or not self._settings.daytona_target:
            raise RenderConfigurationError(
                "DAYTONA_API_KEY and DAYTONA_TARGET are required for the Daytona render backend"
            )
        if self._client is None:
            raise RenderConfigurationError(
                "The 'daytona' Python package is required for the Daytona render backend"
            )

        remote_root = "/home/daytona/workspace/aqshara"
        remote_scene_path = f"{remote_root}/scene.py"
        remote_output_name = f"scene-{scene_index:02d}"
        quality_flag, resolution = self._resolve_quality(render_profile)
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        try:
            sandbox, provision_note = await self._acquire_sandbox()
            if provision_note:
                stdout_chunks.append(provision_note)
        except Exception as error:  # pragma: no cover - depends on Daytona runtime
            raise RenderConfigurationError(
                f"Failed to provision Daytona sandbox: {error}",
            ) from error

        try:
            await sandbox.process.exec(f"mkdir -p {remote_root}/media")
            await sandbox.fs.upload_file(scene_code.encode("utf-8"), remote_scene_path)
            render_result = await sandbox.process.exec(
                "python -m manim "
                f"-q{quality_flag} "
                "--format=mp4 "
                f"--media_dir {remote_root}/media "
                f"-o {remote_output_name} "
                f"{remote_scene_path} {class_name}",
                timeout=self._settings.video_render_timeout_sec,
            )
            stdout_value, stderr_value = self._extract_exec_output(render_result)
            if stdout_value:
                stdout_chunks.append(stdout_value)
            if stderr_value:
                stderr_chunks.append(stderr_value)
            if getattr(render_result, "exit_code", 0) not in (0, None):
                failure_detail = "\n".join(
                    part
                    for part in [
                        stdout_value.strip(),
                        stderr_value.strip(),
                    ]
                    if part and part.strip()
                )
                raise RenderClientError(
                    "Daytona render exited with code "
                    f"{render_result.exit_code}"
                    + (f":\n{failure_detail}" if failure_detail else ""),
                )

            remote_video_path = await self._find_remote_video_path(
                sandbox,
                remote_root=remote_root,
                output_name=remote_output_name,
            )
            video_bytes = await sandbox.fs.download_file(remote_video_path)
            if not video_bytes:
                raise EmptyRenderError("Rendered output was empty")

            return RenderSceneResult(
                video_bytes=video_bytes,
                stdout="\n".join(chunk for chunk in stdout_chunks if chunk).strip(),
                stderr="\n".join(chunk for chunk in stderr_chunks if chunk).strip(),
                resolution=resolution,
                render_profile=render_profile,
            )
        except (TimeoutError, httpx.TimeoutException) as error:
            raise RenderTimeoutError(
                f"Timed out rendering scene {scene_index} in Daytona",
            ) from error
        except EmptyRenderError:
            raise
        except Exception as error:  # pragma: no cover - depends on Daytona runtime
            raise RenderClientError(
                f"Daytona render failed for scene {scene_index}: {error}",
            ) from error
        finally:
            await self._release_sandbox(sandbox)

    async def start_job(self) -> None:
        await self._ensure_pool()

    async def finish_job(self) -> None:
        await self._destroy_pool()

    async def close(self) -> None:
        await self._destroy_pool()
        if self._client is not None:
            await self._client.close()

    async def extract_preview_frames(
        self,
        *,
        video_bytes: bytes,
        sample_count: int,
    ) -> list[bytes]:
        return await _extract_preview_frames(
            ffmpeg_binary=self._settings.ffmpeg_binary,
            ffprobe_binary=self._settings.ffprobe_binary,
            video_bytes=video_bytes,
            sample_count=sample_count,
            timeout_sec=self._settings.video_render_timeout_sec,
        )

    def _create_daytona_client(self):
        if AsyncDaytona is None or DaytonaConfig is None:
            return None

        config_kwargs: dict[str, str] = {
            "api_key": self._settings.daytona_api_key or "",
            "api_url": self._settings.daytona_api_url or self._default_api_url,
            "target": self._settings.daytona_target or "",
        }
        return AsyncDaytona(DaytonaConfig(**config_kwargs))

    async def _create_sandbox_with_fallback(self):
        attempted_profiles: list[DaytonaResourceProfile] = []
        last_error: Exception | None = None

        for profile in self._resource_profiles():
            attempted_profiles.append(profile)
            try:
                sandbox = await self._client.create(
                    self._build_sandbox_params(profile),
                    timeout=self._settings.daytona_create_timeout_sec,
                )
                provision_note = None
                if profile != attempted_profiles[0]:
                    provision_note = (
                        "Provisioned Daytona sandbox with fallback resources "
                        f"cpu={profile.cpu}, memory_gb={profile.memory_gb}, disk_gb={profile.disk_gb}"
                    )
                return sandbox, provision_note
            except Exception as error:
                last_error = error
                if not self._is_resource_limit_error(error):
                    raise

        raise last_error if last_error is not None else RenderConfigurationError(
            "Failed to provision Daytona sandbox",
        )

    def _build_sandbox_params(self, profile: DaytonaResourceProfile):
        if (
            CreateSandboxFromImageParams is None
            or Resources is None
            or Image is None
        ):
            raise RenderConfigurationError(
                "The Daytona SDK does not expose image-based sandbox creation in this environment",
            )

        if self._settings.daytona_render_image:
            sandbox_image = self._settings.daytona_render_image
        else:
            sandbox_image = (
                Image.debian_slim(self._settings.daytona_python_version)
                .run_commands(
                    "apt-get update",
                    "apt-get install -y ffmpeg pkg-config libcairo2-dev libpango1.0-dev",
                    "rm -rf /var/lib/apt/lists/*",
                )
                .pip_install("manim==0.19.0")
            )

        return CreateSandboxFromImageParams(
            image=sandbox_image,
            resources=Resources(
                cpu=profile.cpu,
                memory=profile.memory_gb,
                disk=profile.disk_gb,
            ),
        )

    def _resource_profiles(self) -> list[DaytonaResourceProfile]:
        base = DaytonaResourceProfile(
            cpu=self._settings.daytona_render_cpu,
            memory_gb=self._settings.daytona_render_memory_gb,
            disk_gb=self._settings.daytona_render_disk_gb,
        )
        candidates = [
            base,
            DaytonaResourceProfile(
                cpu=min(base.cpu, 2),
                memory_gb=min(base.memory_gb, 2),
                disk_gb=base.disk_gb,
            ),
            DaytonaResourceProfile(
                cpu=1,
                memory_gb=1,
                disk_gb=base.disk_gb,
            ),
        ]
        unique_profiles: list[DaytonaResourceProfile] = []
        for profile in candidates:
            if profile not in unique_profiles:
                unique_profiles.append(profile)
        return unique_profiles

    @staticmethod
    def _is_resource_limit_error(error: Exception) -> bool:
        message = str(error).lower()
        return "memory limit exceeded" in message or "maximum allowed" in message

    @staticmethod
    def _resolve_quality(render_profile: RenderProfile) -> tuple[str, str]:
        if render_profile == "480p":
            return ("l", "854x480")
        if render_profile == "1080p":
            return ("h", "1920x1080")
        return ("m", "1280x720")

    @staticmethod
    async def _find_remote_video_path(
        sandbox,
        *,
        remote_root: str,
        output_name: str,
    ) -> str:
        result = await sandbox.process.exec(
            f"find {remote_root}/media -type f -name '{output_name}.mp4' -print -quit",
            timeout=30,
        )
        video_path = (result.result or "").strip()
        if not video_path:
            fallback = await sandbox.process.exec(
                f"find {remote_root}/media -type f -name '*.mp4' -print -quit",
                timeout=30,
            )
            video_path = (fallback.result or "").strip()
        if not video_path:
            raise EmptyRenderError("Daytona did not produce an mp4 artifact")
        return video_path

    @staticmethod
    def _extract_exec_output(result) -> tuple[str, str]:
        stdout_parts = [
            getattr(result, "result", ""),
            getattr(result, "stdout", ""),
            getattr(result, "output", ""),
        ]
        stderr_parts = [
            getattr(result, "stderr", ""),
            getattr(result, "error", ""),
        ]
        stdout_value = "\n".join(str(part) for part in stdout_parts if part).strip()
        stderr_value = "\n".join(str(part) for part in stderr_parts if part).strip()
        return stdout_value, stderr_value

    async def _ensure_pool(self) -> None:
        async with self._pool_lock:
            if self._sandbox_queue is not None:
                return

            queue: asyncio.Queue = asyncio.Queue()
            provision_notes: list[str | None] = []
            pool_size = max(
                1,
                min(
                    self._settings.video_daytona_sandbox_pool_size,
                    self._settings.video_render_concurrency,
                ),
            )
            for _ in range(pool_size):
                sandbox, provision_note = await self._create_sandbox_with_fallback()
                await queue.put(sandbox)
                self._sandboxs.append(sandbox)
                provision_notes.append(provision_note)
            self._sandbox_queue = queue
            self._provision_notes = provision_notes

    async def _destroy_pool(self) -> None:
        async with self._pool_lock:
            sandboxs = self._sandboxs
            self._sandboxs = []
            self._sandbox_queue = None
            self._provision_notes = []

        for sandbox in sandboxs:
            try:
                await sandbox.delete()
            except Exception:
                pass

    async def _acquire_sandbox(self):
        await self._ensure_pool()
        assert self._sandbox_queue is not None
        sandbox = await self._sandbox_queue.get()
        provision_note = None
        if getattr(self, "_provision_notes", None):
            provision_note = self._provision_notes.pop(0)
        return sandbox, provision_note

    async def _release_sandbox(self, sandbox) -> None:
        if self._sandbox_queue is None:
            return
        await self._sandbox_queue.put(sandbox)


def create_render_client(settings: WorkerSettings) -> RenderClient:
    backend = settings.video_render_backend.strip().lower()
    if backend == "mock":
        return MockRenderClient(settings)
    if backend == "daytona":
        return DaytonaRenderClient(settings)
    raise RenderConfigurationError(f"Unsupported VIDEO_RENDER_BACKEND: {backend}")


async def _extract_preview_frames(
    *,
    ffmpeg_binary: str,
    ffprobe_binary: str,
    video_bytes: bytes,
    sample_count: int,
    timeout_sec: int,
) -> list[bytes]:
    if sample_count <= 0:
        return []

    try:
        with TemporaryDirectory(prefix="aqshara-frame-sample-") as tmp_dir:
            workspace = Path(tmp_dir)
            video_path = workspace / "scene.mp4"
            video_path.write_bytes(video_bytes)
            duration = await _probe_video_duration(
                ffprobe_binary=ffprobe_binary,
                video_path=video_path,
                timeout_sec=timeout_sec,
            )
            if duration <= 0:
                duration = 1.0

            timestamps = _frame_timestamps(duration, sample_count)
            frames: list[bytes] = []
            for index, timestamp in enumerate(timestamps, start=1):
                frame_path = workspace / f"frame-{index:02d}.png"
                process = await asyncio.create_subprocess_exec(
                    ffmpeg_binary,
                    "-y",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(frame_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    _stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout_sec,
                    )
                except TimeoutError as error:
                    process.kill()
                    await process.communicate()
                    raise RenderTimeoutError("Timed out extracting preview frames") from error

                if process.returncode != 0 or not frame_path.exists():
                    raise RenderClientError(
                        "Failed to extract preview frame: "
                        f"{stderr.decode('utf-8', errors='replace').strip()}"
                    )
                frames.append(frame_path.read_bytes())
            return frames
    except FileNotFoundError as error:
        raise RenderConfigurationError(
            f"Missing required media binary: {error.filename}",
        ) from error


async def _probe_video_duration(
    *,
    ffprobe_binary: str,
    video_path: Path,
    timeout_sec: int,
) -> float:
    process = await asyncio.create_subprocess_exec(
        ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_sec,
        )
    except TimeoutError as error:
        process.kill()
        await process.communicate()
        raise RenderTimeoutError("Timed out probing rendered video duration") from error

    if process.returncode != 0:
        raise RenderClientError(
            "ffprobe failed while sampling preview frames: "
            f"{stderr.decode('utf-8', errors='replace').strip()}"
        )

    raw_value = stdout.decode("utf-8", errors="replace").strip()
    try:
        return float(raw_value)
    except ValueError:
        return 0.0


def _frame_timestamps(duration_sec: float, sample_count: int) -> list[float]:
    if sample_count == 1:
        return [max(duration_sec * 0.5, 0.0)]

    usable_duration = max(duration_sec - 0.1, 0.1)
    return [
        round((usable_duration * index) / (sample_count - 1), 3)
        for index in range(sample_count)
    ]

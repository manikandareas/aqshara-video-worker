from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Protocol

from aqshara_video_worker.config import WorkerSettings


RenderProfile = Literal["480p", "720p", "1080p"]


@dataclass(frozen=True)
class MergeSceneAsset:
    scene_index: int
    video_bytes: bytes
    audio_bytes: bytes
    audio_duration_ms: int


@dataclass(frozen=True)
class MergeVideoResult:
    video_bytes: bytes
    stdout: str
    stderr: str
    duration_sec: float


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str


class MergeClientError(Exception):
    pass


class MergeConfigurationError(MergeClientError):
    pass


class MergeExecutionError(MergeClientError):
    pass


class AudioSyncValidationError(MergeClientError):
    pass


class MergeClient(Protocol):
    async def merge_scenes(
        self,
        *,
        scenes: list[MergeSceneAsset],
        render_profile: RenderProfile,
    ) -> MergeVideoResult: ...


class FfmpegMergeClient:
    _scene_drift_min_sec = 0.1
    _scene_drift_ratio = 0.01
    _mux_safety_pad_sec = 0.15

    def __init__(self, settings: WorkerSettings) -> None:
        self._ffmpeg_binary = self._resolve_binary(settings.ffmpeg_binary)
        self._ffprobe_binary = self._resolve_binary(settings.ffprobe_binary)
        self._timeout_sec = settings.video_merge_timeout_sec
        self._max_drift_pct = settings.video_audio_sync_max_drift_pct
        self._crossfade_sec = settings.video_merge_crossfade_sec

    async def merge_scenes(
        self,
        *,
        scenes: list[MergeSceneAsset],
        render_profile: RenderProfile,
    ) -> MergeVideoResult:
        if not scenes:
            raise MergeExecutionError("Cannot merge a video with zero scenes")

        ordered_scenes = sorted(scenes, key=lambda scene: scene.scene_index)
        expected_duration_sec = (
            sum(scene.audio_duration_ms for scene in ordered_scenes) / 1000.0
        )
        stdout_logs: list[str] = [f"render_profile={render_profile}"]
        stderr_logs: list[str] = []

        try:
            with TemporaryDirectory(prefix="aqshara-merge-") as tmp_dir:
                workspace = Path(tmp_dir)
                clips_dir = workspace / "clips"
                clips_dir.mkdir(parents=True, exist_ok=True)

                clip_paths: list[Path] = []
                for scene in ordered_scenes:
                    raw_video = workspace / f"scene-{scene.scene_index:02d}.mp4"
                    raw_audio = workspace / f"scene-{scene.scene_index:02d}.wav"
                    merged_clip = clips_dir / f"scene-{scene.scene_index:02d}.mp4"
                    raw_video.write_bytes(scene.video_bytes)
                    raw_audio.write_bytes(scene.audio_bytes)
                    width, height = self._resolve_output_dimensions(render_profile)
                    expected_scene_duration_sec = scene.audio_duration_ms / 1000.0
                    raw_video_duration_sec = await self._probe_duration(raw_video)
                    stop_duration_sec = max(
                        expected_scene_duration_sec - raw_video_duration_sec,
                        0.0,
                    ) + self._mux_safety_pad_sec
                    stdout_logs.append(
                        "scene-duration:"
                        f" scene={scene.scene_index:02d}"
                        f" planned_audio={expected_scene_duration_sec:.3f}s"
                        f" raw_video={raw_video_duration_sec:.3f}s"
                        f" stop_pad={stop_duration_sec:.3f}s",
                    )

                    muxed = await self._run_command(
                        [
                            self._ffmpeg_binary,
                            "-y",
                            "-i",
                            str(raw_video),
                            "-i",
                            str(raw_audio),
                            "-filter_complex",
                            (
                                "[0:v]"
                                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                                "fps=30,"
                                f"tpad=stop_mode=clone:stop_duration={stop_duration_sec:.3f}[v]"
                            ),
                            "-map",
                            "[v]",
                            "-map",
                            "1:a",
                            "-af",
                            f"apad=pad_dur={self._mux_safety_pad_sec:.3f}",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            "-pix_fmt",
                            "yuv420p",
                            "-c:a",
                            "aac",
                            "-ar",
                            "48000",
                            "-b:a",
                            "192k",
                            "-t",
                            f"{expected_scene_duration_sec:.3f}",
                            str(merged_clip),
                        ],
                    )
                    stdout_logs.append(
                        f"scene-{scene.scene_index:02d}-mux-stdout:\n{muxed.stdout.strip()}",
                    )
                    if muxed.stderr.strip():
                        stderr_logs.append(
                            f"scene-{scene.scene_index:02d}-mux-stderr:\n{muxed.stderr.strip()}",
                        )
                    measured_scene_duration_sec = await self._probe_duration(merged_clip)
                    stdout_logs.append(
                        "scene-sync:"
                        f" scene={scene.scene_index:02d}"
                        f" expected={expected_scene_duration_sec:.3f}s"
                        f" corrected={measured_scene_duration_sec:.3f}s",
                    )
                    self._validate_scene_duration(
                        scene_index=scene.scene_index,
                        expected_duration_sec=expected_scene_duration_sec,
                        measured_duration_sec=measured_scene_duration_sec,
                        raw_video_duration_sec=raw_video_duration_sec,
                    )
                    clip_paths.append(merged_clip)

                final_path = workspace / "final.mp4"

                if self._crossfade_sec > 0 and len(clip_paths) > 1:
                    concatenated = await self._run_xfade_concat(
                        clip_paths, final_path, stdout_logs,
                    )
                else:
                    concatenated = await self._run_stream_copy_concat(
                        clip_paths, workspace, final_path,
                    )
                stdout_logs.append(f"concat-stdout:\n{concatenated.stdout.strip()}")
                if concatenated.stderr.strip():
                    stderr_logs.append(f"concat-stderr:\n{concatenated.stderr.strip()}")

                if not final_path.exists() or final_path.stat().st_size == 0:
                    raise MergeExecutionError("Merged output was empty")

                measured_duration_sec = await self._probe_duration(final_path)
                if expected_duration_sec > 0:
                    drift_pct = (
                        abs(measured_duration_sec - expected_duration_sec)
                        / expected_duration_sec
                    ) * 100
                    stdout_logs.append(
                        "audio-sync:"
                        f" expected={expected_duration_sec:.3f}s"
                        f" measured={measured_duration_sec:.3f}s"
                        f" drift_pct={drift_pct:.2f}",
                    )
                    if drift_pct > self._max_drift_pct:
                        raise AudioSyncValidationError(
                            "Merged video duration drift exceeded threshold: "
                            f"expected={expected_duration_sec:.3f}s "
                            f"measured={measured_duration_sec:.3f}s "
                            f"drift={drift_pct:.2f}% > {self._max_drift_pct:.2f}%",
                        )

                return MergeVideoResult(
                    video_bytes=final_path.read_bytes(),
                    stdout="\n\n".join(part for part in stdout_logs if part),
                    stderr="\n\n".join(part for part in stderr_logs if part),
                    duration_sec=measured_duration_sec,
                )
        except FileNotFoundError as error:
            raise MergeConfigurationError(
                f"Missing required media binary: {error.filename}",
            ) from error

    async def _run_stream_copy_concat(
        self,
        clip_paths: list[Path],
        workspace: Path,
        final_path: Path,
    ) -> CommandResult:
        concat_file = workspace / "concat.txt"
        concat_file.write_text(
            "".join(f"file '{clip_path.as_posix()}'\n" for clip_path in clip_paths),
            encoding="utf-8",
        )
        return await self._run_command(
            [
                self._ffmpeg_binary,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(final_path),
            ],
        )

    async def _run_xfade_concat(
        self,
        clip_paths: list[Path],
        final_path: Path,
        stdout_logs: list[str],
    ) -> CommandResult:
        durations: list[float] = []
        for clip in clip_paths:
            durations.append(await self._probe_duration(clip))

        fade_dur = self._crossfade_sec
        inputs: list[str] = []
        for clip in clip_paths:
            inputs.extend(["-i", str(clip)])

        n = len(clip_paths)
        # Build xfade filter chain for video
        filter_parts: list[str] = []
        # Fade-in on the very first clip
        filter_parts.append(
            f"[0:v]fade=t=in:d={min(fade_dur, 0.3):.3f}[v0_fi];"
        )
        prev_label = "v0_fi"
        offset = durations[0] - fade_dur
        for i in range(1, n):
            out_label = f"vx{i}"
            if i == n - 1:
                # Last pair: also apply fade-out
                filter_parts.append(
                    f"[{prev_label}][{i}:v]xfade=transition=fade:duration={fade_dur:.3f}:offset={offset:.3f},"
                    f"fade=t=out:d={min(fade_dur, 0.3):.3f}:st={offset + durations[i] - fade_dur - min(fade_dur, 0.3):.3f}[{out_label}];"
                )
            else:
                filter_parts.append(
                    f"[{prev_label}][{i}:v]xfade=transition=fade:duration={fade_dur:.3f}:offset={offset:.3f}[{out_label}];"
                )
            prev_label = out_label
            if i < n - 1:
                offset = offset + durations[i] - fade_dur

        # Build audio amerge with concat filter
        audio_inputs = "".join(f"[{i}:a]" for i in range(n))
        filter_parts.append(
            f"{audio_inputs}concat=n={n}:v=0:a=1[aout]"
        )

        filter_complex = "".join(filter_parts)
        stdout_logs.append(f"crossfade-filter: {filter_complex}")

        cmd = [
            self._ffmpeg_binary,
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{prev_label}]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-b:a",
            "192k",
            str(final_path),
        ]
        return await self._run_command(cmd)

    async def _probe_duration(self, output_path: Path) -> float:
        result = await self._run_command(
            [
                self._ffprobe_binary,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(output_path),
            ],
        )
        raw_value = result.stdout.strip()
        if not raw_value:
            raise MergeExecutionError("ffprobe returned an empty duration")

        try:
            return float(raw_value)
        except ValueError as error:
            raise MergeExecutionError(
                f"Unable to parse merged duration from ffprobe output: {raw_value}",
            ) from error

    @staticmethod
    def _resolve_output_dimensions(render_profile: RenderProfile) -> tuple[int, int]:
        if render_profile == "480p":
            return (854, 480)
        if render_profile == "1080p":
            return (1920, 1080)
        return (1280, 720)

    def _validate_scene_duration(
        self,
        *,
        scene_index: int,
        expected_duration_sec: float,
        measured_duration_sec: float,
        raw_video_duration_sec: float,
    ) -> None:
        tolerance_sec = max(
            self._scene_drift_min_sec,
            expected_duration_sec * self._scene_drift_ratio,
        )
        drift_sec = abs(measured_duration_sec - expected_duration_sec)
        if drift_sec <= tolerance_sec:
            return

        drift_pct = (
            0.0
            if expected_duration_sec <= 0
            else (drift_sec / expected_duration_sec) * 100
        )
        raise AudioSyncValidationError(
            "Scene clip duration drift exceeded tolerance: "
            f"scene={scene_index} "
            f"expected={expected_duration_sec:.3f}s "
            f"measured={measured_duration_sec:.3f}s "
            f"raw_video={raw_video_duration_sec:.3f}s "
            f"tolerance={tolerance_sec:.3f}s "
            f"drift={drift_pct:.2f}%",
        )

    @staticmethod
    def _resolve_binary(binary: str) -> str:
        binary_path = Path(binary)
        if binary_path.parent != Path("."):
            if binary_path.exists():
                return str(binary_path)
            raise MergeConfigurationError(
                "Missing required media binary: "
                f"{binary} (set a valid FFMPEG_BINARY/FFPROBE_BINARY path)",
            )

        resolved = shutil.which(binary)
        if resolved is not None:
            return resolved

        raise MergeConfigurationError(
            "Missing required media binary: "
            f"{binary} (install it or set FFMPEG_BINARY/FFPROBE_BINARY)",
        )

    async def _run_command(self, command: list[str]) -> CommandResult:
        process = await asyncio.create_subprocess_exec(
            *command,
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
            raise MergeExecutionError(
                f"Timed out running media command: {' '.join(command)}",
            ) from error

        decoded_stdout = stdout.decode("utf-8", errors="replace")
        decoded_stderr = stderr.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise MergeExecutionError(
                "Media command failed with exit code "
                f"{process.returncode}: {' '.join(command)}\n{decoded_stderr.strip()}",
            )

        return CommandResult(stdout=decoded_stdout, stderr=decoded_stderr)


def create_merge_client(settings: WorkerSettings) -> MergeClient:
    return FfmpegMergeClient(settings)

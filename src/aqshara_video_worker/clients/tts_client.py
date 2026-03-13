from __future__ import annotations

from io import BytesIO
import wave

from openai import AsyncOpenAI
from openai import APIError
from openai import APIStatusError
from openai import Omit

from aqshara_video_worker.config import WorkerSettings
from aqshara_video_worker.schemas import VideoLanguage


class TtsConfigurationError(RuntimeError):
    pass


class TtsGenerationError(RuntimeError):
    pass


class EmptyAudioError(TtsGenerationError):
    pass


class AudioDurationError(TtsGenerationError):
    pass


class OpenAITtsClient:
    def __init__(
        self,
        settings: WorkerSettings,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._api_key = settings.video_tts_openai_api_key
        self._model = settings.video_tts_openai_model
        self._client = client or AsyncOpenAI(
            api_key=settings.video_tts_openai_api_key,
            base_url=settings.video_tts_openai_base_url,
            timeout=settings.video_tts_openai_timeout_sec,
        )

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: VideoLanguage,
        voice_instruction: str | None = None,
    ) -> bytes:
        if not self._api_key:
            raise TtsConfigurationError(
                "VIDEO_TTS_OPENAI_API_KEY is required for TTS"
            )

        payload = {
            "model": self._model,
            "voice": voice,
            "input": text,
            "response_format": "pcm",
        }
        instructions = self._build_language_instruction(language, voice_instruction)
        payload["instructions"] = instructions if instructions else Omit()

        try:
            async with self._client.audio.speech.with_streaming_response.create(
                **payload,
            ) as response:
                audio_bytes = await response.read()
        except APIStatusError as error:
            detail = (getattr(error, "body", None) or {}).get("message", "")
            raise TtsGenerationError(
                detail or "OpenAI TTS request failed",
            ) from error
        except APIError as error:
            raise TtsGenerationError("OpenAI TTS request failed") from error

        if not audio_bytes:
            raise EmptyAudioError("OpenAI TTS returned an empty audio payload")

        return self._pcm_to_wav(audio_bytes)

    @staticmethod
    def measure_duration_ms(audio_bytes: bytes) -> int:
        try:
            with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
                frame_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
        except wave.Error as error:
            raise AudioDurationError("Unable to decode WAV audio payload") from error

        if frame_rate <= 0:
            raise AudioDurationError("WAV audio payload has invalid frame rate")

        duration_ms = int(round((frame_count / frame_rate) * 1000))
        if duration_ms <= 0:
            raise AudioDurationError("WAV audio payload has zero duration")

        return duration_ms

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _pcm_to_wav(audio_bytes: bytes) -> bytes:
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24_000)
            wav_file.writeframes(audio_bytes)
        return buffer.getvalue()

    @staticmethod
    def _build_language_instruction(
        language: VideoLanguage,
        voice_instruction: str | None = None,
    ) -> str:
        if language == "id":
            base = "Read the narration naturally in Indonesian."
        else:
            base = "Read the narration naturally in English."
        if voice_instruction:
            return f"{base} {voice_instruction}".strip()
        return base

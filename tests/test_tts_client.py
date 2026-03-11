from __future__ import annotations

from io import BytesIO
import wave

import httpx
from openai import APIError
import pytest

from aqshara_video_worker.clients.tts_client import (
    EmptyAudioError,
    OpenAITtsClient,
    TtsGenerationError,
)
from aqshara_video_worker.config import WorkerSettings


def build_wav_bytes(duration_ms: int) -> bytes:
    frame_rate = 24_000
    frame_count = int(frame_rate * (duration_ms / 1000))
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


def build_pcm_bytes(duration_ms: int) -> bytes:
    frame_rate = 24_000
    frame_count = int(frame_rate * (duration_ms / 1000))
    return b"\x00\x00" * frame_count


class FakeStreamingResponse:
    def __init__(self, audio_bytes: bytes) -> None:
        self._audio_bytes = audio_bytes

    async def __aenter__(self) -> "FakeStreamingResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def read(self) -> bytes:
        return self._audio_bytes


class FakeStreamingSpeech:
    def __init__(self, audio_bytes: bytes, error: Exception | None = None) -> None:
        self._audio_bytes = audio_bytes
        self._error = error
        self.calls: list[dict[str, object]] = []
        self.with_streaming_response = self

    def create(self, **kwargs: object) -> FakeStreamingResponse:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return FakeStreamingResponse(self._audio_bytes)


class FakeAudioClient:
    def __init__(self, speech: FakeStreamingSpeech) -> None:
        self.speech = speech


class FakeOpenAIClient:
    def __init__(self, speech: FakeStreamingSpeech) -> None:
        self.audio = FakeAudioClient(speech)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_tts_client_sends_wav_request_and_returns_audio() -> None:
    pcm_bytes = build_pcm_bytes(750)
    speech = FakeStreamingSpeech(pcm_bytes)
    client = OpenAITtsClient(
        WorkerSettings(
            OPENAI_API_KEY="openai-key",
            R2_ENDPOINT="https://example.r2.cloudflarestorage.com",
            R2_ACCESS_KEY_ID="key",
            R2_SECRET_ACCESS_KEY="secret",
            R2_BUCKET="bucket",
        ),
        client=FakeOpenAIClient(speech),  # type: ignore[arg-type]
    )

    try:
        result = await client.generate_speech(
            text="Narration text",
            voice="alloy",
            language="en",
        )
    finally:
        await client.close()

    assert speech.calls == [
        {
            "model": "gpt-4o-mini-tts",
            "voice": "alloy",
            "input": "Narration text",
            "response_format": "pcm",
            "instructions": "Read the narration naturally in English.",
        },
    ]
    assert result == build_wav_bytes(750)
    assert OpenAITtsClient.measure_duration_ms(result) == 750


@pytest.mark.asyncio
async def test_tts_client_rejects_empty_audio_payload() -> None:
    client = OpenAITtsClient(
        WorkerSettings(
            OPENAI_API_KEY="openai-key",
            R2_ENDPOINT="https://example.r2.cloudflarestorage.com",
            R2_ACCESS_KEY_ID="key",
            R2_SECRET_ACCESS_KEY="secret",
            R2_BUCKET="bucket",
        ),
        client=FakeOpenAIClient(FakeStreamingSpeech(b"")),  # type: ignore[arg-type]
    )

    try:
        with pytest.raises(EmptyAudioError):
            await client.generate_speech(
                text="Narration text",
                voice="alloy",
                language="id",
            )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tts_client_maps_http_errors_to_generation_error() -> None:
    client = OpenAITtsClient(
        WorkerSettings(
            OPENAI_API_KEY="openai-key",
            R2_ENDPOINT="https://example.r2.cloudflarestorage.com",
            R2_ACCESS_KEY_ID="key",
            R2_SECRET_ACCESS_KEY="secret",
            R2_BUCKET="bucket",
        ),
        client=FakeOpenAIClient(
            FakeStreamingSpeech(
                b"",
                error=APIError(
                    "provider error",
                    request=httpx.Request("POST", "https://api.openai.com/v1/audio/speech"),
                    body=None,
                ),
            ),
        ),  # type: ignore[arg-type]
    )

    try:
        with pytest.raises(TtsGenerationError):
            await client.generate_speech(
                text="Narration text",
                voice="alloy",
                language="en",
            )
    finally:
        await client.close()

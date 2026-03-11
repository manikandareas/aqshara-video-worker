from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    callback_base_url: str = Field(
        default="http://127.0.0.1:8000/api/v1",
        alias="VIDEO_WORKER_CALLBACK_BASE_URL",
    )
    internal_service_token: str = Field(
        default="local-video-internal-token",
        alias="VIDEO_INTERNAL_SERVICE_TOKEN",
    )
    r2_endpoint: str = Field(alias="R2_ENDPOINT")
    r2_region: str = Field(default="auto", alias="R2_REGION")
    r2_access_key_id: str = Field(alias="R2_ACCESS_KEY_ID")
    r2_secret_access_key: str = Field(alias="R2_SECRET_ACCESS_KEY")
    r2_bucket: str = Field(alias="R2_BUCKET")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
    )
    openai_tts_model: str = Field(
        default="gpt-4o-mini-tts",
        alias="OPENAI_TTS_MODEL",
    )
    openai_timeout_sec: float = Field(default=60, alias="OPENAI_TIMEOUT_SEC")
    video_render_backend: str = Field(default="mock", alias="VIDEO_RENDER_BACKEND")
    video_render_profile: str = Field(default="720p", alias="VIDEO_RENDER_PROFILE")
    video_render_timeout_sec: int = Field(default=180, alias="VIDEO_RENDER_TIMEOUT_SEC")
    video_merge_timeout_sec: int = Field(default=120, alias="VIDEO_MERGE_TIMEOUT_SEC")
    video_audio_sync_max_drift_pct: float = Field(
        default=15.0,
        alias="VIDEO_AUDIO_SYNC_MAX_DRIFT_PCT",
    )
    ffmpeg_binary: str = Field(default="ffmpeg", alias="FFMPEG_BINARY")
    ffprobe_binary: str = Field(default="ffprobe", alias="FFPROBE_BINARY")
    daytona_api_url: str | None = Field(default=None, alias="DAYTONA_API_URL")
    daytona_api_key: str | None = Field(default=None, alias="DAYTONA_API_KEY")
    daytona_target: str | None = Field(default=None, alias="DAYTONA_TARGET")
    daytona_python_version: str = Field(default="3.12", alias="DAYTONA_PYTHON_VERSION")
    daytona_render_image: str | None = Field(
        default=None,
        alias="DAYTONA_RENDER_IMAGE",
    )
    daytona_create_timeout_sec: int = Field(
        default=300,
        alias="DAYTONA_CREATE_TIMEOUT_SEC",
    )
    daytona_render_cpu: int = Field(default=2, alias="DAYTONA_RENDER_CPU")
    daytona_render_memory_gb: int = Field(
        default=4,
        alias="DAYTONA_RENDER_MEMORY_GB",
    )
    daytona_render_disk_gb: int = Field(default=8, alias="DAYTONA_RENDER_DISK_GB")

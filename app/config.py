from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "StackAI RAG Backend"
    app_version: str = "0.1.0"

    data_dir: Path = Path("data")
    upload_dir: Path = Path("data/uploads")
    record_dir: Path = Path("data/records")

    max_upload_size_mb: int = 25
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 180
    min_chunk_chars: int = 250

    ocr_fallback_enabled: bool = True
    mistral_api_key: str | None = None
    mistral_ocr_model: str = "mistral-ocr-latest"


@lru_cache
def get_settings() -> Settings:
    return Settings()

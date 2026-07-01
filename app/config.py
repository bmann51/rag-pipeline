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

    query_top_k: int = 5
    query_min_chars: int = 3
    query_max_chars: int = 500
    intent_gate_enabled: bool = True
    query_rewrite_enabled: bool = True
    keyword_weight: float = 0.4
    semantic_weight: float = 0.6
    relevance_threshold: float = 0.3
    min_query_term_coverage: float = 0.5
    semantic_search_enabled: bool = True
    semantic_candidate_k: int = 20
    semantic_max_new_chunk_embeddings_per_query: int = 24
    mistral_embedding_model: str = "mistral-embed"
    embedding_batch_size: int = 16
    source_consistency_bonus_enabled: bool = True
    source_consistency_bonus: float = 0.05
    source_consistency_max_query_terms: int = 4
    source_consistency_top_window: int = 8
    source_consistency_min_share: float = 0.6
    embedding_min_request_interval_seconds: float = 1.05
    embedding_max_retries_on_rate_limit: int = 3
    embedding_retry_base_delay_seconds: float = 1.0

    generation_enabled: bool = False
    mistral_chat_model: str = "mistral-small-latest"
    generation_temperature: float = 0.2
    generation_max_tokens: int = 600
    generation_min_top_relevance_score: float = 0.45
    generation_min_avg_relevance_score: float = 0.4
    generation_max_chunks: int = 4
    generation_max_chars_per_chunk: int = 1600

    ocr_fallback_enabled: bool = True
    mistral_api_key: str | None = None
    mistral_ocr_model: str = "mistral-ocr-latest"


@lru_cache
def get_settings() -> Settings:
    return Settings()

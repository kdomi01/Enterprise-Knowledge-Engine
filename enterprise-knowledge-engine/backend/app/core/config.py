from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent.parent
ENV_FILE_PATH = ROOT_DIR / ".env"

class Settings(BaseSettings):
    # App Settings
    ENVIRONMENT: Literal["development", "staging", "production", "testing"] = "development"
    PROJECT_NAME: str = "Enterprise Knowledge Engine"
    LOG_LEVEL: Literal["debug", "info", "warning", "error"] = "info"

    # LLM Providers & Tools
    OPENAI_API_KEY: str
    TAVILY_API_KEY: Optional[str] = None

    # Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None

    # Observability
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "http://localhost:3000"

    # Pass the absolute Path object directly to Pydantic
    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8", 
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
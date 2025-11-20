"""
Environment settings loader for the FastAPI application.
"""
from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable settings container built from environment variables."""
    openai_api_key: str | None
    gemini_api_key: str | None
    enable_db_ping: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings built from environment variables."""
    return Settings(
        openai_api_key=os.getenv("MY_OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        enable_db_ping=os.getenv("ENABLE_DB_PING", "false").lower() == "true",
    )


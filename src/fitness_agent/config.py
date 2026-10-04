import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))
    top_k: int = int(os.getenv("TOP_K", "4"))

    pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    training_csv: Path = PROJECT_ROOT / "data" / "guardrail_training.csv"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    collection_name: str = "fitness_docs"


settings = Settings()

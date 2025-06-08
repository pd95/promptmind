import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import argparse

@dataclass
class Settings:
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "phi4-mini"
    ollama_url: str = "http://localhost:11434"
    chunk_size: int = 600
    overlap: int = 100

    @classmethod
    def from_env_and_args(cls):
        # Load .env if present
        load_dotenv(override=False)

        # Start with environment variables
        settings = cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            overlap=int(os.getenv("OVERLAP", cls.overlap)),
        )

        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--embedding-model", type=str, help="Embedding model to use")
        parser.add_argument("--llm-model", type=str, help="LLM model to use")
        parser.add_argument("--ollama-url", type=str, help="Ollama URL")
        parser.add_argument("--chunk-size", type=int, help="Chunk size for indexing")
        parser.add_argument("--overlap", type=int, help="Chunk overlap for indexing")
        args, unknown_args = parser.parse_known_args()

        # Override with CLI args if provided
        if args.embedding_model:
            settings.embedding_model = args.embedding_model
        if args.llm_model:
            settings.llm_model = args.llm_model
        if args.ollama_url:
            settings.ollama_url = args.ollama_url
        if args.chunk_size:
            settings.chunk_size = args.chunk_size
        if args.overlap:
            settings.overlap = args.overlap

        return settings, unknown_args
"""KGK AI configuration module.

All model/provider/runtime settings are loaded from environment variables
via Pydantic Settings. Never hard-code secrets or model-specific values.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KGKSettings(BaseSettings):
    """Central configuration for KGK AI.

    All values can be overridden via environment variables or a .env file.
    Defaults are chosen for free/low-cost Hugging Face deployment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Model Configuration ---
    model_name: str = Field(
        default="Qwen/Qwen3-4B-Instruct-2507",
        description="Hugging Face model ID for the runtime LLM.",
    )
    model_name_fallback: str = Field(
        default="Qwen/Qwen3-0.6B",
        description="Fallback model for CPU-only / low-resource environments.",
    )

    # --- Generation Parameters ---
    max_new_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)

    # --- Quantization ---
    quantization: str = Field(
        default="4bit",
        description="Options: none, 4bit, 8bit.",
    )

    # --- Embedding Model ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    embedding_dimension: int = Field(default=384, ge=1)

    # --- RAG Configuration ---
    enable_rag: bool = Field(default=True)
    vector_store: str = Field(default="faiss")
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=50, ge=0)
    top_k_retrieval: int = Field(default=5, ge=1)
    knowledge_dir: str = Field(default="knowledge/documents")
    vectorstore_dir: str = Field(default="knowledge/vectorstore")

    # --- Memory Configuration ---
    enable_memory: bool = Field(default=True)
    memory_dir: str = Field(default="data/conversations")
    long_term_memory_dir: str = Field(default="data/memory")
    context_window_tokens: int = Field(default=4096, ge=512, description="Maximum tokens for conversation context window.")
    reserved_response_tokens: int = Field(default=1024, ge=128, description="Tokens reserved for model response generation.")
    enable_summarization: bool = Field(default=True, description="Summarize old messages when context window is exceeded.")
    max_summary_chars: int = Field(default=500, ge=100, description="Maximum characters for conversation summaries.")

    # --- Tools Configuration ---
    enable_tools: bool = Field(default=True)
    enable_python_tool: bool = Field(default=True)
    python_tool_timeout: int = Field(default=5, ge=1, le=30)

    # --- Agents Configuration ---
    enable_agents: bool = Field(default=False)

    # --- API Configuration ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=7860, ge=1, le=65535)

    # --- UI Configuration ---
    ui_title: str = Field(default="KGK AI")
    ui_subtitle: str = Field(default="AI Intelligence by KGK")

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")

    # --- Hugging Face ---
    hf_token: Optional[str] = Field(default=None, description="Hugging Face API token for private models.")

    # --- Security ---
    max_input_length: int = Field(default=10000, ge=100)
    max_concurrent_requests: int = Field(default=10, ge=1)

    # --- Runtime Detection ---
    @property
    def is_hf_space(self) -> bool:
        """Detect if running inside a Hugging Face Space."""
        import os

        return os.environ.get("SPACE_ID") is not None

    @property
    def is_zero_gpu(self) -> bool:
        """Detect if ZeroGPU hardware is available."""
        import os

        return os.environ.get("ACCELERATOR") != "none" and self.is_hf_space

    @property
    def device(self) -> str:
        """Determine the best available compute device."""
        if self.is_zero_gpu:
            return "cuda"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"


@lru_cache
def get_settings() -> KGKSettings:
    """Return cached settings instance.

    Use this everywhere in the codebase to access configuration:
        from app.config import get_settings
        settings = get_settings()
    """
    return KGKSettings()

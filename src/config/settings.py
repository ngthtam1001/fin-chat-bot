from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
MODEL_CONFIG_PATH = BASE_DIR / "model_config.yml"
QDRANT_CONFIG_PATH = BASE_DIR / "qdrant_config.yml"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ModelSizeGroup:
    default: str
    small: str | None = None
    large: str | None = None


@dataclass
class EmbeddingModelConfig:
    ollama: ModelSizeGroup
    gemini: ModelSizeGroup
    e5: ModelSizeGroup | None = None
    bge: ModelSizeGroup | None = None


@dataclass
class GeminiLLMConfig:
    flash: str


@dataclass
class LLMModelConfig:
    gemini: GeminiLLMConfig


@dataclass
class MetadataMappingConfig:
    path: str


@dataclass
class ModelConfig:
    doc_patterns: dict[str, str]
    metadata_mapping: MetadataMappingConfig
    embedding_model: EmbeddingModelConfig
    llm_model: LLMModelConfig
    api_key: dict[str, str | None]


@dataclass
class QdrantConfig:
    url: str
    test_collection: str
    default_collection: str


@dataclass
class EmbedDimensionConfig:
    gemini: int
    ollama: int


@dataclass
class AppSettings:
    model: ModelConfig
    qdrant: QdrantConfig
    embed_dimension: EmbedDimensionConfig
    metadata_mapping_path: str = r"C:\Users\admin\Desktop\gic\data\mapping\financebench_document_information.jsonl"

    @property
    def gemini_api_key(self) -> str | None:
        return self.model.api_key.get("gemini")

    @property
    def tavily_api_key(self) -> str | None:
        return self.model.api_key.get("tavily")


def parse_model_group(raw: dict[str, Any], default_fallback: str = "") -> ModelSizeGroup:
    return ModelSizeGroup(
        default=raw.get("default", default_fallback),
        small=raw.get("small"),
        large=raw.get("large"),
    )


def load_settings() -> AppSettings:
    model_raw = load_yaml(MODEL_CONFIG_PATH)
    qdrant_raw = load_yaml(QDRANT_CONFIG_PATH)

    embedding_raw = model_raw.get("embedding_model", {})
    llm_raw = model_raw.get("llm_model", {})
    qdrant_section = qdrant_raw.get("qdrant", {})
    dimension_section = qdrant_raw.get("embed_dimension", {})

    model_config = ModelConfig(
        doc_patterns=model_raw.get("doc_patterns", {}),
        metadata_mapping=MetadataMappingConfig(
            path=model_raw.get("metadata_mapping", {}).get("path", "")
        ),
        embedding_model=EmbeddingModelConfig(
            ollama=parse_model_group(
                embedding_raw.get("ollama", {}),
                default_fallback="nomic-embed-text",
            ),
            gemini=parse_model_group(
                embedding_raw.get("gemini", {}),
                default_fallback="gemini-embedding-001",
            ),

        ),
        llm_model=LLMModelConfig(
            gemini=GeminiLLMConfig(
                flash=llm_raw.get("gemini", {}).get("flash", "gemini-2.5-flash-lite")
            )
        ),
        api_key=model_raw.get("api_key", {}),
    )

    qdrant_config = QdrantConfig(
        url=qdrant_section.get("url", "http://localhost:6333"),
        test_collection=qdrant_section.get("test_collection", "test_collection"),
        default_collection=qdrant_section.get("default_collection", "default_collection"),
    )

    embed_dimension = EmbedDimensionConfig(
        gemini=int(dimension_section.get("gemini", 3072)),
        ollama=int(dimension_section.get("ollama", 768)),
    )

    return AppSettings(
        model=model_config,
        qdrant=qdrant_config,
        embed_dimension=embed_dimension,
    )


settings = load_settings()
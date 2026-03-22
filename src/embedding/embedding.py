import re
from abc import ABC, abstractmethod
from typing import List, Optional

import requests

from src.config.settings import settings


# HELPER FUNCTION
def clean_text(text: str, max_chars: int = 1500) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# BASE CLASS
class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# OLLAMA
class OllamaEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/embed"
        self.session = requests.Session()
        self.timeout = timeout

    def embed(self, text: str) -> List[float]:
        last_exc: Exception | None = None
        for _ in range(2):
            try:
                resp = self.session.post(
                    self.url,
                    json={"model": self.model, "input": text},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()["embeddings"][0]
            except Exception as e:
                last_exc = e
        raise last_exc  # type: ignore[misc]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        last_exc: Exception | None = None
        for _ in range(2):
            try:
                resp = self.session.post(
                    self.url,
                    json={"model": self.model, "input": texts},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()["embeddings"]
            except Exception as e:
                last_exc = e
        raise last_exc  # type: ignore[misc]


# GEMINI
class GeminiEmbedder(BaseEmbedder):
    def __init__(self, model: str, api_key: str, timeout: int = 60) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, text: str) -> List[float]:
        if not self.api_key:
            raise RuntimeError("Gemini API key is missing (settings.gemini_api_key).")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        payload = {"content": {"parts": [{"text": text}]}}
        resp = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]["values"]


# EMBEDDER FACTORY (settings-based)
class EmbedderFactory:
    @staticmethod
    def create(provider: str) -> BaseEmbedder:
        provider = provider.lower().strip()

        if provider == "ollama":
            # from dataclass settings: settings.model.embedding_model.ollama.default
            model = settings.model.embedding_model.ollama.default
            # optionally support base_url if you add it; otherwise default
            base_url = getattr(settings, "ollama_base_url", "http://localhost:11434")
            timeout = int(getattr(settings, "ollama_timeout", 180))
            return OllamaEmbedder(model=model, base_url=base_url, timeout=timeout)

        if provider == "gemini":
            model = settings.model.embedding_model.gemini.default
            api_key = (settings.model.api_key or {}).get("gemini")
            if not api_key:
                raise ValueError("Missing Gemini API key: settings.model.api_key['gemini']")
            return GeminiEmbedder(model=model, api_key=api_key)

        raise ValueError(f"Unknown provider: {provider!r}")
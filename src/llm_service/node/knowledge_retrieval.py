from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any

from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.llm_service.node.metadata_extraction import QueryMetadata
from src.embedding.embedding import EmbedderFactory


class FinancialRAG:
    def __init__(
        self,
        gemini_api_key: str,
        qdrant_url: str,
        collection_name: str,
        embed_model: str,
        llm_model: str,
        embed_provider: str = "gemini",
    ) -> None:
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.embed_provider = embed_provider.lower().strip()

        self.genai_client = genai.Client(api_key=gemini_api_key)
        self.qdrant_client = QdrantClient(url=qdrant_url, timeout=30.0)

        # TODO: Now we are using only OLLAMA model, need re-factor to make it ENUM
        if self.embed_provider == "ollama":
            self._embedder = EmbedderFactory.create("ollama")
        else:
            self._embedder = None

    def embed_query(self, query: str) -> List[float]:
        if self.embed_provider == "ollama":
            return self._embedder.embed(query) 

        result = self.genai_client.models.embed_content(
            model=self.embed_model,
            contents=query,
        )
        return result.embeddings[0].values

    def build_filter(
        self,
        metadata: Optional[Dict[str, Any] | QueryMetadata] = None,
    ) -> Optional[Filter]:
        if metadata is None:
            return None

        if isinstance(metadata, QueryMetadata):
            metadata = metadata.model_dump()

        conditions: List[FieldCondition] = []

        company_name = metadata.get("company_name")
        year = metadata.get("year")
        report_type = metadata.get("report_type")

        if company_name is not None:
            conditions.append(
                FieldCondition(
                    key="company_name",
                    match=MatchValue(value=company_name),
                )
            )

        if year is not None:
            conditions.append(
                FieldCondition(
                    key="time",
                    match=MatchValue(value=year),
                )
            )

        if report_type is not None:
            conditions.append(
                FieldCondition(
                    key="report_type",
                    match=MatchValue(value=report_type),
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    def _extract_chunks_from_results(self, results) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for point in results.points:
            payload = point.payload or {}
            text = payload.get("text")

            if text:
                chunks.append(
                    {
                        "text": text,
                        "score": getattr(point, "score", None),
                        "company_name": payload.get("company_name"),
                        "year": payload.get("time"),
                        "report_type": payload.get("report_type"),
                        "page": payload.get("page"),
                        "source": payload.get("source"),
                    }
                )

        return chunks

    def _join_context(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
    
        sources: List[Dict[str, Any]] = []

        for chunk in chunks:
            url = chunk.get("source")
            page = chunk.get("page")
        
        sources.append({
            "url": url,
            "page": page
        })

        text = "\n\n---\n\n".join(
            chunk["text"] for chunk in chunks if chunk.get("text")
        )

        return text, sources

    def retrieve_context(
        self,
        query: str,
        metadata: Optional[Dict[str, Any] | QueryMetadata] = None,
        top_k: int = 20,
    ) -> Optional[List[Dict[str, Any]]]:
        query_vector = self.embed_query(query)
        q_filter = self.build_filter(metadata)

    
        if q_filter is None:
            return None

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=q_filter,
            limit=top_k,
        )

        chunks = self._extract_chunks_from_results(results)
        if not chunks:
            return None

        return chunks

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        if not sources:
            return "Sources: None"

        lines = []
        for i, src in enumerate(sources, 1):
            url = src.get("url", "Unknown source")
            page = src.get("page")

            if page is not None:
                lines.append(f"{i}. {url} (page {page})")
            else:
                lines.append(f"{i}. {url}")

        return "Sources:\n" + "\n".join(lines)
    

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        if not chunks:
            return "No relevant information found."

        context_text, sources = self._join_context(chunks)

        prompt = f"""
    You are a financial document QA assistant.

    Answer the user's question using ONLY the provided context.

    Rules:
    - Do NOT use external knowledge.
    - Do NOT guess.
    - If the answer is not available in the context, say clearly that it is not found in the provided documents.
    - Keep the answer concise, accurate, and well-structured.

    [QUESTION]
    {query}

    [CONTEXT]
    {context_text}
    """

        response = self.genai_client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
        )

        answer_text = (getattr(response, "text", "") or "").strip()
        sources_text = self._format_sources(sources)

        return f"{answer_text}\n\n{sources_text}"


    def answer_with_rag(
        self,
        query: str,
        metadata: Optional[Dict[str, Any] | QueryMetadata] = None,
        top_k: int = 20,
    ) -> str:
        chunks = self.retrieve_context(
            query=query,
            metadata=metadata,
            top_k=top_k,
        )

        if chunks is None:
            return "No documents match the specified metadata."

        if len(chunks) == 0:
            return "No relevant information found in Qdrant."

        return self.generate_answer(query, chunks)


if __name__ == "__main__":

    from src.config.settings import settings

    GEMINI_API_KEY = settings.gemini_api_key or ""
    QDRANT_URL = settings.qdrant.url
    COLLECTION_NAME = settings.qdrant.default_collection

    EMBED_MODEL = settings.model.embedding_model.gemini.default
    LLM_MODEL = settings.model.llm_model.gemini.flash

    rag = FinancialRAG(
        gemini_api_key=GEMINI_API_KEY,
        qdrant_url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embed_model=EMBED_MODEL,
        llm_model=LLM_MODEL,
        embed_provider="ollama",
    )

    query = "What was Apple's revenue in 2015 10-K?"
    metadata = QueryMetadata(
        company_name="Apple",
        year=2015,
        report_type="10k",
    )

    answer = rag.answer_with_rag(
        query=query,
        metadata=metadata,
        top_k=5,
    )

    print(answer)
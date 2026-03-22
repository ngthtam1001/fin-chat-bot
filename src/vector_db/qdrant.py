from __future__ import annotations

from typing import List
from uuid import uuid4
from tqdm import tqdm
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config.settings import settings
from src.embedding.embedding import EmbedderFactory, clean_text  
from src.ocr.ocr import SimpleOCR


_qurl = urlparse(settings.qdrant.url)
QDRANT_HOST: str = _qurl.hostname or "localhost"
QDRANT_PORT: int = int(_qurl.port or 6333)

COLLECTION_NAME: str = settings.qdrant.test_collection
BATCH_SIZE: int = int(getattr(settings, "qdrant_batch_size", 16)) 
EMBED_PROVIDER: str = getattr(settings, "embed_provider", "ollama")

client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(embedder) -> int:
    dim = len(embedder.embed("dimension check"))

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    return dim


def ingest_pdf(pdf_path: str) -> int:

    ocr = SimpleOCR(pdf_path=pdf_path, chunk_size=800)
    chunks = ocr.process() 

    embedder = EmbedderFactory.create(EMBED_PROVIDER)
    dim = ensure_collection(embedder)
    print(f"Collection '{COLLECTION_NAME}' ready (dim={dim}, provider={EMBED_PROVIDER})")

    total_points = 0

    for i in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc="Ingesting batches",
        unit="batch",
    ):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        texts = [clean_text(item["text"]) for item in batch_chunks]

        try:
            embeddings = embedder.embed_batch(texts)

            points: List[PointStruct] = []
            for item, emb, clean_txt in zip(batch_chunks, embeddings, texts):
                meta = item.get("metadata", {}) or {}

                payload = {
                    "text": item["text"],
                    "clean_text": clean_txt,
                    "page": meta.get("page"),
                    "source": pdf_path,
                    "company_name": meta.get("company_name"),
                    "time": meta.get("time"),
                    "report_type": meta.get("report_type"),
                }

                points.append(PointStruct(id=uuid4().hex, vector=emb, payload=payload))

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_points += len(points)

        except Exception as e:
            print(f"Error at batch {i // BATCH_SIZE}: {e}")

    print(f"\nTotal upserted: {total_points}")
    return total_points

# def ingest_multiple_pdf(pdf_list: list[str]):


if __name__ == "__main__":
    path = r"C:\Users\admin\Desktop\gic\data\pdfs\ACTIVISIONBLIZZARD_2015_10K.pdf"
    ingest_pdf(path)
from __future__ import annotations

from pathlib import Path
from typing import List
from uuid import uuid4
from tqdm import tqdm
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config.settings import settings
from src.embedding.embedding import EmbedderFactory, clean_text
from src.ocr.ocr import SimpleOCR


_qurl = urlparse(settings.qdrant.url)
QDRANT_HOST: str = _qurl.hostname or "localhost"
QDRANT_PORT: int = int(_qurl.port or 6333)

COLLECTION_NAME: str = settings.qdrant.default_collection
BATCH_SIZE: int = int(getattr(settings, "qdrant_batch_size", 16))
EMBED_PROVIDER: str = getattr(settings, "embed_provider", "ollama")

client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(embedder, recreate: bool = False) -> int:
    """
    Ensure Qdrant collection exists with correct vector dimension.
    recreate=True  -> drop and recreate collection
    recreate=False -> create only if not exists
    """
    dim = len(embedder.embed("dimension check"))

    existing = {c.name for c in client.get_collections().collections}

    if recreate:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        if COLLECTION_NAME not in existing:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    return dim


def get_pdf_paths(pdf_dir: str | Path) -> list[str]:
    """
    Recursively get all PDF paths in a directory.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")

    pdf_paths = sorted(str(p) for p in pdf_dir.rglob("*.pdf"))
    return pdf_paths


def ingest_chunks(
    chunks: list[dict],
    pdf_path: str,
    embedder,
) -> int:
    """
    Ingest already-processed chunks into Qdrant.
    """
    total_points = 0

    for i in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc=f"Ingesting {Path(pdf_path).name}",
        unit="batch",
    ):
        batch_chunks = chunks[i:i + BATCH_SIZE]
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

                points.append(
                    PointStruct(
                        id=uuid4().hex,
                        vector=emb,
                        payload=payload,
                    )
                )

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            total_points += len(points)

        except Exception as e:
            print(f"[ERROR] {Path(pdf_path).name} - batch {i // BATCH_SIZE}: {e}")

    return total_points


def ingest_pdf(pdf_path: str, embedder=None) -> int:
    """
    Ingest one PDF into Qdrant.
    NOTE: does NOT recreate collection.
    """
    if embedder is None:
        embedder = EmbedderFactory.create(EMBED_PROVIDER)

    ocr = SimpleOCR(pdf_path=pdf_path, chunk_size=800)
    chunks = ocr.process()

    total_points = ingest_chunks(chunks=chunks, pdf_path=pdf_path, embedder=embedder)

    print(f"[DONE] {Path(pdf_path).name}: upserted {total_points} chunks")
    return total_points


def ingest_multiple_pdf(
    pdf_dir: str | Path | None = None,
    pdf_list: list[str] | None = None,
    recreate_collection: bool = False,
    parallel: bool = False,
    max_workers: int = 4,
) -> int:
    """
    Ingest multiple PDFs from either:
    - a folder (pdf_dir), or
    - an explicit list of paths (pdf_list)

    Args:
        pdf_dir: directory containing pdf files
        pdf_list: list of pdf file paths
        recreate_collection: whether to reset collection before ingest
        parallel: run file-level ingestion in parallel
        max_workers: number of threads for parallel mode
    """
    if pdf_list is None:
        if pdf_dir is None:
            raise ValueError("Provide either pdf_dir or pdf_list.")
        pdf_list = get_pdf_paths(pdf_dir)

    if not pdf_list:
        print("No PDF files found.")
        return 0

    print(f"Found {len(pdf_list)} PDF files.")

    # Create embedder once
    embedder = EmbedderFactory.create(EMBED_PROVIDER)

    # Ensure collection once only
    dim = ensure_collection(embedder, recreate=recreate_collection)
    print(f"Collection '{COLLECTION_NAME}' ready (dim={dim}, provider={EMBED_PROVIDER})")

    total_all = 0

    if not parallel:
        for pdf_path in tqdm(pdf_list, desc="Processing PDFs", unit="file"):
            try:
                total_all += ingest_pdf(pdf_path, embedder=embedder)
            except Exception as e:
                print(f"[ERROR] Failed to ingest {pdf_path}: {e}")
    else:
        # Parallel by file
        # Good for OCR / I/O heavy workloads, but too many workers may overload embedding server
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(ingest_pdf, pdf_path, EmbedderFactory.create(EMBED_PROVIDER)): pdf_path
                for pdf_path in pdf_list
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel ingest", unit="file"):
                pdf_path = futures[future]
                try:
                    total_all += future.result()
                except Exception as e:
                    print(f"[ERROR] Failed to ingest {pdf_path}: {e}")

    print(f"\n[ALL DONE] Total upserted across all PDFs: {total_all}")
    return total_all


if __name__ == "__main__":
    pdf_dir = r"C:\Users\admin\Desktop\gic\data\pdfs"

    ingest_multiple_pdf(
        pdf_dir=pdf_dir,
        recreate_collection=True,   # True if you want reset from scratch
        parallel=True,             # Start with False first
        max_workers=4,
    )
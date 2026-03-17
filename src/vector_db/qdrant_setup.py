from typing import List
from uuid import uuid4
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.model.embedding import embed_ollama_batch, clean_text
from src.model.ocr import ocr_langchain


client = QdrantClient("localhost", port=6333)
collection_name = "test"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=768,   
        distance=Distance.COSINE,
    ),
)

BATCH_SIZE = 16


def ingest_pdf(pdf_path: str) -> None:

    chunks = ocr_langchain(pdf_path)
    total_points = 0

    # 🔥 tqdm progress bar
    for i in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc="🚀 Ingesting batches",
        unit="batch"
    ):

        batch_chunks = chunks[i:i+BATCH_SIZE]

        # CLEAN
        texts = [clean_text(doc.page_content) for doc in batch_chunks]

        try:
            # EMBEDDING
            embeddings = embed_ollama_batch(texts)

            points: List[PointStruct] = []

            for doc, emb, clean_txt in zip(batch_chunks, embeddings, texts):
                payload = {
                    "text": doc.page_content,
                    "clean_text": clean_txt,
                    "page": doc.metadata.get("page"),
                    "source": pdf_path,
                    "company_name": doc.metadata.get("company_name"),
                    "time": doc.metadata.get("time"),
                    "report_type": doc.metadata.get("report_type"),
                }

                points.append(
                    PointStruct(
                        id=uuid4().hex,
                        vector=emb,
                        payload=payload,
                    )
                )

            # UPSERT
            client.upsert(
                collection_name=collection_name,
                points=points,
            )

            total_points += len(points)

        except Exception as e:
            print(f"❌ Error at batch {i//BATCH_SIZE}: {e}")

    print(f"\n✅ Total upserted: {total_points}")

def query(user_input,
          topk,
          )

if __name__ == "__main__":
    path = r"data/pdfs/3M_2015_10K.pdf"
    ingest_pdf(path)
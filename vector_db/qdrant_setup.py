from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.model.embedding import embed_ollama
from src.model.ocr import ocr_langchain


client = QdrantClient("localhost", port=6333)
collection_name = "pdf_docs"

client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=1024,   
        distance=Distance.COSINE
    ),
)

def ingest_pdf(pdf_path):

    docs = ocr_langchain(pdf_path)

    points = []

    for i, doc in enumerate(docs):

        text = doc.page_content
        embedding = embed_ollama(text)

        payload = {
            "text": text,
            "page": doc.metadata["page"],
            "source": pdf_path
        }

        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload=payload
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points
    )

ingest_pdf("test.pdf")
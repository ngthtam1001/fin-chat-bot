import requests
from pathlib import Path
import yaml
from tqdm import tqdm  
import time
from src.model.ocr import ocr_langchain


CONFIG_PATH = Path(__file__).resolve().parent / "model_config.yml"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

OLLAMA_EMBED_MODEL = (
    config.get("embedding_model", {})
          .get("ollama", {})
          .get("nomic", "nomic-embed-text")
)

GEMINI_EMBED_MODEL = (
    config.get("embedding_model", {})
          .get("gemini", {})
          .get("default", "gemini-embedding-001")
)


GEMINI_API_KEY = config.get("api_key", {}).get("gemini")


def embed_ollama(text: str):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": OLLAMA_EMBED_MODEL,
            "input": text,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]

def embed_ollama_batch(
    texts: list[str],
    _session=requests.Session()   
):
    response = _session.post(
        "http://localhost:11434/api/embed",
        json={
            "model": OLLAMA_EMBED_MODEL,
            "input": texts,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def embed_gemini(text: str) -> list[float]:

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent"

    payload = {
        "model": GEMINI_EMBED_MODEL,
        "content": {
            "parts": [
                {"text": text}
            ]
        }
    }

    params = {"key": GEMINI_API_KEY}
    resp = requests.post(url, json=payload, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    return data["embedding"]["values"]


import re

def clean_text(text: str, max_chars: int = 1500) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text[:max_chars]

# if __name__ == "__main__":

#     BATCH_SIZE = 16
#     pdf_path = r"C:\Users\admin\Desktop\gic\data\pdfs\3M_2015_10K.pdf"

#     # ⏱️ Start total timer
#     total_start = time.time()

#     # 1. OCR + chunk
#     ocr_start = time.time()
#     chunks = ocr_langchain(pdf_path)
#     ocr_time = time.time() - ocr_start

#     print(f"📄 Total chunks: {len(chunks)}")
#     print(f"⏱️ OCR time: {ocr_time:.2f}s")

#     # 2. Embedding (BATCH)
#     embedded_data = []
#     embed_start = time.time()

#     for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
#         batch_chunks = chunks[i:i+BATCH_SIZE]
#         texts = [clean_text(chunk.page_content) for chunk in batch_chunks]

#         batch_start = time.time() 

#         try:
#             embeddings = embed_ollama_batch(texts)

#             for chunk, emb in zip(batch_chunks, embeddings):
#                 embedded_data.append({
#                     "text": chunk.page_content,
#                     "embedding": emb,
#                     "metadata": chunk.metadata,
#                 })

#         except Exception as e:
#             print(f"❌ Error at batch {i}: {e}")

#         batch_time = time.time() - batch_start
#         print(f"⚡ Batch {i//BATCH_SIZE}: {batch_time:.2f}s")

#     embed_time = time.time() - embed_start

#     print(f"\n✅ Done embedding {len(embedded_data)} chunks")
#     print(f"⏱️ Total embedding time: {embed_time:.2f}s")

#     # ⏱️ Total pipeline time
#     total_time = time.time() - total_start
#     print(f"🚀 Total pipeline time: {total_time:.2f}s")

#     # 3. Sample output
#     if embedded_data:
#         print("\n🔎 Sample embedded chunk:")
#         print("Text:", embedded_data[0]["text"][:200])
#         print("Metadata:", embedded_data[0]["metadata"])
#         print("Embedding dim:", len(embedded_data[0]["embedding"]))
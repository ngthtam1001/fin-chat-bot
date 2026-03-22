from pathlib import Path
import re
import json
from typing import List, Dict, Any

import fitz

from src.config.settings import settings


def metadata_extraction(pdf_path: str) -> dict[str, Any]:
    mapping_path = Path(settings.metadata_mapping_path)
    metadata_index: dict[str, dict[str, Any]] = {}

    with mapping_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            key = record["doc_name"].upper()
            metadata_index[key] = record

    file_name = Path(pdf_path).stem.upper()
    metadata = metadata_index.get(file_name)

    if metadata is None:
        for key, value in metadata_index.items():
            if key in file_name:
                metadata = value
                break

    if metadata is None:
        company_match = re.match(r"([A-Z0-9]+)", file_name)
        year_match = re.search(r"(19|20)\d{2}", file_name)
        doc_type_match = re.search(r"(10K|10Q|8K)", file_name, flags=re.IGNORECASE)

        metadata = {
            "doc_name": file_name,
            "company": company_match.group(1) if company_match else None,
            "gics_sector": None,
            "doc_type": doc_type_match.group(1).lower() if doc_type_match else None,
            "doc_period": int(year_match.group()) if year_match else None,
            "doc_link": None,
        }

    qdrant_payload = {
        "company": metadata.get("company"),
        "year": metadata.get("doc_period"),
        "doc_type": metadata.get("doc_type"),
        "sector": metadata.get("gics_sector"),
    }

    return {
        "raw_metadata": metadata,
        "qdrant_payload": qdrant_payload,
    }


def normalize_chunk_metadata(raw_metadata: dict[str, Any], page: int) -> dict[str, Any]:

    # FinanceBench mapping JSONL uses keys like: company, doc_period, doc_type
    company_name = raw_metadata.get("company")
    time_value = raw_metadata.get("doc_period")
    report_type = raw_metadata.get("doc_type")

    return {
        "page": page,
        "company_name": company_name,
        "time": time_value,
        "report_type": report_type,

        # keep originals too (handy for debugging / future filtering)
        "raw": raw_metadata,
    }


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


class SimpleOCR:
    def __init__(self, pdf_path: str, chunk_size: int = 800, overlap: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_pdf(self) -> List[Dict[str, Any]]:
        doc = fitz.open(self.pdf_path)
        pages: List[Dict[str, Any]] = []

        for page_num, page in enumerate(doc):
            text = page.get_text()

            if text.strip():
                pages.append({
                    "text": text,
                    "page": page_num,
                })

        doc.close()
        return pages

    def process(self) -> List[Dict[str, Any]]:
        pages = self.load_pdf()
        metadata = metadata_extraction(self.pdf_path)

        results: List[Dict[str, Any]] = []

        for page in pages:
            chunks = chunk_text(
                text=page["text"],
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

            for chunk in chunks:
                results.append({
                    "text": chunk,
                    "page": page["page"],
                    "metadata": normalize_chunk_metadata(
                        raw_metadata=metadata["raw_metadata"],
                        page=page["page"],
                    ),
                    "qdrant_payload": metadata["qdrant_payload"],
                })

        return results


if __name__ == "__main__":
    ocr = SimpleOCR(
        pdf_path=r"C:\Users\admin\Desktop\gic\data\pdfs\3M_2015_10K.pdf",
        chunk_size=800,
        overlap=200,
    )

    chunks = ocr.process()

    print(f"Total chunks: {len(chunks)}")
    print("\n--- SAMPLE TEXT ---\n")
    print(chunks[0]["text"][:300])
    print("\n--- METADATA ---\n")
    print(chunks[0]["metadata"])
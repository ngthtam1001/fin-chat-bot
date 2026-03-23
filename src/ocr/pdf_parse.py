from pathlib import Path
import re
import json
from typing import List, Dict, Any

import pdfplumber

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


def normalize_chunk_metadata(
    raw_metadata: dict[str, Any],
    page: int,
    chunk_id: int,
    source_type: str = "pdfplumber",
) -> dict[str, Any]:
    company_name = raw_metadata.get("company")
    time_value = raw_metadata.get("doc_period")
    report_type = raw_metadata.get("doc_type")

    return {
        "page": page,
        "chunk_id": chunk_id,
        "company_name": company_name,
        "time": time_value,
        "report_type": report_type,
        "source_type": source_type,
        "raw": raw_metadata,
    }


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def clean_table(table: list[list[Any]]) -> list[list[Any]]:
    cleaned = []
    for row in table:
        if row is None:
            continue
        cleaned_row = [
            cell.strip() if isinstance(cell, str) else cell
            for cell in row
        ]
        if any(cell not in (None, "") for cell in cleaned_row):
            cleaned.append(cleaned_row)
    return cleaned


def table_to_text(table: list[list[Any]], table_index: int) -> str:
    """
    Convert a raw table into a text block for embedding.
    """
    cleaned = clean_table(table)
    if not cleaned:
        return ""

    lines = [f"[TABLE {table_index}]"]

    for row in cleaned:
        row_text = " | ".join("" if cell is None else str(cell) for cell in row)
        lines.append(row_text)

    return "\n".join(lines)


class PDFPlumberParser:
    def __init__(self, pdf_path: str, chunk_size: int = 800, overlap: int = 200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_pdf(self) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                raw_tables = page.extract_tables() or []

                table_texts = []
                for i, table in enumerate(raw_tables, start=1):
                    table_text = table_to_text(table, i)
                    if table_text.strip():
                        table_texts.append(table_text)

                combined_parts = []

                if text:
                    combined_parts.append("[PAGE_TEXT]\n" + text)

                if table_texts:
                    combined_parts.append("\n\n".join(table_texts))

                combined_text = "\n\n".join(combined_parts).strip()

                if combined_text:
                    pages.append({
                        "page": page_num,
                        "text": text,
                        "table_texts": table_texts,
                        "combined_text": combined_text,
                    })

        return pages

    def process(self) -> List[Dict[str, Any]]:
        pages = self.load_pdf()
        metadata = metadata_extraction(self.pdf_path)

        results: List[Dict[str, Any]] = []

        for page in pages:
            chunks = chunk_text(
                text=page["combined_text"],
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

            for chunk_id, chunk in enumerate(chunks):
                results.append({
                    "text": chunk,
                    "page": page["page"],
                    "metadata": normalize_chunk_metadata(
                        raw_metadata=metadata["raw_metadata"],
                        page=page["page"],
                        chunk_id=chunk_id,
                        source_type="pdfplumber",
                    ),
                    "qdrant_payload": {
                        **metadata["qdrant_payload"],
                        "page": page["page"],
                        "chunk_id": chunk_id,
                    },
                })

        return results


if __name__ == "__main__":
    parser = PDFPlumberParser(
        pdf_path=r"C:\Users\admin\Desktop\gic\data\pdfs\3M_2015_10K.pdf",
        chunk_size=1200,
        overlap=200,
    )

    chunks = parser.process()

    print(f"Total chunks: {len(chunks)}")
    print("\n--- SAMPLE TEXT ---\n")
    print(chunks[0]["text"][:1000])
    print("\n--- METADATA ---\n")
    print(json.dumps(chunks[0]["metadata"], indent=2, ensure_ascii=False))
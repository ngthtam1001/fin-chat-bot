from pathlib import Path
import re
import yaml 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


CONFIG_PATH = Path(__file__).resolve().parents[1] / "model_config.yml"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
DOC_PATTERNS: dict[str, str] = config.get("doc_patterns", {})



def metadata_extraction(
    pdf_path: str
) -> dict[str | None]:
    
    # Company name extraction
    file_name = Path(pdf_path).stem
    company_name = file_name.split("_")[0]

    # Date extraction
    year_match = re.search(r"(19|20)\d{2}", file_name)
    year = int(year_match.group()) if year_match else None

    # Report type extraction
    report_type: str | None = None
    for doc_type, pattern in DOC_PATTERNS.items():
        if re.search(pattern, file_name, flags=re.IGNORECASE):
            report_type = doc_type
            break

    return {
        "company_name": company_name,
        "time": year,
        "report_type": report_type,
    }


def ocr_langchain(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 300,
) -> tuple[list, dict]:
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(docs)

    return chunks, metadata_extraction(pdf_path=pdf_path)


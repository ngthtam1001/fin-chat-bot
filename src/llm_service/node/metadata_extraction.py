import requests
import json
from pydantic import BaseModel, Field
from typing import Optional

from src.config.settings import settings


GEMINI_LLM_MODEL = settings.model.llm_model.gemini.flash
GEMINI_API_KEY = settings.model.api_key.get("gemini")


PROMPT_TEMPLATE = """
You are an information extraction system.

Your task is to extract structured metadata from the user query.

Return ONLY a valid JSON object with the following fields:
- "company_name": string or null
- "year": integer or null
- "report_type": one of ["10k", "10q", "earnings", "annual_report"] or null

Rules:
1. Normalize company names:
   - Convert abbreviations to full official names when possible
     (e.g., "J&J" → "Johnson & Johnson", "3M" stays "3M").
2. If a field is not mentioned, return null.
3. Extract only explicitly stated information — do NOT infer or guess.
4. "report_type" must strictly be one of:
   - "10k" (annual SEC filing)
   - "10q" (quarterly SEC filing)
   - "earnings" (earnings call/report)
   - "annual_report" (non-SEC annual report)
5. Output must be valid JSON:
   - No explanations
   - No extra text
   - No comments

User query:
"{query}"

JSON:
"""


class QueryMetadata(BaseModel):
    company_name: Optional[str] = Field(
        default=None,
        description="Company name mentioned in the query. Normalize abbreviations to the full official company name when possible (e.g., 'J&J' → 'Johnson & Johnson', while '3M' remains '3M')."
    )
    year: Optional[int] = Field(
        default=None,
        description="Year mentioned in query (e.g., 2015)"
    )
    report_type: Optional[str] = Field(
        default=None,
        description="Type of report: 10k, 10q, earnings, annual_report"
    )


def extract_metadata_gemini(query: str) -> QueryMetadata:

    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini API key is missing (settings.model.api_key['gemini']).")

    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_LLM_MODEL}:generateContent"
    prompt = PROMPT_TEMPLATE.format(query=query)

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    params = {"key": GEMINI_API_KEY}

    resp = requests.post(url, json=payload, params=params, timeout=30)
    resp.raise_for_status()

    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    json_str = text[json_start:json_end]

    data = json.loads(json_str)
    return QueryMetadata(**data)



 # TEST

if __name__ == "__main__":

    queries = [
        "Show me the revenue of 3M in 2015 10K",
        "Apple 2022 annual report net income",
        "What did Tesla report in earnings 2021?",
        "Give me Microsoft 10Q 2020 financials",
        "Tell me about Amazon performance"
    ]

    for q in queries:
        print("\n" + "="*60)
        print(f"🔍 Query: {q}")

        try:
            meta = extract_metadata_gemini(q)

            print("✅ Parsed Metadata:")
            print(meta)

            print("\n📌 Fields:")
            print("company_name:", meta.company_name)
            print("year:", meta.year)
            print("report_type:", meta.report_type)

        except Exception as e:
            print(f"❌ Error: {e}")
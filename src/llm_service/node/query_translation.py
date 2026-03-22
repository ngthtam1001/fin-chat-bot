import re
from typing import List

import requests
from pydantic import BaseModel, Field

from src.config.settings import settings


GEMINI_LLM_MODEL = settings.model.llm_model.gemini.flash
GEMINI_API_KEY = settings.model.api_key.get("gemini")


PROMPT_TEMPLATE = """
You are a reasoning planner for financial document question answering.

Your task is to analyze the user's question and produce the minimum necessary intermediate questions or reasoning steps needed to answer it correctly.

This is NOT simple query splitting.
Your job is to identify whether the question requires:
- direct retrieval,
- multi-step reasoning,
- multi-hop reasoning,
- comparison,
- calculation,
- conditional evaluation,
- or explanation of whether a metric is appropriate.

Core principles:
1. If the original question can be answered directly from a document, return the original question only.
2. If the original question requires reasoning across multiple facts, break it into the minimum necessary intermediate questions.
3. Every intermediate question must be necessary for constructing the final answer.
4. The intermediate questions must collectively form a reasoning path toward the original question.
5. Do NOT generate background or optional questions.
6. Do NOT introduce new assumptions, concepts, or metrics not present in the original question.
7. If the question contains a defined formula, ratio, or condition, preserve it exactly.
8. If the question asks whether a metric is useful, include the necessary intermediate step(s) to evaluate that, not just retrieve the metric.
9. If the question involves any derived quantity (e.g., average, change, growth, ratio, margin, or any value defined by a formula), you MUST decompose it into its underlying components.
    - Determine whether each required value can be directly retrieved from financial statements.
    - If a value is NOT directly retrievable, break it down into the minimum set of base variables needed to compute it.
    - Continue decomposing recursively until all variables correspond to directly retrievable financial line items (e.g., revenue, PP&E, liabilities, etc.).

When decomposition / planning is needed, think in terms of reasoning steps such as:
- retrieve a required fact
- retrieve another required fact
- compare facts
- compute a metric from retrieved values
- identify the driver of a change
- evaluate whether a metric is meaningful for this company/context
- combine all intermediate findings into the final answer

Output requirements:
- Output ONLY the intermediate questions
- One question per line
- No numbering
- No bullet points
- No explanations
- No JSON
- No labels
- No extra text before or after

Examples:

User query:
"What was 3M's revenue in 2022?"
Output:
What was 3M's revenue in 2022?

User query:
"What is the difference between the revenue of Apple and Microsoft in 2025?"
Output:
What was Apple's revenue in 2025?
What was Microsoft's revenue in 2025?

User query:
"What is the operating cash flow ratio for Adobe in FY2015? The ratio is defined as cash from operations divided by total current liabilities."
Output:
What was Adobe's cash from operations in FY2015?
What was Adobe's total current liabilities in FY2015?

User query:
"What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why."
Output:
What was 3M's operating margin in FY2022?
What factors drove the change in 3M's operating margin in FY2022?
Is operating margin a meaningful metric for evaluating 3M in this context?
Why might operating margin be less useful or require caution when evaluating a company like 3M?

Now process the following query:

"{query}"
"""


class QueryTranslation(BaseModel):
    list_questions: List[str] = Field(default_factory=list)


def _call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini API key is missing.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_LLM_MODEL}:generateContent"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "topP": 1,
            "topK": 1,
        },
    }

    resp = requests.post(
        url,
        json=payload,
        params={"key": GEMINI_API_KEY},
        timeout=30,
    )

    print("status_code =", resp.status_code)
    print("response_text =", resp.text)

    if not resp.ok:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _parse_questions(raw_text: str, original_query: str) -> List[str]:
    text = raw_text.strip()

    if not text:
        return [original_query]

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line).strip()

        if line:
            lines.append(line)

    if not lines:
        return [original_query]

    return lines


def translate_query_gemini(query: str) -> QueryTranslation:
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    prompt = PROMPT_TEMPLATE.format(query=query)
    raw_text = _call_gemini(prompt)

    questions = _parse_questions(raw_text, query)

    cleaned_questions = []
    for q in questions:
        q = q.strip()
        if q and q not in cleaned_questions:
            cleaned_questions.append(q)

    if not cleaned_questions:
        cleaned_questions = [query]

    return QueryTranslation(list_questions=cleaned_questions)


def main():
    test_queries = [
        "What was 3M's revenue in 2022?",
        "What is the difference between the revenue of Apple and Microsoft in 2025?",
        "What is the operating cash flow ratio for Adobe in FY2015?",
        "What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric, explain why.",
        "Summarize Apple's business performance in 2022.",
    ]

    for i, query in enumerate(test_queries, 1):
        print("=" * 80)
        print(f"Test Case {i}")
        print(f"Query: {query}\n")

        try:
            result = translate_query_gemini(query)

            print("Decomposed Questions:")
            for j, q in enumerate(result.list_questions, 1):
                print(f"  {j}. {q}")

            print("\nNum questions:", len(result.list_questions))
            if len(result.list_questions) == 1:
                print("👉 Likely NO decomposition")
            else:
                print("👉 Decomposition applied")

        except Exception as e:
            print("❌ ERROR:", str(e))

        print()


if __name__ == "__main__":
    main()
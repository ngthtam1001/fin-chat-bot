from __future__ import annotations

from typing import Any, Dict, List

from google import genai

from src.config.settings import settings
from src.llm_service.pipeline.graph import FinancialAgent
from src.llm_service.node.query_translation import translate_query_gemini


class MultiStepFinancialAgent:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("Missing Gemini API key in settings.gemini_api_key")

        self.base_agent = FinancialAgent()
        self.genai_client = genai.Client(api_key=settings.gemini_api_key)
        self.llm_model = settings.model.llm_model.gemini.flash

    def _format_intermediate_results(
        self,
        sub_questions: List[str],
        step_results: List[Dict[str, Any]],
    ) -> str:
        blocks: List[str] = []

        for i, (sub_q, result) in enumerate(zip(sub_questions, step_results), 1):
            answer = (result.get("answer") or "").strip()
            source = result.get("source", "")
            route = result.get("route", "")
            error = result.get("error", "")

            if not answer:
                answer = "No answer generated."

            block = f"""
[STEP {i}]
Sub-question: {sub_q}
Route used: {route}
Source: {source}
Answer:
{answer}
"""
            if error:
                block += f"\nError: {error}\n"

            blocks.append(block.strip())

        return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(blocks)

    def _generate_final_answer(
        self,
        original_query: str,
        sub_questions: List[str],
        step_results: List[Dict[str, Any]],
    ) -> str:
        intermediate_context = self._format_intermediate_results(
            sub_questions=sub_questions,
            step_results=step_results,
        )

        prompt = f"""
You are a financial QA synthesis assistant.

Your task is to answer the ORIGINAL user question by reasoning over the provided intermediate sub-questions and their answers.

IMPORTANT PRINCIPLES:
- The ORIGINAL QUESTION is the only thing you must answer.
- Sub-questions and intermediate answers are ONLY supporting evidence.
- Do NOT summarize all intermediate answers blindly.
- Select ONLY the relevant pieces of information that help answer the original question.

RULES:
1. Use ONLY the information from the intermediate answers.
2. Do NOT introduce external knowledge or assumptions.
3. IGNORE any intermediate answers that:
   - say "not found", "cannot find", "not available", or similar
   - do not contribute useful information
4. You MUST perform reasoning:
   - combine multiple pieces of evidence if needed
   - compare values if relevant
   - infer relationships ONLY when clearly supported by the given answers
5. Stay strictly grounded in the provided information.
6. If the available information is insufficient to fully answer the question, explicitly say:
   "The answer cannot be fully determined from the provided information."
7. Keep the answer concise, structured, and directly focused on the original question.
8. Preserve numerical accuracy. Perform calculations ONLY if all required numbers are available.

OUTPUT FORMAT:
- Final Answer: <direct answer to the original question>
- Key Evidence:
  - <relevant supporting point 1>
  - <relevant supporting point 2>
  - ...

[ORIGINAL QUESTION]
{original_query}

[SUB-QUESTIONS]
{chr(10).join(f"- {q}" for q in sub_questions)}

[INTERMEDIATE ANSWERS]
{intermediate_context}
"""

        response = self.genai_client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
        )

        return (getattr(response, "text", "") or "").strip()

    def invoke(self, query: str) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        # 1) decompose first
        translation = translate_query_gemini(query)
        sub_questions = translation.list_questions or [query]

        # 2) run base financial agent on each decomposed question
        step_results: List[Dict[str, Any]] = []
        for sub_q in sub_questions:
            result = self.base_agent.invoke(sub_q)
            step_results.append(result)

        # 3) synthesize final answer from all intermediate answers
        final_answer = self._generate_final_answer(
            original_query=query,
            sub_questions=sub_questions,
            step_results=step_results,
        )

        # Keep a top-level shape similar to FinancialAgent.invoke()
        return {
            "query": query,
            "route": "reasoning_graph",
            "source": "multi_step",
            "answer": final_answer,
            "error": None,
            # Extra debug/inspection fields
            "original_query": query,
            "translated_questions": sub_questions,
            "step_results": step_results,
        }

    def answer(self, query: str) -> str:
        result = self.invoke(query)
        return result.get("answer", "No final answer generated.")
    


def main():
    agent = MultiStepFinancialAgent()

    queries = [
        "What was 3M's revenue in 2022?",
        "What is the difference between the revenue of Apple and Microsoft in 2025?",
        "What is the operating cash flow ratio for Adobe in FY2015?",
        "What drove operating margin change as of FY2022 for 3M? If operating margin is not a useful metric for a company like this, then please state that and explain why.",
    ]

    for q in queries:
        print("\n" + "=" * 100)
        print("ORIGINAL QUERY:", q)

        try:
            result = agent.invoke(q)

            print("\nTRANSLATED QUESTIONS:")
            for i, sub_q in enumerate(result["translated_questions"], 1):
                print(f"  {i}. {sub_q}")

            print("\nSTEP RESULTS:")
            for i, step in enumerate(result["step_results"], 1):
                print(f"\n--- STEP {i} ---")
                print("route :", step.get("route"))
                print("source:", step.get("source"))
                print("answer:", step.get("answer"))
                print("error :", step.get("error"))

            print("\nFINAL ANSWER:")
            print(result["final_answer"])

        except Exception as e:
            print("ERROR:", str(e))


if __name__ == "__main__":
    main()
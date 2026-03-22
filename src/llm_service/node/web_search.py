from tavily import TavilyClient

from src.config.settings import settings
from google import genai


class TavilySearcher:
    def __init__(self, api_key: str | None = None) -> None:
        self.client = TavilyClient(api_key=api_key or settings.tavily_api_key)

        if not settings.gemini_api_key:
            raise RuntimeError("Missing Gemini API key in settings (settings.gemini_api_key).")

        self.genai_client = genai.Client(api_key=settings.gemini_api_key)
        self.llm_model = settings.model.llm_model.gemini.flash

    def search(
        self,
        query: str,
        top_k: int = 5,
        topic: str = "general",
        search_depth: str = "advanced",
        include_raw_content: bool = False,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[dict]:
        resp = self.client.search(
            query=query,
            max_results=top_k,
            topic=topic,
            search_depth=search_depth,
            include_raw_content=include_raw_content,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=False,
        )

        return resp.get("results", [])
    
    def _format_sources(self, results: list[dict]) -> str:
        seen_urls = set()
        source_lines = []

        for i, res in enumerate(results):
            title = (res.get("title") or "Untitled source").strip()
            url = (res.get("url") or "").strip()

            seen_urls.add(url)
            source_lines.append(f"{len(source_lines)+1}. {title}\n   {url}")

        if not source_lines:
            return "Sources: None"

        return "Sources:\n" + "\n".join(source_lines)

    def answer_with_tavily(
        self,
        query: str,
        top_k: int = 20,
        topic: str = "finance",
        search_depth: str = "advanced",
    ) -> str:
        
        results = self.search(
            query=query,
            top_k=top_k,
            topic=topic,
            search_depth=search_depth,
            include_raw_content=False,
        )


        if not results:
            return (
                "I could not find relevant web results for this query.\n\n"
                "Note: No answer was generated from internal data."
            )


        context_parts: list[str] = []
        valid_results_for_sources: list[dict] = []

        for i, res in enumerate(results, 1):
            
            title = (res.get("title") or "").strip()
            url = (res.get("url") or "").strip()
            content = (res.get("content") or "").strip()
            
            if not content:
                continue

            context_parts.append(
                f"[{i}] Title: {title}\nURL: {url}\nContent: {content}"
            )
            valid_results_for_sources.append(res)

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
You are a financial document question-answering assistant.

Your task is to answer the user's question using ONLY the provided context.

Guidelines:
1. Grounding:
   - Base your answer strictly on the given context.
   - Do NOT use external knowledge or assumptions.
   - If the information is not explicitly stated or cannot be logically inferred from the context, say:
     "The answer is not found in the provided documents."

2. Reasoning:
   - You are allowed to perform logical reasoning, synthesis, and simple calculations based on the provided context.
   - Combine information from multiple parts of the context if necessary.
   - If making an inference, ensure it is directly supported by the context (no speculation).

3. Accuracy:
   - Do NOT hallucinate facts, numbers, or conclusions.
   - If there is ambiguity, explain it clearly instead of guessing.

4. Structure:
   - Provide a clear and concise answer.
   - When helpful, break the answer into bullet points or steps.
   - If relevant, quote or reference specific parts of the context to justify your answer.

5. Numerical / Financial Analysis:
   - Perform calculations if required (e.g., growth rates, differences, ratios), but only using values present in the context.
   - Clearly show the reasoning steps if calculations are involved.

6. Edge Cases:
   - If the context provides partial information, answer only what can be supported.
   - If multiple interpretations exist, mention them and explain which is most supported.

Output Style:
- Clear, structured, and professional.
- No unnecessary explanation beyond what is needed to answer the question.

[QUESTION]
{query}

[CONTEXT]
{context}
"""
        response = self.genai_client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
        )

        answer_text = (getattr(response, "text", "") or "").strip()
        sources_text = self._format_sources(valid_results_for_sources)

        final_answer = (
            f"{answer_text}\n\n"
            f"{sources_text}\n\n"
            "Note: This response is based on web search results, not our internal database. "
            "Please verify the information before relying on it fully."
        )

        return final_answer

    

if __name__ == "__main__":
    searcher = TavilySearcher()

    q = "Johnson & Johnson 2015 10-K litigation risk"
    print(searcher.answer_with_tavily(q))
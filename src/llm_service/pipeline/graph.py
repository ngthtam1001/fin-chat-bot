from __future__ import annotations

from typing import TypedDict, Optional, Any

from langgraph.graph import StateGraph, START, END

from src.config.settings import settings
from src.llm_service.node.metadata_extraction import (
    QueryMetadata,
    extract_metadata_gemini,
)
from src.llm_service.node.knowledge_retrieval import FinancialRAG
from src.llm_service.node.web_search import TavilySearcher


# STATE
class GraphState(TypedDict, total=False):
    query: str
    metadata: Optional[QueryMetadata]
    route: str
    answer: str
    source: str
    error: Optional[str]
    no_match: bool


# HELPERS
def has_metadata(metadata: Optional[QueryMetadata]) -> bool:
    if metadata is None:
        return False

    if metadata.company_name is None:
        return False

    return any([
        metadata.year is not None,
        metadata.report_type is not None,
    ])


# SERVICE FACTORIES
def build_rag() -> FinancialRAG:
    gemini_api_key = settings.gemini_api_key
    if not gemini_api_key:
        raise RuntimeError("Missing Gemini API key in settings (settings.gemini_api_key).")

    qdrant_url = settings.qdrant.url
    collection_name = settings.qdrant.default_collection
    embed_model = settings.model.embedding_model.gemini.default
    llm_model = settings.model.llm_model.gemini.flash

    return FinancialRAG(
        gemini_api_key=gemini_api_key,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        embed_model=embed_model,
        llm_model=llm_model,
        embed_provider="ollama",
    )


def build_web_searcher() -> TavilySearcher:
    return TavilySearcher(api_key=settings.tavily_api_key)


# NODES
def metadata_extraction_node(state: GraphState) -> GraphState:
    query = state["query"]

    try:
        metadata = extract_metadata_gemini(query)
        return {
            **state,
            "metadata": metadata,
        }
    except Exception as e:
        return {
            **state,
            "metadata": None,
            "error": f"Metadata extraction failed: {str(e)}",
        }


def router_node(state: GraphState) -> GraphState:
    metadata = state.get("metadata")

    if has_metadata(metadata):
        return {
            **state,
            "route": "knowledge_retrieval",
        }

    return {
        **state,
        "route": "web_search",
    }


def knowledge_retrieval_node(state: GraphState) -> GraphState:
    query = state["query"]
    metadata = state.get("metadata")

    try:
        rag = build_rag()

        chunks = rag.retrieve_context(
            query=query,
            metadata=metadata,
            top_k=5,
        )

        if chunks is None or len(chunks) == 0:
            return {
                **state,
                "no_match": True,
                "answer": "",
                "source": "qdrant_rag",
                "error": None,
            }

        answer = rag.generate_answer(query=query, chunks=chunks)

        if answer is None or not str(answer).strip():
            return {
                **state,
                "no_match": True,
                "answer": "",
                "source": "qdrant_rag",
                "error": None,
            }

        return {
            **state,
            "no_match": False,
            "answer": answer,
            "source": "qdrant_rag",
            "error": None,
        }

    except Exception as e:
        return {
            **state,
            "no_match": True,
            "answer": "",
            "source": "qdrant_rag",
            "error": f"Knowledge retrieval failed: {str(e)}",
        }

def web_search_node(state: GraphState) -> GraphState:
    query = state["query"]

    try:
        searcher = build_web_searcher()
        answer = searcher.answer_with_tavily(
            query=query,
            top_k=5,
            topic="finance",
            search_depth="advanced",
        )

        return {
            **state,
            "answer": answer,
            "source": "tavily_web_search",
        }
    except Exception as e:
        return {
            **state,
            "answer": f"Web search failed: {str(e)}",
            "source": "tavily_web_search",
            "error": str(e),
        }


# CONDITIONAL EDGE
def route_after_metadata(state: GraphState) -> str:
    return state["route"]


def route_after_knowledge_retrieval(state: GraphState) -> str:
    if state.get("no_match"):
        return "web_search"

    if state.get("error"):
        return "web_search"

    if not (state.get("answer") or "").strip():
        return "web_search"

    return "end"

# BUILD GRAPH
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("metadata_extraction", metadata_extraction_node)
    graph.add_node("router", router_node)
    graph.add_node("knowledge_retrieval", knowledge_retrieval_node)
    graph.add_node("web_search", web_search_node)

    graph.add_edge(START, "metadata_extraction")
    graph.add_edge("metadata_extraction", "router")

    graph.add_conditional_edges(
        "router",
        route_after_metadata,
        {
            "knowledge_retrieval": "knowledge_retrieval",
            "web_search": "web_search",
        },
    )

    graph.add_conditional_edges(
        "knowledge_retrieval",
        route_after_knowledge_retrieval,
        {
            "web_search": "web_search",
            "end": END,
        },
    )
    graph.add_edge("web_search", END)

    return graph.compile()


#AGENT
class FinancialAgent:
    def __init__(self) -> None:
        self.graph = build_graph()

    def invoke(self, query: str) -> dict[str, Any]:
        result = self.graph.invoke({"query": query})
        return result

    def answer(self, query: str) -> str:
        result = self.invoke(query)
        return result.get("answer", "No answer generated.")


# TEST
if __name__ == "__main__":
    agent = FinancialAgent()

    queries = [
        # "What is the GDP of USA in 2023 ?"
        "Show me the revenue of 3M in 2015",
        # "Apple 2022 annual report net income",
        # "What did Tesla report in earnings 2021?",
        # "Give me Microsoft 10Q 2020 financials",
        # "Tell me about Amazon performance",
        # "What is inflation today?",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print("QUERY:", q)

        result = agent.invoke(q)

        print("METADATA:", result.get("metadata"))
        print("ROUTE:", result.get("route"))
        print("SOURCE:", result.get("source"))
        print("ANSWER:", result.get("answer"))
        print("ERROR:", result.get("error"))
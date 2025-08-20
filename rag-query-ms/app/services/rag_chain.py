import uuid
from loguru import logger
from langgraph.graph import StateGraph, END

from app.services.graph_definition import (
    GraphState,
    retrieve_documents,
    generate_answer,
    check_relevance,
    rewrite_answer,
)

def build_rag_graph():
    """Builds and compiles the LangGraph workflow for handling RAG queries."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("rewrite", rewrite_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges(
        "generate",
        check_relevance,
        {
            "useful": END,
            "not_useful": "rewrite",
        },
    )
    workflow.add_edge("rewrite", END)

    logger.info("LangGraph workflow compiled successfully.")
    return workflow.compile()

app_graph = build_rag_graph()

async def get_rag_response(question: str) -> dict:
    """
    Runs the RAG graph for a given question and returns the final answer along with its sources.
    This is the sole function that should be exposed to the rest of the application.
    """
    request_id = str(uuid.uuid4())
    inputs = {"question": question, "request_id": request_id}
    try:
        final_state = await app_graph.ainvoke(inputs)

        answer = final_state.get("generation", "Sorry, an error occurred while processing your request.")
        documents = final_state.get("documents", [])

        response = {
            "answer": answer,
            "sources": [
                {
                    "filename": doc.metadata.get("original_filename", "Unknown source"),
                    "page_content_snippet": doc.page_content[:200] + "..."
                }
                for doc in documents
            ]
        }
        return response

    except Exception as e:
        
        return {
            "answer": "Sorry, an internal error occurred. Please try again later.",
            "sources": [],
            "error": "An internal error prevented the request from completing."
        }
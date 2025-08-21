import uuid
from loguru import logger
from langgraph.graph import StateGraph, END
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models import BaseChatModel

from app.services.graph_definition import (
    GraphState,
    create_rag_graph_nodes,
)
class RAGService:
    def __init__(self, retriever: VectorStoreRetriever, llm: BaseChatModel):
        self.retriever = retriever
        self.llm = llm
        self.graph_nodes = create_rag_graph_nodes(retriever, llm)
        self.app_graph = self._build_rag_graph()

    def _build_rag_graph(self):
        """Builds and compiles the LangGraph workflow for handling RAG queries."""
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self.graph_nodes.retrieve_documents)
        workflow.add_node("generate", self.graph_nodes.generate_answer)
        workflow.add_node("rewrite", self.graph_nodes.rewrite_answer)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_conditional_edges(
            "generate",
            self.graph_nodes.check_relevance,
            {
                "useful": END,
                "not_useful": "rewrite",
            },
        )
        workflow.add_edge("rewrite", END)

        logger.info("LangGraph workflow compiled successfully.")
        return workflow.compile()
    
    async def get_rag_response(self, question: str) -> dict:
        """
        Runs the RAG graph for a given question and returns the final answer along with its sources.
        This is the sole function that should be exposed to the rest of the application.
        """
        request_id = str(uuid.uuid4())
        inputs = {"question": question, "request_id": request_id}
        try:
            final_state = await self.app_graph.ainvoke(inputs)

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
            logger.error(f"[{request_id}] An error occurred during RAG graph execution: {e}", exc_info=True)
            return {
                "answer": "Sorry, an internal error occurred. Please try again later.",
                "sources": [],
                "error": "An internal error prevented the request from completing."
            }

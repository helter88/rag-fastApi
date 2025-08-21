
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models import BaseChatModel
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from app.core.config import settings

class RelevanceDecision(BaseModel):
    """A Pydantic model for the relevance check decision."""
    is_relevant: bool = Field(description="Set to True if the answer is fully based on the context, False otherwise.")
class GraphState(TypedDict):
    request_id: str
    question: str
    generation: str
    documents: List[Document]

class RAGGraphNodes:
    """Class containing all RAG graph nodes with injected dependencies."""
    def __init__(self, retriever: VectorStoreRetriever, llm: BaseChatModel):
        self.retriever = retriever
        self.llm = llm

    async def retrieve_documents(self, state: GraphState):
        """Retrieves documents from the vector store."""
        request_id = state['request_id']
        logger.info(f"[{request_id}] Node: retrieve_documents")
        question = state["question"]
        documents = await self.retriever.ainvoke(question)
        logger.info(f"Retrieved {len(documents)} documents.")

        return {"documents": documents, "question": question, "request_id": request_id}
    
    async def generate_answer(self, state: GraphState):
        """Generates an answer based on the retrieved documents."""
        request_id = state['request_id']
        question = state["question"]
        documents = state["documents"]
        logger.info(f"[{request_id}] Node: generate_answer")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer the user's question based solely on the context provided below. "
            +"If the context does not contain an answer, state that and do not try to make one up.\n\nContext:\n---\n{context}\n---"),
            ("human", "Question: {question}")
        ])
        
        rag_chain = prompt | self.llm | StrOutputParser()
        
        context_str = "\n\n".join(doc.page_content for doc in documents)
        generation = await rag_chain.ainvoke({"context": context_str, "question": question})
        
        logger.info("Generated an answer.")
        return {"documents": state["documents"], "question": state["question"], "generation": generation, "request_id": request_id}
    
    async def check_relevance(self, state: GraphState):
        """
        Checks if the generated answer is relevant and grounded in the provided context.
        This node uses structured output to ensure reliable boolean responses.
        """
        request_id = state['request_id']
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        logger.info(f"[{request_id}] Node: check_relevance (structured_output)")

        structured_llm = self.llm.with_structured_output(RelevanceDecision)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI judge. Your task is to evaluate whether the generated answer is fully based on the provided context. " 
            +"Respond with a boolean value indicating relevance."),
            ("human", "Context:\n---\n{context}\n---\n\nQuestion: {question}\n\nGenerated Answer: {generation}")
        ])

        checker_chain = prompt | structured_llm
        
        context_str = "\n\n".join(doc.page_content for doc in documents)
        
        decision_object = await checker_chain.ainvoke({
            "context": context_str, 
            "question": question, 
            "generation": generation
        })
        
        logger.info(f"[{request_id}] Relevance check decision: {decision_object.is_relevant}")
        
        if decision_object.is_relevant:
            return "useful"
        else:
            return "not_useful"

    async def rewrite_answer(self,state: GraphState):
        """Generates an alternative answer, informing the user about the lack of specific data."""
        request_id = state['request_id']
        question = state["question"]
        logger.info(f"[{request_id}] Node: rewrite_answer")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer the user's question. Inform them that sufficient information was not found in the provided documents to give a precise answer, but you will attempt to answer based on your general knowledge."),
            ("human", "Question: {question}")
        ])
        
        rewrite_chain = prompt | self.llm | StrOutputParser()
        generation = await rewrite_chain.ainvoke({"question": question})
        
        return {"documents": state["documents"], "question": question, "generation": generation, "request_id": request_id}

def create_rag_graph_nodes(retriever: VectorStoreRetriever, llm: BaseChatModel) -> RAGGraphNodes:
    """Factory function to create RAG graph nodes with dependencies."""
    return RAGGraphNodes(retriever, llm)
from functools import lru_cache
from http.client import HTTPException
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
import chromadb
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from fastapi import Depends
from loguru import logger

from app.core.config import settings
from app.services.rag_service import RAGService

@lru_cache(maxsize=None)
def get_settings():
    return settings

@lru_cache(maxsize=None)
def get_embeddings_model() -> Embeddings:
    """Provides a singleton instance of the embeddings model."""
    return HuggingFaceEndpointEmbeddings(
        model=get_settings().HF_EMBEDDING_MODEL,
        huggingfacehub_api_token=get_settings().HUGGINGFACEHUB_API_TOKEN
    )

@lru_cache(maxsize=None)
def get_chroma_client() -> chromadb.HttpClient:
    """Provides a singleton instance of the ChromaDB client."""
    try:
        return chromadb.HttpClient(host=get_settings().CHROMA_HOST, port=get_settings().CHROMA_PORT)
    
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise HTTPException(status_code=503, detail="Vector database unavailable")

@lru_cache(maxsize=None)
def get_vector_store() -> Chroma:
    """Provides a singleton instance of the vector store, depending on the client and embeddings."""
    return Chroma(
        client=get_chroma_client(),
        collection_name=get_settings().RAG_COLLECTION_NAME,
        embedding_function=get_embeddings_model()
    )

def get_retriever() -> VectorStoreRetriever:
    """
    Provides a retriever. This function is simple, so it doesn't need caching
    if get_vector_store is already cached. It will be called each time
    but will receive the same cached vector store instance.
    """
    return get_vector_store().as_retriever(search_kwargs={"k": get_settings().RAG_K_DOCUMENTS})

@lru_cache(maxsize=None)
def get_llm() -> BaseChatModel:
    """Provides a singleton instance of the ChatGoogleGenerativeAI model."""
    return ChatGoogleGenerativeAI(
        model=get_settings().LLM_MODEL_NAME, 
        google_api_key=get_settings().GOOGLE_API_KEY, 
        temperature=get_settings().LLM_TEMPERATURE
    )

_rag_service_instance = None

def get_rag_service(
    retriever: VectorStoreRetriever = Depends(get_retriever),
    llm: BaseChatModel = Depends(get_llm),
) -> RAGService:
    """
    Provides a singleton instance of the RAGService.
    """
    global _rag_service_instance
    if _rag_service_instance is None:
        logger.info("Creating RAGService instance for the first time...")
        _rag_service_instance = RAGService(retriever=retriever, llm=llm)
    return _rag_service_instance
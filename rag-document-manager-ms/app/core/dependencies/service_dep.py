from functools import lru_cache
import chromadb
from fastapi import Depends, HTTPException
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.core.config import settings
from app.services.document_service import DocumentService

@lru_cache(maxsize=None)
def get_settings():
    return settings

@lru_cache(maxsize=None)
def get_embeddings_model() -> Embeddings:
    """Provides a singleton instance of the embeddings model."""
    try:
        return HuggingFaceEndpointEmbeddings(
            model=get_settings().HF_EMBEDDING_MODEL,
            huggingfacehub_api_token=get_settings().HUGGINGFACEHUB_API_TOKEN
        )
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {e}")
        raise HTTPException(status_code=503, detail="Embeddings service unavailable")

@lru_cache(maxsize=None)
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Provides a singleton of recursive character text splitter"""
    return RecursiveCharacterTextSplitter(chunk_size=get_settings().TEXT_SPLITTER_CHUNK_SIZE,
                                           chunk_overlap=get_settings().TEXT_SPLITTER_CHUNK_OVERLAP)

@lru_cache(maxsize=None)
def get_chroma_client() -> chromadb.HttpClient:
    """Provides a singleton instance of the ChromaDB client."""
    try:
        client = chromadb.HttpClient(
            host=get_settings().CHROMA_HOST, 
            port=get_settings().CHROMA_PORT
        )
        # Connection test
        client.heartbeat()
        logger.info(f"Successfully connected to ChromaDB at {get_settings().CHROMA_HOST}:{get_settings().CHROMA_PORT}")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        raise HTTPException(status_code=503, detail="Vector database unavailable")

@lru_cache(maxsize=None)
def get_vector_store() -> Chroma:
    """Provides a singleton instance of the vector store, depending on the client and embeddings."""
    try:
        vector_store = Chroma(
            client=get_chroma_client(),
            collection_name=get_settings().RAG_COLLECTION_NAME,
            embedding_function=get_embeddings_model()
        )
        logger.info(f"Vector store initialized with collection: {get_settings().RAG_COLLECTION_NAME}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise HTTPException(status_code=503, detail="Vector store initialization failed")
    
@lru_cache(maxsize=None)
def get_document_service(
    vector_store: Chroma = Depends(get_vector_store),
    text_splitter: RecursiveCharacterTextSplitter = Depends(get_text_splitter)
) -> DocumentService:
    """Provides a singleton instance of the DocumentService."""
    return DocumentService(vector_store=vector_store, text_splitter=text_splitter)
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from loguru import logger

from app.api.endpoints import schemas
from app.api.endpoints.dependencies.validation import validate_files_payload
from app.services import rag_processor


router = APIRouter()

@router.post(
    "/ingest-to-rag",
    response_model=schemas.RAGIngestionResponse,
    summary="Upload, parse and ingest documents into RAG vector store",
    description="Accepts multiple documents, processes them, and stores them as vector embeddings in ChromaDB."
)
async def ingest_documents_to_rag(
    files: List[UploadFile] = Depends(validate_files_payload)
):
    try:
        total_chunks, files_with_errors = await rag_processor.process_and_store_files(files)
        
        processed_count = len(files) - len(files_with_errors)

        return schemas.RAGIngestionResponse(
            total_chunks_added=total_chunks,
            processed_files_count=processed_count,
            files_with_errors=files_with_errors,
            message=f"Ingestion process completed. Processed {processed_count} files successfully."
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during ingestion process : {e}")
        raise HTTPException(status_code=500, detail="An unexpected internal error occurred.")
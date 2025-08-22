from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from loguru import logger

from app.api.endpoints import schemas
from app.core.dependencies.validation import validate_files_payload
from app.core.dependencies.service_dep import get_document_service
from app.services.document_service import DocumentService


router = APIRouter()

@router.get(
    "/documents",
    response_model=schemas.DocumentListResponse,
    summary="List all ingested documents",
    description="Retrieves a list of unique filenames of all documents that have been processed and stored in the RAG vector store."
)
async def list_ingested_documents(document_service: DocumentService = Depends(get_document_service)):
    try:
        document_names = await document_service.get_all_document_names()
        return schemas.DocumentListResponse(
            count=len(document_names),
            documents=document_names
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred while listing documents: {e}")
        raise HTTPException(status_code=500, detail="An unexpected internal error occurred.")


@router.post(
    "/ingest-to-rag",
    response_model=schemas.RAGIngestionResponse,
    summary="Upload, parse and ingest documents into RAG vector store",
    description="Accepts multiple documents, processes them, and stores them as vector embeddings in ChromaDB."
)
async def ingest_documents_to_rag(
    files: List[UploadFile] = Depends(validate_files_payload),
    document_service: DocumentService = Depends(get_document_service)
):
    try:
        total_chunks, files_with_errors = await document_service.process_and_store_files(files)
        
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
    
@router.delete(
    "/documents/{filename}",
    response_model=schemas.DocumentDeleteResponse,
    summary="Delete a document and its chunks from the RAG store",
    description="Removes all vector embeddings associated with the specified filename from ChromaDB and updates the document index."
)
async def delete_document(
    filename: str,
    document_service: DocumentService = Depends(get_document_service)
):
    try:
        await document_service.delete_document_by_name(filename)
        return schemas.DocumentDeleteResponse(
            message="Document and all its associated chunks have been successfully deleted.",
            deleted_filename=filename
        )
    except HTTPException as e:
        # Przekaż wyjątki HTTP (np. 404) dalej
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred while deleting document '{filename}': {e}")
        raise HTTPException(status_code=500, detail="An unexpected internal error occurred.")
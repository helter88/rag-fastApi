    
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from app.api.endpoints import schemas
from app.services.rag_service import RAGService
from app.dependencies import get_rag_service

router = APIRouter()

@router.post(
    "/query",
    response_model=schemas.QueryResponse,
    summary="Ask a question to the RAG pipeline",
    description="Receives a question, retrieves relevant documents, generates an answer, and performs compliance checks."
)
async def ask_question(request: schemas.QueryRequest, ragService: RAGService = Depends(get_rag_service)):
    response_dict = await ragService.get_rag_response(request.question)
    if "error" in response_dict:
        logger.error(f"RAG service returned an error for question '{request.question}': {response_dict['error']}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing the query."
        )
    
    return schemas.QueryResponse(
        answer=response_dict.get("answer", "No answer could be generated."),
        sources=response_dict.get("sources", []) # Poprawiona nazwa klucza na 'sources'
    )
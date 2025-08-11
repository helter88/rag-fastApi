from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.endpoints import schemas
from app.api.endpoints.dependencies.validation import validate_files_payload
from app.services import document_parser


router = APIRouter()

@router.post(
    "/upload-and-parse",
    response_model=schemas.ParsingResponse,
    summary="Upload and parse multiple documents",
    description="Accepts a list of documents (pdf, txt, docx, epub, doc) and returns the first 3 lines of each."
)
async def upload_and_parse_documents(
    files: List[UploadFile] = Depends(validate_files_payload)
):

    results = []
    errors = []
    
    for file in files:       
        try:
            snippet = await document_parser.parse_document(file)
            results.append(schemas.DocumentSnippet(filename=file.filename, content_snippet=snippet))
        except HTTPException as e:
            # Przechwytujemy błędy z serwisu i dodajemy do listy
            errors.append(f"Error processing '{file.filename}': {e.detail}")
        finally:
            # Zawsze zamykaj plik po użyciu
            await file.close()

    return schemas.ParsingResponse(
        results=results,
        total_files_processed=len(results),
        errors=errors
    )
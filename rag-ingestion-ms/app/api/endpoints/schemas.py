from typing import List
from pydantic import BaseModel


class DocumentSnippet(BaseModel):
    filename: str
    content_snippet: List[str]

class RAGIngestionResponse(BaseModel):
    total_chunks_added: int
    processed_files_count: int
    files_with_errors: List[str]
    message: str

class DocumentListResponse(BaseModel):
    count: int
    documents: List[str]
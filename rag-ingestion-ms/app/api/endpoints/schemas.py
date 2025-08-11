from typing import List
from pydantic import BaseModel


class DocumentSnippet(BaseModel):
    filename: str
    content_snippet: List[str]

class ParsingResponse(BaseModel):
    results: List[DocumentSnippet]
    total_files_processed: int
    errors: List[str]
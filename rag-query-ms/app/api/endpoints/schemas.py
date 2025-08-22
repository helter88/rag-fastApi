from pydantic import BaseModel, Field, validator

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    conversation_id: str | None = None
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class SourceDocument(BaseModel):
    filename: str
    page_content_snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
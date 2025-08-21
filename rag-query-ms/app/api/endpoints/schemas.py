from pydantic import BaseModel
class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None # Do obsługi historii konwersacji w przyszłości

class QueryResponse(BaseModel):
    answer: str
    context: list[dict] # Prześlemy fragmenty użytych dokumentów
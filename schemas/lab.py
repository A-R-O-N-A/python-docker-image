from pydantic import BaseModel

class LabBase(BaseModel):
    data_input: str

class PingResponse(BaseModel):
    message: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
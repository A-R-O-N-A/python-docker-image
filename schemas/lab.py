from pydantic import BaseModel
from typing import Annotated, Optional, Any
from fastapi import UploadFile, File

class LabBase(BaseModel):
    data_input: str

class PingResponse(BaseModel):
    message: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    embeddings: Optional[dict[str, list[float]]] = None
    documents: Optional[dict[str, dict[str, Any]]] = None

class RAGFileResponse(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    embeddings : Optional[list[list[float]]] = None
    results : Optional[list[dict[str, Any]]] = None
    query : Optional[str] = None,
    answer : Optional[str] = None,
    # embeddings : list[float]

class RAGFileRequest(BaseModel):
    file: Annotated[UploadFile, File()]

class RAGChatRequest(BaseModel):
    file: Annotated[UploadFile, File()]
    query: Optional[str] = None

class RAGVectorizeResponse(BaseModel): 
    filename: str
    content_type: str
    size_bytes: int
    embeddings : Optional[list[list[float]]] = None

class OCRImageRequest(BaseModel):
    image: Annotated[UploadFile, File()]


class OCRImageResponse(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    text: Optional[str] = None
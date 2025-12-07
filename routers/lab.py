# test router
from fastapi import APIRouter

# test ollama chat from langchain
from langchain_ollama import ChatOllama

from ..schemas.lab import LabBase

router = APIRouter(
    prefix='/lab',
    tags=['lab']
)

@router.get('/test')
def test_endpoint():
    return [{ 'UOHHH' : 'Cunny' }]

@router.get('/test/{test_id}/get')
def get_test_endpoint(test_id: int):
    return {"test_id": test_id}

@router.get('/test/{test_chat}/chat')
def get_test_chat_endpoint(test_chat: str):
    return {"chat" : test_chat}

llm = ChatOllama(
    model='llama3.2:1b',
    temperature=0.5,
)

@router.get('/test/{ollama_chat}/chat/ollama')
def get_ollama_chat(ollama_chat: str):

    messages= [
        (
            'system',
            'You are Flandre Scarlet from Touhou Project as a chatbot'
        ),
        (
            'human',
            ollama_chat
        ),
    ]

    ai_response = llm.invoke(messages)

    return {"ai response" : ai_response}

@router.post('/test/chat/ollama')
def post_ollama_chat(request: LabBase):
    messages=[
        (
            'system',
            'You are Remilia Scarlet, a vampire from Touhou Project'
        ),
        (
            'human',
            request.data_input
        )
    ]

    ai_response = llm.invoke(messages)

    return ai_response
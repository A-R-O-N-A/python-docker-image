# test router
from fastapi import APIRouter, File, UploadFile, Form

# test ollama chat from langchain
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pdfplumber

from ..schemas.lab import LabBase, PingResponse, ChatRequest, RAGFileRequest, RAGFileResponse, RAGChatRequest, RAGVectorizeResponse 

from ..core.config import settings

from io import BytesIO

from PyPDF2 import PdfReader

from typing import Dict, Optional

from ..rag_utils.utils import extract_text_from_file, split_text_into_chunks, perform_vector_similarity_search, generate_chat_response, extract_text_from_content, split_text_into_chunks_v2, generate_chat_response_with_history, create_vector_store_docs, vector_search_hits, get_context_similarity_search

router = APIRouter(
    prefix='/lab',
    tags=['lab']
)

embeddings = OllamaEmbeddings(model=settings.LLM_EMBEDDING_MODEL)

# in memory FAISS vector store

vector_store : Optional[FAISS] = None

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
    model=settings.LLM_MODEL,
    # model="llama3.2:1b",
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


@router.post('/test/ping')
def post_ping(request: PingResponse):

    return {"message_ping": request.messages}

@router.post('/test/array')
def post_ollama_array(request: ChatRequest):

    # convert list of ChatMessage to list of tuples
    req_messages = [tuple(message.model_dump().values()) for message in request.messages]

    # now lets test the ai response with memory
    try :

        messages = req_messages
        ai_response = llm.invoke(messages)
        return ai_response
    except Exception as e:

        return {
            "error": str(e),
            "messages" : req_messages
        }
    
@router.post('/test/rag/file', response_model=RAGFileResponse)
# def post_rag_file(request: RAGFileRequest):
async def post_rag_file(file: UploadFile = File(...)):

    # return {"rag_ping" : request}
    contents = await file.read()
    # text = contents.decode('utf-8')

    text = None
    ctype = file.content_type or ""

    if ctype.startswith('text/'):
        text = contents.decode('utf-8', errors='replace')
    
    elif ctype == 'application/pdf':
        reader = PdfReader(BytesIO(contents))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)

    else : 
        import base64
        text = base64.b64encode(contents).decode('utf-8')


    vector = embeddings.embed_query(text) if text else None
    # vector = embeddings.embed_query(str(contents))

    return {
        "filename" : file.filename,
        "content_type" : ctype,
        "size_bytes" : len(contents),
        "embeddings" : vector
    }

@router.post('/test/rag/file/chat', response_model=RAGFileResponse)
async def post_rag_file_chat(file: UploadFile = File(...), query: Optional[str] = Form(None)):

    contents = await file.read()

    text = None
    ctype = file.content_type or ""

    if ctype.startswith('text/'):
        text = contents.decode('utf-8', errors='replace')
    
    elif ctype == 'application/pdf':
        reader = PdfReader(BytesIO(contents))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)

    else : 
        import base64
        text = base64.b64encode(contents).decode('utf-8')


    vector = embeddings.embed_query(text) if text else None

    ## Add to FAISS in-memory

    global vector_store

    if text : 
        doc = Document(
            page_content=text, 
            metadata={
                'filename' : file.filename, 
                "content_type" : ctype
            })

        if vector_store is None:
            vector_store = FAISS.from_documents([doc], embeddings)
        
        else : 
            vector_store.add_documents([doc])

        # Query immediately

        results= [] 

        if vector_store is not None and text:

            search_text = query if query else text

            hits = vector_store.similarity_search(search_text, k=5)
            # hits = vector_store.similarity_search(text, k=5)

            results = [
                {
                    "filename" : h.metadata.get('filename'),
                    "content_type" : h.metadata.get('content_type'),
                    "text_snippet" : h.page_content[:300],
                }
                for h in hits
            ]
        
        return {
            "filename" : file.filename,
            "content_type" : ctype,
            "size_bytes" : len(contents),
            "embeddings" : vector,
            "results" : results,
            "query": query
        }
    
@router.post('/test/rag/file/chat/v2', response_model=RAGFileResponse)
async def post_rag_file_chat_v2(file: UploadFile = File(...), query: Optional[str] = Form(None)):

    ## Extract file contents
    # TODO test the utility function

    text, ctype, contents = await extract_text_from_file(file)

    vector = embeddings.embed_query(text) if text else None

    global vector_store
    results = []
    answer = None

    if text:
        
        ## Split and chunk the document
        # TODO test the utility
        docs = split_text_into_chunks(text, ctype, file, chunk_size=1000, chunk_overlap=200)


        if vector_store is None:
            vector_store = FAISS.from_documents(docs, embeddings)
        else:
            vector_store.add_documents(docs)

        ## Similarity search from vector via query
        # TODO test the utility function

        hits, results = perform_vector_similarity_search(vector_store, query, text, top_k=5)

        ## LLM answer using retrieved context
        # TODO test the utility function

        answer = generate_chat_response(llm, query, hits)


    # Return values in API
    return {
        "filename": file.filename,
        "content_type": ctype,
        "size_bytes": len(contents),
        "embeddings": vector,
        "results": results,
        "query": query,
        "answer": answer,
    }

@router.post('/test/rag/file/vectorize/', response_model=RAGVectorizeResponse)
async def post_rag_file_vectorize(file: UploadFile = File(...)):
    
    text, ctype, contents = await extract_text_from_file(file)

    vector = embeddings.embed_query(text) if text else None

    return {
        'filename' : file.filename,
        'content_type' : ctype,
        'size_bytes' : len(contents),
        'embeddings' : vector,
    }


@router.post('/test/rag/chat/ollama')
async def post_rag_chat_ollama(request: ChatRequest):

    # convert list of ChatMessage to list of tuples
    req_messages = [tuple(message.model_dump().values()) for message in request.messages]

    # now lets test the ai response with memory
    try :

        #######################
        global vector_store
        results = []

        # Extract and add documents to vector store if provided
        if request.documents:

            docs_to_add = []

            docs_to_add, results = await create_vector_store_docs(request.documents, results)
            
            # Add to vector store
            if docs_to_add:
                if vector_store is None:
                    vector_store = FAISS.from_documents(docs_to_add, embeddings)
                else:
                    vector_store.add_documents(docs_to_add)
        
        # Perform similarity search using embeddings
        if vector_store is not None and request.embeddings:


            # turn this into a utility function

            results = vector_search_hits(
                vector_store,
                request.embeddings,
                results,
                k=3
            )


        context = get_context_similarity_search(
            vector_store,
            embeddings=request.embeddings,
            k=3
        )
        
        ai_response = generate_chat_response_with_history(llm, context, req_messages)

        # Generate AI response with context
        # ai_response = llm.invoke(req_messages)

        # return ai_response

        return {
            "ai_response" : ai_response,
            "results" : results,
        }

        # return {
        #     # "ai_response": ai_response,
        #     "ai_response": answer,
        #     "results": results,
        #     "vector_store_exists": vector_store is not None,
        #     "embeddings" : request.embeddings
        # }


    except Exception as e:
    
        return {
        "error": str(e),
        "ai_response": f"Error: {str(e)}",  # Add this
        "results": [],  # Add this
        "messages": req_messages,
        "embeddings": request.embeddings
    }
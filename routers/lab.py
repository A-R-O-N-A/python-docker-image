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

from ..rag_utils.utils import extract_text_from_file, split_text_into_chunks, perform_vector_similarity_search, generate_chat_response, extract_text_from_content, split_text_into_chunks_v2, generate_chat_response_with_history, create_vector_store_docs, vector_search_hits, get_context_similarity_search, perform_vector_bm25_similarity_search, generate_chat_response_with_bm25

router = APIRouter(
    prefix='/lab',
    tags=['lab']
)

embeddings = OllamaEmbeddings(
    # model=settings.LLM_EMBEDDING_MODEL,
    model='mxbai-embed-large:latest',
    base_url="http://72.62.69.183:11434"
    )

# in memory FAISS vector store

vector_store : Optional[FAISS] = None

def embed_all_chunks(text: str, ctype: str, file: UploadFile) -> Optional[list[list[float]]]:
    if not text:
        return None

    chunks = split_text_into_chunks(text, ctype, file, chunk_size=1000, chunk_overlap=200)
    if not chunks:
        return None

    return embeddings.embed_documents([chunk.page_content for chunk in chunks])

def mean_pool_embeddings(vectors: Optional[list[list[float]]]) -> Optional[list[float]]:
    if not vectors:
        return None

    length = len(vectors[0])
    if length == 0:
        return None

    totals = [0.0] * length
    for vector in vectors:
        if len(vector) != length:
            return None
        for idx, value in enumerate(vector):
            totals[idx] += value

    count = float(len(vectors))
    return [value / count for value in totals]

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
    # model=settings.LLM_MODEL,
    # model="mistral:7b",
    # model='mistral:7b-instruct-q2_K',
    model="llama3.2:3b",
    temperature=0.5,
    base_url="http://72.62.69.183:11434"
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

    # return {"message_ping": request.messages}
    return {"message": f'Successfully processed by FastAPI : " {request.message} "'}

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


    vector = embed_all_chunks(text, ctype, file)
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


    vector = embed_all_chunks(text, ctype, file)

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

    vector = None

    global vector_store
    results = []
    answer = None

    if text:
        
        ## Split and chunk the document
        # TODO test the utility
        docs = split_text_into_chunks(text, ctype, file, chunk_size=1000, chunk_overlap=200)
        if docs:
            vector = embeddings.embed_documents([doc.page_content for doc in docs])


        if vector_store is None:
            vector_store = FAISS.from_documents(docs, embeddings)
        else:
            vector_store.add_documents(docs)

        ## Similarity search from vector via query
        # TODO test the utility function

        hits, results = perform_vector_similarity_search(vector_store, query, text, top_k=5)
        # hits, results = perform_vector_bm25_similarity_search(vector_store, query, text, top_k=5)

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
        "answer": answer
    }

@router.post('/test/rag/file/vectorize/', response_model=None)
async def post_rag_file_vectorize(file: UploadFile = File(...)):
    
    text, ctype, contents = await extract_text_from_file(file)

    vectors = embed_all_chunks(text, ctype, file)
    vector = mean_pool_embeddings(vectors)

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

@router.post('/test/rag/chat/ollama/bm25')
async def post_rag_chat_ollama_bm25(request: ChatRequest):

    # convert list of ChatMessage to list of tuples
    req_messages = [tuple(message.model_dump().values()) for message in request.messages]

    try :
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
        
        # Get the last user message as query
        query = req_messages[-1][1] if req_messages else ""
        
        # Perform BM25 hybrid search to get results
        if vector_store is not None and query:
            try:
                hits, bm25_results = perform_vector_bm25_similarity_search(vector_store, query, query, top_k=3)
                results.extend(bm25_results)
            except Exception as search_error:
                print(f"BM25 search error: {search_error}")
        
        # Generate AI response using BM25 hybrid search
        ai_response = generate_chat_response_with_bm25(llm, vector_store, query, req_messages)

        return {
            "ai_response": ai_response,
            "results": results,
        }

    except Exception as e:
        return {
            "error": str(e),
            "ai_response": f"Error: {str(e)}",
            "results": [],
            "messages": req_messages,
            "embeddings": request.embeddings
        }

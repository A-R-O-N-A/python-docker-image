import pdfplumber
from io import BytesIO
from fastapi import UploadFile, File
from langchain_ollama import ChatOllama

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# from langchain.retrievers import BaseRetrieverv{}

# related to google gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# TODO File content extractor
async def extract_text_from_content(content: bytes, filename: str, content_type: str):
    """Extract text from raw file content"""
    text = None
    
    if content_type.startswith('text/'):
        text = content.decode('utf-8', errors='replace')
    
    elif content_type == 'application/pdf':
        from PyPDF2 import PdfReader
        from io import BytesIO
        reader = PdfReader(BytesIO(content))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    
    else:
        import base64
        text = base64.b64encode(content).decode('utf-8')
    
    return text, content_type, content

async def extract_text_from_file(file: UploadFile = File(...)) -> str:
    contents = await file.read()

    text = None

    ctype = file.content_type or ""


    if ctype.startswith('text/'):

        text = contents.decode('utf-8', errors='replace')

    elif ctype == 'application/pdf':

        with pdfplumber.open(BytesIO(contents)) as pdf:

            text = '\n'.join(page.extract_text() or "" for page in pdf.pages) 
    
    else : 

        import base64
        text = base64.b64encode(contents).decode('utf-8')
    
    return text, ctype, contents

# TODO document splitter and chunker

def split_text_into_chunks(text: str, ctype: str, file: UploadFile, chunk_size: int=1000, chunk_overlap: int=200) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    docs = splitter.create_documents(
        [text],
        metadatas=[{
            "filename": file.filename,
            "content_type": ctype,
        }]
    )

    return docs

def split_text_into_chunks_v2(text: str, ctype: str, file: UploadFile = None, chunk_size: int=1000, chunk_overlap: int=200) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    filename = file.filename if file else "unknown"
    
    docs = splitter.create_documents(
        [text],
        metadatas=[{
            "filename": filename,
            "content_type": ctype,
        }]
    )

    return docs
# TODO vector similarity search

def perform_vector_similarity_search(vector_store, query: str,text: str ,top_k: int=5) -> list[dict]:
    
    search_text = query if query else text[:800]

    hits = vector_store.similarity_search(search_text, k=top_k)

    results = [
        {
            "filename" : h.metadata.get('filename'),
            "content_type" : h.metadata.get('content_type'),
            "text_snippet" : h.page_content[:300],
        }
        for h in hits
    ]

    return hits, results

# TODO Vector search for BM2 5  

def perform_vector_bm25_similarity_search(vector_store, query: str, text: str, top_k: int=5) -> list[dict]:
    
    search_text = query if query else text[:800]

    # Get all documents from vector store for BM25
    all_docs = vector_store.similarity_search("", k=vector_store._collection.count() if hasattr(vector_store, '_collection') else 100)
    
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = top_k
    
    # Create vector store retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # Combine both retrievers with ensemble (hybrid search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]  # Equal weight to BM25 and vector search
    )
    
    # Get results using invoke() instead of get_relevant_documents()
    hits = ensemble_retriever.invoke(search_text)[:top_k]

    results = [
        {
            "doc_id" : h.metadata.get('doc_id'),
            "filename": h.metadata.get('filename'),
            "content_type": h.metadata.get('content_type'),
            "text_snippet": h.page_content[:300],
        }
        for h in hits
        # for idx, h in enumerate(hits)
    ]

    return hits, results

def generate_chat_response(llm: ChatOllama, query: str, hits: list) -> str:

    context = "\n\n".join(h.page_content[:800] for h in hits)
    messages = [
        ("system", f"You are the character described in this document. Respond as this character using the provided information about yourself. Stay in character and use first person."),
        ('system', f'Character Information:\n{context}'),
        ('human', query or 'Introduce yourself.')
    ]

    ai_message = llm.invoke(messages)
    answer = ai_message.content if (hasattr(ai_message, 'content')) else str(ai_message)

    return answer

def generate_chat_response_with_history(llm: ChatOllama, context: str, req_messages: list) -> str:
    """Generate chat response with conversation history and context"""
    
    # Build messages with system context first
    messages = [
        # ("system", f"You are a helpful assistant. Use the following information to answer questions:\n\nContext:\n{context}"),
        ("system", f"You are the character described in this document. Respond as this character using the provided information about yourself. Stay in character and use first person."),
        ('system', f'Respond briefly, keep responses simple, clean, with a maximum of one sentence.'),
        ('system', f'Respond only using the information provided, do not make up any facts. Say if you do not know the answer.'),
        ('system', f'Character Information:\n{context}'),
    ]
    
    # Add conversation history
    messages.extend(req_messages)
    
    ai_message = llm.invoke(messages)
    answer = ai_message.content if (hasattr(ai_message, 'content')) else str(ai_message)

    return answer

# TODO Embeddings generator with document splitter


# TODO Vector store manager

## Utility functions for vector chat management 

async def create_vector_store_docs(documents: list, results: list) -> list:

    import base64
    docs_to_add = []


    for doc_id , doc_data in documents.items():
        try: 
            pass
            content_bytes = base64.b64decode(doc_data['content'])
            text, ctype, _ = await extract_text_from_content(
                content_bytes,
                doc_data['name'],
                doc_data['mime_type']
            )

            if text:

                chunks = split_text_into_chunks_v2(text, ctype, file=None, chunk_size=1000, chunk_overlap=200)

                for chunk in chunks :
                    chunk.metadata.update({
                        'doc_id' : doc_id,
                        'filename' : doc_data['name'],
                        'content_type' : doc_data['mime_type'],
                    })
                
                docs_to_add.extend(chunks)

                return docs_to_add, results
        except Exception as chunk_error:
            results.append({"error": f"Failed to process {doc_data['name']}: {str(chunk_error)}"})

            return docs_to_add, results

def vector_search_hits(vector_store, embeddings, results ,  k: int) -> list:

    for doc_id, embedding_vector in embeddings.items():

        hits = vector_store.similarity_search_by_vector(embedding_vector,k=k)

        results.extend([
            {
                'doc_id' : doc_id,
                'filename' : h.metadata.get('filename'),
                'text_snippet' : h.page_content[:500] + '...',
            }

            for h in hits
        ])
    
    return results

def get_context_similarity_search(vector_store, embeddings, k: int)  -> str:
    context = '\n'.join([hit.page_content for hit in vector_store.similarity_search_by_vector(
        list(embeddings.values())[0] , k=k
    ) if vector_store and embeddings]) if vector_store and embeddings else ''

    return context

def generate_chat_response_with_bm25(llm: ChatOllama, vector_store, query: str, req_messages: list) -> str:
    """Generate chat response using BM25 hybrid search with conversation history"""
    
    # Perform BM25 hybrid search to get relevant context
    search_text = query if query else req_messages[-1][1] if req_messages else ""
    
    if vector_store and search_text:
        try:
            # Get all documents from vector store for BM25
            all_docs = vector_store.similarity_search("", k=vector_store._collection.count() if hasattr(vector_store, '_collection') else 100)
            
            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 6
            
            # Create vector store retriever
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": 6})
            
            # Combine both retrievers with ensemble (hybrid search)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
            
            # Get results
            hits = ensemble_retriever.invoke(search_text)[:6]
            context = "\n\n".join(h.page_content[:800] for h in hits)
            
        except Exception as e:
            context = ""
    else:
        context = ""
    
    # Build messages with system context and conversation history
    messages = [
        ("system", f"You are the character described in this document. Respond as this character using the provided information about yourself. Stay in character and use first person."),
        ('system', f'Respond briefly, keep responses simple, clean, with a maximum of one sentence.'),
        ('system', f'Respond only using the information provided, do not make up any facts. Say if you do not know the answer.'),
        ('system', f'Character Information:\n{context}'),
    ]
    
    # Add conversation history
    messages.extend(req_messages)
    
    ai_message = llm.invoke(messages)
    answer = ai_message.content if (hasattr(ai_message, 'content')) else str(ai_message)

    return answer

def generate_chat_response_with_bm25_gemini(
    vector_store,
    query: str,
    req_messages: list,
    # model: str = "gemini-flash-lite-latest"
    model: str = "gemini-2.5-flash"
    # model: str = "gemma-3-27b-it"
) -> str:
    """Generate chat response using Gemini + BM25 hybrid search with conversation history."""

    search_text = query if query else (req_messages[-1][1] if req_messages else "")

    if vector_store and search_text:
        try:
            all_docs = vector_store.similarity_search(
                "",
                k=vector_store._collection.count() if hasattr(vector_store, "_collection") else 100
            )

            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = 6

            vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )

            hits = ensemble_retriever.invoke(search_text)[:6]
            context = "\n\n".join(h.page_content[:800] for h in hits)
        except Exception:
            context = ""
    else:
        context = ""

    messages = [
        SystemMessage(
            content="You are the character described in this document. Respond as this character using the provided information about yourself. Stay in character and use first person."
        ),
        SystemMessage(
            content="Respond briefly, keep responses simple, clean, with a maximum of one sentence."
        ),
        SystemMessage(
            content="Respond only using the information provided, do not make up any facts. Say if you do not know the answer."
        ),
        SystemMessage(content=f"Character Information:\n{context}"),
    ]

    for m in req_messages or []:
        if isinstance(m, (tuple, list)) and len(m) >= 2:
            role, content = str(m[0]).lower(), str(m[1])
        elif isinstance(m, dict):
            role, content = str(m.get("role", "human")).lower(), str(m.get("content", ""))
        else:
            continue

        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    llm = ChatGoogleGenerativeAI(model=model)
    ai_message = llm.invoke(messages)
    return ai_message.content if hasattr(ai_message, "content") else str(ai_message)

def generate_chat_response_with_bm25_gemma_27b(
    vector_store,
    query: str,
    req_messages: list,
    model: str = "gemma-3-27b-it"
) -> str:
    """Generate chat response using Gemma 27B + BM25 hybrid search with conversation history."""

    search_text = query if query else (req_messages[-1][1] if req_messages else "")

    if vector_store and search_text:
        try:
            all_docs = vector_store.similarity_search(
                "",
                k=vector_store._collection.count() if hasattr(vector_store, "_collection") else 100
            )

            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k =  4

            vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )

            hits = ensemble_retriever.invoke(search_text)[:4]
            context = "\n\n".join(h.page_content[:800] for h in hits)
        except Exception:
            context = ""
    else:
        context = ""

    # Flatten conversation history to avoid system-instruction incompatibility
    history_lines = []
    for m in req_messages or []:
        if isinstance(m, (tuple, list)) and len(m) >= 2:
            role, content = str(m[0]).upper(), str(m[1])
            history_lines.append(f"{role}: {content}")
        elif isinstance(m, dict):
            role = str(m.get("role", "USER")).upper()
            content = str(m.get("content", ""))
            history_lines.append(f"{role}: {content}")

    history_text = "\n".join(history_lines).strip()

    prompt = (
        "You are the character described in this document. "
        "Respond as this character using the provided information about yourself. "
        "Stay in character and use first person.\n"
        "Respond briefly, keep responses simple, clean, with a maximum of one sentence.\n"
        "Respond only using the information provided, do not make up any facts. "
        "Say if you do not know the answer.\n\n"
        f"Character Information:\n{context}\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"User Query:\n{search_text}"
    )

    llm = ChatGoogleGenerativeAI(model=model)
    ai_message = llm.invoke([HumanMessage(content=prompt)])
    return ai_message.content if hasattr(ai_message, "content") else str(ai_message)
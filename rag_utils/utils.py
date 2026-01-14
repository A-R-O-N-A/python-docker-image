import pdfplumber
from io import BytesIO
from fastapi import UploadFile, File
from langchain_ollama import ChatOllama

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

# TODO LLM chat handler

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


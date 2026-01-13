import pdfplumber
from io import BytesIO
from fastapi import UploadFile, File
from langchain_ollama import ChatOllama

# TODO File content extractor


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
        ('system', f'You are the character described in the context provided. Use the context to answer the user query. If the context does not contain the answer, respond with "I do not know".\n\nContext:\n{context}'),
        ('system', f'Character Information:\n{context}'),
        ('human', query or 'Introduce yourself.')
    ]

    ai_message = llm.invoke(messages)
    answer = ai_message.content if (hasattr(ai_message, 'content')) else str(ai_message)

    return answer



# TODO Embeddings generator with document splitter


# TODO Vector store manager
# from typing import Union
# from fastapi import FastAPI

# app = FastAPI()

# @app.get('/')
# def read_root():
#     return {'Waifu' : 'Remilia Scarlet'}

# @app.get('/items/{item_id}')
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {'item_id' : item_id, 'q' : q}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Cunny Adventure API",
    description="API for Cunny Adventure game",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
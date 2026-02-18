# This is where we put our OCR routes
# We may utilize either DeepSeek-OCR or GLM-OCR
# Preferably GLM-OCR

from fastapi import APIRouter, UploadFile, File

from ..schemas.lab import OCRImageRequest, OCRImageResponse

import ollama
import base64

router = APIRouter(
    prefix='/ocr',
    tags=['ocr']
)

@router.get('/test/')
def test_ocr():
    return {'message' : 'OCR router is working'}


@router.post('/process-image/', response_model=OCRImageResponse)
async def process_image(image: UploadFile = File(...)):

    # Here we would process the image using the OCR model and return the extracted text
    # For now, we will just return a placeholder response
    # return {'message' : 'Image processed successfully', 'extracted_text' : 'This is a placeholder for the extracted text from the image.'}

    contents = await image.read()
    size_bytes = len(contents)
    await image.seek(0)

    image_b64 = base64.b64encode(contents).decode('utf-8')

    client = ollama.Client('http://72.62.69.183:11434')

    #response = ollama.chat(
    response = client.chat(
        model='glm-ocr:latest',
        messages=[{
            'role': 'user',
            'content': 'Extract text from this image.',
            'images': [image_b64]
        }]
    )

    extracted_text = response['message']['content']

    return OCRImageResponse(
        filename=image.filename,
        content_type=image.content_type or "application/octet-stream",
        size_bytes=size_bytes,
        text=extracted_text
    )

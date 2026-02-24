from fastapi import APIRouter, UploadFile, File
import base64

from ..schemas.lab import DataAnalyticsRequest, DataAnalyticsResponse

import pandas as pd

router = APIRouter(
    prefix='/data-analytics',
    tags=['Data Analytics']
)

@router.get('/test/')
def test_data_analytics():
    return {'message' : 'Data Analytics router is working'}


@router.post('/data-analytics/analyze', response_model=DataAnalyticsResponse)
async def process_data_analytics(dataset: UploadFile = File(...)) :

    contents = await dataset.read()
    size_bytes = len(contents)
    await dataset.seek(0)
    dataset_b64 = base64.b64encode(contents).decode('utf-8')
    content_type = dataset.content_type or "application/octet-stream"

    # read the data as pandas
    df = pd.read_csv(dataset.file)
    data_parsed = {
        'columns' : df.columns.tolist(),
        'rows' : df.to_dict(orient='records'),
        'row_count' : len(df)
    }


    return DataAnalyticsResponse(
        filename=dataset.filename,
        content_type=content_type,
        size_bytes=size_bytes,
        data_parsed=data_parsed
    )

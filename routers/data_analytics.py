from fastapi import APIRouter, UploadFile, File
import base64

from ..schemas.lab import DataAnalyticsRequest, DataAnalyticsResponse

import pandas as pd
import numpy as np

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

        # Auto-detect numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Generate Plotly config
    plotly_config = {
        'numerical_columns': numerical_cols,
        'categorical_columns': categorical_cols,
        'traces': []
    }

    # Create default traces (one per categorical column or single trace if none)
    if categorical_cols and numerical_cols:
        # Group by first categorical column
        category_col = categorical_cols[0]
        x_col = numerical_cols[0] if len(numerical_cols) > 0 else None
        y_col = numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0]

        for category in df[category_col].unique():
            subset = df[df[category_col] == category]
            plotly_config['traces'].append({
                'x': subset[x_col].tolist() if x_col else list(range(len(subset))),
                'y': subset[y_col].tolist(),
                'name': str(category),
                'type': 'scatter',
                'mode': 'markers'
            })
    elif numerical_cols:
        # No categorical; single trace with first two numerical columns
        x_col = numerical_cols[0]
        y_col = numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0]
        plotly_config['traces'].append({
            'x': df[x_col].tolist(),
            'y': df[y_col].tolist(),
            'name': f'{y_col} vs {x_col}',
            'type': 'scatter',
            'mode': 'markers'
        })


    return DataAnalyticsResponse(
        filename=dataset.filename,
        content_type=content_type,
        size_bytes=size_bytes,
        # data_parsed=data_parsed
        data_parsed={
            'plotly_config' : plotly_config
            # 'data' : data_parsed,
        }
    )

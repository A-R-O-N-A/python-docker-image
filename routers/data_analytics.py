from fastapi import APIRouter, UploadFile, File
import base64
import json

from ..schemas.lab import DataAnalyticsRequest, DataAnalyticsResponse, LabBase
from .lab import post_ollama_chat

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
    supported_trace_types = ['scatter', 'box', 'violin', 'histogram']
    plotly_config = {
        'numerical_columns': numerical_cols,
        'categorical_columns': categorical_cols,
        'supported_trace_types': supported_trace_types,
        'default_plot_type': 'scatter' ,
        'traces': []
    }

    # Descriptive statistics
    descriptive_statistics = {
        'overall': df[numerical_cols].describe().to_dict() if numerical_cols else {},
        'per_trace': {}
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
                'type': supported_trace_types,
                'mode': 'markers'
            })
            
            # Per-trace descriptive stats
            descriptive_statistics['per_trace'][str(category)] = subset[numerical_cols].describe().to_dict()

    elif numerical_cols:
        # No categorical; single trace with first two numerical columns
        x_col = numerical_cols[0]
        y_col = numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0]
        plotly_config['traces'].append({
            'x': df[x_col].tolist(),
            'y': df[y_col].tolist(),
            'name': f'{y_col} vs {x_col}',
            'type': supported_trace_types,
            'mode': 'markers'
        })

    # Get AI interpretation of descriptive statistics
    # ai_interpretation = None
    ai_interpretation = 'Currently disabled for dev testing.'

    # IMPORTANT
    # UNCOMMENT the following block to enable AI interpretation of descriptive statistics using Ollama chat

    # try:
    #     stats_json_str = json.dumps(descriptive_statistics, indent=2)
    #     prompt = f"""Analyze the following descriptive statistics from a dataset and provide insights:

    #     {stats_json_str}

    #     Please provide:
    #     1. Key observations about the data distribution
    #     2. Notable patterns or outliers
    #     3. Recommendations for further analysis
    #     """
        
    #     lab_request = LabBase(data_input=prompt)
    #     ai_response = post_ollama_chat(lab_request)
    #     ai_interpretation = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
    
    # except Exception as e:
    #     ai_interpretation = f"Error generating AI interpretation: {str(e)}"

    return DataAnalyticsResponse(
        filename=dataset.filename,
        content_type=content_type,
        size_bytes=size_bytes,
        data_parsed={
            'plotly_config': plotly_config,
            'descriptive_statistics': descriptive_statistics,
            'ai_interpretation': ai_interpretation
        }
    )
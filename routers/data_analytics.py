from fastapi import APIRouter, UploadFile, File
import base64
import json

from ..schemas.lab import DataAnalyticsRequest, DataAnalyticsResponse, LabBase
from .lab import post_ollama_chat

import pandas as pd
import numpy as np

from langchain_google_genai import ChatGoogleGenerativeAI

router = APIRouter(
    prefix='/data-analytics',
    tags=['Data Analytics']
)

@router.get('/test/')
def test_data_analytics():
    return {'message' : 'Data Analytics router is working'}

@router.post('/data-analytics/analyze', response_model=DataAnalyticsResponse)
async def process_data_analytics(dataset: UploadFile = File(...)):
    contents = await dataset.read()
    size_bytes = len(contents)
    await dataset.seek(0)
    content_type = dataset.content_type or "application/octet-stream"

    # read the data as pandas
    df = pd.read_csv(dataset.file)

    # Auto-detect numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Generate Plotly config
    supported_trace_types = ['scatter', 'scatter3d', 'box', 'violin', 'histogram']
    plotly_config = {
        'numerical_columns': numerical_cols,
        'categorical_columns': categorical_cols,
        'supported_trace_types': supported_trace_types,
        'default_plot_type': 'scatter',
        'traces': []
    }

    # Descriptive statistics
    descriptive_statistics = {
        'overall': df[numerical_cols].describe().to_dict() if numerical_cols else {},
        'per_trace': {}
    }

    # Create default traces (one per categorical group or single trace if none)
    if categorical_cols and numerical_cols:
        category_col = categorical_cols[0]
        x_col = numerical_cols[0]
        y_col = numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0]
        z_col = numerical_cols[2] if len(numerical_cols) > 2 else y_col

        for category in df[category_col].dropna().unique():
            subset = df[df[category_col] == category]
            plotly_config['traces'].append({
                'x': subset[x_col].tolist(),
                'y': subset[y_col].tolist(),
                'z': subset[z_col].tolist(),
                'x_col': x_col,
                'y_col': y_col,
                'z_col': z_col,
                'name': str(category),
                'type': 'scatter',      # must be a single string
                'mode': 'markers'       # dots only
            })

            descriptive_statistics['per_trace'][str(category)] = (
                subset[numerical_cols].describe().to_dict()
            )

    elif numerical_cols:
        x_col = numerical_cols[0]
        y_col = numerical_cols[1] if len(numerical_cols) > 1 else numerical_cols[0]
        z_col = numerical_cols[2] if len(numerical_cols) > 2 else y_col

        plotly_config['traces'].append({
            'x': df[x_col].tolist(),
            'y': df[y_col].tolist(),
            'z': df[z_col].tolist(),
            'x_col': x_col,
            'y_col': y_col,
            'z_col': z_col,
            'name': f'{y_col} vs {x_col}',
            'type': 'scatter',          # must be a single string
            'mode': 'markers'           # dots only
        })

        # Parallel coordinates config (separate from trace switching pipeline)
    parallel_coords_config = {
        'enabled': bool(numerical_cols),
        'trace': None,
        'layout': {
            'width': 800
        }
    }

    if numerical_cols:
        dimensions = []
        for col in numerical_cols:
            series = df[col]
            valid = series.dropna()

            if valid.empty:
                col_min, col_max = 0.0, 0.0
            else:
                col_min, col_max = float(valid.min()), float(valid.max())

            dimensions.append({
                'label': col,
                'range': [col_min, col_max],
                'values': [None if pd.isna(v) else float(v) for v in series.tolist()]
            })

        parallel_coords_config['trace'] = {
            'type': 'parcoords',
            'dimensions': dimensions
        }

    # AI interpretation (disabled)
    # ai_interpretation = 'Currently disabled for dev testing.'
    # implement gemini interpretation of descriptive statistics
    # llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

    # # below is a gemini-2.5-flast implementation, we dont have the money to keep using this other than for COR
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # stats_json = json.dumps(descriptive_statistics, ensure_ascii=False, indent=2)
    # prompt = (
    #     "Provide a concise interpretation of the following descriptive statistics for a dataset. "
    #     "Focus on key insights, trends, and any potential data quality issues.\n\n"
    #     f"{stats_json}"
    # )

    # response = llm.invoke(prompt)
    # ai_interpretation = response.content if hasattr(response, "content") else str(response)

    llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it")

    # Limit prompt to ~14k tokens (roughly 4 chars/token)
    MAX_TOKENS = 14_000
    CHARS_PER_TOKEN = 4
    MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN

    instruction = (
        "Provide a concise interpretation of the following descriptive statistics for a dataset. "
        "Focus on key insights, trends, and any potential data quality issues.\n\n"
    )

    # compact JSON to save tokens
    stats_json = json.dumps(descriptive_statistics, ensure_ascii=False, separators=(",", ":"))

    # reserve room for instruction + truncation note
    reserve = len(instruction) + 200
    allowed_json_chars = max(0, MAX_CHARS - reserve)

    if len(stats_json) > allowed_json_chars:
        stats_json = stats_json[:allowed_json_chars] + '..."__truncated__":true}'

    prompt = instruction + stats_json

    response = llm.invoke(prompt)
    ai_interpretation = response.content if hasattr(response, "content") else str(response)
    return DataAnalyticsResponse(
        filename=dataset.filename,
        content_type=content_type,
        size_bytes=size_bytes,
        data_parsed={
            'plotly_config': plotly_config,
            'parallel_coords_config': parallel_coords_config,
            'descriptive_statistics': descriptive_statistics,
            'ai_interpretation': ai_interpretation
        }
    )
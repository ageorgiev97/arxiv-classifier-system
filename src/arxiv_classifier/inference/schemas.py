from pydantic import BaseModel
from typing import List

class PredictionResult(BaseModel):
    label: str
    probability: float

class InferenceResponse(BaseModel):
    abstract_preview: str
    predictions: List[PredictionResult]
    model_version: str = "scibert-v1"
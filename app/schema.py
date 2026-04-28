from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    RIDAGEYR: float = Field(..., ge=0, le=120)
    RIAGENDR: int
    BMXBMI: float = Field(..., ge=0)# This is not required for inference, but we can impute it during training and use the imputed value for prediction. This way we ca maintain consistency between training and inference.
    PAQ605: float
    PAQ620: float
    SLD012: Optional[float] = None
    INDFMMPI: Optional[float] = None
    BPQ020: float
    DR1TKCAL: Optional[float] = None
    DR1TSUGR: Optional[float] = None
    DR1TTFAT: Optional[float] = None
    DR1TPROT: Optional[float] = None
    DR1TSODI: Optional[float] = None
    DBD895: float
    DBD900: Optional[float] = None


class BatchPredictionInput(BaseModel):
    records: List[PredictionInput]


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    latency_seconds: float

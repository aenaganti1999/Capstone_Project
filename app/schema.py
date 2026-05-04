from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    RIDAGEYR: float = Field(..., ge=0, le=120)
    RIAGENDR: Literal[1, 2]
    BMXBMI: float = Field(..., ge=0)  # This is not required for inference.
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

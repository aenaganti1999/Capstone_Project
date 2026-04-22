from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    RIDAGEYR: float
    RIAGENDR: int
    BMXBMI: float
    PAQ605: float
    PAQ620: float
    SLD012: float
    INDFMMPI: float
    BPQ020: float
    DR1TKCAL: float
    DR1TSUGR: float
    DR1TTFAT: float
    DR1TPROT: float
    DR1TSODI: float
    DBD895: float
    DBD900: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    latency_seconds: float
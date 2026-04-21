from pydantic import BaseModel

class PredictionInput(BaseModel):
    RIDAGEYR: float
    RIAGENDR: int
    BMXBMI: float
    ALQ111: float
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
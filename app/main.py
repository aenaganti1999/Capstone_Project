import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request
import logging
import app.model_loader as ml
from .schema import PredictionInput, PredictionResponse, BatchPredictionInput
from .preprocess import preprocess_input

app = FastAPI(title="Healthcare Prediction API")

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}")

    return JSONResponse(
        status_code=500, content={"message": "Something went wrong. Please try again."}
    )


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Single prediction
@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Predict health risk using trained ML model.

    Returns:
    - prediction: 0 or 1
    - probability: likelihood of positive class
    - threshold: decision threshold used
    - latency_seconds: time taken for prediction
    """
    if ml.model is None:
        ml.load_artifacts()
    try:

        # Convert input once (Pydantic v2)
        input_dict = input_data.model_dump()
        logger.info(f"Incoming request: {input_dict}")

        # Start timing
        start = time.time()

        # Preprocess
        processed = preprocess_input(input_dict)

        # Probability + threshold
        probability = ml.model.predict_proba(processed)[0][1]
        prediction = int(probability > ml.threshold)

        # End timing
        latency = time.time() - start

        logger.info(
            f"prob={probability:.4f},"
            f"threshold={ml.threshold},"
            f"pred={prediction},"
            f"latency={latency:.4f}s"
        )

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            threshold=float(ml.threshold),
            latency_seconds=round(latency, 4),
        )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch prediction
@app.post("/batch_predict")
def batch_predict(inputs: BatchPredictionInput):
    if ml.model is None:
        ml.load_artifacts()
    try:
        results = []

        start = time.time()

        for item in inputs.records:
            input_dict = item.model_dump()
            processed = preprocess_input(input_dict)

            probability = ml.model.predict_proba(processed)[0][1]
            prediction = int(probability > ml.threshold)

            results.append(
                {
                    "prediction": prediction,
                    "probability": float(probability),
                    "threshold": float(ml.threshold),
                }
            )

        latency = time.time() - start

        logger.info(f"Batch size={len(inputs.records)}, latency={latency:.4f}s")

        return {"results": results, "batch_latency_seconds": round(latency, 4)}

    except Exception as e:
        logger.error(f"Batch error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

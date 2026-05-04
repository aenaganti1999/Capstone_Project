import time
import logging
import uuid
from pathlib import Path
from contextvars import ContextVar
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
import app.model_loader as ml
from .schema import PredictionInput, PredictionResponse, BatchPredictionInput
from .preprocess import preprocess_input

# Request ID context variable
request_id: ContextVar[str] = ContextVar("request_id", default="")


# Custom logging filter to include request ID
class RequestIDFilter(logging.Filter):
    def filter(self, record):
        rid = request_id.get()
        record.request_id = rid if rid else "STARTUP"
        return True


# -------------------------------
# Logging Setup
# -------------------------------
# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure logging format with request ID
log_format = "%(asctime)s - [%(levelname)s] - [%(request_id)s] - %(message)s"

# File Handler
file_handler = logging.FileHandler(log_dir / "api.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))
file_handler.addFilter(RequestIDFilter())

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(log_format))
console_handler.addFilter(RequestIDFilter())

# Configure app logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configure uvicorn loggers to write to file as well
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)
for handler in [file_handler, console_handler]:
    if handler not in uvicorn_logger.handlers:
        uvicorn_logger.addHandler(handler)

uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.setLevel(logging.WARNING)
for handler in [file_handler, console_handler]:
    if handler not in uvicorn_access.handlers:
        uvicorn_access.addHandler(handler)


# -------------------------------
# FastAPI App
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts at startup...")
    ml.load_artifacts()

    if ml.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    logger.info("Model loaded successfully.")
    yield  # app runs here


app = FastAPI(title="Healthcare Prediction API", lifespan=lifespan)


# Middleware to set request ID
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to context for all logs"""
    rid = str(uuid.uuid4())[:8]
    request_id.set(rid)
    start = time.time()
    logger.info(f"request_start | method={request.method} | path={request.url.path}")
    response = await call_next(request)
    latency = time.time() - start
    logger.info(f"request_end | status={response.status_code} | latency={latency:.4f}s")

    return response


# -------------------------------
# Exception Handlers
# -------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors (422) with details - combine all errors in one line"""
    error_details = exc.errors()
    # Format: field1=error1, field2=error2
    field_errors = ", ".join(
        [f"{error['loc'][-1]}={error['msg']}" for error in error_details]
    )
    logger.error(f"validation_error | {field_errors}")

    return JSONResponse(status_code=422, content={"detail": error_details})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"unhandled_error | type={type(exc).__name__} | error={str(exc)}")
    return JSONResponse(
        status_code=500, content={"message": "Something went wrong. Please try again."}
    )


# -------------------------------
# Health Check
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------------
# Single Prediction
# -------------------------------
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        model = ml.model
        threshold = ml.threshold

        input_dict = input_data.model_dump()
        start = time.time()

        processed = preprocess_input(input_dict)
        try:
            probability = model.predict_proba(processed)[0][1]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        prediction = int(probability >= threshold)
        latency = time.time() - start

        logger.info(
            f"prediction | pred={prediction} | prob={probability:.4f} | "
            f"latency={latency:.4f}s"
        )

        return PredictionResponse(
            prediction=prediction,
            probability=float(probability),
            threshold=float(threshold),
            latency_seconds=round(latency, 4),
        )

    except Exception as e:
        logger.error(f"prediction_error | type={type(e).__name__} | error={str(e)}")
        raise HTTPException(status_code=500, detail="Model inference failed")


# -------------------------------
# Batch Prediction (vectorized)
# -------------------------------
@app.post("/batch_predict")
def batch_predict(inputs: BatchPredictionInput):
    try:
        model = ml.model
        threshold = ml.threshold
        records = inputs.records

        start = time.time()

        # Convert all inputs at once
        input_dicts = [item.model_dump() for item in records]

        # Preprocess entire batch
        processed_batch = preprocess_input(input_dicts)

        # Vectorized prediction
        try:
            probabilities = model.predict_proba(processed_batch)[:, 1]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        predictions = (probabilities >= threshold).astype(int)

        # Format output
        results = [
            {
                "prediction": int(pred),
                "probability": float(prob),
                "threshold": float(threshold),
            }
            for pred, prob in zip(predictions, probabilities)
        ]

        latency = time.time() - start

        logger.info(f"batch_prediction | batch_size={len(records)} ")

        return {
            "results": results,
            "batch_latency_seconds": round(latency, 4),
        }

    except Exception as e:
        logger.error(
            f"batch_prediction_error | type={type(e).__name__} | error={str(e)}"
        )
        raise HTTPException(status_code=500, detail="Model inference failed")

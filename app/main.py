from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import SensorDataRequest, PredictionResponse, SensorReading
from .inference import TransportPredictor
from .config import settings
import time
import logging
from fastapi.responses import JSONResponse

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#App Initialization
app = FastAPI(title="Transport Mode Detection API", version="1.0.0")

#CORS Middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])

#Global Model Instance
predictor: TransportPredictor | None = None

#Startup Event
@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        predictor = TransportPredictor(
            model_path=settings.model_path,
            artifacts_path=settings.artifacts_path  #folder with scaler, encoder, feature stats
        )
        predictor.load_model()  # This now loads model + scaler + encoder + feature stats
        logger.info("✅ Model and preprocessing artifacts loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model or artifacts at startup: {e}")
        predictor = None

#Healthcheck Endpoint
@app.get("/health")
async def health_check():
    model_loaded = predictor is not None and predictor.model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

#Main Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_transport_mode(request: dict):
    try:
        # Validate input manually because aiController sends 'sensor_data_array'
        sensor_data_array = request.get("sensor_data_array")
        trip_id = request.get("trip_id")

        if not sensor_data_array or not isinstance(sensor_data_array, list):
            raise HTTPException(status_code=400, detail="sensor_data_array must be a non-empty list")

        logger.info(f"Received {len(sensor_data_array)} samples for trip {trip_id}")

        # Convert JSON list to Pydantic SensorReading models
        sensor_readings = [SensorReading(**reading) for reading in sensor_data_array]

        # Run model prediction (TransportPredictor.predict already returns PredictionResponse)
        prediction_response = predictor.predict(sensor_readings)

        return prediction_response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

#Error Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    #Log unexpected errors
    #Return generic error response
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
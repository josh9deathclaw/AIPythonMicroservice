from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import SensorDataRequest, PredictionResponse
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
    #Load ML model once at startup
    global predictor
    #initiailise predictor instance
    predictor = TransportPredictor(model_path=settings.model_path)
    predictor.load_model()
    #log successful startup
    logger.info("Model loaded successfully at startup")

#Healthcheck Endpoint
@app.get("/health")
async def health_check():
    #return service status
    #check if model is loaded
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }

#Main Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_transport_mode(request: SensorDataRequest):
    # Start timing
    start_time = time.time()
    
    # Add debug logging
    logger.info(f"Received prediction request for user: {request.user_id}")
    logger.info(f"Window size: {request.window_size}")
    
    try:
        # Make Prediction - assuming your predictor.predict returns (prediction, confidence)
        #prediction, confidence = predictor.predict(request)
        
        # Calculate processing time in milliseconds
        #processing_time_ms = round((time.time() - start_time) * 1000, 3)
        
        # Determine if user confirmation is needed (low confidence)
        #needs_user_check = confidence < settings.confidence_threshold
        
        #Your predictor expects List[SensorReading], but request has single SensorReading
        # So we need to pass it as a list
        sensor_readings_list = [request.sensor_data]  # Convert single reading to list
        
        # Make prediction - passing list of readings as expected by predictor
        prediction_response = predictor.predict(sensor_readings_list)

        return prediction_response

        #return PredictionResponse(
         #   predicted_mode=prediction,
         #   confidence=confidence,
          #  processing_time_ms=processing_time_ms,
         #   needs_user_check=needs_user_check,
          ##  alternative_modes=None  # You can add this if your predictor returns alternative predictions
        #)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(f"Request data: {request}")
        # Add more detailed error logging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
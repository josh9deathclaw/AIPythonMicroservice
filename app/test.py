# app/test.py
from fastapi import FastAPI
from app.models import SensorReading, AccelerometerData, GyroscopeData, GPSData
from app.inference import TransportPredictor
from app.config import settings
from datetime import datetime

app = FastAPI(title=settings.api_title, version=settings.api_version)

# Initialize predictor
predictor = TransportPredictor(model_path=settings.model_path)
predictor.load_model()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    # Example dummy reading
    reading = SensorReading(
        accelerometer=AccelerometerData(x=0.1, y=0.2, z=0.3),
        gyroscope=GyroscopeData(x=0.01, y=0.02, z=0.03),
        gps=GPSData(speed=50.0, altitude=100.0),
        timestamp=datetime.utcnow()
    )

    features = predictor.preprocessor.preprocess_sensor_data(reading, window_size=30)
    prediction = predictor.predict(reading, window_size=30)

    return {
        "features": features.tolist(),  # convert numpy array to list
        "predicted_mode": prediction.predicted_mode,
        "confidence": prediction.confidence
    }
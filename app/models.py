#Data Models for API Request / Response Validation

#Imports
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime

#Sensor Data Models
class AccelerometerData(BaseModel):
    x: float = Field(..., description="X-axis acceleration (m/s²)")
    y: float = Field(..., description="Y-axis acceleration (m/s²)")
    z: float = Field(..., description="Z-axis acceleration (m/s²)")

class GyroscopeData(BaseModel):
    x: float = Field(..., description="X-axis angular velocity (rad/s)")
    y: float = Field(..., description="Y-axis angular velocity (rad/s)")
    z: float = Field(..., description="Z-axis angular velocity (rad/s)")

class GPSData(BaseModel):
    speed: Optional[float] = Field(0.0, description="Speed in km/h")
    altitude: Optional[float] = Field(0.0, description="Altitude in meters")
    lat: Optional[float] = Field(None, description="Latitude")
    lon: Optional[float] = Field(None, description="Longitude")
    accuracy: Optional[float] = Field(None, description="GPS accuracy in meters")

class SensorReading(BaseModel):
    accelerometer: AccelerometerData
    gyroscope: GyroscopeData
    gps: Optional[GPSData] = None
    timestamp: Optional[datetime] = None
    
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

#Request Model
class SensorDataRequest(BaseModel):
    sensor_data_array: List[SensorReading] = Field(
        ..., description="List of 600 sensor samples for 1-minute window"
    )
    user_id: Optional[str] = None
    trip_id: Optional[str] = None

    @validator('sensor_data_array')
    def validate_window_size(cls, v):
        if len(v) != 600:
            raise ValueError("Expected exactly 600 samples per 1-minute window")
        return v

#Response Model
class PredictionResponse(BaseModel):
    predicted_mode: str = Field(..., description="Predicted transport mode")
    confidence: float = Field(..., description="Confidence score (0-1)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    needs_user_check: bool = Field(..., description="Whether user confirmation is needed")
    alternative_modes: Optional[Dict[str, float]] = Field(None, description="Other possible modes with scores")

#Config Model
class Settings(BaseModel):
    model_path: str
    confidence_threshold: float = 0.6
    log_level: str = "INFO"
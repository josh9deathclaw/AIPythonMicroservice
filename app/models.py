#Data Models for API Request / Response Validation

#Imports
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
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
    speed: Optional[float] = Field(None, description="Speed in km/h")
    altitude: Optional[float] = Field(None, description="Altitude in meters")
    accuracy: Optional[float] = Field(None, description="GPS accuracy in meters")

class SensorReading(BaseModel):
    accelerometer: AccelerometerData
    gyroscope: GyroscopeData
    gps: Optional[GPSData] = None
    timestamp: datetime
    
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        # Convert string to datetime if needed
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

#Request Model
class SensorDataRequest(BaseModel):
    sensor_data: SensorReading
    window_size: int = Field(30, description="Time window in seconds")
    user_id: Optional[str] = None
    
    @validator('window_size')
    def validate_window_size(cls, v):
        # Ensure window_size is reasonable (10-300 seconds)
        if not 10 <= v <= 300:
            raise ValueError('Window size must be between 10 and 300 seconds')
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
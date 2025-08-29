# IMPORTS
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # MODEL SETTINGS
    model_path: str = os.getenv("MODEL_PATH", "./models/transport_classifier.pkl")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    artifacts_path: str = "./models"
    
    # API SETTINGS
    api_title: str = "Transport Mode Detection API"
    api_version: str = "1.0.0"
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    
    # LOGGING
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS SETTINGS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]  # Add your frontend URLs
    
    # PREPROCESSING SETTINGS
    max_window_size: int = 300  # seconds
    min_window_size: int = 10   # seconds
    default_window_size: int = 30  # seconds
    
    # TRANSPORT MODES (should match your trained model)
    #unsure if it needs to be plane or flight
    transport_modes: List[str] = [
        "walking",  "car", "bus", "train", "tram", "stationary", "plane"
    ]
    
    # FEATURE SETTINGS
    expected_feature_count: int = 42  # Adjust based on your model
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# CREATE GLOBAL SETTINGS INSTANCE
settings = Settings()

# LOGGING CONFIGURATION
import logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
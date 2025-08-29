#Load the AI Model and Prediction Logic

import joblib  # or pickle, tensorflow, torch depending on model
import numpy as np
from typing import Dict, List, Any
from .preprocessing import DataPreprocessor
from .models import SensorReading, PredictionResponse
from .config import settings
import logging
import os
import time
import logging

class TransportPredictor:
    def __init__(self, model_path: str, artifacts_path: str = None):
        self.model_path = model_path
        self.model = None
        self.preprocessor = DataPreprocessor(model_artifacts_path=artifacts_path)
        self.transport_modes = ["walking", "car", "bus", "train", "tram"]
        self.confidence_threshold = settings.confidence_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
    def load_model(self):
        """Load the trained ML model + preprocessing artifacts"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            self.logger.info(f"✅ Model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        # Log preprocessing artifacts status
        if self.preprocessor.scaler:
            self.logger.info(f"✅ Scaler loaded: mean shape={self.preprocessor.scaler.mean_.shape}, std shape={self.preprocessor.scaler.scale_.shape}")
        if self.preprocessor.label_encoder:
            self.logger.info(f"✅ Label encoder loaded: classes={self.preprocessor.label_encoder.classes_}")
        if self.preprocessor.feature_means is not None and self.preprocessor.feature_stds is not None:
            self.logger.info("✅ Feature stats loaded successfully")

    
    def predict(self, readings: List[SensorReading]) -> PredictionResponse:
        """
        Make transport mode prediction using the loaded model.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.logger.info(f"Starting prediction for {len(readings)} sensor reading(s)")
        start_time = time.time()

        try:
            # STEP 1: Preprocess sensor data
            features = self.preprocessor.preprocess_sensor_data(readings)
            self.logger.info(f"Feature vector generated: {features.shape}")

            # STEP 2: Validate features
            if not self._validate_features(features):
                raise ValueError("Invalid feature vector generated")
            self.logger.info("Feature vector validation passed")

            # STEP 3: Reshape features and make prediction
            features_reshaped = features.reshape(1, -1)
            prediction_probabilities = self._predict_probabilities(features_reshaped)
            self.logger.info(f"Prediction probabilities: {prediction_probabilities}")

            # STEP 4: Get top prediction
            predicted_class_idx = np.argmax(prediction_probabilities)
            predicted_mode = self.transport_modes[predicted_class_idx]
            confidence = float(prediction_probabilities[predicted_class_idx])
            self.logger.info(f"Predicted mode: {predicted_mode}, confidence: {confidence:.3f}")

            # STEP 5: Determine if user confirmation is needed
            needs_user_check = confidence < self.confidence_threshold

            # STEP 6: Get alternative modes
            alternative_modes = self._get_alternative_modes(prediction_probabilities, top_n=3)
            self.logger.info(f"Alternative modes: {alternative_modes}")

            processing_time_ms = (time.time() - start_time) * 1000
            self.logger.info(f"Prediction completed in {processing_time_ms:.2f} ms")

            return PredictionResponse(
                predicted_mode=predicted_mode,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                needs_user_check=needs_user_check,
                alternative_modes=alternative_modes
            )

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from model"""
        features_reshaped = features.reshape(1, -1)
        try:
            # Reshape features if needed (model might expect 2D input)
            features_reshaped = features.reshape(1, -1)
            
            # Different prediction methods based on model type:
            # sklearn: model.predict_proba(features_reshaped)[0]
            # tensorflow: model.predict(features_reshaped)[0]
            # pytorch: model(torch.tensor(features_reshaped))[0]
            
            probabilities = self.model.predict_proba(features_reshaped)[0]
            return probabilities
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise
    
    def _validate_features(self, features: np.ndarray) -> bool:
        """Validate feature vector"""
        # Check for NaN values
        # Check feature vector length matches expected
        # Check for reasonable value ranges
        
        if np.isnan(features).any():
            return False
        if len(features) == 0:
            return False
        # Add more validation based on your model requirements
        return True
    
    def _get_alternative_modes(self, probabilities: np.ndarray, top_n: int = 3) -> Dict[str, float]:
        """Get top N alternative transport modes"""
        # Get indices of top probabilities
        top_indices = np.argsort(probabilities)[::-1][:top_n]
        alternatives = {}

        for idx in top_indices[1:]:  # Skip the top prediction
            mode = self.transport_modes[idx]
            confidence = float(probabilities[idx])
            if confidence > 0.1:  # Only include if reasonably confident
                alternatives[mode] = confidence
        return alternatives
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold"""
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            logging.info(f"Confidence threshold updated to {new_threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

# UTILITY FUNCTIONS
def get_model_info(model_path: str) -> Dict[str, Any]:
    """Return model metadata"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    info = {
        "model_path": model_path,
        "size_bytes": os.path.getsize(model_path),
        "type": "scikit-learn (joblib)"  # update if TF/PyTorch
    }
    return info

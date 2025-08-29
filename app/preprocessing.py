import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
from .models import SensorReading
from sklearn.preprocessing import StandardScaler
import joblib
import logging

class DataPreprocessor:
    def __init__(self, model_artifacts_path: str = None):
        # Initialize StandardScaler (fit during training, here we mock)
        self.scaler = StandardScaler()
        self.feature_means = None
        self.feature_stds = None
        self.label_encoder = None

        if model_artifacts_path:
            self.load_preprocessing_artifacts(model_artifacts_path)

    def load_preprocessing_artifacts(self, artifacts_path: str):
        """Load saved preprocessing artifacts from training"""
        try:
            # Load scaler
            scaler_path = f"{artifacts_path}/scaler.pkl"
            self.scaler = joblib.load(scaler_path)
            logging.info(f"✅ Scaler loaded from {scaler_path}")
            logging.info(f"   Feature means shape: {self.scaler.mean_.shape}")
            logging.info(f"   Feature stds shape: {self.scaler.scale_.shape}")
            
            # Load label encoder
            encoder_path = f"{artifacts_path}/label_encoder.pkl"
            self.label_encoder = joblib.load(encoder_path)
            logging.info(f"✅ Label encoder loaded from {encoder_path}")
            logging.info(f"   Label classes: {self.label_encoder.classes_}")
            
            # Load normalization stats for compatibility
            self.feature_means = np.load(f"{artifacts_path}/feature_means.npy")
            self.feature_stds = np.load(f"{artifacts_path}/feature_stds.npy")
            logging.info(f"✅ Feature statistics loaded from {artifacts_path}")
            
            logging.info(f"Loaded preprocessing artifacts from {artifacts_path}")
            
        except Exception as e:
            logging.error(f"Failed to load preprocessing artifacts: {e}")
            raise
        
    def preprocess_sensor_data(self, sensor_data_list: List[Dict]) -> np.ndarray:

        """
        Main preprocessing pipeline
        
        STEPS:
        1. Extract raw sensor values from 600 samples
        2. Create Feature Dictionary for each sample
        3. Calculate Statistical Features across the window
        4. Normalise features
        5. Return 40-feature vector for ML model
        """
        if len(sensor_data_list) == 0:
            raise ValueError("No sensor data provided")
        
        if len(sensor_data_list) != 600:
            logging.warning(f"Expected 600 samples, got {len(sensor_data_list)}")
        
        # Convert each sensor reading to feature dictionary
        feature_dicts = []
        for sensor_reading in sensor_data_list:
            try:
                features = self._sensor_dict_to_features(sensor_reading)
                feature_dicts.append(features)
            except Exception as e:
                logging.error(f"Error processing sensor reading: {e}")
                # Skip invalid readings
                continue
        
        if len(feature_dicts) == 0:
            raise ValueError("No valid sensor readings found")
        
        # Convert to DataFrame and calculate statistics
        df = pd.DataFrame(feature_dicts)
        
        # Ensure correct feature order (critical for model compatibility)
        expected_features = [
            "acc_x", "acc_y", "acc_z", 
            "gyro_x", "gyro_y", "gyro_z",
            "acc_mag", "gyro_mag", 
            "gps_speed", "gps_alt"
        ]
        
        # Reorder columns to match training
        df = df[expected_features]
        
        # Calculate statistical features (matching training pipeline)
        stats = df.agg(['mean', 'std', 'min', 'max']).fillna(0)
        features = stats.values.flatten()  # 40 features (10 base × 4 stats)
        
        # Normalize features using training statistics
        features = self._normalize_features(features)
        
        logging.info(f"✅ Processed {len(feature_dicts)} samples → {len(features)} features")
        return features

    
    def _sensor_dict_to_features(self, sensor_reading) -> Dict[str, float]:
        """
        Convert sensor dictionary from frontend to feature dictionary
        
        Input format from frontend:
        {
            "accelerometer": {"x": float, "y": float, "z": float},
            "gyroscope": {"x": float, "y": float, "z": float},  # Note: frontend sends alpha/beta/gamma as x/y/z
            "gps": {"speed": float, "altitude": float, "lat": float, "lon": float}
        }
        """
        try:
            if hasattr(sensor_reading, 'accelerometer'):
                # Pydantic SensorReading model
                acc = sensor_reading.accelerometer
                gyro = sensor_reading.gyroscope
                gps = sensor_reading.gps if sensor_reading.gps else {}
                    
                acc_x = float(acc.x)
                acc_y = float(acc.y)
                acc_z = float(acc.z)
                    
                gyro_x = float(gyro.x)
                gyro_y = float(gyro.y)
                gyro_z = float(gyro.z)
                    
                gps_speed = float(gps.speed) if hasattr(gps, 'speed') and gps.speed is not None else 0.0
                gps_alt = float(gps.altitude) if hasattr(gps, 'altitude') and gps.altitude is not None else 0.0
                    
            else:
                # Dict format from frontend
                acc = sensor_reading.get('accelerometer', {})
                gyro = sensor_reading.get('gyroscope', {})
                gps = sensor_reading.get('gps', {})
                    
                # Extract accelerometer values
                acc_x = float(acc.get('x', 0.0))
                acc_y = float(acc.get('y', 0.0))
                acc_z = float(acc.get('z', 0.0))
                    
                # Extract gyroscope values - handle both x/y/z and alpha/beta/gamma
                gyro_x = float(gyro.get('x', gyro.get('alpha', 0.0)))
                gyro_y = float(gyro.get('y', gyro.get('beta', 0.0)))
                gyro_z = float(gyro.get('z', gyro.get('gamma', 0.0)))
                    
                # Extract GPS values (handle missing values)
                gps_speed = float(gps.get('speed', 0.0)) if gps.get('speed') is not None else 0.0
                gps_alt = float(gps.get('altitude', 0.0)) if gps.get('altitude') is not None else 0.0
                
            # Calculate magnitudes
            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
                
            return {
                    "acc_x": acc_x,
                    "acc_y": acc_y,
                    "acc_z": acc_z,
                    "gyro_x": gyro_x,
                    "gyro_y": gyro_y,
                    "gyro_z": gyro_z,
                    "acc_mag": acc_mag,
                    "gyro_mag": gyro_mag,
                    "gps_speed": gps_speed,
                    "gps_alt": gps_alt,
                }
        except Exception as e:
            logging.error(f"Error converting sensor reading to features: {e}")
            raise ValueError(f"Invalid sensor data format: {e}")
        
        
    def _aggregate_time_window(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        """
        Aggregate list of feature dicts into single 1D array
        with mean, std, min, max for each feature.
        """
        if not features_list:
            raise ValueError("No sensor readings provided")

        keys = features_list[0].keys()
        array = np.array([[f[k] for k in keys] for f in features_list], dtype=np.float32)

        # Aggregate statistics
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0)
        min_val = np.min(array, axis=0)
        max_val = np.max(array, axis=0)

        # Concatenate into final vector
        aggregated = np.concatenate([mean, std, min_val, max_val])
        return aggregated

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Standardize features using training statistics"""
        if self.scaler is not None:
            # Use sklearn scaler (preferred method)
            return self.scaler.transform(features.reshape(1, -1)).flatten()
        elif self.feature_means is not None and self.feature_stds is not None:
            # Fallback to manual normalization
            return (features - self.feature_means) / (self.feature_stds + 1e-8)
        else:
            logging.warning("No normalization statistics available, returning raw features")
            return features
    
    def set_normalization_stats(self, means: np.ndarray, stds: np.ndarray):
        """Set normalization statistics from training data"""
        self.feature_means = means
        self.feature_stds = stds
        logging.info(f"Set normalization stats: means={len(means)}, stds={len(stds)}")
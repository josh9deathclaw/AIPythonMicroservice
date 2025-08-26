import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from .models import SensorReading
from sklearn.preprocessing import StandardScaler
import logging

class DataPreprocessor:
    def __init__(self):
        # Initialize StandardScaler (fit during training, here we mock)
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_means = None
        self.feature_stds = None
        
    def preprocess_sensor_data(self, sensor_data_list: List[SensorReading], window_size: int = 30) -> np.ndarray:

        """
        Main preprocessing pipeline
        
        STEPS:
        1. Extract raw sensor values
        2. Simulate time window segmentation
        3. Normalize features
        4. Return feature vector for ML model
        """
        # Allow single reading
        if isinstance(sensor_data_list, SensorReading):
            sensor_data_list = [sensor_data_list]

        if len(sensor_data_list) == 0:
            raise ValueError("No sensor data provided")
        
        # Convert list of SensorReading to DataFrame
        df = pd.DataFrame([self._sensor_to_dict(r) for r in sensor_data_list])
        
        # Calculate statistical features per column
        stats = df.agg(['mean', 'std', 'min', 'max']).fillna(0)
        features = stats.values.flatten()  # flatten to 1D array
        
        # Normalize features
        features = self._normalize_features(features)
        return features

    
    def _sensor_to_dict(self, reading: SensorReading) -> Dict[str, float]:
        """Convert single SensorReading to dict of features"""
        acc = reading.accelerometer
        gyro = reading.gyroscope
        gps = reading.gps

        return {
            "acc_x": acc.x,
            "acc_y": acc.y,
            "acc_z": acc.z,
            "gyro_x": gyro.x,
            "gyro_y": gyro.y,
            "gyro_z": gyro.z,
            "acc_mag": np.sqrt(acc.x**2 + acc.y**2 + acc.z**2),
            "gyro_mag": np.sqrt(gyro.x**2 + gyro.y**2 + gyro.z**2),
            "gps_speed": gps.speed if gps else 0.0,
            "gps_alt": gps.altitude if gps else 0.0,
        }
        
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
        """Standardize features based on training stats"""
        if self.feature_means is None or self.feature_stds is None:
            # If not set, just return raw features
            return features
        return (features - self.feature_means) / (self.feature_stds + 1e-8)

    def set_normalization_stats(self, means: np.ndarray, stds: np.ndarray):
        """Set normalization statistics from training data"""
        self.feature_means = means
        self.feature_stds = stds
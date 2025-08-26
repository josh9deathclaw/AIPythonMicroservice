# example.py
from datetime import datetime, timezone
from .models import SensorReading, AccelerometerData, GyroscopeData, GPSData
from .preprocessing import DataPreprocessor

def main():
    # Create a sample sensor reading
    reading = SensorReading(
        accelerometer=AccelerometerData(x=0.1, y=0.2, z=0.3),
        gyroscope=GyroscopeData(x=0.01, y=0.02, z=0.03),
        gps=GPSData(speed=50.0, altitude=100.0),
        timestamp=datetime.now(timezone.utc)  # timezone-aware datetime
    )

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Preprocess sensor reading to feature vector
    features = preprocessor.preprocess_sensor_data(reading, window_size=30)

    print("Processed feature vector:")
    print(features)

if __name__ == "__main__":
    main()
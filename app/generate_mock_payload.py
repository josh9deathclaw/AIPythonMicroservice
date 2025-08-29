import json
import random
from datetime import datetime, timedelta

# Number of samples
NUM_SAMPLES = 600

# Trip and user IDs for testing
TRIP_ID = "test_trip_001"
USER_ID = "user123"

# Start timestamp
start_time = datetime.utcnow()

# Helper function to generate a single sensor sample
def generate_sample(ts):
    return {
        "accelerometer": {
            "x": round(random.uniform(-0.2, 0.2), 3),
            "y": round(random.uniform(-0.2, 0.2), 3),
            "z": round(random.uniform(9.7, 9.85), 3)
        },
        "gyroscope": {
            "alpha": round(random.uniform(-0.05, 0.05), 3),
            "beta": round(random.uniform(-0.05, 0.05), 3),
            "gamma": round(random.uniform(-0.05, 0.05), 3)
        },
        "gps": {
            "speed": round(random.uniform(0, 15), 2),
            "altitude": round(random.uniform(50, 55), 2),
            "lat": -37.8136,
            "lon": 144.9631
        },
        "timestamp": ts.isoformat() + "Z"
    }

# Generate the array of samples
sensor_data_array = []
for i in range(NUM_SAMPLES):
    ts = start_time + timedelta(seconds=i/10)  # 10Hz sampling
    sensor_data_array.append(generate_sample(ts))

# Full payload - FIXED to match aiController.js expectations
payload = {
    "tripId": TRIP_ID,          # Changed from "trip_id" to "tripId"
    "userId": USER_ID,          # Changed from "user_id" to "userId" 
    "sensorDataArray": sensor_data_array  # Changed from "sensor_data_array" to "sensorDataArray"
}

# Save to JSON file
with open("mock_payload_600.json", "w") as f:
    json.dump(payload, f, indent=2)

print(f"Generated {NUM_SAMPLES} sensor samples in mock_payload_600.json")
print("✅ Fixed field names to match aiController.js expectations:")
print(f"   - tripId: {TRIP_ID}")
print(f"   - userId: {USER_ID}")
print(f"   - sensorDataArray: {len(sensor_data_array)} samples")
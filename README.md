Installation
python -m venv venv
#Activate 
venv\Scripts\activate
#Install Dependences
pip install -r requirements.txt
#Run Service
uvicorn main:app --reload --port 8001
#Project Structure
ai-service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic data models
│   ├── preprocessing.py     # Data preprocessing pipeline
│   ├── prediction.py        # ML model integration
│   └── config.py           # Configuration settings
├── models/                  # Trained ML model files
│   └── transport_classifier.pkl
├── tests/                   # Test files
│   ├── test_api.py
│   └── test_preprocessing.py
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
└── README.md               # This file

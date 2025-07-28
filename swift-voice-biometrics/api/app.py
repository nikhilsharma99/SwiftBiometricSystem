from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import joblib
import tempfile
from src.predict import predict_speaker

app = FastAPI()

# TODO: Adjust paths and model loading as needed
model = joblib.load('models/voice_model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/label_encoder.pkl')
# anti_spoof_model = joblib.load('models/anti_spoof_model.pkl')  # Optional

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    # TODO: Add anti_spoof_model if available
    result = predict_speaker(tmp_path, model, scaler, encoder, anti_spoof_model=None)
    return result

# TODO: Add authentication, logging, and batching for production
# TODO: Document API usage in README

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000) 
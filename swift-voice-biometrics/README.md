# Swift Voice Biometrics

## Features
- Speaker verification and voice recognition using machine learning
- Supports multiple models (Logistic Regression, SVM, Random Forest, Neural Network)
- Anti-spoofing system (basic, see code comments)
- Real-time response measurement
- API for integration (Flask/FastAPI)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
- See `/src/main.py` for CLI usage
- See `/api/` for API usage (Flask/FastAPI)
- See `/notebooks/` for experiments and EDA

## Folder Structure
```
/api/           # Flask/FastAPI app, API scripts
/models/        # Model files, training scripts, anti-spoofing code
/data/          # Dataset, noise samples, metadata
/tests/         # Test scripts, synthetic/real test cases, logs
/docs/          # Documentation, diagrams, summary tables
/notebooks/     # Jupyter notebooks for experiments, EDA
/utils/         # Preprocessing, augmentation, helpers
```

## API Usage
- Start the API:
  ```bash
  cd api
  uvicorn app:app --reload
  ```
- Send a POST request to `/predict` with a WAV file:
  ```bash
  curl -X POST "http://localhost:8000/predict" -F "file=@your_audio.wav"
  ```
- See `/docs/` for pipeline diagram, metrics, and logs.
- See `/notebooks/` for experiments and EDA.

## Dataset
- **Limitation:** Only two speakers, limited emotions, mostly synthetic/test data
- **TODO:** Add more real user data and open datasets
- **Noise simulation:** See `/utils/augmentation.py` for background noise augmentation

## Metrics
| Model              | Accuracy | Precision | Recall | FAR  | FRR  |
|--------------------|----------|-----------|--------|------|------|
| Logistic Regression|   --     |    --     |   --   | --   | --   |
| SVM                |   --     |    --     |   --   | --   | --   |
| Random Forest      |   --     |    --     |   --   | --   | --   |
| Neural Network     |   --     |    --     |   --   | --   | --   |

- **TODO:** Fill in with real results after Phase 3 testing

## Anti-Spoofing
- See code comments in `/src/predict.py` and `/src/model_trainer.py`
- **TODO:** Test with real/generated spoofed voices, measure false accept %

## Real-Time Pipeline
- End-to-end response time measured in logs (see `/tests/`)
- **TODO:** Optimize with lighter models, threading, batching

## Deployment & Integration
- Minimal API in `/api/` (Flask/FastAPI)
- **Integration:** Can be used in call center IVR for speaker verification
- **Challenges:** Latency, security, scaling

## Documentation
- Key parts commented (model loading, prediction)
- See `/docs/` for diagrams, summary tables, and logs

## Future Work
- Add more real-world data
- Improve anti-spoofing
- Optimize for real-time deployment
- Expand API features

---

**For more details, see code comments and `/docs/`.** 
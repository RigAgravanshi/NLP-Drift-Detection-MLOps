from fastapi import FastAPI
from src.api.schemas import PredictionRequest, PredictionResponse
from src.models.predict import BertInference
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

try:
    model = BertInference(config['model']['classifier_name'], config['model']['num_classes'])
except Exception:
    model = None

app = FastAPI()
@app.post("/predict", response_model= PredictionResponse)    # route declares response type & returns proper object
def predict(request : PredictionRequest):                     # tells FastAPI i/p is a PredictionRequest
    if model is None:
        return PredictionResponse(predicted_intent="unavailable", confidence=0.0)
    predicted_intent, confidence = model.predict(request.text) # these have become params of PredictionResponse
    return PredictionResponse(predicted_intent=predicted_intent, confidence=confidence)

@app.get("/health")
def health(): 
    return {"status": "ok"}
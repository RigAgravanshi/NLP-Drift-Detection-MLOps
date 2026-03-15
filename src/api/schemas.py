from pydantic import BaseModel

class PredictionRequest(BaseModel):
    text : str                    # If text is missing/not a string, FastAPI returns a 422 error automatically

class PredictionResponse(BaseModel):
    predicted_intent : str        # it is ensured that only these data-types are returned
    confidence : float  
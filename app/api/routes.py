from fastapi import APIRouter
from app.ml.dummy_model import DummyModel
from app.metrics.prometheus import requests_total

router = APIRouter()
model = DummyModel()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/predict")
def predict(text: str):
    result = model.predict(text)
    requests_total.inc()
    return {"input": text, "prediction": result}

from fastapi import APIRouter
from app.metrics.prometheus import search_requests, embedding_requests
from fastapi import APIRouter, Query
from app.ml.embed import Embed
from app.ml.service import MLService

router = APIRouter()
mlservice = MLService()

@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/status")
def status():
    return mlservice.model_status()


@router.post("/embed")
def embed_text(embeds: list[Embed]):
    return mlservice.embed_and_store(embeds)


@router.get("/search")
def search_text(query: str = Query(...), top_k: int = 3):
    print(f"Received search request: query='{query}', top_k={top_k}")
    return mlservice.search(query, top_k)

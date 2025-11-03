from fastapi import APIRouter
from app.metrics.prometheus import search_requests, embedding_requests
from fastapi import APIRouter, Query
from app.ml.embed import Embed
from app.ml.embeddings import Embedder
from app.ml.vector_store import VectorStore

router = APIRouter()
embedder = Embedder()
store = VectorStore()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.post("/embed")
def embed_text(embeds: list[Embed]):
    """
    Embed a list of texts and add them to the vector store.
    Each item must include 'text', 'domain', and 'intent'.
    """
    embedding_requests.inc()
    embeddings = embedder.encode([embed.text for embed in embeds])
    store.add([embed.text for embed in embeds], embeddings)
    return {"status": "ok", "count": len(embeds)}

@router.get("/search")
def search_text(query: str = Query(...), top_k: int = 3):
    search_requests.inc()
    query_emb = embedder.encode([query])[0]
    results = store.search(query_emb, top_k)
    return {"query": query, "results": results}

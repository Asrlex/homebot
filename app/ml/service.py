from app.ml.embeddings import Embedder
from app.ml.vector_store import VectorStore
from app.ml.embed import Embed
from app.metrics.prometheus import embedding_requests, search_requests

class MLService:
    _instance = None
    
    def __init__(self):
        if MLService._instance is None:
            MLService._instance = self
            self.embedder = Embedder()
            self.vector_store = VectorStore()
    
    def model_status(self):
        """Return the status of the model and vector store."""
        self.vector_store.cursor.execute("SELECT COUNT(*) FROM metadata")
        count = self.vector_store.cursor.fetchone()[0]
        return {"total_vectors": self.vector_store.index.ntotal, "metadata_rows": count}


    def embed_and_store(self, embeds: list[Embed]):
        """
        Embed a list of texts and add them to the vector store.
        Each item must include 'text', 'domain', and 'intent'.
        """
        embedding_requests.inc()
        texts = [embed.text for embed in embeds]
        embeddings = self.embedder.encode(texts)
        self.vector_store.add(texts, 
                  embeddings,
                  [{"domain": embed.domain, "intent": embed.intent} for embed in embeds])
        return {"status": "ok", "count": len(embeds), "embedded_texts": texts}

    def search(self, query: str, top_k: int = 3):
        """
        Search the vector store for the most similar texts to the query.
        Args:
            query (str): The query text.
            top_k (int, optional): Number of top results to return. Defaults to 3.
        """
        search_requests.inc()
        query_embedding = self.embedder.encode([query])
        inference = self.vector_store.infer_intent_and_domain(query_embedding, top_k)
        return {
            "query": query,
            "intent": inference["intent"],
            "domain": inference["domain"],
            "results": inference["results"]
        }

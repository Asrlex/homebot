from threading import Lock
from sentence_transformers import SentenceTransformer

class Embedder:
    _instance = None
    _lock = Lock()
    model: SentenceTransformer

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        with cls._lock:
          if cls._instance is None:
              cls._instance = super().__new__(cls)
              cls._instance.model = SentenceTransformer(model_name)
          return cls._instance

    def encode(self, texts: list[str]):
        """
        Return dense vector embeddings for a list of texts.
        Converts embeddings to numpy arrays for compatibility with FAISS.
        """
        return self.model.encode(texts, convert_to_tensor=False)

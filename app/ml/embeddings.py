from sentence_transformers import SentenceTransformer

class Embedder:
    _instance = None

    def __new__(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer(model_name)
        return cls._instance

    def encode(self, texts: list[str]):
        """Return dense vector embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_tensor=False)

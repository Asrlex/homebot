import faiss
import numpy as np
import uuid

class VectorStore:
    _instance = None

    def __new__(cls, dim: int = 384):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.index = faiss.IndexFlatL2(dim)
            cls._instance.metadata = {}
        return cls._instance

    def add(self, texts: list[str], embeddings: np.ndarray):
        for text, emb in zip(texts, embeddings):
            uid = str(uuid.uuid4())
            self.metadata[uid] = {"text": text}
            self.index.add(np.array([emb]).astype("float32"))

    def search(self, query_emb: np.ndarray, top_k: int = 3):
        D, I = self.index.search(np.array([query_emb]).astype("float32"), top_k)
        results = []
        for idx_list, dist_list in zip(I, D):
            for idx, dist in zip(idx_list, dist_list):
                if idx < len(self.metadata):
                    uid = list(self.metadata.keys())[idx]
                    results.append({
                        "uuid": uid,
                        "text": self.metadata[uid]["text"],
                        "score": float(dist)
                    })
        return results

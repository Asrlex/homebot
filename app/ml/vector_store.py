import faiss
import numpy as np
import uuid
import sqlite3

class VectorStore:
    _instance = None
    index: faiss.IndexFlatL2
    metadata: dict
    dim: int

    def __new__(cls, dim: int = 384):
        """
        Create or return the singleton instance of VectorStore.
        Args:
            dim (int, optional): Dimension of the embedding vectors. Defaults to 384.
        Returns:
            VectorStore: The singleton instance of VectorStore.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.index = faiss.IndexFlatL2(dim)
            cls._instance.metadata = {}
            cls._instance.dim = dim
            cls._instance._init_db()
        return cls._instance


    def _init_db(self):
        """Initialize the SQLite in-memory database."""
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE metadata (
                uuid TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                domain TEXT NOT NULL,
                intent TEXT NOT NULL
            )
        """)
        self.conn.commit()


    def add(self, texts: list[str], embeddings: np.ndarray, metadata: list[dict]):
        """
        Add texts, embeddings, and metadata to the vector store.
        Args:
            texts (list[str]): List of texts to add.
            embeddings (np.ndarray): Corresponding embedding vectors.
            metadata (list[dict]): List of metadata dictionaries (with 'domain' and 'intent').
        """
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {embeddings.shape[1]}.")

        for text, emb, meta in zip(texts, embeddings, metadata):
            id = str(uuid.uuid4())
            self.cursor.execute(
                "INSERT INTO metadata (uuid, text, domain, intent) VALUES (?, ?, ?, ?)",
                (id, text, meta["domain"], meta["intent"])
            )
            self.index.add(np.array([emb], dtype="float32"))
        self.conn.commit()


    def search(self, query_emb: np.ndarray, top_k: int = 3):
        """
        Search for the most similar texts in the vector store.
        Args:
            query_emb (np.ndarray): Embedding vector for the query.
            top_k (int, optional): Number of top results to return. Defaults to 3.
        Returns:
            list[dict]: List of search results with metadata.
        """
        if query_emb.shape[0] != self.dim:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.dim}, got {query_emb.shape[0]}.")

        D, I = self.index.search(np.array([query_emb], dtype="float32"), top_k)
        results = []
        for idx_list, dist_list in zip(I, D):
            for idx, dist in zip(idx_list, dist_list):
                if idx < self.index.ntotal:
                    # Retrieve metadata from SQLite
                    self.cursor.execute("SELECT uuid, text, domain, intent FROM metadata LIMIT 1 OFFSET ?", (idx,))
                    row = self.cursor.fetchone()
                    if row:
                        results.append({
                            "uuid": row[0],
                            "text": row[1],
                            "domain": row[2],
                            "intent": row[3],
                            "score": float(dist)
                        })
        return results
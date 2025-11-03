from prometheus_client import Counter

embedding_requests = Counter("homebot_embeddings_total", "Total embedding requests")
search_requests = Counter("homebot_search_total", "Total search requests")

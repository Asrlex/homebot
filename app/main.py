from fastapi import FastAPI, Lifespan
from prometheus_client import make_asgi_app
from app.api.routes import router as api_router
from app.ml.vector_store import VectorStore
from app.ml.seeder import seed_faiss

store = VectorStore()

def lifespan(app: FastAPI) -> Lifespan:
    async def on_startup():
        """Seed FAISS index on API startup."""
        seed_faiss(store)

    async def on_shutdown():
        """Handle any cleanup if necessary."""
        pass

app = FastAPI(title="HomeBot ML Service", version="0.1.0", lifespan=lifespan)

app.include_router(api_router, prefix="/api")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

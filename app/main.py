from fastapi import FastAPI
from prometheus_client import make_asgi_app
from app.api.routes import router as api_router
from app.ml.vector_store import VectorStore
from app.ml.seeder import seed_faiss
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uvicorn

store = VectorStore()

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    seed_faiss(store)
    yield

main_app = FastAPI(lifespan=lifespan)

app = FastAPI(title="HomeBot ML Service", version="0.1.0", lifespan=lifespan)

app.include_router(api_router, prefix="/api")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
